// Independent numeric and model-geometry checks; experiment driver omitted.
#pragma once
#include "nn_interpretation_check_support.hpp"
namespace inpaint {
using namespace dm::nn::models;
}
namespace denoise {
using namespace dm::nn::models;
}
namespace flow {
using namespace dm::nn::models;
}
namespace color {
using namespace dm::nn::models;
}
namespace depth {
using namespace dm::nn::models;
}
namespace pose {
using namespace dm::nn::models;
}
namespace style {
using namespace dm::nn::models;
}
namespace crnn {
using namespace dm::nn::models;
}
namespace ocr {
using namespace dm::nn::models;
}
namespace cifar {
using namespace dm::nn::models;
}

#include <cmath>
#include <datamunge/datasets/assets.hpp>
#include <iostream>
namespace nn = datamunge::nn;
using nn::Tensor;
using namespace depth;

namespace {

double Sign(double value) {
  return (value > 0) - (value < 0);
}

struct HostLossResult {
  double              log = 0, gradient = 0;
  std::vector<double> derivative;
};

HostLossResult HostLoss(
    const std::vector<float>& prediction,
    const std::vector<float>& truth,
    const std::vector<float>& mask,
    int                       batch,
    int                       height,
    int                       width,
    double                    lambda) {
  const int      pixels = height * width;
  HostLossResult result;
  result.derivative.resize(prediction.size());
  std::vector<double> errors(prediction.size());
  std::vector<int>    valid(prediction.size());
  int                 images = 0, edge_images = 0;
  for (int n = 0; n < batch; ++n) {
    int    count = 0, edges = 0;
    double sum = 0, square = 0, gradient = 0;
    for (int i = 0; i < pixels; ++i) {
      const int    j      = n * pixels + i;
      const double target = truth[std::size_t(j)];
      if (mask[std::size_t(j)] > .5 && std::isfinite(target) && target > .1f && target <= 10) {
        valid[std::size_t(j)]  = 1;
        errors[std::size_t(j)] = std::log(double(prediction[std::size_t(j)])) - std::log(target);
        ++count;
        sum += errors[std::size_t(j)];
        square += errors[std::size_t(j)] * errors[std::size_t(j)];
      }
    }
    if (count) {
      ++images;
      result.log += square / count - lambda * std::pow(sum / count, 2);
    }
    for (int i = 0; i < pixels; ++i)
      if (valid[std::size_t(n * pixels + i)])
        result.derivative[std::size_t(n * pixels + i)] =
            2 * (errors[std::size_t(n * pixels + i)] - lambda * sum / count) /
            (count * prediction[std::size_t(n * pixels + i)]);
    auto edge = [&](int a, int b) {
      if (!valid[std::size_t(a)] || !valid[std::size_t(b)])
        return;
      gradient += std::abs(errors[std::size_t(a)] - errors[std::size_t(b)]);
      ++edges;
    };
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const int j = n * pixels + y * width + x;
        if (x + 1 < width)
          edge(j, j + 1);
        if (y + 1 < height)
          edge(j, j + width);
      }
    if (edges) {
      ++edge_images;
      result.gradient += gradient / edges;
    }
  }
  if (images)
    result.log /= images;
  if (edge_images)
    result.gradient /= edge_images;
  for (auto& value : result.derivative)
    if (images)
      value /= images;
  for (int n = 0; n < batch; ++n) {
    int edges = 0;
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const int j = n * pixels + y * width + x;
        if (x + 1 < width && valid[std::size_t(j)] && valid[std::size_t(j + 1)])
          ++edges;
        if (y + 1 < height && valid[std::size_t(j)] && valid[std::size_t(j + width)])
          ++edges;
      }
    auto edge = [&](int a, int b) {
      if (valid[std::size_t(a)] && valid[std::size_t(b)]) {
        const double d =
            .1 * Sign(errors[std::size_t(a)] - errors[std::size_t(b)]) / (edges * edge_images);
        result.derivative[std::size_t(a)] += d / prediction[std::size_t(a)];
        result.derivative[std::size_t(b)] -= d / prediction[std::size_t(b)];
      }
    };
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const int j = n * pixels + y * width + x;
        if (x + 1 < width)
          edge(j, j + 1);
        if (y + 1 < height)
          edge(j, j + width);
      }
  }
  return result;
}

void LossChecks() {
  std::vector<float> prediction(18, NAN), truth(18, 0.f), mask(18, 1.f);
  prediction[0] = 1.4f, prediction[1] = 3.2f, prediction[2] = 5.4f;
  truth[0] = 1, truth[1] = 2, truth[2] = 3;
  truth[4] = NAN, truth[5] = INFINITY, prediction[6] = 2.5f, truth[6] = 2;
  const Tensor t = nn::from_vector(truth, {3, 2, 3, 1}), m = nn::from_vector(mask, {3, 2, 3, 1});
  for (double lambda : {0., .5, 1.}) {
    Tensor       x   = nn::from_vector(prediction, {3, 2, 3, 1}).requires_grad_();
    const Tensor log = DepthLogLoss(x, t, m, lambda), gradient = DepthGradientLoss(x, t, m);
    const Tensor objective  = log + .1 * gradient;
    const auto   derivative = nn::grad(objective, {x})[0].to_vector<float>();
    const auto   host       = HostLoss(prediction, truth, mask, 3, 2, 3, lambda);
    check::Near(log.item(), host.log, 2e-7, "per-image masked log loss denominator and scale term");
    check::Near(
        gradient.item(), host.gradient, 2e-7, "only valid neighbor edges and nonempty images");
    double error = 0;
    for (std::size_t i = 0; i < prediction.size(); ++i)
      error = std::max(error, std::abs(derivative[i] - host.derivative[i]));
    check::Near(error, 0, 3e-7, "every valid and invalid depth-coordinate derivative");
    std::cout << " masked log lambda=" << lambda
              << ",valid-neighbor gradient and every-coordinate derivative max error=" << error
              << ";unequal counts,invalid NaNs/zeros,empty image passed\n";
  }
  std::vector<float> tv(6), sv(6);
  for (int i = 0; i < 6; ++i)
    tv[std::size_t(i)] = float(1 + i), sv[std::size_t(i)] = float(2 * (1 + i));
  const Tensor tt = nn::from_vector(tv, {1, 2, 3, 1}), scaled = nn::from_vector(sv, {1, 2, 3, 1}),
               valid = nn::ones({1, 2, 3, 1}), empty = nn::zeros({1, 2, 3, 1});
  check::Near(
      DepthLogLoss(scaled, tt, valid, 1).item(), 0, 2e-7, "lambda1 ignores a uniform metric scale");
  check::Near(
      DepthLogLoss(scaled, tt, valid, .5).item(),
      .5 * std::pow(std::log(2.), 2),
      2e-7,
      "lambda.5 retains metric-scale supervision");
  check::Near(
      DepthGradientLoss(scaled, tt, valid).item(), 0, 2e-7, "log-gradient ignores global scale");
  check::Near(DepthLogLoss(scaled, tt, empty).item(), 0, 0, "empty depth batch log loss0");
  check::Near(
      DepthGradientLoss(scaled, tt, empty).item(), 0, 0, "empty depth batch gradient loss0");
  std::cout << " explicit global-scale invariance/scale anchoring and all-empty batches passed\n";
}

void PositiveHeadChecks() {
  const std::vector<float> value = {-80, -2, 0, 2, 80};
  Tensor                   x     = nn::from_vector(value).requires_grad_();
  const Tensor             d     = PositiveMetricDepth(x);
  const auto               g     = nn::grad(nn::sum(d), {x})[0].to_vector<float>();
  const auto               dv    = d.to_vector<float>();
  for (int i = 0; i < 5; ++i) {
    const double z        = value[std::size_t(i)],
                 expected = .1 + std::max(z, 0.) + std::log1p(std::exp(-std::abs(z)));
    check::Near(
        dv[std::size_t(i)], expected, 4e-6, "positive metric depth including extreme logits");
    check::Near(
        g[std::size_t(i)],
        1 / (1 + std::exp(-z)),
        1e-7,
        "every positive-head coordinate derivative");
  }
  std::cout
      << " stable positive metric head values/every-coordinate Jacobian/extreme logits passed\n";
}

void MetricPriorChecks() {
  const Tensor truth = nn::from_vector(std::vector<float>{1, 2, 0, 4, NAN, INFINITY}, {2, 1, 3, 1});
  const Tensor scaled = nn::from_vector(std::vector<float>{2, 4, NAN, 8, NAN, NAN}, {2, 1, 3, 1});
  const Tensor mask   = nn::ones({2, 1, 3, 1});
  const auto   score = ScoreDepth(scaled, truth, mask), perfect = ScoreDepth(truth, truth, mask);
  check::Near(score.abs_rel, 1, 0, "known twice-depth AbsRel");
  check::Near(score.sq_rel, 2.75, 0, "per-image SqRel differs from pooled pixels");
  check::Near(score.rmse, (std::sqrt(2.5) + 4) / 2, 1e-12, "per-image metric RMSE");
  check::Near(score.log_rmse, std::log(2.), 1e-12, "natural log RMSE");
  check::Near(score.log10, std::log10(2.), 1e-12, "absolute log10 error");
  check::Near(score.silog, 0, 1e-8, "scale-invariant log metric at uniform2x");
  check::Require(
      score.delta1 == 0 && score.delta2 == 0 && score.delta3 == 0 && score.pixels == 3 &&
          score.images == 2,
      "strict ratio thresholds and valid pixel/image counts");
  check::Require(
      perfect.abs_rel == 0 && perfect.rmse == 0 && perfect.delta1 == 1 && perfect.delta2 == 1 &&
          perfect.delta3 == 1,
      "perfect metric depth recovery");
  const auto boundary =
      ScoreDepth(nn::full({1, 1, 1, 1}, 5.0), nn::full({1, 1, 1, 1}, 4.0), nn::ones({1, 1, 1, 1}));
  check::Require(
      boundary.delta1 == 0 && boundary.delta2 == 1 && boundary.delta3 == 1,
      "exact1.25 ratio uses strict delta1 threshold");
  std::vector<float> d(2 * 3072), v(2 * 3072, 1.f);
  for (int i = 0; i < 3072; ++i)
    d[std::size_t(i)] = 1, d[3072 + std::size_t(i)] = 4;
  DepthData data;
  data.depth = nn::from_vector(d, {2, 48, 64, 1});
  data.valid = nn::from_vector(v, {2, 48, 64, 1});
  auto prior = FitDepthPriors(data);
  check::Near(prior.geometric, 2, 1e-12, "fit-only geometric rather than arithmetic depth prior");
  for (float s : prior.spatial.to_vector<float>())
    check::Near(s, 2, 0, "per-pixel fit log-depth prior");
  v[0] = 0, d[0] = NAN;
  data.depth = nn::from_vector(d, {2, 48, 64, 1});
  data.valid = nn::from_vector(v, {2, 48, 64, 1});
  prior      = FitDepthPriors(data);
  check::Near(
      prior.spatial.to_vector<float>()[0], 4, 0, "invalid fit observations never enter prior");
  const auto reorder =
      DepthBatch(nn::from_vector(std::vector<std::int32_t>{7, 13, 19, 29}), {3, 1});
  check::Require(
      reorder.dtype() == nn::DType::Int32 &&
          reorder.to_vector<std::int32_t>() == std::vector<std::int32_t>{29, 13},
      "source identity row gathering preserves integer dtype");
  std::cout << " independent metric units/logs/scale/delta boundary/perfect/invalid cases and "
               "fit-only geometric/spatial priors passed\n";
}

void ModelChecks() {
  nn::ParameterStore p(139);
  std::vector<float> image(2 * 48 * 64 * 3), truth(2 * 48 * 64), mask(truth.size());
  for (std::size_t i = 0; i < image.size(); ++i)
    image[i] = float(.5 + .4 * std::sin(double(i) * .17));
  for (std::size_t i = 0; i < truth.size(); ++i)
    truth[i] = float(1 + double(i % 13) * .1), mask[i] = float(i % 7 != 0);
  const auto net    = MonocularDepth(p, nn::from_vector(image, {2, 48, 64, 3}));
  const auto losses = DepthObjective(
      net, nn::from_vector(truth, {2, 48, 64, 1}), nn::from_vector(mask, {2, 48, 64, 1}));
  const auto gradient = nn::grad(losses[0], p.parameters());
  check::Require(
      p.count() == 231745 && p.size() == 16, "exact actually-trained depth model parameter census");
  const std::vector<nn::Shape> expected = {
      {2, 24, 32, 8},
      {2, 12, 16, 16},
      {2, 6, 8, 32},
      {2, 128},
      {2, 192},
      {2, 12, 16, 1},
      {2, 48, 64, 1},
      {2, 48, 64, 16},
      {2, 48, 64, 16},
      {2, 48, 64, 1},
      {2, 48, 64, 1}};
  check::Require(net.stages.size() == expected.size(), "all11 coarse/global/local stages exposed");
  for (std::size_t i = 0; i < expected.size(); ++i)
    check::Require(
        net.stages[i].shape() == expected[i],
        "exact global-vector and spatial batch/channel geometry");
  double minimum = INFINITY, maximum = 0;
  for (const auto& g : gradient) {
    double norm = 0;
    for (float v : g.to_vector<float>()) {
      check::Require(std::isfinite(v), "finite depth parameter gradient");
      norm += double(v) * v;
    }
    norm = std::sqrt(norm);
    check::Require(norm > 0, "every coarse/fine tensor receives gradient");
    minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
  }
  std::cout << " depth231745 parameters/all16 finite nonzero derivative norms[" << minimum << ","
            << maximum << "],all11 exact stage shapes passed\n";
}

} // namespace
