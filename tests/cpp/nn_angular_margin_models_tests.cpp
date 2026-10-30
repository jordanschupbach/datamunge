#include "common/angular_margin.hpp"
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
using metric::AngularMarginConfig;
using metric::AngularMarginKind;
using nn::Tensor;
namespace {
std::pair<double, double> HostPhi(double value, const AngularMarginConfig &cfg) {
  if (cfg.margin == 0)
    return {value, 1};
  if (cfg.kind == AngularMarginKind::kCosFace)
    return {value - cfg.margin, 1};
  const double low = -1.f + cfg.epsilon, high = 1.f - cfg.epsilon,
               c = std::clamp(value, low, high);
  const double clip = value > low && value < high ? 1 : 0;
  if (cfg.monotonic && c <= std::cos(3.14159265358979323846f - cfg.margin))
    return {c - std::sin(cfg.margin) * cfg.margin, clip};
  const double sine = std::sqrt(std::max(1 - c * c, double(cfg.epsilon)));
  return {c * std::cos(cfg.margin) - sine * std::sin(cfg.margin),
          (std::cos(cfg.margin) + c / sine * std::sin(cfg.margin)) * clip};
}

double HostCE(const std::vector<double> &logits, int target,
              std::vector<double> *gradient = nullptr) {
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double sum = 0;
  for (double value : logits)
    sum += std::exp(value - maximum);
  if (gradient) {
    gradient->clear();
    for (std::size_t i = 0; i < logits.size(); ++i)
      gradient->push_back(std::exp(logits[i] - maximum) / sum - (int(i) == target));
  }
  return maximum + std::log(sum) - logits[target];
}

void FormulaChecks() {
  for (bool cosface : {false, true})
    for (bool monotonic : {false, true}) {
      AngularMarginConfig cfg;
      cfg.kind = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
      cfg.margin = cosface ? .35f : .5f;
      cfg.scale = 7;
      cfg.monotonic = monotonic;
      const float target[] = {
          .8f, .1f, -.95f, -1, 1, 0, std::cos(3.14159265358979323846f - .5f), -.6f};
      std::vector<float> values(24);
      std::vector<std::int64_t> labels(8);
      for (int i = 0; i < 8; ++i) {
        labels[i] = i % 3;
        for (int j = 0; j < 3; ++j)
          values[i * 3 + j] = j == i % 3 ? target[i] : j == 0 ? -.2f : .3f;
      }
      Tensor cosine = nn::from_vector(values, {8, 3}).requires_grad_();
      const Tensor y = nn::from_vector(labels);
      const Tensor logits = metric::ApplyAngularMargin(cosine, y, 3, cfg);
      const Tensor loss = metric::AngularCrossEntropy(logits, y, 3);
      const auto gradient = nn::grad(loss, {cosine})[0].to_vector<double>();
      auto zero = cfg;
      zero.margin = 0;
      const auto plain = metric::ApplyAngularMargin(cosine, y, 3, zero).to_vector<double>();
      const auto out0 = logits.to_vector<double>();
      double host = 0, error = 0;
      for (int i = 0; i < 8; ++i) {
        const int label = int(labels[i]);
        std::vector<double> expected;
        for (int j = 0; j < 3; ++j) {
          const double raw = values[i * 3 + j];
          expected.push_back(cfg.scale * (j == label ? HostPhi(raw, cfg).first : raw));
          check::Near(out0[i * 3 + j], expected.back(), 2e-6,
                      "target-only angular/cosine transformed value");
          check::Near(plain[i * 3 + j], cfg.scale * raw, 2e-7,
                      "zero margin normalized-softmax identity");
        }
        std::vector<double> d;
        host += HostCE(expected, label, &d) / 8;
        for (int j = 0; j < 3; ++j) {
          const double derivative =
              cfg.scale * d[j] * (j == label ? HostPhi(values[i * 3 + j], cfg).second : 1) / 8;
          error = std::max(error, std::abs(gradient[i * 3 + j] - derivative));
        }
      }
      check::Near(loss.item(), host, 2e-6, "host stabilized margin CE");
      check::Near(error, 0, 2e-5,
                  "every margin/CE gradient including tail and endpoint clips");
      std::cout << "  " << (cosface ? "CosFace" : "ArcFace") << " monotonic=" << monotonic
                << ": host CE/every-logit derivative error=" << error
                << "; target-only,zero margin,endpoints passed" << std::endl;
    }
  std::vector<float> grid(1001 * 2);
  for (int i = 0; i <= 1000; ++i)
    grid[i * 2] = -1.f + 2.f * i / 1000, grid[i * 2 + 1] = .17f;
  AngularMarginConfig cfg;
  cfg.scale = 1;
  const auto monotone = metric::ApplyAngularMargin(nn::from_vector(grid, {1001, 2}),
                                                   nn::zeros({1001}, nn::DType::Int64), 2, cfg)
                            .to_vector<float>();
  for (int i = 0; i <= 1000; ++i) {
    check::Require(monotone[i * 2] <= grid[i * 2] + 1e-6,
                   "positive angular margin never boosts target logit");
    if (i)
      check::Require(monotone[i * 2] >= monotone[(i - 1) * 2] - 1e-6,
                     "tail extension preserves monotonic target curve");
    check::Near(monotone[i * 2 + 1], .17f, 0, "all non-target logits unchanged");
  }
  for (int invalid = 0; invalid < 5; ++invalid) {
    auto bad = cfg;
    if (invalid == 0)
      bad.margin = -.1f;
    if (invalid == 1)
      bad.margin = 2;
    if (invalid == 2)
      bad.scale = 0;
    if (invalid == 3)
      bad.epsilon = 0;
    if (invalid == 4)
      bad.margin = NAN;
    check::Throws(
        [&] {
          bad.Validate();
        },
        "reject invalid margin configuration");
  }
  std::cout << "  1001-point monotonic tail/target penalty grid and invalid-configuration "
               "checks passed"
            << std::endl;
}

void NormalizedHeadChecks() {
  for (bool cosface : {false, true}) {
    nn::ParameterStore p(17);
    AngularMarginConfig cfg;
    cfg.kind = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
    cfg.margin = cosface ? .2f : .3f;
    cfg.scale = 5;
    const std::vector<float> xv = {.3f, .7f, -.4f, -.8f, .2f, .5f, .1f, -.6f, .9f},
                             wv = {.4f, -.3f, .8f, .1f, -.2f, .6f,
                                   .1f, -.5f, .7f, .2f, -.4f, .9f};
    const Tensor target = nn::tensor({2, 0, 3}).to(nn::DType::Int64);
    const std::int64_t tv[] = {2, 0, 3};
    // The weight is created by the head, then overwritten with the controlled values.
    {
      nn::NoGradGuard guard;
      metric::NormalizedAngularHead(p, nn::zeros({3, 3}), target, 3, 4, cfg);
      p.set("angular/W", nn::from_vector(wv, {3, 4}));
    }
    Tensor raw = nn::from_vector(xv, {3, 3}).requires_grad_();
    const auto head = metric::NormalizedAngularHead(p, raw, target, 3, 4, cfg);
    const Tensor loss = metric::AngularCrossEntropy(head.training, target, 4);
    const auto gradient = nn::grad(loss, {raw, p.at("angular/W")});
    const auto cos_out = head.cosine.to_vector<double>(),
               emb = head.embedding.to_vector<double>(),
               wout = head.weights.to_vector<double>(), gx = gradient[0].to_vector<double>(),
               gw = gradient[1].to_vector<double>();
    double u[3][3] = {}, v[3][4] = {}, xn[3] = {}, wn[4] = {}, du[3][3] = {}, dv[3][4] = {},
           dx[3][3] = {}, dw[3][4] = {}, host = 0, error = 0;
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k)
        xn[i] += double(xv[3 * i + k]) * xv[3 * i + k];
      xn[i] = std::sqrt(xn[i]);
      for (int k = 0; k < 3; ++k)
        u[i][k] = xv[3 * i + k] / xn[i];
    }
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 3; ++k)
        wn[j] += double(wv[4 * k + j]) * wv[4 * k + j];
      wn[j] = std::sqrt(wn[j]);
      for (int k = 0; k < 3; ++k)
        v[k][j] = wv[4 * k + j] / wn[j];
    }
    for (int i = 0; i < 3; ++i) {
      std::vector<double> logits, c;
      for (int j = 0; j < 4; ++j) {
        double value = 0;
        for (int k = 0; k < 3; ++k)
          value += u[i][k] * v[k][j];
        c.push_back(value);
        check::Near(cos_out[i * 4 + j], value, 2e-7,
                    "feature-row/class-column normalization cosine");
        logits.push_back(cfg.scale * (j == tv[i] ? HostPhi(value, cfg).first : value));
      }
      std::vector<double> d;
      host += HostCE(logits, int(tv[i]), &d) / 3;
      for (int j = 0; j < 4; ++j) {
        const double dc = d[j] * cfg.scale * (j == tv[i] ? HostPhi(c[j], cfg).second : 1) / 3;
        for (int k = 0; k < 3; ++k) {
          du[i][k] += dc * v[k][j];
          dv[k][j] += dc * u[i][k];
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      double dot = 0;
      for (int k = 0; k < 3; ++k)
        dot += u[i][k] * du[i][k];
      for (int k = 0; k < 3; ++k) {
        dx[i][k] = (du[i][k] - u[i][k] * dot) / xn[i];
        error = std::max(error, std::abs(gx[i * 3 + k] - dx[i][k]));
        check::Near(emb[i * 3 + k], u[i][k], 1e-7, "unit feature values");
      }
    }
    for (int j = 0; j < 4; ++j) {
      double dot = 0;
      for (int k = 0; k < 3; ++k)
        dot += v[k][j] * dv[k][j];
      for (int k = 0; k < 3; ++k) {
        dw[k][j] = (dv[k][j] - v[k][j] * dot) / wn[j];
        error = std::max(error, std::abs(gw[k * 4 + j] - dw[k][j]));
        check::Near(wout[k * 4 + j], v[k][j], 1e-7, "unit class-column values");
      }
    }
    check::Near(loss.item(), host, 1e-6, "normalized head host CE");
    // Keep the nearly aligned target fixture. Check all coordinates by a
    // separate central difference as well as the double-precision formula.
    auto evaluate = [&](const std::vector<float> &features,
                        const std::vector<float> &weights) {
      nn::NoGradGuard guard;
      p.set("angular/W", nn::from_vector(weights, {3, 4}));
      const auto h = metric::NormalizedAngularHead(p, nn::from_vector(features, {3, 3}),
                                                   target, 3, 4, cfg);
      return metric::AngularCrossEntropy(h.training, target, 4).item();
    };
    double difference_error = 0;
    const float h = .002f;
    for (int coordinate = 0; coordinate < 21; ++coordinate) {
      auto features = xv, weights = wv;
      float &value = coordinate < 9 ? features[coordinate] : weights[coordinate - 9];
      const float base = value;
      value = base + h;
      const double plus = evaluate(features, weights);
      value = base - h;
      const double minus = evaluate(features, weights);
      const double expected = coordinate < 9 ? dx[coordinate / 3][coordinate % 3]
                                             : dw[(coordinate - 9) / 4][(coordinate - 9) % 4];
      difference_error =
          std::max(difference_error, std::abs((plus - minus) / (2 * h) - expected));
    }
    check::Near(error, 0, 1e-5,
                "all raw feature and class-weight derivatives, float32 near-collinear case");
    check::Near(difference_error, 0, 1e-3,
                "independent central differences for every feature and weight");
    // Positive feature/column scaling must disappear under normalization.
    std::vector<float> scaled_x(9), scaled_w(12);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        scaled_x[3 * i + k] = xv[3 * i + k] * (i + 2);
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 4; ++j)
        scaled_w[4 * k + j] = wv[4 * k + j] * (j + 2);
    std::vector<double> original, scaled;
    {
      nn::NoGradGuard guard;
      p.set("angular/W", nn::from_vector(wv, {3, 4}));
      original =
          metric::NormalizedAngularHead(p, nn::from_vector(xv, {3, 3}), Tensor(), 3, 4, cfg)
              .inference.to_vector<double>();
      p.set("angular/W", nn::from_vector(scaled_w, {3, 4}));
      scaled = metric::NormalizedAngularHead(p, nn::from_vector(scaled_x, {3, 3}), Tensor(), 3,
                                             4, cfg)
                   .inference.to_vector<double>();
    }
    for (int i = 0; i < 12; ++i)
      check::Near(scaled[i], original[i], 1e-6,
                  "independent row/column positive scale invariance");
    check::Require(p.count() == 12 && p.size() == 1,
                   "bias-free normalized head has one shared weight matrix");
    std::cout << "  normalized " << (cosface ? "CosFace" : "ArcFace")
              << ": all feature/column-weight Jacobian error=" << error
              << ", central difference error=" << difference_error
              << "; label-free inference and independent scale invariance passed" << std::endl;
  }
}

void GeometryChecks() {
  for (bool cosface : {false, true}) {
    const double angle = 1, margin = cosface ? .2 : .3,
                 boundary = cosface ? angle / 2 - std::asin(margin / (2 * std::sin(angle / 2)))
                                    : (angle - margin) / 2;
    std::vector<float> c(6);
    for (int i = 0; i < 3; ++i) {
      const double theta = boundary + (i - 1) * .02;
      c[i * 2] = float(std::cos(theta));
      c[i * 2 + 1] = float(std::cos(angle - theta));
    }
    AngularMarginConfig cfg;
    cfg.kind = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
    cfg.margin = float(margin);
    cfg.scale = 1;
    const auto out = metric::ApplyAngularMargin(nn::from_vector(c, {3, 2}),
                                                nn::zeros({3}, nn::DType::Int64), 2, cfg)
                         .to_vector<float>();
    check::Require(out[0] > out[1] && out[4] < out[5], "margin boundary sides");
    check::Near(out[2], out[3], 2e-7, "analytic two-class boundary");
    check::Require(boundary < angle / 2, "margin shrinks target training region");
  }
  std::cout << "  two-class angular/cosine boundaries and opposite decision sides passed"
            << std::endl;
}

void ModelChecks(bool full) {
  for (bool compact : {false, true})
    for (bool cosface : {false, true}) {
      const auto cfg = compact ? metric::FaceNetConfig::Compact() : metric::FaceNetConfig{};
      const int classes = compact ? 5 : 1000;
      AngularMarginConfig margin;
      margin.kind = cosface ? AngularMarginKind::kCosFace : AngularMarginKind::kArcFace;
      margin.margin = cosface ? .2f : .3f;
      margin.scale = 16;
      int tensors = 0;
      const std::int64_t census = metric::FaceNetCensus(cfg, &tensors) + 128 * classes;
      check::Require(census == (compact ? 108680 : 140153664) && tensors + 1 == 29,
                     "independent encoder+head census");
      if (!compact && !full) {
        std::cout << "  full " << (cosface ? "CosFace" : "ArcFace")
                  << ":NN1+128x1000 head,140153664 parameters/29 tensors from the layer "
                     "shapes (TF: graph-only; full=1 executes it)"
                  << std::endl;
        continue;
      }
      nn::ParameterStore p(17);
      std::vector<float> in(std::size_t(2) * cfg.side * cfg.side * cfg.input);
      for (std::size_t i = 0; i < in.size(); ++i)
        in[i] = .5 + .45 * std::sin(i * .113f);
      const Tensor label = nn::tensor({0, 3}).to(nn::DType::Int64);
      const auto net =
          metric::FaceNetNN1(p, nn::from_vector(in, {2, cfg.side, cfg.side, cfg.input}), cfg);
      const auto head = metric::NormalizedAngularHead(p, net.raw, label, 128, classes, margin);
      const Tensor loss = metric::AngularCrossEntropy(head.training, label, classes);
      const auto gradient = nn::grad(loss, p.parameters());
      check::Require(p.count() == census && gradient.size() == 29,
                     "independent encoder+head census and gradient connectivity");
      check::Require(std::isfinite(loss.item()) &&
                         head.inference.shape() == nn::Shape({2, classes}),
                     "finite compact loss/classifier shape");
      double low = INFINITY, high = 0;
      for (std::size_t i = 0; i < gradient.size(); ++i) {
        const double n = nn::norm(gradient[i]).item();
        check::Require(std::isfinite(n) && n > 0,
                       "every CNN/head gradient finite nonzero: " + p.names()[i]);
        low = std::min(low, n), high = std::max(high, n);
      }
      std::cout << "  " << (compact ? "compact " : "full ")
                << (cosface ? "CosFace" : "ArcFace") << ":" << census
                << " parameters/all29 numerical derivatives, norms[" << low << ',' << high
                << ']' << std::endl;
    }
}

} // namespace
TEST(AngularMarginModels, IndependentTargetMarginAndEveryLogitDerivative) {
  FormulaChecks();
}
TEST(AngularMarginModels, IndependentNormalizationAndCentralDifferences) {
  NormalizedHeadChecks();
}
TEST(AngularMarginModels, AnalyticTwoClassBoundaries) {
  GeometryChecks();
}
TEST(AngularMarginModels, FullCensusCompactAll29Gradients) {
  nn::set_num_threads(2);
  ModelChecks(false);
}
TEST(AngularMarginModels, ClassConfusionAndDataOrder) {
  const auto logits = nn::tensor({1., 0., 0., 1., 0., 0.}, {3, 2});
  const auto labels = nn::from_vector(std::vector<std::int64_t>{0, 1, 1});
  const auto counts =
      nn::models::AngularClassificationConfusion(logits, labels).to_vector<std::int64_t>();
  EXPECT_EQ(counts, (std::vector<std::int64_t>{1, 0, 1, 1}));
  nn::ImageDataset d{nn::zeros({3, 28, 28, 1}), labels};
  EXPECT_EQ(nn::models::AngularLabels(d, {2, 0, 2}).to_vector<std::int64_t>(),
            (std::vector<std::int64_t>{1, 0, 1}));
  EXPECT_THROW(nn::models::AngularLabels(d, {3}), std::invalid_argument);
  EXPECT_THROW(nn::models::ApplyAngularMargin(
                   logits, labels + nn::full({}, 2, nn::DType::Int64), 2, {}),
               std::invalid_argument);
}
