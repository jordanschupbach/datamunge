#pragma once
// Independent numerical and architecture regression checks retained from the
// original executable; canonical operations are called directly. No training
// driver or example-source dependency is retained here.
#include <dm/dm.hpp>
#include <iostream>
namespace {
namespace nn = dm::nn;
using nn::Padding;
using nn::Tensor;
namespace check {
inline void Require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}
inline void Near(double a, double b, double tolerance, const std::string& message) {
  if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance)
    throw std::runtime_error(message);
}
} // namespace check
void InflationChecks() {
  std::vector<float> spatial(36), input(160);
  for (int i = 0; i < 36; ++i)
    spatial[static_cast<std::size_t>(i)] = static_cast<float>(.09f * std::sin(.37 * i));
  for (int t = 0; t < 5; ++t)
    for (int i = 0; i < 32; ++i)
      input[static_cast<std::size_t>(t * 32 + i)] = static_cast<float>(.3f * std::cos(.29 * i));
  const auto inflated =
      dm::nn::models::InflateI3d(nn::from_vector(spatial, {3, 3, 2, 2}), 3).to_vector<float>();
  check::Require(inflated.size() == 108, "inflated [3,3,3,2,2] filter");
  for (int t = 0; t < 3; ++t)
    for (int i = 0; i < 36; ++i)
      check::Near(
          inflated[static_cast<std::size_t>(36 * t + i)],
          spatial[static_cast<std::size_t>(i)] / 3,
          0,
          "all inflated temporal slices divided by T");
  // Inflate (tile then divide) inside the differentiated computation, as TF does.
  const auto run = [](const std::vector<float>& x_values,
                      const std::vector<float>& w_values,
                      std::vector<Tensor>*      grads) {
    Tensor x = nn::from_vector(x_values, {1, 5, 4, 4, 2}),
           w = nn::from_vector(w_values, {3, 3, 2, 2});
    x.requires_grad_(true);
    w.requires_grad_(true);
    const Tensor w3 = nn::tile(nn::expand_dims(w, 0), {3, 1, 1, 1, 1}) / 3.0;
    const Tensor y  = nn::conv3d(x, w3, nn::conv3d_options(1, Padding::Same));
    if (grads)
      *grads = nn::grad(nn::sum(y * y), {w, x});
    return y.to_vector<float>();
  };
  std::vector<Tensor> g;
  const auto          out  = run(input, spatial, &g);
  const auto          host = [&](const std::vector<float>& a,
                        const std::vector<float>& b,
                        std::vector<double>*      output = nullptr) {
    double sum = 0;
    for (int t = 0; t < 5; ++t)
      for (int h = 0; h < 4; ++h)
        for (int z = 0; z < 4; ++z)
          for (int c = 0; c < 2; ++c) {
            double value = 0;
            for (int kt = 0; kt < 3; ++kt)
              for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw)
                  for (int ci = 0; ci < 2; ++ci) {
                    const int it = t + kt - 1, ih = h + kh - 1, iw = z + kw - 1;
                    if (it >= 0 && it < 5 && ih >= 0 && ih < 4 && iw >= 0 && iw < 4)
                      value +=
                          double(a[static_cast<std::size_t>(((it * 4 + ih) * 4 + iw) * 2 + ci)]) *
                          b[static_cast<std::size_t>(((kh * 3 + kw) * 2 + ci) * 2 + c)] / 3.;
                  }
            sum += value * value;
            if (output)
              output->push_back(value);
          }
    return sum;
  };
  std::vector<double> expected;
  host(input, spatial, &expected);
  double forward = 0, gradient = 0, constant = 0;
  for (std::size_t i = 0; i < 160; ++i)
    forward = std::max(forward, std::abs(expected[i] - out[i]));
  // Independent spatial-only correlation; SAME temporal edge attenuation is 2/3.
  for (int t = 0; t < 5; ++t)
    for (int h = 0; h < 4; ++h)
      for (int z = 0; z < 4; ++z)
        for (int c = 0; c < 2; ++c) {
          double value = 0;
          for (int kh = 0; kh < 3; ++kh)
            for (int kw = 0; kw < 3; ++kw)
              for (int ci = 0; ci < 2; ++ci) {
                const int ih = h + kh - 1, iw = z + kw - 1;
                if (ih >= 0 && ih < 4 && iw >= 0 && iw < 4)
                  value += double(input[static_cast<std::size_t>((ih * 4 + iw) * 2 + ci)]) *
                           spatial[static_cast<std::size_t>(((kh * 3 + kw) * 2 + ci) * 2 + c)];
              }
          if (t == 0 || t == 4)
            value *= 2. / 3;
          constant = std::max(
              constant,
              std::abs(value - out[static_cast<std::size_t>(((t * 4 + h) * 4 + z) * 2 + c)]));
        }
  const auto gw = g[0].to_vector<float>(), gx = g[1].to_vector<float>();
  for (int variable = 0; variable < 2; ++variable) {
    const auto& original = variable == 0 ? spatial : input;
    for (std::size_t i = 0; i < original.size(); ++i) {
      auto plus = original, minus = original;
      plus[i] += .001f;
      minus[i] -= .001f;
      const double denominator = double(plus[i]) - minus[i];
      const double derivative  = variable == 0
                                     ? (host(input, plus) - host(input, minus)) / denominator
                                     : (host(plus, spatial) - host(minus, spatial)) / denominator;
      gradient = std::max(gradient, std::abs(derivative - (variable == 0 ? gw[i] : gx[i])));
    }
  }
  check::Near(forward, 0, 1e-7, "all160 independent inflated output coordinates");
  check::Near(
      constant,
      0,
      1e-7,
      "repeated-image spatial response and explicit temporal zero-padding borders");
  check::Near(gradient, 0, 2e-6, "all36source-kernel+160input finite differences");
  std::cout << "inflation:all160outputs error=" << forward
            << ",constant-video/interior and border error=" << constant
            << ",all196source/input FD error=" << gradient << std::endl;
}

void UnitChecks() {
  nn::ParameterStore       p(691), state(693);
  const std::vector<float> input{-.7f, .2f, .3f, -.6f, .9f, .4f, -.2f, .8f},
      weight{.4f, -.3f, .2f, .7f}, beta{.13f, -.09f};
  std::vector<float> coefficient(8);
  for (int i = 0; i < 8; ++i)
    coefficient[static_cast<std::size_t>(i)] = static_cast<float>(.1f * std::cos(.7 * i));
  Tensor x = nn::from_vector(input, {2, 2, 1, 1, 2});
  {
    nn::NoGradGuard guard;
    dm::nn::models::I3dUnit(p, state, "fixture", x, 2, 1, 1, false, .5, false);
  }
  p.set("fixture/W", nn::from_vector(weight, {1, 1, 1, 2, 2}));
  p.set("fixture/beta", nn::from_vector(beta, {2}));
  x.requires_grad_(true);
  const Tensor y = dm::nn::models::I3dUnit(p, state, "fixture", x, 2, 1, 1, true, .5, false);
  const auto   gradients = nn::grad(
      nn::sum(y * nn::from_vector(coefficient, {2, 2, 1, 1, 2})),
      {p.at("fixture/W"), p.at("fixture/beta"), x});
  std::vector<std::vector<double>> values(3);
  values[0].assign(weight.begin(), weight.end());
  values[1].assign(beta.begin(), beta.end());
  values[2].assign(input.begin(), input.end());
  const auto host = [&](const std::vector<std::vector<double>>& v,
                        std::vector<double>*                    output = nullptr,
                        std::vector<double>* moments                   = nullptr) {
    double z[8], mu[2] = {}, var[2] = {};
    for (int n = 0; n < 4; ++n)
      for (int c = 0; c < 2; ++c) {
        z[2 * n + c] = v[2][2 * n] * v[0][c] + v[2][2 * n + 1] * v[0][2 + c];
        mu[c] += z[2 * n + c] / 4;
      }
    for (int n = 0; n < 4; ++n)
      for (int c = 0; c < 2; ++c)
        var[c] += std::pow(z[2 * n + c] - mu[c], 2) / 4;
    double result = 0;
    for (int n = 0; n < 4; ++n)
      for (int c = 0; c < 2; ++c) {
        const double value =
            std::max(0., (z[2 * n + c] - mu[c]) / std::sqrt(var[c] + .001) + v[1][c]);
        result += value * coefficient[static_cast<std::size_t>(2 * n + c)];
        if (output)
          output->push_back(value);
      }
    if (moments)
      *moments = {mu[0], mu[1], var[0], var[1]};
    return result;
  };
  std::vector<double> expected, moments;
  host(values, &expected, &moments);
  const auto yv    = y.to_vector<float>();
  double     error = 0, derivative_error = 0;
  for (std::size_t i = 0; i < 8; ++i)
    error = std::max(error, std::abs(expected[i] - yv[i]));
  for (std::size_t variable = 0; variable < 3; ++variable) {
    const auto g = gradients[variable].to_vector<float>();
    for (std::size_t i = 0; i < values[variable].size(); ++i) {
      auto plus = values, minus = values;
      plus[variable][i] += 1e-5;
      minus[variable][i] -= 1e-5;
      derivative_error =
          std::max(derivative_error, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
    }
  }
  {
    nn::NoGradGuard guard; // the training step's moving-moment update
    dm::nn::models::I3dUnit(p, state, "fixture", x, 2, 1, 1, true, .5, true);
  }
  const auto inference =
      dm::nn::models::I3dUnit(p, state, "fixture", x, 2, 1, 1, false, .5, false).to_vector<float>();
  const auto moving_mean = state.at("fixture/mean").to_vector<float>(),
             moving_var  = state.at("fixture/variance").to_vector<float>();
  for (std::size_t c = 0; c < 2; ++c) {
    check::Near(moving_mean[c], moments[c] * .5, 1e-7, "moving mean exact population update");
    check::Near(
        moving_var[c], .5 + .5 * moments[2 + c], 1e-7, "moving variance population convention");
    for (std::size_t n = 0; n < 4; ++n) {
      const double z = values[2][2 * n] * values[0][c] + values[2][2 * n + 1] * values[0][2 + c];
      const double e = std::max(
          0., (z - .5 * moments[c]) / std::sqrt(.5 + .5 * moments[2 + c] + .001) + values[1][c]);
      check::Near(
          inference[2 * n + c], e, 1e-6, "stored-stat inference independent of test batch moments");
    }
  }
  check::Near(error, 0, 1e-6, "3D Unit train output versus independent population normalization");
  check::Near(
      derivative_error,
      0,
      1e-6,
      "all14 kernel/offset/input populationBN-ReLU host finite differences");
  std::cout << "I3D Unit populationBN/ReLU forward error=" << error
            << ",all14hostFD error=" << derivative_error
            << ";moving population moments/stored-stat inference passed\n";
}

/// Endpoint shapes [T, S, C] from the SAME padding rules alone.
std::vector<nn::Shape> ShapeTrace(const dm::nn::models::I3dConfig& cfg, const int channels[16]) {
  const auto same = [](std::int64_t in, std::int64_t k, std::int64_t s) {
    return nn::conv_output_size(in, k, s, 1, Padding::Same);
  };
  // (temporal window/stride, spatial window/stride) for each pool; convolutions keep sizes except
  // Conv3d_1a (stride 2).
  std::vector<nn::Shape> shapes;
  std::int64_t           t = same(cfg.frames, 7, 2), s = same(cfg.image, 7, 2);
  const auto push = [&](int i) { shapes.push_back({1, t, s, s, channels[i] / cfg.divisor}); };
  push(0);
  s = same(s, 3, 2); // MaxPool 1x3x3 / 1x2x2
  push(1);
  push(2);
  push(3);
  s = same(s, 3, 2);
  push(4);
  for (int block = 0, i = 5; block < 9; ++block) {
    if (block == 2) {
      t = same(t, 3, 2), s = same(s, 3, 2);
      push(i++);
    }
    if (block == 7) {
      t = same(t, 2, 2), s = same(s, 2, 2);
      push(i++);
    }
    push(i++);
  }
  return shapes;
}

void GraphChecks(bool run_full) {
  const int time[]     = {32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16, 8, 8, 8},
            space[]    = {112, 56, 56, 56, 28, 28, 28, 14, 14, 14, 14, 14, 14, 7, 7, 7},
            channels[] = {
                64, 64, 64, 192, 192, 256, 480, 480, 512, 512, 512, 528, 832, 832, 832, 1024};
  for (bool compact : {false, true}) {
    const dm::nn::models::I3dConfig reference =
        compact ? dm::nn::models::I3dConfig{8, 3, 16, 32} : dm::nn::models::I3dConfig{};
    // Full widths execute on 16x32x32 by default (full=1: at 64x224x224).
    const dm::nn::models::I3dConfig cfg =
        compact || run_full ? reference : dm::nn::models::I3dConfig{1, 400, 16, 32};
    const bool small = cfg.frames == 16;
    const auto table = [&](bool at_small, int divisor) {
      std::vector<nn::Shape> shapes;
      for (int i = 0; i < 16; ++i)
        shapes.push_back(
            {1,
             at_small ? time[i] / 4 : time[i],
             at_small ? (space[i] + 6) / 7 : space[i],
             at_small ? (space[i] + 6) / 7 : space[i],
             channels[i] / divisor});
      return shapes;
    };
    check::Require(
        ShapeTrace(reference, channels) == table(compact, reference.divisor),
        "I3D original or compact endpoint dimensions from the SAME padding rules");
    nn::ParameterStore p(683), state(685);
    nn::Generator      gen(5);
    const Tensor       rgb = nn::uniform({1, cfg.frames, cfg.image, cfg.image, 3}, 0.0, 1.0, gen);
    const auto         net = dm::nn::models::I3d(p, state, rgb, cfg);
    check::Require(net.stages.size() == 16, "three stem convs/four pools/nine Mixed endpoints");
    const auto expected_shapes = table(small, cfg.divisor);
    for (std::size_t i = 0; i < 16; ++i)
      check::Require(
          net.stages[i].shape() == expected_shapes[i], "I3D executed endpoint dimension");
    check::Require(
        net.pooled.shape() == nn::Shape({1, small ? 1 : 7, 1, 1, 1024 / cfg.divisor}) &&
            net.logits.shape() == nn::Shape({1, cfg.classes}),
        "I3D pooled/logit shapes");
    check::Require(
        p.size() == 116 && state.size() == 114, "57W+57beta+headW/b and114 moving moments");
    // Independent per-unit parameter inventory from the published branch table.
    const int table9[9][6] = {
        {64, 96, 128, 16, 32, 32},
        {128, 128, 192, 32, 96, 64},
        {192, 96, 208, 16, 48, 64},
        {160, 112, 224, 24, 64, 64},
        {128, 128, 256, 24, 64, 64},
        {112, 144, 288, 32, 64, 64},
        {256, 160, 320, 32, 128, 128},
        {256, 160, 320, 32, 128, 128},
        {384, 192, 384, 48, 128, 128}};
    const auto   w        = [&](int x) { return x / cfg.divisor; };
    const auto   count    = [](int a, int b, int k) { return std::int64_t{a} * b * k * k * k + b; };
    std::int64_t expected = count(3, w(64), 7) + count(w(64), w(64), 1) + count(w(64), w(192), 3);
    int          in       = w(192);
    for (const auto& c : table9) {
      expected += count(in, w(c[0]), 1) + count(in, w(c[1]), 1) + count(w(c[1]), w(c[2]), 3) +
                  count(in, w(c[3]), 1) + count(w(c[3]), w(c[4]), 3) + count(in, w(c[5]), 1);
      in = w(c[0]) + w(c[2]) + w(c[4]) + w(c[5]);
    }
    expected += std::int64_t{in} * cfg.classes + cfg.classes;
    check::Require(expected == p.count(), "independent57-unit plushead parameter inventory");
    const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
    check::Require(gradients.size() == 116, "all116 I3D learned gradients");
    for (const auto& g : gradients) {
      check::Require(g.defined(), "no missing I3D parameter gradient");
      check::Require(nn::all(nn::is_finite(g)).item<double>() != 0, "finite I3D gradient");
    }
    std::cout << (compact ? "compact" : "full") << " I3D graph:" << p.count()
              << " learned parameters/116 tensors,114 moving tensors;all16 endpoints/9Mixed "
                 "shapes,independent "
                 "inventory and all gradients passed;"
              << (compact ? "executed 16x32x32"
                  : small ? "full width executed on 16x32x32 (64x224 table from padding rules; "
                            "full=1 executes it)"
                          : "full network executed at 64x224x224")
              << "\n";
  }
}

} // namespace
