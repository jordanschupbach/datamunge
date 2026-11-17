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
void FlowStackChecks() {
  std::vector<double> values(16), coeff(16);
  for (int i = 0; i < 16; ++i) {
    values[i] = i == 0 ? 40. : i == 9 ? -35. : 3 * std::sin(.7 * i);
    coeff[i]  = .2 * std::cos(.33 * i);
  }
  auto x = nn::from_vector(values, {1, 2, 2, 2, 2}).requires_grad_(),
       y = dm::nn::models::TwoStreamFlowStack(x, 2, 2);
  auto g = nn::grad(nn::sum(y * nn::from_vector(coeff, {1, 2, 2, 4})), {x})[0].to_vector<double>();
  auto host = [&](const std::vector<double>& v, std::vector<double>* output = nullptr) {
    std::vector<double> z(16);
    double              loss = 0;
    for (int t = 0; t < 2; ++t)
      for (int c = 0; c < 2; ++c) {
        double mean = 0;
        for (int p = 0; p < 4; ++p)
          mean += v[t * 8 + p * 2 + c] / 4;
        for (int p = 0; p < 4; ++p) {
          int j = p * 4 + 2 * t + c;
          z[j]  = std::clamp((v[t * 8 + p * 2 + c] - mean) / 10., -1., 1.);
          loss += z[j] * coeff[j];
        }
      }
    if (output)
      *output = z;
    return loss;
  };
  std::vector<double> expected;
  host(values, &expected);
  auto   output  = y.to_vector<double>();
  double forward = 0, error = 0;
  for (int i = 0; i < 16; ++i) {
    forward   = std::max(forward, std::abs(expected[i] - output[i]));
    auto plus = values, minus = values;
    plus[i] += 1e-5;
    minus[i] -= 1e-5;
    error = std::max(error, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
  }
  check::Near(forward, 0, 1e-7, "signed dx0/dy0/dx1/dy1 order/mean/clipping");
  check::Near(error, 0, 1e-7, "ALL16 centered/clipped/raw flow FD coordinates");
  check::Near(
      nn::max(nn::abs(dm::nn::models::TwoStreamFlowStack(nn::full({1, 2, 2, 2, 2}, 7), 2, 2)))
          .item(),
      0,
      0,
      "constant camera displacement zero");
  std::cout << "flow stack ALL16 host FD=" << error << ";constant displacement passed\n";
}
void FusionChecks() {
  std::vector<double> av{4, -2, 1, -1, 2, .3}, bv{-1, 2, 0, 3, -2, 1};
  auto                a = nn::from_vector(av, {2, 3}).requires_grad_(),
       b                = nn::from_vector(bv, {2, 3}).requires_grad_();
  auto fused            = dm::nn::models::TwoStreamFusion(a, b);
  auto target           = nn::from_vector(std::vector<double>{1, 0, 0, 0, 0, 1}, {2, 3});
  auto g                = nn::grad(-nn::sum(nn::log(fused) * target) * .5, {a, b});
  auto host             = [&](const std::vector<double>& x, const std::vector<double>& y) {
    double loss = 0;
    for (int row = 0; row < 2; ++row) {
      double sumx = 0, sumy = 0;
      for (int c = 0; c < 3; ++c) {
        sumx += std::exp(x[row * 3 + c]);
        sumy += std::exp(y[row * 3 + c]);
      }
      int c = row ? 2 : 0;
      loss -=
          .5 * std::log(.5 * (std::exp(x[row * 3 + c]) / sumx + std::exp(y[row * 3 + c]) / sumy));
    }
    return loss;
  };
  double error = 0;
  for (int stream = 0; stream < 2; ++stream) {
    auto derivative = g[stream].to_vector<double>();
    for (int i = 0; i < 6; ++i) {
      auto plus = stream ? bv : av, minus = plus;
      plus[i] += 1e-5;
      minus[i] -= 1e-5;
      double fd = stream ? (host(av, plus) - host(av, minus)) / 2e-5
                         : (host(plus, bv) - host(minus, bv)) / 2e-5;
      error     = std::max(error, std::abs(fd - derivative[i]));
    }
  }
  check::Near(error, 0, 1e-7, "ALL12 late probability fusion NLL derivatives");
  auto alternate = nn::softmax((a + b) * .5, -1);
  check::Require(
      nn::sum(nn::abs(alternate - fused)).item() > .01,
      "probability average differs from logit average");
  auto actual = fused.to_vector<double>();
  for (int row = 0; row < 2; ++row) {
    double ax = 0, bx = 0;
    for (int c = 0; c < 3; ++c) {
      ax += std::exp(av[row * 3 + c]);
      bx += std::exp(bv[row * 3 + c]);
    }
    for (int c = 0; c < 3; ++c)
      check::Near(
          actual[row * 3 + c],
          .5 * (std::exp(av[row * 3 + c]) / ax + std::exp(bv[row * 3 + c]) / bx),
          1e-7,
          "independent fusion values");
  }
  std::cout << "fusion ALL12 host FD=" << error << " PASS\n";
}
void PoolChecks() {
  std::vector<float> values(16);
  for (int i = 0; i < 16; ++i)
    values[i] = -10 - i;
  auto x      = nn::from_vector(values, {1, 4, 4, 1}).requires_grad_();
  auto padded = nn::pad(x, {{0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY);
  auto y      = nn::max_pool2d(padded, {3, 3}, {2, 2}, nn::Padding::Valid);
  auto g      = nn::grad(nn::sum(y), {x})[0].to_vector<float>();
  auto z      = y.to_vector<float>();
  const float expected[] = {-10, -12, -18, -20};
  for (int i = 0; i < 4; ++i)
    check::Near(z[i], expected[i], 0, "pool2 negative partial border");
  for (int i = 0; i < 16; ++i)
    check::Near(g[i], i == 0 || i == 2 || i == 8 || i == 10 ? 1 : 0, 0, "all16 max derivatives");
}
void GraphChecks() {
  for (bool compact : {false, true}) {
    dm::nn::models::TwoStreamConfig cfg = compact
                                              ? dm::nn::models::TwoStreamConfig{96, 16, 64, 32, 3}
                                              : dm::nn::models::TwoStreamConfig{};
    const int sides[] = {109, 26, 13, 13, 13}, small[] = {45, 10, 5, 5, 5}, pool[] = {54, 13, 6},
              sp[] = {22, 5, 2}, widths[] = {96, 256, 512, 512, 512}, kernel[] = {7, 5, 3, 3, 3};
    std::int64_t total = 0;
    for (bool temporal : {false, true}) {
      int          in = temporal ? 20 : 3, side = cfg.image;
      std::int64_t count = 0;
      for (int i = 0; i < 5; ++i) {
        int out = widths[i] / cfg.divisor;
        count += std::int64_t(kernel[i]) * kernel[i] * in * out + out;
        in   = out;
        side = (side + (i ? 2 : 0) - kernel[i]) / (i < 2 ? 2 : 1) + 1;
        check::Require(
            side == (compact ? small[i] : sides[i]), "full/compact independent convolution side");
        if (i == 0 || i == 1 || i == 4) {
          side  = (side + (i == 1 ? 1 : 0) - 3) / 2 + 1;
          int j = i == 0 ? 0 : i == 1 ? 1 : 2;
          check::Require(
              side == (compact ? sp[j] : pool[j]), "full/compact independent pooling side");
        }
      }
      int flat = side * side * in;
      check::Require(flat == (compact ? 128 : 18432), "CNN-M flattened side");
      count += std::int64_t(flat) * cfg.fc6 + cfg.fc6 + std::int64_t(cfg.fc6) * cfg.fc7 + cfg.fc7 +
               std::int64_t(cfg.fc7) * cfg.classes + cfg.classes;
      total += count;
      std::cout << (temporal ? "flow" : "RGB") << (compact ? " compact" : " full CNN-M-2048") << ":"
                << count << " learned parameters\n";
    }
    if (compact) {
      nn::ParameterStore p(863);
      nn::NoGradGuard    guard;
      auto               path = dm::nn::models::TwoStream(
          p, nn::zeros({1, 96, 96, 3}), nn::zeros({1, 10, 96, 96, 2}), cfg);
      check::Require(p.count() == total && p.size() == 32, "independent32 tensor inventory");
      check::Require(
          path.stack.shape() == nn::Shape({1, 96, 96, 20}) &&
              path.probabilities.shape() == nn::Shape({1, 3}),
          "flow stack and fusion shapes");
      for (const auto* stream : {&path.rgb, &path.flow}) {
        for (int i = 0; i < 5; ++i)
          check::Require(
              stream->conv[i].shape() == nn::Shape({1, small[i], small[i], widths[i] / 16}),
              "all five compact convolution endpoints");
        for (int i = 0; i < 3; ++i)
          check::Require(stream->pool[i].size(1) == sp[i], "three actual pool endpoints");
        check::Require(
            stream->flat.shape() == nn::Shape({1, 128}) &&
                stream->fc6.shape() == nn::Shape({1, 64}) &&
                stream->fc7.shape() == nn::Shape({1, 32}),
            "compact dense endpoints");
      }
    }
    std::cout << (compact ? "compact" : "full") << " two-stream inventory=" << total
              << "/32 tensors PASS\n";
  }
}

} // namespace
