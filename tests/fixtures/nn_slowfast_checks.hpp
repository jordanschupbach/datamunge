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
void SamplingChecks() {
  for (int frames : {16, 32}) {
    std::vector<float> values(frames);
    std::iota(values.begin(), values.end(), 0.f);
    auto x        = nn::from_vector(values, {1, frames, 1, 1, 1}).requires_grad_();
    auto y        = dm::nn::models::SlowFastSample(x, frames);
    auto gradient = nn::grad(nn::sum(y), {x})[0].to_vector<float>();
    auto expected = frames == 16 ? std::vector<int>{0, 15} : std::vector<int>{0, 10, 20, 31};
    check::Require(
        dm::nn::models::SlowFastIndices(frames) == expected,
        "authors linspace endpoints/truncated indices");
    auto out = y.to_vector<float>();
    for (size_t i = 0; i < expected.size(); ++i)
      check::Near(out[i], expected[i], 0, "sample actual frame index");
    for (int i = 0; i < frames; ++i)
      check::Near(
          gradient[i],
          std::find(expected.begin(), expected.end(), i) != expected.end() ? 1 : 0,
          0,
          "sampled-only input gradient routing");
  }
}
void FusionChecks() {
  std::vector<std::vector<double>> values(3);
  int                              sizes[] = {40, 128, 24};
  for (int a = 0; a < 3; ++a) {
    values[a].resize(sizes[a]);
    for (int i = 0; i < sizes[a]; ++i)
      values[a][i] = .2 * std::sin(.27 * i + .3 * a);
  }
  std::vector<double> coefficient(56);
  for (int i = 0; i < 56; ++i)
    coefficient[i] = .15 * std::cos(.39 * i);
  auto kernel    = nn::from_vector(values[0], {5, 1, 1, 2, 4}).requires_grad_(),
       fast      = nn::from_vector(values[1], {1, 16, 2, 2, 2}).requires_grad_(),
       slow      = nn::from_vector(values[2], {1, 2, 2, 2, 3}).requires_grad_();
  auto lateral   = dm::nn::models::FactorConv(fast, kernel, 8),
       joined    = dm::nn::models::SlowFastJoin(slow, lateral);
  auto gradients = nn::grad(
      nn::sum(joined * nn::from_vector(coefficient, {1, 2, 2, 2, 7})), {kernel, fast, slow});
  auto host = [&](const std::vector<std::vector<double>>& a,
                  std::vector<double>*                    output = nullptr) {
    double total = 0;
    for (int t = 0; t < 2; ++t)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
          for (int c = 0; c < 7; ++c) {
            double value = 0;
            if (c < 3)
              value = a[2][((t * 2 + y) * 2 + x) * 3 + c];
            else
              for (int kt = 0; kt < 5; ++kt)
                for (int ci = 0; ci < 2; ++ci) {
                  int it = t * 8 + kt - 2;
                  if (it >= 0 && it < 16)
                    value +=
                        a[1][((it * 2 + y) * 2 + x) * 2 + ci] * a[0][(kt * 2 + ci) * 4 + c - 3];
                }
            int i = ((t * 2 + y) * 2 + x) * 7 + c;
            total += value * coefficient[i];
            if (output)
              output->push_back(value);
          }
    return total;
  };
  std::vector<double> expected;
  host(values, &expected);
  auto   output = joined.to_vector<double>();
  double error = 0, derivative = 0;
  int    count = 0;
  for (int i = 0; i < 56; ++i)
    error = std::max(error, std::abs(expected[i] - output[i]));
  for (int a = 0; a < 3; ++a) {
    auto g = gradients[a].to_vector<double>();
    for (size_t i = 0; i < values[a].size(); ++i) {
      auto plus = values, minus = values;
      plus[a][i] += 1e-6;
      minus[a][i] -= 1e-6;
      derivative = std::max(derivative, std::abs((host(plus) - host(minus)) / 2e-6 - g[i]));
      ++count;
    }
  }
  check::Require(count == 192, "all40lateral+128Fast+24Slow coordinates");
  check::Near(error, 0, 1e-7, "all56 joined outputs host convolution");
  check::Near(derivative, 0, 1e-7, "ALL192 fusion/pathway derivatives host FD");
  std::cout << "lateral5/stride8 all56 values=" << error << " all192 FD=" << derivative
            << std::endl;
}
void ZeroGateChecks() {
  nn::ParameterStore p(761), state;
  auto               input =
      nn::from_vector(std::vector<float>{.1f, .3f, .8f, .2f, .4f, .9f, .6f, .5f}, {2, 2, 1, 1, 2})
          .requires_grad_();
  auto weight = nn::from_vector(std::vector<float>{.4f, -.2f, .3f, .7f}, {1, 1, 1, 2, 2});
  p.get("expand/W", weight.shape(), nn::init::zeros());
  p.set("expand/W", weight);
  std::vector<float> coefficients(8);
  for (int i = 0; i < 8; ++i)
    coefficients[i] = .13 * std::cos(.61 * i);
  auto coefficient = nn::from_vector(coefficients, {2, 2, 1, 1, 2});
  auto run         = [&]() {
    return nn::relu(
        input + dm::nn::models::SlowFastUnit(
                    p, state, "expand", input, 2, 1, 1, 1, 1, 1, 1, false, true, true, false));
  };
  auto y      = run();
  auto leaves = p.parameters();
  leaves.push_back(input);
  auto grad = nn::grad(nn::sum(y * coefficient), leaves);
  check::Near(nn::max(nn::abs(y - input)).item(), 0, 0, "zero-final exact residual identity");
  check::Near(
      nn::max(nn::abs(grad.back() - coefficient)).item(), 0, 1e-7, "identity input gradients");
  auto                              names = p.names();
  std::map<std::string, nn::Tensor> g;
  for (size_t i = 0; i < names.size(); ++i)
    g[names[i]] = grad[i];
  check::Near(
      nn::max(nn::abs(g.at("expand/W"))).item(),
      0,
      0,
      "zero gate blocks every branch kernel derivative");
  auto x = input.to_vector<float>(), w = weight.to_vector<float>();
  for (int c = 0; c < 2; ++c) {
    double z[4], mean = 0, var = 0, expected_gamma = 0, expected_beta = 0;
    for (int n = 0; n < 4; ++n) {
      z[n] = double(x[2 * n]) * w[c] + double(x[2 * n + 1]) * w[2 + c];
      mean += z[n] / 4;
    }
    for (double v : z)
      var += std::pow(v - mean, 2) / 4;
    for (int n = 0; n < 4; ++n) {
      expected_gamma += coefficients[2 * n + c] * (z[n] - mean) / std::sqrt(var + 1e-5);
      expected_beta += coefficients[2 * n + c];
    }
    check::Near(
        g.at("expand/bn/gamma").to_vector<float>()[c],
        expected_gamma,
        1e-6,
        "host initial gamma derivative");
    check::Near(
        g.at("expand/bn/beta").to_vector<float>()[c],
        expected_beta,
        1e-7,
        "host initial beta derivative");
  }
  p.set("expand/bn/gamma", nn::full({2}, .2));
  auto opened = run();
  auto gw     = nn::grad(nn::sum(opened * coefficient), {p.at("expand/W")})[0];
  check::Require(
      nn::sum(nn::square(gw)).item() > 1e-8, "opened final gate unlocks branch kernel derivatives");
}
void GraphChecks() {
  for (bool compact : {false, true}) {
    dm::nn::models::SlowFastConfig cfg =
        compact ? dm::nn::models::SlowFastConfig{8, 3, 16} : dm::nn::models::SlowFastConfig{};
    int          image    = compact ? 32 : 224;
    std::int64_t expected = 0;
    int          units    = 0;
    auto         count    = [&](int in, int out, int volume) {
      expected += std::int64_t(in) * out * volume + 2 * out;
      ++units;
    };
    int b = cfg.base, f = b / 8;
    count(3, b, 49);
    count(3, f, 245);
    int si = b, fi = f;
    count(fi, 2 * fi, 5);
    si += 2 * fi;
    const int depths[] = {3, 4, 6, 3};
    for (int stage = 0; stage < 4; ++stage) {
      for (int path = 0; path < 2; ++path) {
        auto& incoming = path ? fi : si;
        int   width = (b << stage) / (path ? 8 : 1), channels = 4 * width;
        for (int block = 0; block < depths[stage]; ++block) {
          count(incoming, width, path || stage >= 2 ? 3 : 1);
          count(width, width, 9);
          count(width, channels, 1);
          if (incoming != channels || (stage && block == 0))
            count(incoming, channels, 1);
          incoming = channels;
        }
      }
      if (stage < 3) {
        count(fi, 2 * fi, 5);
        si += 2 * fi;
      }
    }
    expected += std::int64_t(36 * b) * cfg.classes + cfg.classes;
    check::Require(units == 110, "independent two-path/lateral/BN/head inventory110 units");
    if (compact) {
      nn::ParameterStore p(757), state;
      nn::NoGradGuard    guard;
      auto               net =
          dm::nn::models::SlowFast(p, state, nn::zeros({1, cfg.frames, image, image, 3}), cfg);
      check::Require(
          p.count() == expected && p.size() == 332 && state.size() == 220,
          "332learned/220moving inventory");
      check::Require(
          net.slow_input.shape() == nn::Shape({1, cfg.frames / 8, image, image, 3}),
          "actual slow sampled endpoint");
      check::Require(
          net.slow_stages.size() == 5 && net.fast_stages.size() == 5 && net.lateral.size() == 4 &&
              net.slow_blocks.size() == 16 && net.fast_blocks.size() == 16,
          "all32 residual blocks/four fusions retained");
      for (int i = 0; i < 5; ++i) {
        int channels = i ? cfg.base * (4 << (i - 1)) : cfg.base,
            space    = image / (i ? 4 << (i - 1) : 4);
        check::Require(
            net.slow_stages[i].shape() == nn::Shape({1, cfg.frames / 8, space, space, channels}) &&
                net.fast_stages[i].shape() ==
                    nn::Shape({1, cfg.frames, space, space, channels / 8}),
            "every actual paired stage endpoint");
        if (i < 4)
          check::Require(
              net.lateral[i].shape() == nn::Shape({1, cfg.frames / 8, space, space, channels / 4}),
              "every actual fusion endpoint");
      }
      int index = 0;
      for (int stage = 0; stage < 4; ++stage)
        for (int block = 0; block < depths[stage]; ++block, ++index) {
          int channels = cfg.base * (4 << stage), space = image / (4 << stage);
          check::Require(
              net.slow_blocks[index].shape() ==
                      nn::Shape({1, cfg.frames / 8, space, space, channels}) &&
                  net.fast_blocks[index].shape() ==
                      nn::Shape({1, cfg.frames, space, space, channels / 8}),
              "all32 actual residual block endpoints");
        }
      check::Require(
          net.pooled.shape() == nn::Shape({1, 36 * cfg.base}) &&
              net.logits.shape() == nn::Shape({1, cfg.classes}),
          "pooled/head endpoints");
    }
    std::cout << (compact ? "compact" : "full reference") << " SlowFast:" << expected
              << " parameters/332learned+220moving;two16-block paths/four fusions PASS\n";
  }
}

} // namespace
