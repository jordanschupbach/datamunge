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
std::vector<double>
HostConv(const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& b) {
  std::vector<double> out(54);
  for (int t = 0; t < 3; ++t)
    for (int y = 0; y < 3; ++y)
      for (int z = 0; z < 3; ++z)
        for (int c = 0; c < 2; ++c) {
          double value = b[static_cast<std::size_t>(c)];
          for (int kt = 0; kt < 3; ++kt)
            for (int ky = 0; ky < 3; ++ky)
              for (int kz = 0; kz < 3; ++kz) {
                const int it = t + kt - 1, iy = y + ky - 1, iz = z + kz - 1;
                if (it < 0 || it >= 3 || iy < 0 || iy >= 3 || iz < 0 || iz >= 3)
                  continue;
                for (int in = 0; in < 2; ++in)
                  value += x[static_cast<std::size_t>(((it * 3 + iy) * 3 + iz) * 2 + in)] *
                           w[static_cast<std::size_t>((((kt * 3 + ky) * 3 + kz) * 2 + in) * 2 + c)];
              }
          out[static_cast<std::size_t>(((t * 3 + y) * 3 + z) * 2 + c)] = value;
        }
  return out;
}

double Objective(
    const std::vector<std::vector<double>>& values,
    const std::vector<double>&              coefficient) {
  const auto out   = HostConv(values[2], values[0], values[1]);
  double     total = 0;
  for (std::size_t i = 0; i < 54; ++i)
    total += coefficient[i] * out[i];
  return total;
}

void ConvChecks() {
  std::vector<float>               input(54), coefficient(54), weight(108);
  std::vector<std::vector<double>> values(3);
  std::vector<double>              host_coefficient;
  for (int i = 0; i < 54; ++i) {
    input[static_cast<std::size_t>(i)]       = static_cast<float>(.2 * std::sin(.31 * i));
    coefficient[static_cast<std::size_t>(i)] = static_cast<float>(.1 * std::cos(.37 * i));
    values[2].push_back(input[static_cast<std::size_t>(i)]);
    host_coefficient.push_back(coefficient[static_cast<std::size_t>(i)]);
  }
  for (int i = 0; i < 108; ++i) {
    weight[static_cast<std::size_t>(i)] = static_cast<float>(.07 * std::sin(.23 * i + .1));
    values[0].push_back(weight[static_cast<std::size_t>(i)]);
  }
  const std::vector<float> bias{.1f, -.2f};
  values[1] = {bias[0], bias[1]};
  nn::ParameterStore p(653);
  Tensor             x = nn::from_vector(input, {1, 3, 3, 3, 2});
  {
    nn::NoGradGuard guard;
    nn::conv3d(p, "conv", x, 2, 3); // create, then assign the asymmetric coefficients
  }
  p.set("conv/W", nn::from_vector(weight, {3, 3, 3, 2, 2}));
  p.set("conv/b", nn::from_vector(bias, {2}));
  x.requires_grad_(true);
  const Tensor y           = nn::conv3d(p, "conv", x, 2, 3);
  const Tensor objective   = nn::sum(y * nn::from_vector(coefficient, {1, 3, 3, 3, 2}));
  const auto   gradient    = nn::grad(objective, {p.at("conv/W"), p.at("conv/b"), x});
  const auto   expected    = HostConv(values[2], values[0], values[1]);
  const auto   yv          = y.to_vector<float>();
  double       value_error = 0, gradient_error = 0;
  for (std::size_t i = 0; i < 54; ++i)
    value_error = std::max(value_error, std::abs(expected[i] - yv[i]));
  int coordinates = 0;
  for (std::size_t v = 0; v < 3; ++v) {
    const auto g = gradient[v].to_vector<float>();
    for (std::size_t i = 0; i < values[v].size(); ++i) {
      auto plus = values, minus = values;
      plus[v][i] += 1e-5;
      minus[v][i] -= 1e-5;
      const double derivative =
          (Objective(plus, host_coefficient) - Objective(minus, host_coefficient)) / 2e-5;
      gradient_error = std::max(gradient_error, std::abs(derivative - g[i]));
      ++coordinates;
    }
  }
  check::Require(coordinates == 164, "all108kernel+2bias+54input coordinates covered");
  check::Near(value_error, 0, 1e-7, "independent signed time-height-width cross correlation");
  check::Near(
      gradient_error,
      0,
      1e-7,
      "ALL164signed3D coordinates versus independent host finite differences");
  std::cout << " signed3x3x3 multichannel forward error=" << value_error
            << ",ALL164 kernel/bias/input host FD error=" << gradient_error << std::endl;
}

void PoolChecks() {
  std::vector<float> input(18);
  for (int i = 0; i < 18; ++i)
    input[static_cast<std::size_t>(i)] = static_cast<float>(-100 - i);
  Tensor x = nn::from_vector(input, {1, 2, 3, 3, 1});
  x.requires_grad_(true);
  const Tensor pooled = dm::nn::models::C3dPool(x, false), first = dm::nn::models::C3dPool(x, true);
  const Tensor gradient = nn::grad(nn::sum(pooled), {x})[0];
  check::Require(
      pooled.shape() == nn::Shape({1, 1, 2, 2, 1}) && first.shape() == nn::Shape({1, 2, 2, 2, 1}),
      "ceil odd spatial pool and pool1preserves time");
  const double expected[] = {-100, -102, -106, -108};
  const auto   pv         = pooled.to_vector<float>();
  for (std::size_t i = 0; i < 4; ++i)
    check::Near(
        pv[i], expected[i], 0, "right/bottom partial window excludes negative-infinity padding");
  const auto gv = gradient.to_vector<float>();
  for (int t = 0; t < 2; ++t)
    for (int y = 0; y < 3; ++y)
      for (int z = 0; z < 3; ++z)
        check::Near(
            gv[static_cast<std::size_t>((t * 3 + y) * 3 + z)],
            float(t == 0 && y % 2 == 0 && z % 2 == 0),
            0,
            "unique-max odd-edge gradient onlyactualsamples");
  // tflow's formulation: PadV2 with -inf on the odd trailing edges, then VALID pooling.
  const Tensor padded =
      nn::pad(x, {{0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant, -INFINITY);
  const auto tf_way = nn::max_pool3d(padded, 2, 2, Padding::Valid).to_vector<float>();
  for (std::size_t i = 0; i < 4; ++i)
    check::Near(tf_way[i], pv[i], 0, "SAME pool equals -inf PadV2 + VALID pool");
  std::cout
      << " Caffe-ceil3Dpool boundary values/all input gradients/temporal pool1 fixtures passed\n";
}

std::int64_t Census(const dm::nn::models::C3dConfig& cfg) {
  const int    multiples[] = {1, 2, 4, 4, 8, 8, 8, 8};
  std::int64_t total = 0, in = 3;
  for (int m : multiples) {
    const std::int64_t out = cfg.base * m;
    total += 27 * in * out + out;
    in = out;
  }
  std::int64_t time = 16, image = cfg.image;
  for (int i = 0; i < 5; ++i) {
    if (i > 0)
      time = nn::conv_output_size(time, 2, 2, 1, Padding::Same);
    image = nn::conv_output_size(image, 2, 2, 1, Padding::Same);
  }
  const std::int64_t flat = time * image * image * in;
  return total + flat * cfg.dense + cfg.dense + std::int64_t{cfg.dense} * cfg.dense + cfg.dense +
         std::int64_t{cfg.dense} * cfg.classes + cfg.classes;
}

void GraphChecks(bool run_full) {
  for (bool compact : {false, true}) {
    const dm::nn::models::C3dConfig cfg =
        compact ? dm::nn::models::C3dConfig{4, 64, 3, 32} : dm::nn::models::C3dConfig{};
    const int time[] = {16, 16, 8, 8, 4, 4, 2, 2}, full_space[] = {112, 56, 28, 28, 14, 14, 7, 7},
              small_space[] = {32, 16, 8, 8, 4, 4, 2, 2}, width[] = {1, 2, 4, 4, 8, 8, 8, 8};
    const int pt[] = {16, 8, 4, 2, 1}, ps[] = {56, 28, 14, 7, 4}, cs[] = {16, 8, 4, 2, 1},
              pc[]                     = {1, 2, 4, 8, 8};
    const std::int64_t expected_params = compact ? 114963 : 79991015;
    // Stage sizes from the TF padding rules: SAME convolutions keep every axis, ceil pools halve
    // (rounding up).
    std::int64_t t = 16, s = cfg.image;
    int          pool = 0;
    for (int i = 0; i < 8; ++i) {
      t = nn::conv_output_size(t, 3, 1, 1, Padding::Same);
      s = nn::conv_output_size(s, 3, 1, 1, Padding::Same);
      check::Require(
          t == time[i] && s == (compact ? small_space[i] : full_space[i]),
          "original C3D full/compact stage dimensions");
      if (i == 0 || i == 1 || i == 3 || i == 5 || i == 7) {
        if (i != 0)
          t = nn::conv_output_size(t, 2, 2, 1, Padding::Same);
        s = nn::conv_output_size(s, 2, 2, 1, Padding::Same);
        check::Require(
            t == pt[pool] && s == (compact ? cs[pool] : ps[pool]),
            "original C3D ceil-pool dimensions");
        ++pool;
      }
    }
    check::Require(
        Census(cfg) == expected_params,
        "independent original-eight-conv/twoFC/head parameter count");
    if (compact || run_full) {
      // Execute: every stage shape, the census and all 22 gradients.
      nn::ParameterStore p(compact ? 659 : 661);
      nn::Generator      gen(5);
      const Tensor       x   = nn::uniform({1, 16, cfg.image, cfg.image, 3}, 0.0, 1.0, gen);
      const auto         net = dm::nn::models::C3d(p, x, cfg);
      for (int i = 0; i < 8; ++i)
        check::Require(
            net.conv[static_cast<std::size_t>(i)].shape() ==
                nn::Shape(
                    {1,
                     time[i],
                     compact ? small_space[i] : full_space[i],
                     compact ? small_space[i] : full_space[i],
                     cfg.base * width[i]}),
            "executed C3D conv stage shapes");
      for (int i = 0; i < 5; ++i)
        check::Require(
            net.pools[static_cast<std::size_t>(i)].shape() ==
                nn::Shape(
                    {1, pt[i], compact ? cs[i] : ps[i], compact ? cs[i] : ps[i], cfg.base * pc[i]}),
            "executed C3D pool stage shapes");
      check::Require(
          net.flat.shape() == nn::Shape({1, compact ? 32 : 8192}) &&
              net.fc6.shape() == nn::Shape({1, cfg.dense}) &&
              net.fc7.shape() == nn::Shape({1, cfg.dense}) &&
              net.logits.shape() == nn::Shape({1, cfg.classes}),
          "flatten/FC shapes");
      check::Require(p.count() == expected_params && p.size() == 22, "executed parameter census");
      const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
      check::Require(gradients.size() == 22, "all22 graph gradients constructed");
      for (const auto& g : gradients) {
        check::Require(g.defined(), "no missing C3D learned gradient");
        for (float v : g.to_vector<float>())
          check::Require(std::isfinite(v), "finite C3D gradient");
      }
    }
    std::cout << (compact ? " compact" : " full Sports1M") << " C3D graph:" << expected_params
              << " parameters/22tensors;all8conv/5ceilpool/FC shapes and gradients verified;"
              << (compact    ? "executed"
                  : run_full ? "full-width forward/backward executed"
                             : "full-width execution NOT part of default CPU demo (full=1)")
              << "\n";
  }
}

} // namespace
