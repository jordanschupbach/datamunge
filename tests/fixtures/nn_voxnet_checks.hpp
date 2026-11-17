#pragma once
// Independent numerical and architecture regression checks retained from the
// original executable; canonical operations are called directly. No training
// driver or example-source dependency is retained here.
#include <dm/dm.hpp>
#include <iostream>
namespace {
namespace nn = dm::nn;
namespace vox {
using dm::nn::models::VoxnetRotate;
inline auto Rotate(const nn::Tensor& x, int k) {
  return VoxnetRotate(x, k);
}
using dm::nn::models::VoxnetRegularized;
inline auto Regularized(nn::ParameterStore& p, const std::vector<nn::Tensor>& g) {
  return VoxnetRegularized(p, g);
}
inline auto Forward(nn::ParameterStore& p, const nn::Tensor& x, int c) {
  return dm::nn::models::Voxnet(p, x, c);
}
} // namespace vox
namespace replay {
inline double MaxAbsDiff(const nn::Tensor& a, const nn::Tensor& b) {
  return nn::max(nn::abs(a - b)).item();
}
} // namespace replay
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
void ConvChecks() {
  std::vector<float> input(686), kernel(500), coefficient(16);
  for (int i = 0; i < 686; ++i)
    input[i] = .2 * std::sin(.31 * i);
  for (int i = 0; i < 500; ++i)
    kernel[i] = .03 * std::cos(.27 * i);
  for (int i = 0; i < 16; ++i)
    coefficient[i] = .13 * std::cos(.41 * i);
  auto              x = nn::from_vector(input, {1, 7, 7, 7, 2}).requires_grad_();
  auto              w = nn::from_vector(kernel, {5, 5, 5, 2, 2}).requires_grad_();
  auto              b = nn::from_vector(std::vector<float>{.3f, -.3f}, {2}).requires_grad_();
  nn::Conv3DOptions opt;
  opt.strides    = {2, 2, 2};
  opt.padding    = nn::Padding::Valid;
  auto y         = nn::leaky_relu(nn::conv3d(x, w, opt) + b, .1);
  auto gradients = nn::grad(nn::sum(y * nn::from_vector(coefficient, {1, 2, 2, 2, 2})), {w, b, x});
  std::vector<std::vector<double>> values{
      w.to_vector<double>(), b.to_vector<double>(), x.to_vector<double>()};
  auto host = [&](const std::vector<std::vector<double>>& values,
                  std::vector<double>*                    outputs = nullptr) {
    double total = 0;
    for (int x = 0; x < 2; ++x)
      for (int y = 0; y < 2; ++y)
        for (int z = 0; z < 2; ++z)
          for (int c = 0; c < 2; ++c) {
            double value = values[1][c];
            for (int a = 0; a < 5; ++a)
              for (int b = 0; b < 5; ++b)
                for (int d = 0; d < 5; ++d)
                  for (int ci = 0; ci < 2; ++ci)
                    value += values[2][(((2 * x + a) * 7 + 2 * y + b) * 7 + 2 * z + d) * 2 + ci] *
                             values[0][(((a * 5 + b) * 5 + d) * 2 + ci) * 2 + c];
            value = value >= 0 ? value : .1 * value;
            int i = ((x * 2 + y) * 2 + z) * 2 + c;
            total += value * coefficient[i];
            if (outputs)
              outputs->push_back(value);
          }
    return total;
  };
  std::vector<double> expected;
  host(values, &expected);
  const auto actual = y.to_vector<float>();
  double     error = 0, derivative = 0;
  int        coordinates = 0;
  for (int i = 0; i < 16; ++i)
    error = std::max(error, std::abs(expected[i] - actual[i]));
  for (int variable = 0; variable < 3; ++variable) {
    const auto g = gradients[variable].to_vector<float>();
    for (std::size_t i = 0; i < values[variable].size(); ++i) {
      auto plus = values, minus = values;
      plus[variable][i] += 1e-5;
      minus[variable][i] -= 1e-5;
      derivative = std::max(derivative, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
      ++coordinates;
    }
  }
  check::Require(coordinates == 1188, "all500kernel+2bias+686input coordinates");
  check::Near(error, 0, 3e-7, "all16 stride2 valid5cube leaky outputs");
  check::Near(derivative, 0, 2e-7, "all1188 independent host finite differences");
  std::cout << "VoxNet5cube/stride2/leaky.1:forward=" << error << ",all1188 FD error=" << derivative
            << std::endl;
}
void RotationPoolChecks() {
  std::vector<float> input(27);
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y)
      for (int z = 0; z < 3; ++z)
        input[(x * 3 + y) * 3 + z] = 100 * x + 10 * y + z;
  auto tensor = nn::from_vector(input, {1, 3, 3, 3, 1}).requires_grad_();
  for (int k = 0; k < 4; ++k) {
    auto v = vox::Rotate(tensor, k).to_vector<float>();
    for (int x = 0; x < 3; ++x)
      for (int y = 0; y < 3; ++y)
        for (int z = 0; z < 3; ++z) {
          int a = x, b = y;
          for (int step = 0; step < k; ++step) {
            int previous = a;
            a            = b;
            b            = 2 - previous;
          }
          check::Near(
              v[(x * 3 + y) * 3 + z],
              100 * a + 10 * b + z,
              0,
              "every C4 coordinate,preserved third axis");
        }
  }
  check::Near(
      replay::MaxAbsDiff(vox::Rotate(vox::Rotate(tensor, 3), 1), tensor),
      0,
      0,
      "four turns compose to identity");
  auto rotation_gradient = nn::grad(nn::sum(vox::Rotate(tensor, 1)), {tensor})[0];
  for (float x : rotation_gradient.to_vector<float>())
    check::Near(x, 1, 0, "all27 rotation input gradients");
  std::vector<float> negatives(64);
  for (int i = 0; i < 64; ++i)
    negatives[i] = -1 - i;
  auto x      = nn::from_vector(negatives, {1, 4, 4, 4, 1}).requires_grad_();
  auto pooled = nn::max_pool3d(x, 2, 2, nn::Padding::Valid);
  auto grad = nn::grad(nn::sum(pooled), {x})[0].to_vector<float>(), v = pooled.to_vector<float>();
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        check::Near(
            v[(x * 2 + y) * 2 + z],
            -1 - ((2 * x * 4 + 2 * y) * 4 + 2 * z),
            0,
            "unique real-sample maximum");
  for (int x = 0; x < 4; ++x)
    for (int y = 0; y < 4; ++y)
      for (int z = 0; z < 4; ++z)
        check::Near(
            grad[(x * 4 + y) * 4 + z],
            x % 2 == 0 && y % 2 == 0 && z % 2 == 0 ? 1 : 0,
            0,
            "all64 pool input derivatives");
  std::cout
      << "C4 all108 coordinates/closure/27 input gradients,3Dpool8 values/all64 derivatives pass\n";
}
void RegularizerChecks() {
  nn::ParameterStore p;
  p.set("layer/W", nn::from_vector(std::vector<float>{2, -1, .5}, {3}));
  p.set("layer/b", nn::from_vector(std::vector<float>{.4f}, {1}));
  const auto coefficient =
      std::vector<double>{double(.3f), double(-.2f), double(.7f), double(1.2f)};
  auto linear =
      nn::sum(p.at("layer/W") * nn::from_vector(std::vector<float>{.3f, -.2f, .7f}, {3})) +
      nn::sum(p.at("layer/b") * 1.2);
  auto                g = vox::Regularized(p, nn::grad(linear, p.parameters()));
  std::vector<double> values{2, -1, .5, double(.4f)}, actual = g[0].to_vector<double>();
  actual.push_back(g[1].item());
  auto host = [&](const std::vector<double>& x) {
    double total = 0;
    for (int i = 0; i < 4; ++i)
      total += coefficient[i] * x[i];
    for (int i = 0; i < 3; ++i)
      total += .001 * x[i] * x[i];
    return total;
  };
  double error = 0;
  for (int i = 0; i < 4; ++i) {
    auto plus = values, minus = values;
    plus[i] += 1e-6;
    minus[i] -= 1e-6;
    error = std::max(error, std::abs((host(plus) - host(minus)) / 2e-6 - actual[i]));
  }
  check::Near(
      error, 0, 1e-7, "all4 host finite differences,exact weight-only L2,unregularized bias");
  std::cout << "weight-only L2 hostFD error=" << error << std::endl;
}
void GraphChecks() {
  for (int classes : {40, 3}) {
    nn::ParameterStore p(941);
    auto               net = vox::Forward(p, nn::zeros({1, 32, 32, 32, 1}), classes);
    check::Require(
        net.conv1.shape() == nn::Shape({1, 14, 14, 14, 32}) &&
            net.conv2.shape() == nn::Shape({1, 12, 12, 12, 32}) &&
            net.pool.shape() == nn::Shape({1, 6, 6, 6, 32}) &&
            net.flat.shape() == nn::Shape({1, 6912}) && net.fc1.shape() == nn::Shape({1, 128}) &&
            net.logits.shape() == nn::Shape({1, classes}),
        "canonical 32->14->12->6/FC128/head geometry");
    const auto expected =
        5 * 5 * 5 * 32 + 32 + 3 * 3 * 3 * 32 * 32 + 32 + 6912 * 128 + 128 + 128 * classes + classes;
    check::Require(
        p.count() == expected && p.size() == 8 && expected == (classes == 40 ? 921736 : 916963),
        "independent all4-layer count");
    const auto g = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
    for (const auto& v : g)
      check::Require(
          v.defined() && nn::all(nn::is_finite(v)).item<bool>(),
          "all8 canonical gradients exist and are finite");
    std::cout << "VoxNet " << classes << " classes:" << expected
              << " parameters/8 tensors;all shapes/counts/gradients pass\n";
  }
}

} // namespace
