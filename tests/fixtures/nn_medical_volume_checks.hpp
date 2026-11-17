#pragma once
// Independent numerical and architecture regression checks retained from the
// original executable; canonical operations are called directly. No training
// driver or example-source dependency is retained here.
#include <dm/dm.hpp>
#include <iostream>
namespace {
namespace nn = dm::nn;
namespace mv = dm::nn::models;
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
struct HostVolume {
  int                 d, h, w, c;
  std::vector<double> v;
};

HostVolume Conv(
    const HostVolume&          x,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    int                        kernel,
    int                        stride) {
  HostVolume out{
      (x.d + stride - 1) / stride,
      (x.h + stride - 1) / stride,
      (x.w + stride - 1) / stride,
      static_cast<int>(bias.size()),
      {}};
  out.v.resize(static_cast<std::size_t>(out.d * out.h * out.w * out.c));
  const int pd = std::max(0, (out.d - 1) * stride + kernel - x.d) / 2,
            ph = std::max(0, (out.h - 1) * stride + kernel - x.h) / 2,
            pw = std::max(0, (out.w - 1) * stride + kernel - x.w) / 2;
  for (int z = 0; z < out.d; ++z)
    for (int y = 0; y < out.h; ++y)
      for (int a = 0; a < out.w; ++a)
        for (int o = 0; o < out.c; ++o) {
          double value = bias[static_cast<std::size_t>(o)];
          for (int kz = 0; kz < kernel; ++kz)
            for (int ky = 0; ky < kernel; ++ky)
              for (int kx = 0; kx < kernel; ++kx) {
                const int iz = z * stride + kz - pd, iy = y * stride + ky - ph,
                          ix = a * stride + kx - pw;
                if (iz < 0 || iy < 0 || ix < 0 || iz >= x.d || iy >= x.h || ix >= x.w)
                  continue;
                for (int c = 0; c < x.c; ++c)
                  value += x.v[static_cast<std::size_t>(((iz * x.h + iy) * x.w + ix) * x.c + c)] *
                           weight[static_cast<std::size_t>(
                               (((kz * kernel + ky) * kernel + kx) * x.c + c) * out.c + o)];
              }
          out.v[static_cast<std::size_t>(((z * out.h + y) * out.w + a) * out.c + o)] = value;
        }
  return out;
}

void NumericalChecks() {
  nn::ParameterStore p(1063);
  Tensor             input = nn::zeros({1, 5, 6, 7, 2});
  {
    nn::NoGradGuard guard;
    mv::MedicalResidual(p, "block", input, 2);
  }
  std::vector<std::vector<double>> values;
  for (std::size_t index = 0; index < p.size(); ++index) {
    const auto&        shape = p.parameters()[index].shape();
    const std::int64_t n     = p.parameters()[index].numel();
    std::vector<float> t(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
      t[static_cast<std::size_t>(i)] =
          index % 2 ? static_cast<float>(.04 * std::cos(.7 * i + index))
                    : static_cast<float>(.07 * std::sin(.19 * i + .3 * index));
    p.set(p.names()[index], nn::from_vector(t, shape));
    values.emplace_back(t.begin(), t.end());
  }
  std::vector<float> x(420), coefficient(72);
  for (int i = 0; i < 420; ++i)
    x[static_cast<std::size_t>(i)] = static_cast<float>(.4f * std::sin(.37 * i));
  for (int i = 0; i < 72; ++i)
    coefficient[static_cast<std::size_t>(i)] = static_cast<float>(.1f * std::cos(.43 * i));
  values.emplace_back(x.begin(), x.end());
  input = nn::from_vector(x, {1, 5, 6, 7, 2});
  input.requires_grad_(true);
  const Tensor y         = mv::MedicalResidual(p, "block", input, 2);
  const Tensor objective = nn::sum(y * nn::from_vector(coefficient, {1, 3, 3, 4, 2}));
  auto         vars      = p.parameters();
  vars.push_back(input);
  const auto gradients = nn::grad(objective, vars);
  const auto host      = [&](const std::vector<std::vector<double>>& v,
                        std::vector<double>*                    result = nullptr) {
    const HostVolume in{5, 6, 7, 2, v[6]};
    auto             first = Conv(in, v[0], v[1], 3, 2);
    for (auto& q : first.v)
      q = std::max(0., q);
    const auto second = Conv(first, v[2], v[3], 3, 1), skip = Conv(in, v[4], v[5], 1, 2);
    double     total = 0;
    for (std::size_t i = 0; i < second.v.size(); ++i) {
      const double q = std::max(0., second.v[i] + skip.v[i]);
      total += q * coefficient[i];
      if (result)
        result->push_back(q);
    }
    return total;
  };
  std::vector<double> expected;
  host(values, &expected);
  const auto yv    = y.to_vector<float>();
  double     error = 0, derivative = 0;
  int        coordinates = 0;
  for (std::size_t i = 0; i < expected.size(); ++i)
    error = std::max(error, std::abs(expected[i] - yv[i]));
  for (std::size_t v = 0; v < values.size(); ++v) {
    const auto g = gradients[v].to_vector<float>();
    for (std::size_t i = 0; i < values[v].size(); ++i) {
      auto plus = values, minus = values;
      plus[v][i] += 1e-5;
      minus[v][i] -= 1e-5;
      derivative = std::max(derivative, std::abs((host(plus) - host(minus)) / 2e-5 - g[i]));
      ++coordinates;
    }
  }
  check::Require(
      coordinates == 646 && expected.size() == 72,
      "all2163D+4projection weights/6biases/420inputs and72outputs");
  check::Near(error, 0, 2e-7, "all72 odd/even SAME3D residual outputs");
  check::Near(derivative, 0, 3e-7, "ALL646 independent host finite differences");
  std::cout << "medical residual3D fixture:all72outputs error=" << error
            << ",ALL646 conv/projection/bias/input hostFD error=" << derivative
            << ";odd/even SAME stride2 padding passed\n";
}

void GraphChecks() {
  nn::ParameterStore p(1069), state(1071);
  nn::Generator      gen(5);
  const auto net = mv::MedicalVolume(p, state, nn::uniform({2, 28, 28, 28, 1}, 0.0, 1.0, gen));
  check::Require(
      net.stages[0].shape() == nn::Shape({2, 14, 14, 14, 4}) &&
          net.stages[1].shape() == nn::Shape({2, 7, 7, 7, 8}) &&
          net.stages[2].shape() == nn::Shape({2, 4, 4, 4, 16}) &&
          net.global.shape() == nn::Shape({2, 16}) && net.logits.shape() == nn::Shape({2, 2}),
      "independent medical endpoint shape");
  std::int64_t count = 0;
  int          in    = 1;
  for (int out : {4, 8, 16}) {
    count += 27 * in * out + out + 27 * out * out + out + in * out + out;
    in = out;
  }
  count += 16 * 2 + 2;
  check::Require(
      count == 13782 && p.count() == count && p.size() == 20 && state.count() == 2,
      "independent sixconv/threeprojection/head inventory");
  std::cout << "custom compact medical3D CNN:28->14->7->4,GAP16/twologits,13782 "
               "parameters/20learned+2normalization "
               "tensors;endpoint/count checks passed\n";
}

/// Loads TF's trained named arrays into our model and replays its saved test cubes.

} // namespace
