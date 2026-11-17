#pragma once
// Independent original host formula and complete-graph regression checks.
#include "nn_sequence_check_support.hpp"
namespace {
std::array<double, 2> Host(const Weights& weights) {
  constexpr double means[3][2]     = {{-.2, 0}, {-.1, .05}, {.1, 0}},
                   variances[3][2] = {{1.3, 1}, {1.2, .7}, {.9, 1}};
  // Constants are rounded to the same float values stored in native state.
  std::vector<std::vector<double>> input(7, std::vector<double>(1));
  for (int t = 0; t < 7; ++t)
    input[t][0] = (float(.7 * std::cos(.8 * t)) - float(.3)) / float(1.2);
  int incoming = 1;
  for (int layer = 0; layer < 3; ++layer) {
    const int width  = layer == 1 ? 2 : 1,
              kernel = layer == 0   ? 8
                       : layer == 1 ? 5
                                    : 3,
              pad    = (kernel - 1) / 2;
    std::vector<std::vector<double>> next(7, std::vector<double>(width));
    for (int t = 0; t < 7; ++t)
      for (int c = 0; c < width; ++c) {
        double value = weights[4 * layer + 1][c];
        for (int k = 0; k < kernel; ++k) {
          const int position = t + k - pad;
          if (position < 0 || position >= 7)
            continue;
          for (int j = 0; j < incoming; ++j)
            value += input[position][j] * weights[4 * layer][(k * incoming + j) * width + c];
        }
        value = weights[4 * layer + 2][c] * (value - float(means[layer][c])) /
                    std::sqrt(float(variances[layer][c]) + float(.001)) +
                weights[4 * layer + 3][c];
        next[t][c] = std::max(0., value);
      }
    input    = next;
    incoming = width;
  }
  double average = 0;
  for (int t = 0; t < 7; ++t)
    average += input[t][0] / 7;
  return {average * weights[12][0] + weights[13][0], average * weights[12][1] + weights[13][1]};
}
double Objective(const Weights& weights) {
  const auto z = Host(weights);
  return .3 * z[0] - .2 * z[1];
}

void HostChecks() {
  nn::ParameterStore         p(433), state(439);
  const series::SeriesConfig config{1, .3f, 1.2f};
  std::vector<float>         x(7);
  for (int t = 0; t < 7; ++t)
    x[t] = static_cast<float>(.7 * std::cos(.8 * t));
  const Tensor input = nn::from_vector(x, {1, 7, 1});
  series::SeriesCnn(p, state, input, config);
  const auto weights =
      sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape&) {
        const bool scale = v < 12 && v % 4 == 2;
        return scale ? 1.f + .1f * static_cast<float>(std::sin(i + v))
                     : .13f + .04f * static_cast<float>(std::sin(.6 * i + .8 * v));
      });
  const std::vector<std::vector<float>> fixed = {
      {.3f}, {1.2f}, {-.2f}, {1.3f}, {-.1f, .05f}, {1.2f, .7f}, {.1f}, {.9f}};
  for (std::size_t v = 0; v < fixed.size(); ++v)
    state.set(
        state.names()[v], nn::reshape(nn::from_vector(fixed[v]), state.parameters()[v].shape()));
  const auto net      = series::SeriesCnn(p, state, input, config);
  const auto gradient = nn::grad(
      nn::sum(nn::mul(net.logits, nn::from_vector(std::vector<float>{.3f, -.2f}, {1, 2}))),
      p.parameters());
  const auto expected = Host(weights);
  const auto z = net.logits.to_vector<double>(), normalized = net.normalized.to_vector<double>();
  double     value_error = 0;
  for (int c = 0; c < 2; ++c)
    value_error = std::max(value_error, std::abs(expected[c] - z[c]));
  for (int t = 0; t < 7; ++t)
    check::Near(
        normalized[t],
        (double(x[t]) - float(.3)) / float(1.2),
        1e-7,
        "saved fit-global centering/scaling");
  int          coordinates = 0;
  const double gradient_error =
      sentence::HostGradientError(weights, gradient, Objective, coordinates);
  check::Require(
      coordinates == 40 && p.size() == 14 && state.size() == 8,
      "all40tiny learned coordinates and8 normalization/BN tensors");
  check::Near(value_error, 0, 2e-7, "independent even8/odd5/3 SAME/BN/ReLU/GAP/head values");
  check::Near(gradient_error, 0, 2e-6, "EVERY convolution/BN-affine/head host finite difference");
  std::cout << " independent temporal8/5/3 SAME/BN/GAP/head error=" << value_error
            << ",ALL40 parameter-coordinate host FD error=" << gradient_error
            << ";nontrivial saved mean/std/BN fixtures passed\n";
}

void BaselineChecks() {
  std::vector<float> f(3 * 150), t(2 * 150);
  for (int n = 0; n < 3; ++n)
    for (int i = 0; i < 150; ++i)
      f[n * 150 + i] = 2.f * n;
  for (int i = 0; i < 150; ++i) {
    t[i]       = 1;
    t[150 + i] = 3.8f;
  }
  const series::SeriesData fit{
      nn::from_vector(f, {3, 150, 1}),
      nn::from_vector(std::vector<int>{0, 1, 1}, {3}),
      nn::zeros({3}, nn::DType::Int32)};
  const series::SeriesData test{
      nn::from_vector(t, {2, 150, 1}),
      nn::zeros({2}, nn::DType::Int32),
      nn::zeros({2}, nn::DType::Int32)};
  const auto nearest  = series::SeriesReference(fit, test, false).to_vector<double>(),
             centroid = series::SeriesReference(fit, test, true).to_vector<double>();
  check::Require(
      nearest[0] == 1 && nearest[3] == 1, "1NN exact distances and first fitting-row tie rule");
  check::Near(centroid[0], -1, 0, "known class0 centroid MSE");
  check::Near(centroid[1], -4, 0, "class1 centroid fits ONLY retained training rows");
  check::Near(centroid[3], -.64, 1e-7, "known centroid off-center MSE");
  std::cout << " independent1NN/tie/fitting-centroid fixtures passed\n";
}

void FullChecks() {
  nn::ParameterStore p(443), state(449);
  std::vector<float> x(150);
  for (int t = 0; t < 150; ++t)
    x[t] = static_cast<float>(std::sin(t * .13) + .2 * std::cos(t * .37));
  const auto network = series::SeriesCnn(p, state, nn::from_vector(x, {1, 150, 1}));
  const auto norms =
      sentence::SquaredNorms(nn::grad(nn::sum(nn::square(network.logits)), p.parameters()));
  for (int i = 0; i < 3; ++i)
    check::Require(
        network.stages[i].shape() == nn::Shape{1, 150, i == 1 ? 256 : 128},
        "full SAME temporal lengths/widths");
  check::Require(
      network.logits.shape() == nn::Shape{1, 2} && p.count() == 264962 && p.size() == 14 &&
          state.size() == 8,
      "independent full parameter/state count and binary head");
  for (double n : norms)
    check::Require(
        std::isfinite(n) && n > 0, "every full inference learned tensor adjoint finite/nonzero");
  std::cout << " full temporal FCN264962 parameters/14 learned "
               "tensors+8state,channels128/256/128,T150,kernels8/5/3;all stage shapes/learned "
               "gradients passed\n";
}

} // namespace
