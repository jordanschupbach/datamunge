#pragma once
// Independent original host formula and complete-graph regression checks.
#include "nn_sequence_check_support.hpp"
namespace {
std::vector<std::array<double, 2>> Host(const Weights& weights) {
  std::vector<double> x(7);
  for (int t = 0; t < 7; ++t)
    x[t] = (double(float(1 + .3 * std::sin(.7 * t))) - float(.2)) / float(1.3);
  int variable = 0;
  for (int level = 0; level < 2; ++level) {
    const auto residual = x;
    for (int layer = 0; layer < 2; ++layer) {
      const auto&  v    = weights[variable];
      const double gain = weights[variable + 1][0], bias = weights[variable + 2][0],
                   norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + 1e-12);
      variable += 3;
      std::vector<double> next(7);
      for (int t = 0; t < 7; ++t) {
        double value = bias;
        for (int k = 0; k < 2; ++k) {
          const int position = t + (k - 1) * (1 << level);
          if (position >= 0)
            value += x[position] * v[k] * gain / norm;
        }
        next[t] = std::max(0., value);
      }
      x = next;
    }
    for (int t = 0; t < 7; ++t)
      x[t] = std::max(0., x[t] + residual[t]);
  }
  std::vector<std::array<double, 2>> out(7);
  for (int t = 0; t < 7; ++t)
    for (int c = 0; c < 2; ++c)
      out[t][c] = x[t] * weights[12][c] + weights[13][c];
  return out;
}
double Objective(const Weights& weights) {
  const auto out = Host(weights);
  return .3 * out.back()[0] - .2 * out.back()[1];
}

void HostChecks() {
  nn::ParameterStore p(479), state(487);
  std::vector<float> x(7);
  for (int t = 0; t < 7; ++t)
    x[t] = static_cast<float>(1 + .3 * std::sin(.7 * t));
  const Tensor         input = nn::from_vector(x, {1, 7, 1});
  const tcn::TcnConfig config{1, 2, 2, .2f, 1.3f};
  tcn::Tcn(p, state, input, config);
  const auto weights =
      sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape&) {
        return v < 12 && v % 3 == 1 ? .7f + .03f * static_cast<float>(std::cos(v))
                                    : .15f + .05f * static_cast<float>(std::sin(i * .8 + v * .5));
      });
  const auto net      = tcn::Tcn(p, state, input, config);
  const auto gradient = nn::grad(
      nn::sum(nn::mul(net.logits, nn::from_vector(std::vector<float>{.3f, -.2f}, {1, 2}))),
      p.parameters());
  const auto expected    = Host(weights);
  const auto z           = net.sequence.to_vector<double>();
  double     value_error = 0;
  for (int t = 0; t < 7; ++t)
    for (int c = 0; c < 2; ++c)
      value_error = std::max(value_error, std::abs(expected[t][c] - z[2 * t + c]));
  int          coordinates = 0;
  const double gradient_error =
      sentence::HostGradientError(weights, gradient, Objective, coordinates);
  check::Require(
      coordinates == 20 && p.size() == 14, "all20 tiny weightnorm/gain/bias/head coordinates");
  check::Near(
      value_error, 0, 2e-6, "independent two-level causal weightnorm/ReLU/residual/sequence head");
  check::Near(gradient_error, 0, 3e-6, "EVERY v/g/b/head host finite difference");
  std::cout << " independent TCN every-time forward error=" << value_error
            << ",ALL20 parameter-coordinate host FD error=" << gradient_error << std::endl;
}

void CausalityChecks() {
  nn::ParameterStore   p(491), state(499);
  const tcn::TcnConfig config{1, 3, 2, 0, 1};
  tcn::Tcn(p, state, nn::zeros({1, 20, 1}), config);
  for (const auto& name : p.names()) {
    const double value = name.find("/v") != std::string::npos   ? .3
                         : name.find("/g") != std::string::npos ? 1.
                                                                : .1;
    p.set(name, nn::full(p.at(name).shape(), value));
  }
  Tensor x = nn::full({1, 20, 1}, 2.0);
  x.requires_grad_(true);
  const auto net = tcn::Tcn(p, state, x, config);
  const auto g   = nn::grad(nn::sum(nn::select(net.hidden, 1, 19)), {x})[0].to_vector<double>();
  const auto reference = net.sequence.to_vector<double>();
  check::Require(tcn::TcnReceptiveField(config) == 15, "two convs per dilation give15-position RF");
  for (int t = 0; t < 20; ++t) {
    if (t < 5)
      check::Near(g[t], 0, 0, "outside causal receptive field Jacobian exactly zero");
    else
      check::Require(g[t] > 0, "EVERY reachable lag in RF has positive derivative");
  }
  std::vector<float> input(20, 2.f);
  for (int t = 8; t < 20; ++t)
    input[t] += t * .2f;
  const auto changed =
      tcn::Tcn(p, state, nn::from_vector(input, {1, 20, 1}), config).sequence.to_vector<double>();
  double future_change = 0;
  for (int t = 0; t < 20; ++t)
    for (int c = 0; c < 2; ++c) {
      const double difference = std::abs(changed[2 * t + c] - reference[2 * t + c]);
      if (t < 8)
        check::Near(difference, 0, 0, "future perturbation cannot affect prefix outputs");
      else
        future_change += difference;
    }
  check::Require(future_change > 0, "future input affects its own later causal output");
  const auto prefix =
      tcn::Tcn(p, state, nn::full({1, 8, 1}, 2.0), config).logits.to_vector<double>();
  for (int c = 0; c < 2; ++c)
    check::Near(
        prefix[c], reference[7 * 2 + c], 1e-6, "prefix last output equals full same-time output");
  std::cout << " exact RF15 input Jacobian/all reachable lags/future zero/prefix equivalence "
               "fixtures passed\n";
}

void FullChecks() {
  nn::ParameterStore   p(503), state(509);
  const tcn::TcnConfig config;
  std::vector<float>   x(150);
  for (int t = 0; t < 150; ++t)
    x[t] = static_cast<float>(.5 + std::sin(t * .11));
  const auto network = tcn::Tcn(p, state, nn::from_vector(x, {1, 150, 1}), config);
  const auto norms =
      sentence::SquaredNorms(nn::grad(nn::sum(nn::square(network.logits)), p.parameters()));
  for (int i = 0; i < 6; ++i)
    check::Require(
        network.blocks[i].shape() == nn::Shape{1, 150, 64},
        "causal dilation preserves time at EVERY residual block");
  check::Require(
      network.sequence.shape() == nn::Shape{1, 150, 2} && p.count() == 137154 && p.size() == 40 &&
          state.size() == 2 && tcn::TcnReceptiveField(config) == 253,
      "independent full parameter/state count and RF253");
  for (double n : norms)
    check::Require(std::isfinite(n) && n > 0, "all40 full learned tensor gradients finite/nonzero");
  std::cout << " full TCN137154 parameters/40 tensors+2state,six levelsC64,k3,RF253;all block "
               "shapes and every learned gradient passed\n";
}

} // namespace
