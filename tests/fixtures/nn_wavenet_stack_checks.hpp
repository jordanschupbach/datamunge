#pragma once
// Independent original host formula and complete-graph regression checks.
#include "nn_sequence_check_support.hpp"
namespace {
std::vector<std::array<double, 3>> Host(const Weights& w) {
  const int           ids[] = {0, 1, 2, 0, 2, 1};
  std::vector<double> x(6), skip(6);
  for (int t = 0; t < 6; ++t) {
    x[t] = w[1][0];
    for (int k = 0; k < 2; ++k) {
      const int position = t + k - 1;
      if (position >= 0)
        x[t] += w[0][3 * k + ids[position]];
    }
  }
  for (int layer = 0; layer < 2; ++layer) {
    const int           first = layer == 0 ? 2 : 8;
    std::vector<double> gate(6);
    for (int t = 0; t < 6; ++t) {
      double joined[2] = {w[first + 1][0], w[first + 1][1]};
      for (int k = 0; k < 2; ++k) {
        const int position = t + (k - 1) * (1 << layer);
        if (position >= 0)
          for (int c = 0; c < 2; ++c)
            joined[c] += x[position] * w[first][2 * k + c];
      }
      gate[t] = std::tanh(joined[0]) / (1 + std::exp(-joined[1]));
      skip[t] += gate[t] * w[first + 2][0] + w[first + 3][0];
    }
    if (layer == 0)
      for (int t = 0; t < 6; ++t)
        x[t] += gate[t] * w[6][0] + w[7][0];
  }
  std::vector<std::array<double, 3>> out(6);
  for (int t = 0; t < 6; ++t) {
    const double hidden = std::max(0., std::max(0., skip[t]) * w[12][0] + w[13][0]);
    for (int c = 0; c < 3; ++c)
      out[t][c] = hidden * w[14][c] + w[15][c];
  }
  return out;
}
double Objective(const Weights& w) {
  const auto out = Host(w);
  return .3 * out.back()[0] - .2 * out.back()[1] + .1 * out.back()[2];
}

void CodecLossChecks() {
  using wavestack::MuLawDecode;
  using wavestack::MuLawEncode;
  check::Require(
      MuLawEncode(-1) == 0 && MuLawEncode(0) == 128 && MuLawEncode(1) == 255 &&
          MuLawEncode(-.5) == 16 && MuLawEncode(.5) == 239,
      "independent mu-law endpoint/half/zero values");
  for (int q = 0; q < 256; ++q)
    check::Require(
        MuLawEncode(MuLawDecode(q)) == q, "every mu-law quantization-bin inverse roundtrip");
  check::Near(MuLawDecode(0), -1, 1e-12, "inverse negative endpoint");
  check::Near(MuLawDecode(255), 1, 1e-12, "inverse positive endpoint");
  const std::vector<float> values{1, 2, 3, -1, 0, 1, 1000, 999, 998, -2, 0, 2};
  const std::vector<int>   y{0, 1, 1, 2};
  Tensor                   z = nn::from_vector(values, {1, 4, 3});
  z.requires_grad_(true);
  const Tensor loss     = wavestack::WaveNll(z, nn::from_vector(y, {1, 4}), 3, 2);
  const auto   g        = nn::grad(loss, {z})[0].to_vector<double>();
  double       expected = 0;
  for (int t = 0; t < 4; ++t) {
    double maximum = -INFINITY, sum = 0;
    for (int c = 0; c < 3; ++c)
      maximum = std::max(maximum, double(values[t * 3 + c]));
    for (int c = 0; c < 3; ++c)
      sum += std::exp(values[t * 3 + c] - maximum);
    if (t >= 2)
      expected += (maximum + std::log(sum) - values[t * 3 + y[t]]) / 2;
    for (int c = 0; c < 3; ++c) {
      const double derivative =
          t < 2 ? 0 : (std::exp(values[t * 3 + c] - maximum) / sum - (c == y[t])) / 2;
      check::Near(
          g[t * 3 + c], derivative, 1e-7, "every masked stable NLL logit derivative/warmup zero");
    }
  }
  check::Near(loss.item<double>(), expected, 2e-7, "independent mean over ONLY scored positions");
  std::cout << " mu-law256 inverse/all-bin tests and stable warmup-masked NLL/all-logit "
               "derivatives passed\n";
}

void HostChecks() {
  nn::ParameterStore p(563);
  std::vector<float> onehot(18, 0.f);
  const int          ids[] = {0, 1, 2, 0, 2, 1};
  for (int t = 0; t < 6; ++t)
    onehot[t * 3 + ids[t]] = 1;
  const Tensor                     input = nn::from_vector(onehot, {1, 6, 3});
  const wavestack::WaveStackConfig config{1, 1, 2, 1, 3};
  wavestack::WaveStack(p, input, config);
  const auto weights =
      sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape&) {
        return .13f + .04f * static_cast<float>(std::sin(.7 * i + .9 * v));
      });
  const auto net       = wavestack::WaveStack(p, input, config);
  const auto endpoint  = nn::select(net.logits, 1, 5);
  const auto gradients = nn::grad(
      nn::sum(nn::mul(endpoint, nn::from_vector(std::vector<float>{.3f, -.2f, .1f}, {1, 3}))),
      p.parameters());
  const auto expected    = Host(weights);
  const auto z           = net.logits.to_vector<double>();
  double     value_error = 0;
  for (int t = 0; t < 6; ++t)
    for (int c = 0; c < 3; ++c)
      value_error = std::max(value_error, std::abs(expected[t][c] - z[t * 3 + c]));
  int          coordinates = 0;
  const double gradient_error =
      sentence::HostGradientError(weights, gradients, Objective, coordinates);
  check::Require(
      coordinates == 33 && p.size() == 16, "all33 tiny coordinates in16 learned tensors");
  check::Near(
      value_error, 0, 1e-7, "independent causal/tanh-sigmoid/residual/skip/post-head values");
  check::Near(
      gradient_error,
      0,
      2e-6,
      "EVERY tiny initial/gate/residual/skip/post/head host finite difference");
  std::cout << " independent gated-stack forward error=" << value_error
            << ",ALL33 host parameter-coordinate FD error=" << gradient_error << std::endl;
}

void CausalChecks() {
  nn::ParameterStore               p(569);
  const wavestack::WaveStackConfig config{1, 1, 2, 1, 3};
  wavestack::WaveStack(p, nn::zeros({1, 8, 3}), config);
  for (const auto& name : p.names())
    p.set(name, nn::full(p.at(name).shape(), .1));
  Tensor x = nn::full({1, 8, 3}, .2);
  x.requires_grad_(true);
  const auto net = wavestack::WaveStack(p, x, config);
  const auto g   = nn::grad(nn::sum(nn::select(net.logits, 1, 7)), {x})[0].to_vector<double>();
  const auto reference = net.logits.to_vector<double>();
  check::Require(
      wavestack::WaveStackField(config) == 5, "initial2 plus gated dilations1/2 givesRF5");
  for (int t = 0; t < 8; ++t)
    for (int c = 0; c < 3; ++c) {
      if (t < 3)
        check::Near(g[t * 3 + c], 0, 0, "outside RF exactlyzero Jacobian");
      else
        check::Require(g[t * 3 + c] > 0, "every reachable lag/channel through gates and skips");
    }
  std::vector<float> input(24, .2f);
  for (int t = 4; t < 8; ++t)
    for (int c = 0; c < 3; ++c)
      input[t * 3 + c] += .4f;
  const auto changed =
      wavestack::WaveStack(p, nn::from_vector(input, {1, 8, 3}), config).logits.to_vector<double>();
  double later = 0;
  for (int t = 0; t < 8; ++t)
    for (int c = 0; c < 3; ++c) {
      const double delta = std::abs(changed[t * 3 + c] - reference[t * 3 + c]);
      if (t < 4)
        check::Near(delta, 0, 0, "gated prefix independent of future");
      else
        later += delta;
    }
  check::Require(later > 0, "later predictions depend on changed later inputs");
  std::cout << " exact RF5 input-Jacobian/all reachable lags/future-zero gated fixtures passed\n";
}

void FullChecks() {
  nn::ParameterStore p(571);
  std::vector<int>   ids(160);
  for (int t = 0; t < 160; ++t)
    ids[t] = (t * 37) % 256;
  const wavestack::WaveStackConfig config;
  const auto                       net =
      wavestack::WaveStack(p, nn::one_hot(nn::from_vector(ids, {1, 160}), 256), config);
  const auto norms =
      sentence::SquaredNorms(nn::grad(nn::mean(nn::square(net.logits)), p.parameters()));
  for (int i = 0; i < 12; ++i)
    check::Require(
        net.gates[i].shape() == nn::Shape{1, 160, 32},
        "every gated layer preserves160positions/R32");
  check::Require(
      net.logits.shape() == nn::Shape{1, 160, 256} && p.count() == 124096 && p.size() == 76 &&
          wavestack::WaveStackField(config) == 128,
      "independent full R32/S64 categorical parameter count/RF128");
  for (double n : norms)
    check::Require(
        std::isfinite(n) && n > 0, "all76 full parameter tensor derivatives finite/nonzero");
  std::cout << " full gated stack124096 parameters/76tensors,12layers,R32/S64,mu-law256,RF128;all "
               "stage shapes/learned gradients passed\n";
}

} // namespace
