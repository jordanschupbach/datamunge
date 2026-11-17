#pragma once
// Independent original host formula and complete-graph regression checks.
#include "nn_sequence_check_support.hpp"
namespace {
using Matrix = std::vector<std::vector<double>>;

Matrix Conv(
    const Matrix&              x,
    const std::vector<double>& w,
    const std::vector<double>& bias,
    int                        kernel,
    int                        stride,
    int                        outgoing,
    bool                       relu) {
  const int incoming = static_cast<int>(x.front().size()), time = static_cast<int>(x.size()),
            length = (time + stride - 1) / stride,
            pad    = std::max(0, (length - 1) * stride + kernel - time) / 2;
  Matrix out(length, std::vector<double>(outgoing));
  for (int t = 0; t < length; ++t)
    for (int c = 0; c < outgoing; ++c) {
      double value = bias[c];
      for (int k = 0; k < kernel; ++k) {
        const int position = t * stride + k - pad;
        if (position < 0 || position >= time)
          continue;
        for (int i = 0; i < incoming; ++i)
          value += x[position][i] * w[(k * incoming + i) * outgoing + c];
      }
      out[t][c] = relu ? std::max(0., value) : value;
    }
  return out;
}

std::vector<double> Host(const Weights& w) {
  Matrix x(10, std::vector<double>(2));
  for (int t = 0; t < 10; ++t) {
    x[t][0] = (double(float(.4 * std::cos(.6 * t))) - float(.3)) / float(1.2);
    x[t][1] = (double(float(.3 * std::sin(.8 * t))) - float(-.2)) / float(.7);
  }
  for (int stage = 0; stage < 3; ++stage) {
    const int  first = 6 * stage, width = 1 << stage;
    const auto main = Conv(
        Conv(x, w[first], w[first + 1], 5, 2, width, true),
        w[first + 2],
        w[first + 3],
        3,
        1,
        width,
        false);
    auto residual = Conv(x, w[first + 4], w[first + 5], 1, 2, width, false);
    for (std::size_t t = 0; t < main.size(); ++t)
      for (int c = 0; c < width; ++c)
        residual[t][c] = std::max(0., residual[t][c] + main[t][c]);
    x = residual;
  }
  std::vector<double> average(4), out(3);
  for (const auto& row : x)
    for (int c = 0; c < 4; ++c)
      average[c] += row[c] / x.size();
  for (int c = 0; c < 3; ++c) {
    out[c] = w[19][c];
    for (int i = 0; i < 4; ++i)
      out[c] += average[i] * w[18][3 * i + c];
  }
  return out;
}
double Objective(const Weights& w) {
  const auto out = Host(w);
  return .3 * out[0] - .2 * out[1] + .1 * out[2];
}

void HostChecks() {
  nn::ParameterStore p(601), state(607);
  std::vector<float> values(20);
  for (int t = 0; t < 10; ++t) {
    values[2 * t]     = static_cast<float>(.4 * std::cos(.6 * t));
    values[2 * t + 1] = static_cast<float>(.3 * std::sin(.8 * t));
  }
  const Tensor               input = nn::from_vector(values, {1, 10, 2});
  const sensor::SensorConfig config{2, 3, 1};
  sensor::SensorCnn(p, state, input, config);
  const auto weights =
      sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape&) {
        return .12f + .04f * static_cast<float>(std::sin(.7 * i + .8 * v));
      });
  state.set("normalization/mean", nn::from_vector(std::vector<float>{.3f, -.2f}, {2}));
  state.set("normalization/std", nn::from_vector(std::vector<float>{1.2f, .7f}, {2}));
  const auto network = sensor::SensorCnn(p, state, input, config);
  const auto g       = nn::grad(
      nn::sum(nn::mul(network.logits, nn::from_vector(std::vector<float>{.3f, -.2f, .1f}, {1, 3}))),
      p.parameters());
  const auto expected    = Host(weights);
  const auto z           = network.logits.to_vector<double>();
  double     value_error = 0;
  for (int c = 0; c < 3; ++c)
    value_error = std::max(value_error, std::abs(expected[c] - z[c]));
  int          coordinates    = 0;
  const double gradient_error = sentence::HostGradientError(weights, g, Objective, coordinates);
  check::Require(
      coordinates == 171 && p.size() == 20 && state.size() == 2,
      "all171 tiny coordinates across20 parameters plus2 normalization tensors");
  check::Near(
      value_error,
      0,
      2e-6,
      "independent multichannel normalization/stridedSAME/residual/GAP/head forward");
  check::Near(gradient_error, 0, 3e-6, "EVERY conv/projection/head host finite difference");
  std::cout << " independent multichannel strided residual forward error=" << value_error
            << ",ALL171 host-coordinate FD error=" << gradient_error << std::endl;
}

void LossMetricChecks() {
  Tensor z = nn::from_vector(std::vector<float>{1000, 999, -2, 3}, {2, 2});
  z.requires_grad_(true);
  const Tensor loss = sensor::SensorCE(
      z,
      nn::from_vector(std::vector<int>{1, 0}, {2}),
      nn::from_vector(std::vector<float>{1.5f, .5f}, {2}),
      2);
  const auto g        = nn::grad(loss, {z})[0].to_vector<double>();
  const auto logits   = z.to_vector<double>();
  double     expected = 0;
  for (int n = 0; n < 2; ++n) {
    const int    label  = 1 - n;
    const double weight = label == 0 ? 1.5 : .5, a = logits[2 * n], b = logits[2 * n + 1],
                 maximum = std::max(a, b), sum = std::exp(a - maximum) + std::exp(b - maximum);
    expected += weight * (maximum + std::log(sum) - (label ? b : a)) / 2;
    for (int c = 0; c < 2; ++c)
      check::Near(
          g[2 * n + c],
          weight * (std::exp((c ? b : a) - maximum) / sum - (c == label)) / 2,
          1e-7,
          "every weighted mean CE logit derivative");
  }
  check::Near(
      loss.item<double>(), expected, 1e-6, "weighted mean objective denominator is raw row count");
  std::vector<float> scores(18, 0.f);
  const int          predictions[] = {0, 1, 1, 2, 2, 2};
  for (int n = 0; n < 6; ++n)
    scores[n * 3 + predictions[n]] = 1;
  const auto metrics = sensor::ScoreSensor(
      nn::from_vector(scores, {6, 3}), nn::from_vector(std::vector<int>{0, 0, 1, 1, 2, 2}, {6}), 3);
  check::Near(metrics.accuracy, 2. / 3, 1e-12, "known multiclass accuracy");
  check::Near(metrics.balanced, 2. / 3, 1e-12, "known mean class recall");
  check::Near(
      metrics.macro_f1,
      (2. / 3 + .5 + .8) / 3,
      1e-12,
      "known macroF1 with different precision/recall");
  check::Require(
      metrics.confusion == std::vector<int>({1, 1, 0, 0, 1, 1, 0, 0, 2}),
      "known ALL confusion cells");
  check::Near(
      sensor::SensorAuc0({.8, .8, .2, .1}, nn::from_vector(std::vector<int>{0, 1, 0, 1}, {4})),
      .625,
      1e-12,
      "sourceclass0 ROC with positive/negative ties gets half credit");
  std::cout << " weighted CE host value error=" << std::abs(loss.item<double>() - expected)
            << ";extreme CE/all logits,ALL confusion/macroF1/balanced recall/tie-aware "
               "sourceclass0AUROC fixtures passed\n";
}

void FullChecks() {
  for (int mode = 0; mode < 2; ++mode) {
    const int          channels = mode ? 6 : 1, classes = mode ? 4 : 2, time = mode ? 100 : 96;
    nn::ParameterStore p(613), state(617);
    std::vector<float> input(static_cast<std::size_t>(time * channels));
    for (int t = 0; t < time; ++t)
      for (int c = 0; c < channels; ++c)
        input[t * channels + c] = static_cast<float>(std::sin(t * .13 + c * .7));
    const auto network = sensor::SensorCnn(
        p, state, nn::from_vector(input, {1, time, channels}), {channels, classes, 64});
    const auto norms =
        sentence::SquaredNorms(nn::grad(nn::sum(nn::square(network.logits)), p.parameters()));
    int length = time;
    for (int stage = 0; stage < 3; ++stage) {
      length = (length + 1) / 2;
      check::Require(
          network.stages[stage].shape() == nn::Shape{1, length, 64 * (1 << stage)},
          "full temporal downsampling/channel stage shapes");
    }
    check::Require(
        network.logits.shape() == nn::Shape{1, classes} && p.count() == (mode ? 508484 : 506050) &&
            p.size() == 20 && state.size() == 2,
        "independent full learned/state count");
    for (double n : norms)
      check::Require(
          std::isfinite(n) && n > 0, "every full conv/projection/head gradient finite/nonzero");
    std::cout << " full " << (mode ? "six-channel sensor" : "ECG") << ":" << p.count()
              << " parameters/20 tensors+2normstate;all stages and learned gradients passed\n";
  }
}

} // namespace
