// Bai/Kolter/Koltun-style TCN: two causal weight-normalized convolutions per
// residual block, dilations 1..2^(levels-1), and short real-sequence
// classification with causal prefixes and saved fit normalization (port of
// tflow's common/nn/tcn.h and common/ml/tcn.h). tflow zero-spaces its
// dilated taps by hand to dodge a TF SDK gradient bug; nn::conv1d takes the
// dilation directly (same arithmetic, no extra learned taps).
#pragma once

#include "common/time_series.hpp"

namespace tcn {

namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Range;
using sentence::Rows;
using series::SeriesData;
using series::SeriesRows;

struct TcnConfig {
  int   channels = 64, levels = 6, kernel = 3;
  float mean = 0, standard_deviation = 1;
};

inline int TcnReceptiveField(const TcnConfig& config) {
  if (config.levels < 1 || config.levels > 15 || config.kernel < 2) throw std::invalid_argument("TCN dilation/kernel configuration");
  return 1 + 2 * (config.kernel - 1) * ((1 << config.levels) - 1);
}

/// Weight-normalized causal convolution: W = g v / |v| per output channel,
/// left-only padding (k - 1) d, then a VALID dilated convolution plus bias.
/// Parameters "<name>/v" [1, k, in, out] ~ N(0, .01), "<name>/g" = sqrt 2, "<name>/b" = 0.
inline Tensor TcnCausalConv(nn::ParameterStore& p, const std::string& name, const Tensor& input, int outgoing, int kernel, int dilation) {
  const std::int64_t incoming = input.size(-1);
  const int          left = (kernel - 1) * dilation;
  const Tensor v = p.get(name + "/v", {1, kernel, incoming, outgoing}, nn::init::normal(.01));
  const Tensor g = p.get(name + "/g", {outgoing}, nn::init::constant(std::sqrt(2.f)));
  const Tensor b = p.get(name + "/b", {outgoing}, nn::init::zeros());
  const Tensor norm = nn::sqrt(nn::add(nn::sum(nn::square(v), {0, 1, 2}), 1e-12));
  const Tensor weight = nn::mul(v, nn::div(g, norm));
  const Tensor padded = nn::pad(input, {{0, 0}, {left, 0}, {0, 0}});
  return nn::bias_add(nn::conv1d(padded, nn::reshape(weight, {kernel, incoming, outgoing}), 1, nn::Padding::Valid, dilation), b);
}

struct TcnOutput {
  Tensor              hidden, sequence, logits;
  std::vector<Tensor> blocks;
};

/// signal [N, T, 1] -> per-time logits sequence [N, T, 2]; logits = last time step.
inline TcnOutput Tcn(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& signal, const TcnConfig& config = {},
                     bool training = false, double keep = .8, nn::Generator* gen = nullptr) {
  (void)TcnReceptiveField(config);
  if (config.channels < 1 || !std::isfinite(config.mean) || !std::isfinite(config.standard_deviation) || config.standard_deviation <= 0)
    throw std::invalid_argument("TCN width/normalization");
  TcnOutput    result;
  const Tensor mean = state.get("normalization/mean", {}, nn::init::constant(config.mean));
  const Tensor stddev = state.get("normalization/std", {}, nn::init::constant(config.standard_deviation));
  Tensor       x = nn::div(nn::sub(signal, mean.detach()), stddev.detach());
  int          incoming = 1;
  for (int level = 0; level < config.levels; ++level) {
    const auto name = "block" + std::to_string(level + 1);
    Tensor     branch = x;
    for (int layer = 0; layer < 2; ++layer) {
      branch = nn::relu(TcnCausalConv(p, name + "/conv" + std::to_string(layer + 1), branch, config.channels, config.kernel, 1 << level));
      if (training && keep < 1) branch = nn::dropout(branch, 1 - keep, true, *gen);
    }
    Tensor residual = x;
    if (incoming != config.channels) {
      nn::ConvLayerOptions opt;
      opt.padding = nn::Padding::Valid;
      opt.init = nn::init::he_uniform();
      residual = nn::conv1d(p, name + "/projection", x, config.channels, 1, opt);
    }
    x = nn::relu(nn::add(branch, residual));
    incoming = config.channels;
    result.blocks.push_back(x);
  }
  result.hidden = x;
  const std::int64_t batch = x.size(0), time = x.size(1);
  const Tensor       scores = nn::dense(p, "head", nn::reshape(x, {-1, config.channels}), 2);
  result.sequence = nn::reshape(scores, {batch, time, 2});
  result.logits = nn::select(result.sequence, 1, time - 1);
  return result;
}

class TcnModel {
 public:
  nn::ParameterStore             parameters{461}, state{463};
  TcnConfig                      config;
  std::unique_ptr<nn::Optimizer> optimizer;
  nn::Generator                  dropout{457};

  explicit TcnModel(TcnConfig c = {8, 6, 3, 0, 1}, bool training = true) : config(c) {
    {
      nn::NoGradGuard guard;
      Tcn(parameters, state, nn::zeros({1, 4, 1}), c);
    }
    if (training) optimizer = std::make_unique<nn::Optimizer>(parameters, .001, replay::Adam());
  }
  TcnOutput Forward(const SeriesData& d, bool training) { return Tcn(parameters, state, d.signals, config, training, .8, &dropout); }
  /// Inference (dropout-free, no gradient) CE.
  double Loss(const SeriesData& d) {
    nn::NoGradGuard guard;
    const double    value = sentence::SentenceCE(Forward(d, false).logits, d.labels).item<double>();
    check::Require(std::isfinite(value) && value >= 0, "finite TCN CE");
    return value;
  }
  Tensor Predict(const SeriesData& d, bool all_times = false) {
    nn::NoGradGuard guard;
    const auto      out = Forward(d, false);
    return (all_times ? out.sequence : out.logits).detach().contiguous();
  }
  replay::Map Snapshot() const { return sentence::Snapshot({&parameters, &state}); }
  void        Save(const std::string& path) const { sentence::SaveStores(path, parameters, &state); }
  void        Restore(const std::string& path) { sentence::RestoreStores(path, parameters, &state); }
  void        Export(const std::filesystem::path& path) const { sentence::ExportArrays(path, parameters, &state); }
};

inline SeriesData TcnPrefix(const SeriesData& input, int length) {
  check::Require(length > 0 && length <= input.signals.size(1), "valid observed causal prefix length");
  SeriesData out = input;
  out.signals = nn::slice_axis(input.signals, 1, 0, length).contiguous();
  return out;
}

}  // namespace tcn
