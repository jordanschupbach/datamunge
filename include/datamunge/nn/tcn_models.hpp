#pragma once
/// @file tcn_models.hpp
/// @brief Weight-normalized causal residual temporal classifier and observed-prefix helpers.
#include "datamunge/nn/time_series_data.hpp"
#include <cmath>
#include <limits>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure causal residual widths,dilation depth and saved fitting-only normalization.
///
/// Architecture metadata only. Existing checkpoint mean/std override creation
/// defaults;no normalization statistic is estimated from inference prefixes.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TcnConfig c{8, 6, 3, 0, 1};
/// const auto field = m::TcnReceptiveField(c); // 253
/// @endcode
/// @par References
/// - Bai,Kolter and Koltun(2018),An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, https://arxiv.org/abs/1803.01271;Salimans and Kingma(2016),Weight Normalization:A Simple Reparameterization to Accelerate Training of Deep Neural Networks, https://arxiv.org/abs/1602.07868. Source teaching variant uses explicit left padding,direct dilated tensor convolutions,two scalar fitting-normalization states and a last-time binary head.
/// @see Tcn(), TcnReceptiveField()
struct TcnConfig {
  int channels = 64; ///< Positive residual feature width;compact8.
  int levels = 6;    ///< Residual block count1..15,dilations powers of2.
  int kernel = 3;    ///< Kernel width>=2 in both convolutions per block.
  float mean = 0;    ///< Finite fitting-only mean,used only to create normalization state.
  float standard_deviation = 1; ///< Positive fitting-only population std creation default.
};

/// @brief Count the exact two-convolution dilated causal receptive-field length.
///
/// Two left-padded convolutions per residual block have dilation2^level.
/// The head uses the last observed time;this is graph connectivity,not an
/// effective influence estimate after nonlinearities or short-input padding.
///
/// @param config levels1..15 and kernel>=2;channels/normalization are unused.
/// @return 1+2*(kernel-1)*(2^levels-1),in source sample positions.
/// @throws std::invalid_argument For invalid levels/kernel or an int-overflow receptive field.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto field = m::TcnReceptiveField(m::TcnConfig{}); // 253
/// @endcode
/// @par References
/// - Bai,Kolter and Koltun(2018),An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, https://arxiv.org/abs/1803.01271;Salimans and Kingma(2016),Weight Normalization:A Simple Reparameterization to Accelerate Training of Deep Neural Networks, https://arxiv.org/abs/1602.07868. Source teaching variant uses explicit left padding,direct dilated tensor convolutions,two scalar fitting-normalization states and a last-time binary head.
/// @see Tcn()
inline int TcnReceptiveField(const TcnConfig &config) {
  if (config.levels < 1 || config.levels > 15 || config.kernel < 2)
    throw std::invalid_argument("TCN dilation/kernel configuration");
  const auto field = 1 + 2LL * (config.kernel - 1) * ((1LL << config.levels) - 1);
  if (field > std::numeric_limits<int>::max())
    throw std::invalid_argument("TCN receptive field exceeds int metadata");
  return int(field);
}

/// @brief Apply weight-normalized biased temporal convolution with left-only padding.
///
/// v shape[1,k,Cin,Cout] initializes N(0,.01),g[Cout]=sqrt(2.f),b0.
/// W=g*v/sqrt(sum(v*v)+1e-12),per output channel. Reshape to native1D
/// kernel then explicit left padding and VALID dilated convolution preserve T.
/// No future input,parameter mutation beyond lazy creation or optimizer.
///
/// @param p Mutable learned store;creates/reuses v,g,b named leaves.
/// @param name Unique learned convolution prefix.
/// @param input Floating[N,T,Cin],positive dimensions,matching p dtype.
/// @param outgoing Positive output channels.
/// @param kernel Kernel width>=2.
/// @param dilation Positive tap spacing;left pad=(kernel-1)*dilation fits int.
/// @return Floating[N,T,outgoing] causal scores,without activation.
/// @throws std::invalid_argument For invalid dimensions/dtype/padding or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(461);
/// const auto y = m::TcnCausalConv(p, "causal", n::ones({2, 11, 1}), 4, 3, 2);
/// @endcode
/// @par References
/// - Bai,Kolter and Koltun(2018),An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, https://arxiv.org/abs/1803.01271;Salimans and Kingma(2016),Weight Normalization:A Simple Reparameterization to Accelerate Training of Deep Neural Networks, https://arxiv.org/abs/1602.07868. Source teaching variant uses explicit left padding,direct dilated tensor convolutions,two scalar fitting-normalization states and a last-time binary head.
/// @see Tcn()
inline Tensor TcnCausalConv(nn::ParameterStore &p, const std::string &name,
                            const Tensor &input, int outgoing, int kernel, int dilation) {
  if (input.dim() != 3 || input.size(0) < 1 || input.size(1) < 1 || input.size(2) < 1 ||
      input.dtype() != p.dtype() || outgoing < 1 || kernel < 2 || dilation < 1 ||
      std::int64_t(kernel - 1) * dilation > std::numeric_limits<int>::max())
    throw std::invalid_argument("valid floating causal convolution dimensions required");
  const std::int64_t incoming = input.size(-1);
  const int left = (kernel - 1) * dilation;
  const Tensor v = p.get(name + "/v", {1, kernel, incoming, outgoing}, nn::init::normal(.01));
  const Tensor g = p.get(name + "/g", {outgoing}, nn::init::constant(std::sqrt(2.f)));
  const Tensor b = p.get(name + "/b", {outgoing}, nn::init::zeros());
  const Tensor norm = nn::sqrt(nn::add(nn::sum(nn::square(v), {0, 1, 2}), 1e-12));
  const Tensor weight = nn::mul(v, nn::div(g, norm));
  const Tensor padded = nn::pad(input, {{0, 0}, {left, 0}, {0, 0}});
  return nn::bias_add(nn::conv1d(padded, nn::reshape(weight, {kernel, incoming, outgoing}), 1,
                                 nn::Padding::Valid, dilation),
                      b);
}

/// @brief Own every causal residual block and the shared per-time/endpoint logits.
///
/// Tensor handles retain graph/storage lifetimes;endpoint logits are selected
/// from the last sequence time,not an independently fitted classifier.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(461), state(463);
/// m::TcnConfig c{8, 3, 3, 0, 1};
/// const auto out = m::Tcn(p, state, n::ones({2, 31, 1}), c);
/// @endcode
/// @par References
/// - Bai,Kolter and Koltun(2018),An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, https://arxiv.org/abs/1803.01271;Salimans and Kingma(2016),Weight Normalization:A Simple Reparameterization to Accelerate Training of Deep Neural Networks, https://arxiv.org/abs/1602.07868. Source teaching variant uses explicit left padding,direct dilated tensor convolutions,two scalar fitting-normalization states and a last-time binary head.
/// @see Tcn()
struct TcnOutput {
  Tensor hidden;              ///< Final causal[N,T,channels] feature sequence.
  Tensor sequence;            ///< Linear[N,T,2] logits at every observed time.
  Tensor logits;              ///< Final-time[N,2] scores sharing the sequence graph.
  std::vector<Tensor> blocks; ///< Every residual block output,preserving[N,T,channels].
};

/// @brief Predict causal per-time and endpoint binary logits with dilated residual temporal blocks.
///
/// Normalize using saved detached fitting scalars. Each block applies two
/// weight-normalized causal convolutions,ReLU and optional dropout before
/// adding identity or learned1x1 projected skip,then ReLU. Dilations1..2^(L-1).
/// A shared per-time linear head produces2 logits;classification selects T-1.
/// Full64/compact8 with six blocks retain40 learned and2 normalization leaves.
/// No BN,prefix-derived normalization,hidden gradient clipping or optimizer.
/// Under frozen inference,prefix endpoint equals the full same-time output
/// within floating tolerance;future input has exactly zero prefix Jacobian.
///
/// @param p Mutable learned store;warm all leaves before optimizer creation.
/// @param state Same-dtype separate store for detached scalar fitting mean/std.
/// @param signal Floating[N,T,1],N/T>0,matching both stores dtype;variable T allowed.
/// @param config Valid residual width,levels,kernel and fitting normalization defaults.
/// @param training False by default;true activates explicit-generator dropout.
/// @param keep Finite dropout keep probability in(0,1],default.8.
/// @param gen Required mutable generator only when training&&keep<1;otherwise unused.
/// @return All residual blocks,final features,per-time logits and final-time logits.
/// @throws std::invalid_argument For invalid architecture,raw input,store dtype,dropout policy or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(461), state(463);
/// m::TcnConfig c{8, 3, 3, 0, 1};
/// const auto out = m::Tcn(p, state, n::ones({2, 31, 1}), c);
/// @endcode
/// @par References
/// - Bai,Kolter and Koltun(2018),An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, https://arxiv.org/abs/1803.01271;Salimans and Kingma(2016),Weight Normalization:A Simple Reparameterization to Accelerate Training of Deep Neural Networks, https://arxiv.org/abs/1602.07868. Source teaching variant uses explicit left padding,direct dilated tensor convolutions,two scalar fitting-normalization states and a last-time binary head.
/// @see TcnReceptiveField(), TcnCausalConv(), TcnPrefix()
inline TcnOutput Tcn(nn::ParameterStore &p, nn::ParameterStore &state, const Tensor &signal,
                     const TcnConfig &config = {}, bool training = false, double keep = .8,
                     nn::Generator *gen = nullptr) {
  (void)TcnReceptiveField(config);
  if (config.channels < 1 || !std::isfinite(config.mean) ||
      !std::isfinite(config.standard_deviation) || config.standard_deviation <= 0)
    throw std::invalid_argument("TCN width/normalization");
  if (signal.dim() != 3 || signal.size(0) < 1 || signal.size(1) < 1 || signal.size(2) != 1 ||
      signal.dtype() != p.dtype() || state.dtype() != p.dtype() || !std::isfinite(keep) ||
      keep <= 0 || keep > 1 || (training && keep < 1 && !gen))
    throw std::invalid_argument(
        "aligned temporal input/stores and explicit dropout generator required");
  TcnOutput result;
  const Tensor mean = state.get("normalization/mean", {}, nn::init::constant(config.mean));
  const Tensor stddev =
      state.get("normalization/std", {}, nn::init::constant(config.standard_deviation));
  Tensor x = nn::div(nn::sub(signal, mean.detach()), stddev.detach());
  int incoming = 1;
  for (int level = 0; level < config.levels; ++level) {
    const auto name = "block" + std::to_string(level + 1);
    Tensor branch = x;
    for (int layer = 0; layer < 2; ++layer) {
      branch = nn::relu(TcnCausalConv(p, name + "/conv" + std::to_string(layer + 1), branch,
                                      config.channels, config.kernel, 1 << level));
      if (training && keep < 1)
        branch = nn::dropout(branch, 1 - keep, true, *gen);
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
  const Tensor scores = nn::dense(p, "head", nn::reshape(x, {-1, config.channels}), 2);
  result.sequence = nn::reshape(scores, {batch, time, 2});
  result.logits = nn::select(result.sequence, 1, time - 1);
  return result;
}

/// @brief Take an observed temporal prefix while preserving original labels and source IDs.
///
/// No new normalization,fitting,source ID or future samples enter the prefix.
///
/// @param input Aligned raw temporal container,with a time axis.
/// @param length Positive observed prefix length<=original T.
/// @return Container sharing metadata with contiguous signals[:,0:length,:].
/// @throws std::invalid_argument For invalid temporal axis or out-of-bounds prefix length.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SeriesData d{n::zeros({2, 150, 1}), n::zeros({2}, n::DType::Int32),
///                 n::zeros({2}, n::DType::Int32)};
/// const auto prefix = m::TcnPrefix(d, 17);
/// @endcode
/// @par References
/// - Bai,Kolter and Koltun(2018),An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, https://arxiv.org/abs/1803.01271;Salimans and Kingma(2016),Weight Normalization:A Simple Reparameterization to Accelerate Training of Deep Neural Networks, https://arxiv.org/abs/1602.07868. Source teaching variant uses explicit left padding,direct dilated tensor convolutions,two scalar fitting-normalization states and a last-time binary head.
/// @see Tcn()
inline SeriesData TcnPrefix(const SeriesData &input, int length) {
  if (input.signals.dim() != 3 || length < 1 || length > input.signals.size(1))
    throw std::invalid_argument("bounded observed causal prefix required");
  SeriesData out = input;
  out.signals = nn::slice_axis(input.signals, 1, 0, length).contiguous();
  return out;
}

} // namespace datamunge::nn::models
