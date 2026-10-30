#pragma once
/// @file wavenet_stack_models.hpp
/// @brief Complete differentiable gated causal audio stack and stable warmup loss.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure the complete gated causal audio stack.
///
/// Integer architecture metadata only. Dilations reset each cycle. Full32/64
/// and compact8/16 retain12gated layers and76learned leaves.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::WaveStackConfig c{8, 16, 6, 2, 256};
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
struct WaveStackConfig {
  int residual = 32; ///< Positive residual/gated width;compact8.
  int skip = 64;     ///< Positive skip/post width;compact16.
  int levels = 6;    ///< Dilation levels1..12;source6.
  int cycles = 2;    ///< Repeated dilation cycles1..4;source2.
  int classes = 256; ///< At least2 categorical audio codes;source256.
};
/// @brief Compute the exact causal receptive field in original samples.
///
/// Includes the initial two-tap convolution plus every dilated two-tap gate.
///
/// @param config Positive widths,levels1..12,cycles1..4,classes>=2.
/// @return 2+cycles*(2^levels-1);source128.
/// @throws std::invalid_argument For invalid architecture metadata.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const int field = m::WaveStackField({8, 16, 6, 2, 256}); // 128
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline int WaveStackField(const WaveStackConfig &config) {
  if (config.residual < 1 || config.skip < 1 || config.levels < 1 || config.levels > 12 ||
      config.cycles < 1 || config.cycles > 4 || config.classes < 2)
    throw std::invalid_argument("WaveNet stack configuration");
  return 2 + config.cycles * ((1 << config.levels) - 1);
}

/// @brief Apply explicitly left-padded Glorot causal convolution.
///
/// Stored W shape[1,kernel,C,out] preserves source checkpoint names.
/// Bias0,uniform radius sqrt(6/(kernel*(C+out)));no activation or optimizer.
///
/// @param p Mutable floating store;lazy named W/b,not fitted here.
/// @param name Nonempty caller prefix with /W and /b.
/// @param x Floating[N,T,C],positive axes,dtype matching p.
/// @param out Positive output width.
/// @param kernel Positive tap count,default1.
/// @param dilation Positive tap spacing,default1.
/// @return Floating[N,T,out],same-time causal inclusive input.
/// @throws std::invalid_argument For invalid shape,width,dtype,kernel,dilation or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(541);
/// const auto y = m::WaveStackConv(p, "c", n::ones({1, 6, 2}), 3, 2, 2);
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline Tensor WaveStackConv(nn::ParameterStore &p, const std::string &name, const Tensor &x,
                            int out, int kernel = 1, int dilation = 1) {
  if (x.dim() != 3 || x.size(0) < 1 || x.size(1) < 1 || x.size(2) < 1 ||
      x.dtype() != p.dtype() || out < 1 || kernel < 1 || dilation < 1)
    throw std::invalid_argument(
        "positive causal convolution shape,width,kernel,dilation and matching dtype required");
  const std::int64_t in = x.size(-1);
  const Tensor w = p.get(name + "/W", {1, kernel, in, out},
                         nn::init::uniform(std::sqrt(6.f / (kernel * (in + out)))));
  const Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  const Tensor padded = nn::pad(x, {{0, 0}, {(kernel - 1) * dilation, 0}, {0, 0}});
  return nn::bias_add(
      nn::conv1d(padded, nn::reshape(w, {kernel, in, out}), 1, nn::Padding::Valid, dilation),
      b);
}

/// @brief Own next-code scores,the summed skip path and every gate.
///
/// Tensor handles retain graph/storage;no normalization or mutable generation state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(541);
/// const auto out =
///     m::WaveStack(p, n::one_hot(n::zeros({1, 6}, n::DType::Int32), 3), {1, 1, 2, 1, 3});
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
struct WaveStackOutput {
  Tensor logits;             ///< Raw next-code[N,T,classes] scores.
  Tensor skip;               ///< Sum of all block skip outputs before post-head ReLU.
  std::vector<Tensor> gates; ///< tanh(filter)*sigmoid(gate) in execution order.
};

/// @brief Predict next audio codes with the full tanh-sigmoid residual/skip stack.
///
/// Every block has a dilated two-tap joined filter/gate and1x1skip.
/// All but the last also have a1x1residual projection. Sum skips,ReLU,
/// 1x1post/ReLU,linear1x1head. No batchnorm,dropout,conditioning or fitting.
/// Input onehot is differentiable for structural Jacobian checks.
///
/// @param p Mutable learned store;warm all leaves before optimizer construction.
/// @param encoded Floating[N,T,classes],positive N/T,dtype matching p;normally onehot.
/// @param config Valid architecture,defaultfull32/64/6/2/256.
/// @return Same-time causal next-code logits,skip sum and all gated activations.
/// @throws std::invalid_argument For invalid architecture,input shape,dtype or stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(541);
/// const auto out =
///     m::WaveStack(p, n::one_hot(n::zeros({1, 6}, n::DType::Int32), 3), {1, 1, 2, 1, 3});
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline WaveStackOutput WaveStack(nn::ParameterStore &p, const Tensor &encoded,
                                 const WaveStackConfig &config = {}) {
  (void)WaveStackField(config);
  if (encoded.dim() != 3 || encoded.size(0) < 1 || encoded.size(1) < 1 ||
      encoded.size(2) != config.classes || encoded.dtype() != p.dtype())
    throw std::invalid_argument("floating encoded[N,T,classes] matching store required");
  WaveStackOutput result;
  Tensor x = WaveStackConv(p, "input", encoded, config.residual, 2);
  std::vector<Tensor> skips;
  const int blocks = config.levels * config.cycles;
  for (int block = 0; block < blocks; ++block) {
    const auto name = "block" + std::to_string(block + 1);
    const Tensor joined = WaveStackConv(p, name + "/gate", x, 2 * config.residual, 2,
                                        1 << (block % config.levels));
    const auto split = nn::split(joined, 2, 2);
    const Tensor gate = nn::mul(nn::tanh(split[0]), nn::sigmoid(split[1]));
    result.gates.push_back(gate);
    skips.push_back(WaveStackConv(p, name + "/skip", gate, config.skip));
    if (block + 1 < blocks)
      x = nn::add(x, WaveStackConv(p, name + "/residual", gate, config.residual));
  }
  result.skip = skips[0];
  for (std::size_t i = 1; i < skips.size(); ++i)
    result.skip = nn::add(result.skip, skips[i]);
  Tensor hidden = nn::relu(result.skip);
  hidden = nn::relu(WaveStackConv(p, "post", hidden, config.skip));
  result.logits = WaveStackConv(p, "output", hidden, config.classes);
  return result;
}

/// @brief Compute stable next-code NLL after an explicit warmup.
///
/// No parameter fitting or mutation. Uses log_softmax to remain stable for extreme logits.
///
/// @param logits Floating[N,T,classes] raw scores.
/// @param target Int32/64[N,T] bounded next-code classes.
/// @param classes Class count>=2,default256.
/// @param start First scored time0<=start<T,default127.
/// @return Scalar mean CE over N*(T-start);earlier logit adjoints exactly0.
/// @throws std::invalid_argument For invalid shape,dtype,class bounds or warmup.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto loss = m::WaveNll(n::zeros({1, 4, 3}), n::zeros({1, 4}, n::DType::Int32), 3, 2);
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline Tensor WaveNll(const Tensor &logits, const Tensor &target, int classes = 256,
                      int start = 127) {
  if (logits.dim() != 3 || logits.size(0) < 1 || logits.size(1) < 1 || classes < 2 ||
      logits.size(2) != classes ||
      target.shape() != nn::Shape{logits.size(0), logits.size(1)} ||
      (target.dtype() != nn::DType::Int32 && target.dtype() != nn::DType::Int64) ||
      start < 0 || start >= logits.size(1) ||
      !nn::all(
           nn::logical_and(nn::greater_equal(target, 0.), nn::less(target, double(classes))))
           .item<bool>())
    throw std::invalid_argument(
        "bounded targets,matching scores and nonempty scored time required");
  const auto z = nn::slice_axis(logits, 1, start, logits.size(1));
  const auto y = nn::slice_axis(target, 1, start, target.size(1));
  return -nn::mean(
      nn::sum(nn::one_hot(y, classes, 1., 0., z.dtype()) * nn::log_softmax(z), {2}));
}
} // namespace datamunge::nn::models
