#pragma once
/// @file inpainting_models.hpp
/// @brief Context-encoder generator/adversary and mask-normalized partial convolution tensor primitives.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;

/// @brief Apply context BN with separately owned running statistics.
///
/// Old-value momentum.9,epsilon1e-5. Lazy creation only beyond controlled
/// state updates;no optimizer. Inference keeps the activation gradient for GAN G steps.
///
/// @param p Mutable floating affine-parameter store.
/// @param state Mutable nontrainable running mean/variance store.
/// @param name Prefix for affine and moving tensors.
/// @param x Floating NHWC activations matching p dtype.
/// @param training Use batch moments when true;stored moments otherwise.
/// @param update Update running moments only when training and true,default true.
/// @return Normalized tensor retaining the input graph.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(71), state(72);
/// const auto y = m::InpaintNorm(p, state, "bn", n::ones({2, 3, 3, 2}), true, false);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline nn::Tensor InpaintNorm(nn::ParameterStore &p, nn::ParameterStore &state,
                              const std::string &name, const nn::Tensor &x, bool training,
                              bool update = true) {
  nn::BatchNormOptions o;
  o.momentum = .9;
  o.eps = 1e-5;
  o.update_stats = update;
  return nn::batch_norm(p, state, name, x, training, o);
}

/// @brief Apply a learned4x4 context convolution.
///
/// Weights N(0,.02),zero biases. No activation,normalization or fitting.
/// Convolution costs O(NHoWo*16*Cin*Cout).
///
/// @param p Mutable parameter store;warm every leaf before constructing its optimizer.
/// @param name Prefix for W[4,4,Cin,Cout] and b[Cout].
/// @param x Floating positive NHWC input matching p dtype.
/// @param out Positive output width.
/// @param down Stride2 SAME when true,stride1 VALID otherwise.
/// @return Biased NHWC convolution retaining input/weight graphs.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(71), state(72);
/// const auto y = m::ContextConv(p, "down", n::ones({1, 8, 8, 3}), 8, true);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline nn::Tensor ContextConv(nn::ParameterStore &p, const std::string &name,
                              const nn::Tensor &x, std::int64_t out, bool down) {
  if (x.dim() != 4 || x.size(0) < 1 || x.size(1) < 1 || x.size(2) < 1 || x.size(3) < 1 ||
      out < 1)
    throw std::invalid_argument("invalid context convolution geometry");
  const nn::Tensor w = p.get(name + "/W", {4, 4, x.size(3), out}, nn::init::normal(.02));
  const nn::Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  return nn::bias_add(
      nn::conv2d(
          x, w, nn::conv_options(down ? 2 : 1, down ? nn::Padding::Same : nn::Padding::Valid)),
      b);
}

/// @brief Apply a learned4x4 transpose convolution.
///
/// Weights N(0,.02),zero biases. No activation or optimizer.
/// Transpose operation retains gradients through all input and learned values.
///
/// @param p Mutable parameter store.
/// @param name Prefix for W[4,4,Cout,Cin] and b[Cout].
/// @param x Floating positive NHWC input matching p dtype.
/// @param out Positive output width.
/// @param first Stride1 VALID to4x4 for1x1 latent input;otherwise stride2 SAME doubles side.
/// @return Biased decoder activation with specified output geometry.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(71), state(72);
/// const auto y = m::ContextUp(p, "up", n::ones({1, 1, 1, 4}), 8, true);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline nn::Tensor ContextUp(nn::ParameterStore &p, const std::string &name,
                            const nn::Tensor &x, std::int64_t out, bool first) {
  if (x.dim() != 4 || x.size(0) < 1 || x.size(1) < 1 || x.size(2) < 1 || x.size(3) < 1 ||
      out < 1 || (first && (x.size(1) != 1 || x.size(2) != 1)))
    throw std::invalid_argument("invalid context transpose geometry");
  const nn::Tensor w = p.get(name + "/W", {4, 4, out, x.size(3)}, nn::init::normal(.02));
  const nn::Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  const nn::Shape shape{x.size(0), first ? 4 : x.size(1) * 2, first ? 4 : x.size(2) * 2, out};
  return nn::bias_add(
      nn::conv2d_transpose(
          x, w, shape,
          nn::conv_options(first ? 1 : 2, first ? nn::Padding::Valid : nn::Padding::Same)),
      b);
}

/// @brief Configure full128px or compact32px context encoder geometry.
///
/// Integer architecture metadata only;default128/64/100. No parameter creation.
///
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// const auto c = dm::nn::models::ContextConfig::Compact();
/// const auto widths = c.Encoder();
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
struct ContextConfig {
  int side = 128;   ///< Input side:128 reference or32 compact.
  int width = 64;   ///< Positive first convolution width.
  int latent = 100; ///< Positive1x1 bottleneck channels.

  /// @brief Return the32px,width8,latent32 teaching architecture.
  ///
  /// Retains all compact encoder/bottleneck/decoder/adversary stages,no skip connections.
  ///
  /// @return Independent32/8/32 architecture metadata.
  /// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
  /// @par Example
  /// @code
/// const auto c = dm::nn::models::ContextConfig::Compact();
/// @endcode
  /// @par References
  /// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
  /// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
  static ContextConfig Compact() {
    return {32, 8, 32};
  }
  /// @brief Validate configuration and enumerate encoder downsampling widths.
  ///
  /// Positive latent/width,valid side and bounded channel multipliers are required.
  ///
  /// @return Widths[width,width,2width] at32 or also4width,8width at128.
  /// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
  /// @par Example
  /// @code
/// const auto w = dm::nn::models::ContextConfig::Compact().Encoder();
/// @endcode
  /// @par References
  /// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
  /// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
  [[nodiscard]] std::vector<int> Encoder() const {
    if (width < 1 || latent < 1 || width > std::numeric_limits<int>::max() / 8)
      throw std::invalid_argument("invalid context width or latent channels");
    if (side == 128)
      return {width, width, width * 2, width * 4, width * 8};
    if (side == 32)
      return {width, width, width * 2};
    throw std::invalid_argument("context encoder supports original128 or compact32");
  }
};

/// @brief Own generator output and its exposed stage activations.
///
/// Tensor handles preserve the graph;stage metadata does not perform fitting.
///
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(71), state(72);
/// const auto result =
///     m::ContextEncoder(p, state, n::zeros({1, 32, 32, 3}), m::ContextConfig::Compact());
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
struct ContextResult {
  nn::Tensor patch;               ///< Tanh RGB patch[N,side/2,side/2,3],range[-1,1].
  std::vector<nn::Tensor> stages; ///< Activations in execution order,including final output.
};

/// @brief Predict a missing center patch from its masked RGB context.
///
/// LeakyReLU.2/BN encoder,bottleneck1x1,learned transpose BN/ReLU decoder.
/// No skip connections or optimizer. Warm all leaves,then use a visible caller loop.
/// Save p and state together;frozen generator fake creation uses training=true/update=false.
///
/// @param p Mutable learned generator store.
/// @param state Mutable generator BN running store.
/// @param masked Floating[N,side,side,3] RGB,convention[-1,1],positive N.
/// @param config Validated architecture,default128px.
/// @param training Batch moments when true,default false.
/// @param update Update BN only in training when true,default true.
/// @return Half-side tanh patch plus every encoder/bottleneck/decoder stage.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(71), state(72);
/// const auto y = m::ContextEncoder(p, state, n::zeros({2, 32, 32, 3}),
///                                  m::ContextConfig::Compact(), true, false);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline ContextResult ContextEncoder(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const nn::Tensor &masked, const ContextConfig &config = {},
                                    bool training = false, bool update = true) {
  const auto widths = config.Encoder();
  if (masked.dim() != 4 || masked.size(0) < 1 || masked.size(1) != config.side ||
      masked.size(2) != config.side || masked.size(3) != 3 || masked.dtype() != p.dtype())
    throw std::invalid_argument("context encoder requires aligned floating NHWC RGB");
  ContextResult result;
  nn::Tensor x = masked;
  for (std::size_t i = 0; i < widths.size(); ++i) {
    const auto name = "encode" + std::to_string(i + 1);
    x = ContextConv(p, name, x, widths[i], true);
    if (i)
      x = InpaintNorm(p, state, name + "/bn", x, training, update);
    x = nn::leaky_relu(x, .2);
    result.stages.push_back(x);
  }
  x = ContextConv(p, "bottleneck", x, config.latent, false);
  x = nn::leaky_relu(InpaintNorm(p, state, "bottleneck/bn", x, training, update), .2);
  result.stages.push_back(x);
  std::int64_t in = widths.back();
  x = ContextUp(p, "decode0", x, in, true);
  x = nn::relu(InpaintNorm(p, state, "decode0/bn", x, training, update));
  result.stages.push_back(x);
  for (int i = int(widths.size()) - 2; i >= 1; --i) {
    const auto name = "decode" + std::to_string(int(widths.size()) - 1 - i);
    x = ContextUp(p, name, x, widths[std::size_t(i)], false);
    x = nn::relu(InpaintNorm(p, state, name + "/bn", x, training, update));
    result.stages.push_back(x);
  }
  result.patch = nn::tanh(ContextUp(p, "output", x, 3, false));
  result.stages.push_back(result.patch);
  return result;
}

/// @brief Own adversary logits and exposed intermediate activations.
///
/// Tensor handles retain the graph;no probabilities or optimizer are created.
///
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(71), state(72);
/// const auto y =
///     m::ContextDiscriminator(p, state, n::zeros({2, 16, 16, 3}), m::ContextConfig::Compact());
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
struct ContextDiscriminatorResult {
  nn::Tensor logits;              ///< Raw independent real/fake logits[N],no sigmoid.
  std::vector<nn::Tensor> stages; ///< Activations in execution order,including final output.
};

/// @brief Score real or generated center patches with a convolutional adversary.
///
/// Four downs at128 or two at32,LeakyReLU.2,BN after first,final4x4 VALID.
/// G step uses training=false to freeze moving state but retains patch gradients.
/// Compute generator-only gradients so adversary parameters do not change.
///
/// @param p Mutable learned discriminator store,separate from generator.
/// @param state Mutable discriminator BN store.
/// @param patch Floating[N,side/2,side/2,3] RGB,positive N.
/// @param config Validated architecture,default128px.
/// @param training Batch moments when true,default false.
/// @param update Update BN only while training and true,default true.
/// @return Raw real/fake logits[N] and all downsampling/final stages.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(71), state(72);
/// const auto y =
///     m::ContextDiscriminator(p, state, n::zeros({2, 16, 16, 3}), m::ContextConfig::Compact());
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline ContextDiscriminatorResult
ContextDiscriminator(nn::ParameterStore &p, nn::ParameterStore &state, const nn::Tensor &patch,
                     const ContextConfig &config = {}, bool training = false,
                     bool update = true) {
  (void)config.Encoder();
  if (patch.dim() != 4 || patch.size(0) < 1 || patch.size(1) != config.side / 2 ||
      patch.size(2) != config.side / 2 || patch.size(3) != 3 || patch.dtype() != p.dtype())
    throw std::invalid_argument("context discriminator requires half-side RGB patches");
  ContextDiscriminatorResult result;
  const int depth = config.side == 128 ? 4 : 2;
  nn::Tensor x = patch;
  for (int i = 0; i < depth; ++i) {
    const auto name = "down" + std::to_string(i + 1);
    x = ContextConv(p, name, x, config.width * (1 << i), true);
    if (i)
      x = InpaintNorm(p, state, name + "/bn", x, training, update);
    x = nn::leaky_relu(x, .2);
    result.stages.push_back(x);
  }
  result.logits = nn::reshape(ContextConv(p, "output", x, 1, false), {-1});
  result.stages.push_back(result.logits);
  return result;
}

/// @brief Compute stable non-saturating real/fake mean logit loss.
///
/// Uses softplus directly,finite at extreme logits and derivative(sigmoid(z)-target)/N.
/// No parameter or optimizer mutation.
///
/// @param logits Floating nonempty raw logits.
/// @param real True uses softplus(-logits),false uses softplus(logits).
/// @return Scalar mean stable Bernoulli loss with logit gradients.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto z = n::tensor({-80., 0., 80.}).requires_grad_();
/// const auto loss = n::models::ContextBCE(z, true);
/// const auto gradient = n::grad(loss, {z});
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline nn::Tensor ContextBCE(const nn::Tensor &logits, bool real) {
  if (logits.numel() < 1 ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64))
    throw std::invalid_argument("nonempty floating context logits required");
  return nn::mean(nn::softplus(real ? -logits : logits));
}

/// @brief Compute mean squared patch reconstruction error.
///
/// Caller supplies aligned target geometry;mean covers all batch/spatial/channel samples.
/// No optimizer or implicit stop-gradient on targets.
///
/// @param patch Floating predicted patch.
/// @param truth Same-shaped target in the same value scale.
/// @return Scalar mean squared difference,gradient2*(patch-truth)/numel.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::ContextReconstruction(n::tensor({0., 1.}), n::tensor({1., 1.}));
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline nn::Tensor ContextReconstruction(const nn::Tensor &patch, const nn::Tensor &truth) {
  if (patch.numel() < 1 || patch.shape() != truth.shape())
    throw std::invalid_argument("aligned nonempty reconstruction patches required");
  return nn::mean(nn::squared_difference(patch, truth));
}

/// @brief Keep observed pixels and insert the center predicted patch.
///
/// Zero-pads patch by side/4;where routes gradients to selected inputs.
/// Valid is a fixed decision mask,no learned parameters or fitting.
///
/// @param masked Floating[N,side,side,3] observed image in patch value scale.
/// @param patch Floating[N,side/2,side/2,3] center prediction.
/// @param valid Numeric[1 or N,side,side,1],observed when>.5.
/// @param side Positive side divisible by4,default32.
/// @return Aligned image of masked shape,observed positions preserved exactly.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto mask = n::ones({1, 32, 32, 1});
/// const auto out =
///     n::models::ContextComposite(n::zeros({2, 32, 32, 3}), n::ones({2, 16, 16, 3}), mask);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379;source alternating training https://github.com/pathak22/context-encoder/blob/master/train.lua. Compact graph and objective weights follow the repository teaching example.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline nn::Tensor ContextComposite(const nn::Tensor &masked, const nn::Tensor &patch,
                                   const nn::Tensor &valid, int side = 32) {
  if (side < 4 || side % 4 || masked.dim() != 4 || masked.size(0) < 1 ||
      masked.shape() != nn::Shape{masked.size(0), side, side, 3} ||
      patch.shape() != nn::Shape{masked.size(0), side / 2, side / 2, 3} || valid.dim() != 4 ||
      valid.size(1) != side || valid.size(2) != side || valid.size(3) != 1 ||
      (valid.size(0) != 1 && valid.size(0) != masked.size(0)))
    throw std::invalid_argument("invalid context composite geometry");
  const std::int64_t border = side / 4;
  const nn::Tensor padded =
      nn::pad(patch, {{0, 0}, {border, border}, {border, border}, {0, 0}});
  return nn::where(nn::greater(valid, .5), masked, padded);
}

/// @brief Own normalized partial convolution,updated validity and support counts.
///
/// Tensor output retains x/weight/bias graphs;mask/count are detached.
///
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto r = n::models::PartialConv(n::ones({1, 3, 3, 1}), n::ones({1, 3, 3, 1}),
///                                       n::ones({3, 3, 1, 2}), n::zeros({2}), 1, 2, 3);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
struct PartialConvResult {
  nn::Tensor value; ///< Masked normalized convolution output[N,Ho,Wo,Cout].
  nn::Tensor mask;  ///< Detached output validity,one iff receptive field has support.
  nn::Tensor count; ///< Detached valid-channel counts,shared across output channels.
};

/// @brief Convolve observed channels with valid-count renormalization.
///
/// Masks are detached before counts/multiplication;empty windows output exact zero
/// including bias and its gradient. SAME padding is absent support. Every output
/// channel has the same count/mask. No optimizer or state mutation.
///
/// @param x Floating[N,H,W,in],positive dimensions.
/// @param mask Same-shape/dtype fixed validity,convention binary0/1.
/// @param weights Learned[kernel,kernel,in,out] convolution.
/// @param bias Learned[out] additive bias.
/// @param in Positive input-channel count matching x.
/// @param out Positive output-channel count.
/// @param kernel Positive square kernel size.
/// @param stride Positive SAME convolution stride,default1.
/// @return Value=(conv(x*mask)*kernel²*in/max(count,1)+bias)*valid;updated repeated output mask/count.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible parameter-store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::ones({1, 3, 3, 1}).requires_grad_();
/// const auto r =
///     n::models::PartialConv(x, n::ones_like(x), n::ones({3, 3, 1, 1}), n::zeros({1}), 1, 1, 3);
/// const auto dx = n::grad(n::sum(r.value), {x});
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723.
/// @see ContextEncoder(), ContextDiscriminator(), PartialConv()
inline PartialConvResult PartialConv(const nn::Tensor &x, const nn::Tensor &mask,
                                     const nn::Tensor &weights, const nn::Tensor &bias,
                                     std::int64_t in, std::int64_t out, std::int64_t kernel,
                                     std::int64_t stride = 1) {
  if (in < 1 || out < 1 || kernel < 1 || stride < 1 || x.dim() != 4 || x.size(0) < 1 ||
      x.size(1) < 1 || x.size(2) < 1 || x.size(3) != in || mask.shape() != x.shape() ||
      mask.dtype() != x.dtype() || weights.shape() != nn::Shape{kernel, kernel, in, out} ||
      bias.shape() != nn::Shape{out})
    throw std::invalid_argument("invalid partial convolution geometry");
  const nn::Tensor fixed = mask.detach();
  const auto opts = nn::conv_options(stride, nn::Padding::Same);
  const nn::Tensor counts =
      nn::conv2d(fixed, nn::ones({kernel, kernel, in, out}, mask.dtype()), opts);
  const nn::Tensor valid = nn::cast(nn::greater(counts, 0.0), x.dtype());
  const nn::Tensor ratio =
      nn::div(double(kernel * kernel * in), nn::maximum(counts, nn::scalar_like(1.0, counts)));
  const nn::Tensor conv = nn::conv2d(x * fixed, weights, opts);
  return {nn::bias_add(conv * ratio, bias) * valid, valid, counts};
}

} // namespace datamunge::nn::models
