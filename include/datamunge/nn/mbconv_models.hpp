#pragma once
/// @file mbconv_models.hpp
/// @brief Tensor MBConv/Fused-MBConv and complete MnasNet/EfficientNet/V2 classifiers.
#include <cmath>
#include <limits>
#include <regex>
#include <sstream>

#include "datamunge/nn/mobilenet_models.hpp"
#include "datamunge/nn/model_plot.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;

/// @brief Decoded mobile block architecture metadata.
///
/// Architecture value metadata; no learned weights or fitting.
/// Configuration does not allocate tensors; the forward validates it.
///
/// @par Example
/// @code
/// const auto b = dm::nn::models::DecodeBlock("r2_k3_s22_e6_i16_o24_se0.25");
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
struct BlockArgs {
  int    repeats   = 1;     ///< Positive repeat count, stride applies only to first repeat.
  int    kernel    = 3;     ///< Positive square filter side.
  int    stride    = 1;     ///< Positive square first-repeat spatial step.
  int    expansion = 1;     ///< Positive integer expansion of actual input channels.
  int    input     = 0;     ///< Positive table input width; caller supplies actual tensor width.
  int    output    = 0;     ///< Positive table output width; caller supplies actual output width.
  double se_ratio  = 0;     ///< Finite squeeze ratio in [0,1], relative to actual input width.
  bool   skip      = true;  ///< Permit raw shortcut only for equal widths and stride one.
  bool   fused     = false; ///< Replace pointwise expansion/depthwise with a dense spatial filter.
};
/// @brief Decode a reference block string into validated architecture metadata.
///
/// O(text length) regex parsing; unknown fields and asymmetric strides fail.
/// SE is in [0,1]; repeat/kernel/stride/expansion/widths are positive.
/// Missing r defaults to one; missing SE/skip/fused use BlockArgs defaults.
/// No tensors or learned state. stoi/stod range errors can propagate.
///
/// @param text Underscore fields r,k,s,e,i,o,se,c0/c1,noskip; k/s/e/i/o required.
/// @return Independent block metadata; s2 and s22 both mean square stride two.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @throws std::out_of_range If a parsed integer or floating field exceeds its numeric range.
/// @par Example
/// @code
/// const auto b = dm::nn::models::DecodeBlock("r3_k5_s22_e6_i40_o80_se0.25");
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline BlockArgs DecodeBlock(const std::string& text) {
  BlockArgs         b;
  bool              kernel = false, stride = false, expansion = false;
  std::stringstream parts(text);
  std::string       part;
  while (std::getline(parts, part, '_')) {
    std::smatch m;
    if (part == "noskip")
      b.skip = false;
    else if (std::regex_match(part, m, std::regex("r(\\d+)")))
      b.repeats = std::stoi(m[1]);
    else if (std::regex_match(part, m, std::regex("k(\\d+)"))) {
      b.kernel = std::stoi(m[1]);
      kernel   = true;
    } else if (std::regex_match(part, m, std::regex("s(\\d)(\\d?)"))) {
      if (m[2].length() && m[1] != m[2])
        throw std::invalid_argument("only square strides are supported: " + text);
      b.stride = std::stoi(m[1]);
      stride   = true;
    } else if (std::regex_match(part, m, std::regex("c([01])")))
      b.fused = m[1] == "1";
    else if (std::regex_match(part, m, std::regex("e(\\d+)"))) {
      b.expansion = std::stoi(m[1]);
      expansion   = true;
    } else if (std::regex_match(part, m, std::regex("i(\\d+)")))
      b.input = std::stoi(m[1]);
    else if (std::regex_match(part, m, std::regex("o(\\d+)")))
      b.output = std::stoi(m[1]);
    else if (std::regex_match(part, m, std::regex("se([0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)")))
      b.se_ratio = std::stod(m[1]);
    else
      throw std::invalid_argument("unknown block field '" + part + "' in " + text);
  }
  if (!kernel || !stride || !expansion || b.kernel < 1 || b.stride < 1 || b.expansion < 1 ||
      b.repeats < 1 || b.input < 1 || b.output < 1 || !(b.se_ratio >= 0 && b.se_ratio <= 1))
    throw std::invalid_argument("incomplete block string: " + text);
  return b;
}
/// @brief Encode validated block metadata in normalized reference syntax.
///
/// Does not mutate metadata. Default stream precision serializes SE; arbitrary
/// ratios can round, so exact round-trip applies to released tables. No fitting.
///
/// @param b Positive repeat/filter/stride/expansion/width metadata; stride at most nine.
/// @return Owned string, using doubled square stride, optional positive SE, c1 and noskip.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// const auto text =
///     dm::nn::models::EncodeBlock(dm::nn::models::DecodeBlock("r1_k3_s1_e1_i8_o8"));
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline std::string EncodeBlock(const BlockArgs& b) {
  if (b.repeats < 1 || b.kernel < 1 || b.stride < 1 || b.stride > 9 || b.expansion < 1 ||
      b.input < 1 || b.output < 1 || !(b.se_ratio >= 0 && b.se_ratio <= 1))
    throw std::invalid_argument("invalid block metadata");
  std::ostringstream text;
  text << 'r' << b.repeats << "_k" << b.kernel << "_s" << b.stride << b.stride << "_e"
       << b.expansion << "_i" << b.input << "_o" << b.output;
  if (b.se_ratio > 0)
    text << "_se" << b.se_ratio;
  if (b.fused)
    text << "_c1";
  if (!b.skip)
    text << "_noskip";
  return text.str();
}
// Depth-multiplier rounding of the TensorFlow reference.
/// @brief Scale a channel width with the reference divisor-eight rounding rule.
///
/// Architecture integer arithmetic only. Nearest divisor rounding cannot
/// reduce scaled width by more than ten percent. Product must be representable.
///
/// @param filters Positive base channel count.
/// @param multiplier Finite positive width multiplier.
/// @param divisor Positive alignment, default eight.
/// @return Unchanged filters at multiplier one; otherwise
/// MakeDivisible(filters*multiplier,divisor).
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// const int width = dm::nn::models::RoundFilters(24, 1.3);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline int RoundFilters(int filters, double multiplier, int divisor = 8) {
  if (filters < 1 || !std::isfinite(multiplier) || multiplier <= 0 || divisor < 1)
    throw std::invalid_argument("invalid filter scaling");
  return multiplier == 1 ? filters : MakeDivisible(filters * multiplier, divisor);
}
/// @brief Apply the differentiable swish activation x times logistic sigmoid.
///
/// Elementwise O(numel) tensor work; no learned parameters or mutation.
///
/// @param x Floating tensor; NaNs propagate; storage is not modified.
/// @return Same-shape differentiable activation, retaining tensor dtype.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// const auto y = dm::nn::models::Swish(dm::nn::linspace(-3, 3, 7));
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline nn::Tensor Swish(const nn::Tensor& x) {
  return x * nn::sigmoid(x);
}
/// @brief Apply an inverted per-example survival mask to an NHWC residual branch.
///
/// Each row shares one mask across all spatial/channels. O(numel) work.
/// Endpoint zero returns x*0 without dividing by zero; one consumes no RNG.
/// Caller synchronizes shared generator access. This helper always requests
/// stochastic behavior; models invoke it only for training residuals.
///
/// @param x Nonempty Float32/64 NHWC branch; dtype and graph retained.
/// @param survival Finite probability in [0,1].
/// @param gen Borrowed mutable random generator, advanced only for interior probabilities.
/// @return x*Bernoulli(survival)/survival per row, zero at zero and x at one.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::Generator gen(17);
/// const auto y = n::models::DropConnect(n::ones({8, 2, 2, 4}), .8, gen);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline nn::Tensor DropConnect(const nn::Tensor& x, double survival, nn::Generator& gen) {
  if (x.dim() != 4 || x.numel() == 0 || !(survival >= 0 && survival <= 1) ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid drop-connect tensor/survival");
  if (survival == 0)
    return x * 0;
  if (survival == 1)
    return x;
  const nn::Tensor keep =
      nn::floor(nn::uniform({x.shape()[0], 1, 1, 1}, 0.0, 1.0, gen, x.dtype()) + survival);
  return x / survival * keep;
}
/// @brief Intermediate tensor handles and output from a mobile inverted bottleneck.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// The stores own learned leaves and moving moments; nothing is fitted here.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::MBConv(c, "b", n::ones({1, 5, 5, 8}), 8,
///                                  n::models::DecodeBlock("r1_k3_s11_e2_i8_o8"), 1, true);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
struct MBConvOutputs {
  nn::Tensor expanded;  ///< Post-expansion activation; raw input for expansion one.
  nn::Tensor filtered;  ///< Spatially filtered activation; raw input for fused expansion one.
  nn::Tensor gate;      ///< Logistic [N,1,1,hidden] SE weights, undefined without SE.
  nn::Tensor projected; ///< Linear BN projection, activated only for fused expansion one.
  nn::Tensor output;    ///< Projected branch plus optional raw shortcut; graph retained.
};
/// @brief Apply an inverted or fused mobile bottleneck with optional SE and residual dropout.
///
/// Architecture tracing records the complete residual/SE unit as a named block;
/// layer and operation detail remain available. Ordinary forwards return the same tensors.
///
/// Nonfused: optional 1x1 expansion/BN/act, SAME depthwise/BN/act,
/// optional global-mean SE, linear 1x1/BN. SE reduces to max(1,floor(Cin*ratio))
/// and uses the chosen activation then logistic sigmoid. Fused replaces
/// expansion/depthwise with dense kxk/BN/act; expansion one instead uses
/// a single activated kxk projection and forbids SE. Raw shortcut applies
/// only if skip, stride one and equal widths; branch dropout uses c.p RNG.
/// He-uniform filters; BN epsilon 1e-3 with learned scale/center. Logged
/// MACs include kernels and pooled SE; no fitting or timing is hidden.
///
/// @param c Borrowed learned/state stores, training options and measured-layer log.
/// @param name Parameter prefix for expand/depthwise/se/project leaves.
/// @param x Nonempty floating NHWC input matching store dtype.
/// @param out Positive actual output width.
/// @param b Positive kernel/expansion and SE in [0,1]; table repeats/input/output unused here.
/// @param stride Positive actual spatial step, independent of table first-repeat stride.
/// @param swish True selects x*sigmoid(x), false ReLU.
/// @param drop_rate Residual drop probability in [0,1), used only in training matching shortcuts.
/// @return Intermediate maps and final NHWC output with full gradients.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::MBConv(c, "b", n::ones({1, 5, 5, 8}), 8,
///                                  n::models::DecodeBlock("r1_k3_s1_e2_i8_o8_se0.25"), 1, true);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline MBConvOutputs MBConv(
    MobileContext&     c,
    const std::string& name,
    const nn::Tensor&  x,
    std::int64_t       out,
    const BlockArgs&   b,
    int                stride,
    bool               swish,
    double             drop_rate = 0) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || b.expansion < 1 || b.kernel < 1 || stride < 1 ||
      !(b.se_ratio >= 0 && b.se_ratio <= 1) || !(drop_rate >= 0 && drop_rate < 1) ||
      x.size(3) > std::numeric_limits<int64_t>::max() / b.expansion)
    throw std::invalid_argument("invalid MBConv dimensions or probabilities");
  if (b.fused && b.expansion == 1 && b.se_ratio > 0)
    throw std::invalid_argument("fused expansion-one block cannot have SE");
  auto               act = [&](const nn::Tensor& h) { return swish ? Swish(h) : nn::relu(h); };
  MBConvOutputs      r;
  const std::int64_t in = x.shape()[3], hidden = in * b.expansion;
  const auto         none = nn::Activation::None;
  if (b.fused) {
    // Fused-MBConv: one strided kxk convolution replaces the 1x1 expansion and
    // the depthwise filter; without expansion it is the whole block.
    if (b.expansion == 1)
      r.expanded = r.filtered = x;
    else
      r.expanded = r.filtered = act(ConvBN(c, name + "/expand", x, hidden, b.kernel, stride, none));
  } else {
    r.expanded = b.expansion == 1 ? x : act(ConvBN(c, name + "/expand", x, hidden, 1, 1, none));
    r.filtered = act(ConvBN(
        c,
        name + "/depthwise",
        r.expanded,
        hidden,
        b.kernel,
        stride,
        none,
        MobileConv::kDepthwise));
  }
  nn::Tensor h = r.filtered;
  if (b.se_ratio > 0) {
    const std::int64_t   reduced = std::max<std::int64_t>(1, std::int64_t(in * b.se_ratio));
    nn::ConvLayerOptions o;
    o.init                    = nn::init::he_uniform();
    const nn::Tensor pooled   = nn::mean(h, {1, 2}, true);
    const nn::Tensor squeezed = act(nn::conv2d(c.p, name + "/se/reduce", pooled, reduced, 1, o));
    r.gate = nn::sigmoid(nn::conv2d(c.p, name + "/se/expand", squeezed, hidden, 1, o));
    h      = h * r.gate;
    c.layers.push_back({name + "/se", {1, 1, 1, 1}, 2 * hidden * reduced});
  }
  if (b.fused && b.expansion == 1)
    r.projected = act(ConvBN(c, name + "/project", h, out, b.kernel, stride, none));
  else
    r.projected = ConvBN(c, name + "/project", h, out, 1, 1, none);
  r.output = r.projected;
  if (b.skip && stride == 1 && in == out) {
    nn::Tensor branch = r.projected;
    if (c.training && drop_rate > 0)
      branch = DropConnect(branch, 1 - drop_rate, c.p.generator());
    r.output = x + branch;
  }
  if (nn::model_plot_detail::recording())
    nn::model_stage((b.fused ? "Fused MBConv / " : "MBConv / ") + name, x, r.output);
  return r;
}
/// @brief Return the seven released MnasNet stage specifications.
///
/// A1 and B1 retain their distinct expansion/kernel/repeat/width tables.
/// Architecture metadata only, no search or fitting is performed.
///
/// @param variant a1 or b1, case-sensitive.
/// @return Owned vector: A1 sixteen repeats with selective SE; B1 seventeen without SE.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// const auto blocks = dm::nn::models::MnasNetBlocks("a1");
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline std::vector<BlockArgs> MnasNetBlocks(const std::string& variant) {
  std::vector<std::string> text;
  if (variant == "a1")
    text = {
        "r1_k3_s11_e1_i32_o16_noskip",
        "r2_k3_s22_e6_i16_o24",
        "r3_k5_s22_e3_i24_o40_se0.25",
        "r4_k3_s22_e6_i40_o80",
        "r2_k3_s11_e6_i80_o112_se0.25",
        "r3_k5_s22_e6_i112_o160_se0.25",
        "r1_k3_s11_e6_i160_o320"};
  else if (variant == "b1")
    text = {
        "r1_k3_s11_e1_i32_o16_noskip",
        "r3_k3_s22_e3_i16_o24",
        "r3_k5_s22_e3_i24_o40",
        "r3_k5_s22_e6_i40_o80",
        "r2_k3_s11_e6_i80_o96",
        "r4_k5_s22_e6_i96_o192",
        "r1_k3_s11_e6_i192_o320_noskip"};
  else
    throw std::invalid_argument("MnasNet variants are a1 and b1");
  std::vector<BlockArgs> blocks;
  for (const auto& t : text)
    blocks.push_back(DecodeBlock(t));
  return blocks;
}
/// @brief MnasNet preset, scaling, classifier and normalization metadata.
///
/// Architecture value metadata; no learned weights or fitting.
/// Configuration does not allocate tensors; the forward validates it.
///
/// @par Example
/// @code
/// dm::nn::models::MnasNetConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
struct MnasNetConfig {
  std::string variant          = "a1";  ///< A1 with selective SE, or B1 without SE: a1 or b1.
  double      multiplier       = 1;     ///< Finite positive channel multiplier; head remains 1280.
  int         classes          = 1000;  ///< Classifier width, at least two.
  bool        small_stem       = false; ///< Use stride one in stem, retaining block downsamplings.
  double      keep_probability = .8;    ///< Training head-dropout survival in (0,1].
  double      momentum         = .9;    ///< BN moving-moment retention in [0,1).
  bool        update_stats     = true;  ///< Permit eager training BN updates.
};
/// @brief Build a complete MnasNet A1 or B1 tensor classifier without fitting.
///
/// 3x3 stem, all seven repeated stage groups, unscaled 1280-channel head,
/// ReLU, global mean, training inverted head dropout and biased classifier.
/// Only stem stride changes for small_stem. BN eps 1e-3; no branch drop
/// connect. RNG comes from p. Store/state mutations require caller synchronization.
/// No latency search, preprocessing, loss, optimizer or fitting is hidden.
///
/// @param p Mutable learned store; lazily creates/reuses named floating leaves.
/// @param state Mutable matching-dtype BN moving-moment store.
/// @param image Nonempty NHWC image tensor matching p dtype.
/// @param cfg Valid preset/scaling/classes/dropout/BN metadata, defaults A1.
/// @param training Use batch BN and head dropout; inference uses saved moments.
/// @return Features/logits and convolution/SE/classifier MAC metadata.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::MnasNetConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::MnasNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline MobileNetOutputs MnasNet(
    nn::ParameterStore&  p,
    nn::ParameterStore&  state,
    const nn::Tensor&    image,
    const MnasNetConfig& cfg      = {},
    bool                 training = false) {
  if (image.dim() != 4 || image.numel() == 0 || !std::isfinite(cfg.multiplier) ||
      !(cfg.momentum >= 0 && cfg.momentum < 1) || !(cfg.multiplier > 0) || cfg.classes < 2 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid MnasNet configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  int              in = RoundFilters(32, cfg.multiplier);
  nn::Tensor h     = ConvBN(c, "stem", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::Relu);
  int        index = 0;
  for (const auto& b : MnasNetBlocks(cfg.variant)) {
    const int o = RoundFilters(b.output, cfg.multiplier);
    for (int r = 0; r < b.repeats; ++r, in = o)
      h = MBConv(c, "block" + std::to_string(++index), h, o, b, r ? 1 : b.stride, false).output;
  }
  out.features      = ConvBN(c, "head", h, 1280, 1, 1, nn::Activation::Relu);
  nn::Tensor pooled = nn::global_avg_pool2d(out.features);
  if (training && cfg.keep_probability < 1)
    pooled = nn::dropout(pooled, 1 - cfg.keep_probability, true, p.generator());
  out.logits          = nn::dense(p, "output", pooled, cfg.classes);
  out.classifier_macs = 1280LL * cfg.classes;
  return out;
}

// EfficientNet-B0..B7 (https://arxiv.org/abs/1905.11946, TensorFlow TPU
// reference): the B0 block strings (all with SE 0.25 and swish), scaled by
// width (RoundFilters on every width, including the 32-channel stem and
// 1280-channel head) and depth (ceil(depth * repeats)); each variant's
// resolution and dropout come from the released table. Block i of n drops
// its residual branch with probability drop_connect * i / n in training.
/// @brief Released EfficientNet width/depth/resolution/dropout coefficients.
///
/// Architecture value metadata; no learned weights or fitting.
/// Configuration does not allocate tensors; the forward validates it.
///
/// @par Example
/// @code
/// const auto b3 = dm::nn::models::EfficientNetCoefficients(3);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
struct EfficientNetScaling {
  double width;      ///< Positive channel scaling, rounded to multiples of eight.
  double depth;      ///< Positive repeat scaling, rounded upwards.
  int    resolution; ///< Reference square input side; model itself accepts other sides.
  double dropout;    ///< Training classifier-dropout probability.
};
/// @brief Look up the released EfficientNet B0 through B7 scaling table.
///
/// Immutable local table is initialized once; concurrent reads are safe.
/// Resolution is metadata, not automatic image resizing.
///
/// @param variant Integer B index in [0,7].
/// @return Independent width/depth/reference-resolution/dropout metadata.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// const auto b7 = dm::nn::models::EfficientNetCoefficients(7);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline EfficientNetScaling EfficientNetCoefficients(int variant) {
  /// @brief Immutable released B0 through B7 coefficient table.
  static const EfficientNetScaling table[8] = {
      {1.0, 1.0, 224, .2},
      {1.0, 1.1, 240, .2},
      {1.1, 1.2, 260, .3},
      {1.2, 1.4, 300, .3},
      {1.4, 1.8, 380, .4},
      {1.6, 2.2, 456, .4},
      {1.8, 2.6, 528, .5},
      {2.0, 3.1, 600, .5}};
  if (variant < 0 || variant > 7)
    throw std::invalid_argument("EfficientNet variants are B0 to B7");
  return table[variant];
}
/// @brief Scale all seven EfficientNet baseline stage specifications.
///
/// All stages use SE .25 and swish. Stem/head scaling occurs in EfficientNet().
/// Architecture metadata only; no search or learned tensors.
///
/// @param variant Integer B index in [0,7].
/// @return Owned block vector with rounded input/output widths and ceil-scaled repeats.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// const auto b0 = dm::nn::models::EfficientNetBlocks(0);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline std::vector<BlockArgs> EfficientNetBlocks(int variant) {
  const auto             scale = EfficientNetCoefficients(variant);
  std::vector<BlockArgs> blocks;
  for (const char* text :
       {"r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25"}) {
    auto b    = DecodeBlock(text);
    b.input   = RoundFilters(b.input, scale.width);
    b.output  = RoundFilters(b.output, scale.width);
    b.repeats = int(std::ceil(scale.depth * b.repeats));
    blocks.push_back(b);
  }
  return blocks;
}
/// @brief EfficientNet B0 through B7 forward configuration.
///
/// Architecture value metadata; no learned weights or fitting.
/// Configuration does not allocate tensors; the forward validates it.
///
/// @par Example
/// @code
/// dm::nn::models::EfficientNetConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
struct EfficientNetConfig {
  int    variant    = 0;     ///< Integer B index in [0,7].
  int    classes    = 1000;  ///< Classifier width, at least two.
  bool   small_stem = false; ///< Change stem stride to one.
  double drop_connect =
      .2; ///< Maximum residual drop probability in [0,1), approached across blocks.
  double momentum     = .9;   ///< BN retention in [0,1).
  bool   update_stats = true; ///< Permit eager training BN updates.
};
/// @brief Build a complete B0 through B7 EfficientNet tensor classifier without fitting.
///
/// Scaled stem/head, all ceil-scaled MBConv repeats, swish/SE .25, global
/// mean and biased classifier. Block zero-based index i of n has drop rate
/// cfg.drop_connect*i/n, used only for matching training shortcuts. Final
/// rate stays below the coefficient. Head dropout follows released table.
/// BN eps 1e-3; changing input side does not resize images automatically.
/// No loss, optimizer or fitting; shared store mutations need synchronization.
///
/// @param p Mutable floating learned leaves and dropout generator.
/// @param state Mutable matching-dtype BN moving moments.
/// @param image Nonempty floating NHWC images matching p.
/// @param cfg Valid B index/classes/drop-connect/BN metadata, default B0.
/// @param training Use batch BN, row branch masks and classifier dropout when true.
/// @return Features/logits plus kernel/SE/classifier MAC metadata.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::EfficientNetConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::EfficientNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline MobileNetOutputs EfficientNet(
    nn::ParameterStore&       p,
    nn::ParameterStore&       state,
    const nn::Tensor&         image,
    const EfficientNetConfig& cfg      = {},
    bool                      training = false) {
  const auto scale  = EfficientNetCoefficients(cfg.variant);
  const auto blocks = EfficientNetBlocks(cfg.variant);
  if (image.dim() != 4 || image.numel() == 0 || !(cfg.momentum >= 0 && cfg.momentum < 1) ||
      cfg.classes < 2 || !(cfg.drop_connect >= 0 && cfg.drop_connect < 1))
    throw std::invalid_argument("invalid EfficientNet configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  int              total = 0, index = 0;
  for (const auto& b : blocks)
    total += b.repeats;
  int        in = RoundFilters(32, scale.width);
  nn::Tensor h =
      Swish(ConvBN(c, "stem", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::None));
  for (const auto& b : blocks)
    for (int r = 0; r < b.repeats; ++r, ++index, in = b.output)
      h = MBConv(
              c,
              "block" + std::to_string(index + 1),
              h,
              b.output,
              b,
              r ? 1 : b.stride,
              true,
              cfg.drop_connect * index / total)
              .output;
  const int head    = RoundFilters(1280, scale.width);
  out.features      = Swish(ConvBN(c, "head", h, head, 1, 1, nn::Activation::None));
  nn::Tensor pooled = nn::global_avg_pool2d(out.features);
  if (training)
    pooled = nn::dropout(pooled, scale.dropout, true, p.generator());
  out.logits          = nn::dense(p, "output", pooled, cfg.classes);
  out.classifier_macs = std::int64_t(head) * cfg.classes;
  return out;
}

// EfficientNetV2 (https://arxiv.org/abs/2104.00298, google/automl
// reference): Fused-MBConv early stages ("c1"), swish, SE 0.25 on MBConv
// stages, a stem as wide as the first block's input, a 1280-channel head and
// the same drop-connect schedule. S/M/L use their own block tables at width
// and depth one; the base table B0-B3 is scaled like EfficientNet.
/// @brief Released EfficientNetV2 block table and scaling coefficients.
///
/// Architecture value metadata; no learned weights or fitting.
/// Configuration does not allocate tensors; the forward validates it.
///
/// @par Example
/// @code
/// const auto s = dm::nn::models::EfficientNetV2Spec("s");
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
struct EfficientNetV2Variant {
  std::vector<std::string> blocks;     ///< Nonempty ordered encoded block table.
  double                   width;      ///< Finite positive channel multiplier.
  double                   depth;      ///< Finite positive repeat multiplier.
  int                      resolution; ///< Reference square training/evaluation side metadata.
  double                   dropout;    ///< Classifier-dropout probability in [0,1).
};
/// @brief Look up a released EfficientNetV2 stage table and scaling coefficients.
///
/// S/M/L have distinct searched tables; B0-B3 scale the base table.
/// Early stages use c1 fused blocks, later stages use SE MBConv. No search.
///
/// @param name One of s,m,l,b0,b1,b2,b3, case-sensitive.
/// @return Owned block strings plus width/depth/reference-side/dropout metadata.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// const auto s = dm::nn::models::EfficientNetV2Spec("s");
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline EfficientNetV2Variant EfficientNetV2Spec(const std::string& name) {
  const std::vector<std::string> base = {
      "r1_k3_s1_e1_i32_o16_c1",
      "r2_k3_s2_e4_i16_o32_c1",
      "r2_k3_s2_e4_i32_o48_c1",
      "r3_k3_s2_e4_i48_o96_se0.25",
      "r5_k3_s1_e6_i96_o112_se0.25",
      "r8_k3_s2_e6_i112_o192_se0.25"};
  if (name == "s")
    return {
        {"r2_k3_s1_e1_i24_o24_c1",
         "r4_k3_s2_e4_i24_o48_c1",
         "r4_k3_s2_e4_i48_o64_c1",
         "r6_k3_s2_e4_i64_o128_se0.25",
         "r9_k3_s1_e6_i128_o160_se0.25",
         "r15_k3_s2_e6_i160_o256_se0.25"},
        1,
        1,
        384,
        .2};
  if (name == "m")
    return {
        {"r3_k3_s1_e1_i24_o24_c1",
         "r5_k3_s2_e4_i24_o48_c1",
         "r5_k3_s2_e4_i48_o80_c1",
         "r7_k3_s2_e4_i80_o160_se0.25",
         "r14_k3_s1_e6_i160_o176_se0.25",
         "r18_k3_s2_e6_i176_o304_se0.25",
         "r5_k3_s1_e6_i304_o512_se0.25"},
        1,
        1,
        480,
        .3};
  if (name == "l")
    return {
        {"r4_k3_s1_e1_i32_o32_c1",
         "r7_k3_s2_e4_i32_o64_c1",
         "r7_k3_s2_e4_i64_o96_c1",
         "r10_k3_s2_e4_i96_o192_se0.25",
         "r19_k3_s1_e6_i192_o224_se0.25",
         "r25_k3_s2_e6_i224_o384_se0.25",
         "r7_k3_s1_e6_i384_o640_se0.25"},
        1,
        1,
        480,
        .4};
  if (name == "b0")
    return {base, 1.0, 1.0, 224, .2};
  if (name == "b1")
    return {base, 1.0, 1.1, 240, .2};
  if (name == "b2")
    return {base, 1.1, 1.2, 260, .3};
  if (name == "b3")
    return {base, 1.2, 1.4, 300, .3};
  throw std::invalid_argument("EfficientNetV2 variants are s, m, l and b0-b3");
}
/// @brief Decode and scale a validated EfficientNetV2 specification.
///
/// Host metadata arithmetic only. Rejects integer repeat overflow.
/// No tensor or learned state is created; no progressive schedule is run.
///
/// @param v Nonempty valid block strings; finite positive width/depth, positive side, dropout in
/// [0,1).
/// @return Owned stages with rounded widths and ceil-scaled positive repeats.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// const auto blocks =
///     dm::nn::models::EfficientNetV2Blocks(dm::nn::models::EfficientNetV2Spec("b0"));
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline std::vector<BlockArgs> EfficientNetV2Blocks(const EfficientNetV2Variant& v) {
  if (v.blocks.empty() || !std::isfinite(v.width) || v.width <= 0 || !std::isfinite(v.depth) ||
      v.depth <= 0 || v.resolution < 1 || !(v.dropout >= 0 && v.dropout < 1))
    throw std::invalid_argument("invalid V2 specification");
  std::vector<BlockArgs> blocks;
  for (const auto& text : v.blocks) {
    auto b               = DecodeBlock(text);
    b.input              = RoundFilters(b.input, v.width);
    b.output             = RoundFilters(b.output, v.width);
    const double repeats = std::ceil(v.depth * b.repeats);
    if (repeats > std::numeric_limits<int>::max())
      throw std::invalid_argument("V2 repeats overflow");
    b.repeats = int(repeats);
    blocks.push_back(b);
  }
  return blocks;
}
/// @brief EfficientNetV2 S/M/L or base B0 through B3 forward configuration.
///
/// Architecture value metadata; no learned weights or fitting.
/// Configuration does not allocate tensors; the forward validates it.
///
/// @par Example
/// @code
/// dm::nn::models::EfficientNetV2Config cfg;
/// cfg.variant = "b0";
/// cfg.classes = 10;
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
struct EfficientNetV2Config {
  std::string variant      = "s";   ///< One of s, m, l, b0, b1, b2, b3.
  int         classes      = 1000;  ///< Classifier width, at least two.
  bool        small_stem   = false; ///< Change only stem stride to one.
  double      drop_connect = .2;    ///< Residual drop-rate schedule coefficient in [0,1).
  double      momentum     = .9;    ///< BN retention in [0,1).
  bool        update_stats = true;  ///< Permit eager training BN updates.
};
/// @brief Build a complete fused/SE EfficientNetV2 tensor classifier without fitting.
///
/// Stem width comes from first scaled table input; early fused blocks and
/// later SE MBConv retain every repeat, swish, scaled 1280-channel head,
/// global mean and biased classifier. BN eps 1e-3. Residual drop rate is
/// cfg.drop_connect*i/n at zero-based block index i, only for matching
/// training shortcuts. Reference resolution is metadata. No progressive
/// resizing/regularization schedule, loss, optimizer or fitting is hidden.
/// Shared state/RNG mutations require caller synchronization.
///
/// @param p Mutable floating learned leaves and dropout generator.
/// @param state Mutable matching-dtype BN moving moments.
/// @param image Nonempty floating NHWC images matching p.
/// @param cfg Valid released variant/classes/drop-connect/BN metadata, default S.
/// @param training Use batch BN, per-row branch dropout and classifier dropout when true.
/// @return Features/logits plus kernel/pooled-SE/classifier MAC metadata.
/// @throws std::invalid_argument For invalid domain, unknown preset, unsupported dtype/shape,
/// overflowing dimensions or incompatible stored entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::EfficientNetV2Config cfg;
/// cfg.variant = "b0";
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::EfficientNetV2(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Tan et al. (2019), https://arxiv.org/abs/1807.11626; Tan and Le (2019),
/// https://arxiv.org/abs/1905.11946; Tan and Le (2021), https://arxiv.org/abs/2104.00298;
/// https://github.com/google/automl/tree/master/efficientnetv2
/// @see BlockArgs, MBConv(), MnasNet(), EfficientNet(), EfficientNetV2()
inline MobileNetOutputs EfficientNetV2(
    nn::ParameterStore&         p,
    nn::ParameterStore&         state,
    const nn::Tensor&           image,
    const EfficientNetV2Config& cfg      = {},
    bool                        training = false) {
  const auto spec   = EfficientNetV2Spec(cfg.variant);
  const auto blocks = EfficientNetV2Blocks(spec);
  if (image.dim() != 4 || image.numel() == 0 || !(cfg.momentum >= 0 && cfg.momentum < 1) ||
      cfg.classes < 2 || !(cfg.drop_connect >= 0 && cfg.drop_connect < 1))
    throw std::invalid_argument("invalid EfficientNetV2 configuration");
  MobileNetOutputs out;
  MobileContext    c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  int              total = 0, index = 0;
  for (const auto& b : blocks)
    total += b.repeats;
  int        in = blocks.front().input;
  nn::Tensor h =
      Swish(ConvBN(c, "stem", image, in, 3, cfg.small_stem ? 1 : 2, nn::Activation::None));
  for (const auto& b : blocks)
    for (int r = 0; r < b.repeats; ++r, ++index, in = b.output)
      h = MBConv(
              c,
              "block" + std::to_string(index + 1),
              h,
              b.output,
              b,
              r ? 1 : b.stride,
              true,
              cfg.drop_connect * index / total)
              .output;
  const int head    = RoundFilters(1280, spec.width);
  out.features      = Swish(ConvBN(c, "head", h, head, 1, 1, nn::Activation::None));
  nn::Tensor pooled = nn::global_avg_pool2d(out.features);
  if (training)
    pooled = nn::dropout(pooled, spec.dropout, true, p.generator());
  out.logits          = nn::dense(p, "output", pooled, cfg.classes);
  out.classifier_macs = std::int64_t(head) * cfg.classes;
  return out;
}
} // namespace datamunge::nn::models
