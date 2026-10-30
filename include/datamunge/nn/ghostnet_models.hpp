#pragma once
/// @file ghostnet_models.hpp
/// @brief Intrinsic/cheap ghost modules and complete GhostNet tensor classifier.
#include "datamunge/nn/mobilenet_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;

/// @brief Pad both NHWC spatial axes symmetrically with zeros.
///
/// Explicit symmetric padding fixes stride-two sample alignment. Work/storage
/// linear in padded tensor size; no parameters or fitting.
///
/// @param x Nonempty rank-four tensor, graph retained.
/// @param pad Nonnegative number of rows/columns on both sides.
/// @return Padded tensor, input storage unchanged.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// const auto y = dm::nn::models::GhostPadSpatial(dm::nn::ones({1, 3, 3, 2}), 1);
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostModule(), GhostBottleneck(), GhostNet()
inline nn::Tensor GhostPadSpatial(const nn::Tensor &x, std::int64_t pad) {
  if (x.dim() != 4 || x.numel() == 0 || pad < 0)
    throw std::invalid_argument("invalid ghost spatial padding");
  return nn::pad(x, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}});
}
/// @brief Apply a symmetric-padded dense/depthwise filter, BN and optional ReLU.
///
/// Bias-free He-uniform filters, learned-scale BN epsilon 1e-5. Depthwise
/// products per position are kernel²*out, dense kernel²*Cin*out.
/// No preprocessing, loss or fitting. Context stores/log are mutated lazily.
///
/// @param c Borrowed mobile learned/state stores and mutable arithmetic log.
/// @param name Prefix for filter and /bn leaves.
/// @param x Nonempty floating NHWC input matching stores.
/// @param out Positive output channels; depthwise requires a positive multiple of Cin.
/// @param kernel Positive odd square filter width.
/// @param stride Positive spatial step.
/// @param depthwise True uses depth multiplier out/Cin, otherwise dense convolution.
/// @param relu True applies ReLU after BN; false returns linear BN.
/// @return Differentiable NHWC map with a logged measured shape/MAC budget.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::GhostConvBN(c, "c", n::ones({1, 5, 5, 3}), 6, 3, 1, true, false);
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostModule(), GhostBottleneck(), GhostNet()
inline nn::Tensor GhostConvBN(MobileContext &c, const std::string &name, nn::Tensor x,
                              std::int64_t out, std::int64_t kernel, std::int64_t stride,
                              bool depthwise, bool relu) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || kernel < 1 || kernel % 2 != 1 ||
      stride < 1 || (depthwise && out % x.size(3)))
    throw std::invalid_argument("invalid ghost convolution dimensions");
  const std::int64_t in = x.shape()[3];
  nn::ConvLayerOptions o;
  o.bias = false;
  o.init = nn::init::he_uniform();
  o.stride = stride;
  o.padding = nn::Padding::Valid;
  if (kernel > 1)
    x = GhostPadSpatial(x, kernel / 2);
  nn::Tensor h = depthwise ? nn::depthwise_conv2d(c.p, name, x, out / in, kernel, o)
                           : nn::conv2d(c.p, name, x, out, kernel, o);
  h = ContextBatchNorm(c, name + "/bn", h, 1e-5);
  c.layers.push_back({name, h.shape(), kernel * kernel * (depthwise ? out : in * out)});
  return relu ? nn::relu(h) : h;
}
/// @brief Intrinsic and cheaply generated maps plus a possibly trimmed concatenation.
///
/// Handles retain storage/graphs. Intrinsic width is ceil(out/ratio), generated
/// width is intrinsic*(ratio-1), and only output is trimmed to requested channels.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::GhostModule(c, "g", n::ones({1, 5, 5, 3}), 5, false);
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostModule()
struct GhostModuleOutputs {
  nn::Tensor intrinsic; ///< Primary 1x1/BN with optional ReLU, ceil(out/ratio) channels.
  nn::Tensor ghost;     ///< Depthwise 3x3/BN-generated channels, before output trimming.
  nn::Tensor output; ///< [intrinsic,ghost] concatenation, first requested out channels only.
};
/// @brief Generate intrinsic maps, derive cheap depthwise maps and trim concatenated channels.
///
/// Primary 1x1 emits ceil(out/ratio); cheap depthwise 3x3 multiplier
/// ratio-1 generates the remaining maps. Output order is intrinsic first,
/// then depthwise channel/multiplier order, trimmed from the end. Unused
/// generated channels have zero data gradient. Work follows both filters;
/// BN/activation makes the general module nonlinear. No fitting.
///
/// @param c Borrowed stores/log and BN mode.
/// @param name Prefix for primary and cheap layers.
/// @param x Nonempty floating NHWC input matching store dtype.
/// @param out Positive requested output width, possibly not divisible by ratio.
/// @param relu Apply ReLU after both primary/cheap BN stages when true.
/// @param ratio Integer expansion ratio at least two, default two.
/// @return Intrinsic, generated and exactly out-channel output maps, graph retained.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::GhostModule(c, "g", n::ones({1, 5, 5, 3}), 5, false);
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostModule(), GhostBottleneck(), GhostNet()
inline GhostModuleOutputs GhostModule(MobileContext &c, const std::string &name,
                                      const nn::Tensor &x, std::int64_t out, bool relu,
                                      int ratio = 2) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || ratio < 2)
    throw std::invalid_argument("invalid ghost module dimensions/ratio");
  const std::int64_t init = (out + ratio - 1) / ratio;
  GhostModuleOutputs g;
  g.intrinsic = GhostConvBN(c, name + "/primary", x, init, 1, 1, false, relu);
  g.ghost = GhostConvBN(c, name + "/cheap", g.intrinsic, init * (ratio - 1), 3, 1, true, relu);
  g.output = nn::concat({g.intrinsic, g.ghost}, 3);
  if (init * ratio != out)
    g.output = nn::slice(g.output, {0, 0, 0, 0}, {-1, -1, -1, out});
  return g;
}
/// @brief One released GhostNet bottleneck specification before width scaling.
///
/// Value metadata without learned allocation/fitting. Widths are rounded
/// with MakeDivisible(...,4) by the model.
///
/// @par Example
/// @code
/// const auto blocks = dm::nn::models::GhostNetBlocks(); // 16 specifications
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostNet()
struct GhostBlock {
  int kernel;   ///< Odd depthwise/shortcut filter width, three or five.
  int expanded; ///< Internal ghost-module target width before global scaling.
  int out;      ///< Output bottleneck width before global scaling.
  double se;    ///< SE reduction ratio, zero or .25 in released blocks.
  int stride;   ///< Spatial step, one or two.
};
/// @brief Return all sixteen released GhostNet bottleneck specifications.
///
/// Contains exact kernel/expanded/output/SE/stride metadata. Does not
/// allocate or fit learned tensors; O(16) copy work/storage.
///
/// @return Value vector in execution order, before width scaling.
/// @par Example
/// @code
/// const auto blocks = dm::nn::models::GhostNetBlocks(); // 16 blocks
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostModule(), GhostBottleneck(), GhostNet()
inline std::vector<GhostBlock> GhostNetBlocks() {
  return {{3, 16, 16, 0, 1},     {3, 48, 24, 0, 2},     {3, 72, 24, 0, 1},
          {5, 72, 40, .25, 2},   {5, 120, 40, .25, 1},  {3, 240, 80, 0, 2},
          {3, 200, 80, 0, 1},    {3, 184, 80, 0, 1},    {3, 184, 80, 0, 1},
          {3, 480, 112, .25, 1}, {3, 672, 112, .25, 1}, {5, 672, 160, .25, 2},
          {5, 960, 160, 0, 1},   {5, 960, 160, .25, 1}, {5, 960, 160, 0, 1},
          {5, 960, 160, .25, 1}};
}
/// @brief Apply two ghost modules, optional strided filtering/SE and a shortcut sum.
///
/// Ghost1 uses ReLU, Ghost2 stays linear. Stride two inserts depthwise/BN
/// between them. SE pools to 1x1, reduces to MakeDivisible(mid*se_ratio,4)
/// with ReLU, expands with hard sigmoid and scales channels. Shape changes
/// use depthwise/BN then pointwise/BN shortcut; equal shape carries x.
/// There is no post-addition activation. No optimizer or fitting.
///
/// @param c Borrowed learned/state stores and arithmetic log.
/// @param name Prefix for ghost1, depthwise, se, ghost2 and shortcut leaves.
/// @param x Nonempty floating NHWC input matching stores.
/// @param mid Positive internal expansion target.
/// @param out Positive bottleneck output target.
/// @param kernel Positive odd stride/shortcut depthwise side.
/// @param stride One or two.
/// @param se_ratio Finite channel-reduction fraction in [0,1]; zero disables SE.
/// @return Signed residual-plus-shortcut NHWC tensor with gradients.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::GhostBottleneck(c, "b", n::ones({1, 8, 8, 8}), 16, 8, 3, 1, .25);
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostModule(), GhostBottleneck(), GhostNet()
inline nn::Tensor GhostBottleneck(MobileContext &c, const std::string &name,
                                  const nn::Tensor &x, std::int64_t mid, std::int64_t out,
                                  int kernel, int stride, double se_ratio) {
  if (x.dim() != 4 || x.numel() == 0 || mid < 1 || out < 1 || kernel < 1 || kernel % 2 != 1 ||
      (stride != 1 && stride != 2) || !(se_ratio >= 0 && se_ratio <= 1))
    throw std::invalid_argument("invalid ghost bottleneck dimensions/SE ratio");
  const std::int64_t in = x.shape()[3];
  nn::Tensor h = GhostModule(c, name + "/ghost1", x, mid, true).output;
  if (stride > 1)
    h = GhostConvBN(c, name + "/depthwise", h, mid, kernel, stride, true, false);
  if (se_ratio > 0) {
    const int reduced = MakeDivisible(mid * se_ratio, 4);
    nn::ConvLayerOptions o;
    o.init = nn::init::he_uniform();
    const nn::Tensor pooled = nn::mean(h, {1, 2}, true);
    const nn::Tensor squeezed =
        nn::relu(nn::conv2d(c.p, name + "/se/reduce", pooled, reduced, 1, o));
    h = h * HardSigmoid(nn::conv2d(c.p, name + "/se/expand", squeezed, mid, 1, o));
    c.layers.push_back({name + "/se", {1, 1, 1, 1}, 2 * mid * reduced});
  }
  h = GhostModule(c, name + "/ghost2", h, out, false).output;
  nn::Tensor shortcut = x;
  if (in != out || stride != 1) {
    shortcut =
        GhostConvBN(c, name + "/shortcut/depthwise", x, in, kernel, stride, true, false);
    shortcut = GhostConvBN(c, name + "/shortcut/pointwise", shortcut, out, 1, 1, false, false);
  }
  return h + shortcut;
}
/// @brief GhostNet width, stem, classifier dropout and normalization settings.
///
/// Defaults describe the 1.0x ImageNet network. Metadata only, no allocation or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::GhostNetConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostNet()
struct GhostNetConfig {
  double width = 1;   ///< Finite positive channel multiplier, aligned to four.
  int classes = 1000; ///< At least two output logits.
  bool small_stem =
      false; ///< First stem stride one for small images, retaining later downsamplings.
  double keep_probability = .8; ///< Head dropout retention in (0,1], training only.
  double momentum = .9;         ///< BN moving retention in [0,1).
  bool update_stats = true;     ///< Permit eager moving moments to update during training.
};
/// @brief Build all sixteen GhostNet bottlenecks and the tensor classifier without fitting.
///
/// Stem is 3x3/2/BN/ReLU, or stride one in small_stem mode. Scaled widths
/// align to four; all sixteen blocks remain. Final 1x1/BN/ReLU targets
/// MakeDivisible(960*width,4), global mean, biased 1x1/ReLU to 1280,
/// training dropout and biased Glorot classifier. Dropout consumes
/// p.generator(); no preprocessing, loss, optimizer or fitting is hidden.
/// He-uniform fan-in is an initialization adaptation; BN eps 1e-5 learns
/// gamma/beta. Logged MACs include convolutions/pooled SE/head/classifier,
/// excluding elementwise/pooling/BN work; they do not measure latency.
///
/// @param p Mutable floating learned store, lazily creates/reuses leaves.
/// @param state Mutable moving BN moments matching p dtype.
/// @param image Nonempty floating NHWC images.
/// @param cfg Valid width, class, dropout and BN settings; defaults ImageNet 1.0x.
/// @param training True uses batch moments and dropout, false saved moments.
/// @return MobileNet-style final features, logits and measured arithmetic metadata.
/// @throws std::invalid_argument For invalid dimensions/settings, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::GhostNetConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::GhostNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Han et al. (2020), https://arxiv.org/abs/1911.11907; https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
/// @see GhostModule(), GhostBottleneck(), GhostNet()
inline MobileNetOutputs GhostNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                 const nn::Tensor &image, const GhostNetConfig &cfg = {},
                                 bool training = false) {
  if (image.dim() != 4 || image.numel() == 0 || !std::isfinite(cfg.width) ||
      cfg.width > double(std::numeric_limits<int>::max() / 1280) ||
      !(cfg.momentum >= 0 && cfg.momentum < 1))
    throw std::invalid_argument("invalid GhostNet input/width/BN settings");
  if (!(cfg.width > 0) || cfg.classes < 2 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid GhostNet configuration");
  MobileNetOutputs out;
  MobileContext c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  int in = MakeDivisible(16 * cfg.width, 4);
  nn::Tensor h = GhostConvBN(c, "stem", image, in, 3, cfg.small_stem ? 1 : 2, false, true);
  int index = 0;
  for (const auto &b : GhostNetBlocks()) {
    const int mid = MakeDivisible(b.expanded * cfg.width, 4),
              o = MakeDivisible(b.out * cfg.width, 4);
    h = GhostBottleneck(c, "block" + std::to_string(++index), h, mid, o, b.kernel, b.stride,
                        b.se);
    in = o;
  }
  const int last = MakeDivisible(960 * cfg.width, 4);
  out.features = GhostConvBN(c, "conv_last", h, last, 1, 1, false, true);
  nn::ConvLayerOptions head;
  head.init = nn::init::he_uniform();
  head.activation = nn::Activation::Relu;
  nn::Tensor v = nn::reshape(
      nn::conv2d(p, "head", nn::mean(out.features, {1, 2}, true), 1280, 1, head), {-1, 1280});
  if (training && cfg.keep_probability < 1)
    v = nn::dropout(v, 1 - cfg.keep_probability, true, p.generator());
  out.logits = nn::dense(p, "output", v, cfg.classes);
  out.classifier_macs = std::int64_t(last) * 1280 + 1280LL * cfg.classes;
  return out;
}
} // namespace datamunge::nn::models
