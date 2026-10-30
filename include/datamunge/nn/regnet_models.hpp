#pragma once
/// @file regnet_models.hpp
/// @brief Quantized RegNet architecture metadata and complete X/Y tensor classifiers.
#include "datamunge/nn/classic_cnn_components.hpp"
#include "datamunge/nn/mobilenet_models.hpp"
#include <cfenv>
#include <cmath>
#include <map>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Quantized linear-width design parameters for one RegNet release.
///
/// Architecture metadata only. Depth counts bottleneck blocks; group is the
/// channels per 3x3 group, not the number of groups. X uses se=0, Y uses .25.
///
/// @par Example
/// @code
/// const auto m = dm::nn::models::RegNetModel("y400mf");
/// @endcode
/// @par References
/// - Radosavovic et al. (2020), https://arxiv.org/abs/2003.13678; https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
/// @see RegNetDesign()
struct RegNetParams {
  int depth; ///< Positive total number of bottleneck blocks.
  int w0;    ///< Positive initial continuous width, divisible by eight.
  double wa; ///< Finite nonnegative additive width slope per block.
  double wm; ///< Finite multiplicative quantization step greater than one.
  int group; ///< Positive target channels per grouped 3x3 transformation.
  double se; ///< Finite SE reduction ratio in [0,1], relative to block input width.
};
/// @brief Return the design parameters for a released RegNetX/Y model name.
///
/// The fourteen tables follow the reference release. No weights or fitting
/// are loaded; lookup/copy is O(log 14). Unknown names are rejected.
///
/// @param name x400mf/x800mf/x1.6gf/x3.2gf/x8gf/x16gf/x32gf or corresponding y names.
/// @return Independent depth/width/group/SE metadata.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/settings, width overflow, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// const auto m = dm::nn::models::RegNetModel("y400mf");
/// @endcode
/// @par References
/// - Radosavovic et al. (2020), https://arxiv.org/abs/2003.13678; https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
/// @see RegNetModel(), RegNetDesign(), RegNet()
inline RegNetParams RegNetModel(const std::string &name) {
  /// @brief Fourteen released X/Y design presets, copied to callers.
  static const std::map<std::string, RegNetParams> models = {
      {"x400mf", {22, 24, 24.48, 2.54, 16, 0}},
      {"x800mf", {16, 56, 35.73, 2.28, 16, 0}},
      {"x1.6gf", {18, 80, 34.01, 2.25, 24, 0}},
      {"x3.2gf", {25, 88, 26.31, 2.25, 48, 0}},
      {"x8gf", {23, 80, 49.56, 2.88, 120, 0}},
      {"x16gf", {22, 216, 55.59, 2.1, 128, 0}},
      {"x32gf", {23, 320, 69.86, 2.0, 168, 0}},
      {"y400mf", {16, 48, 27.89, 2.09, 8, .25}},
      {"y800mf", {14, 56, 38.84, 2.4, 16, .25}},
      {"y1.6gf", {27, 48, 20.71, 2.65, 24, .25}},
      {"y3.2gf", {21, 80, 42.63, 2.66, 24, .25}},
      {"y8gf", {17, 192, 76.82, 2.19, 56, .25}},
      {"y16gf", {18, 200, 106.23, 2.48, 112, .25}},
      {"y32gf", {20, 232, 115.89, 2.53, 232, .25}}};
  if (!models.count(name))
    throw std::invalid_argument("unknown RegNet model " + name);
  return models.at(name);
}
/// @brief Per-block continuous/quantized widths and grouped stage metadata.
///
/// Value vectors contain no activation or learned state. Consecutive equal
/// quantized widths define stages before group-compatibility adjustment.
///
/// @par Example
/// @code
/// const auto d = dm::nn::models::RegNetDesign(dm::nn::models::RegNetModel("y400mf"));
/// @endcode
/// @par References
/// - Radosavovic et al. (2020), https://arxiv.org/abs/2003.13678; https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
/// @see RegNetDesign()
struct RegNetStages {
  std::vector<float> continuous; ///< Float32 w0+wa*j for each block j.
  std::vector<int> quantized; ///< Eight-aligned multiplicatively quantized per-block widths.
  std::vector<int> widths;    ///< Stage output widths, adjusted to group-width compatibility.
  std::vector<int> depths;    ///< Number of consecutive blocks in each stage.
  std::vector<int> groups;    ///< Effective channel width per 3x3 group (not group count).
};
/// @brief Quantize a Float32 linear width rule into grouped constant-width stages.
///
/// Use u_j=w0+wa*j, k_j=round(log(u_j/w0)/log(wm)), then
/// q_j=8*round(w0*wm^k_j/8), matching Float32 reference arithmetic and
/// ties-to-even rounding. Consecutive equal q_j become stages. Effective
/// group width is min(group,width); adjust each stage width with
/// MakeDivisible(width,group_width). Architecture metadata uses host scalars,
/// not learned model computation. O(depth) work/storage; no fitting.
/// Temporarily sets this thread rounding mode to nearest and restores it
/// on normal or exceptional exit. Widths must remain int-representable.
///
/// @param m Valid finite design metadata with positive depth, eight-aligned w0 and group width.
/// @return Per-block continuous/quantized widths and stage widths/depths/group widths.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/settings, width overflow, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// const auto d = dm::nn::models::RegNetDesign(dm::nn::models::RegNetModel("y400mf"));
/// @endcode
/// @par References
/// - Radosavovic et al. (2020), https://arxiv.org/abs/2003.13678; https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
/// @see RegNetModel(), RegNetDesign(), RegNet()
inline RegNetStages RegNetDesign(const RegNetParams &m) {
  if (m.depth < 1 || m.w0 <= 0 || m.w0 % 8 || !std::isfinite(m.wa) || m.wa < 0 ||
      !std::isfinite(m.wm) || m.wm <= 1 || m.group < 1 || !(m.se >= 0 && m.se <= 1))
    throw std::invalid_argument("invalid RegNet parameters");
  RegNetStages d;
  /// @brief Restore this thread rounding mode even if width generation throws.
  struct RestoreRounding {
    int mode = std::fegetround(); ///< Rounding mode observed before setting nearest.
    RestoreRounding() {
      std::fesetround(FE_TONEAREST);
    }
    ~RestoreRounding() {
      std::fesetround(mode);
    }
  } rounding;

  const float w0 = float(m.w0), wa = float(m.wa), wm = float(m.wm),
              log_wm = float(std::log(m.wm));
  for (int j = 0; j < m.depth; ++j) {
    const float u = float(j) * wa + w0, step = std::nearbyint(std::log(u / w0) / log_wm);
    d.continuous.push_back(u);
    const float quantized = std::nearbyint(w0 * std::pow(wm, step) / 8.f);
    if (!std::isfinite(quantized) || quantized < 1 ||
        double(quantized) > double(std::numeric_limits<int>::max() / 8))
      throw std::invalid_argument("RegNet quantized width overflow");
    d.quantized.push_back(int(quantized) * 8);
  }
  for (int w : d.quantized) {
    if (!d.widths.empty() && d.widths.back() == w)
      ++d.depths.back();
    else {
      d.widths.push_back(w);
      d.depths.push_back(1);
    }
  }
  for (auto &w : d.widths) {
    const int g = std::min(m.group, w);
    d.groups.push_back(g);
    w = MakeDivisible(w, g);
  }
  return d;
}

/// @brief Apply logged dense/grouped symmetric-padded convolution, BN and optional ReLU.
///
/// Uses FusedGroupedConv2D with bias-free He-uniform filters. Explicit
/// symmetric padding fixes strided alignment. Learned-scale BN eps 1e-5.
/// MACs per output position are kernel²*Cin*Cout/groups. No fitting.
///
/// @param c Borrowed learned/state stores and mutable arithmetic log.
/// @param name Prefix, retaining per-group groupN/W leaf names.
/// @param x Nonempty floating NHWC input matching stores; handle copied.
/// @param out Positive output channels.
/// @param kernel Positive odd square filter side.
/// @param stride Positive spatial step.
/// @param groups Positive divisor of input/output widths, one for dense.
/// @param relu True applies ReLU after BN, false linear BN.
/// @return Differentiable NHWC map and one appended measured arithmetic entry.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/settings, width overflow, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::RegNetConvBN(c, "c", n::ones({1, 5, 5, 8}), 8, 3, 1, 2, true);
/// @endcode
/// @par References
/// - Radosavovic et al. (2020), https://arxiv.org/abs/2003.13678; https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
/// @see RegNetModel(), RegNetDesign(), RegNet()
inline nn::Tensor RegNetConvBN(MobileContext &c, const std::string &name, nn::Tensor x,
                               int out, int kernel, int stride, int groups, bool relu) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || kernel < 1 || kernel % 2 != 1 ||
      stride < 1 || groups < 1 || x.size(3) % groups || out % groups)
    throw std::invalid_argument("invalid RegNet convolution dimensions");
  const auto in = x.size(3);
  nn::ConvLayerOptions cfg;
  cfg.bias = false;
  cfg.init = nn::init::he_uniform();
  cfg.stride = stride;
  cfg.padding = nn::Padding::Valid;
  if (kernel > 1)
    x = nn::pad(x, {{0, 0}, {kernel / 2, kernel / 2}, {kernel / 2, kernel / 2}, {0, 0}});
  auto h = FusedGroupedConv2D(c.p, name, x, out, kernel, groups, cfg);
  h = ContextBatchNorm(c, name + "/bn", h, 1e-5);
  c.layers.push_back({name, h.shape(), std::int64_t(kernel) * kernel * in * out / groups});
  return relu ? nn::relu(h) : h;
}
/// @brief Apply a grouped residual bottleneck with optional logistic SE and post-add ReLU.
///
/// 1x1/BN/ReLU, grouped 3x3/BN/ReLU, optional global-mean SE,
/// linear 1x1/BN, shortcut sum/ReLU. SE hidden width is lround(se*Cin),
/// not output width; it must be positive when enabled. Matching width/stride
/// uses raw identity, otherwise 1x1/BN projection. No fitting.
///
/// @param c Borrowed learned/state stores and measured-layer log.
/// @param name Prefix for a/b/c convolutions, optional se and projection.
/// @param x Nonempty floating NHWC input matching stores.
/// @param out Positive bottleneck/output width.
/// @param stride One or two, applied in the grouped 3x3 and shortcut projection.
/// @param group_width Positive channels per group, dividing out.
/// @param se Finite reduction ratio in [0,1], relative to input width; zero disables SE.
/// @return NHWC post-addition ReLU map retaining branch/shortcut gradients.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/settings, width overflow, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::models::MobileLayer> layers;
/// n::models::MobileContext c{p, state, layers, false};
/// const auto y = n::models::RegNetBlock(c, "b", n::ones({1, 8, 8, 16}), 24, 2, 8, .25);
/// @endcode
/// @par References
/// - Radosavovic et al. (2020), https://arxiv.org/abs/2003.13678; https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
/// @see RegNetModel(), RegNetDesign(), RegNet()
inline nn::Tensor RegNetBlock(MobileContext &c, const std::string &name, const nn::Tensor &x,
                              int out, int stride, int group_width, double se) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || group_width < 1 || out % group_width ||
      (stride != 1 && stride != 2) || !(se >= 0 && se <= 1) ||
      (se > 0 && std::lround(se * x.size(3)) < 1))
    throw std::invalid_argument("invalid RegNet block dimensions/SE");
  const int in = x.size(3);
  auto h = RegNetConvBN(c, name + "/a", x, out, 1, 1, 1, true);
  h = RegNetConvBN(c, name + "/b", h, out, 3, stride, out / group_width, true);
  if (se > 0) {
    const int squeeze = std::lround(se * in);
    nn::ConvLayerOptions cfg;
    cfg.init = nn::init::he_uniform();
    auto pooled = nn::mean(h, {1, 2}, true);
    auto squeezed = nn::relu(nn::conv2d(c.p, name + "/se/reduce", pooled, squeeze, 1, cfg));
    h = nn::mul(h, nn::sigmoid(nn::conv2d(c.p, name + "/se/expand", squeezed, out, 1, cfg)));
    c.layers.push_back({name + "/se", {1, 1, 1, 1}, 2 * std::int64_t(out) * squeeze});
  }
  h = RegNetConvBN(c, name + "/c", h, out, 1, 1, 1, false);
  auto shortcut = in == out && stride == 1
                      ? x
                      : RegNetConvBN(c, name + "/projection", x, out, 1, stride, 1, false);
  return nn::relu(nn::add(shortcut, h));
}
/// @brief Released RegNet model name, classifier, stem and BN options.
///
/// Defaults describe ImageNet Y-400MF. Metadata only; the forward validates
/// settings and lazily creates tensors. No optimizer or fitting state.
///
/// @par Example
/// @code
/// dm::nn::models::RegNetConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Radosavovic et al. (2020), https://arxiv.org/abs/2003.13678; https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
/// @see RegNet()
struct RegNetConfig {
  std::string model =
      "y400mf";             ///< One of fourteen released X/Y preset names from RegNetModel().
  int classes = 1000;       ///< At least two output classes.
  bool small_stem = false;  ///< Stem stride one, retaining all stage-entry downsamplings.
  double momentum = .9;     ///< BN moving-moment retention in [0,1).
  bool update_stats = true; ///< Permit eager moving updates in training mode.
};
/// @brief Build a complete released RegNetX/Y tensor classifier without fitting.
///
/// A 32-channel 3x3/2/BN/ReLU stem (stride one for small_stem) precedes
/// the generated stages. Every stage entry downsamples by two, including
/// the first. Stage bottlenecks preserve output width (ratio one); X omits
/// SE, Y uses logistic SE with input-based reduction. Global mean and a
/// biased Glorot classifier finish the model. No dropout, preprocessing,
/// loss, optimizer or fitting is hidden. He-uniform fan-in is an adaptation;
/// BN eps 1e-5 learns gamma/beta. Reported MACs count kernels/SE/classifier,
/// excluding pooling/BN/activation/addition and are not measured runtime.
///
/// @param p Mutable floating learned store, lazily creates/reuses all leaves.
/// @param state Mutable moving BN store matching p dtype.
/// @param image Nonempty floating NHWC images.
/// @param cfg Valid model, classes, stem and BN metadata, default Y-400MF.
/// @param training True uses batch moments, optionally updating state; false saved moments.
/// @return MobileNet-style features/logits and measured convolution/SE/classifier metadata.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/settings, width overflow, dtype, normalization or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RegNetConfig cfg;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// const auto y = n::models::RegNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Radosavovic et al. (2020), https://arxiv.org/abs/2003.13678; https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
/// @see RegNetModel(), RegNetDesign(), RegNet()
inline MobileNetOutputs RegNet(nn::ParameterStore &p, nn::ParameterStore &state,
                               const nn::Tensor &image, const RegNetConfig &cfg = {},
                               bool training = false) {
  if (image.dim() != 4 || image.numel() == 0 || !(cfg.momentum >= 0 && cfg.momentum < 1) ||
      cfg.classes < 2)
    throw std::invalid_argument("invalid RegNet configuration");
  auto model = RegNetModel(cfg.model);
  auto design = RegNetDesign(model);
  MobileNetOutputs out;
  MobileContext c{p, state, out.layers, training, cfg.momentum, cfg.update_stats};
  auto h = RegNetConvBN(c, "stem", image, 32, 3, cfg.small_stem ? 1 : 2, 1, true);
  for (std::size_t stage = 0; stage < design.widths.size(); ++stage)
    for (int block = 0; block < design.depths[stage]; ++block)
      h = RegNetBlock(
          c, "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1), h,
          design.widths[stage], block ? 1 : 2, design.groups[stage], model.se);
  out.features = h;
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  out.classifier_macs = h.size(3) * cfg.classes;
  return out;
}
} // namespace datamunge::nn::models
