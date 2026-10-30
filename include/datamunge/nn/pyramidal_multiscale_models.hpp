#pragma once
/// @file pyramidal_multiscale_models.hpp
/// @brief Additive PyramidNet and hierarchical Res2Net tensor models, without fitting.
#include "datamunge/nn/preactivation_densenet_models.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Average-pool a shortcut when strided and append zero channels without parameters.
///
/// No nonlinearities or fitted parameters. Each pooled value averages four input
/// positions and its derivative spreads equally (1/4); appended channels have
/// zero values and no input dependence. Work/storage O(NHW*out).
///
/// @param input Nonempty floating NHWC tensor; stride-two spatial sides must be even.
/// @param out Output channel count, at least the input width.
/// @param stride One for identity spatial geometry or two for VALID 2x2 average pooling.
/// @return NHWC pooled input followed by out-Cin zero channels, retaining input gradients.
/// @throws std::invalid_argument For invalid rank, stride, odd strided sides or decreasing width.
/// @par Example
/// @code
/// const auto y = dm::nn::models::PyramidShortcut(dm::nn::ones({1, 4, 4, 3}), 5, 2);
/// @endcode
/// @par References
/// - Han et al. (2017), https://arxiv.org/abs/1610.02915; https://github.com/jhkim89/PyramidNet/blob/master/addpyramidnet.lua
/// @see PyramidBlock()
inline nn::Tensor PyramidShortcut(const nn::Tensor &input, std::int64_t out, int stride) {
  if (input.dim() != 4 || input.numel() == 0 || (stride != 1 && stride != 2) ||
      (stride == 2 && (input.size(1) % 2 || input.size(2) % 2)))
    throw std::invalid_argument(
        "pyramidal shortcut requires nonempty NHWC, stride 1/2 and even strided sides");
  const std::int64_t in = input.shape().back();
  if (out < in)
    throw std::invalid_argument("pyramidal shortcuts cannot remove channels");
  nn::Tensor h = stride == 2 ? nn::avg_pool2d(input, 2, 2, nn::Padding::Valid) : input;
  if (out == in)
    return h;
  return nn::pad(h, {{0, 0}, {0, 0}, {0, 0}, {0, out - in}});
}

/// @brief Apply a pyramidal basic or bottleneck correction with a parameter-free shortcut.
///
/// The first BN has no following ReLU. Basic order is BN/conv/BN/ReLU/conv/BN;
/// bottleneck order BN/1x1/BN/ReLU/3x3/BN/ReLU/1x1/BN. No activation follows
/// addition. Output width cannot decrease. Convolutions have no bias and use
/// He-uniform fan-in. No loss or fitting; work sums convolution MACs.
///
/// @param p Mutable floating learned-parameter store, lazily creates/reuses leaves.
/// @param state Mutable BN moving moments, controlled by norm.
/// @param name Prefix for bn1..bn3/4 and conv1..conv2/3.
/// @param input Nonempty floating NHWC; even spatial sides when stride two.
/// @param planes Positive inner/basic width; actual output is planes or 4*planes.
/// @param bottleneck Select three filters 1x1/3x3/1x1 instead of two 3x3 filters.
/// @param stride One or two, spatial reduction in the basic first or bottleneck second filter.
/// @param norm Preactivation BN settings; default inference, epsilon 1e-5.
/// @return Branch, pooled/zero-padded shortcut and signed sum, retaining gradients.
/// @throws std::invalid_argument For invalid NHWC, stride, widths, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::PyramidBlock(p, state, "b", n::ones({1, 4, 4, 3}), 4, false);
/// @endcode
/// @par References
/// - Han et al. (2017), https://arxiv.org/abs/1610.02915; https://github.com/jhkim89/PyramidNet/blob/master/addpyramidnet.lua
/// @see PyramidShortcut(), PyramidNet()
inline ResNetBlockOutputs PyramidBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                       const std::string &name, const nn::Tensor &input,
                                       std::int64_t planes, bool bottleneck, int stride = 1,
                                       const PreactivationNormConfig &norm = {}) {
  if (input.dim() != 4 || input.numel() == 0 ||
      (stride == 2 && (input.size(1) % 2 || input.size(2) % 2)))
    throw std::invalid_argument(
        "pyramidal block requires nonempty NHWC and even strided sides");
  const std::int64_t in = input.shape().back(), out = bottleneck ? 4 * planes : planes;
  if (in < 1 || planes < 1 || out < in || (stride != 1 && stride != 2))
    throw std::invalid_argument("invalid pyramidal unit");
  auto bn = [&](const std::string &part, const nn::Tensor &x) {
    return PreactivationNormalize(p, state, name + "/" + part, x, norm);
  };
  nn::Tensor h = bn("bn1", input);
  if (bottleneck) {
    h = RawResidualConv(p, name + "/conv1", h, planes, 1);
    h = RawResidualConv(p, name + "/conv2", nn::relu(bn("bn2", h)), planes, 3, stride);
    h = RawResidualConv(p, name + "/conv3", nn::relu(bn("bn3", h)), out, 1);
    h = bn("bn4", h);
  } else {
    h = RawResidualConv(p, name + "/conv1", h, planes, 3, stride);
    h = RawResidualConv(p, name + "/conv2", nn::relu(bn("bn2", h)), planes, 3);
    h = bn("bn3", h);
  }
  ResNetBlockOutputs result;
  result.residual = h;
  result.shortcut = PyramidShortcut(input, out, stride);
  result.output = result.residual + result.shortcut;
  result.preactivation = result.output; // no post-addition nonlinearity
  return result;
}

/// @brief Additive width schedule and layout metadata for PyramidNet.
///
/// Default CIFAR basic depth 110, alpha 48; no model state or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::PyramidNetConfig cfg;
/// const auto widths = cfg.UnitWidths();
/// @endcode
/// @par References
/// - Han et al. (2017), https://arxiv.org/abs/1610.02915; https://github.com/jhkim89/PyramidNet/blob/master/addpyramidnet.lua
/// @see PyramidNet()
struct PyramidNetConfig {
  int depth =
      110; ///< CIFAR 6n+2 basic or 9n+2 bottleneck; ImageNet 50/101/152/200 bottleneck.
  int classes = 10;        ///< At least two classifier outputs.
  double alpha = 48;       ///< Finite nonnegative total additive inner-width increase.
  bool bottleneck = false; ///< Fourfold output expansion with three-filter units.
  bool imagenet =
      false; ///< Four-stage 64-channel large-image stem instead of three-stage CIFAR.
  double momentum = .9;     ///< BN moving-moment retention in [0,1).
  bool update_stats = true; ///< Permit moving moments to change in training mode.

  /// @brief Return unit counts for the configured dataset and depth.
  ///
  /// Checks layout/depth only, without validating alpha, classes or momentum.
  /// No allocation beyond the small returned vector, no learned state changes.
  ///
  /// @return Three equal CIFAR counts or four ImageNet counts; depth 200 uses 3/24/36/3.
  /// @throws std::invalid_argument For unsupported depth or ImageNet basic units.
  /// @par Example
  /// @code
  /// const auto counts = dm::nn::models::PyramidNetConfig{}.StageCounts(); // {18,18,18}
  /// @endcode
  /// @see UnitWidths()
  std::vector<int> StageCounts() const {
    if (imagenet) {
      if (!bottleneck)
        throw std::invalid_argument("the released ImageNet PyramidNet uses bottlenecks");
      if (depth == 200)
        return {3, 24, 36, 3};
      if (depth < 50)
        throw std::invalid_argument("ImageNet PyramidNet depths are 50/101/152/200");
      const auto c = ResNetBlockCounts(depth);
      return {c[0], c[1], c[2], c[3]};
    }
    const int per = bottleneck ? 9 : 6;
    if (depth < per + 2 || (depth - 2) % per)
      throw std::invalid_argument(
          "CIFAR PyramidNet depth must be 6n+2 (basic) or 9n+2 (bottleneck)");
    return std::vector<int>(3, (depth - 2) / per);
  }
  /// @brief Validate the configuration and compute released rounded additive unit widths.
  ///
  /// Start running width at 16 (CIFAR) or 64 (ImageNet); increment alpha/N
  /// before every unit, then floor(running+.5). Repeated floating additions
  /// match the release. Bottleneck output is four times these values.
  /// Finite alpha must keep every expanded width representable by int. O(N) work/storage.
  ///
  /// @return Positive inner/basic width for every unit in execution order.
  /// @throws std::invalid_argument For invalid layout, alpha, classes, momentum or width overflow.
  /// @par Example
  /// @code
  /// const auto widths = dm::nn::models::PyramidNetConfig{}.UnitWidths(); // 54, ending at 64
  /// @endcode
  /// @par References
  /// - Han et al. (2017), https://arxiv.org/abs/1610.02915; https://github.com/jhkim89/PyramidNet/blob/master/addpyramidnet.lua
  /// @see StageCounts()
  std::vector<int> UnitWidths() const {
    const auto counts = StageCounts();
    int units = 0;
    for (int c : counts)
      units += c;
    if (!std::isfinite(alpha) || alpha < 0 ||
        alpha > double(std::numeric_limits<int>::max() / (bottleneck ? 4 : 1) - 64) ||
        classes < 2 || !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid PyramidNet configuration");
    std::vector<int> widths;
    double running = imagenet ? 64 : 16;
    for (int i = 0; i < units; ++i) {
      running += alpha / units;
      widths.push_back(int(std::floor(running + .5)));
    }
    return widths;
  }
};
/// @brief Pyramidal stem, unit paths, width schedule, stage ends and logits.
///
/// Tensors retain storage/graphs; stores own learned leaves and moving moments.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PyramidNetConfig cfg;
/// cfg.depth = 8;
/// const auto y = n::models::PyramidNet(p, state, n::ones({1, 8, 8, 3}), cfg);
/// @endcode
/// @see PyramidNet()
struct PyramidNetOutputs {
  nn::Tensor stem;     ///< Stem after BN, with ReLU/pool only for ImageNet.
  nn::Tensor features; ///< Final BN/ReLU feature map.
  nn::Tensor logits;   ///< Unnormalised [N,classes] global-average classifier scores.
  std::vector<nn::Tensor> stages;         ///< Three CIFAR or four ImageNet signed stage sums.
  std::vector<ResNetBlockOutputs> blocks; ///< Every unit with its branch and shortcut.
  std::vector<std::int64_t> channels;     ///< Expanded output channel count for every unit.
};
/// @brief Build a complete additive PyramidNet tensor classifier without fitting.
///
/// CIFAR stem is bias-free 3x3/BN without ReLU; ImageNet uses 7x7/2,
/// BN/ReLU and padded max-pool. Each later stage downsamples once, using
/// parameter-free average-pool/zero-pad shortcuts. Final BN/ReLU, global
/// mean and biased Glorot dense classifier. All BN learns scale/shift, eps 1e-5.
/// He-uniform fan-in differs from released normal fan-out initialization.
/// No preprocessing, optimizer, loss or fitting loop is hidden. Work is the
/// sum of per-unit convolution MACs; storage includes graph activations.
///
/// @param p Mutable learned-parameter store, reused by stable stem/stage/output names.
/// @param state Mutable BN moving moments, with dtype matching p.
/// @param image Nonempty floating NHWC tensor; stage-downsampled sides must be even.
/// @param cfg Validated layout and width settings; defaults CIFAR-110 alpha 48 basic.
/// @param training True uses batch moments; updates state only if cfg.update_stats.
/// @return Stem, every unit/width, stages, final features and logits with gradients.
/// @throws std::invalid_argument For invalid configuration, geometry, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PyramidNetConfig cfg;
/// cfg.depth = 8;
/// const auto y = n::models::PyramidNet(p, state, n::ones({1, 8, 8, 3}), cfg);
/// @endcode
/// @par References
/// - Han et al. (2017), https://arxiv.org/abs/1610.02915; https://github.com/jhkim89/PyramidNet/blob/master/addpyramidnet.lua
/// @see PyramidNetConfig, PyramidBlock()
inline PyramidNetOutputs PyramidNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const nn::Tensor &image, const PyramidNetConfig &cfg = {},
                                    bool training = false) {
  if (image.dim() != 4 || image.numel() == 0)
    throw std::invalid_argument("PyramidNet image must be nonempty NHWC");
  const auto counts = cfg.StageCounts();
  const auto widths = cfg.UnitWidths();
  PyramidNetOutputs out;
  const PreactivationNormConfig norm{training, cfg.update_stats, cfg.momentum, 1e-5};
  ResNetNormConfig stem_norm;
  stem_norm.training = training;
  stem_norm.scale = true;
  stem_norm.epsilon = 1e-5;
  stem_norm.momentum = cfg.momentum;
  stem_norm.update_stats = cfg.update_stats;
  std::int64_t in = cfg.imagenet ? 64 : 16;
  nn::Tensor h = ResNetConv(p, state, "stem", image, in, cfg.imagenet ? 7 : 3, stem_norm,
                            cfg.imagenet ? 2 : 1, cfg.imagenet);
  if (cfg.imagenet)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  std::size_t unit = 0;
  for (std::size_t stage = 0; stage < counts.size(); ++stage) {
    for (int block = 0; block < counts[stage]; ++block, ++unit) {
      const auto name =
          "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      out.blocks.push_back(PyramidBlock(p, state, name, h, widths[unit], cfg.bottleneck,
                                        stage > 0 && block == 0 ? 2 : 1, norm));
      h = out.blocks.back().output;
      in = std::int64_t(widths[unit]) * (cfg.bottleneck ? 4 : 1);
      out.channels.push_back(in);
    }
    out.stages.push_back(h);
  }
  out.features = nn::relu(PreactivationNormalize(p, state, "final/bn", h, norm));
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(out.features), cfg.classes);
  return out;
}
/// @brief Ordered multi-scale split activations and their channel concatenation.
///
/// Tensor handles retain storage and autograd history; no fitted state is copied.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig norm;
/// norm.scale = true;
/// const auto y = n::models::Res2ScaleTransform(p, state, "s", n::ones({1, 5, 5, 4}), 1, 4, norm);
/// @endcode
/// @see Res2ScaleTransform()
struct Res2ScaleOutputs {
  nn::Tensor mixed; ///< NHWC concatenation of every group, preserving input channel order.
  std::vector<nn::Tensor>
      groups; ///< Processed groups followed by the final bypass (if scale>1).
};
/// @brief Compute hierarchical or stage-entry multi-scale 3x3 transformations.
///
/// Ordinary groups i<scale-1 compute y0=K0(x0), yi=Ki(xi+y(i-1));
/// last group bypasses. Stage-entry groups process independently, while the
/// last uses padded 3x3 average pooling including zero padding in the divisor.
/// Scale one uses one filtered group without bypass. Explicit symmetric padding
/// aligns stride-two sampling. Work O(NHW*9*width²*max(1,scale-1)); no fitting.
///
/// @param p Mutable learned store for group0..group(scale-2) filters and BN.
/// @param state Mutable BN moving moments controlled by cfg.
/// @param name Group-layer prefix.
/// @param input Nonempty floating NHWC tensor with exactly width*scale channels.
/// @param width Positive channels per split.
/// @param scale Positive split count; one retains one 3x3 filter.
/// @param cfg Post-filter BN/ReLU settings, normally learned scale and eps 1e-5.
/// @param stage True for the first unit of every stage, including stride-one entries.
/// @param stride One or two; two is valid only when stage=true.
/// @return Ordered per-split outputs and their NHWC concatenation, retaining gradients.
/// @throws std::invalid_argument For invalid NHWC, width/scale, stride, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig norm;
/// norm.scale = true;
/// const auto y = n::models::Res2ScaleTransform(p, state, "s", n::ones({1, 5, 5, 4}), 1, 4, norm);
/// @endcode
/// @par References
/// - Gao et al. (2021), https://arxiv.org/abs/1904.01169; https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
/// @see Res2NetBlock()
inline Res2ScaleOutputs Res2ScaleTransform(nn::ParameterStore &p, nn::ParameterStore &state,
                                           const std::string &name, const nn::Tensor &input,
                                           std::int64_t width, int scale,
                                           const ResNetNormConfig &cfg, bool stage = false,
                                           int stride = 1) {
  if (input.dim() != 4 || input.numel() == 0 || width < 1 || scale < 1 ||
      (stride != 1 && stride != 2) || (!stage && stride != 1) ||
      input.shape().back() != width * scale)
    throw std::invalid_argument("invalid Res2Net scale transform");
  const std::vector<nn::Tensor> split =
      scale == 1 ? std::vector<nn::Tensor>{input} : nn::split(input, scale, 3);
  Res2ScaleOutputs out;
  const int processed = scale == 1 ? 1 : scale - 1;
  for (int group = 0; group < processed; ++group) {
    nn::Tensor h = split[static_cast<std::size_t>(group)];
    if (group > 0 && !stage)
      h = h + out.groups.back();
    h = ResNetConv(p, state, name + "/group" + std::to_string(group), h, width, 3, cfg,
                   stride);
    out.groups.push_back(h);
  }
  if (scale > 1) {
    nn::Tensor h = split[static_cast<std::size_t>(scale - 1)];
    if (stage)
      h = nn::avg_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), 3, stride,
                         nn::Padding::Valid);
    out.groups.push_back(h);
  }
  out.mixed = out.groups.size() == 1 ? out.groups[0] : nn::concat(out.groups, 3);
  return out;
}

/// @brief Apply a hierarchical Res2Net bottleneck with expansion, shortcut and ReLU.
///
/// Reduction expands to scale*floor(planes*base_width/64); multi-scale processing
/// then concatenation precedes linear 1x1 expansion and BN. Identity applies only
/// at equal width/stride. Bias-free He-uniform filters; no optimizer or fitting.
/// The original release is used, without the later v1b stem/projection changes.
///
/// @param p Mutable learned parameter store, lazily creates/reuses leaves.
/// @param state Mutable BN moments controlled by cfg.
/// @param name Prefix for conv1, scales/groupN, conv3 and optional shortcut.
/// @param input Nonempty floating NHWC tensor matching store dtype.
/// @param planes Positive base width; output width is 4*planes.
/// @param base_width Positive coefficient; floor(planes*base_width/64) must be positive.
/// @param scale Positive number of internal channel splits.
/// @param cfg Post-filter BN/ReLU configuration.
/// @param stage True at every stage entry, using independent split transforms and pooled bypass.
/// @param stride One or two, with stride two requiring stage=true.
/// @return Branch, raw/projected shortcut, preactivation sum and post-addition ReLU.
/// @throws std::invalid_argument For invalid dimensions, stage/stride, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig norm;
/// norm.scale = true;
/// const auto y = n::models::Res2NetBlock(p, state, "b", n::ones({1, 5, 5, 8}), 4, 16, 4, norm);
/// @endcode
/// @par References
/// - Gao et al. (2021), https://arxiv.org/abs/1904.01169; https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
/// @see Res2ScaleTransform(), Res2Net()
inline ResNetBlockOutputs Res2NetBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                       const std::string &name, const nn::Tensor &input,
                                       std::int64_t planes, std::int64_t base_width, int scale,
                                       const ResNetNormConfig &cfg, bool stage = false,
                                       int stride = 1) {
  if (input.dim() != 4 || input.numel() == 0)
    throw std::invalid_argument("Res2Net block requires nonempty NHWC");
  const std::int64_t in = input.shape().back(), width = planes * base_width / 64,
                     middle = width * scale, out = 4 * planes;
  if (in < 1 || planes < 1 || base_width < 1 || width < 1 || scale < 1)
    throw std::invalid_argument("invalid Res2Net bottleneck");
  auto h = ResNetConv(p, state, name + "/conv1", input, middle, 1, cfg);
  h = Res2ScaleTransform(p, state, name + "/scales", h, width, scale, cfg, stage, stride)
          .mixed;
  ResNetBlockOutputs result;
  result.residual = ResNetConv(p, state, name + "/conv3", h, out, 1, cfg, 1, false);
  result.shortcut = in == out && stride == 1 ? input
                                             : ResNetConv(p, state, name + "/shortcut", input,
                                                          out, 1, cfg, stride, false);
  result.preactivation = result.shortcut + result.residual;
  result.output = nn::relu(result.preactivation);
  return result;
}

/// @brief Original Res2Net topology, internal scale and optional compact stem settings.
///
/// Defaults describe full ImageNet Res2Net-50 26w/4s. Metadata only, no fitting.
///
/// @par Example
/// @code
/// dm::nn::models::Res2NetConfig cfg;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Gao et al. (2021), https://arxiv.org/abs/1904.01169; https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
/// @see Res2Net()
struct Res2NetConfig {
  int depth = 50;           ///< ImageNet bottleneck depth 50/101/152.
  int base_width = 26;      ///< Positive split width coefficient at stage base 64.
  int scale = 4;            ///< Positive internal split count.
  int stage_width = 64;     ///< Positive first stage base/stem width.
  int classes = 1000;       ///< At least two classifier outputs.
  bool small_stem = false;  ///< Use 3x3/1 without initial pool for compact images.
  double momentum = .9;     ///< BN moving-moment retention in [0,1).
  bool update_stats = true; ///< Permit eager moving-moment updates in training mode.

  /// @brief Validate Res2Net depth, dimensions, split width and BN retention.
  ///
  /// Checks metadata without allocating or changing learned state.
  ///
  /// @throws std::invalid_argument For unsupported depth, nonpositive/floored-zero widths, classes or momentum.
  /// @par Example
  /// @code
  /// dm::nn::models::Res2NetConfig{}.Validate();
  /// @endcode
  /// @see Res2NetConfig
  void Validate() const {
    if ((depth != 50 && depth != 101 && depth != 152) || base_width < 1 || scale < 1 ||
        stage_width < 1 || base_width * stage_width < 64 || classes < 2 ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid Res2Net configuration");
  }
};
/// @brief Build the original complete Res2Net tensor classifier without fitting.
///
/// Depth 50 retains 3/4/6/3 blocks, including stage-entry semantics in the
/// first block of all four stages. Standard 7x7/2+pool stem or compact 3x3/1;
/// outer stage widths 4*stage_width*2^s. Final global mean and Glorot dense
/// classifier. Full 26w/4s has 25699120 learned and 59492 moving scalars.
/// BN learns gamma/beta with eps 1e-5; update_stats=false freezes moving values
/// while training still uses batch moments. No preprocessing, loss, optimizer
/// or fitting is hidden. Work/storage sums constituent tensor convolutions.
///
/// @param p Mutable learned floating store, stable stem/stage/output names are reused.
/// @param state Mutable moving-moment store matching p dtype.
/// @param image Nonempty floating NHWC images with positive channels.
/// @param cfg Validated architecture; defaults full ImageNet 50 26w/4s.
/// @param training Use batch BN moments and optionally update state.
/// @return ResNet-style stem, four stages, every block and classifier logits with gradients.
/// @throws std::invalid_argument For invalid configuration, NHWC, normalization, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::Res2NetConfig cfg;
/// cfg.stage_width = 16;
/// cfg.small_stem = true;
/// cfg.classes = 10;
/// const auto y = n::models::Res2Net(p, state, n::ones({1, 16, 16, 3}), cfg);
/// @endcode
/// @par References
/// - Gao et al. (2021), https://arxiv.org/abs/1904.01169; https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py
/// @see Res2NetConfig, Res2NetBlock()
inline ResNetOutputs Res2Net(nn::ParameterStore &p, nn::ParameterStore &state,
                             const nn::Tensor &image, const Res2NetConfig &cfg = {},
                             bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.numel() == 0)
    throw std::invalid_argument("Res2Net image must be nonempty NHWC");
  ResNetOutputs out;
  const auto counts = ResNetBlockCounts(cfg.depth);
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.epsilon = 1e-5;
  norm.momentum = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  auto h = ResNetConv(p, state, "stem", image, cfg.stage_width, cfg.small_stem ? 3 : 7, norm,
                      cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  for (int stage = 0; stage < 4; ++stage) {
    const std::int64_t planes = std::int64_t(cfg.stage_width) << stage;
    for (int block = 0; block < counts[static_cast<std::size_t>(stage)]; ++block) {
      h = Res2NetBlock(p, state,
                       "stage" + std::to_string(stage + 1) + "/block" +
                           std::to_string(block + 1),
                       h, planes, cfg.base_width, cfg.scale, norm, block == 0,
                       stage > 0 && block == 0 ? 2 : 1)
              .output;
      out.blocks.push_back(h);
    }
    out.stages[static_cast<std::size_t>(stage)] = h;
  }
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}
} // namespace datamunge::nn::models
