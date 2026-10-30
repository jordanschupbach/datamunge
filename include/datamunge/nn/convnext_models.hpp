#pragma once
/// @file convnext_models.hpp
/// @brief Complete tensor ConvNeXt V1/V2 models, GRN and explicit arithmetic metadata.
#include "datamunge/nn/mbconv_models.hpp"
#include <limits>
#include <map>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Stage depths and channel dimensions for a ConvNeXt layout.
///
/// Value metadata; no learned tensors or fitting.
///
/// @par Example
/// @code
/// const auto layout = dm::nn::models::ConvNeXtLayout("atto");
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm()
struct ConvNeXtSize {
  std::vector<int> depths; ///< Four positive stage repeat counts.
  std::vector<int> dims;   ///< Four positive corresponding stage channel widths.
};
/// @brief Return four-stage architecture metadata for a named ConvNeXt size.
///
/// Immutable local table is initialized once and safe for concurrent reads.
/// Tiny/Small differ in stage-three depth, not width; V2 releases omit Small.
/// Lookup accepts the shared table for either version; no training is hidden.
///
/// @param name One of atto,femto,pico,nano,tiny,small,base,large,huge.
/// @return Owned depths/dimensions, no parameters allocated.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/probabilities/geometry, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// const auto tiny = dm::nn::models::ConvNeXtLayout("tiny");
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm(), ConvNeXtParameters()
inline ConvNeXtSize ConvNeXtLayout(const std::string &name) {
  /// @brief Immutable released dimensions, copied to each caller.
  static const std::map<std::string, ConvNeXtSize> sizes = {
      {"atto", {{2, 2, 6, 2}, {40, 80, 160, 320}}},
      {"femto", {{2, 2, 6, 2}, {48, 96, 192, 384}}},
      {"pico", {{2, 2, 6, 2}, {64, 128, 256, 512}}},
      {"nano", {{2, 2, 8, 2}, {80, 160, 320, 640}}},
      {"tiny", {{3, 3, 9, 3}, {96, 192, 384, 768}}},
      {"small", {{3, 3, 27, 3}, {96, 192, 384, 768}}},
      {"base", {{3, 3, 27, 3}, {128, 256, 512, 1024}}},
      {"large", {{3, 3, 27, 3}, {192, 384, 768, 1536}}},
      {"huge", {{3, 3, 27, 3}, {352, 704, 1408, 2816}}}};
  if (!sizes.count(name))
    throw std::invalid_argument("unknown ConvNeXt size " + name);
  return sizes.at(name);
}

/// @brief Apply a biased dense or multiplier-one depthwise ConvNeXt filter.
///
/// Weights are untruncated N(0,.02), biases zero. Stride one pads
/// symmetrically by floor(k/2); odd k preserves size, even k enlarges it.
/// Strided filters use VALID padding. No norm/activation/fitting hidden.
///
/// @param p Mutable floating learned store, lazy name/W and name/b.
/// @param name Learned kernel/bias prefix.
/// @param x Nonempty floating NHWC input matching p dtype.
/// @param out Positive dense output width; must equal Cin for depthwise.
/// @param kernel Positive square side.
/// @param stride Positive spatial step.
/// @param depthwise False for dense, true multiplier-one depthwise.
/// @return Differentiable biased NHWC map.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/probabilities/geometry, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::ConvNeXtConv(p, "c", n::ones({1, 5, 5, 8}), 8, 7, 1, true);
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm(), ConvNeXtParameters()
inline nn::Tensor ConvNeXtConv(nn::ParameterStore &p, const std::string &name,
                               const nn::Tensor &x, std::int64_t out, std::int64_t kernel,
                               std::int64_t stride, bool depthwise = false) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || kernel < 1 || stride < 1 ||
      (depthwise && out != x.size(3)))
    throw std::invalid_argument("invalid ConvNeXt convolution dimensions");
  const std::int64_t in = x.shape()[3];
  const nn::Tensor w =
      p.get(name + "/W",
            depthwise ? nn::Shape{kernel, kernel, in, 1} : nn::Shape{kernel, kernel, in, out},
            nn::init::normal(.02));
  const nn::Tensor b = p.get(name + "/b", {depthwise ? in : out}, nn::init::zeros());
  nn::Conv2DOptions o = nn::conv_options(stride, nn::Padding::Valid);
  if (stride == 1 && kernel > 1) {
    o.padding = nn::Padding::Explicit;
    o.explicit_paddings = {kernel / 2, kernel / 2, kernel / 2, kernel / 2};
  }
  return nn::bias_add(depthwise ? nn::depthwise_conv2d(x, w, o) : nn::conv2d(x, w, o), b);
}

/// @brief Global response normalization map and differentiated output.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Learned leaves remain owned by the parameter store; no fitting is performed.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::GlobalResponseNorm(p, "g", n::ones({2, 4, 4, 8}));
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm()
struct GrnOutputs {
  nn::Tensor response; ///< [N,1,1,C] spatial L2 responses divided by channel mean plus 1e-6.
  nn::Tensor output;   ///< gamma*x*response+beta+x, preserving full response derivatives.
};
/// @brief Apply V2 global response normalization with a safe zero-norm derivative.
///
/// Each channel norm is sqrt(sum_H,W(x²)); response divides by its
/// channel mean plus 1e-6. A safe where argument defines zero derivative
/// of the spatial norm at zero without evaluating sqrt(0) in backward.
/// Initial gamma=beta=0 makes output exactly x and dx exactly identity,
/// including all-zero/dead channels. At learned nonzero gamma, response
/// derivatives remain active. Approximate scale invariance is limited by
/// the fixed epsilon; no moving moments or fitting. O(numel) tensor work.
///
/// @param p Mutable floating store creating zero-initialized name/gamma and name/beta [C].
/// @param name GRN affine prefix.
/// @param x Nonempty floating NHWC input matching p dtype.
/// @return Normalized spatial response and gamma*x*response+beta+x.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/probabilities/geometry, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::GlobalResponseNorm(p, "grn", n::ones({2, 4, 4, 8}));
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm(), ConvNeXtParameters()
inline GrnOutputs GlobalResponseNorm(nn::ParameterStore &p, const std::string &name,
                                     const nn::Tensor &x) {
  if (x.dim() != 4 || x.numel() == 0)
    throw std::invalid_argument("GRN needs nonempty NHWC input");
  const std::int64_t channels = x.shape()[3];
  GrnOutputs r;
  const nn::Tensor squared = nn::sum(nn::square(x), {1, 2}, true);
  const nn::Tensor nonzero = nn::greater(squared, 0.0);
  // Define the L2 norm's subgradient at the origin as zero. Evaluating
  // sqrt(0) directly would give an infinite derivative and 0*inf NaNs.
  // Selecting a safe argument retains the exact forward norm, including zero.
  const nn::Tensor norm = nn::where(nonzero, nn::sqrt(nn::where(nonzero, squared, 1.0)), 0.0);
  r.response = norm / (nn::mean(norm, {3}, true) + 1e-6);
  const nn::Tensor gamma = p.get(name + "/gamma", {channels}, nn::init::zeros()),
                   beta = p.get(name + "/beta", {channels}, nn::init::zeros());
  r.output = gamma * (x * r.response) + beta + x;
  return r;
}

/// @brief Residual branch before dropout and complete ConvNeXt block output.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Learned leaves remain owned by the parameter store; no fitting is performed.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::ConvNeXtBlock(p, "b", n::ones({2, 5, 5, 8}), false, false);
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm()
struct ConvNeXtBlockOutputs {
  nn::Tensor branch; ///< Projected branch with V1 layer scale or V2 GRN, before row dropout.
  nn::Tensor output; ///< Input plus optionally dropped/rescaled branch, full graph retained.
};
/// @brief Apply a depthwise/LayerNorm inverted bottleneck and optional row branch dropout.
///
/// Biased depthwise 7x7 -> channels-last LayerNorm eps 1e-6 -> biased
/// 1x1 to 4C -> exact erf GELU -> optional V2 GRN -> biased 1x1 to C.
/// V1 multiplies final branch by a learned C-vector. Matching shortcut is
/// raw signed x. Per-row inverted branch masks use p.generator() in training;
/// all filters still execute. Shared store/RNG access needs synchronization.
///
/// @param p Mutable floating learned leaves and random generator.
/// @param name Prefix for dwconv/norm/pwconv1/optional grn/pwconv2/layer_scale.
/// @param x Nonempty floating NHWC input, channel width expandable by four.
/// @param v2 True uses GRN on expanded features; false final layer scale initialized 1e-6.
/// @param training True permits stochastic branch dropout.
/// @param drop Finite drop probability in [0,1), default zero.
/// @return Pre-dropout branch and raw-input-plus-branch output, full graph retained.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/probabilities/geometry, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::ConvNeXtBlock(p, "b", n::ones({2, 5, 5, 8}), false, false);
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm(), ConvNeXtParameters()
inline ConvNeXtBlockOutputs ConvNeXtBlock(nn::ParameterStore &p, const std::string &name,
                                          const nn::Tensor &x, bool v2, bool training,
                                          double drop = 0) {
  if (x.dim() != 4 || x.numel() == 0 || x.size(3) > std::numeric_limits<int64_t>::max() / 4 ||
      !(drop >= 0 && drop < 1))
    throw std::invalid_argument("invalid ConvNeXt block dimensions/dropout");
  const std::int64_t dim = x.shape()[3];
  nn::Tensor h = nn::layer_norm(p, name + "/norm",
                                ConvNeXtConv(p, name + "/dwconv", x, dim, 7, 1, true), 1e-6);
  h = nn::gelu(ConvNeXtConv(p, name + "/pwconv1", h, 4 * dim, 1, 1));
  if (v2)
    h = GlobalResponseNorm(p, name + "/grn", h).output;
  h = ConvNeXtConv(p, name + "/pwconv2", h, dim, 1, 1);
  if (!v2)
    h = h * p.get(name + "/layer_scale", {dim}, nn::init::constant(1e-6));
  ConvNeXtBlockOutputs r;
  r.branch = h;
  if (training && drop > 0)
    h = DropConnect(h, 1 - drop, p.generator());
  r.output = x + h;
  return r;
}

/// @brief ConvNeXt V1/V2 layout, stem, classifier and branch-dropout metadata.
///
/// Value metadata; no learned tensors or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::ConvNeXtConfig cfg;
/// cfg.size = "atto";
/// cfg.classes = 10;
/// cfg.cifar_stem = true;
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm()
struct ConvNeXtConfig {
  std::string size = "tiny"; ///< atto,femto,pico,nano,tiny,small,base,large,huge.
  bool v2 = false;         ///< False learns tiny layer scales; true uses expanded-channel GRN.
  bool cifar_stem = false; ///< True uses 2x2/2 patchify, false 4x4/4.
  int classes = 1000;      ///< At least two classifier outputs.
  double drop_path =
      0; ///< Maximum row branch-drop probability in [0,1), linearly reached at last block.
};
/// @brief All stage maps and classifier logits from a complete ConvNeXt.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Learned leaves remain owned by the parameter store; no fitting is performed.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::ConvNeXtConfig cfg;
/// cfg.size = "atto";
/// cfg.classes = 10;
/// cfg.cifar_stem = true;
/// const auto y = n::models::ConvNeXt(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm()
struct ConvNeXtOutputs {
  nn::Tensor features;            ///< Last stage NHWC map before global pooling.
  nn::Tensor logits;              ///< [N,classes] scores after pooled LayerNorm/classifier.
  std::vector<nn::Tensor> stages; ///< Four post-stage maps, retaining every block graph.
};
/// @brief Build a complete four-stage V1 or V2 ConvNeXt classifier without fitting.
///
/// Biased 4x4/4 patchify plus LayerNorm (2x2/2 for CIFAR), four full
/// stages joined by LayerNorm and VALID biased 2x2/2 filters, global
/// mean/LayerNorm/classifier. Each block uses drop_path*i/(n-1), so last
/// block reaches the coefficient. No BN/moving state or classifier dropout.
/// All filters and classifier use untruncated N(0,.02), an adaptation of
/// released truncated initialization. No resizing, FCMAE pretraining, loss,
/// optimizer or fitting is hidden. Stores must outlive retained graphs.
///
/// @param p Mutable floating learned store and stochastic-depth generator.
/// @param image Nonempty floating RGB NHWC input matching p; each side at least patch*8.
/// @param cfg Valid named size/classes/stem/drop-path metadata, default V1 Tiny.
/// @param training True uses row branch masks; inference is deterministic.
/// @return Four post-stage maps, last features and classifier logits.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/probabilities/geometry, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::ConvNeXtConfig cfg;
/// cfg.size = "atto";
/// cfg.cifar_stem = true;
/// cfg.classes = 10;
/// const auto y = n::models::ConvNeXt(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm(), ConvNeXtParameters()
inline ConvNeXtOutputs ConvNeXt(nn::ParameterStore &p, const nn::Tensor &image,
                                const ConvNeXtConfig &cfg = {}, bool training = false) {
  const auto layout = ConvNeXtLayout(cfg.size);
  if (image.dim() != 4 || image.numel() == 0 || image.size(3) != 3 ||
      image.size(1) < (cfg.cifar_stem ? 16 : 32) ||
      image.size(2) < (cfg.cifar_stem ? 16 : 32) || cfg.classes < 2 ||
      !(cfg.drop_path >= 0 && cfg.drop_path < 1))
    throw std::invalid_argument("invalid ConvNeXt configuration");
  ConvNeXtOutputs out;
  const int patch = cfg.cifar_stem ? 2 : 4;
  nn::Tensor h = nn::layer_norm(
      p, "stem/norm", ConvNeXtConv(p, "stem", image, layout.dims[0], patch, patch), 1e-6);
  int total = 0, index = 0;
  for (int d : layout.depths)
    total += d;
  for (std::size_t stage = 0; stage < layout.depths.size(); ++stage) {
    if (stage) {
      const std::string name = "downsample" + std::to_string(stage);
      h = ConvNeXtConv(p, name, nn::layer_norm(p, name + "/norm", h, 1e-6), layout.dims[stage],
                       2, 2);
    }
    for (int block = 0; block < layout.depths[stage]; ++block, ++index)
      h = ConvNeXtBlock(
              p, "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1), h,
              cfg.v2, training, total > 1 ? cfg.drop_path * index / (total - 1) : 0)
              .output;
    out.stages.push_back(h);
  }
  out.features = h;
  const nn::Tensor pooled = nn::layer_norm(p, "head/norm", nn::global_avg_pool2d(h), 1e-6);
  p.get("output/W", {layout.dims.back(), cfg.classes}, nn::init::normal(.02));
  out.logits = nn::dense(p, "output", pooled, cfg.classes);
  return out;
}

/// @brief Count convolution and classifier products from actual ConvNeXt stage geometry.
///
/// Per-block spatial work is H*W*(49C+8C²); adds patchify, joins
/// and classifier. Handles rectangular maps using H*W, not H². Excludes
/// biases, norms, GELU, residual/dropout and GRN arithmetic/memory traffic.
/// Caller supplies actual layout/classifier/stem; counts must fit int64.
/// No execution or runtime measurement occurs in this metadata helper.
///
/// @param net Four nonempty NHWC stages with matching layout widths; batch ignored.
/// @param layout Four positive stage depths/dimensions.
/// @param classes At least two classifier outputs.
/// @param patch Stem side/stride two or four for RGB inputs.
/// @return 64-bit per-image convolution/classifier MACs, independent of V1/V2 normalization.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/probabilities/geometry, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::ConvNeXtConfig cfg;
/// cfg.size = "atto";
/// const auto y = n::models::ConvNeXt(p, n::ones({1, 32, 32, 3}), cfg);
/// const auto macs = n::models::ConvNeXtMacs(y, n::models::ConvNeXtLayout("atto"), 1000, 4);
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm(), ConvNeXtParameters()
inline std::int64_t ConvNeXtMacs(const ConvNeXtOutputs &net, const ConvNeXtSize &layout,
                                 int classes, int patch) {
  if (layout.dims.size() != 4 || layout.depths.size() != 4 || net.stages.size() != 4 ||
      classes < 2 || (patch != 2 && patch != 4))
    throw std::invalid_argument("invalid ConvNeXt budget metadata");
  for (size_t i = 0; i < 4; ++i)
    if (layout.dims[i] < 1 || layout.depths[i] < 1 || net.stages[i].dim() != 4 ||
        net.stages[i].numel() == 0 || net.stages[i].size(3) != layout.dims[i])
      throw std::invalid_argument("invalid ConvNeXt budget stage");
  std::int64_t macs = std::int64_t(layout.dims.back()) * classes;
  for (std::size_t stage = 0; stage < layout.depths.size(); ++stage) {
    const std::int64_t area = net.stages[stage].size(1) * net.stages[stage].size(2),
                       d = layout.dims[stage];
    macs += layout.depths[stage] * area * (49 * d + 8 * d * d);
    macs += stage ? area * 4 * layout.dims[stage - 1] * d : area * patch * patch * 3 * d;
  }
  return macs;
}
/// @brief Count all learned ConvNeXt scalars analytically without allocating weights.
///
/// Per-block learned count is 8C²+58C for V1, 8C²+65C for V2.
/// Stem counts patch²*3*C+3C, each join 2Cin+4Cin*Cout+Cout;
/// head counts 2C+C*classes+classes. Checked integer architecture
/// arithmetic avoids signed overflow; values beyond int64 are rejected.
/// O(number of stages) metadata work; no model forward or fitting.
///
/// @param layout Four positive integer stage depths/dimensions.
/// @param classes Classifier outputs at least two, default 1000.
/// @param patch RGB stem side/stride two or four, default four.
/// @param v2 True counts GRN, false V1 layer scales.
/// @return 64-bit learned count including filters/biases/LayerNorm/GRN or layer scale.
/// @throws std::invalid_argument For unknown preset, invalid dimensions/probabilities/geometry, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// const auto count =
///     dm::nn::models::ConvNeXtParameters(dm::nn::models::ConvNeXtLayout("tiny")); // 28589128
/// @endcode
/// @par References
/// - Liu et al. (2022), https://arxiv.org/abs/2201.03545; Woo et al. (2023), https://arxiv.org/abs/2301.00808; https://github.com/facebookresearch/ConvNeXt-V2
/// @see ConvNeXt(), ConvNeXtBlock(), GlobalResponseNorm(), ConvNeXtParameters()
inline std::int64_t ConvNeXtParameters(const ConvNeXtSize &layout, int classes = 1000,
                                       int patch = 4, bool v2 = false) {
  if (layout.dims.size() != 4 || layout.depths.size() != 4 || classes < 2 ||
      (patch != 2 && patch != 4))
    throw std::invalid_argument("invalid ConvNeXt parameter metadata");
  for (size_t i = 0; i < 4; ++i)
    if (layout.dims[i] < 1 || layout.depths[i] < 1)
      throw std::invalid_argument("positive ConvNeXt depths and widths required");
  std::int64_t total = 0;
  /// Checked multiplication of nonnegative architecture dimensions.
  auto mul = [](std::int64_t a, std::int64_t b) {
    if (b && a > std::numeric_limits<std::int64_t>::max() / b)
      throw std::invalid_argument("ConvNeXt parameter count overflow");
    return a * b;
  };
  /// Add a nonnegative term without overflowing the accumulated count.
  auto add = [&](std::int64_t term) {
    if (term > std::numeric_limits<std::int64_t>::max() - total)
      throw std::invalid_argument("ConvNeXt parameter count overflow");
    total += term;
  };
  add(mul(patch * patch * 3LL, layout.dims[0]));
  add(3LL * layout.dims[0]);
  for (size_t i = 0; i < 4; ++i) {
    const std::int64_t d = layout.dims[i];
    const auto square = mul(8, mul(d, d));
    if (square > std::numeric_limits<std::int64_t>::max() - (v2 ? 65 : 58) * d)
      throw std::invalid_argument("ConvNeXt parameter count overflow");
    add(mul(layout.depths[i], square + (v2 ? 65 : 58) * d));
    if (i) {
      add(2LL * layout.dims[i - 1]);
      add(mul(4, mul(layout.dims[i - 1], d)));
      add(d);
    }
  }
  add(2LL * layout.dims.back());
  add(mul(layout.dims.back(), classes));
  add(classes);
  return total;
}

} // namespace datamunge::nn::models
