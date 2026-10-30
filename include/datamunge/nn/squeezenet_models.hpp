#pragma once
/// @file squeezenet_models.hpp
/// @brief Fire modules and complete SqueezeNet tensor classifiers with ceil pooling.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Squeeze activations, both Fire expand branches and their concatenation.
///
/// Tensor handles retain storage and autograd graphs; learned leaves belong to p.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::Fire(p, "fire", n::ones({1, 5, 5, 8}), 2, 4, 4);
/// @endcode
/// @par References
/// - Iandola et al. (2016), https://arxiv.org/abs/1602.07360; release https://github.com/forresti/SqueezeNet
/// @see Fire()
struct FireOutputs {
  nn::Tensor squeeze;   ///< Biased 1x1/ReLU reduced-width map.
  nn::Tensor expand1x1; ///< Biased 1x1/ReLU expansion of squeezed activations.
  nn::Tensor expand3x3; ///< Biased SAME 3x3/ReLU expansion of squeezed activations.
  nn::Tensor output;    ///< Concatenation [expand1x1,expand3x3] on NHWC channels.
};
/// @brief Apply a Fire squeeze layer and parallel 1x1/3x3 expand layers.
///
/// All filters use He-uniform weights, zero biases and ReLU; no BN/dropout.
/// Kernel budget Cin*squeeze+squeeze*expand1x1+9*squeeze*expand3x3, plus
/// biases. The two expand filters can be centre-embedded into one 3x3 after
/// the squeeze, but the nonlinear squeeze is not a single linear input filter.
/// No preprocessing, optimizer or fitting is hidden.
///
/// @param p Mutable floating learned store, lazily creates/reuses leaves.
/// @param name Prefix for squeeze1x1, expand1x1 and expand3x3 W/b.
/// @param x Nonempty floating NHWC input matching p dtype.
/// @param squeeze Positive reduced channel width.
/// @param expand1x1 Positive output width of the 1x1 branch.
/// @param expand3x3 Positive output width of the SAME 3x3 branch.
/// @return Squeeze/expand maps and NHWC concatenation, preserving spatial sides.
/// @throws std::invalid_argument For invalid NHWC, widths, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::Fire(p, "fire", n::ones({1, 5, 5, 8}), 2, 4, 4);
/// @endcode
/// @par References
/// - Iandola et al. (2016), https://arxiv.org/abs/1602.07360; release https://github.com/forresti/SqueezeNet
/// @see SqueezeNet()
inline FireOutputs Fire(nn::ParameterStore &p, const std::string &name, const nn::Tensor &x,
                        int squeeze, int expand1x1, int expand3x3) {
  if (x.dim() != 4 || x.numel() == 0 || squeeze < 1 || expand1x1 < 1 || expand3x3 < 1)
    throw std::invalid_argument("invalid fire module");
  nn::ConvLayerOptions o;
  o.init = nn::init::he_uniform();
  o.activation = nn::Activation::Relu;
  FireOutputs out;
  out.squeeze = nn::conv2d(p, name + "/squeeze1x1", x, squeeze, 1, o);
  out.expand1x1 = nn::conv2d(p, name + "/expand1x1", out.squeeze, expand1x1, 1, o);
  out.expand3x3 = nn::conv2d(p, name + "/expand3x3", out.squeeze, expand3x3, 3, o);
  out.output = nn::concat({out.expand1x1, out.expand3x3}, 3);
  return out;
}
/// @brief Apply Caffe-style ceil-rounded 3x3 stride-two max pooling to a square map.
///
/// If needed append one row/column of negative infinity at bottom/right,
/// then use VALID pooling. Padding cannot win against finite input pixels.
/// Work O(NHoWo*C*9); gradient follows max-pool switches. No fitting.
///
/// @param x Nonempty floating square NHWC map; handle copied, storage not mutated.
/// @param size Input side at least two, matching x; overwritten with output side.
/// @return Pooled NHWC map with side ceil((size_old-3)/2)+1.
/// @throws std::invalid_argument For invalid rank, nonsquare/inconsistent side or a side below two.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// int side = 4;
/// const auto y = n::models::CeilMaxPool(n::ones({1, 4, 4, 2}), side); // side=2
/// @endcode
/// @par References
/// - Iandola et al. (2016), https://arxiv.org/abs/1602.07360; release https://github.com/forresti/SqueezeNet
/// @see SqueezeNet()
inline nn::Tensor CeilMaxPool(nn::Tensor x, int &size) {
  if (x.dim() != 4 || x.numel() == 0 || size < 2 || x.size(1) != size || x.size(2) != size)
    throw std::invalid_argument(
        "ceil pooling requires matching square NHWC sides of at least two");
  const int pad = (size - 3) % 2 ? 1 : 0;
  if (pad)
    x = nn::pad(x, {{0, 0}, {0, 1}, {0, 1}, {0, 0}}, nn::PadMode::Constant,
                -std::numeric_limits<double>::infinity());
  size = (size + pad - 3) / 2 + 1;
  return nn::max_pool2d(x, 3, 2, nn::Padding::Valid);
}
/// @brief SqueezeNet release version, square image size and dropout/bypass settings.
///
/// Defaults describe v1.1 at 224 pixels with 1000 classes and .5 dropout
/// retention. Value metadata only, no allocation or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::SqueezeNetConfig cfg;
/// cfg.classes = 10;
/// cfg.image_size = 32;
/// cfg.cifar_stem = true;
/// @endcode
/// @par References
/// - Iandola et al. (2016), https://arxiv.org/abs/1602.07360; release https://github.com/forresti/SqueezeNet
/// @see SqueezeNet()
struct SqueezeNetConfig {
  int version = 11;     ///< Exactly 10 (v1.0) or 11 (v1.1).
  int classes = 1000;   ///< At least two class maps before global mean.
  int image_size = 224; ///< Square input side, at least 16 and large enough for three pools.
  bool simple_bypass =
      false; ///< Add Fire input where expand output width matches, no parameters added.
  bool cifar_stem =
      false; ///< Use SAME 3x3 stride-one stem instead of release VALID strided stem.
  double keep_probability = .5; ///< Final-feature dropout retention in (0,1], training only.
};
/// @brief Stem, every Fire map, class activation maps and global-average logits.
///
/// Tensor handles retain storage and autograd history; learned leaves remain in p.
/// The logits are averages of ReLU class maps, not unconstrained linear scores.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::SqueezeNetConfig cfg;
/// cfg.image_size = 32;
/// cfg.classes = 10;
/// cfg.cifar_stem = true;
/// const auto y = n::models::SqueezeNet(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @see SqueezeNet()
struct SqueezeNetOutputs {
  nn::Tensor stem;               ///< Initial convolution/ReLU before its first pool.
  nn::Tensor features;           ///< ReLU class maps after conv10, [N,H,W,classes].
  nn::Tensor logits;             ///< Mean class-map scores [N,classes], nonnegative.
  std::vector<nn::Tensor> fires; ///< Fire2..Fire9 outputs, including optional simple bypasses.
  std::vector<int> sizes; ///< Side of each recorded Fire output before any following pool.
};
/// @brief Build complete SqueezeNet-v1.0/v1.1 tensor scores without fitting.
///
/// V1.0 uses VALID 7x7/2 to 96 channels, v1.1 VALID 3x3/2 to 64.
/// Both initially pool then apply eight Fire modules; later pools follow
/// Fire4/8 (v1.0) or Fire3/5 (v1.1), using Caffe ceil geometry. At 224
/// pixels Fire stage sides are 54/27/13 or 55/27/13. Optional same-width
/// simple bypasses add no parameters. Full learned counts 1248424/1235496.
/// No BN or moving-state store exists. Dropout follows the last Fire; conv10
/// is a biased 1x1 initialized N(0,.01) with ReLU, then global mean.
/// A dead final ReLU may eliminate data gradients; the optimizer rate is the
/// caller responsibility. No preprocessing, loss, optimizer or fitting is hidden.
///
/// @param p Mutable floating learned store, lazily creates/reuses named leaves.
/// @param image Nonempty NHWC square images matching cfg.image_size and p dtype.
/// @param cfg Valid release, class, geometry and dropout metadata, default v1.1 ImageNet.
/// @param training True enables final-feature dropout; inference is deterministic after initialization.
/// @param generator Borrowed dropout generator, required only for training with keep_probability<1.
/// @return Stem, Fire2..Fire9 maps/sides, ReLU class maps and global-average scores.
/// @throws std::invalid_argument For invalid dimensions, release, dropout, undersized pool geometry, missing training generator, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::SqueezeNetConfig cfg;
/// cfg.image_size = 32;
/// cfg.classes = 10;
/// cfg.cifar_stem = true;
/// const auto y = n::models::SqueezeNet(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Iandola et al. (2016), https://arxiv.org/abs/1602.07360; release https://github.com/forresti/SqueezeNet
/// @see Fire(), CeilMaxPool(), SqueezeNetConfig
inline SqueezeNetOutputs SqueezeNet(nn::ParameterStore &p, const nn::Tensor &image,
                                    const SqueezeNetConfig &cfg = {}, bool training = false,
                                    nn::Generator *generator = nullptr) {
  if (image.dim() != 4 || image.numel() == 0 || image.size(1) != cfg.image_size ||
      image.size(2) != cfg.image_size)
    throw std::invalid_argument(
        "SqueezeNet requires nonempty NHWC matching configured square side");
  if ((cfg.version != 10 && cfg.version != 11) || cfg.classes < 2 || cfg.image_size < 16 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid SqueezeNet configuration");
  const int fires[8][2] = {{16, 64},  {16, 64},  {32, 128}, {32, 128},
                           {48, 192}, {48, 192}, {64, 256}, {64, 256}};
  const std::vector<int> pools =
      cfg.version == 10 ? std::vector<int>{4, 8} : std::vector<int>{3, 5};
  SqueezeNetOutputs out;
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  conv.activation = nn::Activation::Relu;
  int size = cfg.image_size, channels = cfg.version == 10 ? 96 : 64;
  if (cfg.cifar_stem)
    out.stem = nn::conv2d(p, "conv1", image, channels, 3, conv);
  else {
    const int kernel = cfg.version == 10 ? 7 : 3;
    conv.stride = 2;
    conv.padding = nn::Padding::Valid;
    out.stem = nn::conv2d(p, "conv1", image, channels, kernel, conv);
    size = (size - kernel) / 2 + 1;
  }
  auto h = CeilMaxPool(out.stem, size);
  for (int i = 0; i < 8; ++i) {
    const int width = 2 * fires[i][1];
    const auto fire =
        Fire(p, "fire" + std::to_string(i + 2), h, fires[i][0], fires[i][1], fires[i][1]);
    h = cfg.simple_bypass && channels == width ? nn::add(h, fire.output) : fire.output;
    channels = width;
    out.fires.push_back(h);
    out.sizes.push_back(size);
    if (std::find(pools.begin(), pools.end(), i + 2) != pools.end())
      h = CeilMaxPool(h, size);
  }
  if (training && cfg.keep_probability < 1) {
    if (!generator)
      throw std::invalid_argument("SqueezeNet training requires a dropout generator");
    h = nn::dropout(h, 1 - cfg.keep_probability, true, *generator);
  }
  conv = {};
  conv.init = nn::init::normal(.01);
  conv.activation = nn::Activation::Relu;
  out.features = nn::conv2d(p, "conv10", h, cfg.classes, 1, conv);
  out.logits = nn::global_avg_pool2d(out.features);
  return out;
}
} // namespace datamunge::nn::models
