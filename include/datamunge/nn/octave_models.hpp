#pragma once
/// @file octave_models.hpp
/// @brief Tensor octave frequency exchange, per-stream normalization and complete Oct-ResNet.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/resnet_models.hpp"
#include <cmath>
#include <limits>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief High-resolution and optional half-resolution feature streams.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Channel and nominal-cost integers are architecture metadata, not model values.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::OctaveMaps x{n::ones({1, 8, 8, 6}), n::ones({1, 4, 4, 2}), 6, 2};
/// @endcode
/// @par References
/// - Chen et al. (2019), https://arxiv.org/abs/1904.05049; https://github.com/facebookresearch/OctConv
/// @see OctaveConv(), OctaveNorm(), OctResNet()
struct OctaveMaps {
  nn::Tensor high; ///< Nonempty floating NHWC high map; always present.
  nn::Tensor
      low; ///< Optional same-dtype/batch NHWC low map at exactly half both spatial sides.
  int high_channels = 0; ///< Positive high-map width matching high.size(3).
  int low_channels = 0;  ///< Zero for absent low stream, otherwise low.size(3).
};
/// @brief Compute the integer low-frequency width for a positive total channel count.
///
/// Architecture metadata arithmetic only; no tensors or fitting.
///
/// @param channels Positive integer total width.
/// @param alpha Finite low-frequency fraction in [0,1).
/// @return floor(alpha*channels), leaving at least one high channel.
/// @throws std::invalid_argument For invalid dimensions/split/pooling alignment/BN settings, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// const int low = dm::nn::models::LowChannels(16, .25); // 4
/// @endcode
/// @par References
/// - Chen et al. (2019), https://arxiv.org/abs/1904.05049; https://github.com/facebookresearch/OctConv
/// @see OctaveMaps, OctaveConv(), OctaveNorm(), OctResNet()
inline int LowChannels(int channels, double alpha) {
  if (channels < 1 || !(alpha >= 0 && alpha < 1))
    throw std::invalid_argument("invalid octave channel split");
  return int(alpha * channels);
}

/// @brief Accumulated per-example nominal convolution MAC metadata.
///
/// Architecture/arithmetic metadata only; no learned allocation or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::OctaveCost cost;
/// @endcode
/// @par References
/// - Chen et al. (2019), https://arxiv.org/abs/1904.05049; https://github.com/facebookresearch/OctConv
/// @see OctaveConv(), OctaveNorm(), OctResNet()
struct OctaveCost {
  std::int64_t macs =
      0; ///< Nonnegative total, updated by OctaveConv; dimensions are nominal target sizes.
};

/// @brief Exchange high/low feature streams through four tensor convolution paths.
///
/// Yh=conv(Xh,Whh)+upsample2(conv(Xl,Wlh));
/// Yl=conv(avgpool2(Xh),Whl)+conv(Xl,Wll). Missing low paths are
/// omitted; alpha_out zero merges back to high only. Uses 2x2 VALID
/// average pooling and nearest-neighbour upsampling, symmetric padding
/// for odd kernels, no biases/norm/activation. All path filters use
/// He-uniform fan-in of the FULL input width, matching source adaptation.
/// Actual maps must align exactly through pooling/upsampling; odd low
/// spatial sizes at stride two are rejected for continuing low streams.
/// A stride-two high-only merge uses unpooled Xl directly, without
/// upsampling: its resolution already matches pooled Xh. This fixes the
/// source final-stage shortcut mismatch for odd low maps (e.g.7x7 at224).
/// The corrected LH path MACs use the output high side, not floor(side/2).
/// At alpha zero, stride one equals ordinary convolution; stride two
/// still pre-pools before filtering and differs from unpooled strided conv.
/// Four paths partition an ordinary kernel when both frequencies exist;
/// MAC savings depend on nominal sizes, not measured runtime. No fitting.
///
/// @param p Mutable floating learned store, creates/reuses /hh,/hl,/lh,/ll kernels.
/// @param name Layer prefix.
/// @param x Valid high and optional half-resolution low streams, matching store dtype.
/// @param out Positive total output channels.
/// @param alpha_out Low-output fraction in [0,1).
/// @param kernel Positive odd square side.
/// @param stride One or two; two pre-pools streams, except LH in a high-only merge.
/// @param high_size Positive nominal target high side before stride; exact pooling alignment required.
/// @param cost Optional nonnegative accumulator; checked per-path nominal MAC terms added.
/// @return Differentiable high map and optional half-resolution low map.
/// @throws std::invalid_argument For invalid dimensions/split/pooling alignment/BN settings, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::OctaveMaps x{n::ones({1, 8, 8, 6}), n::ones({1, 4, 4, 2}), 6, 2};
/// const auto y = n::models::OctaveConv(p, "o", x, 8, .25, 3, 1, 8);
/// @endcode
/// @par References
/// - Chen et al. (2019), https://arxiv.org/abs/1904.05049; https://github.com/facebookresearch/OctConv
/// @see OctaveMaps, OctaveConv(), OctaveNorm(), OctResNet()
inline OctaveMaps OctaveConv(nn::ParameterStore &p, const std::string &name,
                             const OctaveMaps &x, int out, double alpha_out, int kernel,
                             int stride, int high_size, OctaveCost *cost = nullptr) {
  if (x.high_channels < 1 || x.low_channels < 0 ||
      int64_t(x.high_channels) + x.low_channels > std::numeric_limits<int>::max() ||
      x.high.dim() != 4 || x.high.numel() == 0 || x.high.size(3) != x.high_channels ||
      (x.low_channels &&
       (x.low.dim() != 4 || x.low.numel() == 0 || x.low.size(0) != x.high.size(0) ||
        x.low.dtype() != x.high.dtype() || x.low.size(3) != x.low_channels ||
        x.high.size(1) != 2 * x.low.size(1) || x.high.size(2) != 2 * x.low.size(2))))
    throw std::invalid_argument("invalid octave stream geometry");
  if (out < 1 || kernel < 1 || kernel % 2 != 1 || (stride != 1 && stride != 2) ||
      high_size < stride || high_size % stride || (cost && cost->macs < 0))
    throw std::invalid_argument("invalid octave convolution metadata");
  const int in = x.high_channels + x.low_channels, out_low = LowChannels(out, alpha_out),
            out_high = out - out_low;
  /// A high-only stride-two merge uses the low stream at the output resolution directly.
  const bool merge_stride2 = stride == 2 && x.low_channels > 0 && out_low == 0;
  if (x.high.size(1) % stride || x.high.size(2) % stride ||
      (x.low_channels && !merge_stride2 &&
       (x.low.size(1) % stride || x.low.size(2) % stride)) ||
      (out_low && ((x.high.size(1) / stride) % 2 || (x.high.size(2) / stride) % 2)) ||
      ((x.low_channels || out_low) && !merge_stride2 && (high_size / stride) % 2))
    throw std::invalid_argument("octave pooling requires exact half-resolution alignment");
  const auto init = nn::init::uniform(
      std::sqrt(6.0 / (double(kernel) * kernel * in))); // He uniform, full-layer fan-in
  /// Apply one frequency path and accumulate its checked nominal MAC term.
  auto conv = [&](const std::string &part, const nn::Tensor &v, int from, int to, int size) {
    const nn::Tensor w = p.get(name + "/" + part + "/W", {kernel, kernel, from, to}, init);
    if (cost) {
      int64_t term = size;
      for (int64_t factor :
           {int64_t(size), int64_t(kernel), int64_t(kernel), int64_t(from), int64_t(to)}) {
        if (factor && term > std::numeric_limits<int64_t>::max() / factor)
          throw std::invalid_argument("octave MAC count overflow");
        term *= factor;
      }
      if (term > std::numeric_limits<int64_t>::max() - cost->macs)
        throw std::invalid_argument("octave MAC count overflow");
      cost->macs += term;
    }
    nn::Conv2DOptions o = nn::conv_options(1, nn::Padding::Valid);
    if (kernel > 1) {
      o.padding = nn::Padding::Explicit;
      o.explicit_paddings = {kernel / 2, kernel / 2, kernel / 2, kernel / 2};
    }
    return nn::conv2d(v, w, o);
  };
  nn::Tensor xh = x.high, xl = x.low;
  int size = high_size;
  if (stride == 2) {
    xh = nn::avg_pool2d(xh, 2, 2, nn::Padding::Valid);
    if (x.low_channels && !merge_stride2)
      xl = nn::avg_pool2d(xl, 2, 2, nn::Padding::Valid);
    size /= 2;
  }
  OctaveMaps y;
  y.high_channels = out_high;
  y.low_channels = out_low;
  y.high = conv("hh", xh, x.high_channels, out_high, size);
  if (x.low_channels)
    y.high =
        y.high +
        (merge_stride2
             ? conv("lh", xl, x.low_channels, out_high, size)
             : nn::upsample_nearest(conv("lh", xl, x.low_channels, out_high, size / 2), 2));
  if (out_low) {
    y.low = conv("hl", nn::avg_pool2d(xh, 2, 2, nn::Padding::Valid), x.high_channels, out_low,
                 size / 2);
    if (x.low_channels)
      y.low = y.low + conv("ll", xl, x.low_channels, out_low, size / 2);
  }
  return y;
}

/// @brief Apply separate learned-scale BN and optional ReLU to each frequency stream.
///
/// BN epsilon 1e-5, learned gamma/beta. Missing low stream stays
/// undefined. No fitting; callers synchronize shared learned/state stores.
///
/// @param p Mutable floating BN affine leaves.
/// @param state Mutable matching-dtype per-stream moving moments.
/// @param name Prefix for /high/bn and optional /low/bn.
/// @param x Valid high and optional half-resolution low streams matching stores.
/// @param relu Apply ReLU after each BN if true.
/// @param training Use batch BN if true, saved moments otherwise.
/// @param momentum Moving retention in [0,1).
/// @param update_stats Permit eager training moving updates, default true.
/// @return Same stream geometry and channel metadata, retaining full derivatives.
/// @throws std::invalid_argument For invalid dimensions/split/pooling alignment/BN settings, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::OctaveMaps x{n::ones({2, 8, 8, 6}), n::ones({2, 4, 4, 2}), 6, 2};
/// const auto y = n::models::OctaveNorm(p, state, "n", x, true, false, .9);
/// @endcode
/// @par References
/// - Chen et al. (2019), https://arxiv.org/abs/1904.05049; https://github.com/facebookresearch/OctConv
/// @see OctaveMaps, OctaveConv(), OctaveNorm(), OctResNet()
inline OctaveMaps OctaveNorm(nn::ParameterStore &p, nn::ParameterStore &state,
                             const std::string &name, const OctaveMaps &x, bool relu,
                             bool training, double momentum, bool update_stats = true) {
  if (x.high_channels < 1 || x.low_channels < 0 ||
      int64_t(x.high_channels) + x.low_channels > std::numeric_limits<int>::max() ||
      x.high.dim() != 4 || x.high.numel() == 0 || x.high.size(3) != x.high_channels ||
      (x.low_channels &&
       (x.low.dim() != 4 || x.low.numel() == 0 || x.low.size(0) != x.high.size(0) ||
        x.low.dtype() != x.high.dtype() || x.low.size(3) != x.low_channels ||
        x.high.size(1) != 2 * x.low.size(1) || x.high.size(2) != 2 * x.low.size(2))))
    throw std::invalid_argument("invalid octave stream geometry");
  if (!(momentum >= 0 && momentum < 1))
    throw std::invalid_argument("invalid octave BN momentum");
  nn::BatchNormOptions bn;
  bn.momentum = momentum;
  bn.eps = 1e-5;
  bn.update_stats = update_stats;
  /// Normalize one stream independently, with optional post-BN ReLU.
  auto norm = [&](const std::string &part, const nn::Tensor &v) {
    const nn::Tensor h = nn::batch_norm(p, state, name + "/" + part + "/bn", v, training, bn);
    return relu ? nn::relu(h) : h;
  };
  OctaveMaps y = x;
  y.high = norm("high", x.high);
  if (x.low_channels)
    y.low = norm("low", x.low);
  return y;
}

/// @brief Octave residual backbone topology, target budget side and normalization options.
///
/// Architecture/arithmetic metadata only; no learned allocation or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::OctResNetConfig cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.image_size = 32;
/// cfg.small_stem = true;
/// @endcode
/// @par References
/// - Chen et al. (2019), https://arxiv.org/abs/1904.05049; https://github.com/facebookresearch/OctConv
/// @see OctaveConv(), OctaveNorm(), OctResNet()
struct OctResNetConfig {
  int depth = 50;        ///< Bottleneck depth 50,101 or152.
  int classes = 1000;    ///< At least two classifier outputs.
  int width_divisor = 1; ///< Positive integer at most64; stage planes use integer division.
  int image_size =
      224; ///< Positive nominal square budget side, multiple32 or multiple8 with small_stem.
  double alpha = .25;       ///< Finite low-channel proportion in [0,1).
  bool small_stem = false;  ///< 3x3/1 without pooling rather than7x7/2 and3x3/2 pool.
  double momentum = .9;     ///< BN moving retention in [0,1).
  bool update_stats = true; ///< Permit eager moving updates in training.
};
/// @brief Final single-frequency features, classifier logits and nominal arithmetic budget.
///
/// Tensor handles own storage references and retain differentiation graphs.
/// Channel and nominal-cost integers are architecture metadata, not model values.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::OctResNetConfig cfg;
/// cfg.width_divisor = 4;
/// cfg.classes = 10;
/// const auto out = n::models::OctResNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Chen et al. (2019), https://arxiv.org/abs/1904.05049; https://github.com/facebookresearch/OctConv
/// @see OctaveConv(), OctaveNorm(), OctResNet()
struct OctResNetOutputs {
  nn::Tensor features; ///< NHWC final high-only map after last-stage merge.
  nn::Tensor logits;   ///< [N,classes] biased global-mean classifier scores.
  std::int64_t macs =
      0; ///< Per-example convolution/classifier MACs at cfg.image_size, not necessarily actual input size.
};

/// @brief Build a complete octave-frequency ResNet bottleneck classifier without fitting.
///
/// A 7x7/2 BN/ReLU stem plus3x3/2 max pool (small3x3/1 without
/// pool) enters high-only. All three bottleneck convolutions/projections
/// become octave layers. Four full stages preserve ResNet repeats; first
/// three use cfg.alpha, while the FIRST conv1 in stage four merges to
/// high-only BEFORE its spatial stride-two layer. This avoids a low half
/// of the final odd7x7 map at reference224 pixels. Its shortcut uses the
/// corrected direct-LH high-only stride-two merge. Stage stride is in3x3.
/// Per-stream BN eps1e-5; He-uniform full-layer fans, global mean/biased
/// classifier. Width division uses positive integer floors. Nominal MACs
/// can describe224 while actual count-only inputs use32; report this
/// difference explicitly. Kernel/classifier arithmetic excludes pooling,
/// upsampling, BN/ReLU/addition and memory traffic. No preprocessing,
/// loss, optimizer or fitting. Shared store mutation needs synchronization.
///
/// @param p Mutable floating learned store, lazily creates/reuses named leaves.
/// @param state Mutable matching-dtype BN moving-moment store.
/// @param image Nonempty floating RGB NHWC input; sides multiples32 or8 with small_stem.
/// @param cfg Valid bottleneck depth, split, widths, classes, nominal square side and BN metadata.
/// @param training True uses batch BN, optionally updating moving values; false saved moments.
/// @return Final high-only features/logits and nominal cfg.image_size MAC budget.
/// @throws std::invalid_argument For invalid dimensions/split/pooling alignment/BN settings, overflow, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::OctResNetConfig cfg;
/// cfg.width_divisor = 4;
/// cfg.classes = 10;
/// cfg.small_stem = true;
/// cfg.image_size = 32;
/// const auto y = n::models::OctResNet(p, state, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Chen et al. (2019), https://arxiv.org/abs/1904.05049; https://github.com/facebookresearch/OctConv
/// @see OctaveMaps, OctaveConv(), OctaveNorm(), OctResNet()
inline OctResNetOutputs OctResNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const nn::Tensor &image, const OctResNetConfig &cfg = {},
                                  bool training = false) {
  const auto counts = ResNetBlockCounts(cfg.depth);
  if (image.dim() != 4 || image.numel() == 0 || image.size(3) != 3 || cfg.width_divisor < 1 ||
      cfg.width_divisor > 64 || cfg.image_size < 1 ||
      cfg.image_size % (cfg.small_stem ? 8 : 32) ||
      image.size(1) % (cfg.small_stem ? 8 : 32) || image.size(2) % (cfg.small_stem ? 8 : 32) ||
      !(cfg.momentum >= 0 && cfg.momentum < 1) || cfg.depth < 50 ||
      !(cfg.alpha >= 0 && cfg.alpha < 1) || cfg.classes < 2)
    throw std::invalid_argument("invalid Oct-ResNet configuration");
  OctResNetOutputs out;
  OctaveCost cost;
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.epsilon = 1e-5;
  norm.momentum = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  int in = 64 / cfg.width_divisor, size = cfg.small_stem ? cfg.image_size : cfg.image_size / 4;
  nn::Tensor stem = ResNetConv(p, state, "stem", image, in, cfg.small_stem ? 3 : 7, norm,
                               cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    stem =
        nn::max_pool2d(nn::pad(stem, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  cost.macs += std::int64_t(cfg.small_stem ? size : 2 * size) *
               (cfg.small_stem ? size : 2 * size) * (cfg.small_stem ? 27 : 147) * in;
  OctaveMaps h;
  h.high = stem;
  h.high_channels = in;
  for (int stage = 0; stage < 4; ++stage) {
    const double alpha = stage < 3 ? cfg.alpha : 0;
    const int planes = (64 << stage) / cfg.width_divisor, outputs = 4 * planes;
    for (int block = 0; block < counts[stage]; ++block) {
      const auto name =
          "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      const int stride = stage > 0 && block == 0 ? 2 : 1;
      /// Compose a frequency-exchange convolution with independent stream BN.
      auto layer = [&](const std::string &part, const OctaveMaps &x, int to, int kernel,
                       int step, int at, bool relu) {
        return OctaveNorm(
            p, state, name + "/" + part,
            OctaveConv(p, name + "/" + part, x, to, alpha, kernel, step, at, &cost), relu,
            training, cfg.momentum, cfg.update_stats);
      };
      OctaveMaps r = layer("conv1", h, planes, 1, 1, size, true);
      r = layer("conv2", r, planes, 3, stride, size, true);
      const int next = stride == 2 ? size / 2 : size;
      r = layer("conv3", r, outputs, 1, 1, next, false);
      OctaveMaps shortcut = h;
      if (in != outputs || stride != 1 || (h.low_channels == 0) != (r.low_channels == 0))
        shortcut = layer("shortcut", h, outputs, 1, stride, size, false);
      h.high = nn::relu(shortcut.high + r.high);
      h.high_channels = r.high_channels;
      h.low_channels = r.low_channels;
      h.low = r.low_channels ? nn::relu(shortcut.low + r.low) : nn::Tensor();
      in = outputs;
      size = next;
    }
  }
  out.features = h.high;
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h.high), cfg.classes);
  out.macs = cost.macs + std::int64_t(in) * cfg.classes;
  return out;
}
} // namespace datamunge::nn::models
