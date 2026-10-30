#pragma once
/// @file alexnet_googlenet.hpp
/// @brief AlexNet and GoogLeNet tensor forward definitions, without fitting loops.
#include "datamunge/nn/classic_cnn_components.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Reference or compact dimensions and dropout settings for AlexNet.
///
/// Configuration is value metadata; it does not allocate or fit model parameters.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AlexNetConfig cfg;
/// cfg.image_size = 67;
/// cfg.width_divisor = 8;
/// cfg.classes = 3;
/// cfg.dense_width = 16;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Krizhevsky et al. (2012), https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
/// @see AlexNet()
struct AlexNetConfig {
  int image_size = 227;         ///< Square RGB input side, at least 67; reference 227.
  int classes = 1000;           ///< Output width, at least two.
  int width_divisor = 1;        ///< One, two, four or eight; scales every convolution width.
  int dense_width = 4096;       ///< Positive hidden dense-layer width.
  double keep_probability = .5; ///< Inverted hidden dropout retention in (0,1].

  /// @brief Reject invalid image, width, class, hidden-layer and dropout settings.
  ///
  /// @throws std::invalid_argument For dimensions or dropout outside the documented domain.
  /// @par Example
  /// @code
  /// dm::nn::models::AlexNetConfig{}.Validate();
  /// @endcode
  /// @see AlexNetConfig
  void Validate() const {
    if (image_size < 67 || classes < 2 || dense_width < 1 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 &&
         width_divisor != 8) ||
        !(keep_probability > 0 && keep_probability <= 1))
      throw std::invalid_argument("invalid AlexNet configuration");
  }
  /// @brief Calculate the last convolutional pool spatial side from configured input geometry.
  ///
  /// @return Positive integer feature-map side; reference 6.
  /// @throws std::invalid_argument For invalid configuration.
  /// @par Example
  /// @code
  /// const auto side = dm::nn::models::AlexNetConfig{}.FinalSpatialSize();
  /// @endcode
  /// @see Validate()
  int FinalSpatialSize() const {
    Validate();
    int size = (image_size - 11) / 4 + 1;
    for (int pool = 0; pool < 3; ++pool)
      size = (size - 3) / 2 + 1;
    return size;
  }
};

/// @brief Use the AlexNet paper LRN constants on the last channel axis.
///
/// The clipped window has radius two; alpha is not divided by five.
///
/// @param x Floating [...,C] tensor with a positive last dimension.
/// @return Differentiable x / (2 + .0001 * sum_five_nearby(x*x))^.75.
/// @throws std::invalid_argument For invalid input axis or dtype.
/// @par Example
/// @code
/// const auto y = dm::nn::models::AlexNetLrn(dm::nn::ones({1, 3, 3, 8}));
/// @endcode
/// @par References
/// - Krizhevsky et al. (2012), https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
/// @see LocalResponseNorm()
inline nn::Tensor AlexNetLrn(const nn::Tensor &x) {
  return LocalResponseNorm(x, 2, 2.0, .0001, .75);
}

/// @brief Apply training-mode inverted dropout with a retention probability.
///
/// This helper always requests training behaviour; the architecture calls it only
/// when training=true. No optimizer runs here.
///
/// @param p Store whose generator supplies the mask; generator advances for keep<1.
/// @param x Floating tensor, any shape.
/// @param keep Finite retention in (0,1]; one returns x without consuming randomness.
/// @return Differentiable randomly masked tensor scaled by 1/keep, or x at keep=1.
/// @throws std::invalid_argument For invalid retention or unsupported tensor dtype.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::KeepDropout(p, n::ones({2, 8}), .5);
/// @endcode
/// @see AlexNet()
inline nn::Tensor KeepDropout(nn::ParameterStore &p, const nn::Tensor &x, double keep) {
  if (!(keep > 0 && keep <= 1))
    throw std::invalid_argument("invalid dropout retention");
  return keep < 1 ? nn::dropout(x, 1 - keep, true, p.generator()) : x;
}

/// @brief Intermediate activations and logits from AlexNet.
///
/// Tensor handles retain storage and their differentiation graphs. No fitted state
/// is copied into this result; learned leaves belong to the parameter store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AlexNetConfig cfg;
/// cfg.image_size = 67;
/// cfg.width_divisor = 8;
/// cfg.classes = 3;
/// cfg.dense_width = 16;
/// const auto out = n::models::AlexNet(p, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// @endcode
/// @see AlexNet()
struct AlexNetOutputs {
  std::array<nn::Tensor, 5> convolution; ///< Five post-ReLU maps in execution order.
  std::array<nn::Tensor, 3> pooling;     ///< Three overlapping VALID 3x3 stride-two pools.
  nn::Tensor fc6;                        ///< First dense ReLU, before dropout.
  nn::Tensor fc7;                        ///< Second dense ReLU, before dropout.
  nn::Tensor logits;                     ///< Unnormalised [N,classes] scores.
};

/// @brief Build AlexNet and expose intermediate activations without fitting parameters.
///
/// Five convolutions, grouped layers 2/4/5, summed-square LRN and overlapping
/// VALID pooling precede two ReLU dense layers and linear logits. Geometry uses
/// BVLC 227-pixel input, while LRN constants follow the paper. He/Glorot
/// initialization and inverted dropout are adaptations. Work is ordinary
/// convolution plus dense matrix multiplication; full model has 60965224 scalars.
///
/// @param p Mutable floating parameter store with lazy named learnable leaves.
/// @param image Floating [N,cfg.image_size,cfg.image_size,3], same dtype as p.
/// @param cfg Valid reference or explicitly compact dimensions and dropout.
/// @param training Enable dropout if true; false by default.
/// @return Stage outputs and [N,classes] main logits, retaining differentiation graphs.
/// @throws std::invalid_argument For invalid cfg, RGB input geometry, dtype or parameter entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AlexNetConfig cfg;
/// cfg.image_size = 67;
/// cfg.width_divisor = 8;
/// cfg.classes = 3;
/// cfg.dense_width = 16;
/// const auto out = n::models::AlexNet(p, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// @endcode
/// @par References
/// - Krizhevsky et al. (2012), https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
/// @see AlexNetConfig
inline AlexNetOutputs AlexNet(nn::ParameterStore &p, const nn::Tensor &image,
                              const AlexNetConfig &cfg = {}, bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(1) != cfg.image_size || image.size(2) != cfg.image_size ||
      image.size(3) != 3)
    throw std::invalid_argument("image does not match model configuration");
  AlexNetOutputs out;
  nn::ConvLayerOptions options;
  options.init = nn::init::he_uniform();
  options.activation = nn::Activation::Relu;
  options.stride = 4;
  options.padding = nn::Padding::Valid;
  const std::int64_t c1 = 96 / cfg.width_divisor, c2 = 256 / cfg.width_divisor,
                     c3 = 384 / cfg.width_divisor;
  out.convolution[0] = nn::conv2d(p, "conv1", image, c1, 11, options);
  out.pooling[0] = nn::max_pool2d(AlexNetLrn(out.convolution[0]), 3, 2);
  options.stride = 1;
  options.padding = nn::Padding::Same;
  out.convolution[1] = GroupedConv2D(p, "conv2", out.pooling[0], c2, 5, 2, options);
  out.pooling[1] = nn::max_pool2d(AlexNetLrn(out.convolution[1]), 3, 2);
  out.convolution[2] = nn::conv2d(p, "conv3", out.pooling[1], c3, 3, options);
  out.convolution[3] = GroupedConv2D(p, "conv4", out.convolution[2], c3, 3, 2, options);
  out.convolution[4] = GroupedConv2D(p, "conv5", out.convolution[3], c2, 3, 2, options);
  out.pooling[2] = nn::max_pool2d(out.convolution[4], 3, 2);
  const std::int64_t features =
      std::int64_t(cfg.FinalSpatialSize()) * cfg.FinalSpatialSize() * c2;
  nn::DenseOptions dense;
  dense.init = nn::init::he_uniform();
  dense.activation = nn::Activation::Relu;
  out.fc6 =
      nn::dense(p, "fc6", nn::reshape(out.pooling[2], {-1, features}), cfg.dense_width, dense);
  nn::Tensor h = training ? KeepDropout(p, out.fc6, cfg.keep_probability) : out.fc6;
  out.fc7 = nn::dense(p, "fc7", h, cfg.dense_width, dense);
  h = training ? KeepDropout(p, out.fc7, cfg.keep_probability) : out.fc7;
  out.logits = nn::dense(p, "fc8", h, cfg.classes);
  return out;
}
/// @brief Return the nine reference Inception branch widths in Table 1.
///
/// The function-local immutable table is initialized once; concurrent reads are safe.
///
/// @return Immutable process-lifetime array in execution order, 3a through 5b.
/// @par Example
/// @code
/// const auto widths = dm::nn::models::GoogLeNetWidths();
/// const auto first = widths[0].OutputChannels(); // 256
/// @endcode
/// @par References
/// - Szegedy et al. (2015), https://arxiv.org/abs/1409.4842
/// @see InceptionWidths, GoogLeNet()
inline const std::array<InceptionWidths, 9> &GoogLeNetWidths() {
  // Immutable Table 1 widths; initialization is thread-safe and no caller can mutate it.
  static const std::array<InceptionWidths, 9> widths = {{{64, 96, 128, 16, 32, 32},
                                                         {128, 128, 192, 32, 96, 64},
                                                         {192, 96, 208, 16, 48, 64},
                                                         {160, 112, 224, 24, 64, 64},
                                                         {128, 128, 256, 24, 64, 64},
                                                         {112, 144, 288, 32, 64, 64},
                                                         {256, 160, 320, 32, 128, 128},
                                                         {256, 160, 320, 32, 128, 128},
                                                         {384, 192, 384, 48, 128, 128}}};
  return widths;
}

/// @brief Reference or compact dimensions and dropout settings for GoogLeNet.
///
/// Configuration is value metadata; it does not allocate or fit model parameters.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::GoogLeNetConfig cfg;
/// cfg.image_size = 65;
/// cfg.width_divisor = 8;
/// cfg.classes = 3;
/// cfg.auxiliary_hidden = 16;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Szegedy et al. (2015), https://arxiv.org/abs/1409.4842
/// @see GoogLeNet()
struct GoogLeNetConfig {
  int image_size = 224;         ///< Square RGB input side, at least 65; reference 224.
  int classes = 1000;           ///< Output width, at least two.
  int width_divisor = 1;        ///< One, two, four or eight; scales all convolution widths.
  int auxiliary_hidden = 1024;  ///< Positive hidden width in both auxiliary heads.
  double keep_probability = .6; ///< Main-head training dropout retention in (0,1].
  double auxiliary_keep_probability = .3; ///< Auxiliary-head training retention in (0,1].

  /// @brief Reject invalid image, width, class, hidden-layer and dropout settings.
  ///
  /// @throws std::invalid_argument For dimensions or dropout outside the documented domain.
  /// @par Example
  /// @code
  /// dm::nn::models::GoogLeNetConfig{}.Validate();
  /// @endcode
  /// @see GoogLeNetConfig
  void Validate() const {
    if (image_size < 65 || classes < 2 || auxiliary_hidden < 1 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 &&
         width_divisor != 8) ||
        !(keep_probability > 0 && keep_probability <= 1) ||
        !(auxiliary_keep_probability > 0 && auxiliary_keep_probability <= 1))
      throw std::invalid_argument("invalid GoogLeNet configuration");
  }
  /// @brief Calculate the auxiliary pooled spatial side from configured input geometry.
  ///
  /// @return Positive integer feature-map side; reference 4.
  /// @throws std::invalid_argument For invalid configuration.
  /// @par Example
  /// @code
  /// const auto side = dm::nn::models::GoogLeNetConfig{}.AuxiliarySpatialSize();
  /// @endcode
  /// @see Validate()
  int AuxiliarySpatialSize() const {
    Validate();
    return ((image_size + 15) / 16 - 5) / 3 + 1;
  }
};

/// @brief Intermediate activations and logits from GoogLeNet.
///
/// Tensor handles retain storage and their differentiation graphs. No fitted state
/// is copied into this result; learned leaves belong to the parameter store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::GoogLeNetConfig cfg;
/// cfg.image_size = 65;
/// cfg.width_divisor = 8;
/// cfg.classes = 3;
/// cfg.auxiliary_hidden = 16;
/// const auto out =
///     n::models::GoogLeNet(p, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// @endcode
/// @see GoogLeNet()
struct GoogLeNetOutputs {
  std::array<nn::Tensor, 3> stem;      ///< ReLU stem convolutions: 7x7, reduction, 3x3.
  std::array<nn::Tensor, 9> inception; ///< Modules 3a/3b/4a/4b/4c/4d/4e/5a/5b.
  nn::Tensor logits;                   ///< Main unnormalised [N,classes] scores.
  std::vector<nn::Tensor> auxiliary; ///< Empty unless requested, then two [N,classes] tensors.
};

/// @brief Build a pooled auxiliary classifier on a module 4a or 4d feature map.
///
/// Uses VALID 5x5 stride-three average pooling, 1x1 ReLU projection, dense
/// ReLU, optional dropout and linear scores. No fitting or auxiliary loss weighting.
///
/// @param p Mutable store; named projection/hidden/output leaves are lazily created.
/// @param name Auxiliary parameter prefix, normally aux1 or aux2.
/// @param input Floating NHWC feature map of spatial side ceil(image_size/16).
/// @param cfg Valid GoogLeNet dimensions, width and auxiliary dropout retention.
/// @param training Apply auxiliary dropout when true; false is deterministic.
/// @return [N,classes] auxiliary logits with gradients through the input feature map.
/// @throws std::invalid_argument For invalid cfg, input geometry or parameter entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::GoogLeNetConfig cfg;
/// cfg.image_size = 65;
/// cfg.width_divisor = 8;
/// cfg.classes = 3;
/// cfg.auxiliary_hidden = 16;
/// const auto y = n::models::GoogLeNetAuxiliary(p, "aux", n::ones({1, 5, 5, 64}), cfg, false);
/// @endcode
/// @par References
/// - Szegedy et al. (2015), https://arxiv.org/abs/1409.4842
/// @see GoogLeNet()
inline nn::Tensor GoogLeNetAuxiliary(nn::ParameterStore &p, const std::string &name,
                                     const nn::Tensor &input, const GoogLeNetConfig &cfg,
                                     bool training) {
  cfg.Validate();
  const auto pooled = nn::avg_pool2d(input, 5, 3, nn::Padding::Valid);
  nn::ConvLayerOptions conv;
  conv.activation = nn::Activation::Relu;
  conv.init = nn::init::he_uniform();
  const std::int64_t channels = 128 / cfg.width_divisor, spatial = cfg.AuxiliarySpatialSize();
  auto h = nn::conv2d(p, name + "/projection", pooled, channels, 1, conv);
  nn::DenseOptions dense;
  dense.activation = nn::Activation::Relu;
  dense.init = nn::init::he_uniform();
  h = nn::dense(p, name + "/hidden", nn::reshape(h, {-1, spatial * spatial * channels}),
                cfg.auxiliary_hidden, dense);
  if (training)
    h = nn::dropout(h, 1 - cfg.auxiliary_keep_probability, true, p.generator());
  return nn::dense(p, name + "/output", h, cfg.classes);
}

/// @brief Build GoogLeNet and expose intermediate activations without fitting parameters.
///
/// Uses all nine Table 1 Inception modules, SAME pooling, LRN and GAP. Auxiliary
/// heads follow 4a and 4d and have no effect on main inference. The caller must
/// add .3 times each auxiliary cross entropy during training. He initialization
/// and SAME stride-two alignment are adaptations. Full main model has 6998552
/// learned scalars; optional auxiliaries add 6379728.
///
/// @param p Mutable floating parameter store with lazy named learnable leaves.
/// @param image Floating [N,cfg.image_size,cfg.image_size,3], same dtype as p.
/// @param cfg Valid reference or explicitly compact dimensions and dropout.
/// @param training Enable dropout if true; false by default.
/// @param auxiliary Construct the two auxiliary heads if true; false by default.
/// @return Stage outputs and [N,classes] main logits, retaining differentiation graphs.
/// @throws std::invalid_argument For invalid cfg, RGB input geometry, dtype or parameter entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::GoogLeNetConfig cfg;
/// cfg.image_size = 65;
/// cfg.width_divisor = 8;
/// cfg.classes = 3;
/// cfg.auxiliary_hidden = 16;
/// const auto out =
///     n::models::GoogLeNet(p, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// @endcode
/// @par References
/// - Szegedy et al. (2015), https://arxiv.org/abs/1409.4842
/// @see GoogLeNetConfig
inline GoogLeNetOutputs GoogLeNet(nn::ParameterStore &p, const nn::Tensor &image,
                                  const GoogLeNetConfig &cfg = {}, bool training = false,
                                  bool auxiliary = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(1) != cfg.image_size || image.size(2) != cfg.image_size ||
      image.size(3) != 3)
    throw std::invalid_argument("image does not match model configuration");
  GoogLeNetOutputs out;
  nn::ConvLayerOptions conv;
  conv.activation = nn::Activation::Relu;
  conv.init = nn::init::he_uniform();
  conv.stride = 2;
  // BVLC divides alpha by the five-channel window; convert to summed-square LRN.
  auto lrn = [](const nn::Tensor &x) {
    return LocalResponseNorm(x, 2, 1.0, .0001 / 5, .75);
  };
  const std::int64_t c1 = 64 / cfg.width_divisor, c2 = 192 / cfg.width_divisor;
  out.stem[0] = nn::conv2d(p, "stem/conv7", image, c1, 7, conv);
  nn::Tensor h = lrn(nn::max_pool2d(out.stem[0], 3, 2, nn::Padding::Same));
  conv.stride = 1;
  out.stem[1] = nn::conv2d(p, "stem/reduce", h, c1, 1, conv);
  out.stem[2] = nn::conv2d(p, "stem/conv3", out.stem[1], c2, 3, conv);
  h = nn::max_pool2d(lrn(out.stem[2]), 3, 2, nn::Padding::Same);
  const std::array<std::string, 9> names = {"3a", "3b", "4a", "4b", "4c",
                                            "4d", "4e", "5a", "5b"};
  for (int i = 0; i < 9; ++i) {
    const auto widths =
        GoogLeNetWidths()[static_cast<std::size_t>(i)].Divided(cfg.width_divisor);
    out.inception[static_cast<std::size_t>(i)] =
        InceptionV1(p, "inception" + names[static_cast<std::size_t>(i)], h, widths)
            .concatenated;
    h = out.inception[static_cast<std::size_t>(i)];
    if (auxiliary && (i == 2 || i == 5))
      out.auxiliary.push_back(
          GoogLeNetAuxiliary(p, i == 2 ? "aux1" : "aux2", h, cfg, training));
    if (i == 1 || i == 6)
      h = nn::max_pool2d(h, 3, 2, nn::Padding::Same);
  }
  h = nn::global_avg_pool2d(h);
  if (training)
    h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  out.logits = nn::dense(p, "output", h, cfg.classes);
  return out;
}
} // namespace datamunge::nn::models
