#pragma once
/// @file efficientdet_models.hpp
/// @brief Tensor EfficientDet D0–D7 scaling metadata and feature-only EfficientNet/BiFPN detector.
#include "datamunge/nn/bifpn_models.hpp"
#include "datamunge/nn/retina_detection_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Retain EfficientDet compound backbone/pyramid/head/resolution metadata.
///
/// Integer architecture metadata, independent of learnable weights or state.
/// D7 retains the source D6 backbone/width/repeats with1536-pixel resolution.
///
/// @par Example
/// @code
/// const auto d0 = dm::nn::models::EfficientDetScaling(0);
/// @endcode
/// @par References
/// - Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070
/// @see EfficientDetScaling()
struct EfficientDetSpec {
  int backbone;     ///< EfficientNet backbone variant0–6.
  int width;        ///< Shared BiFPN/head width.
  int repeats;      ///< Number of independent BiFPN cells.
  int head_repeats; ///< Shared separable layers per classification/regression tower.
  int resolution;   ///< Default square image side.
};
/// @brief Return source compound scaling settings for D0 through D7.
///
/// Constant table, no allocation of learned models, state mutation or fitting.
///
/// @param variant Integer0–7 inclusive.
/// @return Value metadata for EfficientNet backbone, width, BiFPN/head depth and image side.
/// @throws std::invalid_argument For a variant outside0–7.
/// @par Example
/// @code
/// const auto d3 = dm::nn::models::EfficientDetScaling(3);
/// @endcode
/// @par References
/// - Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070
/// @see EfficientDet()
inline EfficientDetSpec EfficientDetScaling(int variant) {
  /// Source compound metadata; D7 uses the same learned geometry as D6.
  static const EfficientDetSpec table[] = {
      {0, 64, 3, 3, 512},   {1, 88, 4, 3, 640},   {2, 112, 5, 3, 768},  {3, 160, 6, 4, 896},
      {4, 224, 7, 4, 1024}, {5, 288, 7, 4, 1280}, {6, 384, 8, 5, 1280}, {6, 384, 8, 5, 1536}};
  if (variant < 0 || variant > 7)
    throw std::invalid_argument("EfficientDet variants are D0-D7");
  return table[variant];
}
/// @brief Configure EfficientDet source variant and explicit compact geometry overrides.
///
/// Zero geometry overrides select compound defaults. No pretrained weights or
/// fitting; separate BN state must be saved with learned weights.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::EfficientDetConfig c;
/// c.classes = 10;
/// c.image_size = 64;
/// c.width = 32;
/// c.repeats = 2;
/// c.head_repeats = 2;
/// c.backbone_multiplier = .25;
/// c.grayscale = true;
/// const auto net = n::models::EfficientDet(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070
/// @see EfficientDet()
struct EfficientDetConfig {
  int variant = 0;    ///< D0–D7 metadata index.
  int classes = 80;   ///< Positive independent foreground sigmoid channels.
  int image_size = 0; ///< Zero uses default resolution, else positive side>=32 divisible by32.
  int width = 0;      ///< Zero uses default BiFPN/head width, else positive width.
  int repeats =
      0; ///< Zero uses default number of independent BiFPN cells, else positive count.
  int head_repeats = 0; ///< Zero uses default shared tower depth, else positive count.
  double backbone_multiplier =
      1.;                 ///< Finite positive multiplier on compound-rounded backbone widths.
  bool grayscale = false; ///< Tile one channel to RGB before EfficientNet.
  bool update_stats = true; ///< Update all BN moving state only in training when true.
};
/// @brief Build feature-only EfficientNet, repeated weighted BiFPN and shared separable detection towers.
///
/// The backbone omits its classifier and emits stride8/16/32. Projected C5
/// creates P6; pooling creates P7. Head kernels share across levels but head BN
/// prefixes/state are per-level. Momentum.99/eps.001; drop-connect rate zero.
/// Full D0/80 has3874217 learned/47136 state scalars and493 learned tensors.
/// Compact D0x.25,width32,two cells/two head layers,10 classes has342038/13568.
/// No optimizer or pretraining is hidden; keep leaves unchanged before backward.
///
/// @param p Mutable Float32/64 learned store; warm all leaves before optimizer creation.
/// @param state Mutable same-dtype BN moving-state store; checkpoint with p.
/// @param image Positive floating NHWC with square resolved size and1/3 channels per cfg.
/// @param cfg Source D0–D7 settings or explicit compact overrides.
/// @param training False(default) uses moving BN state; true uses batch stats and optional updates.
/// @return Five-level RetinaNetOutputs with nine independent sigmoid foreground predictions per cell.
/// @throws std::invalid_argument For invalid cfg/input/dtypes or incompatible store leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::EfficientDetConfig c;
/// c.classes = 10;
/// c.image_size = 64;
/// c.width = 32;
/// c.repeats = 2;
/// c.head_repeats = 2;
/// c.backbone_multiplier = .25;
/// c.grayscale = true;
/// const auto net = n::models::EfficientDet(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070
/// @see RetinaAnchors(), RetinaLoss(), DecodeRetinaDetections()
inline RetinaNetOutputs EfficientDet(nn::ParameterStore &p, nn::ParameterStore &state,
                                     const nn::Tensor &image,
                                     const EfficientDetConfig &cfg = {},
                                     bool training = false) {
  const auto spec = EfficientDetScaling(cfg.variant);
  const int size = cfg.image_size ? cfg.image_size : spec.resolution,
            width = cfg.width ? cfg.width : spec.width,
            repeats = cfg.repeats ? cfg.repeats : spec.repeats,
            heads = cfg.head_repeats ? cfg.head_repeats : spec.head_repeats;
  if (size < 32 || size % 32 || width < 1 || repeats < 1 || heads < 1 || cfg.classes < 1 ||
      (!(cfg.backbone_multiplier > 0) || !std::isfinite(cfg.backbone_multiplier)) ||
      cfg.classes > std::numeric_limits<int>::max() / 9 || image.dim() != 4 ||
      image.shape()[0] < 1 || image.shape()[1] != size || image.shape()[2] != size ||
      image.shape()[3] != (cfg.grayscale ? 1 : 3) || image.dtype() != p.dtype() ||
      state.dtype() != p.dtype())
    throw std::invalid_argument("invalid EfficientDet configuration");
  RetinaNetOutputs result;
  std::vector<MobileLayer> costs;
  MobileContext context{p, state, costs, training, .99, cfg.update_stats};
  const auto scale = EfficientNetCoefficients(spec.backbone);
  const auto blocks = EfficientNetBlocks(spec.backbone);
  int index = 0, stage = 0;
  const auto rgb = cfg.grayscale ? nn::tile(image, {1, 1, 1, 3}) : image;
  nn::Tensor h = Swish(ConvBN(context, "backbone/stem", rgb,
                              RoundFilters(32, scale.width * cfg.backbone_multiplier), 3, 2,
                              nn::Activation::None));
  for (const auto &b : blocks) {
    const int out = RoundFilters(b.output, cfg.backbone_multiplier);
    for (int r = 0; r < b.repeats; ++r, ++index)
      h = MBConv(context, "backbone/block" + std::to_string(index + 1), h, out, b,
                 r ? 1 : b.stride, true, 0)
              .output;
    if (stage == 2 || stage == 4 || stage == 6)
      result.features.push_back(h);
    ++stage;
  }
  nn::ConvLayerOptions projection;
  projection.init = nn::init::he_uniform();
  h = DetBatchNorm(p, state, "p6/bn", nn::conv2d(p, "p6/project", h, width, 1, projection),
                   training, cfg.update_stats);
  h = nn::max_pool2d(h, 3, 2, nn::Padding::Same);
  result.features.push_back(h);
  result.features.push_back(nn::max_pool2d(h, 3, 2, nn::Padding::Same));
  for (int cell = 0; cell < repeats; ++cell)
    result.features = BiFpnCell(p, state, "bifpn" + std::to_string(cell), result.features,
                                width, training, cfg.update_stats);
  std::vector<nn::Tensor> logits, deltas;
  for (int level = 0; level < 5; ++level) {
    const int stride = 8 << level, cells = (size + stride - 1) / stride,
              count = cells * cells * 9;
    if (result.features[std::size_t(level)].shape()[1] != cells)
      throw std::invalid_argument("EfficientDet image_size does not match the input images");
    result.cells.push_back(cells);
    result.anchors += count;
    nn::Tensor classification = result.features[std::size_t(level)],
               regression = classification;
    for (int layer = 0; layer < heads; ++layer)
      for (int branch = 0; branch < 2; ++branch) {
        auto &feature = branch ? regression : classification;
        const auto name = std::string(branch ? "regression/" : "classification/") + "layer" +
                          std::to_string(layer);
        feature = DetSeparable(p, name, feature, width);
        feature = Swish(DetBatchNorm(p, state, name + "/bn_level" + std::to_string(level),
                                     feature, training, cfg.update_stats));
      }
    logits.push_back(nn::reshape(DetSeparable(p, "classification/output", classification,
                                              9 * cfg.classes, std::log(.01 / .99)),
                                 {-1, count, cfg.classes}));
    deltas.push_back(
        nn::reshape(DetSeparable(p, "regression/output", regression, 36), {-1, count, 4}));
  }
  result.logits = nn::concat(logits, 1);
  result.deltas = nn::concat(deltas, 1);
  result.probabilities = nn::sigmoid(result.logits);
  return result;
}

} // namespace datamunge::nn::models
