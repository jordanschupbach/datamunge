// EfficientDet D0-D7: EfficientNet, repeated weighted BiFPN and shared
// separable detection subnets with per-level batch-normalization statistics
// (port of tflow's common/nn/efficientdet.h).
// https://arxiv.org/abs/1911.09070
// https://github.com/google/automl/tree/master/efficientdet
// Full defaults use D0/80 classes. The backbone stops before the
// classification head and emits stride8/16/32 endpoints. A projected/
// downsampled C5 creates P6; pooled P6 creates P7. Independent projection per
// first-cell edge follows the reference graph. Head convolutions share across
// levels; head BN does not. Each prediction has nine Retina-style anchors
// (outputs in retinanet::RetinaNetOutputs). No pretrained weights. Backbone
// BN momentum .99 / eps 1e-3; drop connect is off (rate 0), as in tflow.
#pragma once

#include "common/bifpn.hpp"
#include "common/mbconv.hpp"
#include "common/retinanet.hpp"

namespace efficientdet {

namespace nn = datamunge::nn;

struct EfficientDetSpec {
  int backbone, width, repeats, head_repeats, resolution;
};
inline EfficientDetSpec EfficientDetScaling(int variant) {
  static const EfficientDetSpec table[] = {{0, 64, 3, 3, 512},  {1, 88, 4, 3, 640},  {2, 112, 5, 3, 768},
                                           {3, 160, 6, 4, 896}, {4, 224, 7, 4, 1024}, {5, 288, 7, 4, 1280},
                                           {6, 384, 8, 5, 1280}, {6, 384, 8, 5, 1536}};
  if (variant < 0 || variant > 7) throw std::invalid_argument("EfficientDet variants are D0-D7");
  return table[variant];
}
struct EfficientDetConfig {
  int    variant = 0, classes = 80, image_size = 0, width = 0, repeats = 0, head_repeats = 0;
  double backbone_multiplier = 1.;
  bool   grayscale = false;
  bool   update_stats = true;
};
inline retinanet::RetinaNetOutputs EfficientDet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                                const EfficientDetConfig& cfg = {}, bool training = false) {
  const auto spec    = EfficientDetScaling(cfg.variant);
  const int  size    = cfg.image_size ? cfg.image_size : spec.resolution, width = cfg.width ? cfg.width : spec.width,
            repeats = cfg.repeats ? cfg.repeats : spec.repeats, heads = cfg.head_repeats ? cfg.head_repeats : spec.head_repeats;
  if (size < 32 || size % 32 || width < 1 || repeats < 1 || heads < 1 || cfg.classes < 1 || !(cfg.backbone_multiplier > 0))
    throw std::invalid_argument("invalid EfficientDet configuration");
  retinanet::RetinaNetOutputs     result;
  std::vector<mobilenet::MobileLayer> costs;
  mobilenet::MobileContext        context{p, state, costs, training, .99, cfg.update_stats};
  const auto                      scale  = mbconv::EfficientNetCoefficients(spec.backbone);
  const auto                      blocks = mbconv::EfficientNetBlocks(spec.backbone);
  int        index = 0, stage = 0;
  const auto rgb   = cfg.grayscale ? nn::tile(image, {1, 1, 1, 3}) : image;
  nn::Tensor h     = mbconv::Swish(mbconv::ConvBN(context, "backbone/stem", rgb, mbconv::RoundFilters(32, scale.width * cfg.backbone_multiplier),
                                                  3, 2, nn::Activation::None));
  for (const auto& b : blocks) {
    const int out = mbconv::RoundFilters(b.output, cfg.backbone_multiplier);
    for (int r = 0; r < b.repeats; ++r, ++index)
      h = mbconv::MBConv(context, "backbone/block" + std::to_string(index + 1), h, out, b, r ? 1 : b.stride, true, 0).output;
    if (stage == 2 || stage == 4 || stage == 6) result.features.push_back(h);
    ++stage;
  }
  nn::ConvLayerOptions projection;
  projection.init = nn::init::he_uniform();
  h = bifpn::DetBatchNorm(p, state, "p6/bn", nn::conv2d(p, "p6/project", h, width, 1, projection), training, cfg.update_stats);
  h = nn::max_pool2d(h, 3, 2, nn::Padding::Same);
  result.features.push_back(h);
  result.features.push_back(nn::max_pool2d(h, 3, 2, nn::Padding::Same));
  for (int cell = 0; cell < repeats; ++cell)
    result.features = bifpn::BiFpnCell(p, state, "bifpn" + std::to_string(cell), result.features, width, training, cfg.update_stats);
  std::vector<nn::Tensor> logits, deltas;
  for (int level = 0; level < 5; ++level) {
    const int stride = 8 << level, cells = (size + stride - 1) / stride, count = cells * cells * 9;
    if (result.features[std::size_t(level)].shape()[1] != cells)
      throw std::invalid_argument("EfficientDet image_size does not match the input images");
    result.cells.push_back(cells);
    result.anchors += count;
    nn::Tensor classification = result.features[std::size_t(level)], regression = classification;
    for (int layer = 0; layer < heads; ++layer)
      for (int branch = 0; branch < 2; ++branch) {
        auto&      feature = branch ? regression : classification;
        const auto name    = std::string(branch ? "regression/" : "classification/") + "layer" + std::to_string(layer);
        feature            = bifpn::DetSeparable(p, name, feature, width);
        feature = bifpn::Swish(bifpn::DetBatchNorm(p, state, name + "/bn_level" + std::to_string(level), feature, training, cfg.update_stats));
      }
    logits.push_back(nn::reshape(bifpn::DetSeparable(p, "classification/output", classification, 9 * cfg.classes, std::log(.01 / .99)),
                                 {-1, count, cfg.classes}));
    deltas.push_back(nn::reshape(bifpn::DetSeparable(p, "regression/output", regression, 36), {-1, count, 4}));
  }
  result.logits        = nn::concat(logits, 1);
  result.deltas        = nn::concat(deltas, 1);
  result.probabilities = nn::sigmoid(result.logits);
  return result;
}

}  // namespace efficientdet
