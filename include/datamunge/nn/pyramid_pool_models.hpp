#pragma once
/// @file pyramid_pool_models.hpp
/// @brief Exact adaptive average bins and PSPNet dense-ResNet pyramid context with auxiliary scores.
#include "datamunge/nn/deeplab_models.hpp"
#include "datamunge/nn/vision_architecture_components.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Average exact floor-start/ceil-end adaptive spatial bins.
///
/// Bin starts floor(i*H/bins),ends ceil((i+1)*H/bins). Odd-grid bins
/// overlap;repeated pixels accumulate repeated gradients. Bins1 is global mean.
/// No parameters,optimizer,randomness or mutation.
///
/// @param input Positive floating NHWC tensor.
/// @param bins Positive output bins per spatial dimension,including bins larger than input axes.
/// @return Same-dtype[N,bins,bins,C] adaptive means,with exact source gradients.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::AdaptiveAveragePool(n::arange(0, 35).reshape({1, 5, 7, 1}), 3);
/// @endcode
/// @par References
/// - Zhao,Shi,Qi,Wang and Jia(2017), Pyramid Scene Parsing Network,CVPR, https://arxiv.org/abs/1612.01105.
/// @see PSPNet()
inline nn::Tensor AdaptiveAveragePool(const nn::Tensor &input, int bins) {
  if (bins < 1 || input.dim() != 4 || input.shape()[0] < 1 || input.shape()[1] < 1 ||
      input.shape()[2] < 1 || input.shape()[3] < 1 ||
      (input.dtype() != nn::DType::Float32 && input.dtype() != nn::DType::Float64))
    throw std::invalid_argument("positive pyramid bin count required");
  if (bins == 1)
    return nn::mean(input, {1, 2}, true);
  const auto &s = input.shape();
  const std::int64_t h = s[1], w = s[2];
  std::vector<nn::Tensor> rows;
  for (int y = 0; y < bins; ++y) {
    std::vector<nn::Tensor> columns;
    for (int x = 0; x < bins; ++x) {
      const std::int64_t y0 = h * y / bins, y1 = (h * (y + 1) + bins - 1) / bins;
      const std::int64_t x0 = w * x / bins, x1 = (w * (x + 1) + bins - 1) / bins;
      columns.push_back(nn::mean(
          nn::slice(input, {0, y0, x0, 0}, {s[0], y1 - y0, x1 - x0, s[3]}), {1, 2}, true));
    }
    rows.push_back(nn::concat(columns, 2));
  }
  return nn::concat(rows, 1);
}

/// @brief Configure output-stride8 ResNet PSPNet and optional auxiliary classifier.
///
/// Full defaults ResNet101,21 classes,pyramid/head512,auxiliary256,keep.9.
/// No pretrained weights or hidden optimizer.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PSPNetConfig c;
/// c.classes = 4;
/// c.depth = 18;
/// c.width_divisor = 8;
/// c.pool_width = 16;
/// c.head_width = 32;
/// c.aux_width = 16;
/// c.keep_probability = 1;
/// const auto net = n::models::PSPNet(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Zhao,Shi,Qi,Wang and Jia(2017), Pyramid Scene Parsing Network,CVPR, https://arxiv.org/abs/1612.01105.
/// @see
struct PSPNetConfig {
  int classes = 21;             ///< At least two classes.
  int depth = 101;              ///< Supported ResNet depth18/34/50/101/152.
  int width_divisor = 1;        ///< Positive exact divisor of64 for backbone widths.
  int pool_width = 512;         ///< Positive per-pyramid projection width.
  int head_width = 512;         ///< Positive fused3x3 head width.
  int aux_width = 256;          ///< Positive auxiliary3x3 head width.
  float keep_probability = .9f; ///< Training head/auxiliary dropout keep probability in(0,1].
  bool auxiliary = true;        ///< Create auxiliary classifier on C4 when true(default).
};
/// @brief Own PSPNet context pools,coarse/final logits and optional auxiliary scores.
///
/// Returned graphs borrow parameters;moving BN state belongs to caller stores.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PSPNetConfig c;
/// c.classes = 4;
/// c.depth = 18;
/// c.width_divisor = 8;
/// c.pool_width = 16;
/// c.head_width = 32;
/// c.aux_width = 16;
/// c.keep_probability = 1;
/// const auto net = n::models::PSPNet(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Zhao,Shi,Qi,Wang and Jia(2017), Pyramid Scene Parsing Network,CVPR, https://arxiv.org/abs/1612.01105.
/// @see
struct PSPNetOutputs {
  nn::Tensor features;  ///< Deepest output-stride8 dense ResNet feature map.
  nn::Tensor coarse;    ///< Fused class logits before final resize.
  nn::Tensor logits;    ///< NHWC logits at exact input H/W.
  nn::Tensor auxiliary; ///< Resized C4 auxiliary logits;undefined if disabled.
  std::vector<nn::Tensor>
      pools; ///< Raw adaptive means at1/2/3/6 bins,before normalization/projection.
};
/// @brief Build dense ResNet PSPNet with1/2/3/6 pyramid pools and optional auxiliary logits.
///
/// Pyramid projections use no-bias BN/gamma/ReLU,then aligned-corner
/// resize to the deepest map. Concatenation includes original features;3x3
/// fusion and a linear classifier restore input logits. Auxiliary C4 scores
/// need explicit supervision to fit all leaves. BN epsilon1e-5/momentum.9.
/// Full101/21 count67949930,state110976,334 learned tensors with auxiliary.
/// No optimizer or pretrained weights. Warm all leaves before optimizer
/// creation,and checkpoint learned/moving stores together.
///
/// @param p Same-dtype learned store,creates/reuses encoder,pyramid,head and optional auxiliary leaves.
/// @param state Same-dtype BN moving moments,updated during training.
/// @param images Positive floating RGB NHWC matching stores.
/// @param cfg Valid depth,width,class,pyramid,head and dropout settings.
/// @param training False(default) moving moments/no dropout;true batch moments/updates/dropout.
/// @return Dense features,four adaptive pools,coarse/final and optional auxiliary logits.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::PSPNetConfig c;
/// c.classes = 4;
/// c.depth = 18;
/// c.width_divisor = 8;
/// c.pool_width = 16;
/// c.head_width = 32;
/// c.aux_width = 16;
/// c.keep_probability = 1;
/// const auto net = n::models::PSPNet(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Zhao,Shi,Qi,Wang and Jia(2017), Pyramid Scene Parsing Network,CVPR, https://arxiv.org/abs/1612.01105.
/// @see AdaptiveAveragePool(), DenseResNet()
inline PSPNetOutputs PSPNet(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &images, const PSPNetConfig &cfg = {},
                            bool training = false) {
  if (cfg.classes < 2 || cfg.pool_width < 1 || cfg.head_width < 1 || cfg.aux_width < 1 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid PSPNet configuration");
  ResNetConfig backbone;
  backbone.depth = cfg.depth;
  backbone.width_divisor = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.stride_on_3x3 = true;
  const auto net = DenseResNet(p, state, images, backbone, 8, training);
  PSPNetOutputs r;
  r.features = net.stages[3];
  ConvNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.momentum = .9;
  norm.epsilon = 1e-5;
  const nn::ImageSize feature_size{r.features.shape()[1], r.features.shape()[2]};
  const nn::ImageSize image_size{images.shape()[1], images.shape()[2]};
  std::vector<nn::Tensor> branches = {r.features};
  for (int bins : {1, 2, 3, 6}) {
    const auto pool = AdaptiveAveragePool(r.features, bins);
    r.pools.push_back(pool);
    const auto project = NormalizedConv(p, state, "pyramid/bin" + std::to_string(bins), pool,
                                        cfg.pool_width, 1, 1, norm);
    branches.push_back(nn::resize_bilinear(project, feature_size, true));
  }
  nn::Tensor h =
      NormalizedConv(p, state, "head", nn::concat(branches, 3), cfg.head_width, 3, 3, norm);
  if (training && cfg.keep_probability < 1)
    h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  nn::ConvLayerOptions score; // 1x1, bias, Glorot uniform
  r.coarse = nn::conv2d(p, "classifier", h, cfg.classes, 1, score);
  r.logits = nn::resize_bilinear(r.coarse, image_size, true);
  if (cfg.auxiliary) {
    nn::Tensor aux =
        NormalizedConv(p, state, "aux/conv", net.stages[2], cfg.aux_width, 3, 3, norm);
    if (training && cfg.keep_probability < 1)
      aux = nn::dropout(aux, 1 - cfg.keep_probability, true, p.generator());
    aux = nn::conv2d(p, "aux/classifier", aux, cfg.classes, 1, score);
    r.auxiliary = nn::resize_bilinear(aux, image_size, true);
  }
  return r;
}

} // namespace datamunge::nn::models
