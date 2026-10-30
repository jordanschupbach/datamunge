#pragma once
/// @file multilabel_models.hpp
/// @brief Small-stem ResNet18 independent sigmoid objectives and tensor annotation/decision previews.
#include "datamunge/nn/multilabel_metrics.hpp"
#include "datamunge/nn/resnet_models.hpp"
#include "datamunge/nn/siamese_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Compute positive-bit weighted mean stable Bernoulli logit cross entropy.
///
/// Each class has an independent sigmoid;there is no softmax competition.
/// The denominator is N*C,not summed class weight. Stable sign-selected BCE
/// gives derivative weight_factor*(sigmoid(logit)-target)/(N*C),including
/// exact zero logits. Targets/class weights are detached fitting metadata.
/// No optimizer,weight fitting,threshold calibration or model forward is hidden.
///
/// @param logits Float32/64[N,C] finite independent logits,N/C positive.
/// @param target Same-shaped numeric/bool binary presence targets.
/// @param positive_weights Floating[C] finite positive class weights,converted to logit dtype.
/// @return Same-logit-dtype scalar mean BCE*(1+(weight-1)*target),with logit gradients.
/// @throws std::invalid_argument For invalid binary targets,geometry,dtypes,positive weights,threshold or divisor.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto logits = n::zeros({2, 4}).requires_grad_();
/// const auto labels = n::tensor({1., 0., 1., 0., 0., 1., 0., 0.}, {2, 4});
/// const auto loss = n::models::MultilabelObjective(logits, labels, n::tensor({2., 3., 4., 5.}));
/// const auto gradient = n::grad(loss, {logits});
/// @endcode
/// @par References
/// - He et al.(2016),Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385;positive-bit weighted Bernoulli CE follows the explicit source teaching objective. Stable logit BCE semantics https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html. Compact80 independent COCO-presence heads and fit-only weight caps are teaching choices.
/// @see MultilabelConfig(), MultilabelObjective(), ScoreMultilabel()
inline nn::Tensor MultilabelObjective(const nn::Tensor &logits, const nn::Tensor &target,
                                      const nn::Tensor &positive_weights) {
  multilabel_data_detail::ValidateLabels(target);
  if (logits.shape() != target.shape() || logits.numel() < 1 ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      positive_weights.shape() != nn::Shape{logits.shape()[1]} ||
      (positive_weights.dtype() != nn::DType::Float32 &&
       positive_weights.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(logits), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      !nn::all(nn::logical_and(
                   nn::greater(positive_weights, 0.),
                   nn::less(positive_weights, std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid weighted multilabel logits or class weights");
  const auto truth = target.detach().to(logits.dtype()),
             weights = positive_weights.detach().to(logits.dtype());
  return nn::mean(SiameseBinaryCrossEntropy(logits, truth) * (1. + (weights - 1.) * truth));
}
/// @brief Configure a small-stem ResNet18 with80 independent class logits.
///
/// Retains all eight residual blocks,global spatial mean and a biased80-logit
/// head. BN old-value retention.9,epsilon inherited from ResNet. No optimizer
/// or parameters are created by configuration. Caller subtracts.5 from RGB,
/// controls training/update_stats,and checkpoints learned and BN stores together.
///
/// @param divisor Positive exact divisor of64,default8;1 gives full widths.
/// @return Validated ResNetConfig:depth18,classes80,small stem,width divisor requested.
/// @throws std::invalid_argument For invalid binary targets,geometry,dtypes,positive weights,threshold or divisor.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(251), state(257);
/// const auto cfg = n::models::MultilabelConfig();
/// const auto out = n::models::ResNet(p, state, n::zeros({2, 64, 64, 3}), cfg, true);
/// @endcode
/// @par References
/// - He et al.(2016),Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385;positive-bit weighted Bernoulli CE follows the explicit source teaching objective. Stable logit BCE semantics https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html. Compact80 independent COCO-presence heads and fit-only weight caps are teaching choices.
/// @see MultilabelConfig(), MultilabelObjective(), ScoreMultilabel()
inline ResNetConfig MultilabelConfig(int divisor = 8) {
  ResNetConfig cfg;
  cfg.depth = 18;
  cfg.classes = 80;
  cfg.width_divisor = divisor;
  cfg.small_stem = true;
  cfg.Validate();
  return cfg;
}
/// @brief Render observed images,annotation bits and threshold decisions through tensor masks.
///
/// Each row is64px image,8x10 annotation bitmap,8x10 decision bitmap,with
/// black gaps. Label index grows row-major0..79. Truth cells are bright/dark
/// green;decisions are green TP,red FP,yellow FN,gray TN. Six of eight cell
/// rows and three of four cell columns are painted. Byte rounding promotes
/// the Float32 product before adding.5,matching source lround even just below
/// a half-byte boundary. No host pixel loops,forward,training or state mutation.
///
/// @param data Valid Float32 RGB[N,64,64,3],binary targets[N,80],Int32 IDs[N].
/// @param probability Float32/64[N,80] finite independent probabilities[0,1].
/// @param threshold Finite global positive cutoff,default.5;comparison uses Float64.
/// @return Detached Float32[N,64,160,3] RGB[0,1] image/truth/decision tiles.
/// @throws std::invalid_argument For invalid binary targets,geometry,dtypes,positive weights,threshold or divisor.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::MultilabelData data{n::zeros({2, 64, 64, 3}), n::zeros({2, 80}),
///                                n::arange(1, 3, 1, n::DType::Int32)};
/// const auto tiles = n::models::MultilabelPreviewTiles(data, n::full({2, 80}, .1), .5);
/// @endcode
/// @par References
/// - He et al.(2016),Deep Residual Learning for Image Recognition, https://arxiv.org/abs/1512.03385;positive-bit weighted Bernoulli CE follows the explicit source teaching objective. Stable logit BCE semantics https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html. Compact80 independent COCO-presence heads and fit-only weight caps are teaching choices.
/// @see MultilabelConfig(), MultilabelObjective(), ScoreMultilabel()
inline nn::Tensor MultilabelPreviewTiles(const MultilabelData &data,
                                         const nn::Tensor &probability,
                                         double threshold = .5) {
  nn::NoGradGuard guard;
  multilabel_data_detail::ValidateData(data);
  multilabel_metrics_detail::Validate(probability, data.labels);
  if (!std::isfinite(threshold))
    throw std::invalid_argument("finite multilabel preview threshold required");
  const auto n = data.rgb.shape()[0];
  const auto truth = nn::resize_nearest(data.labels.reshape({n, 8, 10, 1}), {64, 40});
  const auto decisions =
      nn::resize_nearest(nn::greater_equal(probability.to(nn::DType::Float64), threshold)
                             .to(nn::DType::Float32)
                             .reshape({n, 8, 10, 1}),
                         {64, 40});
  const auto t = nn::equal(truth, 1.), p = nn::equal(decisions, 1.);
  const auto x = nn::arange(0, 40).reshape({1, 1, 40, 1}),
             y = nn::arange(0, 64).reshape({1, 64, 1, 1});
  const auto dx = x - nn::floor(x / 4.) * 4., dy = y - nn::floor(y / 8.) * 8.;
  const auto inside = nn::logical_and(
      nn::less(dx, 3.), nn::logical_and(nn::greater_equal(dy, 1.), nn::less(dy, 7.)));
  const auto green = nn::tensor({0., 1., 0.}).reshape({1, 1, 1, 3}),
             dark_green = nn::tensor({0., 35. / 255., 0.}).reshape({1, 1, 1, 3}),
             red = nn::tensor({1., 0., 0.}).reshape({1, 1, 1, 3}),
             yellow = nn::tensor({1., 1., 0.}).reshape({1, 1, 1, 3});
  const auto gray = nn::full({1, 1, 1, 3}, 20. / 255.);
  const auto truth_colour = nn::where(t, green, dark_green);
  const auto predicted_colour =
      nn::where(t, nn::where(p, green, yellow), nn::where(p, red, gray));
  const auto zero = nn::zeros({n, 64, 40, 3});
  const auto true_map = nn::where(inside, truth_colour, zero),
             predicted_map = nn::where(inside, predicted_colour, zero);
  const auto image =
      nn::floor((data.rgb * 255.).to(nn::DType::Float64) + .5).to(nn::DType::Float32) / 255.;
  return nn::concat({image, nn::zeros({n, 64, 4, 3}), true_map, nn::zeros({n, 64, 8, 3}),
                     predicted_map, nn::zeros({n, 64, 4, 3})},
                    2);
}
} // namespace datamunge::nn::models
