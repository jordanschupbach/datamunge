#pragma once
/// @file depth_models.hpp
/// @brief RGB-only tensor coarse/fine metric-depth prediction and safe masked log/neighbor objectives.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <array>
#include <cmath>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Convert raw logits to positive metric depth without display clipping.
///
/// Finite-precision output can round to the.1 floor at very negative logits.
/// Derivative is sigmoid(logits);large positive logits remain unbounded. No
/// optimizer,GT scale alignment,normalization or hidden clamp.
///
/// @param logits Floating raw depth logits.
/// @return Matching-shape.1+softplus(logits),meters,graph retained.
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto depth = n::models::PositiveMetricDepth(n::tensor({-80., 0., 80.}));
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
inline nn::Tensor PositiveMetricDepth(const nn::Tensor &logits) {
  return 0.1 + nn::softplus(logits);
}

/// @brief Own coarse/refined positive depth and every model stage.
///
/// Tensor handles preserve the graph. No GT or validity mask enters forward.
///
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(139);
/// const auto r = n::models::MonocularDepth(p, n::zeros({1, 48, 64, 3}));
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
struct MonocularDepthResult {
  nn::Tensor depth;  ///< Refined positive floating[N,48,64,1] metric depth,meters.
  nn::Tensor coarse; ///< Upsampled positive global-coarse metric depth,same geometry.
  std::vector<nn::Tensor>
      stages; ///< Eleven encoder/global/coarse/local/refined stage activations.
};

/// @brief Predict coarse and locally refined metric depth from RGB alone.
///
/// Three stride2 encoders feed a128-unit global dense map and12x16 coarse
/// logit field. Half-pixel bilinear upsampling feeds the RGB local refinement;
/// positive heads are.1+softplus(coarse) and.1+softplus(coarse+residual).
/// 231745 learned scalars/16 tensors,no BN,targets or hidden fitting. The global
/// map sees all observed pixels. This is an inspired teaching baseline,not Eigen
/// pretrained weights or the original backbone.
///
/// @param p Mutable learned store;warm all16 leaves before optimizer construction.
/// @param rgb Floating[N,48,64,3] observed RGB[0,1],positive N,matching p dtype.
/// @return Positive metric depth/coarse[N,48,64,1] and eleven stages.
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(139);
/// const auto r = n::models::MonocularDepth(p, n::zeros({2, 48, 64, 3}));
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
inline MonocularDepthResult MonocularDepth(nn::ParameterStore &p, const nn::Tensor &rgb) {
  if (rgb.dim() != 4 || rgb.size(0) < 1 || rgb.shape() != nn::Shape{rgb.size(0), 48, 64, 3} ||
      rgb.dtype() != p.dtype())
    throw std::invalid_argument("positive48x64 RGB-only depth input required");
  MonocularDepthResult result;
  nn::ConvLayerOptions o;
  o.activation = nn::Activation::Relu;
  o.stride = 2;
  auto x = nn::conv2d(p, "encoder1", rgb, 8, 5, o);
  result.stages.push_back(x);
  x = nn::conv2d(p, "encoder2", x, 16, 3, o);
  result.stages.push_back(x);
  x = nn::conv2d(p, "encoder3", x, 32, 3, o);
  result.stages.push_back(x);
  x = nn::dense(p, "global", nn::reshape(x, {-1, 6 * 8 * 32}), 128, nn::Activation::Relu);
  result.stages.push_back(x);
  x = nn::dense(p, "coarse_map", x, 12 * 16);
  result.stages.push_back(x);
  auto coarse = nn::reshape(x, {-1, 12, 16, 1});
  result.stages.push_back(coarse);
  auto up = nn::resize_bilinear(coarse, {48, 64}, false, true);
  result.stages.push_back(up);
  o.stride = 1;
  auto local = nn::conv2d(p, "local1", rgb, 16, 5, o);
  result.stages.push_back(local);
  local = nn::conv2d(p, "local2", nn::concat({local, up}, 3), 16, 3, o);
  result.stages.push_back(local);
  o.activation = nn::Activation::None;
  auto residual = nn::conv2d(p, "local3", local, 1, 3, o);
  result.stages.push_back(residual);
  result.coarse = PositiveMetricDepth(up);
  result.depth = PositiveMetricDepth(up + residual);
  result.stages.push_back(result.depth);
  return result;
}

/// @brief Own safe masked log residuals and eligible sensor support.
///
/// Invalid samples are selected out before log,so their NaN/zero values and
/// gradients cannot contaminate eligible observations.
///
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto e = n::models::DepthLogError(n::ones({1, 2, 3, 1}), n::ones({1, 2, 3, 1}),
///                                         n::ones({1, 2, 3, 1}));
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
struct DepthError {
  nn::Tensor error; ///< Safe log(prediction)-log(truth),zero at invalid positions.
  nn::Tensor valid; ///< Detached prediction-dtype binary eligible sensor mask.
};
/// @brief Construct safe log-depth residuals on eligible sensor observations.
///
/// Nonfinite/out-of-range target locations select safe prediction/truth1
/// BEFORE logs. At eligible points,predictions must be positive/finite;else
/// throw. Invalid predictions may be NaN because they are never logged.
///
/// @param prediction Floating positive scalar[N,H,W,1] estimates,finite where eligible.
/// @param truth Matching metric sensor targets;valid only if finite,>.1f and<=10m.
/// @param mask Same-shape numeric support,included when>.5.
/// @return Log residual and detached prediction-dtype binary validity.
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto e = n::models::DepthLogError(n::ones({1, 2, 3, 1}), n::ones({1, 2, 3, 1}),
///                                         n::ones({1, 2, 3, 1}));
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
inline DepthError DepthLogError(const nn::Tensor &prediction, const nn::Tensor &truth,
                                const nn::Tensor &mask) {
  if (prediction.dim() != 4 || prediction.size(0) < 1 || prediction.numel() < 1 ||
      prediction.size(3) != 1 || prediction.shape() != truth.shape() ||
      mask.shape() != truth.shape())
    throw std::invalid_argument("aligned scalar NHWC depth supervision required");
  const nn::Tensor valid = nn::logical_and(
      nn::greater(mask.to(nn::DType::Float64), .5),
      nn::logical_and(nn::is_finite(truth), nn::logical_and(nn::greater(truth, double(.1f)),
                                                            nn::less_equal(truth, 10.0))));
  // Select BEFORE Log: invalid NaN/zero targets or predictions cannot contaminate valid values/gradients.
  if (!nn::all(
           nn::where(valid,
                     nn::logical_and(nn::is_finite(prediction), nn::greater(prediction, 0.)),
                     nn::ones_like(prediction).to(nn::DType::Bool)))
           .item<bool>())
    throw std::invalid_argument(
        "positive finite depth estimates required at eligible targets");
  const nn::Tensor safe_truth = nn::where(valid, truth, 1.0),
                   safe_prediction = nn::where(valid, prediction, 1.0);
  return {nn::log(safe_prediction) - nn::log(safe_truth),
          nn::cast(valid.detach(), prediction.dtype())};
}
/// @brief Average per-image values only where support counts are positive.
///
/// Absent values are selected to zero before summing. Images have equal
/// weight,independent of their pixel/neighbor count. No optimizer or state mutation.
///
/// @param values Nonempty floating per-image values.
/// @param counts Matching per-image nonnegative support counts.
/// @return Scalar mean over present images,graph-connected zero if none present.
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto mean =
///     n::models::DepthNonemptyMean(n::tensor({2., 5., 99.}), n::tensor({1., 3., 0.}));
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
inline nn::Tensor DepthNonemptyMean(const nn::Tensor &values, const nn::Tensor &counts) {
  if (values.shape() != counts.shape() || values.numel() < 1)
    throw std::invalid_argument("aligned nonempty depth per-image values/counts required");
  const auto present = nn::greater(counts, 0.);
  return nn::sum(nn::where(present, values, 0.)) /
         nn::maximum(nn::sum(present.to(values.dtype())), nn::scalar_like(1., values));
}
/// @brief Compute equal-image masked scale-sensitive/invariant log-depth loss.
///
/// Lambda0 is log MSE;lambda1 removes a uniform multiplicative scale error.
/// Default.5 retains metric scale supervision. Denominators count true eligible
/// samples;empty images do not dilute the objective and all-empty gradients are0.
///
/// @param prediction Floating[N,H,W,1] metric estimates.
/// @param truth Matching sensor depth with eligibility from DepthLogError.
/// @param mask Matching numeric sensor support.
/// @param lambda Finite coefficient in[0,1],default.5.
/// @return Mean over supported images of mean(e²)-lambda*mean(e)^2.
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::DepthLogLoss(n::ones({2, 3, 4, 1}) * 2., n::ones({2, 3, 4, 1}),
///                                           n::ones({2, 3, 4, 1}), .5);
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
inline nn::Tensor DepthLogLoss(const nn::Tensor &prediction, const nn::Tensor &truth,
                               const nn::Tensor &mask, double lambda = .5) {
  if (!std::isfinite(lambda) || lambda < 0 || lambda > 1)
    throw std::invalid_argument("log-depth lambda in[0,1]");
  const auto e = DepthLogError(prediction, truth, mask);
  const nn::Tensor count = nn::sum(e.valid, {1, 2, 3});
  const nn::Tensor denom = nn::maximum(count, nn::scalar_like(1.0, count));
  const nn::Tensor mean = nn::sum(e.valid * e.error, {1, 2, 3}) / denom;
  const nn::Tensor square = nn::sum(e.valid * nn::square(e.error), {1, 2, 3}) / denom;
  return DepthNonemptyMean(square - lambda * nn::square(mean), count);
}
/// @brief Match adjacent log-depth differences only where both endpoints are valid.
///
/// Counts eligible neighbor pairs,not image area. Mask gaps never introduce
/// artificial edges. Empty neighbor sets yield graph-connected zero. A uniform
/// scale error cancels in spatial log differences. No fitted coefficients.
///
/// @param prediction Floating[N,H,W,1] metric depth.
/// @param truth Matching eligible sensor depth.
/// @param mask Same-shape numeric support.
/// @return Equal-image mean L1 horizontal/vertical log-residual differences over valid pairs.
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::DepthGradientLoss(n::ones({1, 3, 4, 1}), n::ones({1, 3, 4, 1}),
///                                                n::ones({1, 3, 4, 1}));
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
inline nn::Tensor DepthGradientLoss(const nn::Tensor &prediction, const nn::Tensor &truth,
                                    const nn::Tensor &mask) {
  const auto e = DepthLogError(prediction, truth, mask);
  const std::int64_t H = prediction.size(1), W = prediction.size(2);
  /// @brief Slice one spatial neighbor range without copying pixel values.
  /// @param x Observed error or validity tensor.
  /// @param axis Spatial axis1 or2.
  /// @param start Inclusive coordinate.
  /// @param stop Exclusive coordinate,possibly start for empty neighbor sets.
  /// @return Differentiable tensor slice.
  auto cut = [](const nn::Tensor &x, std::int64_t axis, std::int64_t start,
                std::int64_t stop) {
    return nn::slice_axis(x, axis, start, stop);
  };
  const nn::Tensor left = cut(e.error, 2, 0, W - 1), right = cut(e.error, 2, 1, W),
                   top = cut(e.error, 1, 0, H - 1), bottom = cut(e.error, 1, 1, H);
  const nn::Tensor horizontal = cut(e.valid, 2, 0, W - 1) * cut(e.valid, 2, 1, W);
  const nn::Tensor vertical = cut(e.valid, 1, 0, H - 1) * cut(e.valid, 1, 1, H);
  const nn::Tensor numerator = nn::sum(horizontal * nn::abs(left - right), {1, 2, 3}) +
                               nn::sum(vertical * nn::abs(top - bottom), {1, 2, 3});
  const nn::Tensor count = nn::sum(horizontal, {1, 2, 3}) + nn::sum(vertical, {1, 2, 3});
  return DepthNonemptyMean(numerator / nn::maximum(count, nn::scalar_like(1.0, count)), count);
}
/// @brief Combine refined metric-log,valid-neighbor gradient and coarse objectives.
///
/// Both log objectives use lambda.5. Caller owns the explicit optimizer and
/// iteration loop;this helper only computes tensor objectives,no fitting.
///
/// @param result Coarse/refined positive metric predictions.
/// @param truth Matching sensor metric depths.
/// @param mask Matching sensor support.
/// @return Array{log+.1*gradient+.2*coarse,log,gradient,coarse},all graph retained.
/// @throws std::invalid_argument For invalid geometry,coefficient,eligible predictions,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(139);
/// const auto net = n::models::MonocularDepth(p, n::zeros({1, 48, 64, 3}));
/// const auto losses =
///     n::models::DepthObjective(net, n::ones({1, 48, 64, 1}), n::ones({1, 48, 64, 1}));
/// @endcode
/// @par References
/// - Eigen,Puhrsch and Fergus(2014),Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, https://arxiv.org/abs/1406.2283. This is a small from-scratch teaching coarse/fine network,not the original backbone. NYU aligned raw metric depth https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. Lambda.5,neighbor-gradient coefficient.1 and coarse coefficient.2 are source teaching choices.
/// @see MonocularDepth(), DepthObjective(), DepthLogError()
inline std::array<nn::Tensor, 4> DepthObjective(const MonocularDepthResult &result,
                                                const nn::Tensor &truth,
                                                const nn::Tensor &mask) {
  const nn::Tensor log = DepthLogLoss(result.depth, truth, mask),
                   gradient = DepthGradientLoss(result.depth, truth, mask),
                   coarse = DepthLogLoss(result.coarse, truth, mask);
  return {log + .1 * gradient + .2 * coarse, log, gradient, coarse};
}

} // namespace datamunge::nn::models
