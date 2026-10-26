// Per-image masked log-depth errors and valid-neighbor gradient matching: a
// port of tflow's common/ml/depth_losses.h (with the positive metric head and
// the coarse/fine network of common/nn/monocular_depth.h).
#pragma once

#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <vector>

namespace depth {

namespace nn = datamunge::nn;

/// .1 + softplus(logits): strictly positive metric depth in meters.
inline nn::Tensor PositiveMetricDepth(const nn::Tensor& logits) { return 0.1 + nn::softplus(logits); }

struct MonocularDepthResult {
  nn::Tensor              depth, coarse;
  std::vector<nn::Tensor> stages;
};

/// Fixed 48x64 RGB input. The dense coarse map sees the whole image; no depth/mask input.
inline MonocularDepthResult MonocularDepth(nn::ParameterStore& p, const nn::Tensor& rgb) {
  MonocularDepthResult result;
  nn::ConvLayerOptions o;
  o.activation = nn::Activation::Relu;
  o.stride     = 2;
  auto x       = nn::conv2d(p, "encoder1", rgb, 8, 5, o);
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
  o.stride   = 1;
  auto local = nn::conv2d(p, "local1", rgb, 16, 5, o);
  result.stages.push_back(local);
  local = nn::conv2d(p, "local2", nn::concat({local, up}, 3), 16, 3, o);
  result.stages.push_back(local);
  o.activation  = nn::Activation::None;
  auto residual = nn::conv2d(p, "local3", local, 1, 3, o);
  result.stages.push_back(residual);
  result.coarse = PositiveMetricDepth(up);
  result.depth  = PositiveMetricDepth(up + residual);
  result.stages.push_back(result.depth);
  return result;
}

struct DepthError {
  nn::Tensor error, valid;
};
inline DepthError DepthLogError(const nn::Tensor& prediction, const nn::Tensor& truth, const nn::Tensor& mask) {
  const nn::Tensor valid = nn::logical_and(nn::greater(mask, .5),
                                           nn::logical_and(nn::is_finite(truth), nn::logical_and(nn::greater(truth, double(.1f)), nn::less_equal(truth, 10.0))));
  // Select BEFORE Log: invalid NaN/zero targets or predictions cannot contaminate valid values/gradients.
  const nn::Tensor safe_truth = nn::where(valid, truth, 1.0), safe_prediction = nn::where(valid, prediction, 1.0);
  return {nn::log(safe_prediction) - nn::log(safe_truth), nn::cast(valid, prediction.dtype())};
}
inline nn::Tensor DepthNonemptyMean(const nn::Tensor& values, const nn::Tensor& counts) {
  const nn::Tensor present = nn::cast(nn::greater(counts, 0.0), values.dtype());
  return nn::sum(values * present) / nn::maximum(nn::sum(present), nn::scalar_like(1.0, present));
}
inline nn::Tensor DepthLogLoss(const nn::Tensor& prediction, const nn::Tensor& truth, const nn::Tensor& mask, double lambda = .5) {
  if (!std::isfinite(lambda) || lambda < 0 || lambda > 1) throw std::invalid_argument("log-depth lambda in[0,1]");
  const auto       e      = DepthLogError(prediction, truth, mask);
  const nn::Tensor count  = nn::sum(e.valid, {1, 2, 3});
  const nn::Tensor denom  = nn::maximum(count, nn::scalar_like(1.0, count));
  const nn::Tensor mean   = nn::sum(e.valid * e.error, {1, 2, 3}) / denom;
  const nn::Tensor square = nn::sum(e.valid * nn::square(e.error), {1, 2, 3}) / denom;
  return DepthNonemptyMean(square - lambda * nn::square(mean), count);
}
inline nn::Tensor DepthGradientLoss(const nn::Tensor& prediction, const nn::Tensor& truth, const nn::Tensor& mask) {
  const auto         e = DepthLogError(prediction, truth, mask);
  const std::int64_t H = prediction.size(1), W = prediction.size(2);
  auto               cut = [](const nn::Tensor& x, std::int64_t axis, std::int64_t start, std::int64_t stop) { return nn::slice_axis(x, axis, start, stop); };
  const nn::Tensor   left = cut(e.error, 2, 0, W - 1), right = cut(e.error, 2, 1, W), top = cut(e.error, 1, 0, H - 1), bottom = cut(e.error, 1, 1, H);
  const nn::Tensor   horizontal = cut(e.valid, 2, 0, W - 1) * cut(e.valid, 2, 1, W);
  const nn::Tensor   vertical   = cut(e.valid, 1, 0, H - 1) * cut(e.valid, 1, 1, H);
  const nn::Tensor   numerator  = nn::sum(horizontal * nn::abs(left - right), {1, 2, 3}) + nn::sum(vertical * nn::abs(top - bottom), {1, 2, 3});
  const nn::Tensor   count      = nn::sum(horizontal, {1, 2, 3}) + nn::sum(vertical, {1, 2, 3});
  return DepthNonemptyMean(numerator / nn::maximum(count, nn::scalar_like(1.0, count)), count);
}
/// {total = log + .1 gradient + .2 coarse, log, gradient, coarse}.
inline std::array<nn::Tensor, 4> DepthObjective(const MonocularDepthResult& result, const nn::Tensor& truth, const nn::Tensor& mask) {
  const nn::Tensor log = DepthLogLoss(result.depth, truth, mask), gradient = DepthGradientLoss(result.depth, truth, mask),
                   coarse = DepthLogLoss(result.coarse, truth, mask);
  return {log + .1 * gradient + .2 * coarse, log, gradient, coarse};
}

}  // namespace depth
