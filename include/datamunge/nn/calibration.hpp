#pragma once

/// @file calibration.hpp
/// @brief Bounded inverse-temperature fitting and top-label reliability summaries.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Fit a nonnegative inverse temperature by minimizing validation cross entropy.
///
/// The derivative in inverse temperature is monotone, so bisection finds a
/// bounded minimum. A boundary is returned when the minimum is outside the
/// bracket. Zero inverse temperature yields uniform probabilities. A positive
/// value preserves the order of classes within each row. For a flat objective,
/// returns 1 clipped to the bracket. The final bracket uses 48 bisections.
/// @param logits Finite CPU scores [N,C] with N>0 and C>=2.
/// @param labels CPU Int32 or Int64 targets [N] in [0,C).
/// @param upper Finite positive upper bound for inverse temperature, default 4.
/// @return Fitted inverse temperature in [0,upper]. Multiply logits by this value before softmax.
/// @throws std::invalid_argument For invalid inputs or bounds.
/// @throws std::runtime_error If centering or the derivative overflows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto logits = n::tensor({2,0,2,0,2,0,2,0}, {4,2});
/// const auto labels = n::from_vector<std::int64_t>({0,0,0,1});
/// const double beta = n::fit_inverse_temperature(logits, labels); // log(3)/2
/// @endcode
/// @par References
/// - Guo et al. (2017), On Calibration of Modern Neural Networks.
///   https://proceedings.mlr.press/v70/guo17a.html
/// @see top_label_reliability()
inline double fit_inverse_temperature(const Tensor& logits, const Tensor& labels, double upper = 4.) {
  if (!logits.defined() || logits.dim() != 2 || logits.size(0) == 0 || logits.size(1) < 2 ||
      !labels.defined() || labels.shape() != Shape{logits.size(0)} ||
      (labels.dtype() != DType::Int32 && labels.dtype() != DType::Int64) ||
      logits.device().backend != compute::Backend::CPU ||
      labels.device().backend != compute::Backend::CPU || !std::isfinite(upper) || upper <= 0.)
    throw std::invalid_argument("invalid logits, labels or inverse-temperature bound");
  NoGradGuard guard;
  const auto scores = logits.to(DType::Float64);
  if (!all(is_finite(scores)).item<bool>() ||
      min(labels).item<std::int64_t>() < 0 || max(labels).item<std::int64_t>() >= scores.size(1))
    throw std::invalid_argument("temperature scaling requires finite logits and valid labels");
  const auto centered = scores - max(scores, {1}, true);
  if (!all(is_finite(centered)).item<bool>())
    throw std::runtime_error("temperature scaling centering overflowed");
  const auto targets = one_hot(labels, scores.size(1)).to(DType::Float64);
  const auto slope = [&](double beta) {
    const auto derivative = mean(sum((softmax(centered * beta, 1) - targets) * centered, {1}));
    const double value = derivative.item();
    if (!std::isfinite(value))
      throw std::runtime_error("temperature scaling derivative overflowed");
    return value;
  };
  double lower = 0.;
  const double left = slope(lower), right = slope(upper);
  if (left == 0. && right == 0.)
    return std::min(1., upper);
  if (left >= 0.)
    return lower;
  if (right <= 0.)
    return upper;
  for (int i = 0; i < 48; ++i) {
    const double middle = lower + (upper - lower) / 2.;
    if (slope(middle) < 0.)
      lower = middle;
    else
      upper = middle;
  }
  return lower + (upper - lower) / 2.;
}

/// @brief Binned confidence and empirical accuracy for top-label calibration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::tensor({.75,.25,.25,.75}, {2,2});
/// const auto result = n::top_label_reliability(p, n::from_vector<std::int64_t>({0,1}));
/// const auto error = result.ece; // 0.25
/// @endcode
/// @see top_label_reliability()
struct TopLabelReliability {
  Tensor bins;        ///< Float64 [B,3] containing count, mean confidence and accuracy.
  Tensor predictions; ///< Int64 [N] top labels, ties resolved in favour of the first class.
  double ece{};       ///< Count-weighted mean absolute gap between confidence and accuracy.
};

/// @brief Calculate equal-width top-label reliability bins and expected calibration error.
///
/// Assigns confidence p to floor(B*p), with confidence 1 in the final bin.
/// Empty bins contain three zeros. ECE is an empirical summary dependent on
/// bin count and the evaluation sample.
/// @param probabilities CPU [N,C] with N>0, C>=2 and finite probabilities in [0,1].
/// Each row must sum to one within 1e-6.
/// @param labels CPU Int32 or Int64 targets [N] in [0,C).
/// @param bin_count Positive number of bins, default 10.
/// @return Bin counts, mean confidences, accuracies, predictions and ECE.
/// @throws std::invalid_argument For invalid shapes, probabilities, labels or bin counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::tensor({.8,.2,.2,.8}, {2,2});
/// const auto result = n::top_label_reliability(p, n::from_vector<std::int64_t>({0,0}), 5);
/// const auto count = result.bins.at({4,0}); // 2
/// @endcode
/// @par References
/// - Guo et al. (2017), On Calibration of Modern Neural Networks.
///   https://proceedings.mlr.press/v70/guo17a.html
/// @see fit_inverse_temperature(), TopLabelReliability
inline TopLabelReliability top_label_reliability(const Tensor& probabilities, const Tensor& labels,
                                                  std::int64_t bin_count = 10) {
  if (!probabilities.defined() || probabilities.dim() != 2 || probabilities.size(0) == 0 ||
      probabilities.size(1) < 2 || !labels.defined() ||
      labels.shape() != Shape{probabilities.size(0)} ||
      (labels.dtype() != DType::Int32 && labels.dtype() != DType::Int64) ||
      probabilities.device().backend != compute::Backend::CPU ||
      labels.device().backend != compute::Backend::CPU ||
      bin_count < 1 || bin_count > std::numeric_limits<std::int64_t>::max() / 3)
    throw std::invalid_argument("invalid data or reliability bin count");
  const auto rows = probabilities.size(0), classes = probabilities.size(1);
  const auto p = probabilities.to_vector<double>();
  const auto truth = labels.to_vector<std::int64_t>();
  std::vector<std::int64_t> predictions(rows);
  std::vector<double> bins(3 * bin_count);
  for (std::int64_t i = 0; i < rows; ++i) {
    const auto begin = p.begin() + i * classes, end = begin + classes;
    const auto winner = std::max_element(begin, end);
    double total = 0.;
    for (auto value = begin; value != end; ++value) {
      if (!std::isfinite(*value) || *value < 0. || *value > 1.)
        throw std::invalid_argument("expected finite probabilities in [0,1]");
      total += *value;
    }
    if (std::abs(total - 1.) > 1e-6 || truth[i] < 0 || truth[i] >= classes)
      throw std::invalid_argument("expected normalized probabilities and valid labels");
    const auto bin = std::min(bin_count - 1, static_cast<std::int64_t>(*winner * bin_count));
    predictions[i] = winner - begin;
    bins[3 * bin] += 1.;
    bins[3 * bin + 1] += *winner;
    bins[3 * bin + 2] += predictions[i] == truth[i] ? 1. : 0.;
  }
  double ece = 0.;
  for (std::int64_t b = 0; b < bin_count; ++b) {
    const double count = bins[3 * b];
    if (count == 0.)
      continue;
    auto& confidence = bins[3 * b + 1];
    auto& accuracy = bins[3 * b + 2];
    confidence /= count;
    accuracy /= count;
    ece += count / rows * std::abs(confidence - accuracy);
  }
  return {from_vector(bins, {bin_count, 3}), from_vector(predictions), ece};
}

}  // namespace datamunge::nn
