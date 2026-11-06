#pragma once

/// @file classification_analysis.hpp
/// @brief Tensor probability scores, reliability bins and validation-only temperature fitting.

#include <algorithm>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <numeric>

/// @brief Reproducible dataset studies and descriptive neural-classifier diagnostics.
namespace datamunge::nn::study {

/// @brief Classification scores and equal-width reliability bins for fixed logits.
///
/// Brier is the mean sum over classes, without dividing by class count.
/// ECE uses top-class confidence with bins [0,1/B),...,[1-1/B,1].
/// Empty bins have zero confidence/accuracy and do not contribute to ECE.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::study::classification_diagnostics(n::zeros({2,2}),
/// n::tensor({0,1}).to(n::DType::Int64));
/// @endcode
/// @see classification_diagnostics()
struct ClassificationDiagnostics {
  double                    nll      = 0;   ///< Stable mean negative log likelihood, nats/image.
  double                    brier    = 0;   ///< Mean sum of squared class probability errors.
  double                    ece      = 0;   ///< Sample-weighted absolute confidence/accuracy gap.
  double                    accuracy = 0;   ///< Fraction of correct top-one predictions.
  std::vector<double>       bin_confidence; ///< Mean top-class probability in each nonempty bin.
  std::vector<double>       bin_accuracy;   ///< Empirical accuracy in each nonempty bin.
  std::vector<std::int64_t> bin_count;      ///< Samples in each reliability bin.
  Tensor                    probabilities;  ///< Float64 [N,C] softmax(logits/temperature).
};

/// @brief Score a fixed multiclass classifier and compute descriptive reliability bins.
///
/// No fitting, resampling or state mutation. Equal logits use argmax's first-class
/// tie rule. Labels must be integer, nonempty and in [0,C). Finite logits and a
/// positive temperature are required. This ECE is bin-dependent, not a proper score.
/// @param logits Finite floating raw [N,C] scores, C at least two.
/// @param labels Integer [N] class IDs.
/// @param temperature Positive finite scale, default one.
/// @param bins Positive reliability bin count, default 15.
/// @return NLL, Brier, ECE, accuracy, bin aggregates and probabilities.
/// @throws std::invalid_argument For invalid shapes, values, labels or options.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::study::classification_diagnostics(n::zeros({2,2}),
/// n::tensor({0,1}).to(n::DType::Int64)); const auto nll = d.nll; // log(2)
/// @endcode
/// @par References
/// - Guo et al. (2017), On Calibration of Modern Neural Networks.
///   https://arxiv.org/abs/1706.04599
/// @see fit_temperature()
inline ClassificationDiagnostics classification_diagnostics(
    const Tensor& logits,
    const Tensor& labels,
    double        temperature = 1,
    int           bins        = 15) {
  if (!logits.defined() || logits.dim() != 2 || logits.size(0) < 1 || logits.size(1) < 2 ||
      (logits.dtype() != DType::Float32 && logits.dtype() != DType::Float64) || !labels.defined() ||
      labels.shape() != Shape{logits.size(0)} ||
      (labels.dtype() != DType::Int32 && labels.dtype() != DType::Int64) ||
      !std::isfinite(temperature) || temperature <= 0 || bins < 1 ||
      !all(is_finite(logits)).item<bool>() || min(labels).item() < 0 ||
      max(labels).item() >= logits.size(1))
    throw std::invalid_argument(
        "finite multiclass logits, aligned integer labels and positive options required");
  NoGradGuard               guard;
  ClassificationDiagnostics out;
  const auto                z = logits.to(DType::Float64) / temperature;
  if (!all(is_finite(z)).item<bool>())
    throw std::invalid_argument("temperature scaling overflowed");
  out.probabilities = softmax(z, 1);
  out.nll           = sparse_softmax_cross_entropy(z, labels).item();
  out.brier =
      mean(sum(square(out.probabilities - one_hot(labels, logits.size(1), 1, 0, DType::Float64)),
               {1}))
          .item();
  out.accuracy          = nn::accuracy(z, labels);
  const auto confidence = max(out.probabilities, {1}).to_vector<double>();
  const auto prediction = argmax(z, 1).to_vector<std::int64_t>(),
             truth      = labels.to_vector<std::int64_t>();
  out.bin_confidence.assign(bins, 0);
  out.bin_accuracy.assign(bins, 0);
  out.bin_count.assign(bins, 0);
  for (std::size_t i = 0; i < truth.size(); ++i) {
    const auto b = std::min(bins - 1, static_cast<int>(confidence[i] * bins));
    ++out.bin_count[b];
    out.bin_confidence[b] += confidence[i];
    out.bin_accuracy[b] += prediction[i] == truth[i];
  }
  for (int b = 0; b < bins; ++b)
    if (out.bin_count[b]) {
      out.bin_confidence[b] /= out.bin_count[b];
      out.bin_accuracy[b] /= out.bin_count[b];
      out.ece +=
          out.bin_count[b] * std::abs(out.bin_accuracy[b] - out.bin_confidence[b]) / truth.size();
    }
  return out;
}

/// @brief Fit a scalar temperature on supplied validation logits by bounded golden-section search.
///
/// Minimizes native tensor cross entropy in log(T) over [log(.05),log(10)],
/// with 64 iterations. Weights/logits remain fixed. Bounds and iteration count
/// are explicit numerical choices; the caller must keep test labels out of fitting.
/// Uses the same validation checks as classification_diagnostics().
/// @param validation_logits Fixed finite floating [N,C] validation scores.
/// @param validation_labels Corresponding integer [N] validation classes.
/// @return Positive fitted temperature in [.05,10], with T=1 retained if no improvement.
/// @throws std::invalid_argument For incompatible logits or labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto temperature =
/// n::study::fit_temperature(n::zeros({2,2}),n::tensor({0,1}).to(n::DType::Int64)); // 1
/// @endcode
/// @par References
/// - Guo et al. (2017), On Calibration of Modern Neural Networks.
///   https://arxiv.org/abs/1706.04599
/// @see classification_diagnostics()
inline double fit_temperature(const Tensor& validation_logits, const Tensor& validation_labels) {
  const auto  initial = classification_diagnostics(validation_logits, validation_labels);
  NoGradGuard guard;
  const auto  z         = validation_logits.to(DType::Float64);
  auto        objective = [&](double t) {
    return sparse_softmax_cross_entropy(z / std::exp(t), validation_labels).item();
  };
  double           a = std::log(.05), b = std::log(10.);
  constexpr double ratio = .6180339887498948482;
  double c = b - ratio * (b - a), d = a + ratio * (b - a), fc = objective(c), fd = objective(d);
  for (int i = 0; i < 64; ++i) {
    if (fc < fd) {
      b  = d;
      d  = c;
      fd = fc;
      c  = b - ratio * (b - a);
      fc = objective(c);
    } else {
      a  = c;
      c  = d;
      fc = fd;
      d  = a + ratio * (b - a);
      fd = objective(d);
    }
  }
  const double t = (a + b) / 2;
  return objective(t) < initial.nll ? std::exp(t) : 1.;
}
} // namespace datamunge::nn::study
