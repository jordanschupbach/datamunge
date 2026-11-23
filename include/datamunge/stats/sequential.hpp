#pragma once

/// @file sequential.hpp
/// @brief Sequential simple-hypothesis testing and time-uniform inference for a bounded mean.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Decision reached by a sequential probability ratio test.
/// @par Example
/// @code
/// const auto decision = dm::stats::SequentialDecision::Continue;
/// std::print("{}\n", decision == dm::stats::SequentialDecision::Continue); // true
/// @endcode
/// @see SequentialProbabilityRatioTest
enum class SequentialDecision {
  Continue,          ///< The likelihood ratio is between the boundaries.
  AcceptNull,        ///< The lower likelihood boundary has been reached.
  AcceptAlternative, ///< The upper likelihood boundary has been reached.
};

/// @brief Wald's sequential probability ratio test for two simple hypotheses.
///
/// Accepts log-likelihood increments in favor of the alternative. For error
/// targets alpha and beta, the log boundaries are log(beta/(1-alpha)) and
/// log((1-beta)/alpha). These calibrate nominal errors approximately in the
/// presence of stopping overshoot. The first decision to stop is absorbing.
/// @par Example
/// @code
/// dm::stats::SequentialProbabilityRatioTest test(.05, .2);
/// const auto decision = test.observe(4);
/// std::print("{}\n", decision == dm::stats::SequentialDecision::AcceptAlternative); // true
/// @endcode
/// @par References
/// - Wald (1945), Sequential Tests of Statistical Hypotheses, doi:10.1214/aoms/1177731118.
/// @see BoundedMeanConfidenceSequence
class SequentialProbabilityRatioTest {
public:
  /// @brief Sets Wald's decision boundaries and a zero starting log likelihood ratio.
  /// @param alpha Nominal type-I error, positive and less than 1.
  /// @param beta Nominal type-II error, positive and less than 1, with alpha + beta < 1.
  /// @throws std::invalid_argument Invalid error targets.
  /// @par Example
  /// @code
  /// const dm::stats::SequentialProbabilityRatioTest test(.05, .2);
  /// std::print("{:.3f}\n", test.upper_boundary()); // 2.773
  /// @endcode
  SequentialProbabilityRatioTest(double alpha, double beta) {
    if (!(alpha > 0 && beta > 0 && alpha < 1 && beta < 1))
      throw std::invalid_argument("sequential test: alpha and beta must be positive and sum to less than one");
    const auto gap = (1 - static_cast<long double>(std::max(alpha, beta))) - std::min(alpha, beta);
    if (!(gap > 0))
      throw std::invalid_argument("sequential test: alpha and beta must sum to less than one");
    lower_boundary_ = gap < beta ? static_cast<double>(-std::log1p(gap / beta)) :
                                  std::log(beta) - std::log1p(-alpha);
    upper_boundary_ = gap < alpha ? static_cast<double>(std::log1p(gap / alpha)) :
                                   std::log1p(-beta) - std::log(alpha);
  }

  /// @brief Adds one log likelihood increment while the test is open.
  /// @param log_likelihood_ratio Log of alternative likelihood divided by null likelihood.
  /// Infinite contributions are allowed when one likelihood is zero.
  /// @return The current decision. Valid increments after stopping leave it unchanged.
  /// @throws std::invalid_argument NaN increment.
  /// @par Example
  /// @code
  /// dm::stats::SequentialProbabilityRatioTest test(.05, .2);
  /// const auto decision = test.observe(-2);
  /// std::print("{}\n", decision == dm::stats::SequentialDecision::AcceptNull); // true
  /// @endcode
  SequentialDecision observe(double log_likelihood_ratio) {
    if (std::isnan(log_likelihood_ratio))
      throw std::invalid_argument("sequential test: log likelihood increment cannot be NaN");
    if (decision_ == SequentialDecision::Continue) {
      log_likelihood_ratio_ += log_likelihood_ratio;
      if (log_likelihood_ratio_ <= lower_boundary_)
        decision_ = SequentialDecision::AcceptNull;
      else if (log_likelihood_ratio_ >= upper_boundary_)
        decision_ = SequentialDecision::AcceptAlternative;
    }
    return decision_;
  }

  /// @brief Reads the cumulative log likelihood ratio through the current observation.
  /// @return Current log likelihood ratio, frozen at stopping.
  [[nodiscard]] double log_likelihood_ratio() const noexcept {
    return static_cast<double>(log_likelihood_ratio_);
  }

  /// @brief Reads the lower stopping boundary.
  /// @return Log likelihood threshold for accepting the null.
  [[nodiscard]] double lower_boundary() const noexcept { return lower_boundary_; }

  /// @brief Reads the upper stopping boundary.
  /// @return Log likelihood threshold for accepting the alternative.
  [[nodiscard]] double upper_boundary() const noexcept { return upper_boundary_; }

  /// @brief Reads whether the test has stopped and which boundary was reached.
  /// @return Current sequential decision.
  [[nodiscard]] SequentialDecision decision() const noexcept { return decision_; }

private:
  double lower_boundary_;               ///< Log null acceptance threshold.
  double upper_boundary_;               ///< Log alternative acceptance threshold.
  long double log_likelihood_ratio_{0}; ///< Widened cumulative evidence.
  SequentialDecision decision_{SequentialDecision::Continue}; ///< Absorbing stopping state.
};

/// @brief Current window of a time-uniform confidence sequence for a mean.
/// @par Example
/// @code
/// dm::stats::BoundedMeanConfidenceSequence sequence(0, 1);
/// std::print("{}\n", sequence.current().size); // 0
/// @endcode
/// @see BoundedMeanConfidenceSequence
struct BoundedMeanInterval {
  std::size_t size;               ///< Number of observations incorporated.
  std::optional<double> estimate; ///< Current sample mean, absent before the first observation.
  double lower;                   ///< Lower time-uniform confidence bound.
  double upper;                   ///< Upper time-uniform confidence bound.
};

/// @brief Time-uniform confidence sequence for a constant conditional mean of bounded data.
///
/// Applies a two-sided normal mixture to Hoeffding's sub-Gaussian process.
/// For observations in the declared range with a common conditional mean
/// given the past, the probability that any reported interval misses that
/// mean is at most 1 minus the confidence level. State uses O(1) storage.
/// @par Example
/// @code
/// dm::stats::BoundedMeanConfidenceSequence sequence(0, 1);
/// std::print("{:.1f}\n", *sequence.observe(.5).estimate); // 0.5
/// @endcode
/// @par References
/// - Howard et al. (2021), Time-uniform, nonparametric, nonasymptotic confidence
///   sequences, doi:10.1214/20-AOS1991.
/// @see SequentialProbabilityRatioTest
class BoundedMeanConfidenceSequence {
public:
  /// @brief Initializes the sequence with the full permitted range.
  /// @param lower Known finite lower bound on each observation.
  /// @param upper Known finite upper bound on each observation, greater than lower.
  /// @param confidence_level Simultaneous coverage in (0,1), default .95.
  /// @param mixture_precision Positive finite precision of the normal mixture on the
  /// unit-range scale, default 1.
  /// @throws std::invalid_argument Invalid bounds, confidence or precision.
  /// @par Example
  /// @code
  /// const dm::stats::BoundedMeanConfidenceSequence sequence(2, 3);
  /// std::print("{:.0f}\n", sequence.current().lower); // 2
  /// @endcode
  BoundedMeanConfidenceSequence(double lower, double upper, double confidence_level = .95,
                                double mixture_precision = 1)
      : lower_(lower), upper_(upper), precision_(mixture_precision) {
    if (!(std::isfinite(lower) && std::isfinite(upper) && lower < upper) ||
        !(confidence_level > 0 && confidence_level < 1) ||
        !(mixture_precision > 0 && std::isfinite(mixture_precision)))
      throw std::invalid_argument("bounded mean confidence sequence: invalid bounds, confidence or precision");
    log_error_penalty_ = -2 * std::log1p(-static_cast<long double>(confidence_level));
  }

  /// @brief Incorporates one observation and returns the updated interval.
  /// @param value Finite observation in the declared closed range.
  /// @return Updated owned interval.
  /// @throws std::invalid_argument Observation outside the declared range or nonfinite.
  /// @throws std::overflow_error Count exceeds the representable range.
  /// @par Example
  /// @code
  /// dm::stats::BoundedMeanConfidenceSequence sequence(0, 1);
  /// std::print("{}\n", sequence.observe(1).size); // 1
  /// @endcode
  BoundedMeanInterval observe(double value) {
    if (!(value >= lower_ && value <= upper_))
      throw std::invalid_argument("bounded mean confidence sequence: observation outside bounds");
    if (size_ == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("bounded mean confidence sequence: observation count overflow");
    const long double width = static_cast<long double>(upper_) - lower_;
    const auto scaled = std::isfinite(width) ? (static_cast<long double>(value) - lower_) / width :
        (.5L * value - .5L * lower_) / (.5L * upper_ - .5L * lower_);
    ++size_;
    mean_ += (scaled - mean_) / size_;
    return current();
  }

  /// @brief Reads the interval after all observations received so far.
  /// @return Current interval, or the full range and an absent estimate at time zero.
  [[nodiscard]] BoundedMeanInterval current() const noexcept {
    if (size_ == 0)
      return {0, std::nullopt, lower_, upper_};
    const auto variance = static_cast<long double>(size_) / 4;
    const auto precision = precision_ + variance;
    const auto log_ratio = variance < precision_ ? std::log1p(variance / precision_) :
                                                   std::log(precision) - std::log(precision_);
    const auto radius = (std::sqrt(precision) / size_) * std::sqrt(log_error_penalty_ + log_ratio);
    return {size_, std::lerp(lower_, upper_, static_cast<double>(mean_)),
            std::lerp(lower_, upper_, static_cast<double>(std::max(0.0L, mean_ - radius))),
            std::lerp(lower_, upper_, static_cast<double>(std::min(1.0L, mean_ + radius)))};
  }

private:
  double lower_;                ///< Lower observation bound.
  double upper_;                ///< Upper observation bound.
  long double precision_;       ///< Normal-mixture precision for unit-range observations.
  long double log_error_penalty_; ///< Twice the negative logarithm of the error level.
  std::size_t size_{0};          ///< Number of accepted observations.
  long double mean_{0};          ///< Running mean after rescaling to the unit interval.
};

} // namespace datamunge::stats
