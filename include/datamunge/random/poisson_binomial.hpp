#pragma once

/// @file poisson_binomial.hpp
/// @brief Numerical distribution of a count of independent Bernoulli events with unequal probabilities.

#include <datamunge/random/distributions.hpp>
#include <numeric>
#include <vector>

namespace datamunge::random {

/// @brief Count distribution for independent Bernoulli trials with specified success probabilities.
///
/// Computes the probability mass by convolving one Bernoulli distribution per
/// stochastic trial. With m probabilities strictly between zero and one, setup
/// costs O(m²) time and O(m) storage after reading the input. Trials fixed at zero
/// or one only affect the support. The empty sequence gives the point mass at zero.
/// @par Example
/// @code
/// const dm::random::PoissonBinomial count({.25, .75});
/// std::print("{:.3f}\n", count.pmf(1)); // 0.625
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson_binom.html
/// @see binomial_pdf()
class PoissonBinomial {
public:
  /// @brief Constructs the Poisson-binomial probability mass table.
  /// @param probabilities One finite success probability in [0,1] per independent trial.
  /// @throws std::invalid_argument A probability is outside [0,1] or is NaN.
  /// @par Example
  /// @code
  /// const dm::random::PoissonBinomial fixed({0, 1});
  /// std::print("{:.0f}\n", fixed.pmf(1)); // 1
  /// @endcode
  explicit PoissonBinomial(const std::vector<double>& probabilities) : mass_{1}, minimum_(0) {
    for (const double p : probabilities)
      detail::require_probability(p, "PoissonBinomial");
    for (const long double p : probabilities) {
      if (p == 1) {
        ++minimum_;
      } else if (p > 0) {
        mass_.push_back(0);
        for (auto k = mass_.size() - 1; k > 0; --k)
          mass_[k] = p * mass_[k - 1] + (1 - p) * mass_[k];
        mass_.front() *= 1 - p;
      }
    }
    total_ = std::accumulate(mass_.begin(), mass_.end(), 0.0L);
  }

  /// @brief Returns the probability of exactly k successes in O(1) time.
  /// @param k Number of successes.
  /// @return Probability mass, zero outside the support.
  /// @par Example
  /// @code
  /// std::print("{:.3f}\n", dm::random::PoissonBinomial({.25, .75}).pmf(2)); // 0.188
  /// @endcode
  [[nodiscard]] double pmf(std::size_t k) const noexcept {
    if (mass_.empty())
      return k == minimum_ ? 1 : 0;
    if (k < minimum_ || k - minimum_ >= mass_.size())
      return 0;
    return static_cast<double>(mass_[k - minimum_] / total_);
  }

  /// @brief Returns the inclusive lower or strict upper tail, summing the requested tail directly.
  /// @param k Number of successes.
  /// @param lower_tail True for P(X <= k), otherwise P(X > k), default true.
  /// @return Requested tail probability.
  /// @par Example
  /// @code
  /// const dm::random::PoissonBinomial count({.25, .75});
  /// std::print("{:.4f}\n", count.cdf(1, false)); // 0.1875
  /// @endcode
  [[nodiscard]] double cdf(std::size_t k, bool lower_tail = true) const noexcept {
    if (k < minimum_)
      return lower_tail ? 0 : 1;
    const auto index = k - minimum_;
    if (mass_.size() <= 1 || index >= mass_.size() - 1)
      return lower_tail ? 1 : 0;
    const auto boundary = mass_.begin() + index + 1;
    const auto tail = lower_tail ? std::accumulate(mass_.begin(), boundary, 0.0L)
                                 : std::accumulate(boundary, mass_.end(), 0.0L);
    return static_cast<double>(std::min(tail / total_, 1.0L));
  }

  /// @brief Selects the first supported count whose CDF reaches the probability.
  /// @param probability Cumulative probability in [0,1].
  /// @return Count quantile, including the support endpoints for zero and one.
  /// @throws std::invalid_argument Invalid cumulative probability.
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::random::PoissonBinomial({.25, .75}).quantile(.5)); // 1
  /// @endcode
  [[nodiscard]] std::size_t quantile(double probability) const {
    detail::require_probability(probability, "PoissonBinomial::quantile");
    if (probability == 0 || mass_.empty())
      return minimum_;
    const auto maximum = minimum_ + mass_.size() - 1;
    if (probability == 1)
      return maximum;
    long double cumulative = 0;
    if (probability <= .5) {
      for (std::size_t i = 0; i < mass_.size(); ++i) {
        cumulative += mass_[i];
        if (static_cast<double>(cumulative / total_) >= probability)
          return minimum_ + i;
      }
      return maximum;
    }
    for (auto i = mass_.size(); i > 0; --i) {
      cumulative += mass_[i - 1];
      if (static_cast<double>(1 - cumulative / total_) < probability)
        return minimum_ + i - 1;
    }
    return minimum_;
  }

private:
  std::vector<long double> mass_; ///< Mass entries indexed from the smallest possible count.
  std::size_t minimum_;           ///< Count of trials with probability exactly one.
  long double total_;            ///< Sum of the numerically convolved masses.
};

} // namespace datamunge::random
