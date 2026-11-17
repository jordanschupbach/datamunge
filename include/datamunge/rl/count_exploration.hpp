// datamunge::nn port of tflow_examples src/common/rl/count_bonus.h (TF-free; copied unchanged
// apart from the include guard).
// Count-based exploration and density-model pseudo-counts.
//
// A density model's probability of an observation before and after learning
// it once defines N_hat = rho * (1 - rho_after) / (rho_after - rho).
// An empirical categorical density recovers the visit count; a factorized
// density shares evidence across observations with common features. The
// exploration reward beta / sqrt(N_hat + 1) uses the count BEFORE this visit.
// See Bellemare et al. (2016), https://arxiv.org/abs/1606.01868, equation 2.
#pragma once
/// @file count_exploration.hpp
/// @brief One-observation density updates, consistent pseudo-counts and explicit bonuses.

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::rl::foundations {

/// @brief Recover the source density pseudo-count from probabilities before/after one observation.
///
/// A strict increase ending exactly at1 is inconsistent and rejected, including0->1.
/// No density state is mutated. Infinite counts indicate no observable density change.
///
/// @param before Finite density rho in[0,1].
/// @param after Finite learning-positive recoded density rho_after in[before,1].
/// @return rho*(1-rho_after)/(rho_after-rho); before0 yields0; unchanged positive density yields
/// infinity.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto count=f::PseudoCount(.2,.3);
/// @endcode
/// @par References
/// - Bellemare et al. (2016), Unifying Count-Based Exploration and Intrinsic Motivation,
/// https://arxiv.org/abs/1606.01868 , equation2.
/// @see CountBonus(), FactorizedDensity::Observe()
inline double PseudoCount(double before, double after) {
  if (!std::isfinite(before) || !std::isfinite(after) || before < 0 || after > 1 || after < before)
    throw std::invalid_argument("pseudo-count needs 0 <= rho <= rho_after <= 1");
  if (after == 1 && before < after)
    throw std::invalid_argument("rho < rho_after = 1 has no consistent pseudo-count");
  if (before == 0)
    return 0;
  if (after == before)
    return std::numeric_limits<double>::infinity();
  return before * (1 - after) / (after - before);
}

/// @brief Compute an explicit inverse-square-root exploration bonus.
///
/// @param count Nonnegative finite or positive-infinite pre-observation count; NaNs are rejected.
/// @param beta Finite nonnegative bonus scale.
/// @return beta/sqrt(count+1); positive infinity yields0.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto bonus=f::CountBonus(99,1);
/// @endcode
/// @par References
/// - Bellemare et al. (2016), Unifying Count-Based Exploration and Intrinsic Motivation,
/// https://arxiv.org/abs/1606.01868 , equation2.
/// @see PseudoCount()
inline double CountBonus(double count, double beta) {
  if (std::isnan(count) || count < 0 || !std::isfinite(beta) || beta < 0)
    throw std::invalid_argument("count and bonus scale must be nonnegative");
  return beta / std::sqrt(count + 1);
}

// Product of independent categorical distributions, one per feature, with
// symmetric Dirichlet priors. A single feature gives a joint state counter;
// separate row/column features give a small, deliberately approximate model.
// Every factor's probability of the observed feature increases on update,
// so the joint density is learning-positive. Counts persist across episodes.

/// @brief Own independent categorical density factors with symmetric Dirichlet priors.
///
/// Counts persist until this object is replaced. Copies own independent vectors. Observe learns
/// exactly one supplied observation and returns its pre-update pseudo-count, including prior
/// evidence. There is no optimizer, episode or experiment loop. Concurrent mutation requires
/// external locking.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// f::FactorizedDensity density({3,3});auto before=density.Probability({1,2});auto
/// count=density.Observe({1,2});
/// @endcode
/// @par References
/// - Bellemare et al. (2016), Unifying Count-Based Exploration and Intrinsic Motivation,
/// https://arxiv.org/abs/1606.01868 , equation2.
/// @see Probability(), Observe()
class FactorizedDensity {
 public:
  /// @brief Construct independent categorical factors.
  ///
  /// @param sizes Nonempty positive category counts, each at most1000000.
  /// @param prior Finite positive prior per category, default.5; prior*size must be finite.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::FactorizedDensity density({3,3},.5);
  /// @endcode
  /// @par References
  /// - Bellemare et al. (2016), Unifying Count-Based Exploration and Intrinsic Motivation,
  /// https://arxiv.org/abs/1606.01868 , equation2.
  /// @see Probability()
  explicit FactorizedDensity(const std::vector<int>& sizes, double prior = 0.5) : prior_(prior) {
    if (sizes.empty() || !std::isfinite(prior) || prior <= 0)
      throw std::invalid_argument("density needs features and a positive prior");
    for (int size : sizes) {
      if (size < 1)
        throw std::invalid_argument("feature cardinality must be positive");
      if (size > 1000000 || !std::isfinite(prior * size))
        throw std::invalid_argument("density alphabet/prior normalization overflow");
      counts_.emplace_back(size, 0);
    }
  }

  /// @brief Read the joint product of posterior-predictive marginal probabilities.
  ///
  /// @param features One valid category index per factor.
  /// @return Double product; state unchanged. Underflow to0 is retained for many factors.
  /// @throws std::invalid_argument Feature count mismatches.
  /// @throws std::out_of_range A feature is outside its alphabet.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::FactorizedDensity density({3,3});auto probability=density.Probability({1,2});
  /// @endcode
  /// @par References
  /// - Bellemare et al. (2016), Unifying Count-Based Exploration and Intrinsic Motivation,
  /// https://arxiv.org/abs/1606.01868 , equation2.
  /// @see Observe()
  double Probability(const std::vector<int>& features) const {
    Validate(features);
    double p = 1;
    for (size_t i = 0; i < features.size(); ++i)
      p *= (counts_[i][features[i]] + prior_) / (total_ + prior_ * counts_[i].size());
    return p;
  }

  // Learn exactly once and return the pseudo-count before the update. Both
  // probabilities refer to the same observation, on either side of learning.

  /// @brief Update each factor once and return this observation prior pseudo-count.
  ///
  /// A single non-degenerate factor gives raw category visits plus the prior; factorization shares
  /// evidence across observations. Double probability rounding can yield an unchanged density.
  /// For a certain event with unchanged probability1, the pseudo-count is infinite.
  ///
  /// @param features One valid category index per factor; validated before count mutation.
  /// @return Density pseudo-count before learning, including prior mass; may be infinite.
  /// @throws std::invalid_argument Feature count mismatches.
  /// @throws std::out_of_range Feature index is invalid.
  /// @throws std::overflow_error UInt64 observation total is exhausted.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// f::FactorizedDensity density({4});auto before=density.Observe({1});
  /// @endcode
  /// @par References
  /// - Bellemare et al. (2016), Unifying Count-Based Exploration and Intrinsic Motivation,
  /// https://arxiv.org/abs/1606.01868 , equation2.
  /// @see PseudoCount(), Probability()
  double Observe(const std::vector<int>& features) {
    if (total_ == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("density count overflow");
    const double before = Probability(features); // validates before mutating
    for (size_t i = 0; i < features.size(); ++i)
      ++counts_[i][features[i]];
    ++total_;
    return PseudoCount(before, Probability(features));
  }

 private:
  /// @brief Check one category per owned density factor.
  ///
  /// @param features Candidate category indices.
  /// @throws std::invalid_argument Feature count mismatches.
  /// @throws std::out_of_range Feature index is outside its alphabet.
  void Validate(const std::vector<int>& features) const {
    if (features.size() != counts_.size())
      throw std::invalid_argument("wrong number of density features");
    for (size_t i = 0; i < features.size(); ++i)
      if (features[i] < 0 || static_cast<size_t>(features[i]) >= counts_[i].size())
        throw std::out_of_range("density feature outside its alphabet");
  }

  double   prior_;     ///< Positive symmetric Dirichlet pseudo-count per category.
  uint64_t total_ = 0; ///< Successful observations across episodes.
  std::vector<std::vector<uint64_t>> counts_; ///< Owned marginal category visit counts.
};

} // namespace datamunge::rl::foundations
