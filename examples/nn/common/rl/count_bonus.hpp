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

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rl {

inline double PseudoCount(double before, double after) {
  if (!std::isfinite(before) || !std::isfinite(after) || before < 0 ||
      after > 1 || after < before)
    throw std::invalid_argument("pseudo-count needs 0 <= rho <= rho_after <= 1");
  if (after == 1 && before < after)
    throw std::invalid_argument("rho < rho_after = 1 has no consistent pseudo-count");
  if (before == 0) return 0;
  if (after == before) return std::numeric_limits<double>::infinity();
  return before * (1 - after) / (after - before);
}

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
class FactorizedDensity {
 public:
  explicit FactorizedDensity(const std::vector<int>& sizes, double prior = 0.5)
      : prior_(prior) {
    if (sizes.empty() || !std::isfinite(prior) || prior <= 0)
      throw std::invalid_argument("density needs features and a positive prior");
    for (int size : sizes) {
      if (size < 1) throw std::invalid_argument("feature cardinality must be positive");
      counts_.emplace_back(size, 0);
    }
  }

  double Probability(const std::vector<int>& features) const {
    Validate(features);
    double p = 1;
    for (size_t i = 0; i < features.size(); ++i)
      p *= (counts_[i][features[i]] + prior_) / (total_ + prior_ * counts_[i].size());
    return p;
  }

  // Learn exactly once and return the pseudo-count before the update. Both
  // probabilities refer to the same observation, on either side of learning.
  double Observe(const std::vector<int>& features) {
    const double before = Probability(features);  // validates before mutating
    for (size_t i = 0; i < features.size(); ++i) ++counts_[i][features[i]];
    ++total_;
    return PseudoCount(before, Probability(features));
  }

 private:
  void Validate(const std::vector<int>& features) const {
    if (features.size() != counts_.size())
      throw std::invalid_argument("wrong number of density features");
    for (size_t i = 0; i < features.size(); ++i)
      if (features[i] < 0 || static_cast<size_t>(features[i]) >= counts_[i].size())
        throw std::out_of_range("density feature outside its alphabet");
  }

  double prior_;
  uint64_t total_ = 0;
  std::vector<std::vector<uint64_t>> counts_;
};

}  // namespace rl

