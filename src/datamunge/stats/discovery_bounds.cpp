/// @file discovery_bounds.cpp
/// @brief Fixed-threshold null fraction estimation and accelerated Simes closure.

#include <datamunge/stats/discovery_bounds.hpp>
#include <datamunge/stats/p_adjust.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Shared p-value checks and Simes cutoffs.
namespace {

/// @brief Rejects invalid p-values before order comparisons.
/// @param values Vector of purported p-values.
void check_probabilities(const std::vector<double>& values) {
  for (const auto value : values)
    if (!(value >= 0 && value <= 1))
      throw std::invalid_argument("discovery inference: p-values must be in [0,1]");
}

/// @brief Computes the largest cardinality of a locally nonrejected intersection.
/// @param ordered Valid p-values in ascending order.
/// @param alpha Fixed simultaneous miscoverage probability.
/// @return Largest locally nonrejected intersection size, with zero as a fallback.
std::size_t largest_nonrejected(const std::vector<double>& ordered, double alpha) {
  // For each cardinality, the largest p-values are the least likely to reject.
  // A nonrejected tail remains nonrejected when its smallest value is removed.
  std::size_t lo = 0, hi = ordered.size();
  while (lo < hi) {
    const auto size = std::midpoint(hi, lo);
    bool rejected = false;
    for (std::size_t j = 0; j < size && !rejected; ++j)
      rejected = static_cast<long double>(size) * ordered[ordered.size() - size + j] <=
                 static_cast<long double>(alpha) * (j + 1);
    if (rejected)
      hi = size - 1;
    else
      lo = size;
  }
  return lo;
}

} // namespace

std::vector<StoreyQValueFit> storey_qvalues(const std::vector<double>& p,
                                           const std::vector<double>& thresholds, bool add_one) {
  check_probabilities(p);
  if (p.empty() || thresholds.empty())
    throw std::invalid_argument("Storey q-values: p-values and thresholds must be nonempty");
  auto ordered = p;
  std::sort(ordered.begin(), ordered.end());
  const auto bh = p_adjust(p, PAdjustMethod::BH);

  std::vector<StoreyQValueFit> result;
  result.reserve(thresholds.size());
  for (const auto threshold : thresholds) {
    if (!(threshold >= 0 && threshold < 1))
      throw std::invalid_argument("Storey q-values: threshold must be in [0,1)");
    const auto tail = ordered.end() - std::upper_bound(ordered.begin(), ordered.end(), threshold);
    const auto raw = (static_cast<long double>(tail) + add_one) /
                     (p.size() * (1 - static_cast<long double>(threshold)));
    const auto fraction = threshold == 0 ? 1.0 : static_cast<double>(std::min(1.0L, raw));
    auto& fit = result.emplace_back(StoreyQValueFit{threshold, fraction, bh});
    for (auto& q : fit.q_values)
      q *= fraction;
  }
  return result;
}

SimesDiscoveryBounds::SimesDiscoveryBounds(std::vector<double> p, double alpha)
    : p_(std::move(p)), alpha_(alpha), nonrejected_size_(0) {
  check_probabilities(p_);
  if (!(alpha > 0 && alpha < 1))
    throw std::invalid_argument("Simes discovery bounds: alpha must be in (0,1)");
  auto ordered = p_;
  std::sort(ordered.begin(), ordered.end());
  nonrejected_size_ = largest_nonrejected(ordered, alpha_);
}

std::size_t SimesDiscoveryBounds::lower_bound(const std::vector<std::size_t>& indices) const {
  auto selected = indices;
  std::sort(selected.begin(), selected.end());
  if ((!selected.empty() && selected.back() >= p_.size()) ||
      std::adjacent_find(selected.begin(), selected.end()) != selected.end())
    throw std::invalid_argument("Simes discovery bounds: invalid subset indices");
  if (nonrejected_size_ == 0)
    return selected.size();
  std::sort(selected.begin(), selected.end(), [&](std::size_t a, std::size_t b) { return p_[a] < p_[b]; });
  std::size_t count = 0, bound = 0;
  for (std::size_t rank = 0; rank < selected.size(); ++rank) {
    const long double cutoff = static_cast<long double>(alpha_) * (rank + 1);
    while (count < selected.size() &&
           static_cast<long double>(nonrejected_size_) * p_[selected[count]] <= cutoff)
      ++count;
    if (count > rank)
      bound = std::max(bound, count - rank);
  }
  return bound;
}

} // namespace datamunge::stats
