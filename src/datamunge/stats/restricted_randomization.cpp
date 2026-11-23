/// @file restricted_randomization.cpp
/// @brief Within-block assignment enumeration and sharp-null breakpoint inversion.

#include <datamunge/stats/restricted_randomization.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::stats {

/// @brief Block construction and assignment operations.
namespace {

/// @brief One block and the local indices selected by a reference assignment.
struct RandomizationBlock {
  std::vector<std::size_t> rows;       ///< Original row indices.
  std::vector<std::size_t> selected;   ///< Local positions assigned to treatment.
  std::vector<std::size_t> lottery;    ///< Reusable pool for drawing local positions.
};

/// @brief Organizes rows by first occurrence of each block label.
/// @param labels One block label per observation.
/// @return Blocks containing their original row positions.
std::vector<RandomizationBlock> build_blocks(const std::vector<std::size_t>& labels) {
  std::unordered_map<std::size_t, std::size_t> positions;
  std::vector<RandomizationBlock> blocks;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const auto [entry, inserted] = positions.try_emplace(labels[i], blocks.size());
    if (inserted)
      blocks.emplace_back();
    blocks[entry->second].rows.push_back(i);
  }
  return blocks;
}

/// @brief Advances a block's treated subset, resetting it when exhausted.
/// @param block Current combination and block size.
/// @return True if another combination was available.
bool advance(RandomizationBlock& block) {
  auto& chosen = block.selected;
  for (std::size_t j = chosen.size(); j > 0; --j) {
    const auto index = j - 1;
    if (chosen[index] < block.rows.size() - chosen.size() + index) {
      ++chosen[index];
      for (std::size_t i = j; i < chosen.size(); ++i)
        chosen[i] = chosen[i - 1] + 1;
      return true;
    }
  }
  std::iota(chosen.begin(), chosen.end(), std::size_t{0});
  return false;
}

/// @brief Draws one block's treated subset without replacement.
/// @param block Block storage and treated count.
/// @param rng Uniform integer generator.
void draw_block(RandomizationBlock& block, random::RandomGenerator& rng) {
  for (std::size_t j = 0; j < block.selected.size(); ++j) {
    const auto choice = static_cast<std::size_t>(rng.uniform_u64(j, block.lottery.size() - 1));
    std::swap(block.lottery[j], block.lottery[choice]);
    block.selected[j] = block.lottery[j];
  }
  std::sort(block.selected.begin(), block.selected.end());
}

/// @brief Converts a wide endpoint, rounding outwards to retain breakpoint ties.
/// @param value Endpoint in effect units.
/// @param lower Whether the bound is a lower endpoint.
/// @return Rounded endpoint, possibly infinite.
double outward_endpoint(long double value, bool lower) {
  constexpr auto inf = std::numeric_limits<double>::infinity();
  constexpr auto limit = std::numeric_limits<double>::max();
  if (value < -limit)
    return -inf;
  if (value > limit)
    return inf;
  const auto converted = static_cast<double>(value);
  if (lower ? converted > value : converted < value)
    return std::nextafter(converted, lower ? -inf : inf);
  return converted;
}

/// @brief Computes the proportion of extreme assignments with the selected correction.
/// @param extreme Number of reference assignments as extreme as observed.
/// @param total Number of reference assignments.
/// @param simulated Whether to include the observed assignment separately.
/// @return Exact or plus-one p-value.
double tail(std::size_t extreme, std::size_t total, bool simulated) {
  return static_cast<double>((static_cast<long double>(extreme) + simulated) /
                             (static_cast<long double>(total) + simulated));
}

} // namespace

RestrictedRandomization::RestrictedRandomization(const std::vector<double>& outcomes,
    const std::vector<bool>& treated, const std::vector<std::size_t>& blocks, std::size_t max_assignments)
    : RestrictedRandomization(outcomes, treated, blocks, max_assignments, nullptr) {}

RestrictedRandomization::RestrictedRandomization(const std::vector<double>& outcomes,
    const std::vector<bool>& treated, const std::vector<std::size_t>& blocks,
    std::size_t draws, random::RandomGenerator& rng)
    : RestrictedRandomization(outcomes, treated, blocks, draws, &rng) {}

RestrictedRandomization::RestrictedRandomization(const std::vector<double>& outcomes,
    const std::vector<bool>& treated, const std::vector<std::size_t>& labels,
    std::size_t limit, random::RandomGenerator* rng) : simulated_(rng != nullptr) {
  detail::check_sample(outcomes);
  if (treated.size() != outcomes.size() || labels.size() != outcomes.size() || limit == 0)
    throw std::invalid_argument("restricted randomization: invalid input dimensions or assignment count");
  auto blocks = build_blocks(labels);
  const long double n = outcomes.size();
  std::vector<long double> centered(outcomes.size());
  long double effect = 0;
  for (auto& block : blocks) {
    const long double origin = outcomes[block.rows.front()];
    long double sum_treated = 0, sum_control = 0;
    std::size_t nt = 0;
    for (const auto row : block.rows) {
      centered[row] = outcomes[row] - origin;
      if (treated[row]) {
        sum_treated += centered[row];
        ++nt;
      } else {
        sum_control += centered[row];
      }
    }
    if (nt == 0 || nt == block.rows.size())
      throw std::invalid_argument("restricted randomization: each block must contain both arms");
    const auto nc = block.rows.size() - nt;
    effect += (block.rows.size() / n) * (sum_treated / nt - sum_control / nc);
    block.selected.resize(nt);
    std::iota(block.selected.begin(), block.selected.end(), std::size_t{0});
    if (simulated_) {
      block.lottery.resize(block.rows.size());
      std::iota(block.lottery.begin(), block.lottery.end(), std::size_t{0});
    }
  }
  if (!std::isfinite(effect) || std::abs(effect) > std::numeric_limits<double>::max())
    throw std::overflow_error("restricted randomization: observed effect out of range");
  estimate_ = effect;
  for (std::size_t i = 0; i < centered.size(); ++i)
    if (treated[i])
      centered[i] -= effect;

  /// @brief Records the null effects for which the current assignment is at least as extreme.
  const auto record = [&] {
    long double contrast = 0, one_minus_b = 0, one_plus_b = 0;
    for (const auto& block : blocks) {
      const auto nt = block.selected.size(), nc = block.rows.size() - nt;
      const auto weight = block.rows.size() / n;
      long double sum_treated = 0, sum_control = 0;
      std::size_t moved = 0, chosen = 0;
      for (std::size_t j = 0; j < block.rows.size(); ++j) {
        const auto row = block.rows[j];
        if (chosen < nt && j == block.selected[chosen]) {
          sum_treated += centered[row];
          moved += !treated[row];
          ++chosen;
        } else {
          sum_control += centered[row];
        }
      }
      contrast += weight * (sum_treated / nt - sum_control / nc);
      one_minus_b += weight * (static_cast<long double>(moved) / nt + static_cast<long double>(moved) / nc);
      one_plus_b += weight * (static_cast<long double>(nt - moved) / nt + static_cast<long double>(nc - moved) / nc);
    }

    // At effect + delta the permuted contrast is contrast - b*delta and
    // the observed contrast is -delta. The two absolute-value crossings
    // bound one interval containing delta = 0. The observed assignment
    // and its possible complete complement tie at every effect.
    if (one_minus_b == 0 || one_plus_b == 0) {
      starts_.push_back(-std::numeric_limits<double>::infinity());
      ends_.push_back(std::numeric_limits<double>::infinity());
      return;
    }
    const auto a = effect - contrast / one_minus_b;
    const auto b = effect + contrast / one_plus_b;
    starts_.push_back(outward_endpoint(std::min(a, b), true));
    ends_.push_back(outward_endpoint(std::max(a, b), false));
  };

  if (simulated_) {
    for (std::size_t draw = 0; draw < limit; ++draw) {
      for (auto& block : blocks)
        draw_block(block, *rng);
      record();
    }
  } else {
    bool more;
    do {
      if (starts_.size() == limit)
        throw std::length_error("restricted randomization: exact assignment limit exceeded");
      record();
      more = false;
      for (auto& block : blocks) {
        if (advance(block)) {
          more = true;
          break;
        }
      }
    } while (more);
  }
  std::sort(starts_.begin(), starts_.end());
  std::sort(ends_.begin(), ends_.end());
}

HypothesisTestResult RestrictedRandomization::test(double null_effect) const {
  if (!std::isfinite(null_effect))
    throw std::invalid_argument("restricted randomization: null effect must be finite");
  const auto started = std::upper_bound(starts_.begin(), starts_.end(), null_effect) - starts_.begin();
  const auto ended = std::lower_bound(ends_.begin(), ends_.end(), null_effect) - ends_.begin();
  const auto extreme = static_cast<std::size_t>(started - ended);
  HypothesisTestResult result;
  const auto statistic = estimate_ - null_effect;
  const auto bound = std::numeric_limits<double>::max();
  result.statistic = std::abs(statistic) > bound ?
      std::copysign(std::numeric_limits<double>::infinity(), statistic) : static_cast<double>(statistic);
  result.estimate1 = static_cast<double>(estimate_);
  result.parameter1 = static_cast<double>(starts_.size());
  result.parameter2 = static_cast<double>(extreme);
  result.p_value = tail(extreme, starts_.size(), simulated_);
  result.method = simulated_ ? "Monte Carlo blocked randomization test" : "Exact blocked randomization test";
  return result;
}

std::pair<double, double> RestrictedRandomization::confidence_interval(double confidence_level) const {
  detail::check_confidence(confidence_level);
  const double alpha = 1 - confidence_level;
  std::size_t lo = 0, hi = starts_.size();
  while (lo < hi) {
    const auto count = std::midpoint(lo, hi);
    if (tail(count, starts_.size(), simulated_) > alpha)
      hi = count;
    else
      lo = count + 1;
  }
  if (lo == 0)
    return {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  return {starts_[lo - 1], ends_[ends_.size() - lo]};
}

} // namespace datamunge::stats
