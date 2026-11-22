/// @file
/// @brief Sequential screening, rank comparisons and adaptive replication ratios.

#include <datamunge/optim/simulation_search.hpp>

#include <datamunge/stats/nonparametric.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {
/// @brief Sample statistics and finite-candidate allocation internals.
namespace {

/// @brief Online sample count, mean and second central moment.
struct Moments {
  std::size_t count{}; ///< Number of observations.
  long double mean{}; ///< Online sample mean.
  long double total{}; ///< Unnormalized sum of squared mean deviations.

  /// @brief Add one scalar sample using Welford's recurrence.
  /// @param value Observed value.
  void add(long double value) {
    ++count;
    const auto delta = value - mean;
    mean += delta / count;
    total += delta * (value - mean);
  }

  /// @brief Return the unbiased variance estimate where defined.
  /// @return Nonnegative variance or zero with fewer than two observations.
  long double variance() const {
    return count > 1 ? std::max(0.0L, total / (count - 1)) : 0;
  }
};

/// @brief Compute the next OCBA allocation by target deficit.
/// @param statistics Pilot and adaptive moments, all candidates sampled.
/// @param total Completed replication count.
/// @return Index of the candidate furthest below its target.
std::size_t choose_ocba(const std::vector<Moments>& statistics, std::size_t total) {
  const auto winner = std::min_element(statistics.begin(), statistics.end(),
      [](const Moments& a, const Moments& b) { return a.mean < b.mean; }) - statistics.begin();
  const auto n = statistics.size();
  long double magnitude = 0;
  for (const auto& s : statistics) {
    magnitude = std::max(magnitude, std::abs(s.mean));
    magnitude = std::max(magnitude, std::sqrt(s.variance()));
  }
  if (magnitude == 0) magnitude = 1;
  constexpr long double floor = 1e-12L;
  std::vector<long double> allocation(n);
  long double square = 0, sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (i == static_cast<std::size_t>(winner)) continue;
    const auto deviation = std::max(floor, std::sqrt(statistics[i].variance()) / magnitude);
    const auto gap = std::max(floor, statistics[i].mean / magnitude - statistics[winner].mean / magnitude);
    const auto ratio = deviation / gap;
    allocation[i] = ratio * ratio;
    square += std::pow(allocation[i] / deviation, 2);
    sum += allocation[i];
  }
  allocation[winner] = std::max(floor, std::sqrt(statistics[winner].variance()) / magnitude) * std::sqrt(square);
  sum += allocation[winner];
  std::size_t next = winner;
  long double largest = -std::numeric_limits<long double>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    const long double deficit = (static_cast<long double>(total) + 1) * allocation[i] / sum - statistics[i].count;
    if (deficit > largest) {
      largest = deficit;
      next = i;
    }
  }
  return next;
}

/// @brief Estimate initial variance of every paired Kim–Nelson difference.
/// @param pilot Initial complete paired blocks in original candidate order.
/// @return Symmetric lookup of paired sample variances.
std::vector<std::vector<long double>> paired_variance(const std::vector<std::vector<double>>& pilot) {
  const auto n = pilot.front().size();
  std::vector result(n, std::vector<long double>(n));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      Moments differences;
      for (const auto& block : pilot) differences.add(static_cast<long double>(block[i]) - block[j]);
      result[i][j] = result[j][i] = differences.variance();
    }
  }
  return result;
}

/// @brief Compute active treatments' average within-block ranks.
/// @param blocks Active-candidate outcomes, one row per complete block.
/// @return Mean ranks in active-candidate order.
std::vector<long double> rank_means(const std::vector<std::vector<double>>& blocks) {
  const auto n = blocks.front().size();
  std::vector<long double> means(n);
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  for (const auto& block : blocks) {
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return block[a] < block[b]; });
    std::size_t i = 0;
    while (i < n) {
      auto end = i + 1;
      while (end < n && block[order[end]] == block[order[i]]) ++end;
      const auto rank = 0.5L * (static_cast<long double>(i) + 1 + end) / blocks.size();
      for (; i < end; ++i) means[order[i]] += rank;
    }
  }
  return means;
}

/// @brief Mark treatments separated from the best mean rank after an omnibus screen.
/// @param history All completed outcome blocks, with rows indexed by candidate identifier.
/// @param active Current active identifiers.
/// @param alpha Test level.
/// @return Elimination flags in active-candidate order.
std::vector<bool> friedman_screen(const std::vector<std::vector<double>>& history,
                                  const std::vector<std::size_t>& active, double alpha) {
  std::vector<std::vector<double>> subgroups;
  subgroups.reserve(history.size());
  bool variation = false;
  for (const auto& block : history) {
    auto& row = subgroups.emplace_back();
    row.reserve(active.size());
    for (const auto candidate : active) {
      row.push_back(block[candidate]);
      variation = variation || row.back() != row.front();
    }
  }
  std::vector<bool> remove(active.size());
  if (!variation || stats::friedman_test(subgroups).p_value >= alpha) return remove;
  const auto means = rank_means(subgroups);
  const auto best = *std::ranges::min_element(means);
  const auto k = static_cast<long double>(active.size());
  const auto se = std::sqrt(k * (k + 1) / (6 * history.size()));
  for (std::size_t i = 0; i < means.size(); ++i) {
    const auto z = (means[i] - best) / se;
    remove[i] = std::erfc(z / std::sqrt(2.0L)) < alpha / (k - 1);
  }
  return remove;
}

} // namespace

SimulationSearchResult simulation_search(const SimulationObjective& simulation, std::size_t candidates,
                                          const SimulationSearchOptions& options) {
  const auto method = options.method;
  const bool ocba = method == SimulationSearchMethod::ocba;
  const bool kn = method == SimulationSearchMethod::kim_nelson;
  const bool confidence = method == SimulationSearchMethod::confidence_race;
  const bool friedman = method == SimulationSearchMethod::friedman_race;
  if (!simulation || candidates == 0 || options.evaluations == 0 || options.pilot < 2 ||
      !(ocba || kn || confidence || friedman) ||
      (!ocba && !(options.alpha > 0 && options.alpha < 0.5)) ||
      (kn && !(std::isfinite(options.indifference) && options.indifference > 0)) ||
      (confidence && !(std::isfinite(options.lower_loss) && std::isfinite(options.upper_loss) &&
                       options.lower_loss < options.upper_loss))) {
    throw std::invalid_argument("Invalid simulation search objective or controls");
  }

  SimulationSearchResult result;
  std::vector<Moments> statistics(candidates);
  std::vector<std::size_t> active(candidates);
  std::iota(active.begin(), active.end(), std::size_t{0});
  std::mt19937_64 rng(options.seed);
  /// Score one actual simulation replication and update its estimate.
  auto evaluate = [&](std::size_t i, std::uint64_t seed) {
    const auto value = simulation(i, seed);
    if (!std::isfinite(value) || (confidence && (value < options.lower_loss || value > options.upper_loss))) {
      throw std::domain_error("Simulation outcome must be finite and inside any configured bounds");
    }
    ++result.evaluations;
    statistics[i].add(value);
    return value;
  };

  std::vector<std::vector<double>> history;
  std::vector<std::vector<long double>> variances;
  const auto pilot = options.pilot;
  long double h2 = 0;
  if (kn && candidates > 1) {
    const long double log_error = std::log(2.0L * options.alpha) - std::log(static_cast<long double>(candidates - 1));
    h2 = (pilot - 1) * std::expm1(-2 * log_error / (pilot - 1));
  }

  std::size_t rounds = 0;
  while (result.evaluations < options.evaluations) {
    if (ocba && rounds >= pilot) {
      const auto next = candidates == 1 ? 0 : choose_ocba(statistics, result.evaluations);
      evaluate(next, rng());
      continue;
    }
    const auto shared_seed = ocba ? std::uint64_t{0} : rng();
    std::vector<double> block;
    if (friedman || (kn && rounds < pilot)) block.resize(candidates);
    bool complete = true;
    for (const auto i : active) {
      if (result.evaluations == options.evaluations) {
        complete = false;
        break;
      }
      const auto value = evaluate(i, ocba ? rng() : shared_seed);
      if (!block.empty()) block[i] = value;
    }
    if (!complete) break;
    ++rounds;
    if (!block.empty()) history.push_back(std::move(block));
    if (rounds < pilot || ocba || active.size() < 2) continue;
    if (kn && rounds == pilot) {
      variances = paired_variance(history);
      history.clear();
    }

    std::vector<bool> remove(active.size());
    if (friedman) {
      remove = friedman_screen(history, active, options.alpha);
    } else if (confidence) {
      const long double offset = std::log(2.0L * candidates) + std::log(static_cast<long double>(rounds)) +
          std::log(static_cast<long double>(rounds) + 1) - std::log(options.alpha);
      const auto radius = (static_cast<long double>(options.upper_loss) - options.lower_loss) *
          std::sqrt(offset / (2.0L * rounds));
      long double best = std::numeric_limits<long double>::infinity();
      for (const auto i : active) best = std::min(best, statistics[i].mean);
      for (std::size_t i = 0; i < active.size(); ++i) remove[i] = statistics[active[i]].mean - best > 2 * radius;
    } else {
      const long double delta = options.indifference;
      for (std::size_t i = 0; i < active.size(); ++i) {
        const auto a = active[i];
        for (const auto b : active) {
          if (a == b) continue;
          const auto radius = variances[a][b] == 0 ? 0.0L
              : std::max(0.0L, (h2 * variances[a][b] / (rounds * delta) - delta) / 2);
          if (statistics[a].mean - statistics[b].mean > radius) {
            remove[i] = true;
            break;
          }
        }
      }
    }

    std::vector<std::size_t> retained;
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (!remove[i]) retained.push_back(active[i]);
    }
    active = std::move(retained);
    if (active.size() == 1) {
      result.screened_to_one = true;
      break;
    }
  }
  result.estimates.resize(candidates);
  std::vector<bool> alive(candidates);
  for (const auto i : active) alive[i] = true;
  for (std::size_t i = 0; i < candidates; ++i) {
    const auto& s = statistics[i];
    result.estimates[i] = {s.count, s.count == 0 ? std::numeric_limits<double>::quiet_NaN()
                                               : static_cast<double>(s.mean), alive[i]};
    if (alive[i] && s.count != 0 && (!result.selected || s.mean < statistics[*result.selected].mean)) {
      result.selected = i;
    }
  }
  return result;
}

} // namespace datamunge::optim
