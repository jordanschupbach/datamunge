/// @file
/// @brief Mutual-information clustering and binary subset mixing.

#include <datamunge/optim/linkage_search.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::optim {
/// @brief Binary population and linkage estimation internals.
namespace {

/// @brief Candidate with cached objective and streak of non-improving generations.
struct Member {
  CategoricalPoint point; ///< Bit coordinates.
  double value; ///< Scored objective.
  std::size_t stale{}; ///< Consecutive generations without strict improvement.
};

/// @brief Compute empirical mutual information between two bit positions.
/// @param population Binary population providing counts.
/// @param a First bit index.
/// @param b Second bit index.
/// @return Mutual information in nats.
double mutual_information(const std::vector<Member>& population, std::size_t a, std::size_t b) {
  std::array<std::array<double, 2>, 2> counts{};
  std::array<double, 2> first{}, second{};
  for (const auto& member : population) {
    const auto x = member.point[a], y = member.point[b];
    ++counts[x][y];
    ++first[x];
    ++second[y];
  }
  const auto n = static_cast<double>(population.size());
  double info = 0;
  for (std::size_t x = 0; x < 2; ++x) {
    for (std::size_t y = 0; y < 2; ++y) {
      if (counts[x][y] != 0) info += counts[x][y] / n * std::log(counts[x][y] * n / (first[x] * second[y]));
    }
  }
  return info;
}

/// @brief Learn the singleton and internal linkage subsets with average mutual-information linkage.
/// @param dimension Number of binary coordinates.
/// @param population Current population used for linkage estimation.
/// @return Subset tree excluding the full root for dimension greater than one.
std::vector<std::vector<std::size_t>> subsets(std::size_t dimension, const std::vector<Member>& population) {
  std::vector<std::vector<double>> links(dimension, std::vector<double>(dimension));
  std::vector<std::vector<std::size_t>> active(dimension);
  for (std::size_t i = 0; i < dimension; ++i) {
    active[i] = {i};
    for (std::size_t j = 0; j < i; ++j) links[i][j] = links[j][i] = mutual_information(population, i, j);
  }
  auto tree = active;
  for (std::size_t remaining = dimension; remaining > 2; --remaining) {
    std::size_t a = dimension, b = dimension;
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < dimension; ++i) {
      if (active[i].empty()) continue;
      for (std::size_t j = i + 1; j < dimension; ++j) {
        if (!active[j].empty() && links[i][j] > best) {
          a = i;
          b = j;
          best = links[i][j];
        }
      }
    }
    const auto fraction = static_cast<double>(active[b].size()) / (active[b].size() + active[a].size());
    for (std::size_t i = 0; i < dimension; ++i) {
      if (i == a || i == b || active[i].empty()) continue;
      links[a][i] = links[i][a] = std::lerp(links[a][i], links[b][i], fraction);
    }
    active[a].insert(active[a].end(), active[b].begin(), active[b].end());
    active[b].clear();
    tree.push_back(active[a]);
  }
  return tree;
}

} // namespace

DiscreteDistributionResult linkage_search(const CategoricalObjective& objective, std::size_t dimension,
                                           const LinkageSearchOptions& options) {
  if (!objective || dimension == 0 || options.evaluations == 0 || options.population < 2 ||
      (options.method != LinkageMixingMethod::parent_pair && options.method != LinkageMixingMethod::gene_pool)) {
    throw std::invalid_argument("Invalid linkage search objective or controls");
  }
  if (dimension > std::numeric_limits<std::size_t>::max() / dimension / sizeof(double)) {
    throw std::length_error("Linkage matrix exceeds size limits");
  }
  DiscreteDistributionResult result{{}, std::numeric_limits<double>::infinity(), 0};
  std::mt19937_64 rng(options.seed);
  std::bernoulli_distribution bit;
  /// Score one complete bit vector and retain the historical incumbent.
  auto evaluate = [&](CategoricalPoint point) {
    const auto loss = objective(point);
    if (!std::isfinite(loss)) throw std::domain_error("Linkage search objective must be finite");
    ++result.evaluations;
    if (loss < result.value) {
      result.value = loss;
      result.point = point;
    }
    return Member{std::move(point), loss};
  };
  /// Draw independent fair bits for initialization or diversification.
  const auto random_point = [&] {
    CategoricalPoint point(dimension);
    for (auto& value : point) value = bit(rng);
    return point;
  };

  const auto size = std::min(options.population, options.evaluations);
  std::vector<Member> population;
  for (std::size_t i = 0; i < size; ++i) population.push_back(evaluate(random_point()));
  const bool gene_pool = options.method == LinkageMixingMethod::gene_pool;
  const auto stall_limit = 1 + static_cast<std::size_t>(std::log10(static_cast<double>(size)));
  while (result.evaluations < options.evaluations) {
    const auto before = result.evaluations;
    auto tree = subsets(dimension, population);
    auto next = population;
    std::uniform_int_distribution<std::size_t> donors(0, size - 2);
    for (std::size_t i = 0; i < size && result.evaluations < options.evaluations; ++i) {
      auto& child = next[i];
      /// Draw one cohort member other than the receiver.
      const auto donor_index = [&] {
        const auto candidate = donors(rng);
        return candidate + (candidate >= i);
      };
      /// Transfer one subset and compare the resulting score when it changed.
      const auto mix = [&](const CategoricalPoint& donor, const std::vector<std::size_t>& subset, bool strict) {
        auto point = child.point;
        for (const auto index : subset) point[index] = donor[index];
        if (point == child.point) return false;
        auto candidate = evaluate(std::move(point));
        if (candidate.value < child.value || (!strict && candidate.value == child.value)) {
          child.point = std::move(candidate.point);
          child.value = candidate.value;
          return true;
        }
        return false;
      };
      const auto fixed = gene_pool ? i : donor_index();
      std::shuffle(tree.begin(), tree.end(), rng);
      bool changed = false;
      for (const auto& subset : tree) {
        if (result.evaluations == options.evaluations) break;
        changed |= mix(population[gene_pool ? donor_index() : fixed].point, subset, false);
      }
      child.stale = child.value < population[i].value ? 0 : population[i].stale + 1;
      if (gene_pool && (!changed || child.stale > stall_limit) && result.evaluations < options.evaluations) {
        const auto elite = result.point;
        const auto elite_loss = result.value;
        bool improved = false;
        for (const auto& subset : tree) {
          if (result.evaluations == options.evaluations) break;
          if (mix(elite, subset, true)) {
            improved = true;
            break;
          }
        }
        if (!improved && elite_loss < child.value) {
          child.point = elite;
          child.value = elite_loss;
        }
      }
    }
    if (result.evaluations == before) {
      const auto worst = std::ranges::max_element(next, {}, &Member::value);
      *worst = evaluate(random_point());
    }
    population = std::move(next);
  }
  return result;
}

} // namespace datamunge::optim
