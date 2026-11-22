/// @file
/// @brief Pareto search loops using adaptive grids and canonical real variation.

#include <datamunge/optim/pareto_search.hpp>

#include "pareto_selection_detail.hpp"
#include "boxed_search_detail.hpp"
#include "differential_evolution_detail.hpp"

#include <map>
#include <utility>

namespace datamunge::optim {
/// @brief Grid archive and bounded population mechanics.
namespace {

/// @brief Scale a finite objective component to the grid range.
/// @param value Candidate objective value.
/// @param lo Observed minimum value.
/// @param hi Observed maximum value.
/// @return Scaled component in [0,1].
double normalize(double value, double lo, double hi) {
  if (lo == hi) return 0;
  value = std::clamp(value, lo, hi);
  const auto span = hi - lo;
  if (std::isfinite(span)) return (value - lo) / span;
  return (value / 2 - lo / 2) / (hi / 2 - lo / 2);
}

/// @brief Grid indices across objective axes.
using Cell = std::vector<std::size_t>;

/// @brief Adaptive grid snapshot across an archive's objective ranges.
struct Grid {
  std::size_t divisions; ///< Number of bins per objective.
  std::vector<double> lo; ///< Minimum components in the archive.
  std::vector<double> hi; ///< Maximum components in the archive.
  std::map<Cell, std::vector<std::size_t>> occupants; ///< Member indices for each occupied cell.

  /// @brief Build a grid from a nonempty nondominated set.
  /// @param points Archive points.
  /// @param bins Divisions per axis.
  Grid(const std::vector<ParetoPoint>& points, std::size_t bins)
      : divisions(bins), lo(points.front().objectives), hi(lo) {
    for (const auto& point : points) {
      for (std::size_t j = 0; j < lo.size(); ++j) {
        lo[j] = std::min(lo[j], point.objectives[j]);
        hi[j] = std::max(hi[j], point.objectives[j]);
      }
    }
    for (std::size_t i = 0; i < points.size(); ++i) occupants[cell(points[i])].push_back(i);
  }

  /// @brief Compute the clipped grid cell for one scored point.
  /// @param point Point with a compatible finite objective vector.
  /// @return Integer bin index for each objective.
  Cell cell(const ParetoPoint& point) const {
    Cell result(lo.size());
    const auto scale = static_cast<long double>(divisions);
    for (std::size_t j = 0; j < lo.size(); ++j) {
      const auto position = normalize(point.objectives[j], lo[j], hi[j]) * scale;
      result[j] = position >= scale ? divisions - 1 : static_cast<std::size_t>(position);
    }
    return result;
  }

  /// @brief Count occupants of one point's cell.
  /// @param point Point with a compatible finite objective vector.
  /// @return Cell occupancy or zero for an empty cell.
  std::size_t density(const ParetoPoint& point) const {
    const auto found = occupants.find(cell(point));
    return found == occupants.end() ? 0 : found->second.size();
  }

  /// @brief Draw a sparse-region leader.
  /// @param roulette Use inverse-occupancy weights when true, two-region tournament otherwise.
  /// @param rng Borrowed local engine.
  /// @return Archive index of one uniformly chosen member of the selected region.
  std::size_t leader(bool roulette, std::mt19937_64& rng) const {
    std::vector<const std::vector<std::size_t>*> occupied;
    std::vector<double> weights;
    for (const auto& [key, indices] : occupants) {
      occupied.push_back(&indices);
      weights.push_back(1.0 / indices.size());
    }
    std::size_t choice;
    if (roulette) {
      choice = std::discrete_distribution<std::size_t>(weights.begin(), weights.end())(rng);
    } else {
      std::uniform_int_distribution<std::size_t> select(0, occupied.size() - 1);
      const auto a = select(rng), b = select(rng);
      choice = weights[a] >= weights[b] ? a : b;
    }
    const auto& candidates = *occupied[choice];
    return candidates[std::uniform_int_distribution<std::size_t>(0, candidates.size() - 1)(rng)];
  }
};

/// @brief Insert a nondominated point and prune a crowded cell if necessary.
/// @param archive Owned nondominated archive mutated in place.
/// @param trial Scored trial with matching objective count.
/// @param options Archive capacity and grid controls.
/// @param rng Borrowed local engine used for crowding ties.
/// @return Whether the trial survives in the archive.
bool insert(std::vector<ParetoPoint>& archive, const ParetoPoint& trial,
            const ParetoSearchOptions& options, std::mt19937_64& rng) {
  for (const auto& point : archive) {
    if (point.objectives == trial.objectives || dominates(point.objectives, trial.objectives)) return false;
  }
  std::erase_if(archive, [&](const ParetoPoint& point) {
    return dominates(trial.objectives, point.objectives);
  });
  archive.push_back(trial);
  if (archive.size() <= options.capacity) return true;
  const Grid grid(archive, options.grid_divisions);
  std::vector<std::size_t> crowded;
  std::size_t density = 0;
  for (const auto& [cell, occupants] : grid.occupants) {
    if (occupants.size() < density) continue;
    if (occupants.size() > density) {
      crowded.clear();
      density = occupants.size();
    }
    crowded.insert(crowded.end(), occupants.begin(), occupants.end());
  }
  const auto removed = crowded[std::uniform_int_distribution<std::size_t>(0, crowded.size() - 1)(rng)];
  const bool survives = removed + 1 != archive.size();
  archive.erase(archive.begin() + removed);
  return survives;
}

/// @brief Score normalized positions and keep a fixed objective count.
/// @param function Borrowed objective.
/// @param lower Valid lower box bounds.
/// @param upper Valid upper box bounds.
/// @param point Normalized coordinates, retained in the returned point.
/// @param expected Number of objectives, initialized on the first call.
/// @return Evaluated normalized point.
/// @throws std::invalid_argument Invalid objective shape.
/// @throws std::domain_error Objective returns nonfinite values.
ParetoPoint evaluate(MultiObjectiveFunction& function, const std::vector<double>& lower,
                     const std::vector<double>& upper, std::vector<double> point, std::size_t& expected) {
  std::vector<double> original(point.size());
  for (std::size_t j = 0; j < point.size(); ++j) original[j] = std::lerp(lower[j], upper[j], point[j]);
  auto values = function.evaluate(original);
  if (values.size() < 2 || (expected != 0 && values.size() != expected)) {
    throw std::invalid_argument("Pareto search objective must have a fixed shape of at least two");
  }
  if (!std::ranges::all_of(values, [](double value) { return std::isfinite(value); })) {
    throw std::domain_error("Pareto search objective components must be finite");
  }
  expected = values.size();
  return {std::move(point), std::move(values)};
}

/// @brief Scale each objective column for diversity and indicator calculations.
/// @param members Scored points in a common objective space.
/// @return Copy with objective columns normalized to [0,1].
std::vector<ParetoPoint> normalized(const std::vector<ParetoPoint>& members) {
  auto scaled = members;
  for (std::size_t j = 0; j < members.front().objectives.size(); ++j) {
    auto lo = members.front().objectives[j], hi = lo;
    for (const auto& member : members) {
      lo = std::min(lo, member.objectives[j]);
      hi = std::max(hi, member.objectives[j]);
    }
    for (auto& member : scaled) member.objectives[j] = normalize(member.objectives[j], lo, hi);
  }
  return scaled;
}

/// @brief Trim a combined population to its target size using the selected rule.
/// @param members Mutable joint population, retained in survival order.
/// @param target Maximum survivor count.
/// @param options Selection controls.
/// @param directions NSGA-III reference vectors.
/// @param rng Borrowed local random engine.
void truncate(std::vector<ParetoPoint>& members, std::size_t target, const ParetoSearchOptions& options,
              const std::vector<std::vector<double>>& directions, std::mt19937_64& rng) {
  if (members.size() <= target) return;
  const auto selected = detail::pareto_survivors(normalized(members), target, options, directions, rng);
  std::vector<ParetoPoint> kept;
  kept.reserve(target);
  for (const auto index : selected) kept.push_back(std::move(members[index]));
  members = std::move(kept);
}

} // namespace

ParetoSearchResult pareto_search(MultiObjectiveFunction& function, const std::vector<double>& lower,
                                const std::vector<double>& upper, const ParetoSearchOptions& options) {
  const auto dimension = lower.size();
  detail::validate_box(dimension, lower, upper, options.evaluations);
  const auto method = options.method;
  const bool paes = method == ParetoSearchMethod::paes;
  const bool pesa = method == ParetoSearchMethod::pesa2;
  const bool swarm = method == ParetoSearchMethod::mopso;
  const bool gde = method == ParetoSearchMethod::gde3;
  const bool nsga = method == ParetoSearchMethod::nsga3;
  const bool ibea = method == ParetoSearchMethod::ibea;
  const bool hype = method == ParetoSearchMethod::hype;
  const bool sms = method == ParetoSearchMethod::sms_emoa;
  const bool indicator_or_reference = nsga || ibea || hype || sms;
  const bool generational = gde || indicator_or_reference;
  if (!(paes || pesa || swarm || generational) ||
      (!(options.mutation_rate >= 0 && options.mutation_rate <= 1)) ||
      (!generational && (options.capacity == 0 || options.grid_divisions < 2)) ||
      (!paes && options.population < (gde ? 4u : 2u)) ||
      ((generational || pesa) && !(options.crossover_rate >= 0 && options.crossover_rate <= 1)) ||
      (gde && !(options.differential_weight > 0 && options.differential_weight <= 2)) ||
      (ibea && !(options.indicator_scale > 0 && std::isfinite(options.indicator_scale))) ||
      (hype && options.hypervolume_samples == 0)) {
    throw std::invalid_argument("Invalid Pareto search controls");
  }
  std::mt19937_64 rng(options.seed);
  std::uniform_real_distribution<double> uniform;
  const std::vector<double> zero(dimension), one(dimension, 1);
  const double mutation_rate = options.mutation_rate == 0 ? 1.0 / dimension : options.mutation_rate;
  std::size_t calls = 0, expected = 0;
  std::vector<std::vector<double>> directions;
  /// Score one real objective evaluation.
  auto observe = [&](std::vector<double> x) {
    auto result = evaluate(function, lower, upper, std::move(x), expected);
    if (calls == 0) {
      if (nsga) directions = detail::reference_directions(expected, options);
      if (sms && expected != 2) throw std::invalid_argument("SMS-EMOA requires exactly two objectives");
    }
    ++calls;
    return result;
  };

  const auto population_size = paes ? 1 : std::min(options.population, options.evaluations);
  std::vector<ParetoPoint> population, archive;
  population.reserve(population_size);
  for (std::size_t i = 0; i < population_size; ++i) {
    std::vector<double> x(dimension);
    for (auto& value : x) value = uniform(rng);
    population.push_back(observe(std::move(x)));
    if (!generational) insert(archive, population.back(), options, rng);
  }

  auto personal = swarm ? population : std::vector<ParetoPoint>{};
  auto velocities = swarm ? std::vector(population_size, zero) : std::vector<std::vector<double>>{};
  while (calls < options.evaluations) {
    if (paes) {
      auto& parent = population.front();
      auto x = parent.coordinates;
      detail::polynomial_mutation(x, zero, one, mutation_rate, 20, rng);
      auto trial = observe(std::move(x));
      if (dominates(parent.objectives, trial.objectives) || parent.objectives == trial.objectives) continue;
      if (!insert(archive, trial, options, rng)) continue;
      const Grid grid(archive, options.grid_divisions);
      if (dominates(trial.objectives, parent.objectives) || grid.density(trial) <= grid.density(parent)) {
        parent = std::move(trial);
      }
    } else if (gde) {
      std::vector<ParetoPoint> joined;
      for (std::size_t i = 0; i < population_size; ++i) {
        const auto& parent = population[i];
        if (calls == options.evaluations) {
          joined.push_back(parent);
          continue;
        }
        const auto donors = detail::random_distinct_indices(rng, population_size, i, 3);
        const auto axis = std::uniform_int_distribution<std::size_t>(0, dimension - 1)(rng);
        auto x = parent.coordinates;
        for (std::size_t j = 0; j < dimension; ++j) {
          if (j != axis && uniform(rng) >= options.crossover_rate) continue;
          x[j] = std::clamp(population[donors[0]].coordinates[j] + options.differential_weight *
                            (population[donors[1]].coordinates[j] - population[donors[2]].coordinates[j]), 0.0, 1.0);
        }
        auto trial = observe(std::move(x));
        if (!dominates(trial.objectives, parent.objectives)) joined.push_back(parent);
        if (!dominates(parent.objectives, trial.objectives) && parent.objectives != trial.objectives) {
          joined.push_back(std::move(trial));
        }
      }
      truncate(joined, population_size, options, directions, rng);
      population = std::move(joined);
    } else if (indicator_or_reference) {
      auto joined = population;
      const auto count = std::min(sms ? std::size_t{1} : population_size, options.evaluations - calls);
      const auto fitness = ibea || hype ? detail::pareto_mating_fitness(normalized(population), options, rng)
                                       : std::vector<double>{};
      std::uniform_int_distribution<std::size_t> draw(0, population_size - 1);
      /// Choose an unbiased parent or run a fitness tournament for indicator methods.
      const auto parent = [&] {
        const auto a = draw(rng);
        if (fitness.empty()) return a;
        const auto b = draw(rng);
        return fitness[a] >= fitness[b] ? a : b;
      };
      for (std::size_t i = 0; i < count;) {
        const auto a = parent(), b = parent();
        std::vector<double> first(dimension), second(dimension);
        detail::sbx_crossover(population[a].coordinates, population[b].coordinates, first, second,
                              zero, one, options.crossover_rate, 15, rng);
        detail::polynomial_mutation(first, zero, one, mutation_rate, 20, rng);
        joined.push_back(observe(std::move(first)));
        if (++i < count) {
          detail::polynomial_mutation(second, zero, one, mutation_rate, 20, rng);
          joined.push_back(observe(std::move(second)));
          ++i;
        }
      }
      truncate(joined, population_size, options, directions, rng);
      population = std::move(joined);
    } else {
      const auto count = std::min(population_size, options.evaluations - calls);
      const Grid grid(archive, options.grid_divisions);
      std::vector<ParetoPoint> trials;
      trials.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        const auto& guide = archive[grid.leader(swarm, rng)].coordinates;
        if (pesa) {
          const auto& other = archive[grid.leader(false, rng)].coordinates;
          std::vector<double> x(dimension), unused(dimension);
          detail::sbx_crossover(guide, other, x, unused, zero, one, options.crossover_rate, 15, rng);
          detail::polynomial_mutation(x, zero, one, mutation_rate, 20, rng);
          trials.push_back(observe(std::move(x)));
        } else {
          auto x = population[i].coordinates;
          for (std::size_t j = 0; j < dimension; ++j) {
            auto& velocity = velocities[i][j];
            velocity = std::clamp(0.4 * velocity + 1.5 * uniform(rng) * (personal[i].coordinates[j] - x[j]) +
                                  1.5 * uniform(rng) * (guide[j] - x[j]), -0.5, 0.5);
            x[j] += velocity;
            if (x[j] < 0 || x[j] > 1) velocity = -velocity;
            x[j] = std::clamp(x[j], 0.0, 1.0);
          }
          const double remaining = 1 - static_cast<double>(calls) / options.evaluations;
          detail::polynomial_mutation(x, zero, one, mutation_rate * remaining, 20, rng);
          auto trial = observe(std::move(x));
          if (!dominates(personal[i].objectives, trial.objectives) &&
              (dominates(trial.objectives, personal[i].objectives) || uniform(rng) < 0.5)) {
            personal[i] = trial;
          }
          population[i] = trial;
          trials.push_back(std::move(trial));
        }
      }
      for (const auto& trial : trials) insert(archive, trial, options, rng);
    }
  }

  if (generational) {
    for (const auto& point : non_dominated_front(population)) {
      if (std::ranges::none_of(archive, [&](const ParetoPoint& existing) {
            return existing.objectives == point.objectives;
          })) archive.push_back(point);
    }
  }
  for (auto& point : archive) {
    for (std::size_t j = 0; j < dimension; ++j) {
      point.coordinates[j] = std::lerp(lower[j], upper[j], point.coordinates[j]);
    }
  }
  return {std::move(archive), calls};
}

} // namespace datamunge::optim
