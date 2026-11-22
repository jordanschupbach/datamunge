/// @file
/// @brief Discrete ant-colony tour construction and pheromone updates.

#include <datamunge/optim/ant_colony.hpp>
#include <datamunge/optim/pareto_ant_colony.hpp>
#include <datamunge/optim/nsga2.hpp>

#include "tour_cost_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::optim {
/// @brief Implementation details for symmetric tour construction.
namespace {

/// @brief A scored tour in the current ant batch.
struct Tour {
  std::vector<std::size_t> cities; ///< Permutation with implicit closing edge.
  long double length{}; ///< Sum of all edges, including the closing edge.
};

/// @brief Determine the minimum edge scale while checking the input.
/// @param distances Complete symmetric distance matrix.
/// @param options Ant colony controls.
/// @return Smallest positive off-diagonal edge cost.
/// @throws std::invalid_argument Invalid input or search controls.
long double validate(const std::vector<std::vector<double>>& distances,
                     const AntColonyOptions& options) {
  const auto n = distances.size();
  if (n < 3 || options.evaluations == 0 || options.ants == 0 ||
      !(options.alpha >= 0 && std::isfinite(options.alpha)) ||
      !(options.beta >= 0 && std::isfinite(options.beta)) ||
      !(options.evaporation > 0 && options.evaporation < 1)) {
    throw std::invalid_argument("Invalid ant colony size, budget or transition controls");
  }
  switch (options.policy) {
    case AntColonyPolicy::ant_system:
      break;
    case AntColonyPolicy::ant_colony_system:
      if (!(options.exploitation >= 0 && options.exploitation <= 1 &&
            options.local_decay >= 0 && options.local_decay <= 1)) {
        throw std::invalid_argument("Invalid ACS probability or local decay");
      }
      break;
    case AntColonyPolicy::max_min:
      if (!(options.min_trail_ratio > 0 && options.min_trail_ratio <= 1)) {
        throw std::invalid_argument("Invalid MMAS trail ratio");
      }
      break;
    case AntColonyPolicy::rank_based:
      if (options.rank_count == 0) throw std::invalid_argument("Rank count must be positive");
      break;
    case AntColonyPolicy::elitist:
      if (!(options.elite_weight >= 0 && std::isfinite(options.elite_weight))) {
        throw std::invalid_argument("Invalid elitist deposit weight");
      }
      break;
    default:
      throw std::invalid_argument("Unknown ant colony policy");
  }
  return detail::validate_tour_costs(distances);
}

/// @brief Apply a trail update to both directions of an edge.
/// @param trail Mutable symmetric trail matrix.
/// @param i First city.
/// @param j Second city.
/// @param retain Multiplier of the old pheromone.
/// @param deposit New pheromone to add.
void update_edge(std::vector<std::vector<long double>>& trail, std::size_t i, std::size_t j,
                 long double retain, long double deposit) {
  trail[i][j] = std::max(std::numeric_limits<long double>::min(), retain * trail[i][j] + deposit);
  trail[j][i] = trail[i][j];
}

/// @brief Apply the same update to all edges in a cycle.
/// @param trail Mutable symmetric pheromone matrix.
/// @param tour A complete cycle.
/// @param retain Retention multiplier.
/// @param deposit Pheromone to add on each edge.
void update_tour(std::vector<std::vector<long double>>& trail, const Tour& tour,
                 long double retain, long double deposit) {
  auto previous = tour.cities.back();
  for (const auto city : tour.cities) {
    update_edge(trail, previous, city, retain, deposit);
    previous = city;
  }
}

/// @brief Construct and score a nearest-neighbor tour from city zero.
/// @param distances Valid complete graph.
/// @return Feasible tour used for initializing ACS.
Tour nearest_tour(const std::vector<std::vector<double>>& distances) {
  Tour tour{{0}, 0};
  std::vector<bool> used(distances.size());
  used[0] = true;
  while (tour.cities.size() < distances.size()) {
    const auto from = tour.cities.back();
    auto next = distances.size();
    for (std::size_t j = 0; j < distances.size(); ++j) {
      if (!used[j] && (next == distances.size() || distances[from][j] < distances[from][next])) next = j;
    }
    tour.length += distances[from][next];
    used[next] = true;
    tour.cities.push_back(next);
  }
  tour.length += distances[tour.cities.back()][0];
  return tour;
}

/// @brief Construct and score an ant tour, including ACS local updates.
/// @param distances Valid complete graph.
/// @param trail Symmetric positive pheromone matrix.
/// @param options Valid search controls.
/// @param initial Starting ACS trail, the target for local updates.
/// @param rng Borrowed local engine.
/// @return Complete scored cycle.
Tour construct_tour(const std::vector<std::vector<double>>& distances,
                    std::vector<std::vector<long double>>& trail,
                    const AntColonyOptions& options, long double initial, std::mt19937_64& rng) {
  const auto n = distances.size();
  const auto start = std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
  const bool acs = options.policy == AntColonyPolicy::ant_colony_system;
  const long double alpha = acs ? 1 : options.alpha;
  const long double beta = options.beta;
  std::uniform_real_distribution<double> uniform;
  Tour tour{{start}, 0};
  tour.cities.reserve(n);
  std::vector<bool> used(n);
  used[start] = true;
  std::vector<std::size_t> candidates;
  std::vector<long double> log_weights;
  std::vector<double> weights;
  candidates.reserve(n);
  log_weights.reserve(n);
  weights.reserve(n);
  while (tour.cities.size() < n) {
    const auto from = tour.cities.back();
    candidates.clear();
    log_weights.clear();
    for (std::size_t j = 0; j < n; ++j) {
      if (used[j]) continue;
      candidates.push_back(j);
      log_weights.push_back(alpha * std::log(trail[from][j]) -
                            beta * std::log(static_cast<long double>(distances[from][j])));
    }
    const auto best = std::max_element(log_weights.begin(), log_weights.end());
    auto index = static_cast<std::size_t>(best - log_weights.begin());
    if (!(acs && uniform(rng) < options.exploitation)) {
      weights.clear();
      for (const auto value : log_weights) weights.push_back(static_cast<double>(std::exp(value - *best)));
      index = std::discrete_distribution<std::size_t>(weights.begin(), weights.end())(rng);
    }
    const auto next = candidates[index];
    used[next] = true;
    tour.cities.push_back(next);
    tour.length += distances[from][next];
    if (acs) update_edge(trail, from, next, 1 - options.local_decay, options.local_decay * initial);
  }
  tour.length += distances[tour.cities.back()][start];
  if (acs) {
    update_edge(trail, tour.cities.back(), start, 1 - options.local_decay, options.local_decay * initial);
  }
  return tour;
}

} // namespace

AntColonyResult ant_colony_tour(const std::vector<std::vector<double>>& distances,
                               const AntColonyOptions& options) {
  const auto scale = validate(distances, options);
  const auto n = distances.size();
  const auto policy = options.policy;
  const bool acs = policy == AntColonyPolicy::ant_colony_system;
  const bool max_min = policy == AntColonyPolicy::max_min;
  Tour best{{}, std::numeric_limits<long double>::infinity()};
  std::size_t evaluations = 0, stagnant = 0, restarts = 0;
  long double initial = 1;
  const long double evaporation = options.evaporation;
  if (acs) {
    best = nearest_tour(distances);
    ++evaluations;
    initial = scale / best.length;
  } else if (max_min) {
    initial = 1 / evaporation;
  }
  std::vector trail(n, std::vector<long double>(n, initial));
  std::mt19937_64 rng(options.seed);
  std::vector<Tour> batch;
  batch.reserve(std::min(options.ants, options.evaluations - evaluations));
  while (evaluations < options.evaluations) {
    batch.clear();
    const auto count = std::min(options.ants, options.evaluations - evaluations);
    const auto previous_best = best.length;
    for (std::size_t ant = 0; ant < count; ++ant) {
      batch.push_back(construct_tour(distances, trail, options, initial, rng));
      if (batch.back().length < best.length) best = batch.back();
    }
    evaluations += count;
    if (acs) {
      update_tour(trail, best, 1 - evaporation, evaporation * n * scale / best.length);
      continue;
    }

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) update_edge(trail, i, j, 1 - evaporation, 0);
    }
    if (max_min || policy == AntColonyPolicy::rank_based) {
      std::stable_sort(batch.begin(), batch.end(), [](const Tour& a, const Tour& b) {
        return a.length < b.length;
      });
    }

    if (max_min) {
      update_tour(trail, batch.front(), 1, n * scale / batch.front().length);
      const auto maximum = n * scale / best.length / evaporation;
      const auto minimum = maximum * options.min_trail_ratio;
      stagnant = best.length < previous_best ? 0 : stagnant + 1;
      const bool restart = options.restart_after > 0 && stagnant >= options.restart_after &&
                           evaluations < options.evaluations;
      for (auto& row : trail) {
        for (auto& entry : row) entry = restart ? maximum : std::clamp(entry, minimum, maximum);
      }
      if (restart) {
        stagnant = 0;
        ++restarts;
      }
    } else if (policy == AntColonyPolicy::rank_based) {
      for (std::size_t rank = 0; rank < std::min(batch.size(), options.rank_count - 1); ++rank) {
        const auto weight = options.rank_count - rank - 1;
        update_tour(trail, batch[rank], 1, static_cast<long double>(weight) * n * scale / batch[rank].length);
      }
      update_tour(trail, best, 1, static_cast<long double>(options.rank_count) * n * scale / best.length);
    } else {
      for (const auto& tour : batch) update_tour(trail, tour, 1, n * scale / tour.length);
      if (policy == AntColonyPolicy::elitist) {
        update_tour(trail, best, 1, static_cast<long double>(options.elite_weight) * n * scale / best.length);
      }
    }
  }
  return {std::move(best.cities), static_cast<double>(best.length), evaluations, restarts};
}

/// @brief Pareto ranking view for an ant batch or archive.
namespace {

/// @brief Scale objectives for numerically robust crowding and create ranker input.
/// @param tours Evaluated tours in arrival order.
/// @return Points with normalized objective coordinates and empty decision metadata.
std::vector<ParetoPoint> rank_view(const std::vector<ParetoTour>& tours) {
  std::vector<ParetoPoint> points(tours.size());
  for (const auto& tour : tours) {
    for (const auto value : tour.objectives) {
      if (!std::isfinite(value)) throw std::domain_error("Nonfinite Pareto tour length");
    }
  }
  for (std::size_t j = 0; j < tours.front().objectives.size(); ++j) {
    auto lo = tours.front().objectives[j], hi = lo;
    for (const auto& tour : tours) {
      lo = std::min(lo, tour.objectives[j]);
      hi = std::max(hi, tour.objectives[j]);
    }
    const auto span = hi - lo;
    for (std::size_t i = 0; i < tours.size(); ++i) {
      const auto value = tours[i].objectives[j];
      double normalized = 0;
      if (hi != lo) {
        normalized = std::isfinite(span) ? (value - lo) / span
                                        : (value / 2 - lo / 2) / (hi / 2 - lo / 2);
      }
      points[i].objectives.push_back(normalized);
    }
  }
  return points;
}

} // namespace

ParetoAntResult pareto_ant_colony_tour(const std::vector<std::vector<std::vector<double>>>& costs,
                                      const ParetoAntOptions& options) {
  if (costs.size() < 2 || options.evaluations == 0 || options.ants == 0 || options.capacity == 0 ||
      !(options.alpha >= 0 && std::isfinite(options.alpha) && options.beta >= 0 && std::isfinite(options.beta) &&
        options.evaporation > 0 && options.evaporation < 1)) {
    throw std::invalid_argument("Invalid Pareto ant controls or objective count");
  }
  const auto n = costs.front().size();
  auto normalized = costs;
  for (std::size_t i = 0; i < costs.size(); ++i) {
    const auto& matrix = costs[i];
    if (matrix.size() != n) throw std::invalid_argument("Pareto tour objective shapes must match");
    detail::validate_tour_costs(matrix);
    double maximum = 0;
    for (const auto& row : matrix) maximum = std::max(maximum, *std::max_element(row.begin(), row.end()));
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t b = 0; b < n; ++b) {
        if (a != b) normalized[i][a][b] = std::max(1e-300, matrix[a][b] / maximum);
      }
    }
  }
  std::mt19937_64 rng(options.seed);
  std::exponential_distribution<double> exponential;
  std::vector trail(n, std::vector<long double>(n, 1));
  AntColonyOptions construction;
  construction.alpha = options.alpha;
  construction.beta = options.beta;
  std::vector<double> weights(costs.size());
  std::vector heuristic(n, std::vector<double>(n));
  ParetoAntResult result{{}, 0};
  while (result.evaluations < options.evaluations) {
    const auto count = std::min(options.ants, options.evaluations - result.evaluations);
    std::vector<ParetoTour> batch;
    batch.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      for (auto& weight : weights) weight = std::max(1e-12, exponential(rng));
      const auto total = std::accumulate(weights.begin(), weights.end(), 0.0);
      for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a + 1; b < n; ++b) {
          double value = 0;
          for (std::size_t j = 0; j < weights.size(); ++j) value += weights[j] / total * normalized[j][a][b];
          heuristic[a][b] = heuristic[b][a] = value;
        }
      }
      auto route = construct_tour(heuristic, trail, construction, 1, rng);
      std::vector<double> objectives;
      for (const auto& matrix : costs) objectives.push_back(static_cast<double>(detail::tour_cost(matrix, route.cities)));
      batch.push_back({std::move(route.cities), std::move(objectives)});
      ++result.evaluations;
    }

    const auto ranking = nsga2_rank(rank_view(batch));
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t b = a + 1; b < n; ++b) update_edge(trail, a, b, 1 - options.evaporation, 0);
    }
    for (std::size_t i = 0; i < batch.size(); ++i) {
      update_tour(trail, {batch[i].tour, 0}, 1, 1.0L / (ranking.ranks[i] + 1) / count);
      const auto& candidate = batch[i];
      if (std::ranges::any_of(result.front, [&](const auto& point) {
            return point.objectives == candidate.objectives || dominates(point.objectives, candidate.objectives);
          })) continue;
      std::erase_if(result.front, [&](const auto& point) {
        return dominates(candidate.objectives, point.objectives);
      });
      result.front.push_back(candidate);
    }
    if (result.front.size() > options.capacity) {
      const auto diversity = nsga2_rank(rank_view(result.front));
      std::vector<std::size_t> indices(result.front.size());
      std::iota(indices.begin(), indices.end(), std::size_t{0});
      std::stable_sort(indices.begin(), indices.end(),
                       [&](auto a, auto b) { return diversity.crowding[a] > diversity.crowding[b]; });
      std::vector<ParetoTour> kept;
      for (std::size_t i = 0; i < options.capacity; ++i) kept.push_back(std::move(result.front[indices[i]]));
      result.front = std::move(kept);
    }
    for (const auto& guide : result.front) {
      update_tour(trail, {guide.tour, 0}, 1, 1.0L / result.front.size());
    }
  }
  return result;
}

} // namespace datamunge::optim
