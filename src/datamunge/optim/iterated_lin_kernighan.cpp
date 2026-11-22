/// @file
/// @brief Sequential edge exchanges and perturbation for symmetric cycle search.

#include <datamunge/optim/iterated_lin_kernighan.hpp>

#include <datamunge/optim/permutation_moves.hpp>
#include "tour_cost_detail.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::optim {
/// @brief Internal open-path representation and budget.
namespace {

/// @brief Symmetric edge with the smaller city first.
using Edge = std::pair<std::size_t, std::size_t>;

/// @brief Closed tour permutation or a path in visit order.
using Tour = std::vector<std::size_t>;

/// @brief Exact tour scores and edge cost references for a run.
struct Search {
  const std::vector<std::vector<double>>& distances; ///< Validated costs, borrowed.
  const LinKernighanOptions& options; ///< Borrowed controls.
  LinKernighanResult out; ///< Owned observed best tour and counts.
  long double record{std::numeric_limits<long double>::infinity()}; ///< Precise incumbent cost.

  /// @brief Score a closed tour after validating that work remains in the caller.
  /// @param tour Permutation with implicit closing edge.
  /// @return Exact accumulation of tour-edge costs in long double.
  long double evaluate(const Tour& tour) {
    const auto value = detail::tour_cost(distances, tour);
    ++out.evaluations;
    if (value < record) {
      record = value;
      out.tour = tour;
      out.length = static_cast<double>(value);
    }
    return value;
  }
};

/// @brief Follow sequential positive-gain exchanges from an open tour path.
/// @param search Shared budget and cost recorder.
/// @param path Open path from root to current tail.
/// @param removed Edges removed from the original cycle.
/// @param added Edges inserted in this sequence.
/// @param gain Sum of removed lengths less inserted lengths so far.
/// @param distance Current locally retained tour's loss, updated on success.
/// @param better Output of the first improving closed tour.
/// @return Whether the search found and fully scored a better tour.
bool exchange(Search& search, const Tour& path, std::vector<Edge>& removed, std::vector<Edge>& added,
              long double gain, long double& distance, Tour& better) {
  if (removed.size() >= std::min(search.options.max_depth, path.size())) return false;
  const auto tail = path.back();
  std::vector<std::size_t> candidates;
  for (std::size_t i = 1; i + 2 < path.size(); ++i) {
    const Edge y = std::minmax(tail, path[i]);
    const Edge x = std::minmax(path[i], path[i + 1]);
    if (gain <= search.distances[tail][path[i]] ||
        std::ranges::find(removed, y) != removed.end() ||
        std::ranges::find(added, x) != added.end()) continue;
    candidates.push_back(i);
  }
  std::stable_sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
    return search.distances[tail][path[a]] < search.distances[tail][path[b]];
  });
  const auto count = std::min(candidates.size(), search.options.candidate_limit);
  for (std::size_t j = 0; j < count && search.out.evaluations < search.options.evaluations; ++j) {
    const auto i = candidates[j];
    auto next = path;
    std::reverse(next.begin() + i + 1, next.end());
    added.push_back(std::minmax(tail, path[i]));
    removed.push_back(std::minmax(path[i], path[i+1]));
    const auto potential = gain - search.distances[tail][path[i]] + search.distances[path[i]][path[i+1]];
    const Edge close = std::minmax(next.front(), next.back());
    if (potential > search.distances[next.front()][next.back()] &&
        std::ranges::find(removed, close) == removed.end()) {
      const auto score = search.evaluate(next);
      if (score < distance) {
        distance = score;
        better = std::move(next);
        search.out.deepest_improvement = std::max(search.out.deepest_improvement, removed.size());
        return true;
      }
    }
    if (search.out.evaluations < search.options.evaluations &&
        exchange(search, next, removed, added, potential, distance, better)) return true;
    added.pop_back();
    removed.pop_back();
  }
  return false;
}

/// @brief Find the first locally improving variable-depth move.
/// @param search Shared evaluation cap.
/// @param tour Current closed tour, replaced on improvement.
/// @param distance Current scored loss, updated on improvement.
/// @return Whether the tour improved.
bool improve(Search& search, Tour& tour, long double& distance) {
  const auto n = tour.size();
  for (std::size_t start = 0; start < n && search.out.evaluations < search.options.evaluations; ++start) {
    auto path = tour;
    std::rotate(path.begin(), path.begin() + start, path.end());
    for (int direction = 0; direction < 2; ++direction) {
      if (search.out.evaluations == search.options.evaluations) return false;
      const Edge edge = std::minmax(path.front(), path.back());
      std::vector<Edge> removed{edge}, added;
      Tour better;
      if (exchange(search, path, removed, added, search.distances[edge.first][edge.second], distance, better)) {
        tour = std::move(better);
        return true;
      }
      std::reverse(path.begin() + 1, path.end());
    }
  }
  return false;
}

/// @brief Apply a random double bridge, using a two-edge kick when four separated cuts do not fit.
/// @param tour Valid permutation to perturb in place.
/// @param rng Borrowed local random engine.
void kick(Tour& tour, std::mt19937_64& rng) {
  if (tour.size() < 8) {
    apply_two_opt(tour, *sample_two_opt(tour.size(), rng));
    return;
  }
  std::uniform_int_distribution<std::size_t> draw(0, tour.size() - 1);
  std::array<std::size_t, 4> cuts;
  do {
    for (auto& cut : cuts) cut = draw(rng);
    std::ranges::sort(cuts);
  } while (cuts[1] - cuts[0] < 2 || cuts[2] - cuts[1] < 2 ||
           cuts[3] - cuts[2] < 2 || cuts[3] - cuts[0] > tour.size() - 2);
  const auto offset = cuts.front();
  for (auto& cut : cuts) cut -= offset;
  std::rotate(tour.begin(), tour.begin() + offset, tour.end());
  Tour rearranged;
  rearranged.reserve(tour.size());
  for (const auto segment : {0, 3, 2, 1}) {
    const auto end = segment == 3 ? tour.size() : cuts[segment + 1];
    rearranged.insert(rearranged.end(), tour.begin() + cuts[segment], tour.begin() + end);
  }
  tour.swap(rearranged);
}

} // namespace

LinKernighanResult iterated_lin_kernighan(const std::vector<std::vector<double>>& distances,
                                          std::vector<std::size_t> initial,
                                          const LinKernighanOptions& options) {
  detail::validate_tour_costs(distances);
  const auto n = distances.size();
  if (n < 4 || options.evaluations == 0 || options.max_depth < 2 || options.candidate_limit == 0) {
    throw std::invalid_argument("Invalid iterated Lin–Kernighan dimensions or controls");
  }
  std::mt19937_64 rng(options.seed);
  if (initial.empty()) {
    initial.resize(n);
    std::iota(initial.begin(), initial.end(), std::size_t{0});
    std::shuffle(initial.begin(), initial.end(), rng);
  } else {
    if (initial.size() != n) throw std::invalid_argument("Wrong initial tour length");
    std::vector<bool> visited(n);
    for (const auto city : initial) {
      if (city >= n || visited[city]) throw std::invalid_argument("Initial tour must be a permutation");
      visited[city] = true;
    }
  }
  Search search{distances, options, {{}, 0, 0, 0, 0}};
  auto current = std::move(initial);
  auto length = search.evaluate(current);
  while (search.out.evaluations < options.evaluations) {
    if (improve(search, current, length)) continue;
    if (search.out.evaluations == options.evaluations) break;
    current = search.out.tour;
    kick(current, rng);
    length = search.evaluate(current);
    ++search.out.kicks;
  }
  return std::move(search.out);
}

} // namespace datamunge::optim
