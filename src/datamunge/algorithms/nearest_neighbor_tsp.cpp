/// @file nearest_neighbor_tsp.cpp
/// @brief Canonical nearest-city scan, actual histories and bounded permutation reference.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/nearest_neighbor_tsp.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete graph validation and shared cost/greedy producers.
namespace {
/// @brief Validate complete finite nonnegative matrix and represented scan geometry.
/// @param d Original complete edge matrix.
/// @param start Valid start when nonempty, ignored if empty.
/// @throws std::invalid_argument If shape/weights/start are invalid.
/// @throws std::length_error If n squared is unrepresentable.
void validate(const std::vector<std::vector<double>>& d, std::size_t start) {
  const auto n = d.size();
  for (auto& row : d)
    if (row.size() != n)
      throw std::invalid_argument("nearest TSP: square matrix required");
  for (auto& row : d)
    for (auto value : row)
      if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("nearest TSP: finite nonnegative distances required");
  if (n && start >= n)
    throw std::invalid_argument("nearest TSP: invalid start");
  if (n && n > std::numeric_limits<std::size_t>::max() / n)
    throw std::length_error("nearest TSP: matrix/scan geometry overflow");
}
/// @brief Admit actual owned history units before append/copy.
/// @param out Optional actual capture sink.
/// @param cap Actual quota, zero disables.
/// @param amount New owned units.
/// @throws std::length_error If count addition/quota exceeds its limit.
void admit(NearestTSPTrace* out, std::size_t cap, std::size_t amount) {
  if (!out)
    return;
  if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
      (cap && (out->captured_entries > cap || amount > cap - out->captured_entries)))
    throw std::length_error("nearest TSP: capture quota exceeded");
  out->captured_entries += amount;
}
/// @brief Sum a complete produced route in retained left-to-right double order.
/// @param route Actual closed route, empty allowed.
/// @param d Valid finite nonnegative matrix.
/// @param out Optional actual edge-sum sink.
/// @param cap Optional capture quota.
/// @param saturate True permits an overflowing candidate to return infinity for minimum selection.
/// @return Actual rounded route cost, optionally saturating overflowing candidates.
/// @throws std::overflow_error If any running cost is nonfinite.
/// @throws std::length_error If actual edge capture exceeds quota.
double cost(
    const std::vector<std::size_t>&         route,
    const std::vector<std::vector<double>>& d,
    NearestTSPTrace*                        out,
    std::size_t                             cap,
    bool                                    saturate = false) {
  double sum = 0;
  for (std::size_t i = 1; i < route.size(); ++i) {
    auto from = route[i - 1], to = route[i];
    auto weight = d[from][to];
    sum += weight;
    if (!std::isfinite(sum)) {
      if (saturate)
        return std::numeric_limits<double>::infinity();
      throw std::overflow_error("nearest TSP: nonfinite accumulated cost");
    }
    if (out) {
      admit(out, cap, 1);
      out->edges.push_back({from, to, weight, sum});
    }
  }
  return sum;
}
/// @brief Execute original nearest-unvisited selection with optional actual observations.
/// @param d Valid complete matrix.
/// @param start Valid nonempty start, ignored if empty.
/// @param out Optional actual-history sink.
/// @param cap Actual capture quota.
/// @return Actual greedy result/work.
NearestTSPAnalysis run(
    const std::vector<std::vector<double>>& d,
    std::size_t                             start,
    NearestTSPTrace*                        out,
    std::size_t                             cap) {
  NearestTSPAnalysis a;
  if (d.empty())
    return a;
  std::vector<bool> used(d.size());
  admit(out, cap, 1);
  a.result.tour.push_back(start);
  used[start] = true;
  while (a.result.tour.size() < d.size()) {
    const auto  from  = a.result.tour.back();
    auto        best  = d.size();
    const auto  first = out ? out->trials.size() : 0;
    std::size_t count = 0;
    for (std::size_t v = 0; v < d.size(); ++v) {
      ++a.candidate_checks;
      if (used[v])
        continue;
      const bool initial  = best == d.size();
      bool       improved = initial;
      if (!initial) {
        ++a.distance_comparisons;
        improved = d[from][v] < d[from][best];
      }
      if (improved)
        best = v;
      if (out) {
        admit(out, cap, 1);
        out->trials.push_back({from, v, d[from][v], initial, improved});
      }
      ++count;
    }
    if (out) {
      admit(out, cap, 1);
      out->choices.push_back({from, best, d[from][best], first, count});
    }
    used[best] = true;
    admit(out, cap, 1);
    a.result.tour.push_back(best);
  }
  admit(out, cap, 1);
  a.result.tour.push_back(start);
  a.result.cost  = cost(a.result.tour, d, out, cap);
  a.edges_summed = d.size();
  return a;
}
} // namespace
TourResult nearest_neighbor_tsp(const std::vector<std::vector<double>>& d, std::size_t start) {
  validate(d, start);
  return run(d, start, nullptr, 0).result;
}
NearestTSPAnalysis nearest_neighbor_tsp_analyze(
    const std::vector<std::vector<double>>& d,
    std::size_t                             start) {
  validate(d, start);
  return run(d, start, nullptr, 0);
}
NearestTSPTrace nearest_neighbor_tsp_trace(
    const std::vector<std::vector<double>>& d,
    std::size_t                             start,
    std::size_t                             cap) {
  validate(d, start);
  NearestTSPTrace t;
  admit(&t, cap, 1);
  admit(&t, cap, d.size());
  for (auto& row : d)
    admit(&t, cap, row.size());
  t.input    = d;
  t.start    = start;
  t.analysis = run(d, start, &t, cap);
  return t;
}
TourResult
enumerate_tsp(const std::vector<std::vector<double>>& d, std::size_t start, std::size_t cap) {
  validate(d, start);
  if (cap && d.size() > cap)
    throw std::length_error("enumerate_tsp: vertex cap exceeded");
  if (d.empty())
    return {};
  std::vector<std::size_t> rest;
  for (std::size_t i = 0; i < d.size(); ++i)
    if (i != start)
      rest.push_back(i);
  TourResult best;
  best.cost = std::numeric_limits<double>::infinity();
  do {
    std::vector<std::size_t> route{start};
    route.insert(route.end(), rest.begin(), rest.end());
    route.push_back(start);
    auto value = cost(route, d, nullptr, 0, true);
    if (value < best.cost) {
      best.tour = std::move(route);
      best.cost = value;
    }
  } while (std::next_permutation(rest.begin(), rest.end()));
  if (best.tour.empty())
    throw std::overflow_error("enumerate_tsp: every tour sum overflowed");
  return best;
}
std::vector<std::vector<double>> nearest_neighbor_tsp_adversary(double m) {
  if (!std::isfinite(m) || m < 2)
    throw std::invalid_argument("nearest_neighbor_tsp_adversary: finite M>=2 required");
  return {{0, 1, 2}, {1, 0, m}, {1, 1, 0}};
}
std::vector<std::pair<double, double>> nearest_neighbor_tsp_demo_points() {
  return {{0, 0}, {4, 0}, {4, 3}, {0, 3}, {1, 1}, {3, 2}};
}
} // namespace datamunge::algorithms
