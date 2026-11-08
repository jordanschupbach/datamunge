/// @file floyd_warshall.cpp
/// @brief Shared checked in-place sweeps, actual snapshots and bounded next-hop queries.
#include <cmath>
#include <datamunge/algorithms/floyd_warshall.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private validation, checked admission and canonical sweep implementation.
namespace {
/// @brief Positive infinity denotes unreachable pairs, never an input edge weight.
constexpr double infinity = std::numeric_limits<double>::infinity();
/// @brief Multiply two work/entry quantities without unsigned wrap.
/// @param a First factor.
/// @param b Second factor.
/// @return Representable product.
/// @throws std::length_error Product exceeds size_t.
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("Floyd-Warshall: product overflow");
  return a * b;
}
/// @brief Add two work/entry quantities without unsigned wrap.
/// @param a First quantity.
/// @param b Second quantity.
/// @return Representable sum.
/// @throws std::length_error Sum exceeds size_t.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Floyd-Warshall: sum overflow");
  return a + b;
}
/// @brief Checked full-sweep dimensions prepared before matrix allocation.
struct Dimensions {
  std::size_t square; ///< Cells in one matrix and maximum i-row guard count.
  std::size_t cube;   ///< Maximum inner comparisons and improvements.
};
/// @brief Validate all input edges, dimensions and ordinary matrix capacities.
/// @param n Vertex count.
/// @param edges Borrowed directed records.
/// @return Checked square/cube work bounds.
/// @throws std::invalid_argument Invalid endpoint or nonfinite edge weight.
/// @throws std::length_error Matrix/work size or container capacity exhausted.
Dimensions prepare(
    std::size_t                                                      n,
    const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges) {
  for (const auto& [from, to, weight] : edges) {
    if (from >= n || to >= n)
      throw std::invalid_argument("floyd_warshall: edge endpoint out of range");
    if (!std::isfinite(weight))
      throw std::invalid_argument("floyd_warshall: edge weights must be finite");
  }
  const auto square = product(n, n), cube = product(square, n);
  if (n > std::vector<std::vector<double>>().max_size() ||
      n > std::vector<std::vector<std::size_t>>().max_size() ||
      n > std::vector<double>().max_size() || n > std::vector<std::size_t>().max_size())
    throw std::length_error("Floyd-Warshall: matrix container limit");
  return {square, cube};
}
/// @brief Validate complete external/native matrix shapes and every stored value.
/// @param r Read-only result record; global cycle flag is not trusted.
/// @return Matrix dimension.
/// @throws std::invalid_argument Invalid shapes, distance values or hop IDs.
std::size_t validate_matrices(const AllPairsShortestPaths& r) {
  const auto n = r.distance.size();
  if (r.next.size() != n)
    throw std::invalid_argument("reconstruct_path: next matrix dimension mismatch");
  for (const auto& row : r.distance)
    if (row.size() != n)
      throw std::invalid_argument("reconstruct_path: distance matrix not square");
  for (const auto& row : r.next)
    if (row.size() != n)
      throw std::invalid_argument("reconstruct_path: next matrix not square");
  for (const auto& row : r.distance)
    for (double d : row)
      if (!std::isfinite(d) && d != infinity)
        throw std::invalid_argument("reconstruct_path: invalid distance");
  for (const auto& row : r.next)
    for (auto v : row)
      if (v != kNoVertex && v >= n)
        throw std::invalid_argument("reconstruct_path: hop out of range");
  return n;
}
/// @brief Check whether one validated pair connects through a negative diagonal.
/// @param r Validated raw result.
/// @param i Valid source vertex.
/// @param j Valid destination vertex.
/// @return True exactly when some negative-diagonal vertex connects the pair.
bool unbounded(const AllPairsShortestPaths& r, std::size_t i, std::size_t j) {
  for (std::size_t k = 0; k < r.distance.size(); ++k)
    if (r.distance[k][k] < 0 && r.distance[i][k] != infinity && r.distance[k][j] != infinity)
      return true;
  return false;
}
/// @brief Record the actual current matrices only in the recording specialization.
/// @tparam Record Whether full stage storage is enabled.
/// @param t Owned result under construction.
/// @param admitted Number of fully admitted intermediates.
/// @throws std::bad_alloc Snapshot copy allocation fails.
/// @throws std::length_error Snapshot container capacity exhausted.
template<bool Record>
void stage(FloydWarshallTrace& t, std::size_t admitted) {
  if constexpr (Record)
    t.stages.push_back({admitted, t.result.distance, t.result.next});
}
/// @brief Execute the historical strict k/i/j recurrence once with optional evidence.
/// @tparam Record Whether to copy graph/stages and append actual improvements.
/// @param n Vertex count.
/// @param edges Borrowed finite input graph.
/// @param cap Complete upper trace entry cap, ignored without recording.
/// @return Owned raw result/work and optional actual evidence.
/// @throws std::invalid_argument Invalid endpoints or nonfinite weights.
/// @throws std::length_error Work/matrix/capture bound or container capacity exhausted.
/// @throws std::overflow_error Actual double addition becomes nonfinite.
/// @throws std::bad_alloc Matrices, graph copies or history allocation fails.
template<bool Record>
FloydWarshallTrace kernel(
    std::size_t                                                      n,
    const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
    std::size_t                                                      cap) {
  const auto         dimensions = prepare(n, edges);
  FloydWarshallTrace t;
  if constexpr (Record) {
    const auto bound = floyd_warshall_trace_entries(n, edges.size());
    if ((cap && bound > cap) || add(n, 1) > t.stages.max_size() ||
        dimensions.cube > t.relaxations.max_size())
      throw std::length_error("Floyd-Warshall: complete trace entry cap or container limit");
    t.edges = edges;
    t.stages.reserve(n + 1);
  }
  auto& r        = t.result;
  auto& w        = t.work;
  w.vertices     = n;
  w.edges        = edges.size();
  w.matrix_cells = dimensions.square;
  r.distance.assign(n, std::vector<double>(n, infinity));
  r.next.assign(n, std::vector<std::size_t>(n, kNoVertex));
  for (std::size_t i = 0; i < n; ++i)
    r.distance[i][i] = 0.;
  for (const auto& [from, to, weight] : edges)
    if (weight < r.distance[from][to]) {
      r.distance[from][to] = weight;
      r.next[from][to]     = to;
      ++w.seed_improvements;
    }
  stage<Record>(t, 0);
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t i = 0; i < n; ++i) {
      ++w.row_checks;
      if (r.distance[i][k] == infinity) {
        ++w.skipped_rows;
        continue;
      }
      for (std::size_t j = 0; j < n; ++j) {
        ++w.column_checks;
        if (r.distance[k][j] == infinity) {
          ++w.skipped_columns;
          continue;
        }
        const double left = r.distance[i][k], right = r.distance[k][j], candidate = left + right;
        if (!std::isfinite(candidate))
          throw std::overflow_error("Floyd-Warshall: nonfinite candidate sum");
        ++w.candidates;
        if (candidate < r.distance[i][j]) {
          const auto hop = r.next[i][k];
          if constexpr (Record)
            t.relaxations.push_back(
                {k, i, j, r.distance[i][j], left, right, candidate, r.next[i][j], hop});
          r.distance[i][j] = candidate;
          r.next[i][j]     = hop;
          ++w.relaxations;
        }
      }
    }
    stage<Record>(t, k + 1);
  }
  for (std::size_t i = 0; i < n; ++i) {
    ++w.diagonal_checks;
    if (r.distance[i][i] < 0) {
      r.has_negative_cycle = true;
      break;
    }
  }
  return t;
}
} // namespace
AllPairsShortestPaths floyd_warshall(
    std::size_t                                                      n,
    const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges) {
  return kernel<false>(n, edges, 0).result;
}
FloydWarshallOutcome floyd_warshall_work(
    std::size_t                                                      n,
    const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges) {
  auto t = kernel<false>(n, edges, 0);
  return {std::move(t.result), t.work};
}
std::size_t floyd_warshall_trace_entries(std::size_t n, std::size_t edge_count) {
  const auto square = product(n, n), cube = product(square, n),
             matrix = add(product(2, n), product(2, square));
  return add(add(edge_count, matrix), add(product(add(n, 1), add(1, matrix)), cube));
}
FloydWarshallTrace floyd_warshall_trace(
    std::size_t                                                      n,
    const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
    std::size_t                                                      cap) {
  return kernel<true>(n, edges, cap);
}
FloydWarshallPairStatus
floyd_warshall_pair_status(const AllPairsShortestPaths& r, std::size_t i, std::size_t j) {
  const auto n = validate_matrices(r);
  if (i >= n || j >= n)
    throw std::invalid_argument("Floyd-Warshall: query endpoint out of range");
  if (unbounded(r, i, j))
    return FloydWarshallPairStatus::Unbounded;
  return r.distance[i][j] == infinity ? FloydWarshallPairStatus::Unreachable
                                      : FloydWarshallPairStatus::Finite;
}
std::vector<std::vector<bool>> floyd_warshall_unbounded_pairs(const AllPairsShortestPaths& r) {
  const auto n = validate_matrices(r);
  (void)product(n, n);
  std::vector<std::vector<bool>> mask(n, std::vector<bool>(n));
  for (std::size_t k = 0; k < n; ++k)
    if (r.distance[k][k] < 0)
      for (std::size_t i = 0; i < n; ++i)
        if (r.distance[i][k] != infinity)
          for (std::size_t j = 0; j < n; ++j)
            if (r.distance[k][j] != infinity)
              mask[i][j] = true;
  return mask;
}
std::vector<std::size_t>
reconstruct_path(const AllPairsShortestPaths& r, std::size_t i, std::size_t j) {
  const auto n = validate_matrices(r);
  if (i >= n || j >= n)
    throw std::invalid_argument("reconstruct_path: endpoint out of range");
  if (unbounded(r, i, j))
    throw std::domain_error("reconstruct_path: negative-cycle unbounded pair");
  if (r.distance[i][j] == infinity)
    return {};
  if (i == j)
    return {i};
  if (r.next[i][j] == kNoVertex)
    return {};
  std::vector<std::size_t> path{i};
  auto                     current = i;
  for (std::size_t moves = 0; moves < n; ++moves) {
    current = r.next[current][j];
    if (current == kNoVertex)
      return {};
    path.push_back(current);
    if (current == j)
      return path;
  }
  throw std::invalid_argument("reconstruct_path: cyclic next-hop chain");
}
std::vector<std::tuple<std::size_t, std::size_t, double>> floyd_warshall_example_edges() {
  return {
      {0, 1, 3},
      {0, 2, 8},
      {0, 4, -4},
      {1, 3, 1},
      {1, 4, 7},
      {2, 1, 4},
      {3, 0, 2},
      {3, 2, -5},
      {4, 3, 6}};
}
} // namespace datamunge::algorithms
