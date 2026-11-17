/// @file laplacian_smoothing.cpp
/// @brief Complete reciprocal-graph checks and shared actual Jacobi/energy producer with optional history.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/laplacian_smoothing.hpp>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
namespace datamunge::geometry {
/// @brief Local complete graph metadata, finite conversion and actual update helpers.
namespace laplacian_detail {
/// @brief Add a logical actual count without overflow.
/// @param count Mutable logical count.
/// @param amount Additional entries/operations.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& count, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("Laplacian count overflow");
  count += amount;
}
/// @brief Complete validated original graph sizes used for upfront capture admission.
struct Geometry {
  std::size_t entries{0}; ///< Complete original adjacency multiplicity E.
  std::size_t updates{0}; ///< Complete free nonisolated vertex updates per actual sweep R.
};
/// @brief Validate complete finite undirected adjacency/pin/parameter geometry before access/copies.
/// @param vertices Borrowed original finite points.
/// @param neighbors Complete original reciprocal adjacency.
/// @param fixed Complete0..n original pin flags.
/// @param lambda Actual relaxation factor.
/// @param iterations Actual requested sweep count.
/// @return Checked original E and per-sweep free nonisolated count.
/// @throws std::invalid_argument On complete input/parameter/reciprocity violations.
/// @throws std::length_error On vertex/adjacency count limits.
/// @throws std::bad_alloc On reciprocal-validation map allocation failure.
Geometry validate(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<char>&             fixed,
    double                               lambda,
    int                                  iterations) {
  const auto n = vertices.size();
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("Laplacian vertex ids");
  if (neighbors.size() != n || fixed.size() > n || !std::isfinite(lambda) || lambda < 0 ||
      lambda > 1 || iterations < 0)
    throw std::invalid_argument("Laplacian shape/parameters");
  for (auto p : vertices)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("Laplacian finite vertices");
  std::map<std::pair<int, int>, std::size_t> counts;
  Geometry                                   g;
  for (std::size_t i = 0; i < n; ++i) {
    add(g.entries, neighbors[i].size());
    if ((i >= fixed.size() || !fixed[i]) && !neighbors[i].empty())
      ++g.updates;
    for (int j : neighbors[i]) {
      if (j < 0 || static_cast<std::size_t>(j) >= n)
        throw std::invalid_argument("Laplacian neighbor id");
      add(counts[{static_cast<int>(i), j}], 1);
    }
  }
  for (const auto& [edge, count] : counts) {
    const auto it = counts.find({edge.second, edge.first});
    if (it == counts.end() || it->second != count)
      throw std::invalid_argument("Laplacian reciprocal multiplicity");
  }
  return g;
}
/// @brief Convert one widened finite result to a finite representable coordinate/energy double.
/// @param value Actual widened arithmetic result.
/// @return Rounded finite double.
/// @throws std::overflow_error On nonfinite/out-of-double-range arithmetic.
double coordinate(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
    throw std::overflow_error("Laplacian finite double range");
  return static_cast<double>(value);
}
/// @brief Evaluate actual widened undirected squared-length energy without graph-validation replay.
/// @param vertices Validated complete producing state.
/// @param neighbors Validated original reciprocal graph.
/// @param work Optional actual trace diagnostic counter.
/// @return Finite widened actual energy, before optional double rounding.
/// @throws std::length_error On diagnostic count overflow.
/// @throws std::overflow_error On nonfinite widened energy arithmetic.
long double energy(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    LaplacianWork*                       work) {
  long double sum = 0;
  for (std::size_t i = 0; i < vertices.size(); ++i)
    for (int j : neighbors[i])
      if (static_cast<std::size_t>(j) > i) {
        const long double dx = static_cast<long double>(vertices[i].x) - vertices[j].x,
                          dy = static_cast<long double>(vertices[i].y) - vertices[j].y;
        sum += dx * dx + dy * dy;
        if (!std::isfinite(sum))
          throw std::overflow_error("Laplacian energy range");
        if (work)
          add(work->energy_edges, 1);
      }
  return sum;
}
/// @brief Preserve an ordinary finite coordinate relaxation or use a widened convex fallback.
/// @param before Actual original coordinate.
/// @param mean Actual rounded finite centroid coordinate.
/// @param lambda Validated[0,1] relaxation.
/// @param widened Mutable flag set if fallback is actually used.
/// @return Actual finite next coordinate.
/// @throws std::overflow_error On nonfinite/unrepresentable widened fallback.
double blend(double before, double mean, double lambda, bool& widened) {
  const double delta = mean - before, step = lambda * delta, value = before + step;
  if (std::isfinite(delta) && std::isfinite(step) && std::isfinite(value))
    return value;
  widened = true;
  return coordinate(
      (1 - static_cast<long double>(lambda)) * before + static_cast<long double>(lambda) * mean);
}
/// @brief Execute actual old-state-read/new-state-write sweeps, optionally capturing their real boundaries.
/// @param vertices Validated original points, borrowed unchanged.
/// @param neighbors Validated reciprocal original graph.
/// @param fixed Validated original short/full pin mask.
/// @param lambda Actual validated relaxation.
/// @param iterations Actual nonnegative complete pass count.
/// @param trace Optional fully preadmitted producing observation output.
/// @return Owned final state and actual selected work.
/// @throws std::length_error On actual work/container limits.
/// @throws std::overflow_error On coordinate/diagnostic numeric failure.
/// @throws std::bad_alloc On state/capture append allocation failure.
LaplacianAnalysis run(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<char>&             fixed,
    double                               lambda,
    int                                  iterations,
    LaplacianTrace*                      trace) {
  LaplacianAnalysis    out;
  std::vector<Point2D> v = vertices, next = vertices;
  const auto           n = v.size();
  if (trace) {
    trace->states.push_back(v);
    trace->energies.push_back(energy(v, neighbors, &out.work));
  }
  for (int it = 0; it < iterations; ++it) {
    std::copy(v.begin(), v.end(), next.begin());
    for (std::size_t i = 0; i < n; ++i) {
      if (i < fixed.size() && fixed[i]) {
        add(out.work.pinned_skips, 1);
        continue;
      }
      if (neighbors[i].empty()) {
        add(out.work.isolated_skips, 1);
        continue;
      }
      double      cx = 0, cy = 0;
      long double wx = 0, wy = 0;
      for (int j : neighbors[i]) {
        cx += v[j].x;
        cy += v[j].y;
        wx += static_cast<long double>(v[j].x);
        wy += static_cast<long double>(v[j].y);
        add(out.work.neighbor_reads, 1);
      }
      const bool   wide = !std::isfinite(cx) || !std::isfinite(cy);
      const double inv  = 1. / static_cast<double>(neighbors[i].size());
      Point2D      center{
          std::isfinite(cx) ? cx * inv : coordinate(wx / neighbors[i].size()),
          std::isfinite(cy) ? cy * inv : coordinate(wy / neighbors[i].size())};
      bool wide_update = false;
      next[i]          = {
          blend(v[i].x, center.x, lambda, wide_update),
          blend(v[i].y, center.y, lambda, wide_update)};
      add(out.work.updated_vertices, 1);
      if (trace)
        trace->updates.push_back(
            {static_cast<std::size_t>(it),
             i,
             neighbors[i].size(),
             v[i],
             center,
             next[i],
             wide,
             wide_update});
    }
    v.swap(next);
    add(out.work.sweeps, 1);
    if (trace) {
      trace->states.push_back(v);
      trace->energies.push_back(energy(v, neighbors, &out.work));
    }
  }
  out.vertices = std::move(v);
  return out;
}
} // namespace laplacian_detail
std::vector<Point2D> laplacian_smooth(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<char>&             fixed,
    double                               lambda,
    int                                  iterations) {
  laplacian_detail::validate(vertices, neighbors, fixed, lambda, iterations);
  return laplacian_detail::run(vertices, neighbors, fixed, lambda, iterations, nullptr).vertices;
}
double dirichlet_energy(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors) {
  laplacian_detail::validate(vertices, neighbors, {}, .5, 0);
  return laplacian_detail::coordinate(laplacian_detail::energy(vertices, neighbors, nullptr));
}
LaplacianAnalysis laplacian_smooth_analyze(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<char>&             fixed,
    double                               lambda,
    int                                  iterations) {
  laplacian_detail::validate(vertices, neighbors, fixed, lambda, iterations);
  return laplacian_detail::run(vertices, neighbors, fixed, lambda, iterations, nullptr);
}
LaplacianTrace laplacian_smooth_trace(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<char>&             fixed,
    double                               lambda,
    int                                  iterations,
    std::size_t                          max_entries) {
  const auto  g       = laplacian_detail::validate(vertices, neighbors, fixed, lambda, iterations);
  const auto  n       = vertices.size();
  std::size_t initial = n;
  laplacian_detail::add(initial, n);
  laplacian_detail::add(initial, g.entries);
  laplacian_detail::add(initial, 1);
  std::size_t per = n;
  laplacian_detail::add(per, 1);
  laplacian_detail::add(per, g.updates);
  if (static_cast<std::size_t>(iterations) > std::numeric_limits<std::size_t>::max() / per)
    throw std::length_error("Laplacian complete trajectory geometry");
  std::size_t capture = initial;
  laplacian_detail::add(capture, static_cast<std::size_t>(iterations) * per);
  if (max_entries && capture > max_entries)
    throw std::length_error("Laplacian trace quota");
  LaplacianTrace out;
  out.neighbors = neighbors;
  out.fixed.assign(n, 0);
  std::copy(fixed.begin(), fixed.end(), out.fixed.begin());
  out.lambda           = lambda;
  out.captured_entries = capture;
  out.states.reserve(static_cast<std::size_t>(iterations) + 1);
  out.energies.reserve(static_cast<std::size_t>(iterations) + 1);
  out.updates.reserve(static_cast<std::size_t>(iterations) * g.updates);
  auto a     = laplacian_detail::run(vertices, neighbors, fixed, lambda, iterations, &out);
  out.result = std::move(a.vertices);
  out.work   = a.work;
  return out;
}
} // namespace datamunge::geometry
