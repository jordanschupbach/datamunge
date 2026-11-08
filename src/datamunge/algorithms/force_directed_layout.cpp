/// @file force_directed_layout.cpp
/// @brief Shared direct layout kernel preserving original arithmetic and actual capture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/force_directed_layout.hpp>
#include <limits>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Checked implementation details for direct force layout only.
namespace force_directed_detail {
/// @brief Internal endpoint record type.
using Edge = std::pair<std::size_t, std::size_t>;
/// @brief Add nonnegative counts or throw before wrap.
/// @param a First count.
/// @param b Second count.
/// @return Exact representable sum.
/// @throws std::length_error Sum exceeds size_t.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("force layout count overflow");
  return a + b;
}
/// @brief Multiply nonnegative counts or throw before wrap.
/// @param a First factor.
/// @param b Second factor.
/// @return Exact representable product.
/// @throws std::length_error Product exceeds size_t.
std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b > SIZE_MAX / a)
    throw std::length_error("force layout count overflow");
  return a * b;
}
/// @brief Count unordered pairs without premature multiplication overflow.
/// @param n Vertex count.
/// @return n choose two.
/// @throws std::length_error Count cannot be represented.
std::size_t pairs(std::size_t n) {
  if (n < 2)
    return 0;
  return n % 2 ? mul(n, (n - 1) / 2) : mul(n / 2, n - 1);
}
/// @brief Reject nonfinite intermediate arithmetic, including values not eventually selected.
/// @param v Actual double or widened value.
/// @throws std::overflow_error Value is nonfinite.
template<class T>
void finite(T v) {
  if (!std::isfinite(v))
    throw std::overflow_error("force layout nonfinite arithmetic");
}
/// @brief Check both actual accumulator coordinates.
/// @param p Actual point or vector.
/// @throws std::overflow_error Either coordinate is nonfinite.
void finite_point(const Point2D& p) {
  finite(p.x);
  finite(p.y);
}
/// @brief Validate the complete endpoint list and optional supplied coordinates.
/// @param n Vertex count.
/// @param edges Endpoint records.
/// @param initial Null for seeded runs, otherwise supplied coordinates.
/// @return Number of nonloop records.
/// @throws std::invalid_argument An endpoint or coordinate is invalid.
std::size_t
validate(std::size_t n, const std::vector<Edge>& edges, const std::vector<Point2D>* initial) {
  std::size_t active = 0;
  for (auto [u, v] : edges) {
    if (u >= n || v >= n)
      throw std::invalid_argument("force layout endpoint out of range");
    active += u != v;
  }
  if (initial)
    for (auto p : *initial)
      if (!std::isfinite(p.x) || !std::isfinite(p.y))
        throw std::invalid_argument("force layout coordinate must be finite");
  return active;
}
/// @brief Check all actual work bounds before kernel allocation.
/// @param n Vertex count.
/// @param e Edge record count.
/// @param active Nonloop record count.
/// @param t Requested sweeps.
/// @param seeded Whether initialization draws two variates per vertex.
/// @throws std::length_error Work or scratch capacity cannot be represented.
void bounds(std::size_t n, std::size_t e, std::size_t active, std::size_t t, bool seeded) {
  if (n > std::vector<Point2D>().max_size())
    throw std::length_error("force layout vertex capacity");
  const auto sweeps = n >= 2 ? t : 0;
  mul(pairs(n), sweeps);
  mul(e, sweeps);
  mul(active, sweeps);
  mul(n, sweeps);
  if (seeded)
    mul(n, 2);
}
/// @brief Apply original double mean and radius normalization, retaining actual metadata.
/// @param p Owned raw positions, changed in place.
/// @param w Actual work metadata updated on success.
/// @throws std::overflow_error Actual sum, subtraction or radius becomes nonfinite.
void normalize(std::vector<Point2D>& p, ForceDirectedWork& w) {
  Point2D mean;
  for (const auto& v : p) {
    mean.x += v.x;
    mean.y += v.y;
    finite_point(mean);
  }
  mean.x /= p.size();
  mean.y /= p.size();
  double scale = 0.;
  for (auto& v : p) {
    v.x -= mean.x;
    v.y -= mean.y;
    finite_point(v);
    const double radius = std::hypot(v.x, v.y);
    finite(radius);
    scale = std::max(scale, radius);
  }
  if (scale > 0.)
    for (auto& v : p) {
      v.x /= scale;
      v.y /= scale;
      finite_point(v);
    }
  w.normalized = true;
  w.mean       = mean;
  w.scale      = scale;
}
/// @brief Run once with optional actual phase snapshots; no diagnostic computations or replay.
/// @param n Vertex count.
/// @param edges Validated in full before copies.
/// @param iterations Requested sweeps.
/// @param seed Local seed for seeded runs.
/// @param initial Null for seeded initialization, else finite supplied raw positions.
/// @param capture Whether owned actual history is requested.
/// @param cap Optional whole entry bound; zero disables cap only.
/// @return Fully owned output and actual work/history.
/// @throws std::invalid_argument Invalid input domain.
/// @throws std::length_error Unrepresentable counts/capacity or exceeded cap.
/// @throws std::overflow_error Nonfinite actual arithmetic.
/// @throws std::bad_alloc Allocation fails.
ForceDirectedTrace run(
    std::size_t                 n,
    const std::vector<Edge>&    edges,
    std::size_t                 iterations,
    std::uint64_t               seed,
    const std::vector<Point2D>* initial,
    bool                        capture,
    std::size_t                 cap) {
  const auto active = validate(n, edges, initial);
  bounds(n, edges.size(), active, iterations, !initial);
  if (capture) {
    const auto entries = force_directed_layout_trace_entries(n, edges.size(), iterations);
    if (cap && entries > cap)
      throw std::length_error("force layout whole trace cap");
  }
  ForceDirectedTrace result;
  auto&              w   = result.work;
  w.vertices             = n;
  w.edge_records         = edges.size();
  w.nonloop_edges        = active;
  w.iterations_requested = iterations;
  w.seeded               = !initial;
  w.seed                 = initial ? 0 : seed;
  std::mt19937_64                        rng(seed);
  std::uniform_real_distribution<double> uniform(-0.5, 0.5);
  std::vector<Point2D>                   p(n), displacement(n);
  if (initial)
    p = *initial;
  else
    for (auto& point : p) {
      point = {uniform(rng), uniform(rng)};
      w.uniform_values += 2;
    }
  if (capture) {
    result.edges   = edges;
    result.initial = p;
    result.frames.reserve(n >= 2 ? iterations : 0);
  }
  if (n >= 2) {
    const double k = 1.0 / std::sqrt(static_cast<double>(n));
    w.k            = k;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      std::fill(displacement.begin(), displacement.end(), Point2D{});
      for (std::size_t u = 0; u < n; ++u)
        for (std::size_t v = u + 1; v < n; ++v) {
          double dx = p[u].x - p[v].x, dy = p[u].y - p[v].y;
          finite(dx);
          finite(dy);
          double d = std::max(1e-9, std::hypot(dx, dy)), f = k * k / d;
          finite(d);
          finite(f);
          displacement[u].x += dx / d * f;
          displacement[u].y += dy / d * f;
          displacement[v].x -= dx / d * f;
          displacement[v].y -= dy / d * f;
          finite_point(displacement[u]);
          finite_point(displacement[v]);
          ++w.pair_visits;
        }
      ForceDirectedFrame frame;
      if (capture)
        frame.repulsion = displacement;
      for (const auto& [u, v] : edges) {
        ++w.edge_visits;
        if (u != v) {
          double dx = p[u].x - p[v].x, dy = p[u].y - p[v].y;
          finite(dx);
          finite(dy);
          double d = std::max(1e-9, std::hypot(dx, dy)), f = d * d / k;
          finite(d);
          finite(f);
          displacement[u].x -= dx / d * f;
          displacement[u].y -= dy / d * f;
          displacement[v].x += dx / d * f;
          displacement[v].y += dy / d * f;
          finite_point(displacement[u]);
          finite_point(displacement[v]);
          ++w.attraction_visits;
        }
      }
      const double temperature = 0.1 * (1.0 - static_cast<double>(iteration) / iterations);
      if (capture) {
        frame.iteration    = iteration;
        frame.temperature  = temperature;
        frame.displacement = displacement;
      }
      for (std::size_t u = 0; u < n; ++u) {
        const double d = std::max(1e-9, std::hypot(displacement[u].x, displacement[u].y));
        finite(d);
        const double step = std::min(d, temperature);
        p[u].x += displacement[u].x / d * step;
        p[u].y += displacement[u].y / d * step;
        finite_point(p[u]);
        ++w.vertex_moves;
      }
      ++w.completed_iterations;
      if (capture) {
        frame.positions = p;
        result.frames.push_back(std::move(frame));
      }
    }
  }
  if (capture)
    result.raw_final = p;
  if (n >= 2)
    normalize(p, w);
  result.positions = std::move(p);
  return result;
}
} // namespace force_directed_detail
std::size_t force_directed_layout_trace_entries(std::size_t n, std::size_t e, std::size_t t) {
  using namespace force_directed_detail;
  if (n > std::vector<Point2D>().max_size() || e > std::vector<Edge>().max_size() ||
      (n >= 2 && t > std::vector<ForceDirectedFrame>().max_size()))
    throw std::length_error("force layout trace capacity");
  const auto triple = mul(3, n);
  return add(add(e, triple), mul(n >= 2 ? t : 0, add(1, triple)));
}
std::vector<Point2D> force_directed_layout(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    std::size_t                                             t,
    std::uint64_t                                           seed) {
  return force_directed_detail::run(n, e, t, seed, nullptr, false, 0).positions;
}
std::vector<Point2D> force_directed_layout_from(
    const std::vector<Point2D>&                             p,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    std::size_t                                             t) {
  return force_directed_detail::run(p.size(), e, t, 0, &p, false, 0).positions;
}
ForceDirectedOutcome force_directed_layout_work(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    std::size_t                                             t,
    std::uint64_t                                           seed) {
  auto r = force_directed_detail::run(n, e, t, seed, nullptr, false, 0);
  return {std::move(r.positions), r.work};
}
ForceDirectedOutcome force_directed_layout_work_from(
    const std::vector<Point2D>&                             p,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    std::size_t                                             t) {
  auto r = force_directed_detail::run(p.size(), e, t, 0, &p, false, 0);
  return {std::move(r.positions), r.work};
}
ForceDirectedTrace force_directed_layout_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    std::size_t                                             t,
    std::uint64_t                                           seed,
    std::size_t                                             cap) {
  return force_directed_detail::run(n, e, t, seed, nullptr, true, cap);
}
ForceDirectedTrace force_directed_layout_trace_from(
    const std::vector<Point2D>&                             p,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    std::size_t                                             t,
    std::size_t                                             cap) {
  return force_directed_detail::run(p.size(), e, t, 0, &p, true, cap);
}
ForceDirectedMetrics force_directed_layout_metrics(
    const std::vector<Point2D>&                             p,
    const std::vector<std::pair<std::size_t, std::size_t>>& e) {
  using namespace force_directed_detail;
  const auto           n = p.size();
  ForceDirectedMetrics m;
  m.active_edges = validate(n, e, &p);
  m.vertices     = n;
  m.edge_records = e.size();
  m.pair_visits  = pairs(n);
  for (auto v : p) {
    m.centroid_x += static_cast<long double>(v.x);
    m.centroid_y += static_cast<long double>(v.y);
    finite(m.centroid_x);
    finite(m.centroid_y);
  }
  if (n) {
    m.centroid_x /= n;
    m.centroid_y /= n;
  }
  for (auto v : p) {
    const auto d = std::hypot(
        static_cast<long double>(v.x) - m.centroid_x, static_cast<long double>(v.y) - m.centroid_y);
    finite(d);
    m.max_radius = std::max(m.max_radius, d);
  }
  if (n < 2)
    return m;
  const long double k = 1.0 / std::sqrt(static_cast<double>(n)), delta = 1e-9;
  m.min_pair_distance = std::numeric_limits<long double>::infinity();
  for (std::size_t u = 0; u < n; ++u)
    for (std::size_t v = u + 1; v < n; ++v) {
      const auto d = std::hypot(
          static_cast<long double>(p[u].x) - p[v].x, static_cast<long double>(p[u].y) - p[v].y);
      finite(d);
      m.min_pair_distance = std::min(m.min_pair_distance, d);
      if (d < delta) {
        ++m.clamped_pairs;
        m.repulsion_potential +=
            -k * k * std::log(delta) + k * k / 2 * (1 - (d / delta) * (d / delta));
      } else
        m.repulsion_potential += -k * k * std::log(d);
      finite(m.repulsion_potential);
    }
  for (auto [u, v] : e)
    if (u != v) {
      const auto d = std::hypot(
          static_cast<long double>(p[u].x) - p[v].x, static_cast<long double>(p[u].y) - p[v].y);
      finite(d);
      m.mean_edge_length += d;
      finite(m.mean_edge_length);
      m.attraction_potential += d >= delta
                                    ? d * d * d / (3 * k)
                                    : delta * d * d / (2 * k) - delta * delta * delta / (6 * k);
      finite(m.attraction_potential);
    }
  if (m.active_edges)
    m.mean_edge_length /= m.active_edges;
  m.potential = m.repulsion_potential + m.attraction_potential;
  finite(m.potential);
  return m;
}
std::vector<std::pair<std::size_t, std::size_t>> force_directed_layout_example_edges() {
  return {{0, 1}, {1, 2}, {2, 3}};
}
} // namespace datamunge::algorithms
