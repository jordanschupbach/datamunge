/// @file jump_and_walk.cpp
/// @brief Checked local mesh adjacency and shared exact-sign sampled walking kernel.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/jump_and_walk.hpp>
#include <datamunge/geometry/orientation.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Reproducible scalar RNG, checked packed-edge keys and logical counter details.
namespace jw_detail {
/// @brief Legacy local xorshift64 state with fixed nonzero replacement for seed zero.
struct Rng {
  std::uint64_t state; ///< Private-to-query actual stream state.

  /// @brief Start a local deterministic stream.
  /// @param seed Caller seed, zero replaced by legacy golden-ratio constant.
  explicit Rng(std::uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  /// @brief Advance by the original three xor/shift operations.
  /// @return Actual next unsigned64 output with specified modular bit semantics.
  std::uint64_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  }
};
/// @brief Pack two representable original point ids in ascending endpoint order.
/// @param u Original endpoint id.
/// @param v Original endpoint id.
/// @return Unique uint64 undirected key when ids fit uint32.
/// @throws std::length_error On a too-wide endpoint.
std::uint64_t key(std::size_t u, std::size_t v) {
  if (u > UINT32_MAX || v > UINT32_MAX)
    throw std::length_error("locator packed endpoint");
  if (u > v)
    std::swap(u, v);
  return (std::uint64_t(u) << 32) | std::uint64_t(v);
}
/// @brief Increment a logical query counter with overflow checking.
/// @param n Counter mutated on representable increment only.
/// @throws std::length_error On overflow.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::length_error("locator counter overflow");
  ++n;
}
} // namespace jw_detail
TriangulationLocator::TriangulationLocator(
    const std::vector<Point2D>&  pts,
    const std::vector<Triangle>& ts,
    std::size_t                  max_records)
    : points(&pts), tris(&ts) {
  for (const auto& p : pts) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("locator nonfinite point");
    max_abs_ = std::max({max_abs_, std::abs(p.x), std::abs(p.y)});
  }
  if (ts.size() > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("locator triangle ids");
  for (const auto& t : ts) {
    if (t.a >= pts.size() || t.b >= pts.size() || t.c >= pts.size())
      throw std::invalid_argument("locator face index");
    if (t.a > UINT32_MAX || t.b > UINT32_MAX || t.c > UINT32_MAX)
      throw std::length_error("locator packed point id");
    if (orientation_sign(pts[t.a], pts[t.b], pts[t.c]) <= 0)
      throw std::invalid_argument("locator requires strict CCW faces");
  }
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (ts.size() > maximum - pts.size() || ts.size() > maximum / 3)
    throw std::length_error("locator complete shape overflow");
  if (max_records && pts.size() + ts.size() > max_records)
    throw std::length_error("locator complete mesh ceiling");
  across.reserve(3 * ts.size());
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const auto&       t    = ts[i];
    const std::size_t vs[] = {t.a, t.b, t.c};
    for (int k = 0; k < 3; ++k) {
      const auto u = vs[k], v = vs[(k + 1) % 3];
      const auto code     = jw_detail::key(u, v);
      auto [it, inserted] = across.emplace(code, std::array<int, 2>{static_cast<int>(i), -1});
      if (!inserted) {
        if (it->second[1] >= 0)
          throw std::invalid_argument("locator nonmanifold edge");
        const auto&       other    = ts[it->second[0]];
        const std::size_t ov[]     = {other.a, other.b, other.c};
        bool              opposite = false;
        for (int j = 0; j < 3; ++j)
          if (ov[j] == v && ov[(j + 1) % 3] == u)
            opposite = true;
        if (!opposite)
          throw std::invalid_argument("locator shared edge direction");
        it->second[1] = static_cast<int>(i);
      }
    }
  }
}
int TriangulationLocator::neighbor(std::size_t u, std::size_t v, int from) const {
  if (from < 0 || std::size_t(from) >= tris->size())
    throw std::invalid_argument("locator from face");
  const auto it = across.find(jw_detail::key(u, v));
  if (it == across.end())
    return -1;
  const auto& ids = it->second;
  if (ids[0] == from)
    return ids[1];
  if (ids[1] == from)
    return ids[0];
  throw std::invalid_argument("locator edge not incident to from");
}
LocateTrace TriangulationLocator::execute(
    const Point2D& q,
    std::uint64_t  seed,
    LocateOptions  options,
    bool           record) const {
  if (!std::isfinite(q.x) || !std::isfinite(q.y))
    throw std::invalid_argument("locator nonfinite query");
  LocateTrace out;
  auto&       a = out.analysis;
  a.query       = q;
  a.seed        = seed;
  if (record) {
    out.points    = *points;
    out.triangles = *tris;
  }
  const auto n = tris->size();
  if (n == 0)
    return out;
  if (!options.max_steps && n > (std::numeric_limits<std::size_t>::max() - 16) / 4)
    throw std::length_error("locator default step overflow");
  const auto  bound   = options.max_steps ? options.max_steps : 4 * n + 16;
  std::size_t records = 0;
  /// @brief Admit a scalar record before its actual RNG, predicate or state operation.
  const auto admit = [&]() {
    if (record && options.max_events && records >= options.max_events)
      throw std::length_error("locator actual record ceiling");
    jw_detail::increment(records);
  };
  auto&          w = a.work;
  jw_detail::Rng rng(seed);
  std::size_t    count = 1;
  while (std::uint64_t(count) * count * count < n)
    ++count;
  const long double scale = std::max({max_abs_, std::abs(q.x), std::abs(q.y)}),
                    qx    = static_cast<long double>(q.x) / scale,
                    qy    = static_cast<long double>(q.y) / scale;
  long double best        = std::numeric_limits<long double>::infinity();
  int         current     = 0;
  for (std::size_t s = 0; s < count; ++s) {
    admit();
    const int t = static_cast<int>(rng.next() % n);
    jw_detail::increment(w.random_draws);
    const auto&       tr = (*tris)[t];
    const long double cx = (static_cast<long double>((*points)[tr.a].x) / scale +
                            static_cast<long double>((*points)[tr.b].x) / scale +
                            static_cast<long double>((*points)[tr.c].x) / scale) /
                           3;
    const long double cy = (static_cast<long double>((*points)[tr.a].y) / scale +
                            static_cast<long double>((*points)[tr.b].y) / scale +
                            static_cast<long double>((*points)[tr.c].y) / scale) /
                           3;
    const long double score = (cx - qx) * (cx - qx) + (cy - qy) * (cy - qy);
    if (score < best) {
      best    = score;
      current = t;
    }
    jw_detail::increment(w.samples);
    if (record)
      out.samples.push_back({t, score, current, best});
  }
  a.start_triangle = current;
  for (std::size_t step = 0; step < bound; ++step) {
    admit();
    const int first = static_cast<int>(rng.next() % 3);
    jw_detail::increment(w.random_draws);
    jw_detail::increment(w.steps);
    if (record)
      out.steps.push_back({current, first});
    const auto&       tr    = (*tris)[current];
    const std::size_t vs[]  = {tr.a, tr.b, tr.c};
    bool              moved = false;
    for (int offset = 0; offset < 3; ++offset) {
      admit();
      const int  k = (first + offset) % 3;
      const auto u = vs[k], v = vs[(k + 1) % 3];
      const int  sign = orientation_sign((*points)[u], (*points)[v], q);
      jw_detail::increment(w.edge_tests);
      int next = -2;
      if (sign < 0) {
        next = neighbor(u, v, current);
        jw_detail::increment(w.neighbor_lookups);
      }
      if (record)
        out.tests.push_back({step, current, k, u, v, sign, next});
      if (sign < 0) {
        if (next < 0) {
          a.stop = LocateStop::Boundary;
          return out;
        }
        current = next;
        jw_detail::increment(w.crossings);
        moved = true;
        break;
      }
    }
    if (!moved) {
      a.stop     = LocateStop::Located;
      a.triangle = current;
      return out;
    }
  }
  a.stop = LocateStop::StepLimit;
  return out;
}
LocateAnalysis TriangulationLocator::analyze(const Point2D& q, std::uint64_t seed, LocateOptions o)
    const {
  return execute(q, seed, o, false).analysis;
}
LocateTrace TriangulationLocator::trace(const Point2D& q, std::uint64_t seed, LocateOptions o)
    const {
  return execute(q, seed, o, true);
}
int TriangulationLocator::locate(const Point2D& q, std::uint64_t seed) const {
  const auto a = analyze(q, seed);
  if (a.stop == LocateStop::StepLimit)
    throw std::runtime_error("locator walk step limit");
  return a.triangle;
}
int locate_triangle(
    const std::vector<Point2D>&  points,
    const std::vector<Triangle>& tris,
    const Point2D&               q,
    std::uint64_t                seed) {
  return TriangulationLocator(points, tris).locate(q, seed);
}
} // namespace datamunge::geometry
