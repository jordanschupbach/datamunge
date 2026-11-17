#pragma once
/// @file jump_and_walk.hpp
/// @brief Checked borrowed triangulation locator with exact turns and actual sample/walk records.
#include <array>
#include <cstddef>
#include <cstdint>
#include <datamunge/geometry/delaunay.hpp>
#include <unordered_map>
#include <vector>
namespace datamunge::geometry {
/// @brief Query walk bound and optional actual trace-record ceiling.
/// Zero max_steps selects 4*T+16 for T triangles, not an unbounded walk. A
/// positive value overrides that bound. Reaching it is StepLimit, never proof of
/// exterior location. max_events counts samples+step starts+edge tests, before
/// each actual operation; zero disables that optional ceiling. No partial trace
/// is returned on failure; capacities/bytes/peak RAM are outside these counts.
/// @par Example
/// @code
/// const dm::geometry::LocateOptions o{1,100};
/// std::print("{}\n",o.max_steps); // 1
/// @endcode
/// @see TriangulationLocator::analyze(), TriangulationLocator::trace()
struct LocateOptions {
  std::size_t max_steps{0}; ///< Positive explicit walk-step bound, zero(default) uses 4*T+16.
  std::size_t max_events{
      1000000}; ///< Actual sample/start/test ceiling, million default, zero off; ignored untraced.
};
/// @brief Distinguish successful containment, a crossed boundary, empty mesh and exhausted budget.
/// Boundary proves exterior only for the documented conforming convex-domain
/// triangulation precondition. StepLimit says nothing about containment.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::TriangulationLocator({},{}).analyze({0,0}).stop==dm::geometry::LocateStop::Empty); // true
/// @endcode
/// @see LocateAnalysis
enum class LocateStop {
  Located,  ///< All three actual edge signs were nonnegative.
  Boundary, ///< A negative-sign edge has no adjacent face.
  Empty,    ///< No triangle exists, after finite query validation.
  StepLimit ///< Explicit/default walk bound exhausted without a geometric conclusion.
};
/// @brief Selected actual query work, excluding mesh construction, validation and copies.
/// Random draws equal samples+steps. Edge tests count exact orientation calls,
/// not their internal arithmetic. Crossings exclude the terminal boundary edge.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
/// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
/// std::print("{}\n",dm::geometry::TriangulationLocator(p,f).analyze({1,1}).work.edge_tests); // 3
/// @endcode
/// @see LocateAnalysis
struct LocateWork {
  std::size_t samples{0}; ///< Actual sampled triangles, with replacement.
  std::size_t steps{0};   ///< Actual entered triangles/random cyclic edge starts, repeats allowed.
  std::size_t edge_tests{0};       ///< Actual exact orientation signs computed.
  std::size_t neighbor_lookups{0}; ///< Actual map lookups after a negative sign only.
  std::size_t crossings{0};        ///< Actual transitions to an adjacent triangle.
  std::size_t random_draws{0}; ///< Actual xorshift64 outputs used for samples and cyclic starts.
};
/// @brief Owned scalar query result and selected actual work, with no mesh references.
/// Seed zero uses the legacy nonzero fallback stream. Containment includes edges
/// and vertices; when several faces contain q, the actual walk selects one, not
/// necessarily the least index. Triangle is -1 outside/empty/budget exhaustion;
/// stop distinguishes these cases. All fields are values, no shared state.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
/// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
/// const auto a=dm::geometry::TriangulationLocator(p,f).analyze({1,1});
/// std::print("{} {}\n",a.triangle,a.start_triangle); // 0 0
/// @endcode
/// @see LocateTrace
struct LocateAnalysis {
  Point2D       query;   ///< Owned finite query coordinates.
  std::uint64_t seed{1}; ///< Caller seed, including zero's documented fallback convention.
  int           start_triangle{-1}; ///< Actual best sampled triangle; -1 for empty mesh.
  int           triangle{-1};       ///< Actual containing triangle only on Located, otherwise -1.
  LocateStop    stop{LocateStop::Empty}; ///< Exact termination category, not a timing inference.
  LocateWork    work;                    ///< Producing logical work counts.
};
/// @brief One actual sampled triangle and running centroid-score winner.
/// Scores use coordinates divided by a common max(1,absolute mesh/query coordinate)
/// scale before centroid/difference/square arithmetic. They are rounded dimensionless
/// long-double values, not exact nearest-centroid certificates. Strict decrease
/// keeps the first sampled winner on ties. Sampling is with replacement.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
/// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
/// std::print("{}\n",dm::geometry::TriangulationLocator(p,f).trace({1,1}).samples[0].triangle); // 0
/// @endcode
/// @see LocateTrace
struct LocateSample {
  int         triangle{-1};      ///< Actual RNG-selected triangle id.
  long double score{0};          ///< Actual normalized squared centroid distance.
  int         best_triangle{-1}; ///< Actual winner after this sample.
  long double best_score{0};     ///< Actual winning score after this sample.
};
/// @brief Actual triangle entry and random cyclic edge start for a walk step.
/// Edge order is first_edge,(first_edge+1)%3,(first_edge+2)%3, not a uniform
/// permutation of all six orders. No geometric distance monotonicity is promised.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
/// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
/// std::print("{}\n",dm::geometry::TriangulationLocator(p,f).trace({1,1}).steps[0].triangle); // 0
/// @endcode
/// @see LocateTest
struct LocateStep {
  int triangle{-1};  ///< Actually entered original face, possibly a repeat.
  int first_edge{0}; ///< Actual xorshift output modulo three.
};
/// @brief One actual directed edge inspection in a walk step.
/// Exact sign is for represented finite binary64 endpoints/query. A negative sign
/// triggers exactly one adjacency lookup; neighbor is -1 on boundary, >=0 for a
/// crossing, or -2 when no lookup was made. No tolerance or predicate replay.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
/// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
/// std::print("{}\n",dm::geometry::TriangulationLocator(p,f).trace({1,1}).tests[0].neighbor); // -2
/// @endcode
/// @see LocateTrace
struct LocateTest {
  std::size_t step{0};      ///< Index into LocateTrace::steps.
  int         triangle{-1}; ///< Actual face being inspected.
  int         edge{0};      ///< Directed local edge0:a->b,1:b->c,2:c->a.
  std::size_t u{0};         ///< Actual original origin-point index.
  std::size_t v{0};         ///< Actual original target-point index.
  int         sign{0};      ///< Exact orientation sign in {-1,0,1}.
  int         neighbor{-2}; ///< -2 unqueried, -1 boundary, otherwise actual adjacent triangle.
};
/// @brief Owned original mesh and complete actual sample/start/test records from one query.
/// Samples precede all starts/tests; test.step groups tests under each actual
/// start. A negative test records the actual crossing/boundary decision. No
/// repeated-prefix runs or full intermediate snapshots. Copies own all storage;
/// trace remains valid after the borrowed locator/input are destroyed. Recording
/// adds O(P+T+K) storage/copy work for P points,T triangles,K scalar records.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
/// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
/// const auto t=dm::geometry::TriangulationLocator(p,f).trace({1,1});
/// std::print("{} {} {}\n",t.samples.size(),t.steps.size(),t.tests.size()); // 1 1 3
/// @endcode
/// @see TriangulationLocator::trace()
struct LocateTrace {
  LocateAnalysis            analysis; ///< Actual scalar result and work.
  std::vector<Point2D>      points; ///< Owned complete original point list, unused points included.
  std::vector<Triangle>     triangles; ///< Owned original faces in stable input order.
  std::vector<LocateSample> samples;   ///< Actual sampled starts and running winners.
  std::vector<LocateStep>   steps;     ///< Actual triangle entries and cyclic edge starts.
  std::vector<LocateTest> tests; ///< Actual directed inspections, grouped chronologically by step.
};
/// @brief Borrow a conforming CCW triangulation and cache its edge-to-face adjacency.
/// Both input vectors must outlive this object and remain unchanged, including
/// through public pointer/map fields. Copies retain the same borrowed inputs and
/// own copied adjacency. Concurrent const queries use independent RNG/work state.
/// Construction checks all finite coordinates, face indices, strict exact CCW,
/// at most two faces per edge and opposite shared-edge directions. It does NOT
/// certify global nonoverlap, convex-domain coverage, absence of holes or Delaunay
/// empty circles: those embedding properties remain caller preconditions for
/// interpreting boundary as outside. Canonical delaunay_triangulation() supplies
/// an appropriate mesh. General non-Delaunay walks may exhaust their bound.
/// Packed legacy edge keys require each used point index<=UINT32_MAX; triangle
/// count<=INT_MAX. Expected hash construction O(P+T) work/O(T) adjacency storage,
/// excluding borrowed inputs and exact predicate constant factors. No cube-root
/// query bound is promised for arbitrary meshes or query distributions.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
/// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
/// const dm::geometry::TriangulationLocator loc(p,f);
/// std::print("{} {}\n",loc.locate({1,1}),loc.locate({5,5})); // 0 -1
/// @endcode
/// @see LocateTrace, locate_triangle()
struct TriangulationLocator {
  const std::vector<Point2D>* points{
      nullptr}; ///< Borrowed immutable point vector; caller lifetime precondition.
  const std::vector<Triangle>* tris{
      nullptr}; ///< Borrowed immutable original faces; caller lifetime precondition.
  std::unordered_map<std::uint64_t, std::array<int, 2>>
      across; ///< Owned packed undirected-edge incidence; do not mutate while querying.

  /// @brief Validate/admit a complete borrowed mesh and build cached adjacency once.
  /// Complete finite/index/CCW validation precedes adjacency allocation; topology
  /// checks occur while building. A failed construction exposes no partial object.
  /// @param pts Borrowed finite complete points, including unused records.
  /// @param ts Borrowed strict CCW finite triangles, original stable order.
  /// @param max_records Complete P+T ceiling, default million; zero disables optional cap.
  /// @throws std::invalid_argument On nonfinite coordinates, invalid/degenerate/CW faces or invalid local incidence.
  /// @throws std::length_error On enabled cap or point-index/triangle/size representability limits.
  /// @throws std::bad_alloc On allocation failure.
  /// @par Example
  /// @code
  /// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
  /// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
  /// const dm::geometry::TriangulationLocator loc(p,f,4);
  /// std::print("{}\n",loc.across.size()); // 3
  /// @endcode
  /// @see analyze(), trace()
  TriangulationLocator(
      const std::vector<Point2D>&  pts,
      const std::vector<Triangle>& ts,
      std::size_t                  max_records = 1000000);
  /// @brief Return the other incident triangle, or -1 for a missing/boundary edge.
  /// No geometric predicate, mutation or retained references. Expected hash O(1).
  /// @param u Original endpoint id, representable in uint32.
  /// @param v Original endpoint id, representable in uint32.
  /// @param from Valid original face id incident on a present edge.
  /// @return Other original face id, or -1 for missing/boundary edge.
  /// @throws std::invalid_argument On invalid from or a present edge not incident to from.
  /// @throws std::length_error On an endpoint exceeding packed-key representation.
  /// @par Example
  /// @code
  /// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{4,4},{0,4}};
  /// const std::vector<dm::geometry::Triangle> f={{0,1,2},{0,2,3}};
  /// std::print("{}\n",dm::geometry::TriangulationLocator(p,f).neighbor(0,2,0)); // 1
  /// @endcode
  /// @see locate()
  int neighbor(std::size_t u, std::size_t v, int from) const;
  /// @brief Return a containing original triangle or -1 after boundary/empty termination.
  /// Uses analyze() with default step bound, same RNG/exact-turn kernel. Throws
  /// on exhausted budget instead of conflating it with exterior location. Edge/
  /// vertex ties may select any incident face according to actual sampled walk.
  /// @param q Finite query, unchanged by the call.
  /// @param seed Local xorshift seed, default1; zero uses fixed legacy fallback.
  /// @return Containing original face id, or -1 for boundary/empty.
  /// @throws std::invalid_argument On a nonfinite query.
  /// @throws std::length_error On work/bound representability overflow.
  /// @throws std::runtime_error On default walk-step exhaustion.
  /// @par Example
  /// @code
  /// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
  /// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
  /// std::print("{}\n",dm::geometry::TriangulationLocator(p,f).locate({1,1},0)); // 0
  /// @endcode
  /// @par References
  /// - Mucke,Saias,Zhu(1999), https://doi.org/10.1016/S0925-7721(98)00035-2.
  /// @see analyze(), trace()
  int locate(const Point2D& q, std::uint64_t seed = 1) const;
  /// @brief Analyze one sampled jump and bounded visibility walk without mesh copies/history.
  /// Samples ceil(cuberoot(T)) faces with replacement, first strict minimum of
  /// normalized centroid scores. Random cyclic starts retain legacy xorshift64
  /// draws/modulo mapping; this is deterministic pseudorandomness, not independent
  /// mathematical randomness. Exact finite binary64 signs control crossings; three
  /// nonnegative signs mean containment. O(samples+steps) expected hash work and
  /// constant query storage; at most three signs per started step. Mesh construction
  /// and the point count P are excluded; no monotone-distance or universal expected
  /// cube-root bound. Positive options.max_steps overrides default4*T+16.
  /// @param q Borrowed finite query, also returned by value.
  /// @param seed Local stream seed, default1; zero uses legacy fallback.
  /// @param options Walk-step bound; event ceiling ignored for this untraced call.
  /// @return Owned scalar result including distinct StepLimit/Boundary/Empty states.
  /// @throws std::invalid_argument On nonfinite query.
  /// @throws std::length_error On size/work arithmetic overflow.
  /// @par Example
  /// @code
  /// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
  /// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
  /// std::print("{}\n",dm::geometry::TriangulationLocator(p,f).analyze({5,5}).stop==dm::geometry::LocateStop::Boundary); // true
  /// @endcode
  /// @par References
  /// - Mucke,Saias,Zhu(1999), https://doi.org/10.1016/S0925-7721(98)00035-2; centroid sampling is this implementation's heuristic.
  /// @see trace(), LocateOptions
  LocateAnalysis analyze(const Point2D& q, std::uint64_t seed = 1, LocateOptions options = {})
      const;
  /// @brief Capture the actual sampled candidates, entered faces and inspected edges once.
  /// Same query/mesh/kernel contracts as analyze(). Copies complete original mesh
  /// and actual scalar records; returned trace owns all data after input destruction.
  /// Additional O(P+T+K) work/storage for graph copies and K records; no replay.
  /// Each sample/start/test is admitted before RNG/predicate/state operation.
  /// @param q Borrowed finite query, returned by value.
  /// @param seed Local stream seed, default1, zero fallback as locate().
  /// @param options Positive/default step bound and online actual-record ceiling.
  /// @return Complete owned producing trace, including any StepLimit outcome.
  /// @throws std::invalid_argument On nonfinite query.
  /// @throws std::length_error On enabled record cap or arithmetic/container overflow.
  /// @throws std::bad_alloc On graph/history allocation failure.
  /// @par Example
  /// @code
  /// const std::vector<dm::geometry::Point2D> p={{0,0},{4,0},{0,4}};
  /// const std::vector<dm::geometry::Triangle> f={{0,1,2}};
  /// const auto t=dm::geometry::TriangulationLocator(p,f).trace({1,1});
  /// std::print("{}\n",t.analysis.triangle); // 0
  /// @endcode
  /// @see analyze(), LocateTrace
  LocateTrace trace(const Point2D& q, std::uint64_t seed = 1, LocateOptions options = {}) const;

 private:
  double max_abs_{
      1}; ///< Cached max(1,absolute coordinate) for stable finite score scaling; input immutability required.

  /// @brief Shared producing query implementation with optional owned recording.
  /// @param q Finite query validated before any copies or RNG draws.
  /// @param seed Local stream seed with fixed zero fallback.
  /// @param options Step bound and optional online record ceiling.
  /// @param record Whether to copy mesh and record actual scalar history.
  /// @return Actual result and optional owned history, no partial result on exceptions.
  /// @throws std::invalid_argument On nonfinite query.
  /// @throws std::length_error On record/bound/work representability failures.
  /// @throws std::bad_alloc On recording allocation failure.
  LocateTrace execute(const Point2D& q, std::uint64_t seed, LocateOptions options, bool record)
      const;
};
/// @brief Validate/build a temporary borrowed locator and locate one query.
/// Construction O(P+T) expected hash work/O(T) storage repeats on each call;
/// reuse TriangulationLocator for many queries. Same exact signs, borrowed-call
/// semantics, default step bound and boundary/empty/step-exhaustion rules.
/// @param points Borrowed finite original points for this call.
/// @param tris Borrowed conforming strict CCW original triangles.
/// @param q Borrowed finite query.
/// @param seed Local stream seed, default1, zero legacy fallback.
/// @return Containing triangle id, or -1 for boundary/empty.
/// @throws std::invalid_argument On mesh/query/local-topology violations.
/// @throws std::length_error On default million P+T construction cap or representability failures.
/// @throws std::runtime_error On default query step exhaustion.
/// @throws std::bad_alloc On adjacency allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::locate_triangle({{0,0},{4,0},{0,4}},{{0,1,2}},{1,1})); // 0
/// @endcode
/// @par References
/// - Mucke,Saias,Zhu(1999), https://doi.org/10.1016/S0925-7721(98)00035-2.
/// @see TriangulationLocator
int locate_triangle(
    const std::vector<Point2D>&  points,
    const std::vector<Triangle>& tris,
    const Point2D&               q,
    std::uint64_t                seed = 1);
} // namespace datamunge::geometry
