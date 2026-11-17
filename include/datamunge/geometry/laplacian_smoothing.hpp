#pragma once
/// @file laplacian_smoothing.hpp
/// @brief Checked undirected Jacobi umbrella smoothing with actual centroid/update and energy trajectories.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <vector>
namespace datamunge::geometry {
/// @brief Selected actual smoothing work, excluding complete graph validation, copies and time.
/// neighbor_reads counts point records used in actual free nonisolated centroids,
/// not scalar coordinates. energy_edges counts undirected edge multiplicities
/// visited by trace-only energy diagnostics; ordinary/count-only smoothing has0.
/// @par Example
/// @code
/// const auto a=dm::geometry::laplacian_smooth_analyze({{0,0},{2,5},{4,0}},{{1},{0,2},{1}},{1,0,1},1,1);
/// std::print("{} {}\n",a.work.updated_vertices,a.work.neighbor_reads); // 1 2
/// @endcode
/// @see LaplacianAnalysis
struct LaplacianWork {
  std::size_t sweeps{0};           ///< Actual full requested Jacobi passes completed.
  std::size_t updated_vertices{0}; ///< Actual free nonisolated vertex updates.
  std::size_t pinned_skips{0};     ///< Actual vertex passes skipped because fixed flag is nonzero.
  std::size_t isolated_skips{0};   ///< Actual free vertex passes skipped for an empty neighborhood.
  std::size_t neighbor_reads{0};   ///< Actual complete neighbor point records read for centroids.
  std::size_t energy_edges{
      0}; ///< Actual trace-only undirected energy edge evaluations, otherwise0.
};
/// @brief Owned actual final points and same-execution selected smoothing work.
/// @par Example
/// @code
/// const auto a=dm::geometry::laplacian_smooth_analyze({{0,0},{2,5},{4,0}},{{1},{0,2},{1}},{1,0,1},1,1);
/// std::print("{}\n",a.vertices[1].y); // 0
/// @endcode
/// @see laplacian_smooth_trace()
struct LaplacianAnalysis {
  std::vector<Point2D>
                vertices; ///< Owned actual final Jacobi state, original identities preserved.
  LaplacianWork work;     ///< Actual selected kernel work, no energy replay in ordinary analysis.
};
/// @brief Actual free nonisolated vertex centroid and before/after state from one Jacobi pass.
/// centroid reads only the old complete state, never an earlier update in the same
/// pass. Widened flags identify actual overflow-avoiding coordinate fallbacks;
/// normal finite double accumulation/update order follows historical arithmetic.
/// @par Example
/// @code
/// const auto t=dm::geometry::laplacian_smooth_trace({{0,0},{2,5},{4,0}},{{1},{0,2},{1}},{1,0,1},1,1);
/// std::print("{} {}\n",t.updates[0].vertex,t.updates[0].centroid.y); // 1 0
/// @endcode
/// @see LaplacianTrace
struct LaplacianUpdate {
  std::size_t iteration{0};      ///< Zero-based actual producing sweep.
  std::size_t vertex{0};         ///< Original actual updated vertex identity.
  std::size_t neighbor_count{0}; ///< Actual complete neighborhood multiplicity, including loops.
  Point2D     before;            ///< Actual old point read by this update.
  Point2D     centroid;          ///< Actual rounded neighbor mean used by the producer.
  Point2D     after;             ///< Actual committed point in the next complete state.
  bool        widened_centroid{
      false}; ///< Whether either nonfinite ordinary coordinate sum used widened mean.
  bool widened_update{
      false}; ///< Whether either ordinary coordinate update required widened convex blend.
};
/// @brief Owned graph, full actual Jacobi states/energies and actual vertex updates, without reruns.
/// fixed is normalized to n flags; states[0] is the complete original geometry,
/// followed by each actual complete sweep. energies are widened sums over actual
/// states, not an exact monotonicity/mesh-quality certificate. Captured count is
/// 2*n+E+1 + iterations*(n+1+R): original vertices/normalized flags/adjacency entries/
/// initial energy, then each state/energy and R actual free nonisolated updates.
/// Counts point/record identities, not their coordinates/fields/capacities/bytes;
/// result/runtime copies excluded, not peak RAM. Complete quota admitted before
/// graph/state copies or smoothing. No replay or partial returned trace.
/// @par Example
/// @code
/// const auto t=dm::geometry::laplacian_smooth_trace({{0,0},{2,5},{4,0}},{{1},{0,2},{1}},{1,0,1},1,1);
/// std::print("{} {}\n",t.states.size(),t.energies.back()); // 2 8
/// @endcode
/// @see laplacian_smooth_trace()
struct LaplacianTrace {
  std::vector<std::vector<int>>
                    neighbors; ///< Owned complete original undirected adjacency order/multiplicity.
  std::vector<char> fixed;     ///< Owned n normalized pin flags; omitted original suffix becomes0.
  double            lambda{.5}; ///< Actual finite relaxation factor in[0,1].
  std::vector<std::vector<Point2D>>
                           states;   ///< Owned original and every actual complete Jacobi state.
  std::vector<long double> energies; ///< Actual widened undirected energy of every captured state.
  std::vector<LaplacianUpdate>
                       updates; ///< Actual sweep-then-vertex ordered free nonisolated updates.
  std::vector<Point2D> result; ///< Owned actual final geometry, independent of stored state copies.
  LaplacianWork        work;   ///< Actual smoothing and explicitly additional diagnostic work.
  std::size_t captured_entries{0}; ///< Complete declared logical state/input/update capture count.
};
/// @brief Relax finite planar graph vertices toward actual neighbor means using synchronous Jacobi sweeps.
/// Require n<=INT_MAX, exactly n adjacency rows, valid endpoints and reciprocal
/// edge multiplicities; loops/duplicates allowed. Fixed flags may have length0..n,
/// with missing suffix free; any nonzero flag pins exactly. Require finite points,
/// lambda in[0,1] and iterations>=0. Pinned/isolated points stay unchanged. Read the
/// full old state and write next[i]=old[i]+lambda*(centroid-old[i]) for each free
/// nonisolated vertex. Ordinary finite double sum/mean/update order preserved;
/// parallel widened sums and convex-blend fallbacks avoid ordinary overflow when
/// possible. Produced coordinates must remain finite doubles or throw.
/// Exact undirected Dirichlet energy is nonincreasing for these lambda values;
/// lambda1 can oscillate on an unpinned bipartite graph, and roundoff is not an exact
/// energy certificate. No topology, area, nonfolding, volume or signal-truth error
/// guarantee. Shrinkage/oversmoothing are possible. O(n+E log(E+1)+I*(n+E)) work
/// including complete reciprocal validation, O(n+E) validation/runtime storage.
/// No input mutation, optional capture or shared state; calls independent.
/// @param vertices Borrowed complete finite original points, empty allowed.
/// @param neighbors Borrowed complete undirected adjacency with original entry order.
/// @param fixed Borrowed0..n pin flags; missing entries mean free.
/// @param lambda Finite relaxation factor in[0,1], default0.5; zero leaves coordinates unchanged.
/// @param iterations Nonnegative complete sweep count, default1, zero returns original points.
/// @return Owned actual final points, original identities/order retained.
/// @throws std::invalid_argument On graph/flag/finite/lambda/iteration violations before copies.
/// @throws std::length_error On vertex/count/container representability limits.
/// @throws std::overflow_error On unrepresentable produced finite-coordinate arithmetic.
/// @throws std::bad_alloc On validation map/state allocation failure.
/// @par Example
/// @code
/// const auto v=dm::geometry::laplacian_smooth({{0,0},{2,5},{4,0}},{{1},{0,2},{1}},{1,0,1},1,1);
/// std::print("{} {}\n",v[1].x,v[1].y); // 2 0
/// @endcode
/// @par References
/// - Taubin(1995), A Signal Processing Approach to Fair Surface Design, SIGGRAPH1995,351–358, https://doi.org/10.1145/218380.218473.
///   This implementation uses a single-sign umbrella relaxation, not Taubin's shrinkage-reducing two-pass filter.
/// @see dirichlet_energy(), laplacian_smooth_analyze(), laplacian_smooth_trace()
std::vector<Point2D> laplacian_smooth(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<char>&             fixed,
    double                               lambda     = .5,
    int                                  iterations = 1);
/// @brief Evaluate undirected graph Dirichlet energy as squared edge lengths counted once per multiplicity.
/// Validate complete finite points, matching rows, endpoints and reciprocal
/// multiplicities. Loops contribute0; parallel reciprocal entries count repeated
/// edges. Widen coordinate differences/squares/sum, then require finite double
/// output. No mutation/history, O(n+E log(E+1)) validation plus O(E) evaluation.
/// Energy alone does not certify a well-shaped, nonfolded or truth-preserving mesh.
/// @param vertices Borrowed complete finite geometry.
/// @param neighbors Borrowed complete undirected adjacency, loops/duplicates allowed.
/// @return Actual widened squared-length sum rounded once to finite double,0 for no edges.
/// @throws std::invalid_argument On shape/endpoint/finite/reciprocal violations.
/// @throws std::length_error On vertex/count/container limits.
/// @throws std::overflow_error On nonfinite widened arithmetic or out-of-double-range energy.
/// @throws std::bad_alloc On reciprocal-validation map allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::dirichlet_energy({{0,0},{3,4}},{{1},{0}})); // 25
/// @endcode
/// @par References
/// - Taubin(1995), A Signal Processing Approach to Fair Surface Design, https://doi.org/10.1145/218380.218473.
/// @see laplacian_smooth_trace()
double dirichlet_energy(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors);
/// @brief Return actual final Jacobi geometry and selected work without state/history/energy capture.
/// Same complete validation/arithmetic/complexity as laplacian_smooth(). Does not
/// rerun prefixes or evaluate extra energy diagnostics; energy_edges is0.
/// @param vertices Borrowed complete finite original points.
/// @param neighbors Borrowed complete undirected adjacency.
/// @param fixed Borrowed0..n pin flags, missing suffix free.
/// @param lambda Actual finite[0,1] relaxation, default0.5.
/// @param iterations Actual nonnegative complete sweep count, default1.
/// @return Owned actual final geometry and selected producing work.
/// @throws std::invalid_argument On complete input/parameter violations.
/// @throws std::length_error On vertex/count/container limits.
/// @throws std::overflow_error On nonfinite produced coordinate arithmetic.
/// @throws std::bad_alloc On validation/state allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::laplacian_smooth_analyze({{1,2}},{{}},{}).work.isolated_skips); // 1
/// @endcode
/// @par References
/// - Taubin(1995), A Signal Processing Approach to Fair Surface Design, https://doi.org/10.1145/218380.218473.
/// @see LaplacianWork, laplacian_smooth_trace()
LaplacianAnalysis laplacian_smooth_analyze(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<char>&             fixed,
    double                               lambda     = .5,
    int                                  iterations = 1);
/// @brief Capture every actual Jacobi state, producing centroid/update and widened state energy once.
/// Same input/numerical contracts as laplacian_smooth(); complete capture geometry
/// is admitted before copies/sweeps. Adds O(I*(n+E)) energy observation work and
/// O(I*(n+R)+n+E) owned capture, with default million logical entries, zero off.
/// Captures actual passes rather than independently rerunning every prefix.
/// @param vertices Borrowed complete finite original geometry.
/// @param neighbors Borrowed complete reciprocal adjacency.
/// @param fixed Borrowed0..n pin flags, normalized to n for captured output.
/// @param lambda Actual finite relaxation in[0,1], default0.5.
/// @param iterations Actual nonnegative complete sweep count, default1.
/// @param max_entries Complete logical input/state/update ceiling, default1000000, zero off.
/// @return Owned complete actual trajectory, updates, diagnostics and independent final result.
/// @throws std::invalid_argument On complete input/parameter violations before copies.
/// @throws std::length_error On enabled quota or count/container geometry limits.
/// @throws std::overflow_error On nonfinite coordinate or widened energy arithmetic.
/// @throws std::bad_alloc On validation/trajectory/update allocation failure.
/// @par Example
/// @code
/// const auto t=dm::geometry::laplacian_smooth_trace({{1,2}},{{}},{});
/// std::print("{} {}\n",t.states.size(),t.energies.back()); // 2 0
/// @endcode
/// @par References
/// - Taubin(1995), A Signal Processing Approach to Fair Surface Design, https://doi.org/10.1145/218380.218473.
/// @see LaplacianTrace, laplacian_smooth_analyze()
LaplacianTrace laplacian_smooth_trace(
    const std::vector<Point2D>&          vertices,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<char>&             fixed,
    double                               lambda      = .5,
    int                                  iterations  = 1,
    std::size_t                          max_entries = 1000000);
} // namespace datamunge::geometry
