#pragma once
/// @file spectral_layout.hpp
/// @brief Actual combinatorial Laplacian Jacobi rotations and scaled layout.
#include <cstddef>
#include <datamunge/algorithms/layout_geometry.hpp>
#include <utility>
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief SpectralLayoutOptions owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SpectralLayoutOptions value{};
/// std::print("{}\n",value.max_vertices); // 256
/// @endcode
struct SpectralLayoutOptions {
  double      tolerance{1e-12};  ///< Finite positive maximum off-diagonal stopping threshold.
  std::size_t max_vertices{256}; ///< Complete dense matrix order; zero disables.
  std::size_t max_rotations{}; ///< Actual rotation cap; zero chooses50*n*n, still capacity checked.
  std::size_t max_scan_visits{
      200000000}; ///< Actual upper-triangle pivot-search entries; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual records plus copied post-rotation diagonal entries; zero disables.
};
/// @brief SpectralRotation owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SpectralRotation value{};
/// std::print("{}\n",value.first); // 0
/// @endcode
struct SpectralRotation {
  std::size_t         first{};   ///< Actual chosen largest off-diagonal row index.
  std::size_t         second{};  ///< Actual chosen largest off-diagonal column index.
  double              largest{}; ///< Actual maximum absolute off-diagonal entry before rotation.
  double              angle{};   ///< Actual Jacobi rotation angle in radians.
  std::vector<double> diagonal;  ///< Actual complete diagonal immediately after this rotation.
};
/// @brief SpectralLayoutRun owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SpectralLayoutRun value{};
/// std::print("{}\n",value.converged); // false
/// @endcode
struct SpectralLayoutRun {
  std::vector<Point2D>
      positions; ///< Actual second/third-mode coordinates after centering and common radial scale.
  std::vector<std::vector<double>>
      laplacian; ///< Actual built unweighted multigraph combinatorial matrix; loops ignored, duplicate edges add.
  std::vector<double> eigenvalues; ///< Actual diagonal values sorted ascending after termination.
  std::vector<std::vector<double>>
      eigenvectors; ///< Actual row-major eigenvector matrix with columns in sorted-eigenvalue order.
  bool        converged{}; ///< Actual maximum off-diagonal magnitude is below tolerance.
  double      remaining_off_diagonal{}; ///< Actual final maximum absolute off-diagonal entry.
  std::size_t rotations{};              ///< Actual executed Jacobi rotations.
  std::size_t
      scan_visits{}; ///< Actual pivot-search upper-triangle entries read, including final check.
  std::vector<SpectralRotation> history; ///< Optional actual rotation choices/diagonals.
};
/// @brief Embed a graph through its actual low Laplacian modes.
/// Build symmetric L=D-A from undirected endpoints; repeated edges add unit weight, loops are ignored but endpoints still validated. Largest-off-diagonal Jacobi rotations start from identity eigenvectors, use row-major strict maximum ties, then stable ascending eigenvalue order. Select columns1 and2, falling back to column0 for n=1 and repeating column1 for n=2; center and divide both coordinates by maximum radial norm. Empty gives empty converged result. For connected graphs, these are low nonconstant modes; disconnected graphs can select additional zero modes. Degenerate eigenvalues allow rotations/sign changes and layouts are not unique. Approximate finite arithmetic, not a planarity solver. Actual convergence flag is separate from cap exhaustion. Each largest-pivot scan costs O(n^2), each rotation update O(n); up to O(n^2) rotations therefore permits O(n^4) scan work, not an unconditional O(n^3) bound. Dense storage O(n^2), optional history O(rotations*n). Owned outputs and no shared state.
/// @param vertex_count Complete vertex order, labels0..n-1.
/// @param edges Borrowed complete undirected endpoint pairs.
/// @param options Tolerance and complete vertex/actual rotation/scan/capture policies.
/// @return Owned actual positions, Laplacian/eigensystem, stopping state, counters and optional history.
/// @throws std::invalid_argument Endpoint out of range or invalid tolerance.
/// @throws std::length_error Geometry, actual scan/history or counter/container capacity exceeded.
/// @throws std::overflow_error Actual matrix/eigenvector/layout arithmetic nonfinite.
/// @par Example
/// @code
/// auto r=dm::algorithms::spectral_layout_run(2,{{0,1}});
/// std::print("{} {}\n",r.converged,r.rotations); // true 1
/// @endcode
/// @par References
/// - K. M. Hall, An r-Dimensional Quadratic Placement Algorithm, Management Science17(3),1970, DOI10.1287/mnsc.17.3.219.
/// @see spectral_layout()
SpectralLayoutRun spectral_layout_run(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const SpectralLayoutOptions&                            options = {});
/// @brief Embed a graph through its actual low Laplacian modes with actual Jacobi rotations.
/// Build symmetric L=D-A from undirected endpoints; repeated edges add unit weight, loops are ignored but endpoints still validated. Largest-off-diagonal Jacobi rotations start from identity eigenvectors, use row-major strict maximum ties, then stable ascending eigenvalue order. Select columns1 and2, falling back to column0 for n=1 and repeating column1 for n=2; center and divide both coordinates by maximum radial norm. Empty gives empty converged result. For connected graphs, these are low nonconstant modes; disconnected graphs can select additional zero modes. Degenerate eigenvalues allow rotations/sign changes and layouts are not unique. Approximate finite arithmetic, not a planarity solver. Actual convergence flag is separate from cap exhaustion. Each largest-pivot scan costs O(n^2), each rotation update O(n); up to O(n^2) rotations therefore permits O(n^4) scan work, not an unconditional O(n^3) bound. Dense storage O(n^2), optional history O(rotations*n). Owned outputs and no shared state.
/// @param vertex_count Complete vertex order, labels0..n-1.
/// @param edges Borrowed complete undirected endpoint pairs.
/// @param options Tolerance and complete vertex/actual rotation/scan/capture policies.
/// @return Owned actual positions, Laplacian/eigensystem, stopping state, counters and optional history.
/// @throws std::invalid_argument Endpoint out of range or invalid tolerance.
/// @throws std::length_error Geometry, actual scan/history or counter/container capacity exceeded.
/// @throws std::overflow_error Actual matrix/eigenvector/layout arithmetic nonfinite.
/// @par Example
/// @code
/// auto r=dm::algorithms::spectral_layout_trace(2,{{0,1}});
/// std::print("{} {}\n",r.converged,r.rotations); // true 1
/// @endcode
/// @par References
/// - K. M. Hall, An r-Dimensional Quadratic Placement Algorithm, Management Science17(3),1970, DOI10.1287/mnsc.17.3.219.
/// @see spectral_layout()
SpectralLayoutRun spectral_layout_trace(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const SpectralLayoutOptions&                            options = {});
/// @brief Return the preserved spectral graph coordinates.
/// Same actual checked Jacobi producer and defaults. Throws on actual nonconvergence instead of silently returning capped modes; signs and degenerate eigenspaces need not reproduce exact coordinate bytes.
/// @param vertex_count Complete graph order.
/// @param edges Complete borrowed undirected endpoint pairs.
/// @return Owned centered/common-scaled planar positions.
/// @throws std::invalid_argument Malformed graph/controls.
/// @throws std::length_error Default geometry/work/capacity exceeded.
/// @throws std::overflow_error Actual arithmetic nonfinite.
/// @throws std::runtime_error Jacobi rotation cap reached before convergence.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::spectral_layout(4,{{0,1},{1,2},{2,3}}).size()); // 4
/// @endcode
/// @see spectral_layout_run()
std::vector<Point2D> spectral_layout(
    std::size_t                                             vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges);
/// @brief Generate a bounded path graph edge fixture.
/// Consecutive edges(0,1),...,(n-2,n-1), empty below2. At most one million vertices; pure owned structural fixture.
/// @param n Complete path order.
/// @return Owned ordered endpoint pairs.
/// @throws std::length_error Fixture count exceeds one million.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::spectral_path_edges(4).size()); // 3
/// @endcode
/// @see spectral_layout_run()
std::vector<std::pair<std::size_t, std::size_t>> spectral_path_edges(std::size_t n);
} // namespace datamunge::algorithms
