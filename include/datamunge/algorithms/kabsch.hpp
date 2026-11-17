#pragma once
/// @file kabsch.hpp
/// @brief Checked Horn-quaternion rigid alignment with actual covariance and eigensystem evidence.
#include <array>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned three Cartesian double coordinates; units follow the caller's paired data.
/// @par Example
/// @code
/// const dm::algorithms::Point3 p={1,2,3};std::print("{}\n",p[1]); // 2
/// @endcode
/// @see kabsch()
using Point3 = std::array<double, 3>;
/// @brief Owned proper-rotation/translation fit mapping source onto corresponding target points.
/// Original field order/types retained. No scale/reflection/permutation fitting.
/// RMSD evaluates the returned double transform on the original paired inputs.
/// Degenerate data can admit multiple equally optimal rotations; one is returned.
/// @par Example
/// @code
/// const auto r=dm::algorithms::kabsch({{0,0,0}},{{1,2,3}});
/// std::print("{} {}\n",r.translation[0],r.rmsd); // 1 0
/// @endcode
/// @see kabsch_analyze()
struct KabschResult {
  std::array<std::array<double, 3>, 3> rotation; ///< Returned proper rotation, row-major.
  Point3 translation;                            ///< Returned additive translation, target units.
  double rmsd; ///< Root mean squared Euclidean residual per paired point, not per coordinate.
};
/// @brief Owned same-execution paired fit, normalized covariance/key matrix and actual observations.
/// Normalize each input set by its max(1,absolute coordinate), then center in long
/// double. Covariance is the mean outer product in those normalized units. Divide
/// it by covariance_scale to build the dimensionless Horn key matrix; scale zero
/// uses an all-zero key. Eigenvalues are descending and quaternion is the actual
/// normalized principal eigenvector, scalar-first. No second fit or iteration
/// replay. Eigen residual/gap are additional diagnostics, not exact certificates
/// of rank, uniqueness or convergence in exact arithmetic. All vectors own data.
/// @par Example
/// @code
/// const auto a=dm::algorithms::kabsch_analyze({{0,0,0}},{{1,2,3}});
/// std::print("{} {}\n",a.aligned.size(),a.result.rmsd); // 1 0
/// @endcode
/// @see kabsch_analyze()
struct KabschAnalysis {
  KabschResult               result;            ///< Actual returned transform and RMSD.
  std::array<long double, 3> source_centroid{}; ///< Producing source centroid in original units.
  std::array<long double, 3> target_centroid{}; ///< Producing target centroid in original units.
  long double                source_scale{1};   ///< Source max(1,absolute input coordinate).
  long double                target_scale{1};   ///< Target max(1,absolute input coordinate).
  std::array<std::array<long double, 3>, 3>
              covariance{}; ///< Actual mean centered normalized cross-covariance.
  long double covariance_scale{
      0}; ///< Actual largest absolute covariance entry used to normalize key construction.
  std::array<std::array<long double, 4>, 4>
      key_matrix{}; ///< Actual normalized Horn matrix passed to the eigensolver.
  std::array<long double, 4> eigenvalues{}; ///< Actual descending normalized-key eigenvalues.
  std::array<long double, 4>
                      quaternion{};      ///< Actual normalized scalar-first principal eigenvector.
  long double         eigen_residual{0}; ///< Additional Euclidean norm of Nq-lambda*q.
  long double         eigen_gap{0};      ///< Additional largest-minus-second eigenvalue difference.
  std::vector<Point3> source; ///< Owned original paired source coordinates, analysis only.
  std::vector<Point3> target; ///< Owned original paired target coordinates, analysis only.
  std::vector<Point3>
      aligned; ///< Actual application of returned transform, rounded to double coordinates.
  std::vector<long double>
      residual_norms; ///< Actual widened Euclidean residual norms before coordinate rounding.
};
/// @brief Fit a proper rigid transform to nonempty finite corresponding point sets.
/// Equal sizes are required, order defines correspondence. Uses Horn's principal
/// quaternion formulation of Kabsch's objective. Independent positive input
/// normalization and long-double mean covariance avoid raw square/product
/// overflow; no additional scale is fitted. A normalized quaternion gives a
/// proper rotation, then t=target_mean-R*source_mean. Numerical degeneracy may
/// choose any optimum; zero covariance chooses the eigensolver's identity basis.
/// Computes residuals using the returned double transform with widened arithmetic
/// and stable incremental RMS norm. Floating normalization/eigensolving are not
/// exact predicates; last bits may differ from the former raw-double kernel.
/// O(n) work, constant-sized eigensolve and O(1) auxiliary storage apart from its
/// fixed-size allocations. Input unchanged, result owned, no shared state.
/// @param source Borrowed nonempty finite points.
/// @param target Borrowed finite corresponding points of equal size.
/// @return Owned proper rotation, translation and per-point RMSD.
/// @throws std::invalid_argument On empty/mismatched/nonfinite inputs.
/// @throws std::overflow_error On nonfinite intermediates or outputs outside finite double range.
/// @throws std::runtime_error On nonfinite/zero eigensolver quaternion or failed residual tolerance.
/// @throws std::bad_alloc On fixed-size eigensolver allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::kabsch({{0,0,0},{1,0,0}},{{1,2,3},{2,2,3}});
/// std::print("{:.6f}\n",r.rmsd); // 0.000000
/// @endcode
/// @par References
/// - Horn(1987), Closed-form solution of absolute orientation using unit quaternions,
///   https://doi.org/10.1364/JOSAA.4.000629.
/// - Kabsch(1976), A solution for the best rotation to relate two sets of vectors,
///   https://doi.org/10.1107/S0567739476001873.
/// @see kabsch_analyze()
KabschResult kabsch(const std::vector<Point3>& source, const std::vector<Point3>& target);
/// @brief Capture actual paired alignment, covariance/key matrix and principal eigensystem once.
/// Same contracts/arithmetic/degenerate conventions as kabsch(). Complete finite
/// validation and pair-count admission precede input/output vector copies. The
/// ceiling counts paired records, not the multiple vectors, bytes/capacities or
/// peak RAM; zero disables the optional cap. O(n) extra owned output, no replay.
/// Residual norms are computed before aligned coordinates are rounded to double,
/// so differ by rounding from norms recomputed on the captured aligned vector.
/// @param source Borrowed nonempty finite original source points.
/// @param target Borrowed same-sized finite original target points.
/// @param max_pairs Complete pair ceiling, default million; zero disables optional cap.
/// @return Owned same-execution fit and observations, no input references.
/// @throws std::invalid_argument On empty/mismatched/nonfinite inputs.
/// @throws std::length_error On enabled complete pair ceiling or container limits.
/// @throws std::overflow_error On arithmetic/output representability failure.
/// @throws std::runtime_error On invalid quaternion or excessive eigen residual.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::kabsch_analyze({{0,0,0}},{{1,2,3}},1);
/// std::print("{}\n",a.aligned[0][2]); // 3
/// @endcode
/// @par References
/// - Horn(1987), https://doi.org/10.1364/JOSAA.4.000629.
/// @see KabschAnalysis, kabsch()
KabschAnalysis kabsch_analyze(
    const std::vector<Point3>& source,
    const std::vector<Point3>& target,
    std::size_t                max_pairs = 1000000);
} // namespace datamunge::algorithms
