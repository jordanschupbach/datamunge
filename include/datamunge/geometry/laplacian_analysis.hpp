#pragma once
/// @file laplacian_analysis.hpp
/// @brief Original noisy pinned-curve fixture and widened paired-point error observations.
#include <datamunge/geometry/laplacian_smoothing.hpp>
#include <vector>
namespace datamunge::geometry {
/// @brief Owned original41-vertex path, actual noisy points, pin flags and noiseless reference.
/// Synthetic reference data, not a mesh-quality or ground-truth guarantee for
/// arbitrary graph smoothing. No caller/RNG state retained.
/// @par Example
/// @code
/// const auto d=dm::geometry::laplacian_demo_curve();std::print("{} {}\n",d.vertices.size(),int(d.fixed[0])); // 41 1
/// @endcode
/// @see laplacian_demo_curve()
struct LaplacianCurveStudy {
  std::vector<Point2D>          vertices;  ///< Original noisy path coordinates, x0..40.
  std::vector<std::vector<int>> neighbors; ///< Original reciprocal nearest-path adjacency order.
  std::vector<char>             fixed;     ///< Original n flags, only endpoints pinned.
  std::vector<Point2D>          reference; ///< Original noiseless(x,8*sin(x/6)) points.
};
/// @brief Recreate the original41-point seeded noisy sinusoidal path with pinned endpoints.
/// Use mt19937 seed42 and uniform[-3,3) noise only for the39 interior y values;
/// x=i and noiseless y=8*sin(i/6). Build reciprocal path neighbors in original
/// edge order. Standard-library distribution mapping is implementation-specific.
/// @return Owned complete original data, graph, pin flags and noiseless reference.
/// @throws std::bad_alloc On point/graph/flag allocation failure.
/// @par Example
/// @code
/// const auto d=dm::geometry::laplacian_demo_curve();std::print("{}\n",d.neighbors[1].size()); // 2
/// @endcode
/// @see LaplacianCurveStudy, laplacian_smooth_trace()
LaplacianCurveStudy laplacian_demo_curve();
/// @brief Observe mean squared Euclidean difference of two matched finite nonempty point sequences.
/// Complete input validation before widened differences/squares/sum. Mean is per
/// paired point, not per coordinate. No alignment or smoothing is performed;
/// supplied reference must already match identities/units. O(n), no mutation/copies.
/// @param observed Borrowed actual complete finite point sequence.
/// @param reference Borrowed matched finite reference points, same units/identities.
/// @return Actual finite long-double mean squared Euclidean error.
/// @throws std::invalid_argument On empty/mismatched/nonfinite inputs.
/// @throws std::overflow_error On nonfinite widened arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::laplacian_point_mse({{0,0}},{{3,4}})); // 25
/// @endcode
/// @see laplacian_demo_curve()
long double laplacian_point_mse(
    const std::vector<Point2D>& observed,
    const std::vector<Point2D>& reference);
} // namespace datamunge::geometry
