#pragma once
/// @file embedding_diagnostics.hpp
/// @brief Shared reproducible embedding fixtures and numeric diagnostics.
#include <cstddef>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <string>
#include <vector>
namespace datamunge::stats {
/// @brief Generate the original noiseless Swiss roll with retained truth.
/// Two uniform draws per row: t=1.5*pi*(1+2u), height=21u, x=t*cos(t), y=height, z=t*sin(t). Owned data, no shared state; exact stream depends on C++ standard-library uniform distribution.
/// @param n Number of points, default400; positive, at most2000.
/// @param seed mt19937_64 seed, default42.
/// @return DataFrame x,y,z,t,height; t and height are withheld ground truth.
/// @throws std::invalid_argument Invalid n.
/// @par Example
/// @code
/// auto d=dm::stats::embedding_swiss_roll(8);
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
dstruct::DataFrame embedding_swiss_roll(std::size_t n = 400, std::uint64_t seed = 42);
/// @brief Summarize pair distances within and between external groups.
/// Groups never enter fitting. O(n^2*d) work, constant accumulator storage; empty accepted, at most2000 rows.
/// @param coordinates Finite n-by-d embedding.
/// @param labels Exactly n labels, aligned to coordinates.
/// @return One-row numeric DataFrame: within,between,within_pairs,between_pairs. Missing pair types have NaN mean.
/// @throws std::invalid_argument Shape mismatch or nonfinite coordinates.
/// @throws std::length_error Too many rows.
/// @par Example
/// @code
/// dm::linalg::DenseMatrix<double> y(2,1,0); auto d=dm::stats::embedding_group_distances(y,{"a","b"});
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
dstruct::DataFrame embedding_group_distances(
    const linalg::DenseMatrix<double>& coordinates,
    const std::vector<std::string>&    labels);
/// @brief Compute absolute Pearson correlation of two paired finite vectors.
/// @param a First vector.
/// @param b Second vector of matching size.
/// @return Absolute centered correlation; NaN for fewer than2 points or constant vectors.
/// @throws std::invalid_argument Shape mismatch or nonfinite input.
/// @par Example
/// @code
/// double r=dm::stats::embedding_abs_correlation({1,2,3},{3,2,1}); //1
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
double embedding_abs_correlation(const std::vector<double>& a, const std::vector<double>& b);
} // namespace datamunge::stats
