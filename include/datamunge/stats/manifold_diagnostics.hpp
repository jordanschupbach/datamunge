#pragma once
/// @file manifold_diagnostics.hpp
/// @brief Metric-scale and external centroid diagnostics for dense embeddings.
#include <datamunge/linalg/dense_matrix.hpp>
#include <string>
#include <vector>
namespace datamunge::stats {
/// @brief Squared-distance summaries, without automatically choosing a bandwidth.
/// Upper medians use sorted index size/2, retaining the original report convention
/// for even sample counts. Duplicates contribute zero distances. All values are in
/// squared feature units. No fitted labels or automatic standardization are involved.
/// @par Example
/// See manifold_distance_scales().
/// @see manifold_distance_scales()
struct ManifoldDistanceScales {
    double upper_median_squared; ///< Upper median over unordered distinct-row pairs.
    double mean_squared; ///< Mean over unordered distinct-row pairs.
    double upper_median_neighbor_squared; ///< Upper median of each row's kth other-row squared distance.
};
/// @brief Measures global and local Euclidean scales on an explicitly prepared matrix.
/// Borrowed matrix is unchanged; all entries must be finite, no row dropping.
/// Work O(n*n*(p+log n)), storage O(n*n). The neighbor diagnostic is a scale
/// reference, not an estimator guaranteeing a connected or useful diffusion kernel.
/// @param points n by p raw features, n>=2 and p>=1.
/// @param neighbor One-based other-row neighbor rank, default eight; 1<=neighbor<n.
/// @return Owned upper-median, mean and local upper-median squared distances.
/// @throws std::invalid_argument Bad shape/rank or nonfinite entries.
/// @throws std::length_error Pair-count capacity exceeded before pair allocation.
/// @throws std::overflow_error A squared distance cannot be represented as finite double.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(3,1,{0,1,3});
/// const auto s=dm::stats::manifold_distance_scales(x,1);
/// std::print("{:.0f} {:.0f}\n",s.upper_median_squared,s.upper_median_neighbor_squared); // 4 1
/// @endcode
/// @see DiffusionMapsOptions
[[nodiscard]] ManifoldDistanceScales manifold_distance_scales(const linalg::DenseMatrix<double>& points,std::size_t neighbor=8);

/// @brief Unnormalised class-centroid sums and their external separation ratio.
/// Total scatter equals within plus between, up to floating-point error.
/// Ratios use no degrees-of-freedom correction and are not F statistics.
/// @par Example
/// See embedding_separation().
/// @see embedding_separation()
struct EmbeddingSeparation {
    double within; ///< Sum over rows of squared distance to its labelled centroid.
    double between; ///< Sum over labels of class size times squared centroid-to-grand-mean distance.
    double ratio; ///< Between/within; infinity for positive/zero, NaN for zero/zero.
};
/// @brief Computes descriptive between/within centroid scatter on aligned labelled coordinates.
/// Invariant to translation, orthogonal transformations, uniform nonzero scaling and
/// label renaming; changing individual coordinate scales can change the ratio.
/// Labels are external evaluation inputs, never training inputs. Work O(n*p*log g),
/// storage O(g*p), g distinct labels. No mutation or shared mutable state.
/// @param embedding Finite nonempty n by p matrix.
/// @param labels Exactly n strings, including ordinary empty-string labels.
/// @return Owned sums and ratio; one label has between zero.
/// @throws std::invalid_argument Empty/mismatched shape or nonfinite coordinates.
/// @throws std::overflow_error A scatter sum is outside finite double range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,2,4,6});
/// const auto s=dm::stats::embedding_separation(x,{"a","a","b","b"});
/// std::print("{:.0f} {:.0f} {:.0f}\n",s.within,s.between,s.ratio); // 4 16 4
/// @endcode
/// @see manifold_distance_scales()
[[nodiscard]] EmbeddingSeparation embedding_separation(const linalg::DenseMatrix<double>& embedding,const std::vector<std::string>& labels);
} // namespace datamunge::stats
