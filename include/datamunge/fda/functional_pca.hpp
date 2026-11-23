#pragma once

/// @file functional_pca.hpp
/// @brief Trapezoidal-quadrature principal components for curves on a shared grid.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::fda {

/// @brief Eigenfunctions, scores and integrated sample variances of curves on a common grid.
/// @par Example
/// @code
/// dm::fda::FunctionalPCAFit fit;
/// std::println("{}", fit.scores.cols()); // 0
/// @endcode
struct FunctionalPCAFit {
    std::vector<double> mean;           ///< Pointwise sample mean in grid order.
    std::vector<double> weights;        ///< Trapezoidal quadrature weights in grid units.
    linalg::DenseMatrix<double> eigenfunctions; ///< Grid points by components, orthonormal under the quadrature weights.
    linalg::DenseMatrix<double> scores; ///< Subjects by components, centered weighted inner products.
    std::vector<double> eigenvalues;    ///< Integrated sample variances in descending order.
    std::vector<double> explained_fraction; ///< Eigenvalues divided by the total integrated sample variance.
    std::size_t rank{};                 ///< Number of positive-variance directions at the numerical rank threshold.
};

/// @brief Fits functional PCA to fully observed curves on a shared increasing grid.
/// Applies the thin SVD to the centered data weighted by the square roots
/// of trapezoidal quadrature weights. Score variances use the sample divisor
/// n-1. Each sign makes the largest absolute eigenfunction value nonnegative.
/// @param grid Finite strictly increasing grid points, at least two.
/// @param curves Subjects by grid points, at least two rows, all entries finite.
/// @param components Number of positive-variance directions to return.
/// @return Eigenfunctions and scores with one column per requested component.
/// @throws std::invalid_argument For malformed grid or curves, or an impossible component count.
/// @throws std::domain_error If the centered data have fewer positive-variance directions than requested.
/// @throws std::length_error If the weighted data exceed dense storage capacity.
/// @throws std::runtime_error If the SVD fails to converge.
/// @throws std::overflow_error If weights or results are outside the representable range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(3,2,{-1,-1,0,0,1,1});
/// const auto fit = dm::fda::functional_pca({0,2},x,1);
/// std::println("{:.1f}", fit.eigenvalues[0]); // 2.0
/// @endcode
/// @par References
/// Ramsay and Silverman (2005), Functional Data Analysis, chapter 8, https://doi.org/10.1007/b98888
[[nodiscard]] FunctionalPCAFit functional_pca(const std::vector<double>& grid,
    const linalg::DenseMatrix<double>& curves, std::size_t components);

} // namespace datamunge::fda
