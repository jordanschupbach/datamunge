#pragma once

/// @file distance_correlation.hpp
/// @brief Distance covariance and correlation (Székely, Rizzo and Bakirov 2007): V-statistic and bias-corrected
/// estimates, the permutation independence test, the high-dimensional t test and partial distance correlation.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Options for distance_correlation().
/// @par Example
/// @code
/// dm::stats::DistanceCorrelationOptions o{.permutations = 999};
/// std::println("{}", o.permutations); // 999
/// @endcode
struct DistanceCorrelationOptions {
    double index{1.0};            ///< Exponent on Euclidean distances, in (0, 2).
    std::size_t permutations{0};  ///< Permutation replicates for the nV^2 test (0: no test).
    std::uint64_t seed{1};        ///< Permutation seed.
};

/// @brief Distance covariance/correlation estimates and tests.
/// @par Example
/// @code
/// dm::stats::DistanceCorrelation d{.dcor = 0.5};
/// std::println("{}", d.dcor); // 0.5
/// @endcode
struct DistanceCorrelation {
    std::size_t n{};                ///< Sample size.
    double dcov{};                  ///< V-statistic distance covariance \f$\mathcal V_n(X,Y)\f$ (`energy::dcov`).
    double dvar_x{};                ///< \f$\mathcal V_n(X,X)\f$.
    double dvar_y{};                ///< \f$\mathcal V_n(Y,Y)\f$.
    double dcor{};                  ///< V-statistic distance correlation \f$\mathcal R_n\f$ in [0, 1] (`energy::dcor`).
    double dcov_squared_unbiased{}; ///< U-statistic \f$\Omega_n\f$, unbiased for \f$\mathcal V^2(X,Y)\f$.
    double dcor_bias_corrected{};   ///< Bias-corrected \f$\mathcal R^*_n\f$ (`energy::bcdcor`), may be negative.
    double t_statistic{};           ///< \f$T=\sqrt{M-1}\,\mathcal R^*_n/\sqrt{1-(\mathcal R^*_n)^2}\f$, \f$M=n(n-3)/2\f$.
    double t_df{};                  ///< M - 1.
    double t_p_value{};             ///< One-sided t-test p-value (`energy::dcorT.test`).
    double permutation_statistic{}; ///< \f$n\mathcal V_n^2\f$.
    double permutation_p_value{std::numeric_limits<double>::quiet_NaN()}; ///< (1 + #{replicates >= observed}) / (R + 1).
    std::size_t permutations{};     ///< Replicates used.
};

/// @brief Distance correlation between two samples of vectors (rows are observations).
/// @param x n x p matrix.
/// @param y n x q matrix.
/// @param options Options.
/// @return Estimates and tests (the t test needs n >= 4).
/// @throws std::invalid_argument On row mismatch, n < 4, nonfinite values or an index outside (0, 2).
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5, 1, {1, 2, 3, 4, 5}), y(5, 1, {1, 4, 9, 16, 25});
/// std::println("{:.4f}", dm::stats::distance_correlation(x, y).dcor); // 0.9869
/// @endcode
DistanceCorrelation distance_correlation(const linalg::DenseMatrix<double>& x, const linalg::DenseMatrix<double>& y,
                                         const DistanceCorrelationOptions& options = {});

/// @brief Distance correlation between two univariate samples.
/// @param x First sample.
/// @param y Second sample.
/// @param options Options.
/// @return Estimates and tests.
/// @throws std::invalid_argument As for the matrix overload.
/// @par Example
/// @code
/// const std::vector<double> x{-2, -1, 0, 1, 2}, y{4, 1, 0, 1, 4};
/// std::println("{:.4f}", dm::stats::distance_correlation(x, y).dcor); // 0.5159
/// @endcode
DistanceCorrelation distance_correlation(const std::vector<double>& x, const std::vector<double>& y,
                                         const DistanceCorrelationOptions& options = {});

/// @brief Partial distance correlation of x and y given z (Székely and Rizzo 2014, `energy::pdcor`).
/// @param x n x p matrix.
/// @param y n x q matrix.
/// @param z n x r matrix.
/// @param index Distance exponent in (0, 2).
/// @return \f$(\mathcal R^*_{xy}-\mathcal R^*_{xz}\mathcal R^*_{yz})/\sqrt{(1-\mathcal R^{*2}_{xz})(1-\mathcal R^{*2}_{yz})}\f$
/// (0 when a denominator factor vanishes).
/// @throws std::invalid_argument As for distance_correlation().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5, 1, {1, 2, 3, 4, 5}), y(5, 1, {2, 1, 4, 3, 5}), z(5, 1, {1, 1, 2, 2, 3});
/// std::println("{}", dm::stats::partial_distance_correlation(x, y, z) < 1); // true
/// @endcode
double partial_distance_correlation(const linalg::DenseMatrix<double>& x, const linalg::DenseMatrix<double>& y,
                                    const linalg::DenseMatrix<double>& z, double index = 1.0);

} // namespace datamunge::stats
