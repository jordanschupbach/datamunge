#pragma once

/// @file covariance_shrinkage.hpp
/// @brief Linear shrinkage estimators of a covariance matrix.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Covariance shrunk toward an identity matrix scaled by the empirical average variance.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(3,1,{-1,0,1});
/// const auto fit = dm::stats::ledoit_wolf_covariance(x);
/// std::println("{:.3f}", fit.covariance(0,0)); // 0.667
/// @endcode
struct CovarianceShrinkageResult {
    linalg::DenseMatrix<double> covariance; ///< Shrunk covariance in input-column order.
    std::vector<double> location;           ///< Column means subtracted before fitting, or zero if the mean is assumed known.
    double shrinkage;                       ///< Weight on the spherical target, in [0,1].
    double target_variance;                 ///< Common diagonal of the spherical target.
};

/// @brief Estimates covariance and spherical shrinkage intensity by the Ledoit–Wolf method.
///
/// Uses the empirical covariance with divisor equal to the observation
/// count. Estimates sampling variability from the individual outer
/// products. If the empirical covariance equals the target, the
/// reported shrinkage is one.
/// @param observations Finite data, with rows as observations and columns as variables.
/// @param assume_centered Use zero as the known mean when true. Otherwise subtract sample column means.
/// @return Covariance, centering vector, estimated intensity and target variance.
/// @throws std::invalid_argument Incompatible dimensions, too few observations, or nonfinite values.
/// @throws std::length_error The dense workspace exceeds storage capacity.
/// @throws std::overflow_error A fitted covariance exceeds the reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,2,{1,0,-1,0,0,1,0,-1});
/// std::println("{:.1f}", dm::stats::ledoit_wolf_covariance(x).shrinkage); // 1.0
/// @endcode
/// @par References
/// - Ledoit and Wolf (2004), https://doi.org/10.1016/S0047-259X(03)00096-4
[[nodiscard]] CovarianceShrinkageResult ledoit_wolf_covariance(const linalg::DenseMatrix<double>& observations,
    bool assume_centered = false);

/// @brief Estimates Gaussian covariance by oracle-approximating shrinkage.
///
/// Uses the closed-form expression of Chen and colleagues, including
/// its finite-dimension terms. Uses the observation count in the formula
/// and as the empirical covariance divisor. If the empirical covariance
/// equals the target, the reported shrinkage is one.
/// @param observations Finite data, with rows as observations and columns as variables.
/// @param assume_centered Use zero as the known mean when true. Otherwise subtract sample column means.
/// @return Covariance, centering vector, estimated intensity and target variance.
/// @throws std::invalid_argument Incompatible dimensions, too few observations, or nonfinite values.
/// @throws std::length_error The dense workspace exceeds storage capacity.
/// @throws std::overflow_error A fitted covariance exceeds the reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,2,{1,0,-1,0,0,1,0,-1});
/// std::println("{:.1f}", dm::stats::oracle_approximating_covariance(x).covariance(0,0)); // 0.5
/// @endcode
/// @par References
/// - Chen, Wiesel, Eldar and Hero (2010), https://doi.org/10.1109/TSP.2010.2053029
[[nodiscard]] CovarianceShrinkageResult oracle_approximating_covariance(
    const linalg::DenseMatrix<double>& observations, bool assume_centered = false);

} // namespace datamunge::stats
