#pragma once

/// @file minimum_covariance_determinant.hpp
/// @brief Robust location, covariance and distances from the minimum covariance determinant.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Subset search and Gaussian reweighting controls.
/// @par Example
/// @code
/// const dm::stats::MCDOptions options{.support_size=6};
/// std::println("{}", options.support_size); // 6
/// @endcode
struct MCDOptions {
    std::size_t support_size{0};         ///< Number of raw subset rows. Zero uses floor((n+p+1)/2).
    std::size_t exact_subset_limit{25000}; ///< Enumerate all subsets when their number fits this limit.
    std::size_t random_starts{128};      ///< Number of elemental starts for the concentration search.
    std::size_t max_c_steps{50};         ///< Refinement concentration steps per retained start.
    double reweight_probability{.975};  ///< Chi-square probability cutoff for keeping observations.
    std::uint64_t seed{5489};             ///< Random seed for the concentration search.
};

/// @brief Raw and reweighted MCD estimates and squared Mahalanobis distances.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5,1,{0,1,2,100,200});
/// const auto fit = dm::stats::minimum_covariance_determinant(x,{.support_size=3});
/// std::println("{:.1f}", fit.location.front()); // 1.0
/// @endcode
struct MCDResult {
    std::vector<double> raw_location;       ///< Mean of the determinant-minimizing subset.
    linalg::DenseMatrix<double> raw_covariance; ///< Subset covariance with the Gaussian consistency factor.
    std::vector<std::size_t> support;       ///< Sorted original row indices of the selected subset.
    std::vector<double> raw_squared_distances; ///< Distances from the corrected raw fit in original row order.
    std::vector<double> location;           ///< Mean of the observations kept by reweighting.
    linalg::DenseMatrix<double> covariance; ///< Reweighted covariance with the Gaussian consistency factor.
    std::vector<std::size_t> reweighted_rows; ///< Sorted rows whose raw squared distances meet the cutoff.
    std::vector<double> squared_distances;   ///< Distances from the corrected reweighted fit in original row order.
    double raw_consistency_factor;           ///< Factor applied to the raw covariance with divisor support size.
    double reweight_consistency_factor;      ///< Factor applied to the reweighted covariance with divisor retained count.
    double reweight_cutoff;                  ///< Squared-distance cutoff used to select reweighted rows.
    bool exact_search;                       ///< All subsets of the requested size were considered.
    bool converged;                          ///< The exhaustive search finished or the selected concentration run stabilized.
};

/// @brief Fits minimum covariance determinant location and scatter and recomputes distances after reweighting.
///
/// Enumerates small subset spaces. Larger spaces use deterministic and
/// seeded elemental starts with concentration steps. Gaussian consistency
/// factors compensate for truncation of the raw and reweighted covariance.
/// Reweighting keeps observations whose corrected raw squared distances
/// are at most the specified chi-square quantile.
/// @param observations Finite rows with one variable per column and more rows than columns.
/// @param options Subset size, search controls, and reweighting cutoff.
/// @return The two fits, selected rows, and squared distances.
/// @throws std::invalid_argument Invalid data, subset size, or search controls.
/// @throws std::domain_error Singular data or subset scatter, or insufficient reweighted rows.
/// @throws std::length_error The dense workspace exceeds storage capacity.
/// @throws std::overflow_error A fitted covariance exceeds the reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5,1,{0,1,2,100,200});
/// const auto fit = dm::stats::minimum_covariance_determinant(x,{.support_size=3});
/// std::println("{}", fit.support.back()); // 2
/// @endcode
/// @par References
/// - Rousseeuw and Van Driessen (1999), https://doi.org/10.1080/00401706.1999.10485670
[[nodiscard]] MCDResult minimum_covariance_determinant(const linalg::DenseMatrix<double>& observations,
    const MCDOptions& options = {});

} // namespace datamunge::stats
