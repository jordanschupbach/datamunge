#pragma once

/// @file mean_shift.hpp
/// @brief Density modes and basin assignments by Gaussian mean shift.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Iteration and endpoint-merging controls for Gaussian mean shift.
/// @par Example
/// @code
/// const dm::stats::MeanShiftOptions options{.max_iterations=500};
/// std::println("{}", options.max_iterations); // 500
/// @endcode
struct MeanShiftOptions {
    std::size_t max_iterations{300}; ///< Maximum updates per starting observation.
    double tolerance{1e-8};          ///< Convergence displacement as a fraction of bandwidth.
    double merge_tolerance{1e-3};    ///< Merge distance as a fraction of bandwidth, including zero.
};

/// @brief Recovered modes, log densities, and per-observation basin assignments.
/// @par Example
/// @code
/// const dm::stats::MeanShiftFit fit;
/// std::println("{}", fit.modes.rows()); // 0
/// @endcode
struct MeanShiftFit {
    linalg::DenseMatrix<double> modes; ///< One estimated mode or provisional endpoint per row.
    std::vector<double> log_density;   ///< Normalized Gaussian kernel log density at each reported mode.
    std::vector<std::size_t> labels;   ///< Row of modes reached from each input observation.
    std::vector<std::uint8_t> converged; ///< One when the corresponding trajectory reached a mode within tolerance.
    std::vector<std::size_t> iterations; ///< Updates taken by each trajectory.
};

/// @brief Starts Gaussian mean shift at every observation and clusters their endpoints.
/// Uses equal isotropic Gaussian kernels with the same bandwidth. A stationary
/// endpoint with an uphill curvature direction is perturbed and iterated
/// again. Endpoints are considered in convergence and descending density
/// order. Each joins the nearest existing representative within the merge
/// distance or starts a new cluster. Unfinished endpoints can be reported
/// as provisional modes, with their convergence flags unset.
/// @param observations Finite numeric matrix, observations in rows and features in columns.
/// @param bandwidth Positive finite Gaussian standard deviation in feature units.
/// @param options Iteration limit and tolerances relative to the bandwidth.
/// @return Encountered modes and basin labels, with trajectory diagnostics.
/// @throws std::invalid_argument For invalid data shapes, values or controls.
/// @throws std::length_error If a curvature workspace exceeds storage capacity.
/// @throws std::overflow_error If a mode or reported log density is outside finite range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,.1,4,4.1});
/// const auto fit=dm::stats::fit_mean_shift(x,.25);
/// std::println("{}", fit.modes.rows()); // 2
/// @endcode
/// @par References
/// Carreira-Perpiñán (2007), https://doi.org/10.1109/TPAMI.2007.1057
[[nodiscard]] MeanShiftFit fit_mean_shift(const linalg::DenseMatrix<double>& observations,
    double bandwidth, const MeanShiftOptions& options = {});

} // namespace datamunge::stats
