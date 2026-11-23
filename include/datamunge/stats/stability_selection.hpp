#pragma once

/// @file stability_selection.hpp
/// @brief Subsample selection frequencies across a regularization grid.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief Fits to one subsample at the given regularization level and returns selected feature indices.
using StabilitySelector = std::function<std::vector<std::size_t>(
    const linalg::DenseMatrix<double>& predictors, const std::vector<double>& response, double level)>;

/// @brief Number, size and threshold of the subsample selection procedure.
/// @par Example
/// @code
/// const dm::stats::StabilitySelectionOptions options{.subsamples=50};
/// std::println("{}", options.subsamples); // 50
/// @endcode
struct StabilitySelectionOptions {
    std::size_t subsamples{100}; ///< Number of independent draws without replacement.
    double sample_fraction{.5}; ///< Fraction retained per draw, rounded down.
    double selection_threshold{.8}; ///< Inclusive lower bound on the maximum selection frequency, in (0.5, 1].
    std::uint64_t seed{1}; ///< Local random seed for subsampling.
};

/// @brief Level-specific counts and frequencies, and features selected by the stability threshold.
/// @par Example
/// @code
/// const dm::stats::StabilitySelectionResult result;
/// std::println("{}", result.stable_features.size()); // 0
/// @endcode
struct StabilitySelectionResult {
    std::vector<double> levels; ///< Regularization grid in supplied order.
    linalg::DenseMatrix<std::uint64_t> counts; ///< Number of draws selecting each feature, levels by features.
    linalg::DenseMatrix<double> frequencies; ///< Selection frequencies, levels by features.
    std::vector<double> maximum_frequency;   ///< Largest selection frequency over levels for each feature.
    std::vector<std::size_t> stable_features; ///< Zero-based features meeting the threshold, in ascending order.
    std::size_t sample_size{};               ///< Number of observations used in each draw.
};

/// @brief Fits the same regularization grid within each subsample and counts selected features.
/// Reuses the drawn rows across all levels and sends the selector fresh
/// subsets in original row order. The selector should redo preprocessing
/// and fitting from those rows on each call.
/// @param predictors Nonempty matrix of finite observations in rows and features in columns.
/// @param response Finite response values in the same row order.
/// @param levels Distinct finite nonnegative regularization levels.
/// @param select Callable returning unique zero-based selected feature indices.
/// @param options Sampling and selection threshold controls.
/// @return Level-specific frequencies and features reaching the maximum-frequency threshold.
/// @throws std::invalid_argument If the sampling, grid, data or selector output is malformed.
/// @throws std::length_error If the frequency matrices exceed storage capacity.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,2,1);
/// const dm::stats::StabilitySelector select=[](const auto&,const auto&,double level) {
///     return level < 1 ? std::vector<std::size_t>{0} : std::vector<std::size_t>{};
/// };
/// const auto result=dm::stats::stability_selection(x,{0,1,0,1},{.1,1},select,{.subsamples=5});
/// std::println("{}", result.counts(0,0)); // 5
/// @endcode
/// @par References
/// Meinshausen and Bühlmann (2010), https://doi.org/10.1111/j.1467-9868.2010.00740.x
[[nodiscard]] StabilitySelectionResult stability_selection(
    const linalg::DenseMatrix<double>& predictors, const std::vector<double>& response,
    const std::vector<double>& levels, const StabilitySelector& select,
    const StabilitySelectionOptions& options = {});

} // namespace datamunge::stats
