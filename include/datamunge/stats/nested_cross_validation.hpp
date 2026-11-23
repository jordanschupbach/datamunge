#pragma once

/// @file nested_cross_validation.hpp
/// @brief Fold-local tuning and held-out predictions for scalar-outcome models.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Fits a fresh model from the training rows and predicts the query rows in order.
using NestedPredictor = std::function<std::vector<double>(
    const linalg::DenseMatrix<double>& training_predictors, const std::vector<double>& training_response,
    const linalg::DenseMatrix<double>& query_predictors)>;

/// @brief Finite scalar loss from an observed outcome and prediction, with smaller values preferred.
using NestedLoss = std::function<double(double observed, double prediction)>;

/// @brief One model setting, expressed as a name and an isolated fit-and-predict function.
/// @par Example
/// @code
/// const dm::stats::NestedCandidate candidate{"baseline",{}};
/// std::println("{}", candidate.name); // baseline
/// @endcode
struct NestedCandidate {
    std::string name;          ///< User-defined setting label returned in the selected fold.
    NestedPredictor predict;   ///< Refit callback, using only the supplied training data.
};

/// @brief Inner and outer balanced K-fold controls.
/// @par Example
/// @code
/// const dm::stats::NestedCVOptions options{.outer_folds=3};
/// std::println("{}", options.outer_folds); // 3
/// @endcode
struct NestedCVOptions {
    int outer_folds{5};    ///< Number of disjoint held-out folds.
    int inner_folds{4};    ///< Number of tuning folds within each outer training partition.
    std::uint64_t seed{1}; ///< Seed for reproducible outer and inner shuffles.
    bool shuffle{true};   ///< Shuffle rows before drawing each balanced partition.
};

/// @brief Selection and evaluation details for one outer fold.
/// @par Example
/// @code
/// const dm::stats::NestedCVFold fold;
/// std::println("{}", fold.test_rows.size()); // 0
/// @endcode
struct NestedCVFold {
    std::vector<std::size_t> test_rows; ///< Original row indices in prediction order.
    std::vector<double> inner_loss;    ///< Row-weighted validation loss for each candidate in input order.
    std::size_t selected_index{};      ///< Input candidate selected by minimum inner loss.
    std::string selected_name;         ///< Name of the selected setting.
    double test_loss{};                ///< Mean loss on this outer fold.
};

/// @brief Out-of-fold predictions and the selected settings used to produce them.
/// @par Example
/// @code
/// const dm::stats::NestedCVResult result;
/// std::println("{}", result.predictions.size()); // 0
/// @endcode
struct NestedCVResult {
    std::vector<double> predictions;  ///< One outer prediction per original row.
    std::vector<NestedCVFold> folds;   ///< Selection and loss for each outer fold.
    double mean_loss{};                ///< Mean outer loss across individual rows.
};

/// @brief Tunes each outer training partition by inner CV, then refits and predicts its test rows.
/// All candidates receive identical inner folds within a given outer fold.
/// Lower mean loss wins, with ties resolved by candidate order. The
/// default loss is squared error. Random row splitting assumes exchangeable
/// observations. Callbacks must fit from the supplied training rows on every call.
/// @param predictors Finite rows of numeric features.
/// @param response Finite response values in the same row order.
/// @param candidates Nonempty list of model settings to compare.
/// @param options Fold counts, shuffle and seed.
/// @param loss Optional scalar evaluation function, with smaller values preferred.
/// @return Held-out predictions and selected settings with inner and outer loss.
/// @throws std::invalid_argument For invalid geometry, callbacks or prediction count.
/// @throws std::domain_error For nonfinite predictions or losses.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6,1,1.0);
/// const dm::stats::NestedCandidate mean{"mean",[](const auto&,const auto& y,const auto& query) {
///     return std::vector<double>(query.rows(),y.front());
/// }};
/// const auto result=dm::stats::nested_cross_validate(x,std::vector<double>(6,2),{mean},
///     {.outer_folds=3,.inner_folds=2});
/// std::println("{:.0f}", result.predictions[0]); // 2
/// @endcode
/// @par References
/// Varma and Simon (2006), https://doi.org/10.1186/1471-2105-7-91
[[nodiscard]] NestedCVResult nested_cross_validate(
    const linalg::DenseMatrix<double>& predictors, const std::vector<double>& response,
    const std::vector<NestedCandidate>& candidates, const NestedCVOptions& options = {},
    const NestedLoss& loss = {});

} // namespace datamunge::stats
