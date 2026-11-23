#pragma once

/// @file simpls.hpp
/// @brief SIMPLS regression and held-out component selection.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Preprocessing and numerical controls for SIMPLS.
/// @par Example
/// @code
/// const dm::stats::SIMPLSOptions options{.scale_predictors=true};
/// std::println("{}", options.scale_predictors); // true
/// @endcode
struct SIMPLSOptions {
    bool scale_predictors{false}; ///< Standardize nonconstant predictor columns to unit sample variance.
    bool scale_responses{false};  ///< Standardize nonconstant response columns before finding components.
    double tolerance{1e-12};      ///< Relative cutoff for remaining covariance and numerical rank.
};

/// @brief A SIMPLS component path with weights and loadings in original data units.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,1,2,3}), y(4,1,{1,3,5,7});
/// const auto fit = dm::stats::fit_simpls(x,y,1);
/// std::println("{:.1f}", fit.coefficients()(0,0)); // 2.0
/// @endcode
struct SIMPLSFit {
    std::vector<double> predictor_means;       ///< Column means of the training predictors.
    std::vector<double> response_means;        ///< Column means of the training responses.
    linalg::DenseMatrix<double> score_weights; ///< Original-scale coefficients for scores on centered predictors.
    linalg::DenseMatrix<double> scores;        ///< Unit-norm score columns in training row order.
    linalg::DenseMatrix<double> x_loadings;    ///< Cross-products of the centered predictors and scores.
    linalg::DenseMatrix<double> y_loadings;    ///< Cross-products of the centered responses and scores.

    /// @brief Returns the number of estimable components retained.
    /// @return Number of score columns, possibly zero.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4,1,{0,1,2,3}), y(4,1,{1,3,5,7});
    /// std::println("{}", dm::stats::fit_simpls(x,y,1).components()); // 1
    /// @endcode
    [[nodiscard]] std::size_t components() const noexcept { return score_weights.cols(); }

    /// @brief Returns regression slopes for a prefix of the fitted components.
    /// @param count Number of components. Omit it to use all. Zero returns zero slopes.
    /// @return Predictor-by-response coefficient matrix.
    /// @throws std::invalid_argument Component count or fitted dimensions are invalid.
    /// @throws std::overflow_error A slope exceeds finite double range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4,1,{0,1,2,3}), y(4,1,{1,3,5,7});
    /// std::println("{:.1f}", dm::stats::fit_simpls(x,y,1).coefficients()(0,0)); // 2.0
    /// @endcode
    [[nodiscard]] linalg::DenseMatrix<double> coefficients(std::optional<std::size_t> count = std::nullopt) const;

    /// @brief Predicts all responses from new predictor rows using a component prefix.
    /// @param predictors Finite predictor rows in training column order.
    /// @param count Number of components. Omit it to use all. Zero predicts the training response means.
    /// @return Predictions in row-by-response order.
    /// @throws std::invalid_argument Incompatible predictors, component count or fitted dimensions.
    /// @throws std::overflow_error A prediction exceeds finite double range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4,1,{0,1,2,3}), y(4,1,{1,3,5,7});
    /// const auto fit = dm::stats::fit_simpls(x,y,1);
    /// std::println("{:.1f}", fit.predict(dm::linalg::DenseMatrix<double>(1,1,{4}))(0,0)); // 9.0
    /// @endcode
    [[nodiscard]] linalg::DenseMatrix<double> predict(const linalg::DenseMatrix<double>& predictors,
        std::optional<std::size_t> count = std::nullopt) const;
};

/// @brief Fits partial least squares regression using SIMPLS covariance deflation.
/// @param predictors Finite predictors, one observation per row.
/// @param response One or more finite responses in the same row order.
/// @param max_components Largest component count to fit. Zero fits an intercept only.
/// @param options Scaling and numerical rank controls.
/// @return The estimable prefix of the requested path and training scores.
/// @throws std::invalid_argument Invalid shapes, options or nonfinite data.
/// @throws std::length_error A working matrix exceeds its storage capacity.
/// @throws std::runtime_error The required singular value decomposition failed to converge.
/// @throws std::overflow_error A fitted result exceeds the finite reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,1,2,3}), y(4,1,{1,3,5,7});
/// std::println("{}", dm::stats::fit_simpls(x,y,3).components()); // 1
/// @endcode
/// @par References
/// - de Jong (1993), https://doi.org/10.1016/0169-7439(93)85002-X
[[nodiscard]] SIMPLSFit fit_simpls(const linalg::DenseMatrix<double>& predictors,
    const linalg::DenseMatrix<double>& response, std::size_t max_components, const SIMPLSOptions& options = {});

/// @brief Cross-validation loss path and a model fitted on all rows at the selected component count.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,1,2,3}), y(4,1,{1,3,5,7});
/// const auto cv = dm::stats::cross_validate_simpls(x,y,{1,1,2,2},2);
/// std::println("{}", cv.selected_components); // 1
/// @endcode
struct SIMPLSCrossValidation {
    std::size_t selected_components;         ///< Smallest component count attaining the minimum held-out loss.
    std::vector<double> mean_squared_error; ///< Pooled held-out MSE for each available count, starting at zero.
    SIMPLSFit fit;                           ///< Model refitted on all rows with the selected number of components.
};

/// @brief Selects the number of components by holding out each supplied fold in turn.
///
/// Centering and scaling are fitted on each training fold. Each held-out
/// observation and response contributes equally in original response units.
/// Candidate counts include the intercept model and extend to the largest
/// count estimable in every training fold and the full data.
/// @param predictors Finite predictors, one observation per row.
/// @param response Finite responses in the same row order.
/// @param fold One fold label per observation. Each fold must leave at least two training rows.
/// @param max_components Largest candidate component count.
/// @param options Scaling and numerical rank controls.
/// @return Held-out loss path and selected full-data model.
/// @throws std::invalid_argument Invalid data, options, or fold partition.
/// @throws std::length_error A working matrix exceeds its storage capacity.
/// @throws std::runtime_error The required singular value decomposition failed to converge.
/// @throws std::overflow_error A fitted result or loss exceeds the finite reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,1,2,3}), y(4,1,{1,3,5,7});
/// const auto cv = dm::stats::cross_validate_simpls(x,y,{1,1,2,2},2);
/// std::println("{:.1f}", cv.mean_squared_error[1]); // 0.0
/// @endcode
[[nodiscard]] SIMPLSCrossValidation cross_validate_simpls(const linalg::DenseMatrix<double>& predictors,
    const linalg::DenseMatrix<double>& response, const std::vector<std::int64_t>& fold,
    std::size_t max_components, const SIMPLSOptions& options = {});

} // namespace datamunge::stats
