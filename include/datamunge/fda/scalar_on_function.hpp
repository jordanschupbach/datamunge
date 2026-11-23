#pragma once

/// @file scalar_on_function.hpp
/// @brief Scalar regression on a functional predictor with a smooth spline coefficient.

#include <datamunge/fda/pspline.hpp>

#include <vector>

namespace datamunge::fda {

/// @brief Fitted intercept and spline coefficient for a common-grid functional predictor.
/// @par Example
/// @code
/// const dm::fda::ScalarOnFunctionFit fit;
/// std::println("{}", fit.coefficients.size()); // 0
/// @endcode
struct ScalarOnFunctionFit {
    BSplineBasis basis;                   ///< Owned basis of the coefficient function.
    std::vector<double> grid;              ///< Integration grid in training order.
    double intercept{};                   ///< Unpenalized scalar intercept.
    std::vector<double> coefficients;      ///< Spline coefficients in basis order.
    std::vector<double> fitted;            ///< Fitted scalar responses in training order.
    std::vector<double> residuals;         ///< Observed minus fitted scalar responses.
    double roughness{};                   ///< Sum of squared coefficient differences before weighting by lambda.

    /// @brief Evaluates the estimated coefficient function at finite points.
    /// @param points Evaluation coordinates in any order, with zero outside the basis domain.
    /// @return Coefficient values in point order.
    /// @throws std::invalid_argument For malformed basis coefficients or nonfinite points.
    /// @throws std::overflow_error If a result is outside the finite range.
    /// @par Example
    /// @code
    /// using dm::linalg::DenseMatrix;
    /// const auto x = DenseMatrix<double>(3,2,{0,0,1,0,0,1});
    /// const auto fit = dm::fda::fit_scalar_on_function({0,1},x,{2,3,4},
    ///     dm::fda::BSplineBasis::open_uniform(1,2),{0,1});
    /// std::println("{:.1f}", fit.coefficient_function({1})[0]); // 4.0
    /// @endcode
    [[nodiscard]] std::vector<double> coefficient_function(const std::vector<double>& points) const;

    /// @brief Predicts scalar outcomes from new curves measured on the training grid.
    /// @param curves Subjects by grid points, with all entries finite.
    /// @return Predicted values in subject order.
    /// @throws std::invalid_argument For malformed curves or fit parameters.
    /// @throws std::length_error If the required dense design exceeds storage capacity.
    /// @throws std::overflow_error If design values or predictions exceed the finite range.
    /// @par Example
    /// @code
    /// using dm::linalg::DenseMatrix;
    /// const auto x = DenseMatrix<double>(3,2,{0,0,1,0,0,1});
    /// const auto fit = dm::fda::fit_scalar_on_function({0,1},x,{2,3,4},
    ///     dm::fda::BSplineBasis::open_uniform(1,2),{0,1});
    /// std::println("{:.1f}", fit.predict(DenseMatrix<double>(1,2,{1,1}))[0]); // 5.0
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& curves) const;
};

/// @brief Fits a spline coefficient for one functional predictor and an unpenalized intercept.
/// Forms the integrated predictors by trapezoidal quadrature of each curve
/// times each spline basis function. Minimizes residual sum of squares plus
/// lambda times the squared adjacent coefficient differences of the requested
/// order. An augmented least-squares SVD tests identification after applying
/// the penalty, including its null space.
/// @param grid Finite strictly increasing points within the basis domain.
/// @param curves Subjects by grid points, at least two complete curves.
/// @param response Finite paired response values in subject order.
/// @param basis Native B-spline basis for the coefficient function.
/// @param options P-spline smoothing strength and difference order.
/// @return Intercept, spline coefficient, fitted values, residuals and unscaled roughness.
/// @throws std::invalid_argument For inconsistent dimensions, invalid controls or nonfinite inputs.
/// @throws std::domain_error If the augmented design is rank deficient.
/// @throws std::length_error If design dimensions exceed storage capacity.
/// @throws std::runtime_error If the SVD fails to converge.
/// @throws std::overflow_error If a required weight or result exceeds the finite range.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const auto x = DenseMatrix<double>(3,2,{0,0,1,0,0,1});
/// const auto fit = dm::fda::fit_scalar_on_function({0,1},x,{2,3,4},
///     dm::fda::BSplineBasis::open_uniform(1,2),{0,1});
/// std::println("{:.1f}", fit.intercept); // 2.0
/// @endcode
/// @par References
/// Marx and Eilers (1999), https://doi.org/10.1080/00401706.1999.10485591
[[nodiscard]] ScalarOnFunctionFit fit_scalar_on_function(const std::vector<double>& grid,
    const linalg::DenseMatrix<double>& curves, const std::vector<double>& response,
    const BSplineBasis& basis, const PSplineOptions& options = {});

} // namespace datamunge::fda
