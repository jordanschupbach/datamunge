/// @file scalar_on_function.cpp
/// @brief Curve integration and P-spline regression with a free intercept.

#include <datamunge/fda/scalar_on_function.hpp>
#include <datamunge/linalg/svd.hpp>

#include "functional_grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::fda {
/// @brief Working precision and integrated design helpers.
namespace {

/// @brief Working regression matrix for accumulation and factorization.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Casts a numerical result to a finite double.
double report_value(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("scalar-on-function regression: value is outside the finite range");
    return static_cast<double>(value);
}

/// @brief Checks storage bounds before constructing a design or basis matrix.
template <class T>
void check_storage(std::size_t rows, std::size_t cols) {
    if (cols > 0 && rows > std::vector<T>{}.max_size() / cols)
        throw std::length_error("scalar-on-function regression: dense design exceeds storage capacity");
}

/// @brief Integrates each curve against each spline basis vector with trapezoidal weights.
Matrix integrated_design(const std::vector<double>& grid, const linalg::DenseMatrix<double>& curves,
    const BSplineBasis& basis) {
    if (grid.size() != curves.cols() || grid.size() < 2 ||
        !std::ranges::all_of(curves.data(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("scalar-on-function regression: invalid curve dimensions or values");
    const auto weights = detail::trapezoidal_weights(grid);
    const auto [left,right] = basis.domain();
    if (grid.front() < left || grid.back() > right)
        throw std::invalid_argument("scalar-on-function regression: grid is outside the basis domain");

    const auto size = basis.basis_size();
    check_storage<double>(grid.size(),size);
    check_storage<long double>(curves.rows(),size);
    const auto evaluation = basis.evaluate_dense(grid);
    Matrix design(curves.rows(),size);
    for (std::size_t i = 0; i < curves.rows(); ++i)
        for (std::size_t j = 0; j < grid.size(); ++j) {
            const long double weighted = static_cast<long double>(curves(i,j))*weights[j];
            for (std::size_t k = 0; k < size; ++k)
                design(i,k) += weighted*evaluation(j,k);
        }
    if (!std::ranges::all_of(design.data(), [](long double x) { return std::isfinite(x); }))
        throw std::overflow_error("scalar-on-function regression: integrated design overflowed");
    return design;
}

/// @brief Scores rows of an integrated design using the stored scalar intercept and coefficients.
std::vector<double> score(const Matrix& design, const ScalarOnFunctionFit& fit) {
    std::vector<double> prediction(design.rows());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double value = fit.intercept;
        for (std::size_t k = 0; k < design.cols(); ++k)
            value += design(i,k)*fit.coefficients[k];
        prediction[i] = report_value(value);
    }
    return prediction;
}

/// @brief Verifies finite fitted parameters before predicting with the mutable result.
void validate_fit(const ScalarOnFunctionFit& fit) {
    if (fit.coefficients.size() != fit.basis.basis_size() || !std::isfinite(fit.intercept) ||
        !std::ranges::all_of(fit.coefficients, [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("scalar-on-function regression: invalid fitted parameters");
}

} // namespace

std::vector<double> ScalarOnFunctionFit::coefficient_function(const std::vector<double>& points) const {
    validate_fit(*this);
    std::vector<double> response;
    response.reserve(points.size());
    for (const double point : points) {
        if (!std::isfinite(point))
            throw std::invalid_argument("scalar-on-function regression: evaluation points must be finite");
        const auto local = basis.evaluate_point(point);
        long double value = 0;
        for (std::size_t j = 0; j < local.indices.size(); ++j)
            value += static_cast<long double>(local.values[j])*coefficients[local.indices[j]];
        response.push_back(report_value(value));
    }
    return response;
}

std::vector<double> ScalarOnFunctionFit::predict(const linalg::DenseMatrix<double>& curves) const {
    validate_fit(*this);
    return score(integrated_design(grid,curves,basis),*this);
}

ScalarOnFunctionFit fit_scalar_on_function(const std::vector<double>& grid,
    const linalg::DenseMatrix<double>& curves, const std::vector<double>& response,
    const BSplineBasis& basis, const PSplineOptions& options) {
    const auto n = curves.rows(), k = basis.basis_size();
    if (n < 2 || response.size() != n ||
        !std::ranges::all_of(response, [](double x) { return std::isfinite(x); }) ||
        !(options.lambda >= 0) || !std::isfinite(options.lambda) ||
        options.penalty_order == 0 || options.penalty_order >= k)
        throw std::invalid_argument("scalar-on-function regression: invalid responses or penalty controls");

    const auto design = integrated_design(grid,curves,basis);
    const auto penalty_rows = k-options.penalty_order;
    check_storage<double>(penalty_rows,k);
    const auto difference = pspline_difference_matrix(k,options.penalty_order);
    const auto rows = options.lambda > 0 ? n+penalty_rows : n;
    if (rows < n || k == std::numeric_limits<std::size_t>::max())
        throw std::length_error("scalar-on-function regression: augmented design exceeds storage capacity");
    if (rows < k+1)
        throw std::domain_error("scalar-on-function regression: augmented design has too few rows");
    check_storage<long double>(rows,k+1);

    Matrix augmented(rows,k+1);
    std::vector<long double> target(rows);
    for (std::size_t i = 0; i < n; ++i) {
        augmented(i,0) = 1;
        target[i] = response[i];
        for (std::size_t j = 0; j < k; ++j)
            augmented(i,j+1) = design(i,j);
    }
    if (options.lambda > 0) {
        const long double scale = std::sqrt(static_cast<long double>(options.lambda));
        for (std::size_t i = 0; i < penalty_rows; ++i)
            for (std::size_t j = 0; j < k; ++j)
                augmented(n+i,j+1) = scale*difference(i,j);
    }

    // Equilibrate the augmented columns before taking their singular values.
    std::vector<long double> scale(k+1);
    for (std::size_t j = 0; j <= k; ++j) {
        for (std::size_t i = 0; i < rows; ++i)
            scale[j] = std::max(scale[j],std::abs(augmented(i,j)));
        if (!std::isfinite(scale[j]))
            throw std::overflow_error("scalar-on-function regression: augmented design overflowed");
        if (!(scale[j] > 0))
            throw std::domain_error("scalar-on-function regression: augmented design has a zero column");
        for (std::size_t i = 0; i < rows; ++i)
            augmented(i,j) /= scale[j];
    }
    const long double cutoff = rows*std::numeric_limits<double>::epsilon();
    const auto decomposition = linalg::svd(augmented,cutoff);
    // A near-zero transformed column bounds the smallest singular value
    // even if the Jacobi sweeps are still rotating other columns.
    if (decomposition.rank(cutoff) != k+1)
        throw std::domain_error("scalar-on-function regression: augmented design is rank deficient");
    if (!decomposition.converged)
        throw std::runtime_error("scalar-on-function regression: augmented SVD did not converge");
    const auto parameters = decomposition.solve(target,cutoff);
    ScalarOnFunctionFit fit{basis,grid,report_value(parameters[0]/scale[0]),
        std::vector<double>(k),{},std::vector<double>(n)};
    for (std::size_t j = 0; j < k; ++j)
        fit.coefficients[j] = report_value(parameters[j+1]/scale[j+1]);
    fit.fitted = score(design,fit);
    for (std::size_t i = 0; i < n; ++i)
        fit.residuals[i] = report_value(static_cast<long double>(response[i])-fit.fitted[i]);

    long double roughness = 0;
    for (std::size_t i = 0; i < penalty_rows; ++i) {
        long double value = 0;
        for (std::size_t j = 0; j < k; ++j)
            value += static_cast<long double>(difference(i,j))*fit.coefficients[j];
        roughness += value*value;
    }
    fit.roughness = report_value(roughness);
    return fit;
}

} // namespace datamunge::fda
