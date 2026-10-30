#include <datamunge/fda/pspline.hpp>

/// @file pspline.cpp
/// @brief Difference-penalty assembly, Cholesky fitting and fixed-smoother diagnostics.

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/csr.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::fda {
/// @brief Internal validation and scalar algebra for P-spline fitting.
namespace {

/// @brief Validates paired finite data, basis domain/knots and finite penalty options.
/// @param x Training coordinates.
/// @param y Paired responses.
/// @param basis Supplied spline basis.
/// @param opts Smoothing multiplier and order.
/// @param fn_name Function name prepended to failures.
/// @throws std::invalid_argument For invalid shapes, values, domain or options.
void validate_pspline_inputs(const std::vector<double>& x,
                             const std::vector<double>& y,
                             const BSplineBasis&        basis,
                             const PSplineOptions&      opts,
                             const char*                fn_name) {
    if (x.size() != y.size())
        throw std::invalid_argument(std::string(fn_name) + ": x.size() must equal y.size()");
    if (x.empty())
        throw std::invalid_argument(std::string(fn_name) + ": inputs must not be empty");
    if (!std::isfinite(opts.lambda) || opts.lambda < 0.0)
        throw std::invalid_argument(std::string(fn_name) + ": lambda must be finite and nonnegative");
    if (opts.penalty_order == 0)
        throw std::invalid_argument(std::string(fn_name) + ": penalty order must be positive");
    if (basis.basis_size() <= opts.penalty_order)
        throw std::invalid_argument(
            std::string(fn_name) + ": basis size must exceed penalty order");
    for (double knot : basis.knots())
        if (!std::isfinite(knot))
            throw std::invalid_argument(std::string(fn_name) + ": knots must be finite");
    const auto [lo, hi] = basis.domain();
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || x[i] < lo || x[i] > hi)
            throw std::invalid_argument(std::string(fn_name) + ": finite data within the basis domain required");
}

/// @brief Arithmetic mean of a nonempty response vector.
/// @param y Nonempty finite responses.
/// @return Sum divided by length.
double mean(const std::vector<double>& y) {
    double sum = 0.0;
    for (double value : y)
        sum += value;
    return sum / static_cast<double>(y.size());
}

/// @brief Forms signed binomial weights by repeated first differences.
/// @param order Positive difference order.
/// @return order+1 finite weights.
/// @throws std::overflow_error If a weight overflows double.
std::vector<double> difference_coefficients(std::size_t order) {
    std::vector<double> coeffs = {1.0};
    for (std::size_t k = 0; k < order; ++k) {
        std::vector<double> next(coeffs.size() + 1, 0.0);
        for (std::size_t i = 0; i < coeffs.size(); ++i) {
            next[i] -= coeffs[i];
            next[i + 1] += coeffs[i];
        }
        coeffs = std::move(next);
    }
    for (double c : coeffs)
        if (!std::isfinite(c)) throw std::overflow_error("pspline: difference weights overflow");
    return coeffs;
}

} // namespace

std::vector<double> PSplineFitResult::predict(const std::vector<double>& points) const {
    if (coefficients.size() != basis.basis_size())
        throw std::invalid_argument("PSplineFitResult::predict: coefficient size mismatch");
    for (double x : points)
        if (!std::isfinite(x)) throw std::invalid_argument("PSplineFitResult::predict: points must be finite");
    const auto csr = linalg::to_csr(basis.evaluate(points));
    return csr.spmv(coefficients);
}

double PSplineFitResult::gcv() const noexcept {
    const double n = static_cast<double>(observations);
    const double df = n - effective_degrees_of_freedom;
    if (!(df > 64 * std::numeric_limits<double>::epsilon() * std::max(1.0,n)))
        return std::numeric_limits<double>::infinity();
    return n * residual_sum_of_squares / (df * df);
}

std::vector<double> PSplineFitResult::loo_residuals() const {
    if (leverage.size() != residuals.size())
        throw std::invalid_argument("PSplineFitResult::loo_residuals: leverage size mismatch");
    auto out = residuals;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double denominator = 1.0 - leverage[i];
        out[i] = denominator > 64 * std::numeric_limits<double>::epsilon()
            ? out[i] / denominator : std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

double PSplineFitResult::loo_cv() const {
    if (residuals.empty()) return std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (double r : loo_residuals()) {
        if (!std::isfinite(r)) return std::numeric_limits<double>::infinity();
        sum += r*r;
    }
    return sum / residuals.size();
}

linalg::DenseMatrix<double> pspline_difference_matrix(std::size_t coefficient_count,
                                                      std::size_t penalty_order) {
    if (penalty_order == 0 || coefficient_count <= penalty_order)
        throw std::invalid_argument("pspline_difference_matrix: require 0 < order < coefficient count");
    linalg::DenseMatrix<double> difference(coefficient_count-penalty_order,coefficient_count,0.0);
    const auto weights = difference_coefficients(penalty_order);
    for (std::size_t i = 0; i < difference.rows(); ++i)
        for (std::size_t j = 0; j < weights.size(); ++j) difference(i,i+j) = weights[j];
    return difference;
}

linalg::DenseMatrix<double> pspline_penalty_matrix(std::size_t coefficient_count,
                                                   std::size_t penalty_order) {
    if (penalty_order == 0)
        throw std::invalid_argument("pspline_penalty_matrix: penalty order must be positive");
    if (coefficient_count <= penalty_order)
        throw std::invalid_argument(
            "pspline_penalty_matrix: coefficient count must exceed penalty order");

    linalg::DenseMatrix<double> penalty(coefficient_count, coefficient_count, 0.0);
    const auto coeffs = difference_coefficients(penalty_order);

    for (std::size_t row = 0; row + penalty_order < coefficient_count; ++row) {
        for (std::size_t i = 0; i < coeffs.size(); ++i)
            for (std::size_t j = 0; j < coeffs.size(); ++j)
                penalty(row + i, row + j) += coeffs[i] * coeffs[j];
    }
    for (std::size_t i = 0; i < coefficient_count; ++i)
        for (std::size_t j = 0; j < coefficient_count; ++j)
            if (!std::isfinite(penalty(i,j))) throw std::overflow_error("pspline: penalty entries overflow");

    return penalty;
}

PSplineFitResult fit_pspline(const std::vector<double>& x,
                             const std::vector<double>& y,
                             const BSplineBasis&        basis,
                             PSplineOptions             opts) {
    validate_pspline_inputs(x, y, basis, opts, "fit_pspline");

    const auto design_coo = basis.evaluate(x);
    const auto design_csr = linalg::to_csr(design_coo);

    linalg::DenseMatrix<double> xtx(basis.basis_size(), basis.basis_size(), 0.0);
    std::vector<double>         xty(basis.basis_size(), 0.0);

    const auto& rows = design_coo.row_indices();
    const auto& cols = design_coo.col_indices();
    const auto& vals = design_coo.values();

    std::size_t first = 0;
    while (first < vals.size()) {
        const std::size_t row = rows[first];
        std::size_t       last = first;
        while (last < vals.size() && rows[last] == row) ++last;

        for (std::size_t a = first; a < last; ++a) {
            xty[cols[a]] += vals[a] * y[row];
            xtx(cols[a], cols[a]) += vals[a] * vals[a];
            for (std::size_t b = a + 1; b < last; ++b) {
                const double prod = vals[a] * vals[b];
                xtx(cols[a], cols[b]) += prod;
                xtx(cols[b], cols[a]) += prod;
            }
        }

        first = last;
    }

    auto penalty = pspline_penalty_matrix(basis.basis_size(), opts.penalty_order);
    auto system = xtx + opts.lambda * penalty;
    for (std::size_t i = 0; i < system.rows(); ++i) {
        if (!std::isfinite(xty[i])) throw std::runtime_error("fit_pspline: non-finite normal equations");
        for (std::size_t j = 0; j < system.cols(); ++j)
            if (!std::isfinite(system(i,j))) throw std::runtime_error("fit_pspline: non-finite normal equations");
    }
    auto chol = linalg::cholesky(system);
    if (!chol.ok)
        throw std::runtime_error("fit_pspline: penalized normal equations are not positive definite");

    PSplineFitResult result{basis};
    result.lambda = opts.lambda;
    result.penalty_order = opts.penalty_order;
    result.observations = y.size();
    result.penalty_matrix = penalty;
    result.coefficients = chol.solve(xty);
    result.fitted = design_csr.spmv(result.coefficients);
    result.residuals.resize(y.size());

    const double y_bar = mean(y);
    for (std::size_t i = 0; i < y.size(); ++i)
        result.residuals[i] = y[i] - result.fitted[i];

    for (std::size_t i = 0; i < result.residuals.size(); ++i) {
        const double residual = result.residuals[i];
        const double centered = y[i] - y_bar;
        result.residual_sum_of_squares += residual * residual;
        result.total_sum_of_squares += centered * centered;
    }

    // Squaring differences avoids cancellation in beta' P beta for almost-linear coefficients.
    const auto differences = pspline_difference_matrix(basis.basis_size(),opts.penalty_order)
                                 * result.coefficients;
    for (double value : differences) result.penalty += value * value;
    result.objective = result.residual_sum_of_squares + opts.lambda * result.penalty;

    if (result.total_sum_of_squares == 0.0)
        result.r_squared = (result.residual_sum_of_squares == 0.0) ? 1.0 : 0.0;
    else
        result.r_squared = 1.0 - result.residual_sum_of_squares / result.total_sum_of_squares;

    const auto system_inv = chol.solve(linalg::DenseMatrix<double>::identity(basis.basis_size()));
    result.leverage.assign(x.size(),0.0);
    first = 0;
    while (first < vals.size()) {
        const std::size_t row = rows[first];
        std::size_t last = first;
        while (last < vals.size() && rows[last] == row) ++last;
        for (std::size_t a = first; a < last; ++a)
            for (std::size_t b = first; b < last; ++b)
                result.leverage[row] += vals[a]*system_inv(cols[a],cols[b])*vals[b];
        result.effective_degrees_of_freedom += result.leverage[row];
        first = last;
    }
    result.residual_degrees_of_freedom =
        static_cast<double>(y.size()) - result.effective_degrees_of_freedom;

    if (result.residual_degrees_of_freedom > 0.0 && y.size() > 1 && result.total_sum_of_squares > 0.0)
        result.adjusted_r_squared =
            1.0 - (1.0 - result.r_squared)
                    * (static_cast<double>(y.size() - 1) / result.residual_degrees_of_freedom);
    else
        result.adjusted_r_squared = result.r_squared;

    return result;
}

PSplineFitResult fit_pspline(const std::vector<double>& x,
                             const std::vector<double>& y,
                             std::size_t                degree,
                             std::size_t                basis_size,
                             PSplineOptions             opts) {
    if (x.empty())
        throw std::invalid_argument("fit_pspline: inputs must not be empty");
    for (double value : x)
        if (!std::isfinite(value)) throw std::invalid_argument("fit_pspline: x must be finite");
    return fit_pspline(x, y, BSplineBasis::open_uniform(degree, basis_size,
                                                        *std::min_element(x.begin(), x.end()),
                                                        *std::max_element(x.begin(), x.end())),
                       opts);
}

} // namespace datamunge::fda
