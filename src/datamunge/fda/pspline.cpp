#include <datamunge/fda/pspline.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/csr.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::fda {
namespace {

void validate_pspline_inputs(const std::vector<double>& x,
                             const std::vector<double>& y,
                             const BSplineBasis&        basis,
                             const PSplineOptions&      opts,
                             const char*                fn_name) {
    if (x.size() != y.size())
        throw std::invalid_argument(std::string(fn_name) + ": x.size() must equal y.size()");
    if (x.empty())
        throw std::invalid_argument(std::string(fn_name) + ": inputs must not be empty");
    if (opts.lambda < 0.0)
        throw std::invalid_argument(std::string(fn_name) + ": lambda must be nonnegative");
    if (opts.penalty_order == 0)
        throw std::invalid_argument(std::string(fn_name) + ": penalty order must be positive");
    if (basis.basis_size() <= opts.penalty_order)
        throw std::invalid_argument(
            std::string(fn_name) + ": basis size must exceed penalty order");
}

double mean(const std::vector<double>& y) {
    double sum = 0.0;
    for (double value : y)
        sum += value;
    return sum / static_cast<double>(y.size());
}

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
    return coeffs;
}

double trace_of_product(const linalg::DenseMatrix<double>& A,
                        const linalg::DenseMatrix<double>& B) {
    if (A.rows() != B.cols() || A.cols() != B.rows())
        throw std::invalid_argument("trace_of_product: shape mismatch");

    double trace = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t k = 0; k < A.cols(); ++k)
            trace += A(i, k) * B(k, i);
    return trace;
}

} // namespace

std::vector<double> PSplineFitResult::predict(const std::vector<double>& points) const {
    const auto csr = linalg::to_csr(basis.evaluate(points));
    return csr.spmv(coefficients);
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

    for (std::size_t i = 0; i < result.coefficients.size(); ++i)
        for (std::size_t j = 0; j < result.coefficients.size(); ++j)
            result.penalty += result.coefficients[i] * penalty(i, j) * result.coefficients[j];
    result.objective = result.residual_sum_of_squares + opts.lambda * result.penalty;

    if (result.total_sum_of_squares == 0.0)
        result.r_squared = (result.residual_sum_of_squares == 0.0) ? 1.0 : 0.0;
    else
        result.r_squared = 1.0 - result.residual_sum_of_squares / result.total_sum_of_squares;

    const auto system_inv = chol.solve(linalg::DenseMatrix<double>::identity(basis.basis_size()));
    result.effective_degrees_of_freedom = trace_of_product(xtx, system_inv);
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
    return fit_pspline(x, y, BSplineBasis::open_uniform(degree, basis_size,
                                                        *std::min_element(x.begin(), x.end()),
                                                        *std::max_element(x.begin(), x.end())),
                       opts);
}

} // namespace datamunge::fda
