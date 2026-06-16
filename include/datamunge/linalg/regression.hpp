#pragma once

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/qr.hpp>
#include <datamunge/linalg/solvers.hpp>
#include <datamunge/linalg/sparse_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

template <typename T = double>
struct LinearRegressionResult {
    std::vector<T> coefficients;
    std::vector<T> fitted;
    std::vector<T> residuals;
    DenseMatrix<T> covariance;
    std::vector<T> parameter_variances;
    std::vector<T> standard_errors;
    T              residual_sum_of_squares{0};
    T              explained_sum_of_squares{0};
    T              total_sum_of_squares{0};
    T              sigma2{0};
    T              sigma{0};
    T              r_squared{0};
    T              adjusted_r_squared{0};
    std::size_t    observations{0};
    std::size_t    predictors{0};
    std::size_t    degrees_of_freedom{0};
};

struct SparseLinearRegressionOptions {
    SolverOptions solver{};
    bool          use_jacobi{true};
};

namespace regression_detail {

template <typename T>
T mean(const std::vector<T>& x) {
    T sum = T{};
    for (const auto& v : x) sum += v;
    return sum / static_cast<T>(x.size());
}

template <typename T>
void transpose_spmv(const SparseCOO<T>& X,
                    const std::vector<T>& x,
                    std::vector<T>&       y) {
    y.assign(X.cols(), T{});
    const auto& rows = X.row_indices();
    const auto& cols = X.col_indices();
    const auto& vals = X.values();
    for (std::size_t k = 0; k < vals.size(); ++k)
        y[cols[k]] += vals[k] * x[rows[k]];
}

template <typename T>
DenseMatrix<T> dense_gram(const SparseCOO<T>& X) {
    DenseMatrix<T> gram(X.cols(), X.cols(), T{});
    const auto&    rows = X.row_indices();
    const auto&    cols = X.col_indices();
    const auto&    vals = X.values();

    std::size_t first = 0;
    while (first < vals.size()) {
        const std::size_t row = rows[first];
        std::size_t       last = first;
        while (last < vals.size() && rows[last] == row) ++last;

        for (std::size_t a = first; a < last; ++a) {
            gram(cols[a], cols[a]) += vals[a] * vals[a];
            for (std::size_t b = a + 1; b < last; ++b) {
                const T prod = vals[a] * vals[b];
                gram(cols[a], cols[b]) += prod;
                gram(cols[b], cols[a]) += prod;
            }
        }

        first = last;
    }

    return gram;
}

template <typename T>
std::vector<T> diagonal_of_gram(const SparseCOO<T>& X) {
    std::vector<T> diag(X.cols(), T{});
    const auto&    cols = X.col_indices();
    const auto&    vals = X.values();
    for (std::size_t k = 0; k < vals.size(); ++k)
        diag[cols[k]] += vals[k] * vals[k];
    return diag;
}

template <typename T>
void fill_summary(LinearRegressionResult<T>& result,
                  const std::vector<T>&      y,
                  std::size_t                predictor_count) {
    const T y_bar = mean(y);
    T       rss   = T{};
    T       ess   = T{};
    T       tss   = T{};
    for (std::size_t i = 0; i < y.size(); ++i) {
        const T r = result.residuals[i];
        const T f = result.fitted[i] - y_bar;
        const T d = y[i] - y_bar;
        rss += r * r;
        ess += f * f;
        tss += d * d;
    }

    result.residual_sum_of_squares  = rss;
    result.explained_sum_of_squares = ess;
    result.total_sum_of_squares     = tss;
    result.observations             = y.size();
    result.predictors               = predictor_count;
    result.degrees_of_freedom       = y.size() - predictor_count;
    result.sigma2                   = rss / static_cast<T>(result.degrees_of_freedom);
    result.sigma                    = std::sqrt(result.sigma2);
    result.r_squared                = (tss == T{}) ? (rss == T{} ? T{1} : T{}) : (T{1} - rss / tss);

    if (y.size() > 1 && tss != T{})
        result.adjusted_r_squared =
            T{1} - (T{1} - result.r_squared)
                     * (static_cast<T>(y.size() - 1)
                        / static_cast<T>(result.degrees_of_freedom));
    else
        result.adjusted_r_squared = result.r_squared;
}

template <typename T>
void validate_regression_inputs(std::size_t rows,
                                std::size_t cols,
                                std::size_t y_size,
                                const char* fn_name) {
    if (rows != y_size)
        throw std::invalid_argument(std::string(fn_name) + ": X.rows() must equal y.size()");
    if (rows < cols)
        throw std::invalid_argument(std::string(fn_name) + ": require X.rows() >= X.cols()");
    if (rows == cols)
        throw std::invalid_argument(std::string(fn_name) + ": need X.rows() > X.cols() to estimate variance");
}

template <typename T>
SparseCOO<T> materialize_coo(const SparseCOO<T>& X) {
    SparseCOO<T> out = X;
    if (!out.is_compressed()) out.compress();
    return out;
}

template <typename T>
SparseCOO<T> materialize_coo(const SparseCSR<T>& X) {
    return to_coo(X);
}

template <typename T>
SparseCOO<T> materialize_coo(const SparseCSC<T>& X) {
    return to_coo(X);
}

template <typename T>
SparseCOO<T> materialize_coo(const SparseELL<T>& X) {
    return to_coo(X);
}

template <typename T>
SparseCOO<T> materialize_coo(const SparseDIA<T>& X) {
    return to_coo(X);
}

template <typename T>
SparseCOO<T> materialize_coo(const SparseMatrix<T>& X) {
    if (const auto* coo = X.as_coo()) return materialize_coo(*coo);
    if (const auto* csr = X.as_csr()) return materialize_coo(*csr);
    if (const auto* csc = X.as_csc()) return materialize_coo(*csc);
    if (const auto* ell = X.as_ell()) return materialize_coo(*ell);
    if (const auto* dia = X.as_dia()) return materialize_coo(*dia);
    throw std::runtime_error("materialize_coo: unsupported sparse matrix format");
}

template <typename T>
struct DiagonalPreconditioner {
    std::vector<T> inv_diag;

    void apply(const std::vector<T>& r, std::vector<T>& z) const {
        z.resize(r.size());
        for (std::size_t i = 0; i < r.size(); ++i)
            z[i] = inv_diag[i] * r[i];
    }
};

template <typename T>
struct NormalEquationOperator {
    explicit NormalEquationOperator(const SparseCOO<T>& X_) : X(X_) {}

    std::size_t rows() const { return X.cols(); }
    std::size_t cols() const { return X.cols(); }

    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        std::vector<T> tmp(X.rows(), T{});
        X.spmv(x, tmp);
        transpose_spmv(X, tmp, y);
    }

    const SparseCOO<T>& X;
};

} // namespace regression_detail

template <typename T>
LinearRegressionResult<T> linear_regression(const DenseMatrix<T>& X,
                                            const std::vector<T>& y) {
    regression_detail::validate_regression_inputs<T>(
        X.rows(), X.cols(), y.size(), "linear_regression");

    auto qr_dec = qr(X);
    if (!qr_dec.is_full_rank())
        throw std::runtime_error("linear_regression: design matrix is rank-deficient");

    LinearRegressionResult<T> result;
    result.coefficients = qr_dec.solve(y);
    result.fitted       = X * result.coefficients;
    result.residuals.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i)
        result.residuals[i] = y[i] - result.fitted[i];

    auto XtX = X.transpose() * X;
    auto chol = cholesky(XtX);
    if (!chol.ok)
        throw std::runtime_error("linear_regression: X^T X is not positive definite");

    regression_detail::fill_summary(result, y, X.cols());
    result.covariance = chol.solve(DenseMatrix<T>::identity(X.cols()));
    result.covariance *= result.sigma2;
    result.parameter_variances.resize(X.cols());
    result.standard_errors.resize(X.cols());
    for (std::size_t j = 0; j < X.cols(); ++j) {
        result.parameter_variances[j] = result.covariance(j, j);
        result.standard_errors[j]     = std::sqrt(std::max(T{}, result.covariance(j, j)));
    }

    return result;
}

template <typename SparseMatrixLike, typename T = typename SparseMatrixLike::value_type>
LinearRegressionResult<T> sparse_linear_regression(
    const SparseMatrixLike&          X_in,
    const std::vector<T>&            y,
    SparseLinearRegressionOptions    opts = {}) {
    auto X = regression_detail::materialize_coo(X_in);
    regression_detail::validate_regression_inputs<T>(
        X.rows(), X.cols(), y.size(), "sparse_linear_regression");

    regression_detail::NormalEquationOperator<T> normal_eq(X);
    std::vector<T> rhs;
    regression_detail::transpose_spmv(X, y, rhs);

    LinearRegressionResult<T> result;

    if (opts.use_jacobi) {
        auto diag = regression_detail::diagonal_of_gram(X);
        regression_detail::DiagonalPreconditioner<T> M;
        M.inv_diag.resize(diag.size());
        for (std::size_t j = 0; j < diag.size(); ++j) {
            if (diag[j] == T{})
                throw std::runtime_error(
                    "sparse_linear_regression: design matrix is rank-deficient");
            M.inv_diag[j] = T{1} / diag[j];
        }
        auto solve = pcg(normal_eq, rhs, M, {}, opts.solver);
        if (!solve.converged)
            throw std::runtime_error("sparse_linear_regression: PCG did not converge");
        result.coefficients = std::move(solve.x);
    } else {
        auto solve = cg(normal_eq, rhs, {}, opts.solver);
        if (!solve.converged)
            throw std::runtime_error("sparse_linear_regression: CG did not converge");
        result.coefficients = std::move(solve.x);
    }

    result.fitted = X.spmv(result.coefficients);
    result.residuals.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i)
        result.residuals[i] = y[i] - result.fitted[i];

    auto gram = regression_detail::dense_gram(X);
    auto chol = cholesky(gram);
    if (!chol.ok)
        throw std::runtime_error("sparse_linear_regression: X^T X is not positive definite");

    regression_detail::fill_summary(result, y, X.cols());
    result.covariance = chol.solve(DenseMatrix<T>::identity(X.cols()));
    result.covariance *= result.sigma2;
    result.parameter_variances.resize(X.cols());
    result.standard_errors.resize(X.cols());
    for (std::size_t j = 0; j < X.cols(); ++j) {
        result.parameter_variances[j] = result.covariance(j, j);
        result.standard_errors[j]     = std::sqrt(std::max(T{}, result.covariance(j, j)));
    }

    return result;
}

} // namespace datamunge::linalg
