#pragma once

/// @file regression.hpp
/// @brief Ordinary least-squares regression with inference summaries, for dense designs
///        (Householder QR) and sparse designs (conjugate gradients on the normal equations).

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

/// @brief Fit and inference summary returned by linear_regression() and sparse_linear_regression().
/// @tparam T  Floating-point type.
template <typename T = double>
struct LinearRegressionResult {
    std::vector<T> coefficients;        ///< Least-squares coefficients \f$\hat\beta\f$.
    std::vector<T> fitted;              ///< Fitted values \f$X\hat\beta\f$.
    std::vector<T> residuals;           ///< Residuals \f$y-X\hat\beta\f$.
    DenseMatrix<T> covariance;          ///< Estimated covariance \f$\hat\sigma^2(X^\top X)^{-1}\f$ of the coefficients.
    std::vector<T> parameter_variances; ///< Diagonal of `covariance`.
    std::vector<T> standard_errors;     ///< Square roots of `parameter_variances`.
    T              residual_sum_of_squares{0};  ///< RSS \f$\sum_i r_i^2\f$.
    T              explained_sum_of_squares{0}; ///< ESS \f$\sum_i(\hat y_i-\bar y)^2\f$.
    T              total_sum_of_squares{0};     ///< TSS \f$\sum_i(y_i-\bar y)^2\f$.
    T              sigma2{0};           ///< Residual variance \f$\hat\sigma^2=\mathrm{RSS}/(n-p)\f$.
    T              sigma{0};            ///< Residual standard error \f$\hat\sigma\f$.
    T              r_squared{0};        ///< \f$R^2=1-\mathrm{RSS}/\mathrm{TSS}\f$.
    T              adjusted_r_squared{0}; ///< \f$1-(1-R^2)(n-1)/(n-p)\f$.
    std::size_t    observations{0};     ///< Number of rows n.
    std::size_t    predictors{0};       ///< Number of columns p (including any intercept column).
    std::size_t    degrees_of_freedom{0}; ///< Residual degrees of freedom n - p.
};

/// @brief Options for sparse_linear_regression().
struct SparseLinearRegressionOptions {
    SolverOptions solver{};          ///< Tolerance and iteration limits for CG/PCG.
    bool          use_jacobi{true};  ///< Precondition with the diagonal of \f$X^\top X\f$ (PCG) instead of plain CG.
};

/// @brief Internal helpers of the regression routines.
namespace regression_detail {

/// @brief Arithmetic mean of @p x.
/// @param x  Non-empty vector.
/// @return The mean.
template <typename T>
T mean(const std::vector<T>& x) {
    T sum = T{};
    for (const auto& v : x) sum += v;
    return sum / static_cast<T>(x.size());
}

/// @brief \f$y\leftarrow X^\top x\f$ for a COO matrix.
/// @param X  Sparse matrix.
/// @param x  Vector of length X.rows().
/// @param y  Output of length X.cols() (overwritten).
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

/// @brief The dense Gram matrix \f$X^\top X\f$ of a row-sorted (compressed) COO matrix.
/// @param X  Sparse matrix, entries grouped by row.
/// @return The p x p Gram matrix.
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

/// @brief The diagonal of \f$X^\top X\f$ (squared column norms).
/// @param X  Sparse matrix.
/// @return The p column sums of squares.
template <typename T>
std::vector<T> diagonal_of_gram(const SparseCOO<T>& X) {
    std::vector<T> diag(X.cols(), T{});
    const auto&    cols = X.col_indices();
    const auto&    vals = X.values();
    for (std::size_t k = 0; k < vals.size(); ++k)
        diag[cols[k]] += vals[k] * vals[k];
    return diag;
}

/// @brief Fills the sums of squares, \f$\hat\sigma\f$, \f$R^2\f$ and degrees of freedom from residuals and fitted values.
/// @param result           Result whose `residuals` and `fitted` are already set.
/// @param y                Response.
/// @param predictor_count  Number of columns p.
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

/// @brief Checks that n = y.size() and n > p.
/// @param rows     n.
/// @param cols     p.
/// @param y_size   Length of the response.
/// @param fn_name  Caller name for the error message.
/// @throws std::invalid_argument  If a check fails.
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

/// @brief A compressed copy of a COO matrix.
/// @param X  Matrix.
/// @return Compressed COO copy.
template <typename T>
SparseCOO<T> materialize_coo(const SparseCOO<T>& X) {
    SparseCOO<T> out = X;
    if (!out.is_compressed()) out.compress();
    return out;
}

/// @brief COO copy of a CSR matrix.
/// @param X  Matrix.
/// @return COO copy.
template <typename T>
SparseCOO<T> materialize_coo(const SparseCSR<T>& X) {
    return to_coo(X);
}

/// @brief COO copy of a CSC matrix.
/// @param X  Matrix.
/// @return COO copy.
template <typename T>
SparseCOO<T> materialize_coo(const SparseCSC<T>& X) {
    return to_coo(X);
}

/// @brief COO copy of an ELL matrix.
/// @param X  Matrix.
/// @return COO copy.
template <typename T>
SparseCOO<T> materialize_coo(const SparseELL<T>& X) {
    return to_coo(X);
}

/// @brief COO copy of a DIA matrix.
/// @param X  Matrix.
/// @return COO copy.
template <typename T>
SparseCOO<T> materialize_coo(const SparseDIA<T>& X) {
    return to_coo(X);
}

/// @brief COO copy of a format-erased SparseMatrix.
/// @param X  Matrix.
/// @return COO copy.
/// @throws std::runtime_error  For an unknown storage format.
template <typename T>
SparseCOO<T> materialize_coo(const SparseMatrix<T>& X) {
    if (const auto* coo = X.as_coo()) return materialize_coo(*coo);
    if (const auto* csr = X.as_csr()) return materialize_coo(*csr);
    if (const auto* csc = X.as_csc()) return materialize_coo(*csc);
    if (const auto* ell = X.as_ell()) return materialize_coo(*ell);
    if (const auto* dia = X.as_dia()) return materialize_coo(*dia);
    throw std::runtime_error("materialize_coo: unsupported sparse matrix format");
}

/// @brief Jacobi preconditioner \f$z=D^{-1}r\f$.
template <typename T>
struct DiagonalPreconditioner {
    std::vector<T> inv_diag; ///< Reciprocal diagonal entries.

    /// @brief Applies \f$z=D^{-1}r\f$.
    /// @param r  Residual.
    /// @param z  Output (resized).
    void apply(const std::vector<T>& r, std::vector<T>& z) const {
        z.resize(r.size());
        for (std::size_t i = 0; i < r.size(); ++i)
            z[i] = inv_diag[i] * r[i];
    }
};

/// @brief Matrix-free operator \f$v\mapsto X^\top Xv\f$ for CG on the normal equations.
template <typename T>
struct NormalEquationOperator {
    /// @brief Wraps (does not copy) @p X_.
    /// @param X_  Design matrix; must outlive the operator.
    explicit NormalEquationOperator(const SparseCOO<T>& X_) : X(X_) {}

    /// @brief Operator order p.
    /// @return X.cols().
    std::size_t rows() const { return X.cols(); }
    /// @brief Operator order p.
    /// @return X.cols().
    std::size_t cols() const { return X.cols(); }

    /// @brief \f$y\leftarrow X^\top Xx\f$.
    /// @param x  Input of length p.
    /// @param y  Output of length p.
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        std::vector<T> tmp(X.rows(), T{});
        X.spmv(x, tmp);
        transpose_spmv(X, tmp, y);
    }

    const SparseCOO<T>& X; ///< The wrapped design matrix.
};

} // namespace regression_detail

/// @brief Ordinary least squares on a dense design by Householder QR.
///
/// The coefficients solve \f$R\hat\beta=Q^\top y\f$, and the covariance is
/// \f$\hat\sigma^2R^{-1}R^{-\top}\f$, so \f$X^\top X\f$ (condition number \f$\kappa(X)^2\f$) is
/// never formed. Include an intercept column in @p X yourself if you want one.
///
/// @tparam T  Floating-point type.
/// @param X  n x p design matrix, n > p, full column rank.
/// @param y  Response of length n.
/// @return Coefficients, fit summary and covariance.
/// @throws std::invalid_argument  If sizes disagree or n <= p.
/// @throws std::runtime_error     If X is rank-deficient to working precision.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> X(4, 2, {1, 0, 1, 1, 1, 2, 1, 3});
/// const auto fit = dm::linalg::linear_regression(X, std::vector<double>{1, 3, 5, 8});
/// std::print("{:.2f} {:.2f}\n", fit.coefficients[0], fit.coefficients[1]);  // 0.80 2.30
/// @endcode
///
/// @see least_squares(), sparse_linear_regression()
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

    // (X^T X)^{-1} = R^{-1} R^{-T}: no X^T X, whose condition number is kappa(X)^2.
    const auto Ri = qr_dec.r_inverse();
    const std::size_t p = X.cols();
    regression_detail::fill_summary(result, y, p);
    result.covariance = DenseMatrix<T>(p, p, T{});
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = a; b < p; ++b) {
            T s = T{};
            for (std::size_t k = b; k < p; ++k) s += Ri(a, k) * Ri(b, k);
            result.covariance(a, b) = result.covariance(b, a) = s * result.sigma2;
        }
    result.parameter_variances.resize(X.cols());
    result.standard_errors.resize(X.cols());
    for (std::size_t j = 0; j < X.cols(); ++j) {
        result.parameter_variances[j] = result.covariance(j, j);
        result.standard_errors[j]     = std::sqrt(std::max(T{}, result.covariance(j, j)));
    }

    return result;
}

/// @brief Ordinary least squares on a sparse design by (preconditioned) conjugate gradients on
///        \f$X^\top X\beta=X^\top y\f$.
///
/// Never forms \f$X^\top X\f$ for the solve, only for the covariance at the end, so it suits
/// large sparse designs; the convergence rate depends on \f$\kappa(X)^2\f$.
///
/// @tparam SparseMatrixLike  Any of the sparse formats (COO, CSR, CSC, ELL, DIA, SparseMatrix).
/// @param X_in  n x p design matrix.
/// @param y     Response of length n.
/// @param opts  Solver options.
/// @return Coefficients, fit summary and covariance.
/// @throws std::invalid_argument  If sizes disagree or n <= p.
/// @throws std::runtime_error     If a column is zero, CG does not converge, or \f$X^\top X\f$ is not positive definite.
///
/// @see linear_regression()
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
