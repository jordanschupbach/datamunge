#pragma once

/// @file least_squares.hpp
/// @brief Dense linear least squares \f$\min_x\lVert Xx-y\rVert_2\f$ by the three standard
///        direct methods (normal equations, Householder QR, SVD), plus error measures.
///
/// The methods differ in how rounding errors are amplified. With
/// \f$\kappa=\kappa_2(X)\f$ and unit roundoff \f$u\f$, QR and SVD are backward stable, so the
/// relative error of \f$\hat x\f$ is about \f$u\kappa + u\kappa^2\tan\theta\f$ (\f$\theta\f$ the
/// angle between \f$y\f$ and the column space). The normal equations form
/// \f$X^\top X\f$, whose condition number is \f$\kappa^2\f$, and lose about
/// \f$u\kappa^2\f$ even when the residual is zero.

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/qr.hpp>
#include <datamunge/linalg/svd.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace datamunge::linalg {

/// @brief Algorithm used by least_squares().
enum class LeastSquaresMethod {
    NormalEquations, ///< Cholesky factorization of \f$X^\top X\f$: about \f$np^2+p^3/3\f$ flops, error grows like \f$\kappa^2\f$.
    HouseholderQR,   ///< Householder QR of \f$X\f$: about \f$2np^2-2p^3/3\f$ flops (plus forming Q), backward stable.
    SVD,             ///< One-sided Jacobi SVD: slowest, backward stable, handles rank deficiency (minimum-norm solution).
};

/// @brief Short display name of a LeastSquaresMethod.
/// @param method  The method.
/// @return `"normal equations"`, `"Householder QR"` or `"SVD"`.
inline std::string_view to_string(LeastSquaresMethod method) {
    switch (method) {
    case LeastSquaresMethod::NormalEquations: return "normal equations";
    case LeastSquaresMethod::HouseholderQR: return "Householder QR";
    case LeastSquaresMethod::SVD: return "SVD";
    }
    return "?";
}

/// @brief Result of least_squares().
/// @tparam T  Floating-point type.
template <typename T = double>
struct LeastSquaresResult {
    std::vector<T>     coefficients;  ///< The solution \f$\hat x\f$ (length p).
    std::vector<T>     residuals;     ///< \f$y-X\hat x\f$ (length n).
    T                  residual_norm{}; ///< \f$\lVert y-X\hat x\rVert_2\f$.
    std::size_t        rank{0};       ///< Numerical rank detected by the method (p for the normal equations).
    LeastSquaresMethod method{LeastSquaresMethod::HouseholderQR}; ///< The method that produced the result.
};

/// @brief The Gram matrix \f$X^\top X\f$ (p x p), computed with symmetric accumulation.
/// @tparam T  Floating-point type.
/// @param X  An n x p matrix.
/// @return \f$X^\top X\f$, exactly symmetric.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> X(2, 2, {1, 2, 3, 4});
/// std::print("{}\n", dm::linalg::gram(X)(0, 1));  // 14
/// @endcode
template <typename T>
DenseMatrix<T> gram(const DenseMatrix<T>& X) {
    const std::size_t n = X.rows(), p = X.cols();
    DenseMatrix<T>    G(p, p, T{});
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < p; ++a) {
            const T xa = X(i, a);
            for (std::size_t b = a; b < p; ++b) G(a, b) += xa * X(i, b);
        }
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < a; ++b) G(a, b) = G(b, a);
    return G;
}

/// @brief Solves \f$\min_x\lVert Xx-y\rVert_2\f$ with the chosen direct method.
///
/// - NormalEquations: \f$X^\top X\,x = X^\top y\f$ by Cholesky.
/// - HouseholderQR: \f$X=QR\f$, then \f$Rx=Q^\top y\f$ by back substitution.
/// - SVD: \f$x=V\Sigma^{+}U^\top y\f$, the minimum-norm minimizer (singular values below
///   \f$\max(n,p)\,\varepsilon\,s_1\f$ are dropped).
///
/// @tparam T  Floating-point type.
/// @param X       An n x p design matrix with \f$n\ge p\f$.
/// @param y       Response of length n.
/// @param method  Algorithm (default Householder QR).
/// @return Coefficients, residuals, residual norm and detected rank.
/// @throws std::invalid_argument  If the sizes disagree or \f$n<p\f$.
/// @throws std::runtime_error     If the normal-equations matrix is not numerically positive
///         definite, or QR finds \f$X\f$ rank-deficient (use the SVD method then).
///
/// @par Example
/// @code
/// // Fit y = 1 + 2t through three exact points.
/// const dm::linalg::DenseMatrix<double> X(3, 2, {1, 0, 1, 1, 1, 2});
/// const auto fit = dm::linalg::least_squares(X, std::vector<double>{1, 3, 5});
/// std::print("{:.3f} {:.3f}\n", fit.coefficients[0], fit.coefficients[1]);  // 1.000 2.000
/// @endcode
///
/// @see svd(), qr(), cholesky(), gram()
template <typename T>
LeastSquaresResult<T> least_squares(const DenseMatrix<T>& X, const std::vector<T>& y,
                                    LeastSquaresMethod method = LeastSquaresMethod::HouseholderQR) {
    const std::size_t n = X.rows(), p = X.cols();
    if (y.size() != n) throw std::invalid_argument("least_squares: y.size() != X.rows()");
    if (n < p) throw std::invalid_argument("least_squares: X has fewer rows than columns");

    LeastSquaresResult<T> r;
    r.method = method;
    switch (method) {
    case LeastSquaresMethod::NormalEquations: {
        const auto chol = cholesky(gram(X));
        if (!chol.ok) throw std::runtime_error("least_squares: X^T X is not numerically positive definite");
        std::vector<T> rhs(p, T{});
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < p; ++j) rhs[j] += X(i, j) * y[i];
        r.coefficients = chol.solve(rhs);
        r.rank         = p;
        break;
    }
    case LeastSquaresMethod::HouseholderQR: {
        const auto f = qr(X);
        if (!f.is_full_rank()) throw std::runtime_error("least_squares: X is rank-deficient; use LeastSquaresMethod::SVD");
        r.coefficients = f.solve(y);
        r.rank         = p;
        break;
    }
    case LeastSquaresMethod::SVD: {
        const auto f   = svd(X);
        r.coefficients = f.solve(y);
        r.rank         = f.rank();
        break;
    }
    }
    r.residuals = y;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) r.residuals[i] -= X(i, j) * r.coefficients[j];
    T ss = T{};
    for (const T v : r.residuals) ss += v * v;
    r.residual_norm = std::sqrt(ss);
    return r;
}

/// @brief Relative error \f$\lVert x-x_\star\rVert_2/\lVert x_\star\rVert_2\f$ of `x` against a reference.
/// @tparam T  Floating-point type.
/// @param x          Approximation.
/// @param reference  Reference value \f$x_\star\f$ (same length, not all zero).
/// @return The relative 2-norm error.
/// @throws std::invalid_argument  If the lengths differ.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::linalg::relative_error(std::vector<double>{3, 4.5}, std::vector<double>{3, 4}));  // 0.1
/// @endcode
template <typename T>
T relative_error(const std::vector<T>& x, const std::vector<T>& reference) {
    if (x.size() != reference.size()) throw std::invalid_argument("relative_error: length mismatch");
    T num = T{}, den = T{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        num += (x[i] - reference[i]) * (x[i] - reference[i]);
        den += reference[i] * reference[i];
    }
    return std::sqrt(num / den);
}

/// @brief Normwise backward error of an approximate solution of \f$Ax=b\f$.
///
/// The Rigal-Gaches formula
/// \f$\eta(\hat x)=\lVert b-A\hat x\rVert/(\lVert A\rVert\,\lVert\hat x\rVert+\lVert b\rVert)\f$
/// (Frobenius norm for \f$A\f$, 2-norm for vectors) is the smallest \f$\epsilon\f$ for which
/// \f$\hat x\f$ solves a system \f$(A+\Delta A)\hat x=b+\Delta b\f$ with
/// \f$\lVert\Delta A\rVert\le\epsilon\lVert A\rVert\f$ and \f$\lVert\Delta b\rVert\le\epsilon\lVert b\rVert\f$.
/// A backward-stable solver gives \f$\eta\f$ of order the unit roundoff.
///
/// @tparam T  Floating-point type.
/// @param A  Square (or any) matrix.
/// @param x  Approximate solution (length `A.cols()`).
/// @param b  Right-hand side (length `A.rows()`).
/// @return \f$\eta(\hat x)\f$.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(2, 2, {2, 0, 0, 2});
/// std::print("{}\n", dm::linalg::backward_error(A, std::vector<double>{1, 1}, std::vector<double>{2, 2}));  // 0
/// @endcode
template <typename T>
T backward_error(const DenseMatrix<T>& A, const std::vector<T>& x, const std::vector<T>& b) {
    const auto Ax = A * x;
    T          r = T{}, nx = T{}, nb = T{};
    for (std::size_t i = 0; i < b.size(); ++i) {
        r += (b[i] - Ax[i]) * (b[i] - Ax[i]);
        nb += b[i] * b[i];
    }
    for (const T v : x) nx += v * v;
    return std::sqrt(r) / (A.norm_frobenius() * std::sqrt(nx) + std::sqrt(nb));
}

/// @brief Converts every element of a matrix to another type, e.g. to `long double` for a
///        reference solution in extended precision.
/// @tparam U  Target element type.
/// @tparam T  Source element type.
/// @param A  Matrix to convert.
/// @return The converted matrix (same shape).
///
/// @par Example
/// @code
/// const auto L = dm::linalg::matrix_cast<long double>(dm::linalg::DenseMatrix<double>(1, 2, {0.5, 2}));
/// std::print("{}\n", static_cast<double>(L(0, 1)));  // 2
/// @endcode
template <typename U, typename T>
DenseMatrix<U> matrix_cast(const DenseMatrix<T>& A) {
    return DenseMatrix<U>(A.rows(), A.cols(), std::vector<U>(A.data().begin(), A.data().end()));
}

/// @brief The polynomial design matrix with columns \f$1, x, x^2, \dots, x^{d}\f$ (raw monomials).
///
/// Raw monomials on a positive variable are nearly collinear, so \f$\kappa\f$ grows roughly
/// geometrically with the degree; centring and scaling \f$x\f$ first, or orthogonalizing the
/// columns (QR), cures most of it.
///
/// @tparam T  Floating-point type.
/// @param x       The n sample points.
/// @param degree  Highest power d.
/// @return The n x (d+1) Vandermonde matrix.
///
/// @par Example
/// @code
/// const auto V = dm::linalg::vandermonde(std::vector<double>{2, 3}, 2);
/// std::print("{} {} {}\n", V(1, 0), V(1, 1), V(1, 2));  // 1 3 9
/// @endcode
template <typename T>
DenseMatrix<T> vandermonde(const std::vector<T>& x, std::size_t degree) {
    DenseMatrix<T> V(x.size(), degree + 1, T{1});
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = 1; j <= degree; ++j) V(i, j) = V(i, j - 1) * x[i];
    return V;
}

/// @brief Scales every column to unit Euclidean norm: \f$XD^{-1}\f$ with \f$D=\mathrm{diag}(\lVert x_j\rVert)\f$.
///
/// The condition number of the result is within a factor \f$\sqrt p\f$ of the smallest
/// achievable by any column scaling (van der Sluis), which is the part of \f$\kappa\f$ that
/// actually limits the accuracy of Cholesky, QR and SVD least squares.
///
/// @tparam T  Floating-point type.
/// @param X  Matrix (zero columns are left unchanged).
/// @return The column-equilibrated copy.
///
/// @par Example
/// @code
/// const auto E = dm::linalg::equilibrate_columns(dm::linalg::DenseMatrix<double>(2, 1, {3, 4}));
/// std::print("{} {}\n", E(0, 0), E(1, 0));  // 0.6 0.8
/// @endcode
template <typename T>
DenseMatrix<T> equilibrate_columns(DenseMatrix<T> X) {
    for (std::size_t j = 0; j < X.cols(); ++j) {
        T norm = T{};
        for (std::size_t i = 0; i < X.rows(); ++i) norm += X(i, j) * X(i, j);
        norm = std::sqrt(norm);
        if (norm == T{}) continue;
        for (std::size_t i = 0; i < X.rows(); ++i) X(i, j) /= norm;
    }
    return X;
}

/// @brief Centres every column at mean 0 and scales it to standard deviation 1 (denominator n-1),
///        as R's `scale()`.
/// @tparam T  Floating-point type.
/// @param X  Matrix with at least two rows (constant columns become 0).
/// @return The standardized copy \f$Z\f$.
///
/// @par Example
/// @code
/// const auto Z = dm::linalg::standardize_columns(dm::linalg::DenseMatrix<double>(3, 1, {1, 2, 3}));
/// std::print("{} {} {}\n", Z(0, 0), Z(1, 0), Z(2, 0));  // -1 0 1
/// @endcode
template <typename T>
DenseMatrix<T> standardize_columns(DenseMatrix<T> X) {
    const std::size_t n = X.rows();
    for (std::size_t j = 0; j < X.cols(); ++j) {
        T mean = T{};
        for (std::size_t i = 0; i < n; ++i) mean += X(i, j);
        mean /= static_cast<T>(n);
        T ss = T{};
        for (std::size_t i = 0; i < n; ++i) ss += (X(i, j) - mean) * (X(i, j) - mean);
        const T sd = std::sqrt(ss / static_cast<T>(n - 1));
        for (std::size_t i = 0; i < n; ++i) X(i, j) = sd > T{} ? (X(i, j) - mean) / sd : T{};
    }
    return X;
}

/// @brief Pearson correlation matrix of the columns, \f$Z^\top Z/(n-1)\f$ with Z = standardize_columns(X).
/// @tparam T  Floating-point type.
/// @param X  n x p data matrix (n >= 2).
/// @return The p x p correlation matrix.
///
/// @par Example
/// @code
/// const auto R = dm::linalg::correlation_matrix(dm::linalg::DenseMatrix<double>(3, 2, {1, 3, 2, 2, 3, 1}));
/// std::print("{}\n", R(0, 1));  // -1
/// @endcode
template <typename T>
DenseMatrix<T> correlation_matrix(const DenseMatrix<T>& X) {
    auto R = gram(standardize_columns(X));
    R *= T{1} / static_cast<T>(X.rows() - 1);
    return R;
}

/// @brief Accuracy of the three least-squares methods on one problem, from least_squares_accuracy().
struct LeastSquaresAccuracy {
    double              condition_number{};             ///< \f$\kappa_2(X)\f$.
    double              equilibrated_condition_number{}; ///< \f$\kappa_2(XD^{-1})\f$ with D the column norms (the scaling-free part of \f$\kappa\f$).
    double              tan_theta{};                     ///< \f$\lVert r\rVert/\lVert X\hat x\rVert\f$: how far y is from the column space.
    std::vector<double> reference;                       ///< The solution in `long double` (one-sided Jacobi SVD), rounded to double.
    double              normal_equations_error{};        ///< Relative error of LeastSquaresMethod::NormalEquations (NaN if Cholesky broke down).
    double              qr_error{};                      ///< Relative error of LeastSquaresMethod::HouseholderQR (NaN if QR declared X rank-deficient).
    double              svd_error{};                     ///< Relative error of LeastSquaresMethod::SVD.
};

/// @brief Solves \f$\min\lVert Xx-y\rVert\f$ by each method in double precision and measures the
///        relative error against a `long double` SVD solution of the same (exactly converted) data.
///
/// Since the reference sees the identical inputs, the errors are those of the algorithms alone.
/// Compare them with \f$u\kappa\f$ and \f$u\kappa^2\f$, \f$u=2^{-53}\f$; use
/// `equilibrated_condition_number`, because column scaling does not hurt any of the methods.
/// The reference is reliable while \f$\kappa\lesssim 10^{17}\f$ (on x86 `long double` has
/// \f$u\approx 5\cdot 10^{-20}\f$; where `long double` is plain double the errors read as zero).
///
/// @param X  n x p design matrix (full column rank).
/// @param y  Response of length n.
/// @return Condition numbers, \f$\tan\theta\f$, the reference solution and each method's error.
///
/// @par Example
/// @code
/// const auto X = dm::linalg::vandermonde(std::vector<double>{1, 2, 3, 4, 5, 6}, 3);
/// const auto a = dm::linalg::least_squares_accuracy(X, std::vector<double>{1, 0, 2, 1, 3, 2});
/// std::print("{}\n", a.qr_error < 1e-12);  // true
/// @endcode
///
/// @see least_squares(), svd()
inline LeastSquaresAccuracy least_squares_accuracy(const DenseMatrix<double>& X, const std::vector<double>& y) {
    LeastSquaresAccuracy a;
    const auto           XL  = matrix_cast<long double>(X);
    const auto           ref = svd(XL).solve(std::vector<long double>(y.begin(), y.end()), 1e-30L);
    a.reference.assign(ref.begin(), ref.end());
    const auto  fitted = XL * ref;
    long double rr = 0, ff = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        rr += (y[i] - fitted[i]) * (y[i] - fitted[i]);
        ff += fitted[i] * fitted[i];
    }
    a.tan_theta        = static_cast<double>(std::sqrt(rr / ff));
    a.condition_number = svd(X).condition_number();
    a.equilibrated_condition_number = svd(equilibrate_columns(X)).condition_number();
    const auto error = [&](LeastSquaresMethod m) {
        try {
            return relative_error(least_squares(X, y, m).coefficients, a.reference);
        } catch (const std::runtime_error&) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };
    a.normal_equations_error = error(LeastSquaresMethod::NormalEquations);
    a.qr_error               = error(LeastSquaresMethod::HouseholderQR);
    a.svd_error              = error(LeastSquaresMethod::SVD);
    return a;
}

} // namespace datamunge::linalg
