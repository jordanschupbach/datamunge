#pragma once

/// @file eigen.hpp
/// @brief Symmetric eigendecomposition \f$A=V\Lambda V^\top\f$: the cyclic Jacobi method for
///        small matrices, and Householder tridiagonalisation with implicit QL for larger ones.

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::linalg {

/// @brief Eigenvalues and eigenvectors returned by jacobi_eigen() and symmetric_eigen().
/// @tparam T  Floating-point type.
template <typename T = double>
struct SymmetricEigenResult {
    std::vector<T> eigenvalues;   ///< Eigenvalues in decreasing order.
    DenseMatrix<T> eigenvectors;  ///< Orthonormal eigenvectors as columns, in the order of `eigenvalues`.
};

/// @brief Cyclic Jacobi eigenvalue algorithm for a symmetric matrix.
///
/// Each step applies the plane rotation that zeroes one off-diagonal pair \f$(p,q)\f$; this
/// lowers \f$\mathrm{off}(A)^2=\sum_{i\ne j}a_{ij}^2\f$ by exactly \f$2a_{pq}^2\f$ while
/// keeping \f$\lVert A\rVert_F\f$ fixed, so sweeps over all pairs converge (quadratically in the
/// end). The iteration stops when \f$\mathrm{off}(A)\le\mathrm{tol}\cdot\lVert A\rVert_F\f$, a
/// test relative to the size of A, so scaling A by any constant scales the eigenvalues and
/// leaves the eigenvectors unchanged. Only the upper triangle of `A` is read (it is
/// symmetrized defensively). Simple and accurate for the small matrices (a handful of
/// predictors) that show up in statistics; each sweep costs \f$O(n^3)\f$.
///
/// @tparam T  Floating-point type.
/// @param A           Symmetric matrix (taken by value).
/// @param tol         Relative off-diagonal tolerance.
/// @param max_sweeps  Upper bound on the number of sweeps.
/// @return Eigenvalues in decreasing order and the matching orthonormal eigenvectors.
/// @throws std::invalid_argument  If A is not square.
///
/// @par Example
/// @code
/// const auto e = dm::linalg::jacobi_eigen(dm::linalg::DenseMatrix<double>(2, 2, {2, 1, 1, 2}));
/// std::print("{:.6f} {:.6f}\n", e.eigenvalues[0], e.eigenvalues[1]);  // 3.000000 1.000000
/// @endcode
///
/// @see svd()
template <typename T = double>
SymmetricEigenResult<T> jacobi_eigen(DenseMatrix<T> A, T tol = static_cast<T>(1e-12),
                                     std::size_t max_sweeps = 100) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("jacobi_eigen: matrix must be square");
    const std::size_t n = A.rows();

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            A(j, i) = A(i, j);

    DenseMatrix<T> V = DenseMatrix<T>::identity(n);
    const T        scale = A.norm_frobenius();  // invariant under the rotations

    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        T off = T{};
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                off += A(i, j) * A(i, j);
        if (std::sqrt(T{2} * off) <= tol * scale) break;

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                if (A(p, q) == T{} || std::abs(A(p, q)) < std::numeric_limits<T>::epsilon() * tol * scale) continue;

                const T theta = (A(q, q) - A(p, p)) / (T{2} * A(p, q));
                const T sign  = (theta >= T{}) ? T{1} : T{-1};
                const T t     = sign / (std::abs(theta) + std::sqrt(theta * theta + T{1}));
                const T c     = T{1} / std::sqrt(t * t + T{1});
                const T s     = t * c;

                const T app = A(p, p);
                const T aqq = A(q, q);

                A(p, p) = app - t * A(p, q);
                A(q, q) = aqq + t * A(p, q);
                A(p, q) = T{};
                A(q, p) = T{};

                for (std::size_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    const T aip = A(i, p);
                    const T aiq = A(i, q);
                    A(i, p) = c * aip - s * aiq;
                    A(p, i) = A(i, p);
                    A(i, q) = s * aip + c * aiq;
                    A(q, i) = A(i, q);
                }

                for (std::size_t i = 0; i < n; ++i) {
                    const T vip = V(i, p);
                    const T viq = V(i, q);
                    V(i, p) = c * vip - s * viq;
                    V(i, q) = s * vip + c * viq;
                }
            }
        }
    }

    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return A(a, a) > A(b, b); });

    SymmetricEigenResult<T> result;
    result.eigenvalues.resize(n);
    result.eigenvectors = DenseMatrix<T>(n, n, T{});
    for (std::size_t new_index = 0; new_index < n; ++new_index) {
        const std::size_t old_index   = order[new_index];
        result.eigenvalues[new_index] = A(old_index, old_index);
        for (std::size_t i = 0; i < n; ++i)
            result.eigenvectors(i, new_index) = V(i, old_index);
    }
    return result;
}

/// @brief Eigendecomposition of a symmetric matrix by Householder tridiagonalisation and the
///        implicit QL algorithm (EISPACK's `tred2` and `tql2`, the method behind LAPACK's `dsteqr`).
///
/// The matrix is first reduced to tridiagonal form \f$Q^\top AQ=T\f$ by \f$n-2\f$ Householder
/// reflections (\f$\tfrac43n^3\f$ flops), then QL steps with Wilkinson-type implicit shifts
/// diagonalise \f$T\f$ while the plane rotations are accumulated into \f$Q\f$ (about
/// \f$3n^3\f$ more flops in total; usually 1-2 QL steps per eigenvalue). The method is
/// backward stable: each computed eigenvalue is within a small multiple of
/// \f$\varepsilon\lVert A\rVert_2\f$ of an exact one, and an eigenvector's error is about
/// \f$\varepsilon\lVert A\rVert_2\f$ divided by the gap to the nearest other eigenvalue.
/// Unlike jacobi_eigen(), whose stopping test leaves off-diagonal entries of size
/// `tol` \f$\cdot\lVert A\rVert_F\f$, it resolves eigenvectors of eigenvalues that are tiny and
/// close together, as in the bottom of the spectrum of a graph Laplacian or of LLE's
/// \f$(I-W)^\top(I-W)\f$. Work is done in a column-major copy, so it handles a few thousand
/// rows. Only the upper triangle of @p A is read.
///
/// @tparam T  Floating-point type.
/// @param A  Symmetric matrix.
/// @return Eigenvalues in decreasing order and the matching orthonormal eigenvectors (columns).
/// @throws std::invalid_argument  If A is not square.
///
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> A(3, 3, {2, -1, 0, -1, 2, -1, 0, -1, 2});
/// const auto e = dm::linalg::symmetric_eigen(A);
/// std::print("{:.6f} {:.6f} {:.6f}\n", e.eigenvalues[0], e.eigenvalues[1], e.eigenvalues[2]);
/// // 3.414214 2.000000 0.585786
/// @endcode
///
/// @par References
/// - H. Bowdler, R. S. Martin, C. Reinsch and J. H. Wilkinson (1968). "The QR and QL algorithms
///   for symmetric matrices". Numerische Mathematik 11(4), 293-306. doi:10.1007/BF02166681
/// - R. S. Martin, C. Reinsch and J. H. Wilkinson (1968). "Householder's tridiagonalization of a
///   symmetric matrix". Numerische Mathematik 11(3), 181-195. doi:10.1007/BF02161841
/// - G. H. Golub and C. F. Van Loan (2013). *Matrix Computations*, 4th ed., section 8.3. Johns
///   Hopkins University Press.
///
/// @see jacobi_eigen(), svd()
template <typename T = double>
SymmetricEigenResult<T> symmetric_eigen(const DenseMatrix<T>& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("symmetric_eigen: matrix must be square");
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(A.rows());
    SymmetricEigenResult<T> result;
    if (n == 0) return result;

    // Column-major working copy: v(r, c) = buf[c * n + r], so every inner loop below runs down
    // a column and is contiguous.
    std::vector<T> buf(static_cast<std::size_t>(n * n));
    auto v = [&](std::ptrdiff_t r, std::ptrdiff_t c) -> T& { return buf[static_cast<std::size_t>(c * n + r)]; };
    for (std::ptrdiff_t i = 0; i < n; ++i)
        for (std::ptrdiff_t j = 0; j < n; ++j)
            v(i, j) = i <= j ? A(static_cast<std::size_t>(i), static_cast<std::size_t>(j))
                             : A(static_cast<std::size_t>(j), static_cast<std::size_t>(i));
    std::vector<T> d(static_cast<std::size_t>(n)), e(static_cast<std::size_t>(n));

    // tred2: Householder reduction to tridiagonal form (diagonal d, subdiagonal e).
    for (std::ptrdiff_t j = 0; j < n; ++j) d[j] = v(n - 1, j);
    for (std::ptrdiff_t i = n - 1; i > 0; --i) {
        T scale{}, h{};
        for (std::ptrdiff_t k = 0; k < i; ++k) scale += std::abs(d[k]);
        if (scale == T{}) {
            e[i] = d[i - 1];
            for (std::ptrdiff_t j = 0; j < i; ++j) {
                d[j]    = v(i - 1, j);
                v(i, j) = T{};
                v(j, i) = T{};
            }
        } else {
            for (std::ptrdiff_t k = 0; k < i; ++k) {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            T f = d[i - 1];
            T g = std::sqrt(h);
            if (f > T{}) g = -g;
            e[i]     = scale * g;
            h        = h - f * g;
            d[i - 1] = f - g;
            for (std::ptrdiff_t j = 0; j < i; ++j) e[j] = T{};
            for (std::ptrdiff_t j = 0; j < i; ++j) {
                f       = d[j];
                v(j, i) = f;
                g       = e[j] + v(j, j) * f;
                for (std::ptrdiff_t k = j + 1; k <= i - 1; ++k) {
                    g += v(k, j) * d[k];
                    e[k] += v(k, j) * f;
                }
                e[j] = g;
            }
            f = T{};
            for (std::ptrdiff_t j = 0; j < i; ++j) {
                e[j] /= h;
                f += e[j] * d[j];
            }
            const T hh = f / (h + h);
            for (std::ptrdiff_t j = 0; j < i; ++j) e[j] -= hh * d[j];
            for (std::ptrdiff_t j = 0; j < i; ++j) {
                f = d[j];
                g = e[j];
                for (std::ptrdiff_t k = j; k <= i - 1; ++k) v(k, j) -= (f * e[k] + g * d[k]);
                d[j]    = v(i - 1, j);
                v(i, j) = T{};
            }
        }
        d[i] = h;
    }
    // Accumulate the Householder reflections into v.
    for (std::ptrdiff_t i = 0; i < n - 1; ++i) {
        v(n - 1, i) = v(i, i);
        v(i, i)     = T{1};
        const T h   = d[i + 1];
        if (h != T{}) {
            for (std::ptrdiff_t k = 0; k <= i; ++k) d[k] = v(k, i + 1) / h;
            for (std::ptrdiff_t j = 0; j <= i; ++j) {
                T g{};
                for (std::ptrdiff_t k = 0; k <= i; ++k) g += v(k, i + 1) * v(k, j);
                for (std::ptrdiff_t k = 0; k <= i; ++k) v(k, j) -= g * d[k];
            }
        }
        for (std::ptrdiff_t k = 0; k <= i; ++k) v(k, i + 1) = T{};
    }
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        d[j]        = v(n - 1, j);
        v(n - 1, j) = T{};
    }
    v(n - 1, n - 1) = T{1};
    e[0]            = T{};

    // tql2: implicit QL iteration on the tridiagonal matrix, accumulating rotations.
    for (std::ptrdiff_t i = 1; i < n; ++i) e[i - 1] = e[i];
    e[n - 1]      = T{};
    T       f{}, tst1{};
    const T eps = std::numeric_limits<T>::epsilon();
    for (std::ptrdiff_t l = 0; l < n; ++l) {
        tst1             = std::max(tst1, std::abs(d[l]) + std::abs(e[l]));
        std::ptrdiff_t m = l;
        while (m < n - 1 && std::abs(e[m]) > eps * tst1) ++m;
        if (m > l) {
            int iterations = 0;
            do {
                if (++iterations > 60) throw std::runtime_error("symmetric_eigen: QL iteration did not converge");
                T g       = d[l];
                T p       = (d[l + 1] - g) / (T{2} * e[l]);
                T r       = std::hypot(p, T{1});
                if (p < T{}) r = -r;
                d[l]           = e[l] / (p + r);
                d[l + 1]       = e[l] * (p + r);
                const T dl1    = d[l + 1];
                T       h      = g - d[l];
                for (std::ptrdiff_t i = l + 2; i < n; ++i) d[i] -= h;
                f += h;
                p        = d[m];
                T c = T{1}, c2 = c, c3 = c;
                const T el1 = e[l + 1];
                T s{}, s2{};
                for (std::ptrdiff_t i = m - 1; i >= l; --i) {
                    c3       = c2;
                    c2       = c;
                    s2       = s;
                    g        = c * e[i];
                    h        = c * p;
                    r        = std::hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s        = e[i] / r;
                    c        = p / r;
                    p        = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    T* col_i  = &v(0, i);
                    T* col_i1 = &v(0, i + 1);
                    for (std::ptrdiff_t k = 0; k < n; ++k) {
                        const T hk = col_i1[k];
                        col_i1[k]  = s * col_i[k] + c * hk;
                        col_i[k]   = c * col_i[k] - s * hk;
                    }
                }
                p    = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
            } while (std::abs(e[l]) > eps * tst1);
        }
        d[l] += f;
        e[l] = T{};
    }

    std::vector<std::size_t> order(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return d[a] > d[b]; });
    const auto un = static_cast<std::size_t>(n);
    result.eigenvalues.resize(un);
    result.eigenvectors = DenseMatrix<T>(un, un, T{});
    for (std::size_t c = 0; c < un; ++c) {
        result.eigenvalues[c] = d[order[c]];
        const T* col          = &buf[order[c] * un];
        for (std::size_t r = 0; r < un; ++r) result.eigenvectors(r, c) = col[r];
    }
    return result;
}

} // namespace datamunge::linalg
