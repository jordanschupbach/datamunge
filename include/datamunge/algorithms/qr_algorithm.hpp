#pragma once

// The QR algorithm (Francis & Kublanovskaya, 1961): compute all eigenvalues of a
// symmetric matrix by repeatedly factoring it as A = QR (orthogonal times upper
// triangular) and reforming A <- RQ. The reversed product RQ = Q^T A Q is
// orthogonally similar to A, so it has the same eigenvalues, and the iteration
// drives the off-diagonal entries to zero -- the diagonal converging to the
// eigenvalues. A Wilkinson shift (chosen from the trailing 2x2 block) accelerates
// this to cubic convergence per eigenvalue, and once the last off-diagonal entry
// is negligible that eigenvalue is *deflated* off and the iteration continues on
// the smaller leading block. It is the standard dense eigenvalue method.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

using QrMatrix = std::vector<std::vector<double>>;

// Classical Gram-Schmidt QR factorization of the leading m x m block of A.
inline void qr_factor(const QrMatrix& A, int m, QrMatrix& Q, QrMatrix& R) {
    Q.assign(m, std::vector<double>(m, 0.0));
    R.assign(m, std::vector<double>(m, 0.0));
    for (int j = 0; j < m; ++j) {
        std::vector<double> v(m);
        for (int i = 0; i < m; ++i) v[i] = A[i][j];
        for (int k = 0; k < j; ++k) {
            double dot = 0;
            for (int i = 0; i < m; ++i) dot += Q[i][k] * A[i][j];
            R[k][j] = dot;
            for (int i = 0; i < m; ++i) v[i] -= dot * Q[i][k];
        }
        double norm = 0;
        for (int i = 0; i < m; ++i) norm += v[i] * v[i];
        norm = std::sqrt(norm);
        R[j][j] = norm;
        if (norm > 1e-300)
            for (int i = 0; i < m; ++i) Q[i][j] = v[i] / norm;
    }
}

} // namespace detail

// Eigenvalues of a symmetric matrix `A` (as a vector of rows) by the shifted QR
// algorithm with deflation. Returned in ascending order.
inline std::vector<double> qr_eigenvalues(std::vector<std::vector<double>> A, int max_iter = 1000,
                                          double tol = 1e-12) {
    const int           n = static_cast<int>(A.size());
    std::vector<double> eig;
    detail::QrMatrix    Q, R;

    for (int m = n; m >= 1; --m) {
        if (m == 1) { eig.push_back(A[0][0]); break; }
        int iter = 0;
        while (iter++ < max_iter) {
            // Wilkinson shift from the trailing 2x2 of the active m x m block.
            const double a = A[m - 2][m - 2], b = A[m - 2][m - 1], c = A[m - 1][m - 1];
            const double delta = (a - c) / 2.0;
            const double sign  = delta >= 0 ? 1.0 : -1.0;
            double       mu    = c;
            const double denom = std::fabs(delta) + std::sqrt(delta * delta + b * b);
            if (denom > 1e-300) mu = c - sign * b * b / denom;

            for (int i = 0; i < m; ++i) A[i][i] -= mu;         // shift
            detail::qr_factor(A, m, Q, R);
            // A_block <- R Q  (m x m)
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j) {
                    double s = 0;
                    for (int k = 0; k < m; ++k) s += R[i][k] * Q[k][j];
                    A[i][j] = s;
                }
            for (int i = 0; i < m; ++i) A[i][i] += mu;         // unshift

            if (std::fabs(A[m - 1][m - 2]) < tol * (std::fabs(A[m - 2][m - 2]) + std::fabs(A[m - 1][m - 1]) + 1e-300))
                break; // last eigenvalue converged -> deflate
        }
        eig.push_back(A[m - 1][m - 1]);
    }

    std::sort(eig.begin(), eig.end());
    return eig;
}

} // namespace datamunge::algorithms
