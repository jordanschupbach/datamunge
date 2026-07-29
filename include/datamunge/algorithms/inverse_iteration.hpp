#pragma once

// Inverse iteration: find the eigenvector of a matrix A associated with a known
// approximate eigenvalue mu. The trick is that the matrix (A - mu I)^{-1} has the
// same eigenvectors as A, but with eigenvalues 1/(lambda_i - mu); the one closest
// to mu is hugely amplified. So repeatedly solving
//
//     (A - mu I) y = x_k,     x_{k+1} = y / ||y||,
//
// makes x converge rapidly to the eigenvector whose eigenvalue is nearest mu.
// Because A - mu I is nearly singular when mu is a good estimate, each solve
// amplifies the wanted component enormously -- which is exactly why it works. It
// is the standard way to recover an eigenvector once an eigenvalue is known.

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

using EigMatrix = std::vector<std::vector<double>>;

// Solve M x = b by Gaussian elimination with partial pivoting.
inline std::vector<double> eig_solve(EigMatrix M, std::vector<double> b) {
    const int n = static_cast<int>(b.size());
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
        std::swap(M[col], M[piv]);
        std::swap(b[col], b[piv]);
        double d = M[col][col];
        if (std::fabs(d) < 1e-300) d = 1e-300; // near-singular: regularize
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = M[r][col] / d;
            for (int c = col; c < n; ++c) M[r][c] -= f * M[col][c];
            b[r] -= f * b[col];
        }
    }
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = M[i][i] != 0 ? b[i] / M[i][i] : 0.0;
    return x;
}

inline std::vector<double> eig_matvec(const EigMatrix& A, const std::vector<double>& x) {
    const int n = static_cast<int>(A.size());
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) y[i] += A[i][j] * x[j];
    return y;
}

inline double eig_normalize(std::vector<double>& v) {
    double nrm = 0;
    for (double x : v) nrm += x * x;
    nrm = std::sqrt(nrm);
    if (nrm > 1e-300)
        for (double& x : v) x /= nrm;
    return nrm;
}

} // namespace detail

struct EigenPair {
    double              value{0};
    std::vector<double> vector;
};

// Eigenvector of `A` for the eigenvalue nearest `mu`, by inverse iteration.
// Returns the refined eigenvalue (Rayleigh quotient) and the unit eigenvector.
inline EigenPair inverse_iteration(const std::vector<std::vector<double>>& A, double mu,
                                   int iters = 50) {
    const int           n = static_cast<int>(A.size());
    std::vector<double> x(n, 1.0 / std::sqrt(static_cast<double>(n)));
    detail::EigMatrix   shifted = A;
    for (int i = 0; i < n; ++i) shifted[i][i] -= mu;

    for (int k = 0; k < iters; ++k) {
        std::vector<double> y = detail::eig_solve(shifted, x);
        if (detail::eig_normalize(y) < 1e-300) break;
        x = y;
    }
    // Rayleigh quotient x^T A x / x^T x (x is unit).
    const std::vector<double> Ax = detail::eig_matvec(A, x);
    double                    lam = 0;
    for (int i = 0; i < n; ++i) lam += x[i] * Ax[i];
    return {lam, x};
}

} // namespace datamunge::algorithms
