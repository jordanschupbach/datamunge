#pragma once

// Lanczos iteration: the symmetric special case of Arnoldi. When A is symmetric,
// the projection V_m^T A V_m is not just upper Hessenberg but *tridiagonal*, so
// the full Gram-Schmidt collapses to a three-term recurrence
//
//     beta_j v_{j+1} = A v_j - alpha_j v_j - beta_{j-1} v_{j-1},
//     alpha_j = v_j^T A v_j,     beta_j = ||...||,
//
// building an orthonormal Krylov basis using only the two previous vectors. The
// tridiagonal matrix T_m with diagonal (alpha) and off-diagonal (beta) has
// eigenvalues -- the *Ritz values* -- that converge extremely fast to the extreme
// eigenvalues of A, which is why Lanczos is the method of choice for a few
// eigenvalues of a huge sparse symmetric matrix.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

struct LanczosResult {
    std::vector<double>              alpha; // T diagonal (size m)
    std::vector<double>              beta;  // T off-diagonal (size m-1 used)
    std::vector<std::vector<double>> V;     // Lanczos basis vectors
};

// Run up to m steps of Lanczos on symmetric A from start vector b.
inline LanczosResult lanczos_iteration(const std::vector<std::vector<double>>& A,
                                       const std::vector<double>& b, int m) {
    const int     n = static_cast<int>(A.size());
    LanczosResult res;
    auto matvec = [&](const std::vector<double>& x) {
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) y[i] += A[i][j] * x[j];
        return y;
    };
    auto dot  = [&](const std::vector<double>& a, const std::vector<double>& c) { double s = 0; for (int i = 0; i < n; ++i) s += a[i] * c[i]; return s; };
    auto norm = [&](const std::vector<double>& x) { return std::sqrt(dot(x, x)); };

    std::vector<double> v = b;
    const double        b0 = norm(v);
    if (b0 < 1e-300) return res;
    for (double& x : v) x /= b0;

    std::vector<double> v_prev(n, 0.0);
    double              beta_prev = 0.0;
    for (int j = 0; j < m; ++j) {
        res.V.push_back(v);
        std::vector<double> w = matvec(v);
        const double        a = dot(v, w);
        res.alpha.push_back(a);
        for (int i = 0; i < n; ++i) w[i] -= a * v[i] + beta_prev * v_prev[i];
        // Full reorthogonalization for numerical stability.
        for (const auto& q : res.V) { const double h = dot(q, w); for (int i = 0; i < n; ++i) w[i] -= h * q[i]; }
        const double bta = norm(w);
        if (j + 1 < m) res.beta.push_back(bta);
        if (bta < 1e-12) break; // invariant subspace
        v_prev    = v;
        beta_prev = bta;
        for (int i = 0; i < n; ++i) w[i] /= bta;
        v = w;
    }
    return res;
}

} // namespace datamunge::algorithms
