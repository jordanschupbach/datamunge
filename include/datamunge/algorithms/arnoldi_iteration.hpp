#pragma once

// Arnoldi iteration: build an orthonormal basis of the Krylov subspace
// K_m(A, b) = span{b, A b, A^2 b, ..., A^{m-1} b} and, along the way, the
// projection of A onto it. Starting from v_1 = b/||b||, each step forms A v_j,
// orthogonalizes it against the previous basis vectors (modified Gram-Schmidt),
// and normalizes -- yielding an orthonormal V and an upper Hessenberg H with the
// Arnoldi relation
//
//     A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T,     i.e.  H_m = V_m^T A V_m.
//
// The eigenvalues of the small matrix H_m (the *Ritz values*) approximate the
// eigenvalues of A -- especially the extreme ones -- with excellent accuracy for
// m << n. It is the foundation of GMRES and of large-scale eigensolvers for
// nonsymmetric matrices.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

struct ArnoldiResult {
    std::vector<std::vector<double>> V; // basis vectors: V[k] is v_{k+1} (size up to m+1)
    std::vector<std::vector<double>> H; // (m+1) x m upper Hessenberg
};

// Run m steps of Arnoldi on A starting from b. Stops early (a shorter H) if an
// invariant subspace is found (a zero subdiagonal).
inline ArnoldiResult arnoldi_iteration(const std::vector<std::vector<double>>& A,
                                       const std::vector<double>& b, int m) {
    const int     n = static_cast<int>(A.size());
    ArnoldiResult res;
    auto matvec = [&](const std::vector<double>& x) {
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) y[i] += A[i][j] * x[j];
        return y;
    };
    auto norm = [](const std::vector<double>& x) { double s = 0; for (double v : x) s += v * v; return std::sqrt(s); };

    std::vector<double> v0 = b;
    const double        b0 = norm(v0);
    if (b0 < 1e-300) return res;
    for (double& x : v0) x /= b0;
    res.V.push_back(v0);
    res.H.assign(m + 1, std::vector<double>(m, 0.0));

    for (int j = 0; j < m; ++j) {
        std::vector<double> w = matvec(res.V[j]);
        for (int i = 0; i <= j; ++i) {
            double h = 0;
            for (int k = 0; k < n; ++k) h += res.V[i][k] * w[k];
            res.H[i][j] = h;
            for (int k = 0; k < n; ++k) w[k] -= h * res.V[i][k];
        }
        const double hn = norm(w);
        res.H[j + 1][j] = hn;
        if (hn < 1e-12) { res.H.resize(j + 2); for (auto& row : res.H) row.resize(m); break; }
        for (double& x : w) x /= hn;
        res.V.push_back(w);
    }
    return res;
}

} // namespace datamunge::algorithms
