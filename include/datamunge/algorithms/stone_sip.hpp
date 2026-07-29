#pragma once

// Stone's Strongly Implicit Procedure (SIP, 1968): an iterative solver for the
// sparse linear systems that arise from 5-point finite-difference stencils on a
// structured 2-D grid. Plain iterative methods (Jacobi, Gauss-Seidel) converge
// slowly because they couple only immediate neighbors. SIP instead builds an
// *incomplete* LU factorization M = LU whose triangular factors have the same
// sparsity as the original operator plus a small amount of controlled fill; the
// fill terms are partially cancelled with a parameter alpha so that M closely
// approximates A. Each iteration then solves M dx = r (a cheap forward/back
// sweep) and updates x, driving the residual down far faster than point methods.
//
// The operator is given by five coefficient arrays over the Nx-by-Ny grid
// (node p = j*Nx + i): Ap (center), Aw/Ae (west/east), As/An (south/north), with
// the coefficient set to 0 where a neighbor is outside the grid.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

struct StoneResult {
    std::vector<double> x;
    int                 iterations{0};
    double              residual{0}; // final L2 norm of b - A x
};

inline StoneResult stone_sip(int Nx, int Ny, const std::vector<double>& Ap,
                             const std::vector<double>& Aw, const std::vector<double>& Ae,
                             const std::vector<double>& As, const std::vector<double>& An,
                             const std::vector<double>& b, double alpha = 0.92,
                             int max_iter = 2000, double tol = 1e-10) {
    const int           M = Nx * Ny;
    std::vector<double> Lw(M, 0), Ls(M, 0), Lp(M, 0), Ue(M, 0), Un(M, 0);

    auto idx = [Nx](int i, int j) { return j * Nx + i; };

    // Incomplete LU factorization (the SIP recurrence).
    for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i) {
            const int p  = idx(i, j);
            const int w  = i > 0 ? idx(i - 1, j) : -1;
            const int s  = j > 0 ? idx(i, j - 1) : -1;
            const double UnW = w >= 0 ? Un[w] : 0.0, UeW = w >= 0 ? Ue[w] : 0.0;
            const double UeS = s >= 0 ? Ue[s] : 0.0, UnS = s >= 0 ? Un[s] : 0.0;

            Lw[p] = w >= 0 ? Aw[p] / (1.0 + alpha * UnW) : 0.0;
            Ls[p] = s >= 0 ? As[p] / (1.0 + alpha * UeS) : 0.0;
            const double p1 = alpha * Lw[p] * UnW;
            const double p2 = alpha * Ls[p] * UeS;
            Lp[p] = Ap[p] + p1 + p2 - Lw[p] * UeW - Ls[p] * UnS;
            Ue[p] = (Ae[p] - p1) / Lp[p];
            Un[p] = (An[p] - p2) / Lp[p];
        }

    std::vector<double> x(M, 0.0), r(M, 0.0), rho(M, 0.0), delta(M, 0.0);
    StoneResult         out;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Residual r = b - A x.
        double norm = 0.0;
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                const int p  = idx(i, j);
                double    ax = Ap[p] * x[p];
                if (i > 0)      ax += Aw[p] * x[idx(i - 1, j)];
                if (i < Nx - 1) ax += Ae[p] * x[idx(i + 1, j)];
                if (j > 0)      ax += As[p] * x[idx(i, j - 1)];
                if (j < Ny - 1) ax += An[p] * x[idx(i, j + 1)];
                r[p] = b[p] - ax;
                norm += r[p] * r[p];
            }
        norm = std::sqrt(norm);
        out.residual   = norm;
        out.iterations = iter;
        if (norm < tol) break;

        // Forward solve L rho = r.
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                const int p = idx(i, j);
                double    v = r[p];
                if (i > 0) v -= Lw[p] * rho[idx(i - 1, j)];
                if (j > 0) v -= Ls[p] * rho[idx(i, j - 1)];
                rho[p] = v / Lp[p];
            }
        // Back solve U delta = rho.
        for (int j = Ny - 1; j >= 0; --j)
            for (int i = Nx - 1; i >= 0; --i) {
                const int p = idx(i, j);
                double    v = rho[p];
                if (i < Nx - 1) v -= Ue[p] * delta[idx(i + 1, j)];
                if (j < Ny - 1) v -= Un[p] * delta[idx(i, j + 1)];
                delta[p] = v;
            }
        for (int p = 0; p < M; ++p) x[p] += delta[p];
    }
    out.x = std::move(x);
    return out;
}

} // namespace datamunge::algorithms
