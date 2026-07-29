#pragma once

// The simplex algorithm (Dantzig, 1947) for linear programming. A linear program
// maximizes a linear objective over a polytope defined by linear inequalities;
// its optimum, if finite, is attained at a vertex. The simplex method walks from
// vertex to adjacent vertex along improving edges of the polytope until no
// improving move remains. Here it is implemented as a tableau pivot for the
// standard inequality form
//
//     maximize   c . x   subject to   A x <= b,   x >= 0   (with b >= 0),
//
// so the slack variables form an initial feasible basis (no Phase I needed).
// Bland's rule chooses the pivots, which guarantees termination (no cycling).

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

enum class SimplexStatus { Optimal, Unbounded };

struct SimplexResult {
    SimplexStatus       status{SimplexStatus::Optimal};
    double              value{0};    // optimal objective
    std::vector<double> x;           // optimal decision variables (size n)
};

// Maximize c.x subject to A x <= b, x >= 0, with every b_i >= 0.
inline SimplexResult simplex_maximize(const std::vector<double>&              c,
                                      const std::vector<std::vector<double>>& A,
                                      const std::vector<double>&              b) {
    const std::size_t m = A.size();          // constraints
    const std::size_t n = c.size();          // structural variables
    const std::size_t total = n + m;         // + slacks
    const double      eps = 1e-9;

    // Tableau: m constraint rows + 1 objective row; columns = total vars + RHS.
    std::vector<std::vector<double>> T(m + 1, std::vector<double>(total + 1, 0.0));
    std::vector<std::size_t>         basis(m);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) T[i][j] = A[i][j];
        T[i][n + i]   = 1.0;      // slack
        T[i][total]   = b[i];     // RHS
        basis[i]      = n + i;
    }
    for (std::size_t j = 0; j < n; ++j) T[m][j] = -c[j]; // objective row (reduced costs)

    for (int iter = 0; iter < 100000; ++iter) {
        // Entering variable: smallest index with negative reduced cost (Bland).
        std::size_t entering = total;
        for (std::size_t j = 0; j < total; ++j)
            if (T[m][j] < -eps) { entering = j; break; }
        if (entering == total) break; // optimal

        // Ratio test: smallest RHS/col among positive column entries; Bland tie-break.
        std::size_t leaving = m;
        double      best_ratio = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            if (T[i][entering] > eps) {
                const double ratio = T[i][total] / T[i][entering];
                if (leaving == m || ratio < best_ratio - eps ||
                    (std::fabs(ratio - best_ratio) <= eps && basis[i] < basis[leaving])) {
                    best_ratio = ratio;
                    leaving    = i;
                }
            }
        }
        if (leaving == m) return {SimplexStatus::Unbounded, 0.0, {}};

        // Pivot on (leaving, entering).
        const double piv = T[leaving][entering];
        for (double& v : T[leaving]) v /= piv;
        for (std::size_t i = 0; i <= m; ++i) {
            if (i == leaving) continue;
            const double factor = T[i][entering];
            if (std::fabs(factor) < eps) continue;
            for (std::size_t j = 0; j <= total; ++j) T[i][j] -= factor * T[leaving][j];
        }
        basis[leaving] = entering;
    }

    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        if (basis[i] < n) x[basis[i]] = T[i][total];
    return {SimplexStatus::Optimal, T[m][total], x};
}

} // namespace datamunge::algorithms
