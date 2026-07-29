#pragma once

// The difference map (Elser, 2003): a general iterative scheme for constraint
// satisfaction -- find a point in the intersection of two sets A and B given only
// how to *project* onto each. It powers phase retrieval, protein folding,
// packing, and Sudoku solvers. Each iterate combines the two projections in a way
// that escapes the traps a naive alternating projection would fall into:
//
//   f_A(x) = (1 - 1/beta) P_A(x) + (1/beta) x
//   f_B(x) = (1 + 1/beta) P_B(x) - (1/beta) x
//   x <- x + beta ( P_A(f_B(x)) - P_B(f_A(x)) )
//
// At a fixed point the two projected estimates coincide and give a solution -- a
// point in A cap B. With beta = 1 the update reduces to the Douglas-Rachford
// iteration, which converges for convex A, B. The projections are supplied as
// callables, so the same driver solves any feasibility problem.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

struct DifferenceMapResult {
    std::vector<double> solution; // a point in (approximately) A cap B
    int                 iterations{0};
    double              residual{0}; // ||x_{n+1} - x_n||
    bool                converged{false};
};

// Run the difference map from x0 with projections projA, projB (each callable
// std::vector<double>(const std::vector<double>&)). Stops when the step size falls
// below `tol`.
template <class ProjA, class ProjB>
DifferenceMapResult difference_map(ProjA projA, ProjB projB, std::vector<double> x0,
                                   double beta = 1.0, int max_iter = 10000, double tol = 1e-12) {
    const std::size_t   n = x0.size();
    std::vector<double> x = std::move(x0);
    DifferenceMapResult res;

    for (int it = 0; it < max_iter; ++it) {
        const std::vector<double> pa = projA(x);
        const std::vector<double> pb = projB(x);
        std::vector<double>       fA(n), fB(n);
        for (std::size_t i = 0; i < n; ++i) {
            fA[i] = (1.0 - 1.0 / beta) * pa[i] + (1.0 / beta) * x[i];
            fB[i] = (1.0 + 1.0 / beta) * pb[i] - (1.0 / beta) * x[i];
        }
        const std::vector<double> pafB = projA(fB);
        const std::vector<double> pbfA = projB(fA);

        double step = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double delta = beta * (pafB[i] - pbfA[i]);
            x[i] += delta;
            step += delta * delta;
        }
        step             = std::sqrt(step);
        res.iterations   = it + 1;
        res.residual     = step;
        if (step < tol) { res.converged = true; break; }
    }
    // The solution estimate is the projection of the fixed point onto either set.
    res.solution = projB(x);
    return res;
}

} // namespace datamunge::algorithms
