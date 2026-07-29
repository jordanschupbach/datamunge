#pragma once

// The Frank-Wolfe algorithm (conditional gradient, 1956) for minimizing a smooth
// convex function over a compact convex set C. Unlike projected gradient it never
// projects; it only needs a *linear minimization oracle* (LMO) that returns a
// vertex of C minimizing the linearized objective. Each step moves toward that
// vertex:
//
//   s_k = argmin_{s in C} <grad f(x_k), s>          (LMO)
//   x_{k+1} = (1 - gamma_k) x_k + gamma_k s_k,   gamma_k = 2/(k+2).
//
// Because every iterate is a convex combination of vertices, the iterates stay
// feasible and (for structured C like the simplex or an L1 ball) sparse. The
// objective gap f(x_k) - f* shrinks at the O(1/k) rate. This is the workhorse
// behind projection-free optimization over polytopes.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

struct FrankWolfeResult {
    std::vector<double> x;         // final iterate
    std::vector<double> objective; // f(x_k) recorded before each step (size iters+1)
};

// Minimize f over C. `grad(x)` returns the gradient; `lmo(g)` returns a vertex of
// C minimizing <g, s>; `x0` is a feasible starting point.
template <class F, class Grad, class Lmo>
FrankWolfeResult frank_wolfe(F f, Grad grad, Lmo lmo, std::vector<double> x0, int iters) {
    FrankWolfeResult res;
    res.x = std::move(x0);
    for (int k = 0; k < iters; ++k) {
        res.objective.push_back(f(res.x));
        const std::vector<double> g = grad(res.x);
        const std::vector<double> s = lmo(g);
        const double              gamma = 2.0 / (k + 2.0);
        for (std::size_t i = 0; i < res.x.size(); ++i)
            res.x[i] = (1.0 - gamma) * res.x[i] + gamma * s[i];
    }
    res.objective.push_back(f(res.x));
    return res;
}

// Linear minimization oracle for the probability simplex {x >= 0, sum x = 1}:
// the minimizer of <g, s> is the unit vector on the smallest-gradient coordinate.
inline std::vector<double> simplex_lmo(const std::vector<double>& g) {
    std::size_t         best = 0;
    for (std::size_t i = 1; i < g.size(); ++i)
        if (g[i] < g[best]) best = i;
    std::vector<double> s(g.size(), 0.0);
    if (!g.empty()) s[best] = 1.0;
    return s;
}

} // namespace datamunge::algorithms
