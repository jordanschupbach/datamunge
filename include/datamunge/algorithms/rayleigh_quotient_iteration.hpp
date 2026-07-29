#pragma once

// Rayleigh quotient iteration: a supercharged inverse iteration that recomputes
// the shift every step. Inverse iteration solves (A - mu I) y = x with a *fixed*
// mu; here mu is replaced each iteration by the Rayleigh quotient of the current
// estimate,
//
//     mu_k = (x_k^T A x_k) / (x_k^T x_k),      (A - mu_k I) y = x_k,   x_{k+1} = y/||y||.
//
// The Rayleigh quotient is the best scalar approximation to an eigenvalue for a
// given vector, so as the vector homes in on an eigenvector the shift homes in on
// its eigenvalue, and the two reinforce each other. For symmetric matrices this
// gives *cubic* convergence -- the number of correct digits roughly triples each
// step -- so a handful of iterations reach machine precision. The cost is that
// the matrix (A - mu_k I) must be re-factored every step.

#include <datamunge/algorithms/inverse_iteration.hpp> // detail helpers, EigenPair

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

// Eigenpair of `A` reached from the starting vector `x0` by Rayleigh quotient
// iteration. Converges to the eigenpair whose eigenvalue is nearest the starting
// Rayleigh quotient. `residuals` (if non-null) receives ||A x - mu x|| per step.
inline EigenPair rayleigh_quotient_iteration(const std::vector<std::vector<double>>& A,
                                             std::vector<double> x0, int iters = 20,
                                             std::vector<double>* residuals = nullptr) {
    const int n = static_cast<int>(A.size());
    std::vector<double> x = std::move(x0);
    detail::eig_normalize(x);
    double mu = 0;

    for (int k = 0; k < iters; ++k) {
        const std::vector<double> Ax = detail::eig_matvec(A, x);
        mu = 0;
        for (int i = 0; i < n; ++i) mu += x[i] * Ax[i]; // Rayleigh quotient (x unit)

        if (residuals) {
            double r = 0;
            for (int i = 0; i < n; ++i) { const double d = Ax[i] - mu * x[i]; r += d * d; }
            residuals->push_back(std::sqrt(r));
        }

        detail::EigMatrix shifted = A;
        for (int i = 0; i < n; ++i) shifted[i][i] -= mu;
        std::vector<double> y = detail::eig_solve(shifted, x);
        if (detail::eig_normalize(y) < 1e-300) break; // exactly singular -> converged
        x = y;
    }
    return {mu, x};
}

} // namespace datamunge::algorithms
