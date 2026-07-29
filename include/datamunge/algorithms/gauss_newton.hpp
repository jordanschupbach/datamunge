#pragma once

// The Gauss-Newton algorithm for nonlinear least squares: fit parameters p of a
// model m(p, x) to data (x_i, y_i) by minimizing the sum of squared residuals
// S(p) = sum_i r_i(p)^2 with r_i = m(p, x_i) - y_i. Newton's method would need
// the Hessian of S; Gauss-Newton approximates it by dropping the second-order
// term, leaving only the Jacobian J of the residuals. Each step solves the normal
// equations
//
//     (J^T J) delta = -J^T r,      p <- p + delta,
//
// which is just the linear least-squares step for the locally linearized model.
// Near a good fit the neglected term is small and convergence is fast (quadratic
// for zero-residual problems). It is the ungated core that Levenberg-Marquardt
// wraps with damping for robustness.

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

struct GaussNewtonResult {
    std::vector<double> params;
    int                 iterations{0};
    double              sse{0}; // final sum of squared residuals
};

namespace detail {

// Solve the symmetric system A x = b (A is n x n) by Gaussian elimination.
inline std::vector<double> gn_solve(std::vector<std::vector<double>> A, std::vector<double> b) {
    const std::size_t n = b.size();
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < n; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        std::swap(A[col], A[piv]);
        std::swap(b[col], b[piv]);
        const double d = A[col][col];
        if (std::fabs(d) < 1e-300) continue;
        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = A[r][col] / d;
            for (std::size_t c = col; c < n; ++c) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = A[i][i] != 0 ? b[i] / A[i][i] : 0.0;
    return x;
}

} // namespace detail

// Fit `model`(params, x) to data (xs, ys) starting from params0. `model` is any
// callable double(const std::vector<double>&, double). The Jacobian is computed
// by finite differences, so only the model is required.
template <class Model>
GaussNewtonResult gauss_newton(Model model, const std::vector<double>& xs,
                               const std::vector<double>& ys, std::vector<double> params0,
                               int max_iter = 100, double tol = 1e-12) {
    const std::size_t m = xs.size(), n = params0.size();
    std::vector<double> p = std::move(params0);
    GaussNewtonResult   out;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Residuals and Jacobian J[i][k] = d r_i / d p_k (finite difference).
        std::vector<double>              r(m);
        std::vector<std::vector<double>> J(m, std::vector<double>(n));
        double                           sse = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double fi = model(p, xs[i]);
            r[i] = fi - ys[i];
            sse += r[i] * r[i];
            for (std::size_t k = 0; k < n; ++k) {
                const double h = 1e-6 * (std::fabs(p[k]) + 1e-6);
                std::vector<double> pp = p;
                pp[k] += h;
                J[i][k] = (model(pp, xs[i]) - fi) / h;
            }
        }
        out.sse        = sse;
        out.iterations = iter;

        // Normal equations (J^T J) delta = -J^T r.
        std::vector<std::vector<double>> JtJ(n, std::vector<double>(n, 0.0));
        std::vector<double>              Jtr(n, 0.0);
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t l = 0; l < n; ++l) {
                double s = 0;
                for (std::size_t i = 0; i < m; ++i) s += J[i][k] * J[i][l];
                JtJ[k][l] = s;
            }
            double s = 0;
            for (std::size_t i = 0; i < m; ++i) s += J[i][k] * r[i];
            Jtr[k] = -s;
        }
        const std::vector<double> delta = detail::gn_solve(JtJ, Jtr);
        double                    step = 0;
        for (std::size_t k = 0; k < n; ++k) { p[k] += delta[k]; step += delta[k] * delta[k]; }
        if (std::sqrt(step) < tol) { out.iterations = iter + 1; break; }
    }
    out.params = p;
    return out;
}

} // namespace datamunge::algorithms
