#pragma once

/// @file extreme_detail.hpp
/// @brief Internal numerical helpers shared by the extreme-value fitters (GEV, GPD): central-difference
/// derivatives and a damped Newton-Raphson maximizer.

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::stats::detail {

/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief A log-likelihood in k free coordinates.
using Objective = std::function<double(const std::vector<double>&)>;

/// @brief Central-difference gradient and Hessian.
/// @param f Function.
/// @param p Point.
/// @param g Output gradient.
/// @param h Output Hessian.
inline void derivatives(const Objective& f, const std::vector<double>& p, std::vector<double>& g, Matrix& h) {
    const std::size_t k = p.size();
    g.assign(k, 0.0);
    h = Matrix(k, k, 0.0);
    const double f0 = f(p);
    std::vector<double> step(k);
    for (std::size_t i = 0; i < k; ++i)
        step[i] = 1e-4 * std::max(1.0, std::abs(p[i]));
    for (std::size_t i = 0; i < k; ++i) {
        auto a = p, b = p;
        a[i] += step[i], b[i] -= step[i];
        const double fa = f(a), fb = f(b);
        g[i] = (fa - fb) / (2 * step[i]);
        h(i, i) = (fa - 2 * f0 + fb) / (step[i] * step[i]);
        for (std::size_t j = 0; j < i; ++j) {
            auto pp = p, pm = p, mp = p, mm = p;
            pp[i] += step[i], pp[j] += step[j];
            pm[i] += step[i], pm[j] -= step[j];
            mp[i] -= step[i], mp[j] += step[j];
            mm[i] -= step[i], mm[j] -= step[j];
            h(i, j) = h(j, i) = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * step[i] * step[j]);
        }
    }
    // Refine the gradient with a smaller step (the Hessian only steers Newton).
    for (std::size_t i = 0; i < k; ++i) {
        const double e = 1e-6 * std::max(1.0, std::abs(p[i]));
        auto a = p, b = p;
        a[i] += e, b[i] -= e;
        const double fa = f(a), fb = f(b);
        if (std::isfinite(fa) && std::isfinite(fb))
            g[i] = (fa - fb) / (2 * e);
    }
}

/// @brief Newton-Raphson maximization with step halving and a gradient-ascent fallback.
/// @param f Function to maximize.
/// @param p Start (updated).
/// @param max_it Iteration cap.
/// @param tol Step tolerance.
/// @param it Output iterations.
/// @return Whether converged.
inline bool newton(const Objective& f, std::vector<double>& p, std::size_t max_it, double tol, std::size_t& it) {
    double fp = f(p);
    if (!std::isfinite(fp))
        return false;
    std::vector<double> g;
    Matrix h;
    for (it = 0; it < max_it; ++it) {
        derivatives(f, p, g, h);
        const std::size_t k = p.size();
        Matrix neg(k, k, 0.0);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                neg(i, j) = -h(i, j);
        auto chol = linalg::cholesky(neg);
        std::vector<double> step;
        if (chol.ok)
            step = chol.solve(g);
        else
            step = g; // not concave here: gradient ascent with halving
        double size = 0;
        for (std::size_t i = 0; i < k; ++i)
            size = std::max(size, std::abs(step[i]) / std::max(1.0, std::abs(p[i])));
        if (chol.ok && size < tol)
            return true;
        double t = 1;
        bool moved = false;
        for (int halve = 0; halve < 60; ++halve, t /= 2) {
            auto trial = p;
            for (std::size_t i = 0; i < k; ++i)
                trial[i] += t * step[i];
            const double ft = f(trial);
            if (std::isfinite(ft) && ft >= fp) {
                moved = ft > fp || t * size < tol;
                p = trial, fp = ft;
                break;
            }
        }
        if (!moved)
            return chol.ok && size < 1e3 * tol;
    }
    return false;
}

} // namespace datamunge::stats::detail
