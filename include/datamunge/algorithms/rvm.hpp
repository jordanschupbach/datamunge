#pragma once

/// \file rvm.hpp
/// \brief The Relevance Vector Machine (RVM) for regression: sparse Bayesian
///        kernel learning (Tipping 2001).
///
/// The RVM is a Bayesian counterpart to the support vector machine. It fits a
/// kernel expansion \f$y(x)=w_0+\sum_j w_j\,k(x,x_j)\f$ but places an independent
/// Gaussian prior \f$w_j\sim\mathcal{N}(0,\alpha_j^{-1})\f$ on each weight, with its
/// *own* precision hyperparameter \f$\alpha_j\f$. Maximizing the marginal likelihood
/// (evidence) over the \f$\alpha_j\f$ drives most of them to infinity -- pruning their
/// weights to exactly zero. The few surviving basis points are the *relevance
/// vectors*. Compared with the SVM, the RVM typically needs far fewer kernel terms,
/// produces *probabilistic* predictions (a full predictive variance), and needs no
/// cross-validation of a margin parameter -- at the cost of a non-convex evidence
/// optimization.
///
/// The evidence is maximized by the fixed-point re-estimation (with
/// \f$\Sigma=(A+\beta\Phi^\top\Phi)^{-1}\f$, \f$\mu=\beta\Sigma\Phi^\top t\f$,
/// \f$A=\mathrm{diag}(\alpha)\f$):
/// \f[
///   \gamma_j = 1-\alpha_j\Sigma_{jj},\quad
///   \alpha_j \leftarrow \frac{\gamma_j}{\mu_j^2},\quad
///   \beta \leftarrow \frac{N-\sum_j\gamma_j}{\lVert t-\Phi\mu\rVert^2}.
/// \f]

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A trained RVM regressor.
struct RVMModel {
    std::vector<std::vector<double>> centers;   ///< Kernel centers (the training inputs).
    std::vector<double>              weights;    ///< Posterior mean weights (index 0 = bias, then per-center).
    std::vector<char>               relevant;   ///< Whether each center survived pruning (size = centers).
    double                          gamma = 1.0; ///< RBF kernel width parameter.
    double                          beta  = 1.0; ///< Learned noise precision.
    std::size_t                     iterations = 0;
    std::size_t                     num_relevance_vectors = 0;
};

namespace detail {

inline double rvm_sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; s += d * d; }
    return s;
}

/// Invert an m x m matrix in place by Gauss-Jordan with partial pivoting; returns false if singular.
inline bool rvm_inverse(std::vector<std::vector<double>>& A) {
    const std::size_t m = A.size();
    std::vector<std::vector<double>> I(m, std::vector<double>(m, 0.0));
    for (std::size_t i = 0; i < m; ++i) I[i][i] = 1.0;
    for (std::size_t col = 0; col < m; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < m; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        if (std::fabs(A[piv][col]) < 1e-300) return false;
        std::swap(A[col], A[piv]);
        std::swap(I[col], I[piv]);
        const double d = A[col][col];
        for (std::size_t j = 0; j < m; ++j) { A[col][j] /= d; I[col][j] /= d; }
        for (std::size_t r = 0; r < m; ++r) {
            if (r == col) continue;
            const double f = A[r][col];
            for (std::size_t j = 0; j < m; ++j) { A[r][j] -= f * A[col][j]; I[r][j] -= f * I[col][j]; }
        }
    }
    A.swap(I);
    return true;
}

}  // namespace detail

/// \brief Train an RVM for regression with a Gaussian (RBF) kernel.
///
/// \param X            Training inputs.
/// \param t            Training targets.
/// \param gamma        RBF kernel width in \f$k(x,x')=e^{-\gamma\lVert x-x'\rVert^2}\f$.
/// \param max_iters    Maximum evidence re-estimation iterations.
/// \param prune_threshold \f$\alpha_j\f$ above this is treated as pruned (weight forced to 0).
/// \param tol          Convergence tolerance on the largest log-alpha change.
inline RVMModel rvm_train(const std::vector<std::vector<double>>& X, const std::vector<double>& t,
                          double gamma = 1.0, std::size_t max_iters = 200, double prune_threshold = 1e9,
                          double tol = 1e-4) {
    const std::size_t n = X.size();
    const std::size_t m = n + 1;  // bias + one basis per training point
    RVMModel          model;
    model.centers = X;
    model.gamma   = gamma;
    if (n == 0) return model;

    // Design matrix Phi (n x m): column 0 = bias, column j+1 = k(x_i, x_j).
    std::vector<std::vector<double>> Phi(n, std::vector<double>(m, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        Phi[i][0] = 1.0;
        for (std::size_t j = 0; j < n; ++j) Phi[i][j + 1] = std::exp(-gamma * detail::rvm_sq_dist(X[i], X[j]));
    }

    double tvar = 0.0, tmean = 0.0;
    for (double v : t) tmean += v;
    tmean /= n;
    for (double v : t) tvar += (v - tmean) * (v - tmean);
    tvar /= (n > 1 ? n - 1 : 1);

    std::vector<double> alpha(m, 1.0);
    double              beta = 1.0 / (0.1 * tvar + 1e-9);
    std::vector<double> mu(m, 0.0);

    for (std::size_t it = 0; it < max_iters; ++it) {
        // Sigma = (A + beta Phi^T Phi)^{-1}.
        std::vector<std::vector<double>> S(m, std::vector<double>(m, 0.0));
        for (std::size_t a = 0; a < m; ++a)
            for (std::size_t b = 0; b < m; ++b) {
                double s = 0.0;
                for (std::size_t i = 0; i < n; ++i) s += Phi[i][a] * Phi[i][b];
                S[a][b] = beta * s + (a == b ? alpha[a] : 0.0);
            }
        if (!detail::rvm_inverse(S)) break;

        // mu = beta Sigma Phi^T t.
        std::vector<double> pt(m, 0.0);
        for (std::size_t a = 0; a < m; ++a)
            for (std::size_t i = 0; i < n; ++i) pt[a] += Phi[i][a] * t[i];
        for (std::size_t a = 0; a < m; ++a) {
            double s = 0.0;
            for (std::size_t b = 0; b < m; ++b) s += S[a][b] * pt[b];
            mu[a] = beta * s;
        }

        // Re-estimate alpha and beta.
        double max_change = 0.0, sum_gamma = 0.0;
        for (std::size_t a = 0; a < m; ++a) {
            const double g = 1.0 - alpha[a] * S[a][a];
            sum_gamma += g;
            double new_alpha = (mu[a] * mu[a] > 1e-300) ? g / (mu[a] * mu[a]) : prune_threshold * 10.0;
            if (new_alpha <= 0.0 || !std::isfinite(new_alpha)) new_alpha = prune_threshold * 10.0;
            max_change = std::max(max_change, std::fabs(std::log(new_alpha + 1e-300) - std::log(alpha[a] + 1e-300)));
            alpha[a] = new_alpha;
        }
        double resid = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double pred = 0.0;
            for (std::size_t a = 0; a < m; ++a) pred += Phi[i][a] * mu[a];
            resid += (t[i] - pred) * (t[i] - pred);
        }
        beta = (static_cast<double>(n) - sum_gamma) / (resid + 1e-12);
        if (beta <= 0.0 || !std::isfinite(beta)) beta = 1.0 / (tvar + 1e-9);

        model.iterations = it + 1;
        if (max_change < tol && it > 1) break;
    }

    model.weights = mu;
    model.beta    = beta;
    model.relevant.assign(n, 0);
    model.num_relevance_vectors = 0;
    for (std::size_t j = 0; j < n; ++j)
        if (alpha[j + 1] < prune_threshold) {
            model.relevant[j] = 1;
            ++model.num_relevance_vectors;
        } else {
            model.weights[j + 1] = 0.0;  // pruned
        }
    return model;
}

/// Predict the RVM regression output at \p x.
inline double rvm_predict(const RVMModel& model, const std::vector<double>& x) {
    double y = model.weights[0];  // bias
    for (std::size_t j = 0; j < model.centers.size(); ++j)
        if (model.relevant[j])
            y += model.weights[j + 1] * std::exp(-model.gamma * detail::rvm_sq_dist(x, model.centers[j]));
    return y;
}

}  // namespace datamunge::algorithms
