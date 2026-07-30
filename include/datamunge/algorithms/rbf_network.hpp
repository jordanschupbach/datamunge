#pragma once

/// \file rbf_network.hpp
/// \brief Radial basis function (RBF) network: a two-layer network that fits a
///        function as a weighted sum of Gaussian bumps placed at data-driven centers.
///
/// An RBF network approximates \f$f(x)\f$ as
/// \f[
///   \hat f(x) = w_0 + \sum_{k=1}^{K} w_k\,\phi_k(x),\qquad
///   \phi_k(x) = \exp\!\Big(-\frac{\|x-c_k\|^2}{2\sigma^2}\Big),
/// \f]
/// a linear combination of \f$K\f$ *radial basis functions* -- Gaussians centered at
/// points \f$c_k\f$ with common width \f$\sigma\f$. Training has two decoupled stages
/// (Broomhead & Lowe 1988; Moody & Darken 1989):
///   1. *Unsupervised placement of centers.* The centers \f$c_k\f$ are chosen from
///      the input distribution -- here by k-means -- and \f$\sigma\f$ is set by a
///      spread heuristic.
///   2. *Linear output weights.* With the centers fixed, the map from basis
///      activations to targets is *linear*, so the output weights solve a
///      (ridge-regularized) least-squares problem in closed form -- no iterative
///      gradient training needed.
/// This makes RBF networks fast to fit and a clean example of the "fixed nonlinear
/// features + linear readout" recipe that also underlies kernel methods and extreme
/// learning machines.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A trained RBF network for scalar regression.
struct RBFNetwork {
    std::vector<std::vector<double>> centers;   ///< \f$c_k\f$, \f$K\f$ centers in input space.
    double                           width = 1.0;///< Shared Gaussian width \f$\sigma\f$.
    std::vector<double>              weights;    ///< Output weights \f$w_1..w_K\f$.
    double                           bias = 0.0; ///< Output bias \f$w_0\f$.
};

namespace detail {

inline double rbf_sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

/// Lloyd's k-means to place \p k centers (few iterations; deterministic given \p seed).
inline std::vector<std::vector<double>> rbf_kmeans(const std::vector<std::vector<double>>& X, std::size_t k,
                                                   std::size_t iters, std::uint64_t seed) {
    std::mt19937_64                            rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, X.size() - 1);
    std::vector<std::vector<double>>           c(k);
    for (auto& ck : c) ck = X[pick(rng)];  // random-sample init

    const std::size_t dim = X.front().size();
    for (std::size_t it = 0; it < iters; ++it) {
        std::vector<std::vector<double>> sum(k, std::vector<double>(dim, 0.0));
        std::vector<std::size_t>         cnt(k, 0);
        for (const auto& x : X) {
            std::size_t best = 0;
            double      bd   = std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < k; ++j) {
                const double d = rbf_sq_dist(x, c[j]);
                if (d < bd) {
                    bd   = d;
                    best = j;
                }
            }
            for (std::size_t i = 0; i < dim; ++i) sum[best][i] += x[i];
            ++cnt[best];
        }
        for (std::size_t j = 0; j < k; ++j)
            if (cnt[j] > 0)
                for (std::size_t i = 0; i < dim; ++i) c[j][i] = sum[j][i] / static_cast<double>(cnt[j]);
    }
    return c;
}

/// Solve the symmetric positive-definite system A x = b by Gaussian elimination with
/// partial pivoting (A is m x m, destroyed in place).
inline std::vector<double> rbf_solve(std::vector<std::vector<double>> A, std::vector<double> b) {
    const std::size_t m = b.size();
    for (std::size_t col = 0; col < m; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < m; ++r)
            if (std::abs(A[r][col]) > std::abs(A[piv][col])) piv = r;
        std::swap(A[col], A[piv]);
        std::swap(b[col], b[piv]);
        const double d = A[col][col];
        if (std::abs(d) < 1e-15) continue;
        for (std::size_t r = 0; r < m; ++r) {
            if (r == col) continue;
            const double f = A[r][col] / d;
            for (std::size_t cc = col; cc < m; ++cc) A[r][cc] -= f * A[col][cc];
            b[r] -= f * b[col];
        }
    }
    std::vector<double> x(m, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        if (std::abs(A[i][i]) > 1e-15) x[i] = b[i] / A[i][i];
    return x;
}

}  // namespace detail

/// \brief Train an RBF network for scalar regression.
///
/// Centers are placed by k-means; the shared width defaults to the spread heuristic
/// \f$\sigma = d_{\max}/\sqrt{2K}\f$ where \f$d_{\max}\f$ is the largest distance
/// between centers. The output weights (including a bias) solve the ridge least
/// squares \f$(\Phi^\top\Phi + \lambda I)\,w = \Phi^\top y\f$, where \f$\Phi\f$ is the
/// design matrix of basis activations (plus a constant column).
///
/// \param X            Input vectors.
/// \param y            Scalar targets, one per input.
/// \param num_centers  Number of RBF centers \f$K\f$.
/// \param ridge_lambda Tikhonov regularization on the output weights.
/// \param width        Gaussian width; <=0 selects the spread heuristic.
/// \param kmeans_iters Lloyd iterations for center placement.
/// \param seed         RNG seed for center initialization.
inline RBFNetwork rbf_train(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                            std::size_t num_centers, double ridge_lambda = 1e-6, double width = 0.0,
                            std::size_t kmeans_iters = 20, std::uint64_t seed = 0) {
    if (X.size() != y.size() || X.empty()) throw std::invalid_argument("rbf: bad data");
    const std::size_t k = std::min(num_centers, X.size());
    RBFNetwork        net;
    net.centers = detail::rbf_kmeans(X, k, kmeans_iters, seed);

    if (width > 0.0) {
        net.width = width;
    } else {
        double dmax = 0.0;
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t b = a + 1; b < k; ++b) dmax = std::max(dmax, std::sqrt(detail::rbf_sq_dist(net.centers[a], net.centers[b])));
        net.width = (k > 1 && dmax > 0.0) ? dmax / std::sqrt(2.0 * static_cast<double>(k)) : 1.0;
    }

    // Design matrix Phi (n x (k+1)); column 0 is the bias.
    const std::size_t              n   = X.size();
    const std::size_t              m   = k + 1;
    const double                   two_s2 = 2.0 * net.width * net.width;
    std::vector<std::vector<double>> Phi(n, std::vector<double>(m, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        Phi[i][0] = 1.0;
        for (std::size_t j = 0; j < k; ++j)
            Phi[i][j + 1] = std::exp(-detail::rbf_sq_dist(X[i], net.centers[j]) / two_s2);
    }

    // Normal equations with ridge: (Phi^T Phi + lambda I) w = Phi^T y.
    std::vector<std::vector<double>> A(m, std::vector<double>(m, 0.0));
    std::vector<double>              rhs(m, 0.0);
    for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t b = 0; b < m; ++b) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) s += Phi[i][a] * Phi[i][b];
            A[a][b] = s + (a == b ? ridge_lambda : 0.0);
        }
        double r = 0.0;
        for (std::size_t i = 0; i < n; ++i) r += Phi[i][a] * y[i];
        rhs[a] = r;
    }

    const std::vector<double> w = detail::rbf_solve(A, rhs);
    net.bias                    = w[0];
    net.weights.assign(w.begin() + 1, w.end());
    return net;
}

/// Predict the scalar output of a trained RBF network at \p x.
inline double rbf_predict(const RBFNetwork& net, const std::vector<double>& x) {
    double out = net.bias;
    const double two_s2 = 2.0 * net.width * net.width;
    for (std::size_t j = 0; j < net.centers.size(); ++j)
        out += net.weights[j] * std::exp(-detail::rbf_sq_dist(x, net.centers[j]) / two_s2);
    return out;
}

}  // namespace datamunge::algorithms
