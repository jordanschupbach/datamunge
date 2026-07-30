#pragma once

/// \file fuzzy_c_means.hpp
/// \brief Fuzzy c-means (FCM): soft clustering where each point holds a graded
///        membership in every cluster (Dunn 1973; Bezdek 1981).
///
/// Hard clustering (k-means, k-medoids) assigns each point to exactly one cluster.
/// *Fuzzy c-means* instead gives each point \f$x_i\f$ a *membership* \f$u_{ij}\in[0,1]\f$
/// in each cluster \f$j\f$, with \f$\sum_j u_{ij}=1\f$, and minimizes the fuzzy
/// objective
/// \f[
///   J_m = \sum_{i=1}^{n}\sum_{j=1}^{c} u_{ij}^{\,m}\,\|x_i - v_j\|^2,
/// \f]
/// where \f$m>1\f$ is the *fuzzifier* controlling how soft the assignment is
/// (\f$m\to1\f$ recovers hard k-means; larger \f$m\f$ blurs the boundaries).
/// Alternating minimization gives the update equations
/// \f[
///   v_j = \frac{\sum_i u_{ij}^m\,x_i}{\sum_i u_{ij}^m},\qquad
///   u_{ij} = \Bigg(\sum_{l=1}^{c}\Big(\tfrac{\|x_i-v_j\|}{\|x_i-v_l\|}\Big)^{\frac{2}{m-1}}\Bigg)^{-1},
/// \f]
/// iterated until the memberships stop changing. FCM is the fuzzy analogue of
/// Lloyd's algorithm and is widely used where cluster boundaries are genuinely
/// gradual (image segmentation, bioinformatics).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// Result of a fuzzy c-means run.
struct FuzzyCMeansResult {
    std::vector<std::vector<double>> centers;     ///< \f$c\f$ cluster centers \f$v_j\f$.
    std::vector<std::vector<double>> membership;  ///< \f$u_{ij}\f$, \f$n\times c\f$ (rows sum to 1).
    double                           objective  = 0.0;  ///< Final \f$J_m\f$.
    std::size_t                      iterations = 0;
};

namespace detail {

inline double fcm_sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

}  // namespace detail

/// \brief Cluster points with fuzzy c-means.
///
/// Memberships are initialized randomly (normalized per point), then centers and
/// memberships alternate until the maximum membership change drops below \p tol or
/// \p max_iterations is reached.
///
/// \param data           Points to cluster.
/// \param c              Number of clusters.
/// \param fuzzifier      Exponent \f$m>1\f$ (2.0 is standard).
/// \param max_iterations Iteration cap.
/// \param tol            Convergence tolerance on the largest membership change.
/// \param seed           RNG seed for membership initialization.
inline FuzzyCMeansResult fuzzy_c_means(const std::vector<std::vector<double>>& data, std::size_t c,
                                       double fuzzifier = 2.0, std::size_t max_iterations = 100,
                                       double tol = 1e-5, std::uint64_t seed = 0) {
    const std::size_t n = data.size();
    if (c == 0 || c > n) throw std::invalid_argument("fuzzy_c_means: need 0 < c <= n");
    if (fuzzifier <= 1.0) throw std::invalid_argument("fuzzy_c_means: fuzzifier m must exceed 1");
    const std::size_t dim = data.front().size();

    // Random initial memberships, normalized per point.
    std::mt19937_64                        rng(seed);
    std::uniform_real_distribution<double> unit(0.1, 1.0);
    std::vector<std::vector<double>>       u(n, std::vector<double>(c, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < c; ++j) {
            u[i][j] = unit(rng);
            sum += u[i][j];
        }
        for (std::size_t j = 0; j < c; ++j) u[i][j] /= sum;
    }

    std::vector<std::vector<double>> centers(c, std::vector<double>(dim, 0.0));
    const double                     p = 2.0 / (fuzzifier - 1.0);
    FuzzyCMeansResult                result;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        // Update centers.
        for (std::size_t j = 0; j < c; ++j) {
            std::vector<double> num(dim, 0.0);
            double              den = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double w = std::pow(u[i][j], fuzzifier);
                for (std::size_t d = 0; d < dim; ++d) num[d] += w * data[i][d];
                den += w;
            }
            if (den > 0.0)
                for (std::size_t d = 0; d < dim; ++d) centers[j][d] = num[d] / den;
        }

        // Update memberships; track the largest change.
        double max_change = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<double> dist(c, 0.0);
            bool                coincident = false;
            std::size_t         coincident_j = 0;
            for (std::size_t j = 0; j < c; ++j) {
                dist[j] = std::sqrt(detail::fcm_sq_dist(data[i], centers[j]));
                if (dist[j] < 1e-12) {
                    coincident   = true;
                    coincident_j = j;
                }
            }
            for (std::size_t j = 0; j < c; ++j) {
                double newu;
                if (coincident) {
                    newu = (j == coincident_j) ? 1.0 : 0.0;
                } else {
                    double s = 0.0;
                    for (std::size_t l = 0; l < c; ++l) s += std::pow(dist[j] / dist[l], p);
                    newu = 1.0 / s;
                }
                max_change = std::max(max_change, std::abs(newu - u[i][j]));
                u[i][j]    = newu;
            }
        }
        result.iterations = iter + 1;
        if (max_change < tol) break;
    }

    // Final objective.
    double J = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < c; ++j) J += std::pow(u[i][j], fuzzifier) * detail::fcm_sq_dist(data[i], centers[j]);

    result.centers    = centers;
    result.membership = u;
    result.objective  = J;
    return result;
}

}  // namespace datamunge::algorithms
