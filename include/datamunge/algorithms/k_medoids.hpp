#pragma once

/// \file k_medoids.hpp
/// \brief k-medoids clustering by Partitioning Around Medoids (PAM).
///
/// Like k-means, k-medoids partitions points into \f$k\f$ clusters by minimizing
/// within-cluster dissimilarity, but each cluster is represented by an *actual data
/// point* -- its *medoid* -- rather than a coordinate mean. It minimizes the total
/// deviation
/// \f[
///   \text{cost} = \sum_{i} d\big(x_i,\, m_{c(i)}\big),
/// \f]
/// the sum of distances from each point to its cluster's medoid. Using medoids
/// instead of means makes k-medoids far more *robust to outliers* (a stray point
/// cannot drag a center off into empty space) and applicable to *any* dissimilarity
/// -- it never averages coordinates, so it works on arbitrary metrics.
///
/// This implements the classic PAM algorithm (Kaufman & Rousseeuw 1990): a greedy
/// *build* of an initial medoid set, then a *swap* phase that repeatedly exchanges a
/// medoid with a non-medoid whenever doing so lowers the total cost, until no
/// improving swap remains (a local optimum).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// Result of a k-medoids run.
struct KMedoidsResult {
    std::vector<std::size_t> medoids;     ///< Indices (into the input) of the \f$k\f$ chosen medoids.
    std::vector<std::size_t> assignment;  ///< Cluster index (0..k-1) of each point.
    double                   total_cost = 0.0;  ///< Sum of distances of points to their medoid.
    std::size_t              iterations = 0;     ///< Swap iterations performed.
};

namespace detail {

inline double km_euclidean(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

}  // namespace detail

/// \brief Cluster points with PAM (Partitioning Around Medoids).
///
/// Medoids are initialized to \p k distinct random points, then the swap phase runs:
/// for every (medoid, non-medoid) pair it evaluates the total cost after the swap and
/// commits the single best improving swap, repeating until no swap helps or
/// \p max_iterations is reached. Distances are Euclidean.
///
/// \param data           Points to cluster.
/// \param k              Number of clusters/medoids.
/// \param max_iterations Cap on swap iterations.
/// \param seed           RNG seed for initial medoid selection.
inline KMedoidsResult k_medoids(const std::vector<std::vector<double>>& data, std::size_t k,
                                std::size_t max_iterations = 100, std::uint64_t seed = 0) {
    const std::size_t n = data.size();
    if (k == 0 || k > n) throw std::invalid_argument("k_medoids: need 0 < k <= n");

    // Precompute the full distance matrix (n small in practice).
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) D[i][j] = D[j][i] = detail::km_euclidean(data[i], data[j]);

    // Initialize medoids: k distinct random points.
    std::mt19937_64          rng(seed);
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    std::vector<std::size_t> medoids(idx.begin(), idx.begin() + k);
    std::vector<char>        is_medoid(n, 0);
    for (std::size_t m : medoids) is_medoid[m] = 1;

    auto assign_cost = [&](const std::vector<std::size_t>& meds, std::vector<std::size_t>& assign) {
        double cost = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t best = 0;
            double      bd   = std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < meds.size(); ++c)
                if (D[i][meds[c]] < bd) {
                    bd   = D[i][meds[c]];
                    best = c;
                }
            assign[i] = best;
            cost += bd;
        }
        return cost;
    };

    std::vector<std::size_t> assignment(n, 0);
    double                   cost = assign_cost(medoids, assignment);

    KMedoidsResult result;
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        double      best_cost = cost;
        std::size_t best_m = 0, best_o = 0;
        bool        improved = false;
        for (std::size_t mi = 0; mi < k; ++mi)
            for (std::size_t o = 0; o < n; ++o) {
                if (is_medoid[o]) continue;
                std::vector<std::size_t> trial = medoids;
                trial[mi]                       = o;
                std::vector<std::size_t> tmp(n);
                const double             tc = assign_cost(trial, tmp);
                if (tc < best_cost - 1e-12) {
                    best_cost = tc;
                    best_m    = mi;
                    best_o    = o;
                    improved  = true;
                }
            }
        if (!improved) {
            result.iterations = iter;
            break;
        }
        is_medoid[medoids[best_m]] = 0;
        medoids[best_m]            = best_o;
        is_medoid[best_o]          = 1;
        cost                       = assign_cost(medoids, assignment);
        result.iterations          = iter + 1;
    }

    result.medoids    = medoids;
    result.assignment = assignment;
    result.total_cost = cost;
    return result;
}

}  // namespace datamunge::algorithms
