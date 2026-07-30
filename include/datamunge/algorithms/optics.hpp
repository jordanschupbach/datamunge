#pragma once

/// \file optics.hpp
/// \brief OPTICS: Ordering Points To Identify the Clustering Structure
///        (Ankerst, Breunig, Kriegel & Sander 1999).
///
/// DBSCAN finds density-based clusters but needs a single global density threshold
/// \f$\varepsilon\f$, which fails when clusters have very different densities.
/// *OPTICS* removes that limitation: instead of producing a flat labeling, it
/// produces an *ordering* of the points together with a *reachability distance* for
/// each, such that points of a cluster appear consecutively in the ordering and
/// clusters show up as *valleys* in the reachability plot. From that single ordering
/// one can extract DBSCAN-style clusterings at *any* threshold \f$\varepsilon'\le\varepsilon\f$,
/// or hierarchical clusters of varying density.
///
/// Two per-point quantities drive it, given a radius \f$\varepsilon\f$ and a density
/// parameter \p min_pts:
///   - the *core distance* of \f$p\f$: the smallest radius that makes \f$p\f$ a core
///     point (the distance to its \p min_pts-th nearest neighbor), or undefined if
///     \f$p\f$ has fewer than \p min_pts neighbors within \f$\varepsilon\f$;
///   - the *reachability distance* of \f$o\f$ from \f$p\f$:
///     \f$\max(\text{core-dist}(p),\, d(p,o))\f$.
/// OPTICS repeatedly processes the not-yet-processed point with the smallest
/// reachability, expanding a priority queue ("seeds") of density-reachable points --
/// a best-first traversal of the density landscape.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// The OPTICS ordering and its per-point distances.
struct OpticsResult {
    std::vector<std::size_t> ordering;       ///< Points in processing order.
    std::vector<double>      reachability;   ///< Reachability distance per point id (inf if undefined).
    std::vector<double>      core_distance;  ///< Core distance per point id (inf if not a core point).
};

namespace detail {

inline double optics_euclidean(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

}  // namespace detail

/// \brief Compute the OPTICS ordering and reachability of a point set.
///
/// \param data     Points.
/// \param eps      The generating radius \f$\varepsilon\f$ (an upper bound on the
///                 densities that can be recovered later).
/// \param min_pts  Minimum neighbors (including the point itself) for a core point.
/// \return the processing order and the reachability/core distances (indexed by point id).
inline OpticsResult optics(const std::vector<std::vector<double>>& data, double eps, std::size_t min_pts) {
    const std::size_t n   = data.size();
    constexpr double  kInf = std::numeric_limits<double>::infinity();

    OpticsResult result;
    result.reachability.assign(n, kInf);
    result.core_distance.assign(n, kInf);
    result.ordering.reserve(n);
    std::vector<char> processed(n, 0);

    // Neighbors of p within eps, as (point id, distance) pairs.
    auto neighbors = [&](std::size_t p) {
        std::vector<std::pair<std::size_t, double>> out;
        for (std::size_t q = 0; q < n; ++q) {
            const double d = detail::optics_euclidean(data[p], data[q]);
            if (d <= eps) out.push_back({q, d});
        }
        return out;
    };

    // Core distance: the min_pts-th smallest neighbor distance (self included), else inf.
    auto core_distance = [&](const std::vector<std::pair<std::size_t, double>>& nbrs) {
        if (nbrs.size() < min_pts) return kInf;
        std::vector<double> ds;
        ds.reserve(nbrs.size());
        for (const auto& [q, d] : nbrs) ds.push_back(d);
        std::sort(ds.begin(), ds.end());
        return ds[min_pts - 1];
    };

    std::vector<std::size_t> seeds;  // candidate point ids, min-reachability extracted by scan

    auto update = [&](const std::vector<std::pair<std::size_t, double>>& nbrs, std::size_t p) {
        const double cd = result.core_distance[p];
        for (const auto& [o, d] : nbrs) {
            if (processed[o]) continue;
            const double new_reach = std::max(cd, d);
            if (result.reachability[o] == kInf) {
                result.reachability[o] = new_reach;
                seeds.push_back(o);
            } else if (new_reach < result.reachability[o]) {
                result.reachability[o] = new_reach;  // decrease-key (seeds already contains o)
            }
        }
    };

    auto extract_min = [&]() {
        std::size_t best_pos  = 0;
        double      best_val  = kInf;
        for (std::size_t i = 0; i < seeds.size(); ++i)
            if (result.reachability[seeds[i]] < best_val) {
                best_val = result.reachability[seeds[i]];
                best_pos = i;
            }
        const std::size_t id = seeds[best_pos];
        seeds[best_pos]      = seeds.back();
        seeds.pop_back();
        return id;
    };

    for (std::size_t p = 0; p < n; ++p) {
        if (processed[p]) continue;
        auto nbrs = neighbors(p);
        processed[p] = 1;
        result.core_distance[p] = core_distance(nbrs);
        result.ordering.push_back(p);  // p keeps reachability inf (start of a new density region)

        if (result.core_distance[p] == kInf) continue;
        seeds.clear();
        update(nbrs, p);
        while (!seeds.empty()) {
            const std::size_t q = extract_min();
            if (processed[q]) continue;
            auto qn      = neighbors(q);
            processed[q] = 1;
            result.core_distance[q] = core_distance(qn);
            result.ordering.push_back(q);
            if (result.core_distance[q] != kInf) update(qn, q);
        }
    }
    return result;
}

/// \brief Extract a flat DBSCAN-equivalent clustering from an OPTICS ordering.
///
/// Walks the ordering and cuts the reachability plot at \p eps_cluster: a point whose
/// reachability exceeds \p eps_cluster starts a new cluster if it is itself a core
/// point at that scale, otherwise it is labeled noise (\f$-1\f$); a point whose
/// reachability is within \p eps_cluster continues the current cluster.
/// Equivalent to DBSCAN(\p eps_cluster, min_pts) for any \p eps_cluster ≤ the eps used
/// to build the ordering.
///
/// \return a label per point id (0-based cluster indices, or -1 for noise).
inline std::vector<int> optics_extract_dbscan(const OpticsResult& r, double eps_cluster) {
    const std::size_t n = r.reachability.size();
    std::vector<int>  labels(n, -1);
    int               cluster = -1;
    for (std::size_t pos = 0; pos < r.ordering.size(); ++pos) {
        const std::size_t p = r.ordering[pos];
        if (r.reachability[p] > eps_cluster) {
            if (r.core_distance[p] <= eps_cluster) {
                ++cluster;
                labels[p] = cluster;
            } else {
                labels[p] = -1;  // noise
            }
        } else {
            labels[p] = cluster;
        }
    }
    return labels;
}

}  // namespace datamunge::algorithms
