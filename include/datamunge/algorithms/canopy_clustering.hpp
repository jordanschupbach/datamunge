#pragma once

/// \file canopy_clustering.hpp
/// \brief Canopy clustering: a fast, approximate *pre-clustering* that cheaply groups
///        points into overlapping "canopies" (McCallum, Nigam & Ungar 2000).
///
/// Exact clustering (k-means, hierarchical) is expensive because it compares many
/// point pairs with an accurate -- and often costly -- distance. Canopy clustering is
/// a cheap first pass that partitions the data into overlapping subsets ("canopies")
/// using a *cheap* approximate distance and two thresholds \f$T_1 > T_2\f$, so that an
/// expensive clustering afterwards need only compare points that *share a canopy* --
/// a large speedup on big data. The rule:
///   1. While points remain, pick one as a new canopy *center*.
///   2. Every point within \f$T_1\f$ of the center *joins* that canopy (canopies may
///      overlap -- a point can be in several).
///   3. Every point within the tighter \f$T_2\f$ is *removed from the pool*, so it can
///      never start its own canopy (this bounds the number of canopies).
/// The guarantee that matters downstream: any two points that are truly close share at
/// least one canopy, so restricting expensive comparisons to shared canopies loses
/// little while skipping the vast majority of far-apart pairs.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// One canopy: a center point index and the members within \f$T_1\f$ of it.
struct Canopy {
    std::size_t              center;   ///< Index of the point that seeded this canopy.
    std::vector<std::size_t> members;  ///< Indices of all points within T1 of the center.
};

namespace detail {
inline double canopy_dist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}
}  // namespace detail

/// \brief Compute canopies over \p points with loose/tight thresholds \p t1 > \p t2.
///
/// Points are consumed from a pool in index order: the first remaining point seeds a
/// canopy, all points within \p t1 join it, and all within \p t2 are removed from the
/// pool. Repeats until the pool is empty.
///
/// \param points  Data points.
/// \param t1      Loose threshold (canopy membership radius); must be >= \p t2.
/// \param t2      Tight threshold (removal radius).
inline std::vector<Canopy> canopy_clustering(const std::vector<std::vector<double>>& points, double t1,
                                             double t2) {
    const std::size_t   n = points.size();
    std::vector<char>   removed(n, 0);
    std::vector<Canopy> canopies;

    for (std::size_t i = 0; i < n; ++i) {
        if (removed[i]) continue;
        Canopy c;
        c.center = i;
        for (std::size_t j = 0; j < n; ++j) {
            const double d = detail::canopy_dist(points[i], points[j]);
            if (d < t1) c.members.push_back(j);      // joins the canopy (overlap allowed)
            if (d < t2) removed[j] = 1;              // claimed: cannot seed a future canopy
        }
        canopies.push_back(std::move(c));
    }
    return canopies;
}

/// Convenience: the canopy center coordinates.
inline std::vector<std::vector<double>> canopy_centers(const std::vector<std::vector<double>>& points,
                                                       const std::vector<Canopy>&               canopies) {
    std::vector<std::vector<double>> centers;
    centers.reserve(canopies.size());
    for (const auto& c : canopies) centers.push_back(points[c.center]);
    return centers;
}

}  // namespace datamunge::algorithms
