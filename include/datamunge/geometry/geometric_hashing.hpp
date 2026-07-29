#pragma once

// Geometric hashing: recognize a rigid 2-D point pattern (a "model") inside a
// larger, cluttered "scene" even after an unknown similarity transform
// (rotation, uniform scale, translation).
//
// The trick is a transform-invariant coordinate frame. Pick an ordered pair of
// points as a *basis* and express every other point in the frame that sends the
// first basis point to (0,0) and the second to (1,0) -- a complex division
// (p - b0) / (b1 - b0). Those coordinates are invariant to any similarity, so a
// model point's frame coordinates are the same in the scene, whatever transform
// was applied.
//
//   Preprocess (offline): for every model basis, hash the invariant coordinates
//     of the other model points into a table, keyed by quantized coordinate.
//   Recognize (online): for a scene basis, look up each other scene point's
//     invariant coordinates and let matching table entries *vote* for a model
//     basis. The basis pair with the most votes is the correspondence; from it
//     the similarity transform is recovered and verified.

#include <datamunge/geometry/point2d.hpp>

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::geometry {

namespace detail {

// Invariant coordinates of p in the basis frame (b0 -> 0, b1 -> 1): the complex
// quotient (p - b0) / (b1 - b0). Returns false if the basis is degenerate.
inline bool gh_frame_coord(const Point2D& p, const Point2D& b0, const Point2D& b1, double& u, double& v) {
    const double dx = b1.x - b0.x, dy = b1.y - b0.y;
    const double L2 = dx * dx + dy * dy;
    if (L2 < 1e-18) return false;
    const double ax = p.x - b0.x, ay = p.y - b0.y;
    u = (ax * dx + ay * dy) / L2;
    v = (ay * dx - ax * dy) / L2;
    return true;
}

inline std::uint64_t gh_bin_key(double u, double v, double quant) {
    const auto bu = static_cast<std::int32_t>(std::floor(u / quant));
    const auto bv = static_cast<std::int32_t>(std::floor(v / quant));
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(bu)) << 32) |
           static_cast<std::uint32_t>(bv);
}

} // namespace detail

struct GeoHashTable {
    double                                                    quant{0.1};
    std::unordered_map<std::uint64_t, std::vector<std::pair<int, int>>> bins; // bin -> model bases (i,j)
};

struct GeoMatch {
    bool    found{false};
    int     votes{0};
    int     model_i{-1}, model_j{-1}; // model basis
    int     scene_a{-1}, scene_b{-1}; // matching scene basis
    double  scale{0}, angle{0};       // scene = scale * Rot(angle) * model + translation
    Point2D translation{0, 0};
    int     inliers{0};
};

// Offline: hash every basis of the model. `quant` is the coordinate bin size.
inline GeoHashTable build_geometric_hash(const std::vector<Point2D>& model, double quant = 0.1) {
    GeoHashTable table;
    table.quant   = quant;
    const int n   = static_cast<int>(model.size());
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            for (int k = 0; k < n; ++k) {
                if (k == i || k == j) continue;
                double u, v;
                if (!detail::gh_frame_coord(model[k], model[i], model[j], u, v)) continue;
                table.bins[detail::gh_bin_key(u, v, quant)].emplace_back(i, j);
            }
        }
    return table;
}

// Online: find the model inside `scene`. `tol` is the inlier distance for the
// final geometric verification of the recovered transform.
inline GeoMatch recognize(const GeoHashTable&         table,
                          const std::vector<Point2D>& model,
                          const std::vector<Point2D>& scene,
                          double                      tol      = 1e-6,
                          int                         min_votes = 2) {
    using namespace detail;
    GeoMatch  best;
    const int m = static_cast<int>(scene.size());

    for (int a = 0; a < m; ++a)
        for (int b = 0; b < m; ++b) {
            if (a == b) continue;
            std::unordered_map<std::uint64_t, int> tally; // packed model basis -> votes
            for (int k = 0; k < m; ++k) {
                if (k == a || k == b) continue;
                double u, v;
                if (!gh_frame_coord(scene[k], scene[a], scene[b], u, v)) continue;
                auto it = table.bins.find(gh_bin_key(u, v, table.quant));
                if (it == table.bins.end()) continue;
                for (const auto& mb : it->second) {
                    const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(mb.first)) << 32) |
                                     static_cast<std::uint32_t>(mb.second);
                    ++tally[key];
                }
            }
            for (const auto& [key, votes] : tally) {
                if (votes <= best.votes) continue;
                best.votes   = votes;
                best.scene_a = a;
                best.scene_b = b;
                best.model_i = static_cast<int>(key >> 32);
                best.model_j = static_cast<std::int32_t>(static_cast<std::uint32_t>(key));
            }
        }

    if (best.votes < min_votes) return best;

    // Recover the similarity from the basis correspondence model(i,j) -> scene(a,b),
    // as the complex ratio (scene_b - scene_a) / (model_j - model_i).
    const Point2D& mi = model[best.model_i];
    const Point2D& mj = model[best.model_j];
    const Point2D& sa = scene[best.scene_a];
    const Point2D& sb = scene[best.scene_b];
    const double   mdx = mj.x - mi.x, mdy = mj.y - mi.y;
    const double   sdx = sb.x - sa.x, sdy = sb.y - sa.y;
    const double   mL2 = mdx * mdx + mdy * mdy;
    if (mL2 < 1e-18) return best;
    const double cs = (sdx * mdx + sdy * mdy) / mL2; // scale * cos(angle)
    const double sn = (sdy * mdx - sdx * mdy) / mL2; // scale * sin(angle)
    best.scale = std::sqrt(cs * cs + sn * sn);
    best.angle = std::atan2(sn, cs);
    // translation = sa - [cs -sn; sn cs] * mi
    best.translation = {sa.x - (cs * mi.x - sn * mi.y), sa.y - (sn * mi.x + cs * mi.y)};
    best.found = true;

    // Verify: count model points that map within tol of some scene point.
    for (const auto& p : model) {
        const double tx = cs * p.x - sn * p.y + best.translation.x;
        const double ty = sn * p.x + cs * p.y + best.translation.y;
        for (const auto& q : scene) {
            const double dx = tx - q.x, dy = ty - q.y;
            if (dx * dx + dy * dy <= tol * tol) { ++best.inliers; break; }
        }
    }
    return best;
}

} // namespace datamunge::geometry
