#pragma once

// Iterative Closest Point (ICP): rigid registration of two 2-D point sets. Given
// a source cloud and a target cloud whose correspondence is unknown, ICP recovers
// the rotation and translation that best overlays source on target by alternating
// two steps until convergence:
//
//   1. Correspondence: match each (current) source point to its nearest target.
//   2. Alignment: solve the closed-form least-squares rigid transform (Kabsch/
//      Umeyama) for those correspondences, apply it, and repeat.
//
// Each iteration cannot increase the mean-squared alignment error, so ICP
// converges to a local optimum; from a reasonable initial overlap it recovers the
// true transform. In 2-D the optimal rotation has a closed form (no SVD): with
// centered clouds, theta = atan2( sum p_x q_y - p_y q_x , sum p_x q_x + p_y q_y ).

#include <datamunge/geometry/point2d.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::geometry {

struct IcpResult {
    double  angle{0};          // total rotation applied to the source (radians)
    Point2D translation{0, 0}; // total translation
    int     iterations{0};
    double  rmse{0};           // final root-mean-square alignment error
};

// Register `source` onto `target`. Returns the rigid transform (angle, translation)
// such that R(angle) * source + translation approximates target.
inline IcpResult icp(const std::vector<Point2D>& source, const std::vector<Point2D>& target,
                     int max_iters = 100, double tol = 1e-14) {
    IcpResult            R;
    std::vector<Point2D> cur = source;
    if (source.empty() || target.empty()) return R;

    double prev_rmse = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < max_iters; ++iter) {
        // 1. Nearest-target correspondence for each current source point.
        std::vector<Point2D> match(cur.size());
        double               sse = 0.0;
        for (std::size_t i = 0; i < cur.size(); ++i) {
            double best = std::numeric_limits<double>::infinity();
            std::size_t bj = 0;
            for (std::size_t j = 0; j < target.size(); ++j) {
                const double dx = cur[i].x - target[j].x, dy = cur[i].y - target[j].y;
                const double d = dx * dx + dy * dy;
                if (d < best) { best = d; bj = j; }
            }
            match[i] = target[bj];
            sse += best;
        }
        const double rmse = std::sqrt(sse / static_cast<double>(cur.size()));

        // 2. Closed-form optimal rigid transform for these correspondences.
        Point2D cc{0, 0}, cq{0, 0};
        for (std::size_t i = 0; i < cur.size(); ++i) {
            cc.x += cur[i].x;  cc.y += cur[i].y;
            cq.x += match[i].x; cq.y += match[i].y;
        }
        const double inv = 1.0 / static_cast<double>(cur.size());
        cc.x *= inv; cc.y *= inv; cq.x *= inv; cq.y *= inv;
        double a = 0.0, b = 0.0;
        for (std::size_t i = 0; i < cur.size(); ++i) {
            const double px = cur[i].x - cc.x, py = cur[i].y - cc.y;
            const double qx = match[i].x - cq.x, qy = match[i].y - cq.y;
            a += px * qx + py * qy;
            b += px * qy - py * qx;
        }
        const double dtheta = std::atan2(b, a);
        const double cs = std::cos(dtheta), sn = std::sin(dtheta);
        // dt maps the current centroid onto the matched centroid after rotation.
        const Point2D dt{cq.x - (cs * cc.x - sn * cc.y), cq.y - (sn * cc.x + cs * cc.y)};

        for (auto& p : cur) {
            const double x = cs * p.x - sn * p.y + dt.x;
            const double y = sn * p.x + cs * p.y + dt.y;
            p = {x, y};
        }
        // Accumulate: R_total <- dR * R_total, t_total <- dR * t_total + dt.
        R.angle += dtheta;
        R.translation = {cs * R.translation.x - sn * R.translation.y + dt.x,
                         sn * R.translation.x + cs * R.translation.y + dt.y};
        R.iterations = iter + 1;
        R.rmse       = rmse;

        if (std::fabs(prev_rmse - rmse) < tol) break;
        prev_rmse = rmse;
    }
    return R;
}

} // namespace datamunge::geometry
