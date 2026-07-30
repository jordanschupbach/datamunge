#pragma once

/// \file ransac.hpp
/// \brief RANSAC (RANdom SAmple Consensus): robust model fitting in the presence of
///        gross outliers (Fischler & Bolles 1981), specialized here to 2-D lines.
///
/// Least-squares fitting minimizes squared residuals and so is wrecked by a few
/// gross outliers -- a single far-off point can tilt the fit arbitrarily. RANSAC
/// takes the opposite stance: instead of fitting all the data and hoping outliers
/// are small, it *hypothesizes* models from tiny random samples of the data and
/// keeps the one that the most points *agree with*. For each trial it draws the
/// minimal sample needed to define a model (two points for a line), scores the model
/// by its *consensus set* -- the inliers within a residual threshold -- and after many
/// trials returns the model with the largest consensus, optionally refit on its
/// inliers. With enough trials it finds a model supported by the inliers even when
/// outliers are a large fraction of the data.
///
/// The number of trials needed to see at least one all-inlier sample with
/// probability \f$p\f$, given an inlier fraction \f$w\f$ and sample size \f$s\f$, is
/// \f$N = \lceil \log(1-p) / \log(1 - w^s) \rceil\f$.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace datamunge::algorithms {

/// A 2-D line in normalized implicit form \f$a x + b y + c = 0\f$ with \f$a^2+b^2=1\f$,
/// so \f$|a x_i + b y_i + c|\f$ is the perpendicular distance of point \f$i\f$ to the line.
struct RansacLine {
    double            a = 0.0, b = 0.0, c = 0.0;  ///< Normalized line coefficients.
    std::vector<char> inliers;                    ///< Per-point inlier flag (1 = inlier).
    std::size_t       inlier_count = 0;           ///< Size of the consensus set.
    std::size_t       trials       = 0;           ///< Trials actually performed.
};

namespace detail {

/// Perpendicular distance of (x, y) to a normalized line.
inline double ransac_point_line_distance(double a, double b, double c, double x, double y) {
    return std::abs(a * x + b * y + c);
}

/// Total-least-squares (orthogonal) line fit of a point subset, returned normalized.
inline void ransac_fit_line(const std::vector<double>& xs, const std::vector<double>& ys,
                            const std::vector<std::size_t>& idx, double& a, double& b, double& c) {
    double mx = 0.0, my = 0.0;
    for (std::size_t i : idx) {
        mx += xs[i];
        my += ys[i];
    }
    mx /= static_cast<double>(idx.size());
    my /= static_cast<double>(idx.size());
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (std::size_t i : idx) {
        const double dx = xs[i] - mx, dy = ys[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    // Line direction = principal eigenvector of the scatter matrix; normal is orthogonal.
    const double theta = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
    a                  = -std::sin(theta);  // normal direction
    b                  = std::cos(theta);
    const double norm  = std::sqrt(a * a + b * b);
    a /= norm;
    b /= norm;
    c = -(a * mx + b * my);
}

}  // namespace detail

/// \brief Fit a 2-D line robustly with RANSAC.
///
/// Repeatedly samples two distinct points, forms the line through them, counts inliers
/// within \p threshold perpendicular distance, and keeps the largest consensus set;
/// the returned line is refit (orthogonal least squares) on that consensus set.
///
/// \param xs, ys      Point coordinates (equal length).
/// \param threshold   Max perpendicular distance for a point to count as an inlier.
/// \param iterations  Number of random trials.
/// \param seed        RNG seed.
inline RansacLine ransac_line(const std::vector<double>& xs, const std::vector<double>& ys, double threshold,
                              std::size_t iterations, std::uint64_t seed = 0) {
    RansacLine  best;
    const std::size_t n = xs.size();
    if (n < 2) return best;

    std::mt19937_64                            rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    for (std::size_t t = 0; t < iterations; ++t) {
        std::size_t i = pick(rng), j = pick(rng);
        if (i == j) continue;
        // Line through points i and j, normalized.
        double a = ys[j] - ys[i];
        double b = xs[i] - xs[j];
        double c = -(a * xs[i] + b * ys[i]);
        const double norm = std::sqrt(a * a + b * b);
        if (norm < 1e-12) continue;
        a /= norm;
        b /= norm;
        c /= norm;

        std::size_t count = 0;
        for (std::size_t k = 0; k < n; ++k)
            if (detail::ransac_point_line_distance(a, b, c, xs[k], ys[k]) <= threshold) ++count;

        if (count > best.inlier_count) {
            best.a            = a;
            best.b            = b;
            best.c            = c;
            best.inlier_count = count;
        }
    }
    best.trials = iterations;
    if (best.inlier_count == 0) return best;

    // Recompute the inlier set for the best minimal model, then refit on it.
    best.inliers.assign(n, 0);
    std::vector<std::size_t> inlier_idx;
    for (std::size_t k = 0; k < n; ++k)
        if (detail::ransac_point_line_distance(best.a, best.b, best.c, xs[k], ys[k]) <= threshold) {
            best.inliers[k] = 1;
            inlier_idx.push_back(k);
        }
    if (inlier_idx.size() >= 2) detail::ransac_fit_line(xs, ys, inlier_idx, best.a, best.b, best.c);
    best.inlier_count = inlier_idx.size();
    return best;
}

/// Trials needed to see an all-inlier minimal sample with probability \p success_prob,
/// given inlier fraction \p inlier_fraction and minimal sample size \p sample_size.
inline std::size_t ransac_required_iterations(double inlier_fraction, std::size_t sample_size,
                                              double success_prob = 0.99) {
    const double w_s = std::pow(inlier_fraction, static_cast<double>(sample_size));
    if (w_s <= 0.0 || w_s >= 1.0) return 1;
    return static_cast<std::size_t>(std::ceil(std::log(1.0 - success_prob) / std::log(1.0 - w_s)));
}

}  // namespace datamunge::algorithms
