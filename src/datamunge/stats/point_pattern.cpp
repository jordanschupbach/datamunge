/// @file point_pattern.cpp
/// @brief Edge-correction weights and pair sums for K, L and the pair correlation function, and
/// Berman-Turner Poisson intensity regression.
#include <datamunge/stats/point_pattern.hpp>

#include <datamunge/linalg/cholesky.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Implementation helpers for point-pattern summaries.
namespace {
/// @brief Checks the points, window and distances.
/// @param p Points.
/// @param w Window.
/// @param r Distances.
void validate(const linalg::DenseMatrix<double>& p, const RectangleWindow& w, const std::vector<double>& r) {
    if (!(w.xmax > w.xmin) || !(w.ymax > w.ymin) || !std::isfinite(w.area()))
        throw std::invalid_argument("point pattern: the window must have positive finite area");
    if (p.rows() < 2 || p.cols() != 2)
        throw std::invalid_argument("point pattern: need an n-by-2 matrix with at least two points");
    for (std::size_t i = 0; i < p.rows(); ++i)
        if (!(p(i, 0) >= w.xmin && p(i, 0) <= w.xmax && p(i, 1) >= w.ymin && p(i, 1) <= w.ymax))
            throw std::invalid_argument("point pattern: all points must lie inside the window");
    if (!std::ranges::all_of(r, [](double v) { return v >= 0 && std::isfinite(v); }))
        throw std::invalid_argument("point pattern: distances must be nonnegative and finite");
}

/// @brief Translation weight |W| / |W cap (W + v)| for a separation v.
/// @param w Window.
/// @param dx Horizontal separation.
/// @param dy Vertical separation.
/// @return The weight.
double translation_weight(const RectangleWindow& w, double dx, double dy) {
    const double a = (w.xmax - w.xmin) - std::abs(dx), b = (w.ymax - w.ymin) - std::abs(dy);
    return w.area() / (a * b);
}

/// @brief Ripley's isotropic weight: inverse fraction of the circle of radius d around (x, y) inside w.
/// @param w Window.
/// @param x Centre x.
/// @param y Centre y.
/// @param d Radius.
/// @return The weight, capped at 100 as in spatstat.
double isotropic_weight(const RectangleWindow& w, double x, double y, double d) {
    if (d <= 0)
        return 1.0;
    const double pi = std::numbers::pi;
    std::vector<std::pair<double, double>> out;
    const auto add = [&](double centre, double dist) {
        if (dist >= d)
            return;
        const double a = std::acos(std::max(-1.0, dist / d));
        double lo = centre - a, hi = centre + a;
        if (lo < 0) {
            out.emplace_back(lo + 2 * pi, 2 * pi);
            lo = 0;
        }
        if (hi > 2 * pi) {
            out.emplace_back(0, hi - 2 * pi);
            hi = 2 * pi;
        }
        out.emplace_back(lo, hi);
    };
    add(0, w.xmax - x);          // right edge, around angle 0
    add(pi / 2, w.ymax - y);     // top
    add(pi, x - w.xmin);         // left
    add(3 * pi / 2, y - w.ymin); // bottom
    std::ranges::sort(out);
    double excluded = 0, cur_lo = -1, cur_hi = -1;
    for (const auto& [lo, hi] : out) {
        if (lo > cur_hi) {
            if (cur_hi > cur_lo)
                excluded += cur_hi - cur_lo;
            cur_lo = lo;
            cur_hi = hi;
        } else {
            cur_hi = std::max(cur_hi, hi);
        }
    }
    if (cur_hi > cur_lo)
        excluded += cur_hi - cur_lo;
    const double inside = 1 - excluded / (2 * pi);
    return inside <= 0.01 ? 100.0 : std::min(100.0, 1 / inside);
}

/// @brief Edge weight of the ordered pair (i, j).
/// @param p Points.
/// @param w Window.
/// @param i Centre point.
/// @param j Other point.
/// @param d Their distance.
/// @param c Correction (none, translation or isotropic).
/// @return The weight.
double pair_weight(const linalg::DenseMatrix<double>& p, const RectangleWindow& w, std::size_t i, std::size_t j,
                   double d, EdgeCorrection c) {
    switch (c) {
    case EdgeCorrection::translation:
        return translation_weight(w, p(i, 0) - p(j, 0), p(i, 1) - p(j, 1));
    case EdgeCorrection::isotropic:
        return isotropic_weight(w, p(i, 0), p(i, 1), d);
    default:
        return 1.0;
    }
}
} // namespace

std::vector<double> ripley_k(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                             const std::vector<double>& r, EdgeCorrection correction) {
    validate(points, window, r);
    const std::size_t n = points.rows();
    const double nd = static_cast<double>(n), area = window.area();
    const double rmax = r.empty() ? 0.0 : *std::ranges::max_element(r);
    std::vector<double> out(r.size(), 0.0);
    if (correction == EdgeCorrection::border) {
        std::vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = std::min({points(i, 0) - window.xmin, window.xmax - points(i, 0), points(i, 1) - window.ymin,
                             window.ymax - points(i, 1)});
        for (std::size_t k = 0; k < r.size(); ++k) {
            double num = 0, centres = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (b[i] < r[k])
                    continue;
                ++centres;
                for (std::size_t j = 0; j < n; ++j)
                    if (j != i && std::hypot(points(i, 0) - points(j, 0), points(i, 1) - points(j, 1)) <= r[k])
                        ++num;
            }
            out[k] = centres == 0 ? std::numeric_limits<double>::quiet_NaN() : num / (nd / area * centres);
        }
        return out;
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i)
                continue;
            const double d = std::hypot(points(i, 0) - points(j, 0), points(i, 1) - points(j, 1));
            if (d > rmax)
                continue;
            const double e = pair_weight(points, window, i, j, d, correction);
            for (std::size_t k = 0; k < r.size(); ++k)
                if (d <= r[k])
                    out[k] += e;
        }
    for (double& v : out)
        v *= area / (nd * (nd - 1));
    return out;
}

std::vector<double> besag_l(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                            const std::vector<double>& r, EdgeCorrection correction) {
    std::vector<double> k = ripley_k(points, window, r, correction);
    for (double& v : k)
        v = std::sqrt(v / std::numbers::pi);
    return k;
}

std::vector<double> pair_correlation(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                                     const std::vector<double>& r, const PairCorrelationOptions& options) {
    validate(points, window, r);
    if (options.correction == EdgeCorrection::border)
        throw std::invalid_argument("pair_correlation: the border correction is not supported");
    const std::size_t n = points.rows();
    const double nd = static_cast<double>(n), area = window.area();
    const double h = std::isnan(options.bandwidth) ? options.stoyan / std::sqrt(5 * nd / area) : options.bandwidth;
    if (!(h > 0) || !std::isfinite(h))
        throw std::invalid_argument("pair_correlation: the bandwidth must be positive");
    const double half = std::sqrt(5.0) * h; // Epanechnikov half-width with standard deviation h
    const double rmax = r.empty() ? 0.0 : *std::ranges::max_element(r);
    std::vector<double> out(r.size(), 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i)
                continue;
            const double d = std::hypot(points(i, 0) - points(j, 0), points(i, 1) - points(j, 1));
            if (d >= rmax + half)
                continue;
            const double e = pair_weight(points, window, i, j, d, options.correction);
            for (std::size_t k = 0; k < r.size(); ++k) {
                const double u = (r[k] - d) / half;
                if (std::abs(u) >= 1)
                    continue;
                const double kern = 0.75 / half * (1 - u * u);
                out[k] += kern * e / (options.divide_by_distance ? 2 * std::numbers::pi * d : 1.0);
            }
        }
    for (std::size_t k = 0; k < r.size(); ++k) {
        out[k] *= area / (nd * (nd - 1));
        if (!options.divide_by_distance)
            out[k] = r[k] > 0 ? out[k] / (2 * std::numbers::pi * r[k]) : std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

QuadratureScheme berman_turner_quadrature(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                                          std::size_t nx, std::size_t ny) {
    if (!(window.xmax > window.xmin) || !(window.ymax > window.ymin) || !std::isfinite(window.area()))
        throw std::invalid_argument("berman_turner_quadrature: the window must have positive finite area");
    if (points.cols() != 2 && points.rows() > 0)
        throw std::invalid_argument("berman_turner_quadrature: points must be n-by-2");
    const std::size_t n = points.rows();
    for (std::size_t i = 0; i < n; ++i)
        if (!(points(i, 0) >= window.xmin && points(i, 0) <= window.xmax && points(i, 1) >= window.ymin &&
              points(i, 1) <= window.ymax))
            throw std::invalid_argument("berman_turner_quadrature: all points must lie inside the window");
    if ((nx == 0) != (ny == 0))
        throw std::invalid_argument("berman_turner_quadrature: set both grid sizes or neither");
    if (nx == 0) {
        const auto rule = static_cast<std::size_t>(10 * std::ceil(2 * std::sqrt(static_cast<double>(n)) / 10));
        nx = ny = std::max<std::size_t>(32, rule);
    }
    const double dx = (window.xmax - window.xmin) / static_cast<double>(nx);
    const double dy = (window.ymax - window.ymin) / static_cast<double>(ny);
    const std::size_t m = n + nx * ny + 4;
    QuadratureScheme q;
    q.points = linalg::DenseMatrix<double>(m, 2);
    q.data_points = n;
    std::size_t row = 0;
    for (std::size_t i = 0; i < n; ++i, ++row)
        q.points(row, 0) = points(i, 0), q.points(row, 1) = points(i, 1);
    for (std::size_t b = 0; b < ny; ++b)
        for (std::size_t a = 0; a < nx; ++a, ++row) {
            q.points(row, 0) = window.xmin + (static_cast<double>(a) + 0.5) * dx;
            q.points(row, 1) = window.ymin + (static_cast<double>(b) + 0.5) * dy;
        }
    for (const double cy : {window.ymin, window.ymax})
        for (const double cx : {window.xmin, window.xmax}) {
            q.points(row, 0) = cx;
            q.points(row, 1) = cy;
            ++row;
        }
    const auto tile = [&](std::size_t r) {
        const auto a = std::min(nx - 1, static_cast<std::size_t>((q.points(r, 0) - window.xmin) / dx));
        const auto b = std::min(ny - 1, static_cast<std::size_t>((q.points(r, 1) - window.ymin) / dy));
        return b * nx + a;
    };
    std::vector<std::size_t> count(nx * ny, 0);
    for (std::size_t r = 0; r < m; ++r)
        ++count[tile(r)];
    q.weights.resize(m);
    for (std::size_t r = 0; r < m; ++r)
        q.weights[r] = dx * dy / static_cast<double>(count[tile(r)]);
    return q;
}

PoissonProcessFit poisson_process_regression(const linalg::DenseMatrix<double>& points, const RectangleWindow& window,
                                             const PointCovariates& covariates, const PoissonProcessOptions& options) {
    if (points.rows() == 0)
        throw std::invalid_argument("poisson_process_regression: need at least one point");
    const QuadratureScheme q = berman_turner_quadrature(points, window, options.grid_x, options.grid_y);
    const std::size_t m = q.points.rows(), n = q.data_points;
    std::vector<std::vector<double>> z(m);
    for (std::size_t r = 0; r < m; ++r) {
        z[r] = covariates(q.points(r, 0), q.points(r, 1));
        if (z[r].empty() || z[r].size() != z[0].size() ||
            !std::ranges::all_of(z[r], [](double v) { return std::isfinite(v); }))
            throw std::invalid_argument("poisson_process_regression: covariate rows must be finite and equal length");
    }
    const std::size_t p = z[0].size();
    std::vector<double> data_sum(p, 0.0);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t k = 0; k < p; ++k)
            data_sum[k] += z[r][k];
    const auto eta = [&](const std::vector<double>& b, std::size_t r) {
        double s = 0;
        for (std::size_t k = 0; k < p; ++k)
            s += b[k] * z[r][k];
        return s;
    };
    const auto loglik = [&](const std::vector<double>& b) {
        double l = 0;
        for (std::size_t k = 0; k < p; ++k)
            l += b[k] * data_sum[k];
        for (std::size_t r = 0; r < m; ++r)
            l -= q.weights[r] * std::exp(eta(b, r));
        return l;
    };
    const auto information = [&](const std::vector<double>& b, std::vector<double>* grad) {
        linalg::DenseMatrix<double> info(p, p);
        if (grad)
            *grad = data_sum;
        for (std::size_t r = 0; r < m; ++r) {
            const double wl = q.weights[r] * std::exp(eta(b, r));
            for (std::size_t a = 0; a < p; ++a) {
                if (grad)
                    (*grad)[a] -= wl * z[r][a];
                for (std::size_t c = 0; c <= a; ++c)
                    info(a, c) += wl * z[r][a] * z[r][c];
            }
        }
        for (std::size_t a = 0; a < p; ++a)
            for (std::size_t c = 0; c < a; ++c)
                info(c, a) = info(a, c);
        return info;
    };
    PoissonProcessFit fit;
    std::vector<double> beta(p, 0.0);
    double current = loglik(beta);
    for (std::size_t it = 0; it < options.max_iterations; ++it) {
        std::vector<double> grad;
        const auto chol = linalg::cholesky(information(beta, &grad));
        if (!chol.ok)
            throw std::runtime_error("poisson_process_regression: the Fisher information is singular");
        const std::vector<double> step = chol.solve(grad);
        double scale = 1, next = current;
        std::vector<double> trial(p);
        for (int halve = 0; halve < 60; ++halve, scale /= 2) {
            for (std::size_t k = 0; k < p; ++k)
                trial[k] = beta[k] + scale * step[k];
            next = loglik(trial);
            if (std::isfinite(next) && next >= current - 1e-12 * std::abs(current))
                break;
        }
        beta = trial;
        fit.iterations = it + 1;
        const double change = std::abs(next - current);
        current = next;
        if (change <= options.tolerance * (1 + std::abs(current))) {
            fit.converged = true;
            break;
        }
    }
    fit.coefficients = beta;
    fit.log_likelihood = current;
    fit.aic = -2 * current + 2 * static_cast<double>(p);
    const auto info = information(beta, nullptr);
    const auto chol = linalg::cholesky(info);
    if (!chol.ok)
        throw std::runtime_error("poisson_process_regression: the Fisher information is singular");
    fit.covariance = linalg::DenseMatrix<double>(p, p);
    std::vector<double> e(p, 0.0);
    for (std::size_t c = 0; c < p; ++c) {
        e[c] = 1;
        const auto col = chol.solve(e);
        e[c] = 0;
        for (std::size_t a = 0; a < p; ++a)
            fit.covariance(a, c) = col[a];
    }
    for (std::size_t k = 0; k < p; ++k)
        fit.standard_errors.push_back(std::sqrt(fit.covariance(k, k)));
    for (std::size_t r = 0; r < m; ++r)
        fit.expected_count += q.weights[r] * std::exp(eta(beta, r));
    return fit;
}

double poisson_process_intensity(const PoissonProcessFit& fit, const PointCovariates& covariates, double x, double y) {
    const std::vector<double> z = covariates(x, y);
    if (z.size() != fit.coefficients.size())
        throw std::invalid_argument("poisson_process_intensity: covariate row length does not match the fit");
    double s = 0;
    for (std::size_t k = 0; k < z.size(); ++k)
        s += fit.coefficients[k] * z[k];
    return std::exp(s);
}

} // namespace datamunge::stats
