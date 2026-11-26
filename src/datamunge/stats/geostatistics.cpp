/// @file geostatistics.cpp
/// @brief Lag binning, semivariance estimators, variogram models, their least-squares fit, and kriging.
#include <datamunge/stats/geostatistics.hpp>

#include <datamunge/linalg/cholesky.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace datamunge::stats {

std::vector<VariogramBin> empirical_variogram(const linalg::DenseMatrix<double>& coordinates,
                                              const std::vector<double>& values, const VariogramOptions& options) {
    const std::size_t n = coordinates.rows(), d = coordinates.cols();
    if (n < 2 || values.size() != n || d == 0)
        throw std::invalid_argument("empirical_variogram: need at least two located values");
    if (!std::ranges::all_of(values, [](double v) { return std::isfinite(v); }) ||
        !std::ranges::all_of(coordinates.data(), [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("empirical_variogram: inputs must be finite");
    if (!options.directions.empty() && d != 2)
        throw std::invalid_argument("empirical_variogram: directional variograms need 2-D coordinates");
    std::vector<double> bounds = options.boundaries;
    if (bounds.empty()) {
        double cutoff = options.cutoff;
        if (std::isnan(cutoff)) {
            double diag = 0;
            for (std::size_t a = 0; a < d; ++a) {
                double lo = coordinates(0, a), hi = lo;
                for (std::size_t i = 1; i < n; ++i) {
                    lo = std::min(lo, coordinates(i, a));
                    hi = std::max(hi, coordinates(i, a));
                }
                diag += (hi - lo) * (hi - lo);
            }
            cutoff = std::sqrt(diag) / 3;
        }
        const double width = std::isnan(options.width) ? cutoff / 15 : options.width;
        if (!(cutoff > 0) || !(width > 0) || !std::isfinite(cutoff))
            throw std::invalid_argument("empirical_variogram: cutoff and width must be positive");
        const auto count = static_cast<std::size_t>(std::ceil(cutoff / width - 1e-9));
        for (std::size_t k = 0; k <= count; ++k)
            bounds.push_back(std::min(cutoff, static_cast<double>(k) * width));
    }
    if (bounds.size() < 2)
        throw std::invalid_argument("empirical_variogram: need at least two boundaries");
    for (std::size_t k = 1; k < bounds.size(); ++k)
        if (!(bounds[k] > bounds[k - 1]))
            throw std::invalid_argument("empirical_variogram: boundaries must increase");
    std::vector<double> dirs = options.directions;
    const bool isotropic = dirs.empty();
    if (isotropic)
        dirs = {0.0};
    const double tol = std::isnan(options.angular_tolerance) ? 90.0 / static_cast<double>(dirs.size())
                                                            : options.angular_tolerance;
    if (!isotropic && !(tol > 0 && tol <= 90))
        throw std::invalid_argument("empirical_variogram: angular tolerance must be in (0, 90]");
    const std::size_t nb = bounds.size() - 1;
    std::vector<VariogramBin> out;
    for (const double dir : dirs) {
        std::vector<double> sum_d(nb, 0.0), sum_g(nb, 0.0);
        std::vector<std::size_t> np(nb, 0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) {
                double h2 = 0;
                for (std::size_t a = 0; a < d; ++a)
                    h2 += (coordinates(i, a) - coordinates(j, a)) * (coordinates(i, a) - coordinates(j, a));
                const double h = std::sqrt(h2);
                if (!(h > bounds.front()) || h > bounds.back())
                    continue;
                if (!isotropic) {
                    const double dx = coordinates(j, 0) - coordinates(i, 0), dy = coordinates(j, 1) - coordinates(i, 1);
                    double az = std::atan2(dx, dy) * 180 / std::numbers::pi; // clockwise from north
                    az = std::fmod(az + 360.0, 180.0);
                    double diff = std::abs(az - std::fmod(std::fmod(dir, 180.0) + 180.0, 180.0));
                    diff = std::min(diff, 180 - diff);
                    if (diff > tol + 1e-12)
                        continue;
                }
                const auto it = std::lower_bound(bounds.begin() + 1, bounds.end(), h);
                const auto k = static_cast<std::size_t>(it - bounds.begin()) - 1;
                const double diff = values[i] - values[j];
                sum_d[k] += h;
                sum_g[k] += options.cressie ? std::sqrt(std::abs(diff)) : diff * diff;
                ++np[k];
            }
        for (std::size_t k = 0; k < nb; ++k) {
            if (np[k] == 0)
                continue;
            VariogramBin b;
            const double m = static_cast<double>(np[k]);
            b.distance = sum_d[k] / m;
            b.pairs = np[k];
            b.gamma = options.cressie ? std::pow(sum_g[k] / m, 4) / (2 * (0.457 + 0.494 / m)) : sum_g[k] / (2 * m);
            b.direction = isotropic ? std::numeric_limits<double>::quiet_NaN() : dir;
            b.lower = bounds[k];
            b.upper = bounds[k + 1];
            out.push_back(b);
        }
    }
    return out;
}

/// @brief Implementation helpers for variogram models and fitting.
namespace {
/// @brief Checks the parameters of a variogram model.
/// @param m The model.
void validate_model(const VariogramModel& m) {
    if (!(m.nugget >= 0) || !(m.partial_sill >= 0) || !(m.range > 0) || !std::isfinite(m.range) ||
        (m.kind == VariogramModelKind::matern && !(m.kappa > 0)))
        throw std::invalid_argument("variogram model: need nonnegative sills, a positive range and kappa");
}

/// @brief Unit-sill shape of the structured component at lag h > 0.
/// @param kind Shape.
/// @param h Lag.
/// @param range Range or scale.
/// @param kappa Matern smoothness.
/// @return The shape value in [0, 1].
double unit_shape(VariogramModelKind kind, double h, double range, double kappa) {
    const double u = h / range;
    switch (kind) {
    case VariogramModelKind::spherical:
        return u >= 1 ? 1.0 : 1.5 * u - 0.5 * u * u * u;
    case VariogramModelKind::exponential:
        return 1 - std::exp(-u);
    case VariogramModelKind::gaussian:
        return 1 - std::exp(-u * u);
    case VariogramModelKind::matern: {
        if (u > 700)
            return 1.0;
        const double c = std::pow(u, kappa) * std::cyl_bessel_k(kappa, u) /
                         (std::pow(2.0, kappa - 1) * std::tgamma(kappa));
        return std::clamp(1 - c, 0.0, 1.0);
    }
    }
    return 1.0;
}

/// @brief Sills and weighted SSE at a fixed range.
struct SillFit {
    double nugget{};       ///< Nugget.
    double partial_sill{}; ///< Partial sill.
    double sse{std::numeric_limits<double>::infinity()}; ///< Weighted SSE.
};

/// @brief Nonnegative weighted least squares for y = c0 + c1 a, with c0 optionally fixed.
/// @param a Shape values.
/// @param y Empirical semivariances.
/// @param w Weights.
/// @param fit_nugget Whether c0 is estimated.
/// @param fixed_nugget c0 when it is not estimated.
/// @return The best feasible sills.
SillFit nonnegative_sills(const std::vector<double>& a, const std::vector<double>& y, const std::vector<double>& w,
                          bool fit_nugget, double fixed_nugget) {
    const auto sse = [&](double c0, double c1) {
        double s = 0;
        for (std::size_t j = 0; j < y.size(); ++j)
            s += w[j] * (y[j] - c0 - c1 * a[j]) * (y[j] - c0 - c1 * a[j]);
        return s;
    };
    SillFit best;
    const auto consider = [&](double c0, double c1) {
        if (!(c0 >= 0) || !(c1 >= 0) || !std::isfinite(c0) || !std::isfinite(c1))
            return;
        const double s = sse(c0, c1);
        if (s < best.sse)
            best = {c0, c1, s};
    };
    double sw = 0, sa = 0, sy = 0, saa = 0, say = 0;
    for (std::size_t j = 0; j < y.size(); ++j) {
        sw += w[j];
        sa += w[j] * a[j];
        sy += w[j] * y[j];
        saa += w[j] * a[j] * a[j];
        say += w[j] * a[j] * y[j];
    }
    if (!fit_nugget) {
        consider(fixed_nugget, saa > 0 ? std::max(0.0, (say - fixed_nugget * sa) / saa) : 0.0);
        return best;
    }
    const double det = sw * saa - sa * sa;
    if (det > 1e-12 * sw * saa)
        consider((saa * sy - sa * say) / det, (sw * say - sa * sy) / det);
    consider(0.0, saa > 0 ? std::max(0.0, say / saa) : 0.0);
    consider(std::max(0.0, sy / sw), 0.0);
    return best;
}

/// @brief Profile fit over the range for fixed weights.
/// @param h Bin distances.
/// @param y Empirical semivariances.
/// @param w Weights.
/// @param initial Shape, kappa and fixed values.
/// @param options Which parameters are fitted.
/// @return Fitted model and weighted SSE.
std::pair<VariogramModel, double> profile_fit(const std::vector<double>& h, const std::vector<double>& y,
                                              const std::vector<double>& w, const VariogramModel& initial,
                                              const VariogramFitOptions& options) {
    std::vector<double> a(h.size());
    const auto at = [&](double range) {
        for (std::size_t j = 0; j < h.size(); ++j)
            a[j] = unit_shape(initial.kind, h[j], range, initial.kappa);
        return nonnegative_sills(a, y, w, options.fit_nugget, initial.nugget);
    };
    double range = initial.range;
    if (options.fit_range) {
        const double lo = std::log(*std::ranges::min_element(h) / 10), hi = std::log(10 * *std::ranges::max_element(h));
        constexpr int grid = 400;
        std::size_t best_k = 0;
        double best = std::numeric_limits<double>::infinity();
        std::vector<double> lr(grid);
        for (int k = 0; k < grid; ++k) {
            lr[k] = lo + (hi - lo) * k / (grid - 1);
            const double s = at(std::exp(lr[k])).sse;
            if (s < best)
                best = s, best_k = static_cast<std::size_t>(k);
        }
        double left = lr[best_k == 0 ? 0 : best_k - 1], right = lr[std::min<std::size_t>(grid - 1, best_k + 1)];
        const double g = (std::sqrt(5.0) - 1) / 2;
        double x1 = right - g * (right - left), x2 = left + g * (right - left);
        double f1 = at(std::exp(x1)).sse, f2 = at(std::exp(x2)).sse;
        while (right - left > 1e-13) {
            if (f1 <= f2) {
                right = x2, x2 = x1, f2 = f1;
                x1 = right - g * (right - left);
                f1 = at(std::exp(x1)).sse;
            } else {
                left = x1, x1 = x2, f1 = f2;
                x2 = left + g * (right - left);
                f2 = at(std::exp(x2)).sse;
            }
        }
        range = std::exp(f1 <= f2 ? x1 : x2);
        if (at(range).sse > best)
            range = std::exp(lr[best_k]);
    }
    const SillFit s = at(range);
    VariogramModel m = initial;
    m.nugget = s.nugget;
    m.partial_sill = s.partial_sill;
    m.range = range;
    return {m, s.sse};
}
} // namespace

double variogram_model_value(const VariogramModel& model, double h) {
    validate_model(model);
    if (!(h >= 0))
        throw std::invalid_argument("variogram_model_value: lag must be nonnegative");
    if (h == 0)
        return 0.0;
    return model.nugget + model.partial_sill * unit_shape(model.kind, h, model.range, model.kappa);
}

double variogram_model_covariance(const VariogramModel& model, double h) {
    validate_model(model);
    if (!(h >= 0))
        throw std::invalid_argument("variogram_model_covariance: lag must be nonnegative");
    if (h == 0)
        return model.nugget + model.partial_sill;
    return model.partial_sill * (1 - unit_shape(model.kind, h, model.range, model.kappa));
}

VariogramFit fit_variogram(const std::vector<VariogramBin>& bins, const VariogramModel& initial,
                           const VariogramFitOptions& options) {
    validate_model(initial);
    if (bins.size() < 2)
        throw std::invalid_argument("fit_variogram: need at least two bins");
    std::vector<double> h, y, np;
    for (const auto& b : bins) {
        if (!(b.distance > 0) || !std::isfinite(b.distance) || !std::isfinite(b.gamma) || b.pairs == 0)
            throw std::invalid_argument("fit_variogram: bins need positive distances, finite gamma and pairs");
        h.push_back(b.distance);
        y.push_back(b.gamma);
        np.push_back(static_cast<double>(b.pairs));
    }
    std::vector<double> w(h.size());
    const auto set_weights = [&](const VariogramModel& m) {
        for (std::size_t j = 0; j < h.size(); ++j) {
            switch (options.weighting) {
            case VariogramWeighting::pairs:
                w[j] = np[j];
                break;
            case VariogramWeighting::uniform:
                w[j] = 1;
                break;
            case VariogramWeighting::pairs_over_distance_squared:
                w[j] = np[j] / (h[j] * h[j]);
                break;
            case VariogramWeighting::cressie: {
                const double g = variogram_model_value(m, h[j]);
                w[j] = g > 0 ? np[j] / (g * g) : np[j] / 1e-300;
                break;
            }
            }
        }
    };
    VariogramFit out;
    set_weights(initial);
    auto [model, sse] = profile_fit(h, y, w, initial, options);
    out.iterations = 1;
    if (options.weighting == VariogramWeighting::cressie) {
        out.converged = false;
        for (std::size_t it = 1; it < options.max_iterations; ++it) {
            set_weights(model);
            auto [next, next_sse] = profile_fit(h, y, w, initial, options);
            ++out.iterations;
            const double change = std::abs(next.nugget - model.nugget) + std::abs(next.partial_sill - model.partial_sill) +
                                  std::abs(next.range - model.range) / next.range;
            model = next;
            sse = next_sse;
            if (change < 1e-9) {
                out.converged = true;
                break;
            }
        }
    }
    out.model = model;
    out.weighted_sse = sse;
    return out;
}

/// @brief Implementation helpers for kriging.
namespace {
/// @brief Euclidean distance between row i of a and row j of b.
/// @param a First coordinate matrix.
/// @param i Row of a.
/// @param b Second coordinate matrix.
/// @param j Row of b.
/// @return The distance.
double row_distance(const linalg::DenseMatrix<double>& a, std::size_t i, const linalg::DenseMatrix<double>& b,
                    std::size_t j) {
    double s = 0;
    for (std::size_t k = 0; k < a.cols(); ++k)
        s += (a(i, k) - b(j, k)) * (a(i, k) - b(j, k));
    return std::sqrt(s);
}

/// @brief A factored kriging system on a subset of observations.
struct KrigingSystem {
    std::vector<std::size_t> index;          ///< Observations in the system.
    linalg::CholeskyDecomposition<double> c; ///< Factor of the covariance matrix among them.
    linalg::DenseMatrix<double> cix;         ///< C^{-1} X (k-by-p).
    linalg::CholeskyDecomposition<double> g; ///< Factor of X' C^{-1} X.
    std::vector<double> beta;                ///< GLS trend coefficients.
    std::vector<double> residual_weights;    ///< C^{-1}(z - X beta) or C^{-1}(z - mean).
    double mean{};                           ///< Known mean (simple kriging).
    std::size_t p{};                         ///< Trend columns (0 for simple kriging).
};

/// @brief Builds and factors the kriging system for the given observations.
/// @param xy Observation coordinates.
/// @param z Observations.
/// @param x Trend design (p = 0 columns for simple kriging).
/// @param index Observations to use.
/// @param model Variogram model.
/// @param mean Known mean for simple kriging.
/// @return The factored system.
KrigingSystem build_system(const linalg::DenseMatrix<double>& xy, const std::vector<double>& z,
                           const linalg::DenseMatrix<double>& x, std::vector<std::size_t> index,
                           const VariogramModel& model, double mean) {
    KrigingSystem s;
    s.index = std::move(index);
    s.p = x.cols();
    s.mean = mean;
    const std::size_t k = s.index.size();
    linalg::DenseMatrix<double> cov(k, k);
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t b = 0; b <= a; ++b)
            cov(a, b) = cov(b, a) = variogram_model_covariance(model, row_distance(xy, s.index[a], xy, s.index[b]));
    s.c = linalg::cholesky(cov);
    if (!s.c.ok)
        throw std::runtime_error("kriging: covariance matrix is not positive definite (duplicate locations?)");
    std::vector<double> zz(k);
    for (std::size_t a = 0; a < k; ++a)
        zz[a] = z[s.index[a]];
    if (s.p == 0) {
        for (double& v : zz)
            v -= mean;
        s.residual_weights = s.c.solve(zz);
        return s;
    }
    linalg::DenseMatrix<double> xs(k, s.p);
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t j = 0; j < s.p; ++j)
            xs(a, j) = x(s.index[a], j);
    s.cix = s.c.solve(xs);
    linalg::DenseMatrix<double> gm(s.p, s.p);
    std::vector<double> rhs(s.p, 0.0);
    for (std::size_t i = 0; i < s.p; ++i) {
        for (std::size_t j = 0; j < s.p; ++j) {
            double v = 0;
            for (std::size_t a = 0; a < k; ++a)
                v += xs(a, i) * s.cix(a, j);
            gm(i, j) = v;
        }
        for (std::size_t a = 0; a < k; ++a)
            rhs[i] += s.cix(a, i) * zz[a];
    }
    s.g = linalg::cholesky(gm);
    if (!s.g.ok)
        throw std::runtime_error("kriging: the trend design is singular within the neighbourhood");
    s.beta = s.g.solve(rhs);
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t j = 0; j < s.p; ++j)
            zz[a] -= xs(a, j) * s.beta[j];
    s.residual_weights = s.c.solve(zz);
    return s;
}

/// @brief Prediction and variance from a factored system.
/// @param s The system.
/// @param xy Observation coordinates.
/// @param target Target coordinates.
/// @param t Target row.
/// @param x0 Trend design at the target (size p).
/// @param model Variogram model.
/// @return Prediction and variance.
std::pair<double, double> predict(const KrigingSystem& s, const linalg::DenseMatrix<double>& xy,
                                  const linalg::DenseMatrix<double>& target, std::size_t t,
                                  const std::vector<double>& x0, const VariogramModel& model) {
    const std::size_t k = s.index.size();
    std::vector<double> c0(k);
    for (std::size_t a = 0; a < k; ++a)
        c0[a] = variogram_model_covariance(model, row_distance(target, t, xy, s.index[a]));
    double pred = s.p == 0 ? s.mean : 0.0;
    for (std::size_t j = 0; j < s.p; ++j)
        pred += x0[j] * s.beta[j];
    for (std::size_t a = 0; a < k; ++a)
        pred += c0[a] * s.residual_weights[a];
    const std::vector<double> cic0 = s.c.solve(c0);
    double var = variogram_model_covariance(model, 0.0);
    for (std::size_t a = 0; a < k; ++a)
        var -= c0[a] * cic0[a];
    if (s.p > 0) {
        std::vector<double> u(x0);
        for (std::size_t j = 0; j < s.p; ++j)
            for (std::size_t a = 0; a < k; ++a)
                u[j] -= s.cix(a, j) * c0[a];
        const std::vector<double> gu = s.g.solve(u);
        for (std::size_t j = 0; j < s.p; ++j)
            var += u[j] * gu[j];
    }
    return {pred, std::max(0.0, var)};
}

/// @brief Nearest observations to a target, within the distance limit and excluding one index.
/// @param xy Observation coordinates.
/// @param target Target coordinates.
/// @param t Target row.
/// @param options Neighbourhood.
/// @param exclude Observation to leave out (n for none).
/// @return Selected observation indices.
std::vector<std::size_t> neighbours(const linalg::DenseMatrix<double>& xy, const linalg::DenseMatrix<double>& target,
                                    std::size_t t, const KrigingOptions& options, std::size_t exclude) {
    std::vector<std::pair<double, std::size_t>> d;
    for (std::size_t i = 0; i < xy.rows(); ++i) {
        if (i == exclude)
            continue;
        const double h = row_distance(target, t, xy, i);
        if (h <= options.max_distance)
            d.emplace_back(h, i);
    }
    const std::size_t keep =
        options.max_neighbours == 0 ? d.size() : std::min(d.size(), options.max_neighbours);
    std::partial_sort(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(keep), d.end());
    std::vector<std::size_t> out;
    for (std::size_t a = 0; a < keep; ++a)
        out.push_back(d[a].second);
    std::ranges::sort(out);
    return out;
}

/// @brief Shared driver for simple (p = 0), ordinary and universal kriging.
/// @param xy Observation coordinates.
/// @param z Observations.
/// @param x Trend design at the observations.
/// @param targets Target coordinates.
/// @param x_targets Trend design at the targets.
/// @param model Variogram model.
/// @param mean Known mean (simple kriging).
/// @param options Neighbourhood.
/// @return Predictions, variances and global trend coefficients.
KrigingResult krige(const linalg::DenseMatrix<double>& xy, const std::vector<double>& z,
                    const linalg::DenseMatrix<double>& x, const linalg::DenseMatrix<double>& targets,
                    const linalg::DenseMatrix<double>& x_targets, const VariogramModel& model, double mean,
                    const KrigingOptions& options) {
    validate_model(model);
    const std::size_t n = xy.rows(), m = targets.rows(), p = x.cols();
    if (n == 0 || z.size() != n || targets.cols() != xy.cols() || xy.cols() == 0)
        throw std::invalid_argument("kriging: observations, values and targets must match");
    if ((p > 0 && x.rows() != n) || x_targets.cols() != p || (p > 0 && x_targets.rows() != m))
        throw std::invalid_argument("kriging: trend designs must match the observations and targets");
    const auto finite = [](const linalg::DenseMatrix<double>& a) {
        return std::ranges::all_of(a.data(), [](double v) { return std::isfinite(v); });
    };
    if (!std::ranges::all_of(z, [](double v) { return std::isfinite(v); }) || !finite(xy) || !finite(targets) ||
        !finite(x) || !finite(x_targets) || !std::isfinite(mean))
        throw std::invalid_argument("kriging: inputs must be finite");
    if (!(options.max_distance > 0))
        throw std::invalid_argument("kriging: max_distance must be positive");
    std::vector<std::size_t> all(n);
    std::iota(all.begin(), all.end(), std::size_t{0});
    KrigingResult out;
    out.prediction.assign(m, std::numeric_limits<double>::quiet_NaN());
    out.variance = out.prediction;
    const bool local = (options.max_neighbours > 0 && options.max_neighbours < n) || std::isfinite(options.max_distance);
    std::vector<double> x0(p);
    const auto trend_row = [&](std::size_t t) {
        for (std::size_t j = 0; j < p; ++j)
            x0[j] = x_targets(t, j);
    };
    if (p > 0 && n >= p) {
        const KrigingSystem global = build_system(xy, z, x, all, model, mean);
        out.trend_coefficients = global.beta;
        if (!local)
            for (std::size_t t = 0; t < m; ++t) {
                trend_row(t);
                std::tie(out.prediction[t], out.variance[t]) = predict(global, xy, targets, t, x0, model);
            }
    } else if (!local) {
        const KrigingSystem global = build_system(xy, z, x, all, model, mean);
        for (std::size_t t = 0; t < m; ++t)
            std::tie(out.prediction[t], out.variance[t]) = predict(global, xy, targets, t, x0, model);
    }
    if (local)
        for (std::size_t t = 0; t < m; ++t) {
            auto idx = neighbours(xy, targets, t, options, n);
            if (idx.size() < std::max<std::size_t>(p, 1))
                continue;
            trend_row(t);
            const KrigingSystem s = build_system(xy, z, x, std::move(idx), model, mean);
            std::tie(out.prediction[t], out.variance[t]) = predict(s, xy, targets, t, x0, model);
        }
    return out;
}

/// @brief An n-by-1 matrix of ones (the ordinary-kriging trend).
/// @param n Rows.
/// @return The intercept design.
linalg::DenseMatrix<double> ones(std::size_t n) { return linalg::DenseMatrix<double>(n, 1, std::vector<double>(n, 1.0)); }
} // namespace

KrigingResult simple_kriging(const linalg::DenseMatrix<double>& coordinates, const std::vector<double>& values,
                             const linalg::DenseMatrix<double>& targets, const VariogramModel& model, double mean,
                             const KrigingOptions& options) {
    const linalg::DenseMatrix<double> none(coordinates.rows(), 0), none_t(targets.rows(), 0);
    return krige(coordinates, values, none, targets, none_t, model, mean, options);
}

KrigingResult ordinary_kriging(const linalg::DenseMatrix<double>& coordinates, const std::vector<double>& values,
                               const linalg::DenseMatrix<double>& targets, const VariogramModel& model,
                               const KrigingOptions& options) {
    return krige(coordinates, values, ones(coordinates.rows()), targets, ones(targets.rows()), model, 0.0, options);
}

KrigingResult universal_kriging(const linalg::DenseMatrix<double>& coordinates, const std::vector<double>& values,
                                const linalg::DenseMatrix<double>& trend, const linalg::DenseMatrix<double>& targets,
                                const linalg::DenseMatrix<double>& target_trend, const VariogramModel& model,
                                const KrigingOptions& options) {
    if (trend.cols() == 0)
        throw std::invalid_argument("universal_kriging: the trend needs at least one column");
    return krige(coordinates, values, trend, targets, target_trend, model, 0.0, options);
}

KrigingCrossValidation kriging_cross_validation(const linalg::DenseMatrix<double>& coordinates,
                                                const std::vector<double>& values, const VariogramModel& model,
                                                const linalg::DenseMatrix<double>& trend,
                                                const KrigingOptions& options) {
    const std::size_t n = coordinates.rows();
    if (n < 3 || values.size() != n)
        throw std::invalid_argument("kriging_cross_validation: need three or more observations");
    const linalg::DenseMatrix<double> x = trend.rows() == 0 ? ones(n) : trend;
    if (x.rows() != n || x.cols() == 0)
        throw std::invalid_argument("kriging_cross_validation: trend rows must match the observations");
    KrigingCrossValidation cv;
    const std::size_t p = x.cols(), d = coordinates.cols();
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::size_t> keep;
        for (std::size_t j = 0; j < n; ++j)
            if (j != i)
                keep.push_back(j);
        linalg::DenseMatrix<double> xy(n - 1, d), xs(n - 1, p), target(1, d), x0(1, p);
        std::vector<double> z(n - 1);
        for (std::size_t a = 0; a < n - 1; ++a) {
            for (std::size_t k = 0; k < d; ++k)
                xy(a, k) = coordinates(keep[a], k);
            for (std::size_t k = 0; k < p; ++k)
                xs(a, k) = x(keep[a], k);
            z[a] = values[keep[a]];
        }
        for (std::size_t k = 0; k < d; ++k)
            target(0, k) = coordinates(i, k);
        for (std::size_t k = 0; k < p; ++k)
            x0(0, k) = x(i, k);
        const KrigingResult r = krige(xy, z, xs, target, x0, model, 0.0, options);
        const double res = values[i] - r.prediction[0];
        cv.prediction.push_back(r.prediction[0]);
        cv.variance.push_back(r.variance[0]);
        cv.residual.push_back(res);
        cv.zscore.push_back(res / std::sqrt(r.variance[0]));
    }
    double me = 0, mse = 0, mz = 0;
    std::size_t used = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(cv.residual[i]))
            continue;
        ++used;
        me += cv.residual[i];
        mse += cv.residual[i] * cv.residual[i];
        mz += cv.zscore[i] * cv.zscore[i];
    }
    const double u = static_cast<double>(used);
    cv.mean_error = me / u;
    cv.rmse = std::sqrt(mse / u);
    cv.mean_squared_zscore = mz / u;
    return cv;
}

} // namespace datamunge::stats
