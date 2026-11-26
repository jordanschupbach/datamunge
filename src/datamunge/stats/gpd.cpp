/// @file gpd.cpp
/// @brief GPD likelihood, Newton fitting, return levels with profile likelihood, and threshold diagnostics.
#include <datamunge/stats/gpd.hpp>

#include "extreme_detail.hpp"

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for GPD fitting.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;
/// @brief Objective alias from the shared extreme-value helpers.
using detail::Objective;

/// @brief Quiet NaN.
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/// @brief \f$\log(1+\xi y/\sigma)/\xi\f$ continuous at 0 (value y/sigma); NaN outside the support.
/// @param xi Shape.
/// @param z y / sigma.
/// @return Reduced log.
double reduced_log(double xi, double z) {
    if (xi == 0)
        return z;
    const double a = xi * z;
    if (a <= -1)
        return nan;
    return std::abs(a) < 1e-8 ? z * (1 - a / 2 + a * a / 3) : std::log1p(a) / xi;
}

/// @brief GPD log-likelihood of excesses.
/// @param y Excesses.
/// @param sigma Scale.
/// @param xi Shape.
/// @return Log-likelihood.
double log_likelihood(const std::vector<double>& y, double sigma, double xi) {
    if (!(sigma > 0) || !std::isfinite(xi))
        return -std::numeric_limits<double>::infinity();
    double ll = -static_cast<double>(y.size()) * std::log(sigma);
    for (const double v : y) {
        const double r = reduced_log(xi, v / sigma);
        if (std::isnan(r))
            return -std::numeric_limits<double>::infinity();
        ll -= (xi + 1) * r;
    }
    return ll;
}

/// @brief \f$[(m\zeta)^\xi-1]/\xi\f$, continuous at 0 (\f$\log m\zeta\f$).
/// @param xi Shape.
/// @param lmz \f$\log(m\zeta)\f$.
/// @return Factor.
double factor(double xi, double lmz) { return std::abs(xi) < 1e-10 ? lmz : std::expm1(xi * lmz) / xi; }

/// @brief Excesses of data over u.
/// @param data Observations.
/// @param u Threshold.
/// @return Excesses.
std::vector<double> excesses(const std::vector<double>& data, double u) {
    std::vector<double> y;
    for (const double v : data)
        if (v > u)
            y.push_back(v - u);
    return y;
}
} // namespace

double gpd_cdf(double y, double scale, double shape) {
    if (!(scale > 0))
        throw std::invalid_argument("gpd_cdf: scale must be positive");
    if (y <= 0)
        return 0.0;
    const double r = reduced_log(shape, y / scale);
    return std::isnan(r) ? 1.0 : -std::expm1(-r);
}

double gpd_quantile(double p, double scale, double shape) {
    if (!(p >= 0 && p < 1) || !(scale > 0))
        throw std::invalid_argument("gpd_quantile: need p in [0, 1) and scale > 0");
    return scale * factor(shape, -std::log1p(-p));
}

double gpd_log_density(double y, double scale, double shape) {
    if (!(scale > 0))
        throw std::invalid_argument("gpd_log_density: scale must be positive");
    if (y < 0)
        return -std::numeric_limits<double>::infinity();
    return log_likelihood({y}, scale, shape);
}

GpdFit fit_gpd(const std::vector<double>& data, double threshold, const GpdOptions& options) {
    if (!std::ranges::all_of(data, [](double v) { return std::isfinite(v); }) || !std::isfinite(threshold))
        throw std::invalid_argument("fit_gpd: data and threshold must be finite");
    const auto y = excesses(data, threshold);
    if (y.size() < 3)
        throw std::invalid_argument("fit_gpd: need at least 3 exceedances");
    GpdFit fit;
    fit.threshold = threshold, fit.exponential = options.exponential;
    fit.exceedances = y.size(), fit.observations = data.size();
    fit.exceedance_rate = static_cast<double>(y.size()) / static_cast<double>(data.size());
    // Moment start: mean m and variance v of the excesses give xi = (1 - m^2/v)/2, sigma = m (1 - xi).
    double m = 0, v = 0;
    for (const double e : y)
        m += e / static_cast<double>(y.size());
    for (const double e : y)
        v += (e - m) * (e - m) / static_cast<double>(y.size() - 1);
    double xi0 = options.exponential ? 0.0 : std::clamp(0.5 * (1 - m * m / v), -0.4, 0.4);
    const Objective f = [&](const std::vector<double>& p) {
        return log_likelihood(y, std::exp(p[0]), options.exponential ? 0.0 : p[1]);
    };
    std::vector<double> p{std::log(m * (1 - xi0))};
    if (!options.exponential)
        p.push_back(xi0);
    if (!std::isfinite(f(p)))
        p = options.exponential ? std::vector<double>{std::log(m)} : std::vector<double>{std::log(m), 0.0};
    fit.converged = detail::newton(f, p, options.max_iterations, options.tolerance, fit.iterations);
    fit.scale = std::exp(p[0]), fit.shape = options.exponential ? 0.0 : p[1];
    fit.log_likelihood = f(p);
    fit.scale_se = fit.shape_se = nan;
    const Objective natural = [&](const std::vector<double>& q) {
        return log_likelihood(y, q[0], options.exponential ? 0.0 : q[1]);
    };
    std::vector<double> q{fit.scale};
    if (!options.exponential)
        q.push_back(fit.shape);
    std::vector<double> g;
    Matrix h;
    detail::derivatives(natural, q, g, h);
    const std::size_t k = q.size();
    Matrix info(k, k, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
            info(i, j) = -h(i, j);
    const auto chol = linalg::cholesky(info);
    if (chol.ok) {
        fit.covariance = Matrix(k, k, 0.0);
        for (std::size_t i = 0; i < k; ++i) {
            std::vector<double> e(k, 0.0);
            e[i] = 1;
            const auto col = chol.solve(e);
            for (std::size_t j = 0; j < k; ++j)
                fit.covariance(j, i) = col[j];
        }
        fit.scale_se = std::sqrt(fit.covariance(0, 0));
        if (!options.exponential)
            fit.shape_se = std::sqrt(fit.covariance(1, 1));
    }
    return fit;
}

GpdReturnLevel gpd_return_level(const GpdFit& fit, const std::vector<double>& data, double period, double per_period,
                                double confidence, double extremal_index) {
    if (!(per_period > 0) || !(confidence > 0 && confidence < 1))
        throw std::invalid_argument("gpd_return_level: need per_period > 0 and confidence in (0, 1)");
    if (!(extremal_index > 0 && extremal_index <= 1))
        throw std::invalid_argument("gpd_return_level: extremal_index must be in (0, 1]");
    if (fit.covariance.rows() == 0)
        throw std::invalid_argument("gpd_return_level: the fit has no covariance");
    GpdReturnLevel r;
    r.period = period, r.observations = period * per_period;
    const double zeta = fit.exceedance_rate, n = static_cast<double>(fit.observations);
    const double theta = extremal_index;
    const double lmz = std::log(r.observations * zeta * theta);
    if (!(lmz > 0))
        throw std::invalid_argument("gpd_return_level: period too short (m * zeta * theta must exceed 1)");
    const double u = fit.threshold, sigma = fit.scale, xi = fit.shape;
    r.level = u + sigma * factor(xi, lmz);
    // Delta method over (zeta, sigma[, xi]) with var(zeta) = zeta (1 - zeta) / n, independent of (sigma, xi).
    const double dzeta = sigma * std::pow(r.observations * zeta * theta, xi) / zeta; // d x_m / d zeta (xi = 0: sigma/zeta)
    const double dsigma = factor(xi, lmz);
    double var = dzeta * dzeta * zeta * (1 - zeta) / n + dsigma * dsigma * fit.covariance(0, 0);
    if (!fit.exponential) {
        const double e = 1e-6;
        const double dxi = sigma * (factor(xi + e, lmz) - factor(xi - e, lmz)) / (2 * e);
        var += dxi * dxi * fit.covariance(1, 1) + 2 * dsigma * dxi * fit.covariance(0, 1);
    }
    r.standard_error = std::sqrt(var);
    const double zq = random::normal_quantile(1 - (1 - confidence) / 2);
    r.wald_lower = r.level - zq * r.standard_error, r.wald_upper = r.level + zq * r.standard_error;

    // Profile likelihood in x_m: sigma = (x_m - u) / factor(xi); maximize over xi (or nothing when exponential).
    const auto y = excesses(data, u);
    const double crit = random::chi_squared_quantile(confidence, 1.0) / 2;
    const auto profile = [&](double level, double& xi_start) -> double {
        if (!(level > u))
            return -std::numeric_limits<double>::infinity();
        const auto ll = [&](double s) {
            const double fac = factor(s, lmz);
            return fac > 0 ? log_likelihood(y, (level - u) / fac, s) : -std::numeric_limits<double>::infinity();
        };
        if (fit.exponential)
            return ll(0.0);
        // Golden-section search on xi around the warm start, after a coarse scan.
        double best = xi_start, best_v = ll(xi_start);
        for (double s = -0.9; s <= 1.5; s += 0.05) {
            const double v = ll(s);
            if (v > best_v)
                best_v = v, best = s;
        }
        double a = best - 0.06, b = best + 0.06;
        const double gr = (std::sqrt(5.0) - 1) / 2;
        double c = b - gr * (b - a), d = a + gr * (b - a), fc = ll(c), fd = ll(d);
        for (int i = 0; i < 100 && b - a > 1e-12; ++i) {
            if (fc > fd)
                b = d, d = c, fd = fc, c = b - gr * (b - a), fc = ll(c);
            else
                a = c, c = d, fc = fd, d = a + gr * (b - a), fd = ll(d);
        }
        xi_start = (a + b) / 2;
        return std::max(best_v, ll(xi_start));
    };
    const double lmax = fit.log_likelihood;
    const auto search = [&](double direction) {
        double xi_start = xi;
        const double unit = std::isfinite(r.standard_error) && r.standard_error > 0 ? r.standard_error : 0.1 * (r.level - u);
        double inside = r.level, outside = nan, step = unit;
        for (int i = 0; i < 60; ++i) {
            const double z = r.level + direction * step;
            if (lmax - profile(z, xi_start) > crit) {
                outside = z;
                break;
            }
            inside = z;
            step *= 1.5;
        }
        if (std::isnan(outside))
            return nan;
        for (int i = 0; i < 80 && std::abs(outside - inside) > 1e-10 * std::max(1.0, std::abs(inside)); ++i) {
            const double mid = (inside + outside) / 2;
            (lmax - profile(mid, xi_start) > crit ? outside : inside) = mid;
        }
        return (inside + outside) / 2;
    };
    r.profile_lower = search(-1);
    r.profile_upper = search(1);
    return r;
}

std::vector<ThresholdDiagnostic> threshold_diagnostics(const std::vector<double>& data,
                                                       const std::vector<double>& thresholds, double confidence) {
    if (!std::ranges::all_of(data, [](double v) { return std::isfinite(v); }) ||
        !(confidence > 0 && confidence < 1))
        throw std::invalid_argument("threshold_diagnostics: finite data and confidence in (0, 1) required");
    const double zq = random::normal_quantile(1 - (1 - confidence) / 2);
    std::vector<ThresholdDiagnostic> out;
    for (const double u : thresholds) {
        const auto y = excesses(data, u);
        if (y.size() < 2)
            continue;
        ThresholdDiagnostic d;
        d.threshold = u, d.exceedances = y.size();
        const double k = static_cast<double>(y.size());
        double m = 0, v = 0;
        for (const double e : y)
            m += e / k;
        for (const double e : y)
            v += (e - m) * (e - m) / (k - 1);
        d.mean_excess = m;
        d.mean_excess_lower = m - zq * std::sqrt(v / k), d.mean_excess_upper = m + zq * std::sqrt(v / k);
        d.shape = d.shape_lower = d.shape_upper = nan;
        d.modified_scale = d.modified_scale_lower = d.modified_scale_upper = nan;
        if (y.size() >= 10) {
            const auto f = fit_gpd(data, u);
            if (f.covariance.rows() == 2) {
                d.shape = f.shape;
                d.shape_lower = f.shape - zq * f.shape_se, d.shape_upper = f.shape + zq * f.shape_se;
                d.modified_scale = f.scale - f.shape * u;
                // var(sigma - xi u) = var(sigma) - 2u cov + u^2 var(xi).
                const double vs = f.covariance(0, 0) - 2 * u * f.covariance(0, 1) + u * u * f.covariance(1, 1);
                d.modified_scale_lower = d.modified_scale - zq * std::sqrt(vs);
                d.modified_scale_upper = d.modified_scale + zq * std::sqrt(vs);
            }
        }
        out.push_back(d);
    }
    return out;
}

} // namespace datamunge::stats
