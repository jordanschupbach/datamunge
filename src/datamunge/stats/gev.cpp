/// @file gev.cpp
/// @brief GEV likelihood, Newton fitting with numerical derivatives, L-moments, return levels and profiles.
#include <datamunge/stats/gev.hpp>

#include "extreme_detail.hpp"

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for GEV fitting.
namespace {
/// @brief Dense matrix alias.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Objective alias from the shared extreme-value helpers.
using detail::Objective;
/// @brief Newton maximizer from the shared extreme-value helpers.
using detail::newton;
/// @brief Numerical derivatives from the shared extreme-value helpers.
using detail::derivatives;

/// @brief Quiet NaN.
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/// @brief Euler-Mascheroni constant.
constexpr double euler_gamma = 0.57721566490153286;

/// @brief \f$\log(1+\xi z)/\xi\f$, continuous at \f$\xi=0\f$ (value z).
/// @param xi Shape.
/// @param z Standardized value.
/// @return The reduced log term (NaN outside the support).
double reduced_log(double xi, double z) {
    if (xi == 0)
        return z;
    const double a = xi * z;
    if (a <= -1)
        return nan;
    return std::abs(a) < 1e-8 ? z * (1 - a / 2 + a * a / 3) : std::log1p(a) / xi;
}

/// @brief GEV log-likelihood of the data.
/// @param x Data.
/// @param mu Location.
/// @param sigma Scale.
/// @param xi Shape.
/// @return Log-likelihood (\f$-\infty\f$ outside the support or for sigma <= 0).
double log_likelihood(const std::vector<double>& x, double mu, double sigma, double xi) {
    if (!(sigma > 0) || !std::isfinite(mu) || !std::isfinite(xi))
        return -INFINITY;
    double ll = -static_cast<double>(x.size()) * std::log(sigma);
    for (const double v : x) {
        const double z = (v - mu) / sigma;
        const double r = reduced_log(xi, z);
        if (std::isnan(r))
            return -INFINITY;
        // log t = xi * r; density term -(1 + 1/xi) log t = -(xi + 1) r.
        ll += -(xi + 1) * r - std::exp(-r);
    }
    return ll;
}

/// @brief Sample L-moments l1, l2 and the L-skewness t3.
/// @param x Data.
/// @return {l1, l2, t3}.
std::array<double, 3> l_moments(std::vector<double> x) {
    std::ranges::sort(x);
    const double n = static_cast<double>(x.size());
    double b0 = 0, b1 = 0, b2 = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double j = static_cast<double>(i);
        b0 += x[i] / n;
        b1 += j / (n - 1) * x[i] / n;
        b2 += j * (j - 1) / ((n - 1) * (n - 2)) * x[i] / n;
    }
    const double l2 = 2 * b1 - b0, l3 = 6 * b2 - 6 * b1 + b0;
    return {b0, l2, l3 / l2};
}

/// @brief Return level \f$z_T\f$ for given parameters.
/// @param mu Location.
/// @param sigma Scale.
/// @param xi Shape.
/// @param yp \f$-\log(1-1/T)\f$.
/// @return Level.
double level_of(double mu, double sigma, double xi, double yp) {
    // z = mu + sigma (yp^-xi - 1) / xi, continuous at xi = 0 (mu - sigma log yp).
    const double ly = std::log(yp);
    const double factor = std::abs(xi) < 1e-10 ? -ly : std::expm1(-xi * ly) / xi;
    return mu + sigma * factor;
}

/// @brief Validates data.
/// @param x Data.
void validate(const std::vector<double>& x) {
    if (x.size() < 3)
        throw std::invalid_argument("fit_gev: need at least 3 maxima");
    if (!std::ranges::all_of(x, [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("fit_gev: maxima must be finite");
    if (std::ranges::max(x) == std::ranges::min(x))
        throw std::invalid_argument("fit_gev: maxima must not all be equal");
}
} // namespace

double gev_cdf(double x, double location, double scale, double shape) {
    if (!(scale > 0))
        throw std::invalid_argument("gev_cdf: scale must be positive");
    const double z = (x - location) / scale;
    const double r = reduced_log(shape, z);
    if (std::isnan(r))
        return shape > 0 ? 0.0 : 1.0;
    return std::exp(-std::exp(-r));
}

double gev_quantile(double p, double location, double scale, double shape) {
    if (!(p > 0 && p < 1) || !(scale > 0))
        throw std::invalid_argument("gev_quantile: need p in (0, 1) and scale > 0");
    return level_of(location, scale, shape, -std::log(p));
}

double gev_log_density(double x, double location, double scale, double shape) {
    if (!(scale > 0))
        throw std::invalid_argument("gev_log_density: scale must be positive");
    return log_likelihood({x}, location, scale, shape);
}

GevFit fit_gev(const std::vector<double>& maxima, const GevOptions& options) {
    validate(maxima);
    GevFit fit;
    fit.n = maxima.size(), fit.gumbel = options.gumbel;
    const auto [l1, l2, t3] = l_moments(maxima);
    // L-moment estimates (Hosking, Wallis and Wood 1985); k = -xi.
    double mu = 0, sigma = 0, xi = 0;
    if (options.gumbel) {
        sigma = l2 / std::numbers::ln2;
        mu = l1 - euler_gamma * sigma;
    } else {
        const double c = 2 / (3 + t3) - std::numbers::ln2 / std::log(3.0);
        const double k = 7.8590 * c + 2.9554 * c * c;
        const double g = std::tgamma(1 + k);
        sigma = std::abs(k) < 1e-12 ? l2 / std::numbers::ln2 : l2 * k / (-std::expm1(-k * std::numbers::ln2) * g);
        mu = std::abs(k) < 1e-12 ? l1 - euler_gamma * sigma : l1 - sigma * (1 - g) / k;
        xi = -k;
    }
    fit.location = mu, fit.scale = sigma, fit.shape = xi;
    fit.location_se = fit.scale_se = fit.shape_se = nan;
    if (options.method == GevMethod::l_moments) {
        fit.log_likelihood = log_likelihood(maxima, mu, sigma, xi);
        fit.converged = true;
    } else {
        // Newton in (mu, log sigma[, xi]); fall back to the Gumbel start if the L-moment start is infeasible.
        const Objective f = [&](const std::vector<double>& p) {
            return log_likelihood(maxima, p[0], std::exp(p[1]), options.gumbel ? 0.0 : p[2]);
        };
        std::vector<double> p{mu, std::log(sigma)};
        if (!options.gumbel)
            p.push_back(xi);
        if (!std::isfinite(f(p)) && !options.gumbel) {
            const double s = l2 / std::numbers::ln2;
            p = {l1 - euler_gamma * s, std::log(s), 0.0};
        }
        fit.converged = newton(f, p, options.max_iterations, options.tolerance, fit.iterations);
        fit.location = p[0], fit.scale = std::exp(p[1]), fit.shape = options.gumbel ? 0.0 : p[2];
        fit.log_likelihood = f(p);
        // Observed information in the natural (mu, sigma[, xi]) parameters.
        const Objective natural = [&](const std::vector<double>& q) {
            return log_likelihood(maxima, q[0], q[1], options.gumbel ? 0.0 : q[2]);
        };
        std::vector<double> q{fit.location, fit.scale};
        if (!options.gumbel)
            q.push_back(fit.shape);
        std::vector<double> g;
        Matrix h;
        derivatives(natural, q, g, h);
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
            fit.location_se = std::sqrt(fit.covariance(0, 0));
            fit.scale_se = std::sqrt(fit.covariance(1, 1));
            if (!options.gumbel)
                fit.shape_se = std::sqrt(fit.covariance(2, 2));
        }
    }
    const double params = options.gumbel ? 2.0 : 3.0;
    fit.aic = -2 * fit.log_likelihood + 2 * params;
    return fit;
}

GumbelTest gumbel_test(const std::vector<double>& maxima) {
    GumbelTest t;
    t.gev = fit_gev(maxima);
    t.gumbel = fit_gev(maxima, {.gumbel = true});
    t.statistic = std::max(0.0, 2 * (t.gev.log_likelihood - t.gumbel.log_likelihood));
    t.p_value = random::chi_squared_survival(t.statistic, 1.0);
    return t;
}

ReturnLevel gev_return_level(const GevFit& fit, const std::vector<double>& maxima, double period, double confidence) {
    if (!(period > 1) || !(confidence > 0 && confidence < 1))
        throw std::invalid_argument("gev_return_level: need period > 1 and confidence in (0, 1)");
    if (fit.covariance.rows() == 0)
        throw std::invalid_argument("gev_return_level: needs a maximum-likelihood fit with a covariance");
    validate(maxima);
    ReturnLevel r;
    r.period = period;
    const double yp = -std::log1p(-1 / period);
    r.level = level_of(fit.location, fit.scale, fit.shape, yp);
    // Delta method: gradient of z_T with respect to (mu, sigma[, xi]).
    const std::size_t k = fit.covariance.rows();
    std::vector<double> grad(k);
    grad[0] = 1;
    grad[1] = level_of(0, 1, fit.shape, yp);
    if (k == 3) {
        const double e = 1e-6;
        grad[2] = (level_of(fit.location, fit.scale, fit.shape + e, yp) -
                   level_of(fit.location, fit.scale, fit.shape - e, yp)) /
                  (2 * e);
    }
    double var = 0;
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
            var += grad[i] * fit.covariance(i, j) * grad[j];
    r.standard_error = std::sqrt(var);
    const double zq = random::normal_quantile(1 - (1 - confidence) / 2);
    r.wald_lower = r.level - zq * r.standard_error, r.wald_upper = r.level + zq * r.standard_error;

    // Profile likelihood: mu = z - sigma * factor(xi); maximize over (log sigma[, xi]).
    const double crit = random::chi_squared_quantile(confidence, 1.0) / 2;
    const auto profile = [&](double z, std::vector<double>& start) -> double {
        const Objective f = [&](const std::vector<double>& p) {
            const double sigma = std::exp(p[0]), xi = k == 3 ? p[1] : 0.0;
            return log_likelihood(maxima, z - (level_of(0, sigma, xi, yp)), sigma, xi);
        };
        std::vector<double> best;
        double best_val = -INFINITY;
        std::vector<std::vector<double>> starts{start};
        for (const double xi : {fit.shape, 0.0, -0.1, 0.1, -0.2, 0.2, -0.4, 0.4})
            for (const double sm : {1.0, 0.5, 2.0})
                starts.push_back(k == 3 ? std::vector<double>{std::log(fit.scale * sm), xi}
                                        : std::vector<double>{std::log(fit.scale * sm)});
        for (const auto& s : starts) {
            const double v = f(s);
            if (v > best_val)
                best_val = v, best = s;
        }
        if (!std::isfinite(best_val))
            return -INFINITY;
        std::size_t it = 0;
        newton(f, best, 200, 1e-9, it);
        start = best;
        return f(best);
    };
    const double lmax = fit.log_likelihood;
    const auto search = [&](double direction) {
        std::vector<double> start = k == 3 ? std::vector<double>{std::log(fit.scale), fit.shape}
                                           : std::vector<double>{std::log(fit.scale)};
        const double unit = std::isfinite(r.standard_error) && r.standard_error > 0 ? r.standard_error
                                                                                     : 0.1 * std::abs(r.level) + 1e-3;
        double inside = r.level, outside = nan;
        double step = unit;
        for (int i = 0; i < 60; ++i) {
            const double z = r.level + direction * step;
            const double v = profile(z, start);
            if (lmax - v > crit) {
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
            (lmax - profile(mid, start) > crit ? outside : inside) = mid;
        }
        return (inside + outside) / 2;
    };
    r.profile_lower = search(-1);
    r.profile_upper = search(1);
    return r;
}

} // namespace datamunge::stats
