/// @file circular.cpp
/// @brief Trigonometric moments, circular spread, mean-direction intervals and von Mises sampling.
#include <datamunge/stats/circular.hpp>

#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

/// @brief Implementation helpers for circular statistics.
namespace {
/// @brief Mean resultant lengths below this are treated as zero (no defined direction).
constexpr double zero_resultant = 1e-12;

/// @brief Checks that a sample is nonempty and finite.
/// @param a Angles.
/// @param what Function name for messages.
void check_angles(const std::vector<double>& a, const char* what) {
    if (a.empty() || !std::ranges::all_of(a, [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument(std::string(what) + ": need a nonempty sample of finite angles");
}

/// @brief Wraps an angle to (-pi, pi].
/// @param x Angle.
/// @return The wrapped angle.
double wrap(double x) {
    const double two_pi = 2 * std::numbers::pi;
    double y = std::fmod(x + std::numbers::pi, two_pi);
    if (y <= 0)
        y += two_pi;
    return y - std::numbers::pi;
}

/// @brief Mean direction of a sample (NaN when the resultant vanishes).
/// @param a Angles.
/// @return atan2 of the mean resultant.
double mean_direction(const std::vector<double>& a) {
    double c = 0, s = 0;
    for (const double v : a)
        c += std::cos(v), s += std::sin(v);
    const double r = std::hypot(c, s) / static_cast<double>(a.size());
    return r < zero_resultant ? std::numeric_limits<double>::quiet_NaN() : std::atan2(s, c);
}
} // namespace

TrigonometricMoment trigonometric_moment(const std::vector<double>& angles, std::size_t order) {
    check_angles(angles, "trigonometric_moment");
    if (order == 0)
        throw std::invalid_argument("trigonometric_moment: order must be at least 1");
    const double p = static_cast<double>(order), n = static_cast<double>(angles.size());
    TrigonometricMoment m;
    for (const double v : angles)
        m.cosine += std::cos(p * v) / n, m.sine += std::sin(p * v) / n;
    m.resultant_length = std::hypot(m.cosine, m.sine);
    m.direction = m.resultant_length < zero_resultant ? std::numeric_limits<double>::quiet_NaN()
                                                       : std::atan2(m.sine, m.cosine);
    return m;
}

CircularSummary circular_summary(const std::vector<double>& angles, double confidence) {
    check_angles(angles, "circular_summary");
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument("circular_summary: confidence must be in (0, 1)");
    const TrigonometricMoment m = trigonometric_moment(angles, 1);
    CircularSummary s;
    s.n = angles.size();
    const double n = static_cast<double>(s.n), r = m.resultant_length;
    s.mean_direction = m.direction;
    s.resultant_length = r;
    s.circular_variance = 1 - r;
    s.circular_sd = r > 0 ? std::sqrt(-2 * std::log(r)) : std::numeric_limits<double>::infinity();
    s.angular_deviation = std::sqrt(2 * (1 - r));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (r < zero_resultant) {
        s.second_central_cosine = s.circular_dispersion = s.standard_error = s.lower = s.upper = nan;
        return s;
    }
    double c2 = 0;
    for (const double v : angles)
        c2 += std::cos(2 * (v - s.mean_direction)) / n;
    s.second_central_cosine = c2;
    s.circular_dispersion = (1 - c2) / (2 * r * r);
    s.standard_error = std::sqrt(s.circular_dispersion / n);
    const double z = random::normal_quantile(0.5 + confidence / 2);
    const double arg = z * s.standard_error;
    if (arg < 1) {
        s.lower = s.mean_direction - std::asin(arg);
        s.upper = s.mean_direction + std::asin(arg);
    } else {
        s.lower = s.upper = nan;
    }
    return s;
}

CircularInterval circular_mean_bootstrap_interval(const std::vector<double>& angles, double confidence,
                                                  std::size_t replicates, std::uint64_t seed) {
    check_angles(angles, "circular_mean_bootstrap_interval");
    if (angles.size() < 2 || !(confidence > 0 && confidence < 1) || replicates < 100)
        throw std::invalid_argument("circular_mean_bootstrap_interval: need n >= 2, confidence in (0, 1) "
                                    "and at least 100 replicates");
    const double centre = mean_direction(angles);
    if (std::isnan(centre))
        throw std::invalid_argument("circular_mean_bootstrap_interval: the mean resultant is zero");
    random::SplitMix64 rng(seed);
    const std::size_t n = angles.size();
    std::vector<double> dev;
    dev.reserve(replicates);
    std::vector<double> resample(n);
    for (std::size_t b = 0; b < replicates; ++b) {
        for (std::size_t i = 0; i < n; ++i)
            resample[i] = angles[std::min(n - 1, static_cast<std::size_t>(rng.uniform01() * static_cast<double>(n)))];
        const double m = mean_direction(resample);
        if (!std::isnan(m))
            dev.push_back(wrap(m - centre));
    }
    std::ranges::sort(dev);
    const auto quantile = [&](double q) {
        const double pos = q * static_cast<double>(dev.size() - 1);
        const auto i = static_cast<std::size_t>(pos);
        return i + 1 < dev.size() ? dev[i] + (pos - static_cast<double>(i)) * (dev[i + 1] - dev[i]) : dev[i];
    };
    const double alpha = 1 - confidence;
    return {centre + quantile(alpha / 2), centre + quantile(1 - alpha / 2)};
}

std::vector<double> von_mises_sample(std::size_t n, double mean_direction, double concentration, std::uint64_t seed) {
    if (!(concentration >= 0) || !std::isfinite(concentration) || !std::isfinite(mean_direction))
        throw std::invalid_argument("von_mises_sample: need a finite mean and a nonnegative finite concentration");
    random::SplitMix64 rng(seed);
    std::vector<double> out(n);
    const double pi = std::numbers::pi;
    if (concentration < 1e-12) {
        for (double& v : out)
            v = wrap(mean_direction + pi * (2 * rng.uniform01() - 1));
        return out;
    }
    // Best and Fisher (1979).
    const double tau = 1 + std::sqrt(1 + 4 * concentration * concentration);
    const double rho = (tau - std::sqrt(2 * tau)) / (2 * concentration);
    const double r = (1 + rho * rho) / (2 * rho);
    for (double& v : out) {
        while (true) {
            const double u1 = rng.uniform01(), u2 = rng.uniform01(), u3 = rng.uniform01();
            const double z = std::cos(pi * u1);
            const double f = (1 + r * z) / (r + z);
            const double c = concentration * (r - f);
            if (c * (2 - c) - u2 > 0 || std::log(c / u2) + 1 - c >= 0) {
                const double theta = (u3 > 0.5 ? 1.0 : -1.0) * std::acos(std::clamp(f, -1.0, 1.0));
                v = wrap(mean_direction + theta);
                break;
            }
        }
    }
    return out;
}

/// @brief Implementation helpers for circular uniformity tests.
namespace {
/// @brief Rayleigh p-value with the Greenwood-Durand correction for n < 50.
/// @param rbar Mean resultant length.
/// @param n Sample size.
/// @return The p-value.
double rayleigh_p(double rbar, double n) {
    const double z = n * rbar * rbar;
    double p = std::exp(-z);
    if (n < 50)
        p *= 1 + (2 * z - z * z) / (4 * n) - (24 * z - 132 * z * z + 76 * z * z * z - 9 * z * z * z * z) / (288 * n * n);
    return std::clamp(p, 0.0, 1.0);
}

/// @brief Mean resultant length.
/// @param a Angles.
/// @return R-bar.
double rbar_of(const std::vector<double>& a) {
    double c = 0, s = 0;
    for (const double v : a)
        c += std::cos(v), s += std::sin(v);
    return std::hypot(c, s) / static_cast<double>(a.size());
}

/// @brief Projected mean resultant toward a direction.
/// @param a Angles.
/// @param mu Direction.
/// @return R-bar_0.
double r0_of(const std::vector<double>& a, double mu) {
    double c = 0;
    for (const double v : a)
        c += std::cos(v - mu);
    return c / static_cast<double>(a.size());
}

/// @brief Kuiper's unmodified V from probability-integral transforms (sorted in place).
/// @param u Values in [0, 1].
/// @return V.
double kuiper_v(std::vector<double>& u) {
    std::ranges::sort(u);
    const double n = static_cast<double>(u.size());
    double dp = -1e300, dm = -1e300;
    for (std::size_t i = 0; i < u.size(); ++i) {
        const double k = static_cast<double>(i);
        dp = std::max(dp, (k + 1) / n - u[i]);
        dm = std::max(dm, u[i] - k / n);
    }
    return dp + dm;
}

/// @brief Stephens' modification of Kuiper's V.
/// @param v V.
/// @param n Sample size.
/// @return V*.
double kuiper_modified(double v, double n) { return v * (std::sqrt(n) + 0.155 + 0.24 / std::sqrt(n)); }

/// @brief Asymptotic upper tail of the Kuiper distribution.
/// @param v Modified statistic V*.
/// @return The p-value in [0, 1].
double kuiper_tail(double v) {
    if (v < 0.4)
        return 1.0;
    double s = 0;
    for (int k = 1; k <= 100; ++k) {
        const double kk = static_cast<double>(k) * static_cast<double>(k), term = (4 * kk * v * v - 1) * std::exp(-2 * kk * v * v);
        s += term;
        if (std::abs(term) < 1e-16)
            break;
    }
    return std::clamp(2 * s, 0.0, 1.0);
}

/// @brief Monte Carlo p-value from simulated statistics of uniform samples.
/// @param n Sample size.
/// @param observed Observed statistic (large values extreme).
/// @param options Simulations and seed.
/// @param stat Statistic of a vector of uniform angles in [0, 2 pi).
/// @return The p-value (NaN without simulations).
template <typename Statistic>
double monte_carlo(std::size_t n, double observed, const CircularTestOptions& options, Statistic stat) {
    if (options.simulations == 0)
        return std::numeric_limits<double>::quiet_NaN();
    random::SplitMix64 rng(options.seed);
    std::vector<double> sim(n);
    std::size_t ge = 0;
    const double tol = 1e-12 * std::max(1.0, std::abs(observed));
    for (std::size_t s = 0; s < options.simulations; ++s) {
        for (double& v : sim)
            v = 2 * std::numbers::pi * rng.uniform01();
        ge += stat(sim) >= observed - tol;
    }
    return (1.0 + static_cast<double>(ge)) / (1.0 + static_cast<double>(options.simulations));
}
} // namespace

CircularTest rayleigh_test(const std::vector<double>& angles, const CircularTestOptions& options) {
    check_angles(angles, "rayleigh_test");
    const double n = static_cast<double>(angles.size());
    CircularTest t;
    t.statistic = rbar_of(angles);
    t.p_value = rayleigh_p(t.statistic, n);
    t.monte_carlo_p_value = monte_carlo(angles.size(), t.statistic, options, [](const std::vector<double>& a) { return rbar_of(a); });
    return t;
}

CircularTest rayleigh_test_direction(const std::vector<double>& angles, double direction,
                                     const CircularTestOptions& options) {
    check_angles(angles, "rayleigh_test_direction");
    if (!std::isfinite(direction))
        throw std::invalid_argument("rayleigh_test_direction: the direction must be finite");
    const double n = static_cast<double>(angles.size());
    CircularTest t;
    t.statistic = r0_of(angles, direction);
    const double z = std::sqrt(2 * n) * t.statistic;
    const double phi = std::exp(-z * z / 2) / std::sqrt(2 * std::numbers::pi);
    const double z3 = z * z * z, z5 = z3 * z * z, z7 = z5 * z * z;
    t.p_value = std::clamp(1 - random::normal_cdf(z) +
                               phi * ((3 * z - z3) / (16 * n) + (15 * z + 305 * z3 - 125 * z5 + 9 * z7) / (4608 * n * n)),
                           0.0, 1.0);
    t.monte_carlo_p_value = monte_carlo(angles.size(), t.statistic, options,
                                        [direction](const std::vector<double>& a) { return r0_of(a, direction); });
    return t;
}

CircularTest kuiper_test(const std::vector<double>& angles, const std::function<double(double)>& cdf,
                         const CircularTestOptions& options) {
    check_angles(angles, "kuiper_test");
    const double two_pi = 2 * std::numbers::pi, n = static_cast<double>(angles.size());
    std::vector<double> u;
    u.reserve(angles.size());
    for (const double a : angles) {
        double r = std::fmod(a, two_pi);
        if (r < 0)
            r += two_pi;
        const double f = cdf ? cdf(r) : r / two_pi;
        if (!(f >= 0 && f <= 1))
            throw std::invalid_argument("kuiper_test: the CDF must return values in [0, 1]");
        u.push_back(f);
    }
    CircularTest t;
    t.statistic = kuiper_modified(kuiper_v(u), n);
    t.p_value = kuiper_tail(t.statistic);
    t.monte_carlo_p_value = monte_carlo(angles.size(), t.statistic, options, [&](const std::vector<double>& a) {
        std::vector<double> w;
        w.reserve(a.size());
        for (const double v : a)
            w.push_back(v / two_pi);
        return kuiper_modified(kuiper_v(w), n);
    });
    return t;
}

/// @brief Implementation helpers for von Mises fitting.
namespace {
/// @brief Bessel-function ratios A_p = I_p(kappa) / I_0(kappa) for p = 1..m by backward recurrence.
/// @param kappa Concentration (> 0).
/// @param m Number of ratios.
/// @return A_1..A_m.
std::vector<double> bessel_ratios(double kappa, std::size_t m) {
    // r_nu = I_{nu+1} / I_nu satisfies r_nu = kappa / (2 (nu + 1) + kappa r_{nu+1}).
    const std::size_t start = m + 30 + static_cast<std::size_t>(kappa + 10 * std::sqrt(kappa));
    std::vector<double> r(start + 1, 0.0);
    for (std::size_t nu = start; nu-- > 0;)
        r[nu] = kappa / (2 * static_cast<double>(nu + 1) + kappa * r[nu + 1]);
    std::vector<double> a(m);
    double prod = 1;
    for (std::size_t p = 0; p < m; ++p)
        a[p] = prod *= r[p];
    return a;
}

/// @brief log I_0(kappa) without overflow.
/// @param kappa Concentration >= 0.
/// @return log I_0.
double log_i0(double kappa) {
    if (kappa < 500)
        return std::log(std::cyl_bessel_i(0.0, kappa));
    const double k = kappa;
    return k - 0.5 * std::log(2 * std::numbers::pi * k) + std::log1p(1 / (8 * k) + 9 / (128 * k * k) + 225 / (3072 * k * k * k));
}

/// @brief Fisher's (1993) approximation to the inverse of A1, as `circular::A1inv`.
/// @param r Mean resultant length in [0, 1).
/// @return Approximate kappa.
double a1_inverse_fisher(double r) {
    if (r < 0.53)
        return 2 * r + r * r * r + 5 * std::pow(r, 5) / 6;
    if (r < 0.85)
        return -0.4 + 1.39 * r + 0.43 / (1 - r);
    return 1 / (r * r * r - 4 * r * r + 3 * r);
}

/// @brief Profile log-likelihood of kappa (per observation) with the mean fixed.
/// @param kappa Concentration.
/// @param r0 Mean of cos(theta - mu).
/// @return kappa r0 - log I_0(kappa).
double profile(double kappa, double r0) { return kappa * r0 - log_i0(kappa); }
} // namespace

double von_mises_a1(double concentration) {
    if (!(concentration >= 0))
        throw std::invalid_argument("von_mises_a1: the concentration must be nonnegative");
    if (concentration == 0)
        return 0;
    if (std::isinf(concentration))
        return 1;
    return bessel_ratios(concentration, 1)[0];
}

double von_mises_a1_inverse(double resultant_length) {
    const double r = resultant_length;
    if (!(r >= 0 && r <= 1))
        throw std::invalid_argument("von_mises_a1_inverse: the resultant length must be in [0, 1]");
    if (r == 0)
        return 0;
    if (r == 1)
        return std::numeric_limits<double>::infinity();
    // Newton on log kappa from Fisher's approximation; A1 is increasing with A1' = 1 - A1/k - A1^2.
    double k = std::max(1e-8, a1_inverse_fisher(r));
    for (int it = 0; it < 100; ++it) {
        const double a = von_mises_a1(k), d = 1 - a / k - a * a;
        double next = k - (a - r) / d;
        if (!(next > 0))
            next = k / 2;
        if (std::abs(next - k) <= 1e-14 * k) {
            k = next;
            break;
        }
        k = next;
    }
    return k;
}

double von_mises_log_likelihood(const std::vector<double>& angles, double mean_direction, double concentration) {
    check_angles(angles, "von_mises_log_likelihood");
    if (!std::isfinite(mean_direction) || !(concentration >= 0) || !std::isfinite(concentration))
        throw std::invalid_argument("von_mises_log_likelihood: need a finite mean and concentration >= 0");
    double s = 0;
    for (const double v : angles)
        s += std::cos(v - mean_direction);
    const double n = static_cast<double>(angles.size());
    return concentration * s - n * (std::log(2 * std::numbers::pi) + log_i0(concentration));
}

double von_mises_cdf(double angle, double mean_direction, double concentration) {
    if (!std::isfinite(angle) || !std::isfinite(mean_direction) || !(concentration >= 0) || !std::isfinite(concentration))
        throw std::invalid_argument("von_mises_cdf: need finite inputs and concentration >= 0");
    const double two_pi = 2 * std::numbers::pi;
    double th = std::fmod(angle, two_pi);
    if (th < 0)
        th += two_pi;
    double f = th / two_pi;
    if (concentration > 0) {
        const auto m = static_cast<std::size_t>(concentration + 40 * std::sqrt(concentration) + 40);
        const auto a = bessel_ratios(concentration, m);
        double s = 0;
        for (std::size_t p = 1; p <= m; ++p) {
            const double pp = static_cast<double>(p);
            s += a[p - 1] / pp * (std::sin(pp * (th - mean_direction)) + std::sin(pp * mean_direction));
            if (a[p - 1] < 1e-17)
                break;
        }
        f += s / std::numbers::pi;
    }
    return std::clamp(f, 0.0, 1.0);
}

VonMisesFit von_mises_fit(const std::vector<double>& angles, const VonMisesFitOptions& options) {
    check_angles(angles, "von_mises_fit");
    if (angles.size() < 2 || !(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("von_mises_fit: need two or more angles and confidence in (0, 1)");
    const bool estimate_mean = std::isnan(options.mean_direction);
    if (!estimate_mean && !std::isfinite(options.mean_direction))
        throw std::invalid_argument("von_mises_fit: the known mean direction must be finite");
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    VonMisesFit f;
    f.n = angles.size();
    const double n = static_cast<double>(f.n);
    const double rbar = rbar_of(angles);
    f.rayleigh_p_value = rayleigh_p(rbar, n);
    f.vanishing_resultant = f.rayleigh_p_value > 1 - options.confidence;
    f.mean_direction = estimate_mean ? mean_direction(angles) : options.mean_direction;
    f.resultant_length = std::isnan(f.mean_direction) ? 0.0 : r0_of(angles, f.mean_direction);
    const double r0 = std::min(1.0, f.resultant_length);
    double kappa = 0;
    if (r0 > 0)
        kappa = r0 >= 1 ? inf : (options.exact_inverse ? von_mises_a1_inverse(r0) : a1_inverse_fisher(r0));
    if (options.bias_correction && std::isfinite(kappa))
        kappa = kappa < 2 ? std::max(kappa - 2 / (n * kappa), 0.0) : std::pow(n - 1, 3) * kappa / (n * n * n + n);
    f.concentration = kappa;
    f.boundary = kappa == 0;
    if (f.boundary && estimate_mean)
        f.mean_direction = nan;
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    if (std::isinf(kappa)) {
        f.mean_se = f.concentration_se = 0;
        f.log_likelihood = inf;
        f.aic = -inf;
        f.mean_lower = f.mean_upper = f.mean_direction;
        f.concentration_lower = f.concentration_upper = inf;
        return f;
    }
    const double a1 = von_mises_a1(kappa);
    f.mean_se = !estimate_mean ? 0.0 : kappa > 0 ? 1 / std::sqrt(n * kappa * a1) : inf;
    f.concentration_se = kappa > 0 ? 1 / std::sqrt(n * (1 - a1 / kappa - a1 * a1)) : 1 / std::sqrt(n / 2);
    f.log_likelihood = std::isnan(f.mean_direction)
                           ? -n * std::log(2 * std::numbers::pi)
                           : von_mises_log_likelihood(angles, f.mean_direction, kappa);
    f.aic = -2 * f.log_likelihood + 2 * (estimate_mean ? 2.0 : 1.0);
    if (estimate_mean && kappa > 0 && z * f.mean_se < std::numbers::pi) {
        f.mean_lower = f.mean_direction - z * f.mean_se;
        f.mean_upper = f.mean_direction + z * f.mean_se;
    } else if (!estimate_mean) {
        f.mean_lower = f.mean_upper = f.mean_direction;
    } else {
        f.mean_lower = f.mean_upper = nan;
    }
    // Profile-likelihood interval for kappa at the fitted mean (uses the uncorrected maximiser).
    const double kmax = r0 > 0 ? von_mises_a1_inverse(r0) : 0.0;
    const double top = profile(kmax, r0), drop = z * z / 2 / n;
    const auto below = [&](double k) { return top - profile(k, r0) - drop; };
    const auto bisect = [&](double lo, double hi) {
        for (int it = 0; it < 200; ++it) {
            const double mid = (lo + hi) / 2;
            ((below(lo) > 0) == (below(mid) > 0) ? lo : hi) = mid;
            if (hi - lo <= 1e-12 * std::max(1.0, hi))
                break;
        }
        return (lo + hi) / 2;
    };
    f.concentration_lower = below(0.0) <= 0 ? 0.0 : bisect(0.0, kmax);
    double hi = std::max(1.0, 2 * kmax);
    while (below(hi) < 0 && hi < 1e12)
        hi *= 2;
    f.concentration_upper = bisect(kmax, hi);
    return f;
}

} // namespace datamunge::stats
