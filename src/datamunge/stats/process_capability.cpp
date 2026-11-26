/// @file process_capability.cpp
/// @brief Capability indices, their intervals and the d2 / c4 unbiasing constants.
#include <datamunge/stats/process_capability.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for capability analysis.
namespace {
/// @brief Bissell interval for a one-sided-type index.
/// @param c Estimate.
/// @param n Observations.
/// @param z Normal quantile.
/// @return Index with limits.
CapabilityIndex bissell(double c, double n, double z) {
    if (!std::isfinite(c))
        return {};
    const double h = z * std::sqrt(1 / (9 * n * c * c) + 1 / (2 * (n - 1)));
    return {c, c * (1 - h), c * (1 + h)};
}

/// @brief Chi-square interval for a Cp-type index.
/// @param c Estimate.
/// @param df Degrees of freedom.
/// @param conf Confidence level.
/// @return Index with limits.
CapabilityIndex chi_interval(double c, double df, double conf) {
    if (!std::isfinite(c))
        return {};
    const double a = (1 - conf) / 2;
    return {c, c * std::sqrt(random::chi_squared_quantile(a, df) / df),
            c * std::sqrt(random::chi_squared_quantile(1 - a, df) / df)};
}

/// @brief Fills the four indices of one family (C or P) for a given sigma.
/// @param o Options.
/// @param mean Process mean.
/// @param sigma Standard deviation.
/// @param n Observations.
/// @param z Normal quantile.
/// @param p, l, u, k Outputs.
void indices(const CapabilityOptions& o, double mean, double sigma, double n, double z, CapabilityIndex& p,
             CapabilityIndex& l, CapabilityIndex& u, CapabilityIndex& k) {
    const bool lo = !std::isnan(o.lsl), hi = !std::isnan(o.usl);
    if (lo && hi)
        p = chi_interval((o.usl - o.lsl) / (6 * sigma), n - 1, o.confidence);
    if (lo)
        l = bissell((mean - o.lsl) / (3 * sigma), n, z);
    if (hi)
        u = bissell((o.usl - mean) / (3 * sigma), n, z);
    k = bissell(lo && hi ? std::min(l.value, u.value) : (lo ? l.value : u.value), n, z);
}

/// @brief Normal-theory fraction outside the limits, in ppm.
/// @param o Options.
/// @param mean Mean.
/// @param sigma SD.
/// @return ppm.
double expected_ppm(const CapabilityOptions& o, double mean, double sigma) {
    double f = 0;
    if (!std::isnan(o.lsl))
        f += random::normal_cdf((o.lsl - mean) / sigma);
    if (!std::isnan(o.usl))
        f += random::normal_cdf((mean - o.usl) / sigma);
    return 1e6 * f;
}
} // namespace

double range_constant_d2(std::size_t n) {
    if (n < 2)
        throw std::invalid_argument("range_constant_d2: n must be at least 2");
    // Composite Simpson on [-12, 12]; the integrand is smooth and negligible beyond.
    const int m = 6000;
    const double a = -12, b = 12, h = (b - a) / m, nd = static_cast<double>(n);
    double s = 0;
    for (int i = 0; i <= m; ++i) {
        const double x = a + i * h, phi = random::normal_cdf(x), q = random::normal_cdf(-x);
        const double f = 1 - std::pow(phi, nd) - std::pow(q, nd);
        s += f * (i == 0 || i == m ? 1 : (i % 2 ? 4 : 2));
    }
    return s * h / 3;
}

double sd_constant_c4(std::size_t n) {
    if (n < 2)
        throw std::invalid_argument("sd_constant_c4: n must be at least 2");
    const double nd = static_cast<double>(n);
    return std::sqrt(2 / (nd - 1)) * std::exp(std::lgamma(nd / 2) - std::lgamma((nd - 1) / 2));
}

ProcessCapability process_capability(const std::vector<std::vector<double>>& subgroups,
                                     const CapabilityOptions& options) {
    CapabilityOptions o = options;
    if (std::isnan(o.lsl) && std::isnan(o.usl))
        throw std::invalid_argument("process_capability: need at least one specification limit");
    if (!std::isnan(o.lsl) && !std::isnan(o.usl) && !(o.lsl < o.usl))
        throw std::invalid_argument("process_capability: LSL must be below USL");
    if (!(o.confidence > 0 && o.confidence < 1))
        throw std::invalid_argument("process_capability: confidence must be in (0, 1)");
    ProcessCapability c;
    std::vector<double> all;
    for (const auto& g : subgroups) {
        if (g.empty())
            throw std::invalid_argument("process_capability: empty subgroup");
        for (const double v : g) {
            if (!std::isfinite(v))
                throw std::invalid_argument("process_capability: values must be finite");
            all.push_back(v);
        }
    }
    c.n = all.size();
    c.subgroups = subgroups.size();
    if (c.n < 3)
        throw std::invalid_argument("process_capability: need at least three observations");
    const double nd = static_cast<double>(c.n);
    for (const double v : all)
        c.mean += v / nd;
    double ss = 0;
    for (const double v : all)
        ss += (v - c.mean) * (v - c.mean);
    c.sigma_overall = std::sqrt(ss / (nd - 1)) / (o.unbiased_overall ? sd_constant_c4(c.n) : 1.0);
    const bool needs_groups = o.within != WithinSigma::moving_range;
    if (needs_groups && std::ranges::any_of(subgroups, [](const auto& g) { return g.size() < 2; }))
        throw std::invalid_argument("process_capability: this within-SD estimator needs subgroups of size >= 2");
    switch (o.within) {
    case WithinSigma::pooled_sd: {
        double num = 0, d = 0;
        for (const auto& g : subgroups) {
            double m = 0;
            for (const double v : g)
                m += v / static_cast<double>(g.size());
            for (const double v : g)
                num += (v - m) * (v - m);
            d += static_cast<double>(g.size() - 1);
        }
        c.sigma_within = std::sqrt(num / d) / sd_constant_c4(static_cast<std::size_t>(d) + 1);
        break;
    }
    case WithinSigma::average_range:
        for (const auto& g : subgroups) {
            const auto [lo, hi] = std::ranges::minmax(g);
            c.sigma_within += (hi - lo) / range_constant_d2(g.size()) / static_cast<double>(c.subgroups);
        }
        break;
    case WithinSigma::average_sd:
        for (const auto& g : subgroups) {
            double m = 0, s = 0;
            for (const double v : g)
                m += v / static_cast<double>(g.size());
            for (const double v : g)
                s += (v - m) * (v - m);
            s = std::sqrt(s / static_cast<double>(g.size() - 1));
            c.sigma_within += s / sd_constant_c4(g.size()) / static_cast<double>(c.subgroups);
        }
        break;
    case WithinSigma::moving_range: {
        const std::size_t k = o.moving_range_span;
        if (k < 2 || k > c.n)
            throw std::invalid_argument("process_capability: invalid moving-range span");
        double total = 0;
        for (std::size_t j = k - 1; j < c.n; ++j) {
            const auto first = all.begin() + static_cast<std::ptrdiff_t>(j + 1 - k);
            const auto [lo, hi] = std::minmax_element(first, first + static_cast<std::ptrdiff_t>(k));
            total += *hi - *lo;
        }
        c.sigma_within = total / static_cast<double>(c.n - k + 1) / range_constant_d2(k);
        break;
    }
    }
    if (!(c.sigma_within > 0) || !(c.sigma_overall > 0))
        throw std::invalid_argument("process_capability: the data show no spread");
    const double z = random::normal_quantile(1 - (1 - o.confidence) / 2);
    indices(o, c.mean, c.sigma_within, nd, z, c.cp, c.cpl, c.cpu, c.cpk);
    indices(o, c.mean, c.sigma_overall, nd, z, c.pp, c.ppl, c.ppu, c.ppk);
    if (!std::isnan(o.lsl) && !std::isnan(o.usl)) {
        const double target = std::isnan(o.target) ? (o.lsl + o.usl) / 2 : o.target;
        const double dev = (c.mean - target) / c.sigma_overall;
        const double value = (o.usl - o.lsl) / (6 * std::sqrt(c.sigma_overall * c.sigma_overall +
                                                              (c.mean - target) * (c.mean - target)));
        const double df = nd * (1 + dev * dev) / (1 + 2 * dev * dev);
        c.cpm = chi_interval(value, df, o.confidence);
    }
    c.expected_ppm_within = expected_ppm(o, c.mean, c.sigma_within);
    c.expected_ppm_overall = expected_ppm(o, c.mean, c.sigma_overall);
    double out = 0;
    for (const double v : all)
        out += (!std::isnan(o.lsl) && v < o.lsl) || (!std::isnan(o.usl) && v > o.usl);
    c.observed_ppm = 1e6 * out / nd;
    return c;
}

ProcessCapability process_capability(const std::vector<double>& values, CapabilityOptions options) {
    options.within = WithinSigma::moving_range;
    std::vector<std::vector<double>> groups;
    for (const double v : values)
        groups.push_back({v});
    return process_capability(groups, options);
}

} // namespace datamunge::stats
