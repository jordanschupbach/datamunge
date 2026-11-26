/// @file acceptance_sampling.cpp
/// @brief OC functions, plan search and rectifying-inspection measures for attribute sampling.
#include <datamunge/stats/acceptance_sampling.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::stats {

/// @brief Implementation helpers for acceptance sampling.
namespace {
/// @brief log of the binomial coefficient.
/// @param n Total.
/// @param k Chosen.
/// @return log C(n, k).
double log_choose(double n, double k) { return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1); }

/// @brief Binomial CDF P(X <= c) by summing probabilities in log space.
/// @param n Trials.
/// @param c Cutoff.
/// @param p Probability.
/// @return CDF.
double binomial_cdf(std::size_t n, std::size_t c, double p) {
    if (c >= n || p <= 0)
        return 1.0;
    if (p >= 1)
        return 0.0;
    const double nd = static_cast<double>(n), lp = std::log(p), lq = std::log1p(-p);
    double s = 0;
    for (std::size_t k = 0; k <= c; ++k) {
        const double kd = static_cast<double>(k);
        s += std::exp(log_choose(nd, kd) + kd * lp + (nd - kd) * lq);
    }
    return std::min(1.0, s);
}

/// @brief Hypergeometric CDF: sample n from N containing D defectives.
///
/// Terms are built by the ratio recurrence outward from the mode (so all are at most 1) and normalized by
/// their total, which avoids the cancellation of log-factorial differences.
/// @param n Sample size.
/// @param c Cutoff.
/// @param big_n Lot size.
/// @param d Defectives in the lot.
/// @return CDF.
double hypergeometric_cdf(std::size_t n, std::size_t c, std::size_t big_n, std::size_t d) {
    const std::size_t lo = n + d > big_n ? n + d - big_n : 0, hi = std::min(n, d);
    if (c >= hi)
        return 1.0;
    if (c < lo)
        return 0.0;
    const double nd = static_cast<double>(n), bn = static_cast<double>(big_n), dd = static_cast<double>(d);
    // ratio(k) = P(k + 1) / P(k).
    const auto ratio = [&](std::size_t k) {
        const double kd = static_cast<double>(k);
        return (dd - kd) * (nd - kd) / ((kd + 1) * (bn - dd - nd + kd + 1));
    };
    auto mode = static_cast<std::size_t>(std::floor((nd + 1) * (dd + 1) / (bn + 2)));
    mode = std::clamp(mode, lo, hi);
    std::vector<double> term(hi - lo + 1, 0.0);
    term[mode - lo] = 1;
    for (std::size_t k = mode; k > lo; --k)
        term[k - 1 - lo] = term[k - lo] / ratio(k - 1);
    for (std::size_t k = mode; k < hi; ++k)
        term[k + 1 - lo] = term[k - lo] * ratio(k);
    double total = 0, below = 0;
    for (std::size_t k = lo; k <= hi; ++k) {
        total += term[k - lo];
        if (k <= c)
            below += term[k - lo];
    }
    return std::min(1.0, below / total);
}

/// @brief Poisson CDF P(X <= c) with mean lambda.
/// @param c Cutoff.
/// @param lambda Mean.
/// @return CDF.
double poisson_cdf(std::size_t c, double lambda) {
    if (lambda <= 0)
        return 1.0;
    double s = 0;
    for (std::size_t k = 0; k <= c; ++k) {
        const double kd = static_cast<double>(k);
        s += std::exp(kd * std::log(lambda) - lambda - std::lgamma(kd + 1));
    }
    return std::min(1.0, s);
}
} // namespace

double acceptance_probability(const AttributePlan& plan, double p, const SamplingOptions& options) {
    if (plan.n == 0)
        throw std::invalid_argument("acceptance_probability: n must be positive");
    if (!(p >= 0) || (options.model != SamplingModel::poisson && p > 1))
        throw std::invalid_argument("acceptance_probability: quality out of range");
    switch (options.model) {
    case SamplingModel::binomial:
        return binomial_cdf(plan.n, plan.c, p);
    case SamplingModel::poisson:
        return poisson_cdf(plan.c, static_cast<double>(plan.n) * p);
    case SamplingModel::hypergeometric: {
        if (options.lot_size < plan.n)
            throw std::invalid_argument("acceptance_probability: lot size must be at least n");
        const auto d = static_cast<std::size_t>(std::llround(p * static_cast<double>(options.lot_size)));
        return hypergeometric_cdf(plan.n, plan.c, options.lot_size, d);
    }
    }
    return 0.0;
}

AttributePlan find_attribute_plan(double aql, double producer_risk, double rql, double consumer_risk,
                                  const SamplingOptions& options) {
    const bool poisson = options.model == SamplingModel::poisson;
    if (!(aql >= 0) || !(rql > aql) || (!poisson && rql > 1))
        throw std::invalid_argument("find_attribute_plan: need 0 <= AQL < RQL (<= 1)");
    if (!(producer_risk > 0 && producer_risk < 1) || !(consumer_risk > 0 && consumer_risk < 1))
        throw std::invalid_argument("find_attribute_plan: risks must be in (0, 1)");
    const std::size_t limit = options.model == SamplingModel::hypergeometric
                                  ? std::min(options.max_sample, options.lot_size)
                                  : options.max_sample;
    if (options.model == SamplingModel::hypergeometric && options.lot_size == 0)
        throw std::invalid_argument("find_attribute_plan: the hypergeometric model needs a lot size");
    AttributePlan plan{1, 0};
    while (plan.n <= limit) {
        if (acceptance_probability(plan, rql, options) > consumer_risk)
            ++plan.n;
        else if (acceptance_probability(plan, aql, options) < 1 - producer_risk)
            ++plan.c;
        else
            return plan;
        if (plan.c >= plan.n)
            ++plan.n;
    }
    throw std::invalid_argument("find_attribute_plan: no plan within the sample-size limit");
}

RectifyingPerformance rectifying_performance(const AttributePlan& plan, double p, std::size_t lot_size,
                                             SamplingOptions options) {
    if (lot_size < plan.n)
        throw std::invalid_argument("rectifying_performance: lot size must be at least n");
    options.lot_size = lot_size;
    RectifyingPerformance r;
    r.acceptance = acceptance_probability(plan, p, options);
    const double big_n = static_cast<double>(lot_size), n = static_cast<double>(plan.n);
    r.aoq = p * r.acceptance * (big_n - n) / big_n;
    r.ati = n + (1 - r.acceptance) * (big_n - n);
    return r;
}

Aoql average_outgoing_quality_limit(const AttributePlan& plan, std::size_t lot_size, SamplingOptions options) {
    const auto aoq = [&](double p) { return rectifying_performance(plan, p, lot_size, options).aoq; };
    // AOQ is unimodal in p; locate the maximum on a grid, then refine (continuous models only).
    const double top = options.model == SamplingModel::poisson ? 20.0 / static_cast<double>(plan.n) + 1 : 1.0;
    const int grid = 2000;
    double best = 0, at = 0;
    for (int i = 1; i <= grid; ++i) {
        const double p = top * i / grid, v = aoq(p);
        if (v > best)
            best = v, at = p;
    }
    if (options.model != SamplingModel::hypergeometric) {
        double a = std::max(0.0, at - top / grid), b = std::min(top, at + top / grid);
        const double g = (std::sqrt(5.0) - 1) / 2;
        for (int it = 0; it < 200; ++it) {
            const double x1 = b - g * (b - a), x2 = a + g * (b - a);
            if (aoq(x1) < aoq(x2))
                a = x1;
            else
                b = x2;
        }
        const double p = (a + b) / 2;
        if (aoq(p) > best)
            best = aoq(p), at = p;
    } else {
        // Hypergeometric quality is discrete: D / N.
        for (std::size_t d = 0; d <= lot_size; ++d) {
            const double p = static_cast<double>(d) / static_cast<double>(lot_size), v = aoq(p);
            if (v > best)
                best = v, at = p;
        }
    }
    return {best, at};
}

} // namespace datamunge::stats
