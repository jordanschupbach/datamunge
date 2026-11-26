/// @file rosenbaum.cpp
/// @brief Signed-rank, sign and m-test sensitivity bounds, sensitivity values and amplification.
#include <datamunge/causal/rosenbaum.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::causal {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Rejects gamma values below 1 or nonfinite.
/// @param gamma Sensitivity parameter.
/// @param who Function name for the message.
void check_gamma(double gamma, const char* who) {
    if (!(gamma >= 1) || !std::isfinite(gamma))
        throw std::invalid_argument(std::string(who) + ": gamma must be finite and at least 1");
}

/// @brief Upper normal tail probability.
/// @param z Deviate.
/// @return P(Z > z).
double upper_tail(double z) { return 0.5 * std::erfc(z / std::sqrt(2.0)); }

/// @brief Upper binomial tail P(X >= t) for X ~ Binomial(n, p), summed in log space.
/// @param t Threshold.
/// @param n Trials.
/// @param p Success probability.
/// @return The tail probability.
double binomial_upper(std::size_t t, std::size_t n, double p) {
    if (t == 0)
        return 1.0;
    if (t > n)
        return 0.0;
    double total = 0;
    for (std::size_t k = t; k <= n; ++k) {
        const double lk = std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0) +
                          static_cast<double>(k) * std::log(p) + static_cast<double>(n - k) * std::log1p(-p);
        total += std::exp(lk);
    }
    return std::min(1.0, total);
}

/// @brief Type-7 quantile of a sample (sorted in place).
/// @param v Values.
/// @param p Probability.
/// @return The quantile.
double quantile7(std::vector<double>& v, double p) {
    std::ranges::sort(v);
    const double h = (static_cast<double>(v.size()) - 1) * p;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

/// @brief Extreme null mean and variance of one set's treated score under bias gamma.
/// @param scores Scores of the units in the set, sorted ascending.
/// @param gamma Sensitivity parameter.
/// @param largest True for the largest expectation, false for the smallest.
/// @param mean Receives the expectation.
/// @param var Receives the variance (the larger one among tied expectations).
void separable(const std::vector<double>& scores, double gamma, bool largest, double& mean, double& var) {
    const std::size_t m = scores.size();
    mean = largest ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    var = -std::numeric_limits<double>::infinity();
    for (std::size_t a = 1; a < m; ++a) {
        // The a smallest scores get odds w_low, the rest odds w_high.
        const double w_low = largest ? 1.0 : gamma, w_high = largest ? gamma : 1.0;
        double s1 = 0, s2 = 0, den = 0;
        for (std::size_t j = 0; j < m; ++j) {
            const double w = j < a ? w_low : w_high;
            s1 += w * scores[j];
            s2 += w * scores[j] * scores[j];
            den += w;
        }
        const double mu = s1 / den, v = s2 / den - mu * mu;
        if (largest ? mu > mean : mu < mean) {
            mean = mu;
            var = v;
        } else if (mu == mean) {
            var = std::max(var, v);
        }
    }
}
} // namespace

RosenbaumBound rosenbaum_wilcoxon(const std::vector<double>& differences, double gamma) {
    check_gamma(gamma, "rosenbaum_wilcoxon");
    std::vector<double> d;
    for (const double x : differences) {
        if (!std::isfinite(x))
            throw std::invalid_argument("rosenbaum_wilcoxon: differences must be finite");
        if (x != 0)
            d.push_back(x);
    }
    if (d.empty())
        throw std::invalid_argument("rosenbaum_wilcoxon: no nonzero differences");
    const std::size_t n = d.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return std::abs(d[a]) < std::abs(d[b]); });
    std::vector<double> rank(n);
    for (std::size_t i = 0; i < n;) {
        std::size_t j = i;
        while (j + 1 < n && std::abs(d[order[j + 1]]) == std::abs(d[order[i]]))
            ++j;
        const double avg = 0.5 * static_cast<double>(i + j) + 1;
        for (std::size_t k = i; k <= j; ++k)
            rank[order[k]] = avg;
        i = j + 1;
    }
    double t = 0, s1 = 0, s2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (d[i] > 0)
            t += rank[i];
        s1 += rank[i];
        s2 += rank[i] * rank[i];
    }
    const double p_plus = gamma / (1 + gamma), p_minus = 1 / (1 + gamma);
    RosenbaumBound b;
    b.gamma = gamma;
    b.statistic = t;
    b.expectation = p_plus * s1;
    b.variance = p_plus * (1 - p_plus) * s2;
    b.deviate = (t - b.expectation) / std::sqrt(b.variance);
    b.p_upper = upper_tail(b.deviate);
    b.p_lower = upper_tail((t - p_minus * s1) / std::sqrt(b.variance));
    return b;
}

RosenbaumBound rosenbaum_sign_test(std::size_t treated_events, std::size_t control_events, double gamma) {
    check_gamma(gamma, "rosenbaum_sign_test");
    const std::size_t d = treated_events + control_events;
    if (d == 0)
        throw std::invalid_argument("rosenbaum_sign_test: no discordant pairs");
    const double p_plus = gamma / (1 + gamma);
    RosenbaumBound b;
    b.gamma = gamma;
    b.statistic = static_cast<double>(treated_events);
    b.expectation = static_cast<double>(d) * p_plus;
    b.variance = static_cast<double>(d) * p_plus * (1 - p_plus);
    b.deviate = std::numeric_limits<double>::quiet_NaN();
    b.p_upper = binomial_upper(treated_events, d, p_plus);
    b.p_lower = binomial_upper(treated_events, d, 1 - p_plus);
    return b;
}

RosenbaumBound rosenbaum_m_test(const linalg::DenseMatrix<double>& sets, double gamma, const MTestOptions& options) {
    check_gamma(gamma, "rosenbaum_m_test");
    if (!(options.inner >= 0) || !(options.trim >= options.inner) || !(options.scale_quantile > 0) ||
        !(options.scale_quantile < 1) || !std::isfinite(options.shift) || std::isnan(options.trim))
        throw std::invalid_argument("rosenbaum_m_test: bad options");
    const std::size_t n = sets.rows(), m = sets.cols();
    if (m < 2)
        throw std::invalid_argument("rosenbaum_m_test: need a treated column and at least one control column");
    linalg::DenseMatrix<double> y = sets;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(y(i, 0)))
            throw std::invalid_argument("rosenbaum_m_test: treated responses must be finite");
        y(i, 0) -= options.shift;
        for (std::size_t j = 1; j < m; ++j)
            if (std::isinf(y(i, j)))
                throw std::invalid_argument("rosenbaum_m_test: responses must be finite or NaN");
    }
    const bool scaled = std::isfinite(options.trim) || options.inner > 0;
    double h = 1;
    if (scaled) {
        std::vector<double> diffs;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < m; ++j)
                for (std::size_t k = 0; k < m; ++k)
                    if (j != k && !std::isnan(y(i, j)) && !std::isnan(y(i, k)))
                        diffs.push_back(std::abs(y(i, j) - y(i, k)));
        if (diffs.empty())
            throw std::invalid_argument("rosenbaum_m_test: no set with a control");
        h = quantile7(diffs, options.scale_quantile);
        if (!(h > 0))
            throw std::invalid_argument("rosenbaum_m_test: the scale is zero; raise scale_quantile");
    }
    const auto psi = [&](double x) {
        if (!scaled)
            return x;
        x /= h;
        const double ax = std::abs(x), sg = (x > 0) - (x < 0);
        if (!std::isfinite(options.trim))
            return sg * std::max(0.0, ax - options.inner);
        if (options.trim == options.inner)
            return sg * (ax > options.inner ? 1.0 : 0.0);
        return sg * std::min(1.0, std::max(0.0, ax - options.inner) / (options.trim - options.inner));
    };
    RosenbaumBound b;
    b.gamma = gamma;
    double mean_low = 0, var_low = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::size_t> present;
        for (std::size_t j = 0; j < m; ++j)
            if (!std::isnan(y(i, j)))
                present.push_back(j);
        if (present.size() < 2)
            continue;
        std::vector<double> scores;
        for (const std::size_t j : present) {
            double s = 0;
            for (const std::size_t k : present)
                if (k != j)
                    s += psi(y(i, j) - y(i, k));
            scores.push_back(s / static_cast<double>(present.size()));
        }
        b.statistic += scores[0];
        std::ranges::sort(scores);
        double mu = 0, v = 0;
        separable(scores, gamma, true, mu, v);
        b.expectation += mu;
        b.variance += v;
        separable(scores, gamma, false, mu, v);
        mean_low += mu;
        var_low += v;
    }
    if (!(b.variance > 0))
        throw std::invalid_argument("rosenbaum_m_test: no set with a control or no variation");
    b.deviate = (b.statistic - b.expectation) / std::sqrt(b.variance);
    b.p_upper = upper_tail(b.deviate);
    b.p_lower = upper_tail((b.statistic - mean_low) / std::sqrt(var_low));
    return b;
}

double rosenbaum_sensitivity_value(const std::function<double(double)>& upper_p_value, double alpha,
                                   double max_gamma) {
    if (!(alpha > 0 && alpha < 1) || !(max_gamma >= 1))
        throw std::invalid_argument("rosenbaum_sensitivity_value: bad alpha or max_gamma");
    if (upper_p_value(1.0) > alpha)
        return 1.0;
    if (upper_p_value(max_gamma) <= alpha)
        return max_gamma;
    double lo = 0, hi = std::log(max_gamma);
    for (int it = 0; it < 200 && hi - lo > 1e-12; ++it) {
        const double mid = 0.5 * (lo + hi);
        (upper_p_value(std::exp(mid)) <= alpha ? lo : hi) = mid;
    }
    return std::exp(lo);
}

double rosenbaum_amplify(double gamma, double lambda) {
    if (!(gamma > 1) || !(lambda > gamma) || !std::isfinite(lambda))
        throw std::invalid_argument("rosenbaum_amplify: need 1 < gamma < lambda");
    return (gamma * lambda - 1) / (lambda - gamma);
}

} // namespace datamunge::causal
