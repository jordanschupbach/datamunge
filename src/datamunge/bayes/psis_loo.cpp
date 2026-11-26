/// @file psis_loo.cpp
/// @brief Generalized Pareto tail fitting, PSIS weights, PSIS-LOO and paired model comparison.
#include <datamunge/bayes/psis_loo.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::bayes {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Log of the sum of exponentials.
/// @param v Values.
/// @return log(sum(exp(v))).
double log_sum_exp(const std::vector<double>& v) {
    const double m = *std::ranges::max_element(v);
    if (!std::isfinite(m))
        return m;
    double s = 0;
    for (const double x : v)
        s += std::exp(x - m);
    return m + std::log(s);
}

/// @brief Generalized Pareto fit.
struct GpdFit {
    double k{};     ///< Shape (after the weakly informative adjustment).
    double sigma{}; ///< Scale.
};

/// @brief Zhang-Stephens empirical Bayes fit of a generalized Pareto distribution, as loo::gpdfit.
/// @param x Positive excesses sorted ascending.
/// @return Shape and scale.
GpdFit gpd_fit(const std::vector<double>& x) {
    const std::size_t n = x.size();
    const double prior = 3;
    const std::size_t m = 30 + static_cast<std::size_t>(std::floor(std::sqrt(static_cast<double>(n))));
    const double xstar = x[static_cast<std::size_t>(std::floor(n / 4.0 + 0.5)) - 1];
    std::vector<double> theta(m), l_theta(m);
    const auto profile_k = [&](double t) {
        double s = 0;
        for (const double v : x)
            s += std::log1p(-t * v);
        return s / static_cast<double>(n);
    };
    for (std::size_t j = 0; j < m; ++j) {
        theta[j] = 1 / x[n - 1] + (1 - std::sqrt(static_cast<double>(m) / (static_cast<double>(j) + 0.5))) /
                                      prior / xstar;
        const double k = profile_k(theta[j]);
        l_theta[j] = static_cast<double>(n) * (std::log(-theta[j] / k) - k - 1);
    }
    const double norm = log_sum_exp(l_theta);
    double theta_hat = 0;
    for (std::size_t j = 0; j < m; ++j)
        theta_hat += theta[j] * std::exp(l_theta[j] - norm);
    double k = profile_k(theta_hat);
    const double sigma = -k / theta_hat;
    const double a = 10, nd = static_cast<double>(n);
    k = k * nd / (nd + a) + a * 0.5 / (nd + a);
    if (std::isnan(k))
        k = std::numeric_limits<double>::infinity();
    return {k, sigma};
}
} // namespace

ParetoSmoothedWeights pareto_smoothed_importance_sampling(const std::vector<double>& log_ratios,
                                                          double relative_efficiency) {
    const std::size_t s = log_ratios.size();
    if (s < 2)
        throw std::invalid_argument("pareto_smoothed_importance_sampling: need at least two draws");
    if (!(relative_efficiency > 0) || !std::isfinite(relative_efficiency))
        throw std::invalid_argument("pareto_smoothed_importance_sampling: r_eff must be positive");
    if (!std::ranges::all_of(log_ratios, [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("pareto_smoothed_importance_sampling: log ratios must be finite");
    const double top = *std::ranges::max_element(log_ratios);
    std::vector<double> lw(s);
    for (std::size_t i = 0; i < s; ++i)
        lw[i] = log_ratios[i] - top;
    ParetoSmoothedWeights out;
    out.pareto_k = std::numeric_limits<double>::infinity();
    const double sd = static_cast<double>(s);
    const auto tail = static_cast<std::size_t>(
        std::ceil(std::min(0.2 * sd, 3 * std::sqrt(sd / relative_efficiency))));
    out.tail_length = 0;
    if (tail >= 5 && tail < s) {
        std::vector<std::size_t> order(s);
        std::iota(order.begin(), order.end(), 0);
        std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) { return lw[a] < lw[b]; });
        const std::size_t first = s - tail;
        const double lo = lw[order[first]], hi = lw[order[s - 1]];
        if (std::abs(hi - lo) >= std::numeric_limits<double>::epsilon() / 100) {
            const double exp_cutoff = std::exp(lw[order[first - 1]]);
            std::vector<double> excess(tail);
            for (std::size_t j = 0; j < tail; ++j)
                excess[j] = std::exp(lw[order[first + j]]) - exp_cutoff;
            const GpdFit fit = gpd_fit(excess);
            out.pareto_k = fit.k;
            if (std::isfinite(fit.k) && fit.sigma > 0) {
                for (std::size_t j = 0; j < tail; ++j) {
                    const double p = (static_cast<double>(j) + 0.5) / static_cast<double>(tail);
                    const double q = fit.sigma * std::expm1(-fit.k * std::log1p(-p)) / fit.k;
                    lw[order[first + j]] = std::log(q + exp_cutoff);
                }
                out.tail_length = tail;
            }
        }
    }
    for (double& v : lw)
        v = std::min(v, 0.0);
    const double norm = log_sum_exp(lw);
    double ss = 0;
    for (double& v : lw) {
        v -= norm;
        ss += std::exp(2 * v);
    }
    out.log_weights = std::move(lw);
    out.effective_sample_size = relative_efficiency / ss;
    return out;
}

double pareto_k_threshold(std::size_t draws) {
    if (draws < 2)
        throw std::invalid_argument("pareto_k_threshold: need at least two draws");
    return std::min(1 - 1 / std::log10(static_cast<double>(draws)), 0.7);
}

PsisLooResult psis_loo(const std::vector<std::vector<double>>& log_likelihood,
                       const std::vector<double>& relative_efficiency) {
    const std::size_t s = log_likelihood.size();
    if (s < 2 || log_likelihood[0].empty())
        throw std::invalid_argument("psis_loo: need at least two draws and one observation");
    const std::size_t n = log_likelihood[0].size();
    for (const auto& row : log_likelihood) {
        if (row.size() != n)
            throw std::invalid_argument("psis_loo: ragged log-likelihood rows");
        if (!std::ranges::all_of(row, [](double x) { return std::isfinite(x); }))
            throw std::invalid_argument("psis_loo: log likelihoods must be finite");
    }
    if (!relative_efficiency.empty() && relative_efficiency.size() != n)
        throw std::invalid_argument("psis_loo: one r_eff per observation is required");
    PsisLooResult out;
    out.k_threshold = pareto_k_threshold(s);
    std::vector<double> ll(s), ratio(s), terms(s);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < s; ++d) {
            ll[d] = log_likelihood[d][i];
            ratio[d] = -ll[d];
        }
        const auto w = pareto_smoothed_importance_sampling(
            ratio, relative_efficiency.empty() ? 1.0 : relative_efficiency[i]);
        for (std::size_t d = 0; d < s; ++d)
            terms[d] = ll[d] + w.log_weights[d];
        PsisLooPoint p;
        p.elpd = log_sum_exp(terms);
        const double lpd = log_sum_exp(ll) - std::log(static_cast<double>(s));
        p.effective_parameters = lpd - p.elpd;
        p.pareto_k = w.pareto_k;
        p.effective_sample_size = w.effective_sample_size;
        out.lpd += lpd;
        out.elpd_loo += p.elpd;
        out.p_loo += p.effective_parameters;
        out.high_k += p.pareto_k > out.k_threshold;
        out.pointwise.push_back(p);
    }
    const auto se = [&](auto get) {
        if (n < 2)
            return std::numeric_limits<double>::quiet_NaN();
        double mean = 0, ss = 0;
        for (const auto& p : out.pointwise)
            mean += get(p) / static_cast<double>(n);
        for (const auto& p : out.pointwise)
            ss += (get(p) - mean) * (get(p) - mean);
        return std::sqrt(static_cast<double>(n) * ss / static_cast<double>(n - 1));
    };
    out.se_elpd_loo = se([](const PsisLooPoint& p) { return p.elpd; });
    out.se_p_loo = se([](const PsisLooPoint& p) { return p.effective_parameters; });
    out.looic = -2 * out.elpd_loo;
    out.se_looic = 2 * out.se_elpd_loo;
    return out;
}

LooComparison loo_compare(const PsisLooResult& first, const PsisLooResult& second) {
    const std::size_t n = first.pointwise.size();
    if (n != second.pointwise.size() || n < 2)
        throw std::invalid_argument("loo_compare: fits must score the same (at least two) observations");
    std::vector<double> diff(n);
    double mean = 0;
    for (std::size_t i = 0; i < n; ++i) {
        diff[i] = second.pointwise[i].elpd - first.pointwise[i].elpd;
        mean += diff[i] / static_cast<double>(n);
    }
    double ss = 0;
    for (const double d : diff)
        ss += (d - mean) * (d - mean);
    return {mean * static_cast<double>(n), std::sqrt(static_cast<double>(n) * ss / static_cast<double>(n - 1))};
}

} // namespace datamunge::bayes
