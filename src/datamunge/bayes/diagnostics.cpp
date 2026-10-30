/// @file diagnostics.cpp
/// @brief Effective sample size, split R-hat, E-BFMI and posterior summaries.

#include <datamunge/bayes/diagnostics.hpp>

#include <datamunge/bayes/nuts.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::bayes {

/// @brief Variance components, autocovariances and quantiles shared by the diagnostics.
namespace {

/// @brief Throws unless the chains are non-empty, equal-length and at least 4 draws long.
void check_chains(const std::vector<std::vector<double>>& chains, const char* who) {
    if (chains.empty()) throw std::invalid_argument(std::string(who) + ": no chains");
    const std::size_t n = chains.front().size();
    if (n < 4) throw std::invalid_argument(std::string(who) + ": chains need at least 4 draws");
    for (const auto& c : chains)
        if (c.size() != n) throw std::invalid_argument(std::string(who) + ": chains differ in length");
}

/// @brief Arithmetic mean.
double mean(const std::vector<double>& x) {
    double s = 0.0;
    for (const double v : x) s += v;
    return s / static_cast<double>(x.size());
}

/// @brief Within-chain variance W and the pooled estimate var+ for equal-length chains.
void variance_components(const std::vector<std::vector<double>>& chains, double& w, double& var_plus) {
    const auto m = static_cast<double>(chains.size());
    const auto n = static_cast<double>(chains.front().size());
    std::vector<double> means;
    w = 0.0;
    for (const auto& c : chains) {
        const double mu = mean(c);
        means.push_back(mu);
        double ss = 0.0;
        for (const double v : c) ss += (v - mu) * (v - mu);
        w += ss / (n - 1.0);
    }
    w /= m;
    double b_over_n = 0.0;
    if (chains.size() > 1) {
        const double grand = mean(means);
        for (const double mu : means) b_over_n += (mu - grand) * (mu - grand);
        b_over_n /= (m - 1.0);
    }
    var_plus = (n - 1.0) / n * w + b_over_n;
}

/// @brief Biased (divide-by-N) autocovariance of one chain at lags 0..N-1.
std::vector<double> autocovariance(const std::vector<double>& x) {
    const std::size_t n = x.size();
    const double mu = mean(x);
    std::vector<double> g(n, 0.0);
    for (std::size_t t = 0; t < n; ++t) {
        double s = 0.0;
        for (std::size_t i = 0; i + t < n; ++i) s += (x[i] - mu) * (x[i + t] - mu);
        g[t] = s / static_cast<double>(n);
    }
    return g;
}

/// @brief Linearly interpolated quantile (R type 7).
double quantile(std::vector<double> x, const double p) {
    std::sort(x.begin(), x.end());
    const double h = (static_cast<double>(x.size()) - 1.0) * p;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, x.size() - 1);
    return x[lo] + (h - static_cast<double>(lo)) * (x[hi] - x[lo]);
}

} // namespace

std::vector<double> draws_of(const std::vector<std::vector<double>>& samples, const std::size_t index) {
    std::vector<double> out;
    out.reserve(samples.size());
    for (const auto& row : samples) out.push_back(row.at(index));
    return out;
}

std::vector<double> draws_mean(const std::vector<std::vector<double>>& samples) {
    if (samples.empty()) throw std::invalid_argument("draws_mean: no draws");
    const std::size_t d = samples.front().size();
    std::vector<double> m(d, 0.0);
    for (const auto& row : samples) {
        if (row.size() != d) throw std::invalid_argument("draws_mean: ragged draws");
        for (std::size_t j = 0; j < d; ++j) m[j] += row[j];
    }
    for (double& v : m) v /= static_cast<double>(samples.size());
    return m;
}

std::vector<std::vector<double>> draws_covariance(const std::vector<std::vector<double>>& samples) {
    if (samples.size() < 2) throw std::invalid_argument("draws_covariance: need at least 2 draws");
    const auto m = draws_mean(samples);
    const std::size_t d = m.size();
    std::vector<std::vector<double>> c(d, std::vector<double>(d, 0.0));
    for (const auto& row : samples)
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j < d; ++j) c[i][j] += (row[i] - m[i]) * (row[j] - m[j]);
    for (auto& r : c)
        for (double& v : r) v /= static_cast<double>(samples.size() - 1);
    return c;
}

std::vector<std::vector<double>> chains_of(const std::vector<NUTSResult>& chains, const std::size_t index) {
    std::vector<std::vector<double>> out;
    out.reserve(chains.size());
    for (const auto& c : chains) out.push_back(draws_of(c.samples, index));
    return out;
}

double effective_sample_size(const std::vector<std::vector<double>>& chains) {
    check_chains(chains, "effective_sample_size");
    const std::size_t m = chains.size();
    const std::size_t n = chains.front().size();
    double w = 0.0, var_plus = 0.0;
    variance_components(chains, w, var_plus);
    if (!(var_plus > 0.0)) return static_cast<double>(m * n);

    std::vector<double> mean_acov(n, 0.0);
    for (const auto& c : chains) {
        const auto g = autocovariance(c);
        for (std::size_t t = 0; t < n; ++t) mean_acov[t] += g[t] / static_cast<double>(m);
    }
    const auto rho = [&](const std::size_t t) { return 1.0 - (w - mean_acov[t]) / var_plus; };

    // Geyer's initial positive sequence of pair sums, made monotone.
    double sum_pairs = 0.0;
    double prev = std::numeric_limits<double>::infinity();
    for (std::size_t t = 0; t + 1 < n; t += 2) {
        double pair = (t == 0 ? 1.0 : rho(t)) + rho(t + 1);
        if (!(pair > 0.0)) break;
        pair = std::min(pair, prev);
        prev = pair;
        sum_pairs += pair;
    }
    const double tau = std::max(-1.0 + 2.0 * sum_pairs, 1.0 / std::log10(static_cast<double>(m * n)));
    return static_cast<double>(m * n) / tau;
}

double split_rhat(const std::vector<std::vector<double>>& chains) {
    check_chains(chains, "split_rhat");
    const std::size_t half = chains.front().size() / 2;
    const std::size_t n = chains.front().size();
    std::vector<std::vector<double>> split;
    for (const auto& c : chains) {
        split.emplace_back(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(half));
        split.emplace_back(c.begin() + static_cast<std::ptrdiff_t>(n - half), c.end());
    }
    double w = 0.0, var_plus = 0.0;
    variance_components(split, w, var_plus);
    if (!(w > 0.0)) return 1.0;
    return std::sqrt(var_plus / w);
}

double energy_bfmi(const std::vector<double>& energy) {
    if (energy.size() < 2) throw std::invalid_argument("energy_bfmi: need at least 2 energies");
    const double mu = mean(energy);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < energy.size(); ++i) {
        den += (energy[i] - mu) * (energy[i] - mu);
        if (i > 0) num += (energy[i] - energy[i - 1]) * (energy[i] - energy[i - 1]);
    }
    if (!(den > 0.0)) throw std::invalid_argument("energy_bfmi: energy is constant");
    return num / den;
}

PosteriorSummary summarize_draws(const std::vector<std::vector<double>>& chains) {
    check_chains(chains, "summarize_draws");
    std::vector<double> pooled;
    for (const auto& c : chains) pooled.insert(pooled.end(), c.begin(), c.end());
    PosteriorSummary s;
    s.mean = mean(pooled);
    double ss = 0.0;
    for (const double v : pooled) ss += (v - s.mean) * (v - s.mean);
    s.sd = std::sqrt(ss / (static_cast<double>(pooled.size()) - 1.0));
    s.q05 = quantile(pooled, 0.05);
    s.q50 = quantile(pooled, 0.50);
    s.q95 = quantile(pooled, 0.95);
    s.ess = effective_sample_size(chains);
    s.rhat = split_rhat(chains);
    s.mcse = s.sd / std::sqrt(s.ess);
    return s;
}

DensityHistogram posterior_histogram(const std::vector<double>& draws, const double lo, const double hi,
                                     const std::size_t bins) {
    if (draws.empty() || !(hi > lo) || bins == 0)
        throw std::invalid_argument("posterior_histogram: need draws, hi > lo and bins >= 1");
    DensityHistogram out;
    out.width = (hi - lo) / static_cast<double>(bins);
    std::vector<double> counts(bins, 0.0);
    for (const double v : draws) {
        if (!(v >= lo && v <= hi)) continue;
        const auto k = std::min(bins - 1, static_cast<std::size_t>((v - lo) / out.width));
        counts[k] += 1.0;
    }
    const double scale = 1.0 / (static_cast<double>(draws.size()) * out.width);
    for (std::size_t k = 0; k < bins; ++k) {
        out.centers.push_back(lo + (static_cast<double>(k) + 0.5) * out.width);
        out.density.push_back(counts[k] * scale);
    }
    return out;
}

} // namespace datamunge::bayes
