/// @file diagnostics.cpp
/// @brief Effective sample size, split R-hat, E-BFMI, posterior summaries and WAIC.

#include <datamunge/bayes/diagnostics.hpp>

#include <datamunge/algorithms/fft.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace datamunge::bayes {

/// @brief Variance components, autocovariances and quantiles shared by the diagnostics.
namespace {

/// @brief Throws unless the chains are nonempty, equal-length and meet the draw minimum.
void check_chains(const std::vector<std::vector<double>>& chains, const char* who, std::size_t minimum = 4) {
    if (chains.empty()) throw std::invalid_argument(std::string(who) + ": no chains");
    const std::size_t n = chains.front().size();
    if (n < minimum)
        throw std::invalid_argument(std::string(who) + ": chains need at least " + std::to_string(minimum) + " draws");
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

/// @brief Splits each chain in half, omitting the middle draw of odd-length chains.
std::vector<std::vector<double>> split_chains(const std::vector<std::vector<double>>& chains) {
    const auto half = static_cast<std::ptrdiff_t>(chains.front().size() / 2);
    std::vector<std::vector<double>> split;
    for (const auto& chain : chains) {
        split.emplace_back(chain.begin(), chain.begin() + half);
        split.emplace_back(chain.end() - half, chain.end());
    }
    return split;
}

/// @brief Validates and sorts original draws for a pooled median or pooled tail quantiles.
std::vector<double> sorted_finite_draws(
    const std::vector<std::vector<double>>& chains, const char* who, std::size_t minimum = 4) {
    check_chains(chains, who, minimum);
    std::vector<double> pooled;
    for (const auto& chain : chains) {
        for (const double draw : chain) {
            if (!std::isfinite(draw))
                throw std::invalid_argument(std::string(who) + ": draws must be finite");
            pooled.push_back(draw);
        }
    }
    std::sort(pooled.begin(), pooled.end());
    return pooled;
}

/// @brief Normalizes retained ranks to Blom normal scores, optionally folding before ranking.
std::vector<std::vector<double>> rank_scores(
    const std::vector<std::vector<double>>& split, const double* fold_center = nullptr) {
    const std::size_t n = split.front().size();
    std::vector<std::pair<long double, std::size_t>> ordered;
    for (const auto& chain : split) {
        for (const long double draw : chain) {
            // Half the distance preserves its order and avoids overflow on platforms
            // where long double has the same exponent range as double.
            const long double value = fold_center ? std::abs(draw / 2 - *fold_center / 2.0L) : draw;
            ordered.emplace_back(value, ordered.size());
        }
    }
    std::sort(ordered.begin(), ordered.end());
    std::vector<std::vector<double>> scores(split.size(), std::vector<double>(n));
    for (std::size_t first = 0, last = 0; first < ordered.size(); first = last) {
        last = first + 1;
        while (last < ordered.size() && ordered[last].first == ordered[first].first)
            ++last;
        const double rank = std::midpoint(static_cast<double>(first + 1), static_cast<double>(last));
        const double reflection = static_cast<double>(ordered.size()) + 1.0 - rank;
        const double probability = (std::min(rank, reflection) - 0.375) / (ordered.size() + 0.25);
        double score = random::normal_quantile(probability);
        // Refine the approximate inverse in the lower tail, using erfc to avoid cancellation.
        score -= (0.5 * std::erfc(-score / std::numbers::sqrt2) - probability) /
            random::normal_pdf(score);
        if (rank > reflection)
            score = -score;
        for (std::size_t i = first; i < last; ++i) {
            const auto index = ordered[i].second;
            scores[index / n][index % n] = score;
        }
    }
    return scores;
}

/// @brief Computes biased autocovariances of bounded scores, using zero-padded FFT for long chains.
std::vector<double> score_autocovariance(const std::vector<double>& scores) {
    const std::size_t n = scores.size();
    if (n < 64)
        return autocovariance(scores);
    const std::size_t max_power = std::bit_floor(std::numeric_limits<std::size_t>::max());
    if (n > max_power / 2)
        throw std::length_error("effective sample size: FFT padding exceeds size limit");
    std::vector<std::complex<double>> centered(std::bit_ceil(2 * n));
    const double mu = mean(scores);
    for (std::size_t i = 0; i < n; ++i)
        centered[i] = scores[i] - mu;
    auto power = algorithms::fft(std::move(centered));
    for (auto& coefficient : power)
        coefficient = std::norm(coefficient);
    const auto sums = algorithms::fft(std::move(power), true);
    std::vector<double> covariance(n);
    for (std::size_t lag = 0; lag < n; ++lag)
        covariance[lag] = sums[lag].real() / static_cast<double>(n);
    return covariance;
}

/// @brief Estimates split-chain ESS using monotone positive pairs and the final even-lag adjustment.
double split_score_ess(const std::vector<std::vector<double>>& scores) {
    double w, var_plus;
    variance_components(scores, w, var_plus);
    if (var_plus == 0)
        return std::numeric_limits<double>::quiet_NaN();
    const std::size_t n = scores.front().size();
    std::vector<double> mean_covariance(n);
    for (const auto& chain : scores) {
        const auto covariance = score_autocovariance(chain);
        for (std::size_t lag = 0; lag < n; ++lag)
            mean_covariance[lag] += covariance[lag] / static_cast<double>(scores.size());
    }
    const auto rho = [&](std::size_t lag) {
        return lag == 0 ? 1.0 : 1.0 - (w - mean_covariance[lag]) / var_plus;
    };
    double tau = -1;
    double previous = std::numeric_limits<double>::infinity();
    std::size_t lag = 0;
    for (; lag + 5 < n; lag += 2) {
        const double pair = rho(lag) + rho(lag + 1);
        if (pair <= 0)
            break;
        previous = std::min(previous, pair);
        tau += 2 * previous;
    }
    tau += std::max(0.0, rho(lag));
    const double draws = static_cast<double>(scores.size()) * static_cast<double>(n);
    return draws / std::max(tau, 1.0 / std::log10(draws));
}

/// @brief Computes ESS of an inclusive pooled tail-quantile indicator in retained draws.
double tail_score_ess(std::vector<std::vector<double>> split, const std::vector<double>& sorted,
                      double probability, bool upper_tail) {
    const double position = static_cast<double>(sorted.size() - 1) * probability;
    const auto lo = static_cast<std::size_t>(std::floor(position));
    const auto hi = std::min(lo + 1, sorted.size() - 1);
    const double threshold = std::lerp(sorted[lo], sorted[hi], position - static_cast<double>(lo));
    for (auto& chain : split)
        for (auto& draw : chain)
            draw = upper_tail ? draw >= threshold : draw <= threshold;
    return split_score_ess(split);
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
    double w = 0.0, var_plus = 0.0;
    variance_components(split_chains(chains), w, var_plus);
    if (!(w > 0.0)) return 1.0;
    return std::sqrt(var_plus / w);
}

RankNormalizedRhat rank_normalized_rhat(const std::vector<std::vector<double>>& chains) {
    const auto pooled = sorted_finite_draws(chains, "rank_normalized_rhat");
    const auto split = split_chains(chains);
    const auto median = std::midpoint(pooled[(pooled.size() - 1) / 2], pooled[pooled.size() / 2]);
    const auto rhat = [](const std::vector<std::vector<double>>& scores) {
        double w, var_plus;
        variance_components(scores, w, var_plus);
        if (std::ranges::all_of(scores, [](const auto& chain) {
                return std::ranges::adjacent_find(chain, std::ranges::not_equal_to{}) == chain.end();
            })) {
            return var_plus > 0 ? std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt(var_plus / w);
    };
    const double rank = rhat(rank_scores(split));
    const double folded = rhat(rank_scores(split, &median));
    const double maximum = std::isnan(rank) || std::isnan(folded)
        ? std::numeric_limits<double>::quiet_NaN() : std::max(rank, folded);
    return {rank, folded, maximum};
}

BulkTailEffectiveSampleSize bulk_tail_effective_sample_size(
    const std::vector<std::vector<double>>& chains, double tail_probability) {
    if (!(tail_probability > 0 && tail_probability < 0.5))
        throw std::invalid_argument("bulk_tail_effective_sample_size: tail probability must be in (0, 0.5)");
    const auto pooled = sorted_finite_draws(chains, "bulk_tail_effective_sample_size", 6);
    auto split = split_chains(chains);
    const double bulk = split_score_ess(rank_scores(split));
    const double lower = tail_score_ess(split, pooled, tail_probability, false);
    const double upper = tail_score_ess(std::move(split), pooled, 1.0 - tail_probability, true);
    const double tail = std::isnan(lower) || std::isnan(upper)
        ? std::numeric_limits<double>::quiet_NaN() : std::min(lower, upper);
    return {bulk, lower, upper, tail};
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

WaicResult waic(const std::vector<std::vector<double>>& log_likelihood) {
    if (log_likelihood.size() < 2 || log_likelihood.front().empty())
        throw std::invalid_argument("waic: need at least two draws and one observation");
    std::vector<double> maxima = log_likelihood.front();
    for (const auto& draw : log_likelihood) {
        if (draw.size() != maxima.size())
            throw std::invalid_argument("waic: log likelihood rows must have equal length");
        for (std::size_t col = 0; col < draw.size(); ++col) {
            if (!std::isfinite(draw[col]))
                throw std::invalid_argument("waic: log likelihoods must be finite");
            maxima[col] = std::max(maxima[col], draw[col]);
        }
    }

    const auto as_score = [](long double value) {
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
            throw std::overflow_error("waic: score exceeds double range");
        return static_cast<double>(value);
    };
    WaicResult result{};
    result.pointwise.reserve(maxima.size());
    long double total_lpd = 0.0L;
    long double total_penalty = 0.0L;
    long double total_elpd = 0.0L;
    const auto draws = static_cast<long double>(log_likelihood.size());
    for (std::size_t col = 0; col < maxima.size(); ++col) {
        const long double maximum = maxima[col];
        long double exp_sum = 0.0L;
        long double mean = 0.0L;
        long double sum_squared_deviations = 0.0L;
        long double count = 0.0L;
        for (const auto& draw : log_likelihood) {
            const long double shifted = static_cast<long double>(draw[col]) - maximum;
            exp_sum += std::exp(shifted);
            const long double delta = shifted - mean;
            mean += delta / ++count;
            sum_squared_deviations += delta * (shifted - mean);
        }
        const long double lpd = maximum + std::log(exp_sum / draws);
        const long double penalty = sum_squared_deviations / (draws - 1.0L);
        const long double elpd = lpd - penalty;
        result.pointwise.push_back({as_score(lpd), as_score(penalty), as_score(elpd)});
        total_lpd += lpd;
        total_penalty += penalty;
        total_elpd += elpd;
    }
    result.log_predictive_density = as_score(total_lpd);
    result.effective_parameters = as_score(total_penalty);
    result.elpd = as_score(total_elpd);
    result.waic = as_score(-2.0L * total_elpd);
    return result;
}

} // namespace datamunge::bayes
