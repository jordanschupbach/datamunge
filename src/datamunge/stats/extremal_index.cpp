/// @file extremal_index.cpp
/// @brief Intervals and runs extremal-index estimators, runs declustering and the cluster bootstrap.
#include <datamunge/stats/extremal_index.hpp>

#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for extremal-index estimation.
namespace {
/// @brief Quiet NaN.
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/// @brief Intervals estimate from N - 1 inter-exceedance times.
/// @param gaps Inter-exceedance times (all >= 1).
/// @param short_gaps Output: whether the all-gaps-<=2 form was used.
/// @return \f$\hat\theta\f$ capped at 1.
double intervals_theta(const std::vector<std::size_t>& gaps, bool& short_gaps) {
    short_gaps = std::ranges::all_of(gaps, [](std::size_t t) { return t <= 2; });
    double s1 = 0, s2 = 0;
    for (const std::size_t g : gaps) {
        const double t = static_cast<double>(g);
        if (short_gaps)
            s1 += t, s2 += t * t;
        else
            s1 += t - 1, s2 += (t - 1) * (t - 2);
    }
    return std::min(1.0, 2 * s1 * s1 / (static_cast<double>(gaps.size()) * s2));
}

/// @brief Run length of the intervals estimator: the K-th largest gap, K = ceil(theta N) (0 when K >= N).
/// @param gaps Inter-exceedance times.
/// @param theta Estimate.
/// @return r.
std::size_t intervals_run_length(const std::vector<std::size_t>& gaps, double theta) {
    const double n = static_cast<double>(gaps.size() + 1);
    const auto k = static_cast<std::size_t>(std::ceil(theta * n - 1e-9));
    if (k >= gaps.size() + 1 || k == 0)
        return 0;
    auto sorted = gaps;
    std::ranges::sort(sorted, std::greater<>());
    return sorted[k - 1];
}

/// @brief Number of clusters for gaps split at T > r.
/// @param gaps Inter-exceedance times.
/// @param r Run length.
/// @return Cluster count.
std::size_t count_clusters(const std::vector<std::size_t>& gaps, std::size_t r) {
    return 1 + static_cast<std::size_t>(std::ranges::count_if(gaps, [r](std::size_t t) { return t > r; }));
}

/// @brief Point estimate from gaps (and its run length and cluster count).
/// @param gaps Inter-exceedance times.
/// @param options Options.
/// @param short_gaps Output flag (intervals).
/// @param r Output run length.
/// @param clusters Output cluster count.
/// @return \f$\hat\theta\f$.
double estimate(const std::vector<std::size_t>& gaps, const ExtremalIndexOptions& options, bool& short_gaps,
                std::size_t& r, std::size_t& clusters) {
    short_gaps = false;
    if (options.method == ExtremalIndexMethod::runs) {
        r = options.run_length;
        clusters = count_clusters(gaps, r);
        return static_cast<double>(clusters) / static_cast<double>(gaps.size() + 1);
    }
    const double theta = intervals_theta(gaps, short_gaps);
    r = intervals_run_length(gaps, theta);
    clusters = count_clusters(gaps, r);
    return theta;
}

/// @brief Type-7 sample quantile of sorted values.
/// @param v Sorted values.
/// @param p Probability.
/// @return Quantile.
double quantile7(const std::vector<double>& v, double p) {
    const double h = (static_cast<double>(v.size()) - 1) * p;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

/// @brief Exceedance time indices.
/// @param data Series.
/// @param u Threshold.
/// @return Indices with data > u.
std::vector<std::size_t> exceedance_times(const std::vector<double>& data, double u) {
    if (!std::ranges::all_of(data, [](double v) { return std::isfinite(v); }) || !std::isfinite(u))
        throw std::invalid_argument("extremal_index: data and threshold must be finite");
    std::vector<std::size_t> s;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (data[i] > u)
            s.push_back(i);
    return s;
}
} // namespace

ExtremalIndex extremal_index(const std::vector<double>& data, double threshold, const ExtremalIndexOptions& options) {
    if (options.run_length == 0 || !(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("extremal_index: need run_length >= 1 and confidence in (0, 1)");
    const auto s = exceedance_times(data, threshold);
    if (s.size() < 2)
        throw std::invalid_argument("extremal_index: need at least 2 exceedances");
    std::vector<std::size_t> gaps(s.size() - 1);
    for (std::size_t i = 1; i < s.size(); ++i)
        gaps[i - 1] = s[i] - s[i - 1];
    ExtremalIndex e;
    e.threshold = threshold, e.observations = data.size(), e.exceedances = s.size();
    e.theta = estimate(gaps, options, e.all_gaps_short, e.run_length, e.clusters);
    e.lower = e.upper = nan;
    if (options.bootstrap == 0 || e.clusters < 2)
        return e;

    // Cluster bootstrap: times[c] holds the gap preceding each exceedance of cluster c (the first cluster's
    // leading gap is a placeholder 1, as in Ferro and Segers / extRemes).
    std::vector<std::vector<std::size_t>> times(1, std::vector<std::size_t>{1});
    for (const std::size_t g : gaps) {
        if (g > e.run_length)
            times.emplace_back();
        times.back().push_back(g);
    }
    const std::size_t nc = times.size();
    random::SplitMix64 rng(options.seed);
    std::vector<double> thetas;
    thetas.reserve(options.bootstrap);
    for (std::size_t b = 0; b < options.bootstrap; ++b) {
        std::vector<std::size_t> boot;
        for (std::size_t c = 0; c < nc; ++c) {
            const auto& cl = times[rng.uniform_u64(0, nc - 1)];
            // Leading gap: kept for the first cluster (dropped when differencing); otherwise a resampled
            // inter-cluster gap.
            const std::size_t lead = c == 0 ? cl.front() : times[rng.uniform_u64(1, nc - 1)].front();
            if (c > 0)
                boot.push_back(lead);
            boot.insert(boot.end(), cl.begin() + 1, cl.end());
        }
        if (boot.empty())
            continue;
        bool sg = false;
        std::size_t r = 0, k = 0;
        thetas.push_back(estimate(boot, options, sg, r, k));
    }
    std::ranges::sort(thetas);
    const double a = (1 - options.confidence) / 2;
    e.lower = quantile7(thetas, a), e.upper = quantile7(thetas, 1 - a);
    e.bootstrap = thetas.size();
    return e;
}

Declustered decluster(const std::vector<double>& data, double threshold, std::size_t run_length) {
    Declustered d;
    d.exceedance_index = exceedance_times(data, threshold);
    for (std::size_t i = 0; i < d.exceedance_index.size(); ++i) {
        const std::size_t t = d.exceedance_index[i];
        if (i == 0 || t - d.exceedance_index[i - 1] > run_length) {
            d.cluster_maxima.push_back(data[t]);
            d.cluster_maximum_index.push_back(t);
        } else if (data[t] > d.cluster_maxima.back()) {
            d.cluster_maxima.back() = data[t];
            d.cluster_maximum_index.back() = t;
        }
        d.cluster.push_back(d.cluster_maxima.size() - 1);
    }
    return d;
}

} // namespace datamunge::stats
