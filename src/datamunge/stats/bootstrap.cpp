#include <datamunge/stats/bootstrap.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::stats {

namespace {

// R "type 7" quantile of an already-sorted vector: linear interpolation at h = (N-1) p.
double quantile_sorted(const std::vector<double>& s, double p) {
    if (s.empty()) return 0.0;
    if (s.size() == 1) return s[0];
    const double h = static_cast<double>(s.size() - 1) * std::min(std::max(p, 0.0), 1.0);
    const std::size_t lo = static_cast<std::size_t>(std::floor(h));
    if (lo + 1 >= s.size()) return s.back();
    return s[lo] + (h - static_cast<double>(lo)) * (s[lo + 1] - s[lo]);
}

} // namespace

Bootstrap::Bootstrap(BootstrapOptions options) : options_(options) {
    if (!(options_.confidence_level > 0.0 && options_.confidence_level < 1.0))
        throw std::invalid_argument("Bootstrap: confidence_level must be in (0, 1)");
    if (options_.num_resamples == 0) throw std::invalid_argument("Bootstrap: num_resamples must be >= 1");
}

BootstrapResult Bootstrap::run(const std::vector<double>& sample, const SampleStatistic& statistic) const {
    const std::size_t n = sample.size();
    if (n == 0) throw std::invalid_argument("Bootstrap::run: sample must not be empty");

    BootstrapResult r;
    r.confidence_level = options_.confidence_level;
    r.estimate = statistic(sample);

    // ---- Draw B resamples with replacement and evaluate the statistic on each ----
    std::mt19937_64 rng(options_.seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    r.replicates.resize(options_.num_resamples);
    std::vector<double> resample(n);
    for (std::size_t b = 0; b < options_.num_resamples; ++b) {
        for (std::size_t i = 0; i < n; ++i) resample[i] = sample[pick(rng)];
        r.replicates[b] = statistic(resample);
    }

    const double mean = std::accumulate(r.replicates.begin(), r.replicates.end(), 0.0) /
                        static_cast<double>(options_.num_resamples);
    r.bias = mean - r.estimate;
    double var = 0.0;
    for (const double v : r.replicates) var += (v - mean) * (v - mean);
    r.standard_error =
        options_.num_resamples > 1 ? std::sqrt(var / static_cast<double>(options_.num_resamples - 1)) : 0.0;

    const double alpha = 1.0 - options_.confidence_level;
    const double z_lo = random::normal_quantile(alpha / 2.0);
    const double z_hi = random::normal_quantile(1.0 - alpha / 2.0);

    std::vector<double> sorted = r.replicates;
    std::sort(sorted.begin(), sorted.end());
    const double q_lo = quantile_sorted(sorted, alpha / 2.0);
    const double q_hi = quantile_sorted(sorted, 1.0 - alpha / 2.0);

    r.percentile_interval = {q_lo, q_hi};
    r.basic_interval = {2.0 * r.estimate - q_hi, 2.0 * r.estimate - q_lo};
    r.normal_interval = {(r.estimate - r.bias) + z_lo * r.standard_error,
                         (r.estimate - r.bias) + z_hi * r.standard_error};

    // ---- BCa: bias-correction z0 from the replicate ECDF, acceleration a from the jackknife ----
    std::size_t less = 0;
    for (const double v : r.replicates) less += (v < r.estimate);
    double bca_lo = q_lo, bca_hi = q_hi; // fall back to percentile if BCa is ill-defined
    if (less > 0 && less < options_.num_resamples) {
        const double z0 = random::normal_quantile(static_cast<double>(less) /
                                                       static_cast<double>(options_.num_resamples));
        // Jackknife leave-one-out replicates of the statistic for the acceleration.
        std::vector<double> loo(n);
        std::vector<double> minus(n > 0 ? n - 1 : 0);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t t = 0;
            for (std::size_t j = 0; j < n; ++j)
                if (j != i) minus[t++] = sample[j];
            loo[i] = statistic(minus);
        }
        const double loo_mean = std::accumulate(loo.begin(), loo.end(), 0.0) / static_cast<double>(n);
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = loo_mean - loo[i];
            num += d * d * d;
            den += d * d;
        }
        const double a = den > 0.0 ? num / (6.0 * std::pow(den, 1.5)) : 0.0;
        auto adjust = [&](double z_alpha) {
            const double denom = 1.0 - a * (z0 + z_alpha);
            if (std::fabs(denom) < 1e-12) return random::normal_cdf(z0);
            return random::normal_cdf(z0 + (z0 + z_alpha) / denom);
        };
        const double p1 = adjust(z_lo), p2 = adjust(z_hi);
        if (std::isfinite(p1) && std::isfinite(p2)) {
            bca_lo = quantile_sorted(sorted, p1);
            bca_hi = quantile_sorted(sorted, p2);
        }
    }
    r.bca_interval = {bca_lo, bca_hi};
    return r;
}

} // namespace datamunge::stats
