#include <datamunge/stats/jackknife.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::stats {

namespace {

double binomial_coefficient(std::size_t n, std::size_t k) {
    if (k > n) return 0.0;
    k = std::min(k, n - k);
    double c = 1.0;
    for (std::size_t i = 0; i < k; ++i) c = c * static_cast<double>(n - i) / static_cast<double>(i + 1);
    return c;
}

// Evaluate the statistic on `sample` with the sorted index set `deleted` removed.
double statistic_without(const std::vector<double>& sample, const std::vector<std::size_t>& deleted,
                         const SampleStatistic& statistic) {
    std::vector<double> kept;
    kept.reserve(sample.size() - deleted.size());
    std::size_t d = 0;
    for (std::size_t i = 0; i < sample.size(); ++i) {
        if (d < deleted.size() && deleted[d] == i) { ++d; continue; }
        kept.push_back(sample[i]);
    }
    return statistic(kept);
}

} // namespace

JackknifeResult jackknife(const std::vector<double>& sample, const SampleStatistic& statistic) {
    const std::size_t n = sample.size();
    if (n < 2) throw std::invalid_argument("jackknife: sample must have at least 2 observations");

    JackknifeResult r;
    r.estimate = statistic(sample);
    r.leave_one_out.resize(n);
    std::vector<double> minus(n - 1);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t t = 0;
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) minus[t++] = sample[j];
        r.leave_one_out[i] = statistic(minus);
    }

    const double mean_loo = std::accumulate(r.leave_one_out.begin(), r.leave_one_out.end(), 0.0) /
                            static_cast<double>(n);
    r.bias = static_cast<double>(n - 1) * (mean_loo - r.estimate);
    r.bias_corrected = r.estimate - r.bias;

    double ss = 0.0;
    r.pseudo_values.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        ss += (r.leave_one_out[i] - mean_loo) * (r.leave_one_out[i] - mean_loo);
        r.pseudo_values[i] = static_cast<double>(n) * r.estimate - static_cast<double>(n - 1) * r.leave_one_out[i];
    }
    r.standard_error = std::sqrt(static_cast<double>(n - 1) / static_cast<double>(n) * ss);
    return r;
}

DeleteDJackknifeResult jackknife_delete_d(const std::vector<double>& sample, const SampleStatistic& statistic,
                                          DeleteDJackknifeOptions options) {
    const std::size_t n = sample.size();
    const std::size_t d = options.d;
    if (d == 0 || d >= n) throw std::invalid_argument("jackknife_delete_d: require 1 <= d < n");

    DeleteDJackknifeResult r;
    r.d = d;
    r.estimate = statistic(sample);

    const double total = binomial_coefficient(n, d);
    std::vector<double> reps;
    if (total <= static_cast<double>(options.max_subsets)) {
        // Enumerate every size-d deletion set via the standard next-combination algorithm.
        r.complete = true;
        std::vector<std::size_t> combo(d);
        std::iota(combo.begin(), combo.end(), 0);
        while (true) {
            reps.push_back(statistic_without(sample, combo, statistic));
            // Find the rightmost element that can still be incremented.
            std::size_t i = d;
            while (i-- > 0 && combo[i] == i + n - d) {}
            if (i == static_cast<std::size_t>(-1)) break; // all at their maxima: done
            ++combo[i];
            for (std::size_t j = i + 1; j < d; ++j) combo[j] = combo[j - 1] + 1;
        }
    } else {
        // Monte Carlo: sample max_subsets random size-d deletion sets.
        r.complete = false;
        std::mt19937_64 rng(options.seed);
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        reps.reserve(options.max_subsets);
        for (std::size_t s = 0; s < options.max_subsets; ++s) {
            for (std::size_t i = 0; i < d; ++i) { // partial Fisher-Yates: first d entries are the deletion set
                std::uniform_int_distribution<std::size_t> pick(i, n - 1);
                std::swap(idx[i], idx[pick(rng)]);
            }
            std::vector<std::size_t> deleted(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(d));
            std::sort(deleted.begin(), deleted.end());
            reps.push_back(statistic_without(sample, deleted, statistic));
        }
    }

    r.num_subsets = reps.size();
    const double mean = std::accumulate(reps.begin(), reps.end(), 0.0) / static_cast<double>(reps.size());
    double ss = 0.0;
    for (const double v : reps) ss += (v - mean) * (v - mean);
    // Shao & Wu (1989) delete-d jackknife variance: ((n-d)/(d * N)) * sum (theta_S - mean)^2.
    const double variance = static_cast<double>(n - d) / static_cast<double>(d) * (ss / static_cast<double>(reps.size()));
    r.standard_error = std::sqrt(std::max(0.0, variance));
    return r;
}

} // namespace datamunge::stats
