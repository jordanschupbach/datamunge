#include <datamunge/stats/u_statistic.hpp>

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

// First Hoeffding projection variance estimate: zeta_1 = Var of g_hat(x_i), where g_hat(x_i) is
// the average kernel value over all evaluated tuples containing observation i; then the
// leading-order Var(U_n) ~= m^2 zeta_1 / n.
double zeta1_from_projections(const std::vector<double>& g_sum, const std::vector<double>& g_cnt, double u) {
    double ss = 0.0;
    std::size_t used = 0;
    for (std::size_t i = 0; i < g_sum.size(); ++i) {
        if (g_cnt[i] <= 0.0) continue;
        const double gi = g_sum[i] / g_cnt[i];
        ss += (gi - u) * (gi - u);
        ++used;
    }
    return used > 1 ? ss / static_cast<double>(used - 1) : 0.0;
}

} // namespace

UStatistic::UStatistic(std::size_t degree, UStatisticOptions options) : degree_(degree), options_(options) {
    if (degree_ == 0) throw std::invalid_argument("UStatistic: degree must be >= 1");
}

UStatisticResult UStatistic::run(const std::vector<double>& sample, const SymmetricKernel& kernel) const {
    const std::size_t n = sample.size();
    const std::size_t m = degree_;
    if (m > n) throw std::invalid_argument("UStatistic::run: degree must not exceed the sample size");

    UStatisticResult r;
    r.degree = m;

    std::vector<double> g_sum(n, 0.0), g_cnt(n, 0.0);
    double u_sum = 0.0;

    const double total = binomial_coefficient(n, m);
    const bool complete = options_.max_terms == 0;
    r.complete = complete;

    auto accumulate = [&](const std::vector<std::size_t>& idx) {
        std::vector<double> tuple(m);
        for (std::size_t k = 0; k < m; ++k) tuple[k] = sample[idx[k]];
        const double h = kernel(tuple);
        u_sum += h;
        for (std::size_t k = 0; k < m; ++k) {
            g_sum[idx[k]] += h;
            g_cnt[idx[k]] += 1.0;
        }
    };

    if (complete) {
        // Enumerate every size-m subset (next-combination algorithm).
        std::vector<std::size_t> combo(m);
        std::iota(combo.begin(), combo.end(), 0);
        std::size_t count = 0;
        while (true) {
            accumulate(combo);
            ++count;
            std::size_t i = m;
            while (i-- > 0 && combo[i] == i + n - m) {}
            if (i == static_cast<std::size_t>(-1)) break;
            ++combo[i];
            for (std::size_t j = i + 1; j < m; ++j) combo[j] = combo[j - 1] + 1;
        }
        r.num_terms = count;
        r.estimate = u_sum / static_cast<double>(count);
        // For a complete U-statistic, g_hat(x_i) = g_sum[i] / C(n-1, m-1).
        const double denom = binomial_coefficient(n - 1, m - 1);
        std::vector<double> cnt(n, denom);
        r.zeta1 = zeta1_from_projections(g_sum, cnt, r.estimate);
    } else {
        // Incomplete U-statistic: draw max_terms random m-subsets (distinct indices per subset).
        std::mt19937_64 rng(options_.seed);
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<std::size_t> combo(m);
        for (std::size_t s = 0; s < options_.max_terms; ++s) {
            for (std::size_t k = 0; k < m; ++k) { // partial Fisher-Yates for a random m-subset
                std::uniform_int_distribution<std::size_t> pick(k, n - 1);
                std::swap(idx[k], idx[pick(rng)]);
                combo[k] = idx[k];
            }
            accumulate(combo);
        }
        r.num_terms = options_.max_terms;
        r.estimate = u_sum / static_cast<double>(options_.max_terms);
        r.zeta1 = zeta1_from_projections(g_sum, g_cnt, r.estimate);
    }
    (void)total;

    r.variance = static_cast<double>(m) * static_cast<double>(m) / static_cast<double>(n) * r.zeta1;
    r.standard_error = std::sqrt(std::max(0.0, r.variance));
    return r;
}

} // namespace datamunge::stats
