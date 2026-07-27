#include <datamunge/stats/v_statistic.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::stats {

VStatistic::VStatistic(std::size_t degree, VStatisticOptions options) : degree_(degree), options_(options) {
    if (degree_ == 0) throw std::invalid_argument("VStatistic: degree must be >= 1");
}

VStatisticResult VStatistic::run(const std::vector<double>& sample, const SymmetricKernel& kernel) const {
    const std::size_t n = sample.size();
    const std::size_t m = degree_;
    if (n == 0) throw std::invalid_argument("VStatistic::run: sample must not be empty");

    VStatisticResult r;
    r.degree = m;

    // Total number of tuples with replacement is n^m; enumerate exactly only if it is small.
    double total = 1.0;
    for (std::size_t k = 0; k < m; ++k) total *= static_cast<double>(n);
    const bool exact = options_.max_terms == 0 && total <= 5.0e6;
    r.exact = exact;

    std::vector<double> tuple(m);
    double sum = 0.0;

    if (exact) {
        std::vector<std::size_t> digit(m, 0); // an m-digit counter in base n enumerates every tuple
        const std::size_t tuples = static_cast<std::size_t>(total);
        for (std::size_t t = 0; t < tuples; ++t) {
            for (std::size_t k = 0; k < m; ++k) tuple[k] = sample[digit[k]];
            sum += kernel(tuple);
            for (std::size_t k = m; k-- > 0;) { // increment the base-n counter
                if (++digit[k] < n) break;
                digit[k] = 0;
            }
        }
        r.num_terms = tuples;
        r.estimate = sum / static_cast<double>(tuples);
    } else {
        // Monte Carlo: each index drawn uniformly and independently with replacement.
        const std::size_t N = options_.max_terms == 0 ? 200000 : options_.max_terms;
        std::mt19937_64 rng(options_.seed);
        std::uniform_int_distribution<std::size_t> pick(0, n - 1);
        for (std::size_t t = 0; t < N; ++t) {
            for (std::size_t k = 0; k < m; ++k) tuple[k] = sample[pick(rng)];
            sum += kernel(tuple);
        }
        r.num_terms = N;
        r.estimate = sum / static_cast<double>(N);
    }
    return r;
}

} // namespace datamunge::stats
