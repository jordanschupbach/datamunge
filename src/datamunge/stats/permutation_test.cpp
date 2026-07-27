#include <datamunge/stats/permutation_test.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::stats {

namespace {

double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

} // namespace

PermutationTestResult permutation_test_two_sample(const std::vector<double>& x, const std::vector<double>& y,
                                                  const TwoSampleStatistic& statistic,
                                                  PermutationTestOptions options) {
    const std::size_t nx = x.size(), ny = y.size();
    if (nx == 0 || ny == 0) throw std::invalid_argument("permutation_test_two_sample: both samples must be non-empty");
    if (options.num_permutations == 0)
        throw std::invalid_argument("permutation_test_two_sample: num_permutations must be >= 1");

    const TwoSampleStatistic stat =
        statistic ? statistic : TwoSampleStatistic([](const std::vector<double>& a, const std::vector<double>& b) {
            return mean(a) - mean(b);
        });

    const double observed = stat(x, y);

    std::vector<double> pool;
    pool.reserve(nx + ny);
    pool.insert(pool.end(), x.begin(), x.end());
    pool.insert(pool.end(), y.begin(), y.end());

    std::mt19937_64 rng(options.seed);
    PermutationTestResult r;
    r.observed = observed;
    r.num_permutations = options.num_permutations;
    r.null_distribution.resize(options.num_permutations);

    std::vector<double> gx(nx), gy(ny);
    std::size_t extreme = 0;
    for (std::size_t b = 0; b < options.num_permutations; ++b) {
        std::shuffle(pool.begin(), pool.end(), rng); // relabel: first nx -> group x, rest -> group y
        for (std::size_t i = 0; i < nx; ++i) gx[i] = pool[i];
        for (std::size_t i = 0; i < ny; ++i) gy[i] = pool[nx + i];
        const double t = stat(gx, gy);
        r.null_distribution[b] = t;
        switch (options.alternative) {
        case Alternative::Greater: extreme += (t >= observed); break;
        case Alternative::Less: extreme += (t <= observed); break;
        case Alternative::TwoSided:
        default: extreme += (std::fabs(t) >= std::fabs(observed)); break;
        }
    }

    // Monte Carlo p-value with the +1 correction (never returns exactly 0).
    r.p_value = static_cast<double>(1 + extreme) / static_cast<double>(options.num_permutations + 1);
    r.null_mean = mean(r.null_distribution);
    double ss = 0.0;
    for (const double v : r.null_distribution) ss += (v - r.null_mean) * (v - r.null_mean);
    r.null_sd = options.num_permutations > 1
                    ? std::sqrt(ss / static_cast<double>(options.num_permutations - 1))
                    : 0.0;
    return r;
}

} // namespace datamunge::stats
