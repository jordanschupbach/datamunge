#include <datamunge/stats/detail/ranking.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>

namespace datamunge::stats::detail {

std::vector<double> rank_with_ties(const std::vector<double>& x) {
    const std::size_t n = x.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&x](std::size_t i, std::size_t j) { return x[i] < x[j]; });

    std::vector<double> ranks(n);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j + 1 < n && x[order[j + 1]] == x[order[i]]) ++j;
        const double average_rank = 0.5 * (static_cast<double>(i + 1) + static_cast<double>(j + 1));
        for (std::size_t k = i; k <= j; ++k) ranks[order[k]] = average_rank;
        i = j + 1;
    }
    return ranks;
}

double tie_correction_sum(const std::vector<double>& x) {
    std::vector<double> sorted = x;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    std::size_t i = 0;
    while (i < sorted.size()) {
        std::size_t j = i;
        while (j + 1 < sorted.size() && sorted[j + 1] == sorted[i]) ++j;
        const double t = static_cast<double>(j - i + 1);
        sum += t * t * t - t;
        i = j + 1;
    }
    return sum;
}

} // namespace datamunge::stats::detail
