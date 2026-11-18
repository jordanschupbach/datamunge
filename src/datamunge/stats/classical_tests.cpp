/// @file classical_tests.cpp
/// @brief Additional classical rank, blocked and categorical tests.
#include <datamunge/stats/classical_tests.hpp>

#include <datamunge/stats/detail/ranking.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/nonparametric.hpp>
#include <datamunge/stats/wilcoxon_test.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Shared finite-reference rank-score calculations.
namespace {
/// @brief Two-sample conditional score statistic with either inward rank scheme.
/// @param x First sample.
/// @param y Independent second sample.
/// @param alternative Direction of first-sample scale.
/// @param siegel Use asymmetric Siegel–Tukey scores instead of symmetric scores.
/// @return Score sum and normal-reference test.
/// @throws std::invalid_argument For invalid inputs or zero score variation.
HypothesisTestResult scale_score_test(const std::vector<double>& x, const std::vector<double>& y,
                                      Alternative alternative, bool siegel) {
    detail::check_sample(x);
    detail::check_sample(y);
    detail::check_alternative(alternative);
    auto pooled = x;
    pooled.insert(pooled.end(), y.begin(), y.end());
    const auto n = pooled.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, {}, [&](std::size_t i) { return pooled[i]; });
    std::vector<double> scores(n);
    if (siegel) {
        std::size_t lo = 0, hi = n;
        for (std::size_t rank = 0; rank < n; ++rank)
            scores[(rank % 4 == 0 || rank % 4 == 3) ? lo++ : --hi] = rank + 1;
    } else {
        for (std::size_t i = 0; i < n; ++i) scores[i] = std::min(i + 1, n - i);
    }
    long double observed = 0;
    for (std::size_t i = 0; i < n;) {
        auto j = i + 1;
        std::size_t first_count = order[i] < x.size();
        while (j < n && pooled[order[j]] == pooled[order[i]]) first_count += order[j++] < x.size();
        const auto average = std::accumulate(scores.begin() + i, scores.begin() + j, 0.L) / (j - i);
        std::fill(scores.begin() + i, scores.begin() + j, static_cast<double>(average));
        observed += first_count * average;
        i = j;
    }
    const auto [mean, variance] = detail::test_moments(scores);
    if (!(variance > 0)) throw std::invalid_argument("scale test: pooled scores have no variation");
    const auto sd = std::sqrt(variance * x.size() * y.size() / n);
    HypothesisTestResult result;
    result.statistic = static_cast<double>(observed);
    result.alternative = alternative;
    // Large first-sample spread produces a small score sum.
    result.p_value = detail::normal_test_p(static_cast<double>((x.size() * mean - observed) / sd), alternative);
    result.method = siegel ? "Siegel-Tukey scale test (normal approximation)"
                           : "Ansari-Bradley scale test (normal approximation)";
    return result;
}
} // namespace

HypothesisTestResult mann_whitney_test(const std::vector<double>& x, const std::vector<double>& y,
                                       Alternative alternative) {
    auto result = wilcoxon_rank_sum_test(x, y, alternative);
    result.estimate1 = result.statistic / (static_cast<double>(x.size()) * y.size());
    return result;
}

HypothesisTestResult siegel_tukey_test(const std::vector<double>& x, const std::vector<double>& y,
                                       Alternative alternative) {
    return scale_score_test(x, y, alternative, true);
}

HypothesisTestResult ansari_bradley_test(const std::vector<double>& x, const std::vector<double>& y,
                                         Alternative alternative) {
    return scale_score_test(x, y, alternative, false);
}

HypothesisTestResult kendall_w_test(const std::vector<std::vector<double>>& blocks) {
    auto result = friedman_test(blocks);
    result.estimate1 = std::clamp(result.statistic / (result.parameter1 * result.parameter2), 0., 1.);
    result.method = "Kendall concordance (Friedman chi-square reference)";
    return result;
}

HypothesisTestResult page_trend_test(const std::vector<std::vector<double>>& blocks) {
    if (blocks.size() < 2 || blocks.front().size() < 2)
        throw std::invalid_argument("Page test: need at least two rows and columns");
    const auto k = blocks.front().size();
    const long double center = (static_cast<long double>(k) + 1) / 2;
    long double statistic = 0, centered = 0, spread = 0, weights_ss = 0;
    for (std::size_t j = 0; j < k; ++j) {
        const auto weight = j + 1 - center;
        weights_ss += weight * weight;
    }
    for (const auto& block : blocks) {
        if (block.size() != k) throw std::invalid_argument("Page test: ragged blocks");
        detail::check_sample(block);
        const auto ranks = detail::rank_with_ties(block);
        for (std::size_t j = 0; j < k; ++j) {
            const auto deviation = ranks[j] - center;
            statistic += (j + 1) * static_cast<long double>(ranks[j]);
            centered += (j + 1 - center) * deviation;
            spread += deviation * deviation;
        }
    }
    if (!(spread > 0)) throw std::invalid_argument("Page test: no within-block rank variation");
    HypothesisTestResult result;
    result.statistic = static_cast<double>(statistic);
    result.parameter1 = static_cast<double>(blocks.size());
    result.parameter2 = static_cast<double>(k);
    result.alternative = Alternative::Greater;
    result.p_value = detail::normal_test_p(static_cast<double>(centered / std::sqrt(spread * weights_ss / (k - 1))),
                                           result.alternative);
    result.method = "Page ordered rank test (normal approximation)";
    return result;
}

HypothesisTestResult cochran_q_test(const std::vector<std::vector<int>>& blocks) {
    std::vector<std::vector<double>> values;
    for (const auto& row : blocks) {
        if (!std::ranges::all_of(row, [](int x) { return x == 0 || x == 1; }))
            throw std::invalid_argument("Cochran Q: outcomes must be binary");
        values.emplace_back(row.begin(), row.end());
    }
    auto result = friedman_test(values);
    result.method = "Cochran Q test";
    return result;
}

HypothesisTestResult mcnemar_test(std::size_t first_only, std::size_t second_only, Alternative alternative) {
    detail::check_alternative(alternative);
    if (first_only > std::numeric_limits<std::size_t>::max() - second_only)
        throw std::invalid_argument("McNemar: discordant count overflows");
    const auto n = first_only + second_only;
    HypothesisTestResult result;
    result.statistic = static_cast<double>(first_only);
    result.parameter1 = static_cast<double>(n);
    result.estimate1 = n == 0 ? 0 : static_cast<double>(first_only) / n;
    result.alternative = alternative;
    result.p_value = n == 0 ? 1 : detail::binomial_test_p(first_only, n, 0.5, alternative);
    result.method = "Exact McNemar test";
    return result;
}

HypothesisTestResult mcnemar_test(const std::vector<int>& first, const std::vector<int>& second,
                                 Alternative alternative) {
    if (first.empty() || first.size() != second.size())
        throw std::invalid_argument("McNemar: need nonempty paired observations");
    std::size_t first_only = 0, second_only = 0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        const int a = first[i], b = second[i];
        if (a < 0 || a > 1 || b < 0 || b > 1) throw std::invalid_argument("McNemar: outcomes must be binary");
        first_only += a > b;
        second_only += b > a;
    }
    return mcnemar_test(first_only, second_only, alternative);
}

HypothesisTestResult mantel_haenszel_test(const std::vector<std::array<std::size_t, 4>>& strata, bool correct) {
    long double residual = 0, variance = 0, numerator = 0, denominator = 0;
    for (const auto& table : strata) {
        const long double a = table[0], b = table[1], c = table[2], d = table[3];
        const auto n = a + b + c + d;
        if (n <= 1) continue;
        residual += (a * d - b * c) / n;
        variance += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1));
        numerator += a * d / n;
        denominator += b * c / n;
    }
    if (!(variance > 0)) throw std::invalid_argument("Mantel-Haenszel: no informative strata");
    const auto distance = std::max(0.L, std::abs(residual) - (correct ? 0.5L : 0.L));
    HypothesisTestResult result;
    result.statistic = static_cast<double>(distance * distance / variance);
    result.parameter1 = 1;
    result.estimate1 = denominator == 0 ? std::numeric_limits<double>::infinity()
                                        : static_cast<double>(numerator / denominator);
    result.p_value = detail::chi_squared_test_survival(result.statistic, 1);
    result.method = correct ? "Cochran-Mantel-Haenszel test with continuity correction"
                            : "Cochran-Mantel-Haenszel test";
    return result;
}

} // namespace datamunge::stats
