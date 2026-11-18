/// @file nonparametric.cpp
/// @brief Classical rank and sign inference and empirical CDF plots.
#include <datamunge/stats/nonparametric.hpp>

#include <datamunge/stats/anova_test.hpp>
#include <datamunge/stats/chi_squared_test.hpp>
#include <datamunge/stats/correlation_test.hpp>
#include <datamunge/stats/detail/ranking.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/fisher_exact_test.hpp>
#include <datamunge/stats/wilcoxon_test.hpp>
#include <datamunge/vector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Private shared calculations for classical nonparametric inference.
namespace {
/// @brief Validates and concatenates independent groups in source order.
/// @param groups At least two finite, nonempty groups.
/// @return Owned concatenation.
/// @throws std::invalid_argument For empty or nonfinite groups.
std::vector<double> pooled_sample(const std::vector<std::vector<double>>& groups) {
    if (groups.size() < 2) throw std::invalid_argument("nonparametric test: need at least two groups");
    std::vector<double> pooled;
    for (const auto& group : groups) {
        detail::check_sample(group);
        pooled.insert(pooled.end(), group.begin(), group.end());
    }
    return pooled;
}

/// @brief Validates paired inputs and computes differences in input order.
/// @param x Finite first measurements.
/// @param y Same-length finite second measurements.
/// @return Owned finite differences x minus y.
/// @throws std::invalid_argument For invalid inputs or nonfinite differences.
std::vector<double> paired_differences(const std::vector<double>& x, const std::vector<double>& y) {
    detail::check_sample(x);
    detail::check_sample(y);
    if (x.size() != y.size()) throw std::invalid_argument("paired test: lengths differ");
    std::vector<double> differences(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) differences[i] = x[i] - y[i];
    detail::check_sample(differences);
    return differences;
}

/// @brief Constructs a right-tailed chi-square result for a rank statistic.
/// @param statistic Observed nonnegative statistic.
/// @param df Positive degrees of freedom.
/// @param method Human-readable method name.
/// @return Owned test result.
HypothesisTestResult rank_chi_square(double statistic, double df, const char* method) {
    HypothesisTestResult result;
    result.statistic = statistic;
    result.parameter1 = df;
    result.p_value = detail::chi_squared_test_survival(statistic, df);
    result.method = method;
    return result;
}

/// @brief Tie group sums for the Kendall score variance.
struct KendallTieSums {
    long double pairs = 0; ///< Sum of t(t-1).
    long double triples = 0; ///< Sum of t(t-1)(t-2).
    long double weighted = 0; ///< Sum of t(t-1)(2t+5).
};

/// @brief Calculates sums over equal-coordinate runs.
/// @param values Finite coordinates copied for sorting.
/// @return Tie moments used to adjust the null score variance.
KendallTieSums kendall_tie_sums(std::vector<double> values) {
    std::ranges::sort(values);
    KendallTieSums result;
    for (auto first = values.begin(); first != values.end();) {
        const auto last = std::find_if(first + 1, values.end(), [v = *first](double x) { return x != v; });
        const long double n = last - first;
        const auto pairs = n * (n - 1);
        result.pairs += pairs;
        result.triples += pairs * (n - 2);
        result.weighted += pairs * (2 * n + 5);
        first = last;
    }
    return result;
}

/// @brief Exact null tail of the inversion count among independent untied pairs.
/// @param n Number of pairs, at most 50.
/// @param observed Observed signed Kendall score.
/// @param alternative Valid direction for association.
/// @return Inclusive one-sided or absolute-score two-sided probability.
double exact_kendall_p(std::size_t n, long long observed, Alternative alternative) {
    std::vector<long double> probabilities{1};
    for (std::size_t m = 2; m <= n; ++m) {
        std::vector<long double> next(probabilities.size() + m - 1);
        for (std::size_t k = 0; k < probabilities.size(); ++k) {
            const auto mass = probabilities[k] / m;
            for (std::size_t d = 0; d < m; ++d) next[k + d] += mass;
        }
        probabilities.swap(next);
    }
    long double p = 0;
    const auto pairs = static_cast<long long>(probabilities.size() - 1);
    for (std::size_t k = 0; k < probabilities.size(); ++k) {
        const auto score = pairs - 2 * static_cast<long long>(k);
        const bool selected = alternative == Alternative::Less ? score <= observed
                            : alternative == Alternative::Greater ? score >= observed
                            : std::abs(score) >= std::abs(observed);
        if (selected) p += probabilities[k];
    }
    return std::min(1., static_cast<double>(p));
}

/// @brief Appends the empirical step function onto a common finite plotting domain.
/// @param plot Destination plot.
/// @param sample Finite, nonempty data copied for sorting.
/// @param lo Common left endpoint.
/// @param hi Common right endpoint.
/// @param label Legend label.
/// @param color Series color.
void add_ecdf(graphics::RPlot& plot, std::vector<double> sample, double lo, double hi,
              const std::string& label, graphics::RGBA color) {
    std::ranges::sort(sample);
    std::vector<double> x{lo}, y{0};
    for (auto first = sample.begin(); first != sample.end();) {
        const auto next = std::find_if(first + 1, sample.end(), [v = *first](double z) { return z != v; });
        x.push_back(*first);
        y.push_back(y.back());
        x.push_back(*first);
        y.push_back(static_cast<double>(next - sample.begin()) / sample.size());
        first = next;
    }
    x.push_back(hi);
    y.push_back(1);
    plot.lines(std::move(x), std::move(y), label, color);
}
} // namespace

HypothesisTestResult kendall_correlation_test(const std::vector<double>& x, const std::vector<double>& y,
                                             Alternative alternative) {
    detail::check_alternative(alternative);
    const auto tau = kendall_tau(x, y);
    const auto score = tau.concordant - tau.discordant;
    const long double n = x.size();
    const long double pairs = n * (n - 1) / 2;
    if (tau.ties_x + tau.ties_xy == pairs || tau.ties_y + tau.ties_xy == pairs)
        throw std::invalid_argument("Kendall test: both margins must vary");
    const bool exact = x.size() <= 50 && tau.ties_x == 0 && tau.ties_y == 0 && tau.ties_xy == 0;
    HypothesisTestResult result;
    result.statistic = static_cast<double>(score);
    result.parameter1 = static_cast<double>(n);
    result.estimate1 = tau.tau_b;
    result.alternative = alternative;
    if (exact) {
        result.p_value = exact_kendall_p(x.size(), score, alternative);
        result.method = "Kendall rank correlation (exact permutation)";
    } else {
        const auto a = kendall_tie_sums(x);
        const auto b = kendall_tie_sums(y);
        const auto variance = (2 * pairs * (2 * n + 5) - a.weighted - b.weighted) / 18
            + a.pairs * b.pairs / (4 * pairs)
            + a.triples * b.triples / (18 * pairs * (n - 2));
        result.p_value = detail::normal_test_p(static_cast<double>(score / std::sqrt(variance)), alternative);
        result.method = "Kendall rank correlation (tie-corrected normal approximation)";
    }
    return result;
}

HypothesisTestResult quadrant_test(const std::vector<double>& x, const std::vector<double>& y,
                                   Alternative alternative) {
    detail::check_sample(x, 2);
    detail::check_sample(y, 2);
    if (x.size() != y.size()) throw std::invalid_argument("quadrant test: lengths differ");
    const double mx = datamunge::median(x), my = datamunge::median(y);
    std::array<std::size_t, 4> counts{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] != mx && y[i] != my) ++counts[2 * (x[i] < mx) + (y[i] < my)];
    }
    const auto [a, b, c, d] = counts;
    if (a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0)
        throw std::invalid_argument("quadrant test: median signs must vary in both retained margins");
    const auto n = a + b + c + d;
    auto result = fisher_exact_test_2x2(a, b, c, d, alternative);
    result.parameter1 = static_cast<double>(n);
    result.estimate1 = (static_cast<double>(a + d) - static_cast<double>(b + c)) / n;
    result.method = "Median quadrant association (conditional Fisher exact)";
    return result;
}

HypothesisTestResult sign_test(const std::vector<double>& values, double median, Alternative alternative) {
    detail::check_sample(values);
    detail::check_alternative(alternative);
    if (!std::isfinite(median)) throw std::invalid_argument("sign test: median must be finite");
    std::size_t positive = 0, negative = 0;
    for (double value : values) {
        positive += value > median;
        negative += value < median;
    }
    const auto n = positive + negative;
    if (n == 0) throw std::invalid_argument("sign test: every value equals the null median");
    HypothesisTestResult result;
    result.statistic = static_cast<double>(positive);
    result.parameter1 = static_cast<double>(n);
    result.estimate1 = static_cast<double>(positive) / n;
    result.p_value = detail::binomial_test_p(positive, n, 0.5, alternative);
    result.alternative = alternative;
    result.method = "Exact sign test";
    return result;
}

HypothesisTestResult paired_sign_test(const std::vector<double>& x, const std::vector<double>& y,
                                     Alternative alternative) {
    auto result = sign_test(paired_differences(x, y), 0, alternative);
    result.method = "Paired exact sign test";
    return result;
}

HypothesisTestResult wilcoxon_paired_test(const std::vector<double>& x, const std::vector<double>& y,
                                         Alternative alternative) {
    const auto differences = paired_differences(x, y);
    if (std::ranges::count_if(differences, [](double x) { return x != 0; }) <= 100)
        return wilcoxon_signed_rank_exact_test(differences, 0, alternative);
    return wilcoxon_signed_rank_test(differences, 0, alternative);
}

HypothesisTestResult wilcoxon_signed_rank_exact_test(const std::vector<double>& values, double mu,
                                                     Alternative alternative) {
    detail::check_sample(values);
    detail::check_alternative(alternative);
    if (!std::isfinite(mu)) throw std::invalid_argument("exact signed rank: invalid null location");
    std::vector<double> magnitude;
    std::vector<bool> positive;
    for (double value : values) {
        if (value == mu) continue;
        const double difference = value - mu;
        if (!std::isfinite(difference) || magnitude.size() == 100)
            throw std::invalid_argument("exact signed rank: need at most 100 finite nonzero differences");
        magnitude.push_back(std::abs(difference));
        positive.push_back(difference > 0);
    }
    detail::check_sample(magnitude);
    const auto ranks = detail::rank_with_ties(magnitude);
    std::vector<long double> probabilities{1};
    int observed = 0;
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        // Doubling average ranks gives integral indices even when there are ties.
        const auto weight = static_cast<int>(2 * ranks[i]);
        if (positive[i]) observed += weight;
        const auto count = probabilities.size();
        probabilities.resize(count + weight, 0);
        for (std::size_t j = count; j-- > 0;) {
            probabilities[j] *= 0.5L;
            probabilities[j + weight] += probabilities[j];
        }
    }
    const auto total = static_cast<int>(probabilities.size() - 1);
    long double p = 0;
    for (int i = 0; i <= total; ++i) {
        const bool selected = alternative == Alternative::Less ? i <= observed
                            : alternative == Alternative::Greater ? i >= observed
                            : std::abs(2 * i - total) >= std::abs(2 * observed - total);
        if (selected) p += probabilities[i];
    }
    HypothesisTestResult result;
    result.statistic = observed / 2.;
    result.parameter1 = static_cast<double>(ranks.size());
    result.estimate1 = mu;
    result.p_value = std::min(1., static_cast<double>(p));
    result.alternative = alternative;
    result.method = "Wilcoxon signed rank test (exact conditional sign flips)";
    return result;
}

HypothesisTestResult median_test(const std::vector<std::vector<double>>& groups) {
    const auto median = datamunge::median(Vector<>(pooled_sample(groups)));
    std::vector<double> table;
    for (const auto& group : groups) {
        double above = 0, below = 0;
        for (double x : group) {
            above += x > median;
            below += x < median;
        }
        table.push_back(above);
        table.push_back(below);
    }
    auto result = chi_squared_test_independence(table, groups.size(), 2, false);
    result.method = "Mood median test (Pearson chi-square)";
    result.estimate1 = median;
    return result;
}

HypothesisTestResult van_der_waerden_test(const std::vector<std::vector<double>>& groups) {
    auto scores = detail::rank_with_ties(pooled_sample(groups));
    const double total = static_cast<double>(scores.size());
    for (auto& score : scores) score = random::normal_quantile(score / (total + 1));
    const auto [mean, variance] = detail::test_moments(scores);
    if (!(variance > 0)) throw std::invalid_argument("van der Waerden test: no score variation");
    long double between = 0;
    auto first = scores.begin();
    for (const auto& group : groups) {
        const auto end = first + group.size();
        const auto delta = std::accumulate(first, end, 0.L) / group.size() - mean;
        between += group.size() * delta * delta;
        first = end;
    }
    return rank_chi_square(static_cast<double>(between / variance), groups.size() - 1,
                           "van der Waerden normal scores test");
}

HypothesisTestResult kruskal_wallis_test(const std::vector<std::vector<double>>& groups) {
    const auto pooled = pooled_sample(groups);
    std::vector<std::size_t> sizes;
    for (const auto& group : groups) sizes.push_back(group.size());
    return kruskal_wallis_test(pooled, sizes);
}

HypothesisTestResult friedman_test(const std::vector<std::vector<double>>& blocks) {
    if (blocks.size() < 2 || blocks.front().size() < 2)
        throw std::invalid_argument("Friedman test: need at least two blocks and treatments");
    const auto k = blocks.front().size();
    std::vector<long double> sums(k);
    long double total_ties = 0;
    const long double n = blocks.size(), treatments = k;
    for (const auto& block : blocks) {
        if (block.size() != k) throw std::invalid_argument("Friedman test: ragged blocks");
        detail::check_sample(block);
        const auto ranks = detail::rank_with_ties(block);
        for (std::size_t j = 0; j < k; ++j) sums[j] += ranks[j] - (treatments + 1) / 2;
        total_ties += detail::tie_correction_sum(block);
    }
    const auto correction = 1 - total_ties / (n * treatments * (treatments * treatments - 1));
    if (!(correction > 0)) throw std::invalid_argument("Friedman test: no within-block variation");
    const auto statistic = 12 * std::inner_product(sums.begin(), sums.end(), sums.begin(), 0.L)
                         / (n * treatments * (treatments + 1) * correction);
    auto result = rank_chi_square(static_cast<double>(statistic), k - 1, "Friedman rank sum test");
    result.parameter2 = static_cast<double>(n);
    return result;
}

graphics::RPlot empirical_cdf_plot(const std::vector<double>& x, const std::vector<double>& y,
                                  const std::string& x_label, const std::string& y_label) {
    detail::check_sample(x);
    detail::check_sample(y);
    const auto [xmin, xmax] = std::ranges::minmax_element(x);
    const auto [ymin, ymax] = std::ranges::minmax_element(y);
    const double lo = std::min(*xmin, *ymin), hi = std::max(*xmax, *ymax);
    auto plot = graphics::RPlot::create().title("Empirical distributions")
        .x_label("Observation").y_label("Cumulative probability").y_limits(0, 1.17);
    add_ecdf(plot, x, lo, hi, x_label, {36, 105, 172});
    add_ecdf(plot, y, lo, hi, y_label, {182, 83, 29});
    return plot;
}

} // namespace datamunge::stats
