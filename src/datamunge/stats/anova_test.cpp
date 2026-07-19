#include <datamunge/stats/anova_test.hpp>

#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/ranking.hpp>

#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

namespace {

void validate_groups(const std::vector<double>& values, const std::vector<std::size_t>& group_sizes,
                      const char* fn) {
    if (group_sizes.size() < 2) throw std::invalid_argument(std::string(fn) + ": need at least 2 groups");
    const std::size_t total = std::accumulate(group_sizes.begin(), group_sizes.end(), std::size_t{0});
    if (total != values.size())
        throw std::invalid_argument(std::string(fn) + ": group_sizes must sum to values.size()");
}

} // namespace

HypothesisTestResult one_way_anova(const std::vector<double>& values, const std::vector<std::size_t>& group_sizes) {
    validate_groups(values, group_sizes, "one_way_anova");

    const auto n_total = static_cast<double>(values.size());
    double grand_mean = 0.0;
    for (const double v : values) grand_mean += v;
    grand_mean /= n_total;

    double ss_between = 0.0, ss_within = 0.0;
    std::size_t offset = 0;
    for (const std::size_t g : group_sizes) {
        double group_mean = 0.0;
        for (std::size_t i = 0; i < g; ++i) group_mean += values[offset + i];
        group_mean /= static_cast<double>(g);
        for (std::size_t i = 0; i < g; ++i) {
            const double d = values[offset + i] - group_mean;
            ss_within += d * d;
        }
        const double dg = group_mean - grand_mean;
        ss_between += static_cast<double>(g) * dg * dg;
        offset += g;
    }

    const double df1 = static_cast<double>(group_sizes.size() - 1);
    const double df2 = n_total - static_cast<double>(group_sizes.size());
    const double f_stat = (ss_between / df1) / (ss_within / df2);

    HypothesisTestResult result;
    result.statistic = f_stat;
    result.parameter1 = df1;
    result.parameter2 = df2;
    result.p_value = 1.0 - random::f_cdf(f_stat, df1, df2);
    result.method = "One-way analysis of means";
    return result;
}

HypothesisTestResult kruskal_wallis_test(const std::vector<double>& values, const std::vector<std::size_t>& group_sizes) {
    validate_groups(values, group_sizes, "kruskal_wallis_test");

    const auto ranks = detail::rank_with_ties(values);
    const auto n_total = static_cast<double>(values.size());

    double h = 0.0;
    std::size_t offset = 0;
    for (const std::size_t g : group_sizes) {
        double rank_sum = 0.0;
        for (std::size_t i = 0; i < g; ++i) rank_sum += ranks[offset + i];
        h += rank_sum * rank_sum / static_cast<double>(g);
        offset += g;
    }
    h = 12.0 / (n_total * (n_total + 1.0)) * h - 3.0 * (n_total + 1.0);

    const double tie_sum = detail::tie_correction_sum(values);
    h /= (1.0 - tie_sum / (n_total * n_total * n_total - n_total));

    const double df = static_cast<double>(group_sizes.size() - 1);

    HypothesisTestResult result;
    result.statistic = h;
    result.parameter1 = df;
    result.p_value = 1.0 - random::chi_squared_cdf(h, df);
    result.method = "Kruskal-Wallis rank sum test";
    return result;
}

} // namespace datamunge::stats
