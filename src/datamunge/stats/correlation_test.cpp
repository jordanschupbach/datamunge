#include <datamunge/stats/correlation_test.hpp>

#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/ranking.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

namespace {

double pearson_r(const std::vector<double>& x, const std::vector<double>& y) {
    const auto n = static_cast<double>(x.size());
    double mean_x = 0.0, mean_y = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    double cov = 0.0, var_x = 0.0, var_y = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - mean_x;
        const double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    return cov / std::sqrt(var_x * var_y);
}

double t_p_value(double t, double df, Alternative alternative) {
    switch (alternative) {
        case Alternative::Less: return random::student_t_cdf(t, df);
        case Alternative::Greater: return 1.0 - random::student_t_cdf(t, df);
        case Alternative::TwoSided:
        default: return 2.0 * (1.0 - random::student_t_cdf(std::fabs(t), df));
    }
}

HypothesisTestResult correlation_result_from_r(double r, double n, Alternative alternative) {
    const double df = n - 2.0;
    const double t = r * std::sqrt(df / (1.0 - r * r));

    HypothesisTestResult result;
    result.statistic = t;
    result.parameter1 = df;
    result.p_value = t_p_value(t, df, alternative);
    result.estimate1 = r;
    result.alternative = alternative;
    return result;
}

} // namespace

HypothesisTestResult pearson_correlation_test(const std::vector<double>& x, const std::vector<double>& y,
                                               Alternative alternative, double conf_level) {
    if (x.size() != y.size() || x.size() < 3)
        throw std::invalid_argument("pearson_correlation_test: x and y must have equal length >= 3");
    const auto n = static_cast<double>(x.size());
    const double r = pearson_r(x, y);

    auto result = correlation_result_from_r(r, n, alternative);
    result.method = "Pearson's product-moment correlation";

    const double z = std::atanh(r);
    const double se_z = 1.0 / std::sqrt(n - 3.0);
    result.has_conf_int = true;
    if (alternative == Alternative::TwoSided) {
        const double crit = random::normal_quantile(1.0 - 0.5 * (1.0 - conf_level));
        result.conf_int_lower = std::tanh(z - crit * se_z);
        result.conf_int_upper = std::tanh(z + crit * se_z);
    } else if (alternative == Alternative::Less) {
        const double crit = random::normal_quantile(conf_level);
        result.conf_int_lower = -1.0;
        result.conf_int_upper = std::tanh(z + crit * se_z);
    } else {
        const double crit = random::normal_quantile(conf_level);
        result.conf_int_lower = std::tanh(z - crit * se_z);
        result.conf_int_upper = 1.0;
    }
    return result;
}

HypothesisTestResult spearman_correlation_test(const std::vector<double>& x, const std::vector<double>& y,
                                                Alternative alternative) {
    if (x.size() != y.size() || x.size() < 3)
        throw std::invalid_argument("spearman_correlation_test: x and y must have equal length >= 3");
    const auto rank_x = detail::rank_with_ties(x);
    const auto rank_y = detail::rank_with_ties(y);
    const double rho = pearson_r(rank_x, rank_y);

    auto result = correlation_result_from_r(rho, static_cast<double>(x.size()), alternative);
    result.method = "Spearman's rank correlation rho";
    return result;
}

} // namespace datamunge::stats
