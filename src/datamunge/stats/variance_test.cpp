#include <datamunge/stats/variance_test.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

namespace {

double sample_variance(const std::vector<double>& x) {
    const auto n = static_cast<double>(x.size());
    double mean = 0.0;
    for (const double v : x) mean += v;
    mean /= n;
    double ss = 0.0;
    for (const double v : x) ss += (v - mean) * (v - mean);
    return ss / (n - 1.0);
}

} // namespace

HypothesisTestResult f_test_variance(const std::vector<double>& x, const std::vector<double>& y,
                                      Alternative alternative, double conf_level) {
    if (x.size() < 2 || y.size() < 2) throw std::invalid_argument("f_test_variance: each sample needs at least 2 observations");

    const double df1 = static_cast<double>(x.size() - 1);
    const double df2 = static_cast<double>(y.size() - 1);
    const double estimate = sample_variance(x) / sample_variance(y);

    HypothesisTestResult result;
    result.statistic = estimate;
    result.parameter1 = df1;
    result.parameter2 = df2;
    result.estimate1 = estimate;
    result.alternative = alternative;
    result.method = "F test to compare two variances";

    const double cdf = random::f_cdf(estimate, df1, df2);
    const double alpha = 1.0 - conf_level;
    const double inf = std::numeric_limits<double>::infinity();
    result.has_conf_int = true;
    switch (alternative) {
        case Alternative::Less:
            result.p_value = cdf;
            result.conf_int_lower = 0.0;
            result.conf_int_upper = estimate / random::f_quantile(alpha, df1, df2);
            break;
        case Alternative::Greater:
            result.p_value = 1.0 - cdf;
            result.conf_int_lower = estimate / random::f_quantile(1.0 - alpha, df1, df2);
            result.conf_int_upper = inf;
            break;
        case Alternative::TwoSided:
        default:
            result.p_value = 2.0 * std::min(cdf, 1.0 - cdf);
            result.conf_int_lower = estimate / random::f_quantile(1.0 - 0.5 * alpha, df1, df2);
            result.conf_int_upper = estimate / random::f_quantile(0.5 * alpha, df1, df2);
            break;
    }
    if (result.p_value > 1.0) result.p_value = 1.0;
    return result;
}

} // namespace datamunge::stats
