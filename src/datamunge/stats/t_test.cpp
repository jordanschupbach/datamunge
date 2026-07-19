#include <datamunge/stats/t_test.hpp>

#include <datamunge/random/distributions.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

namespace {

struct MeanVar {
    double mean;
    double variance; // sample (n-1 denominator)
    std::size_t n;
};

MeanVar mean_variance(const std::vector<double>& x) {
    if (x.size() < 2) throw std::invalid_argument("t_test: need at least 2 observations");
    double mean = 0.0;
    for (const double v : x) mean += v;
    mean /= static_cast<double>(x.size());
    double ss = 0.0;
    for (const double v : x) ss += (v - mean) * (v - mean);
    return {mean, ss / static_cast<double>(x.size() - 1), x.size()};
}

double t_p_value(double t, double df, Alternative alternative) {
    switch (alternative) {
        case Alternative::Less: return random::student_t_cdf(t, df);
        case Alternative::Greater: return 1.0 - random::student_t_cdf(t, df);
        case Alternative::TwoSided:
        default: return 2.0 * (1.0 - random::student_t_cdf(std::fabs(t), df));
    }
}

void fill_ci(HypothesisTestResult& result, double estimate, double se, double df, double conf_level,
             Alternative alternative) {
    result.has_conf_int = true;
    const double inf = std::numeric_limits<double>::infinity();
    if (alternative == Alternative::TwoSided) {
        const double crit = random::student_t_quantile(1.0 - 0.5 * (1.0 - conf_level), df);
        result.conf_int_lower = estimate - crit * se;
        result.conf_int_upper = estimate + crit * se;
    } else if (alternative == Alternative::Less) {
        const double crit = random::student_t_quantile(conf_level, df);
        result.conf_int_lower = -inf;
        result.conf_int_upper = estimate + crit * se;
    } else {
        const double crit = random::student_t_quantile(conf_level, df);
        result.conf_int_lower = estimate - crit * se;
        result.conf_int_upper = inf;
    }
}

} // namespace

HypothesisTestResult t_test_one_sample(const std::vector<double>& x, double mu, Alternative alternative,
                                        double conf_level) {
    const auto mv = mean_variance(x);
    const double se = std::sqrt(mv.variance / static_cast<double>(mv.n));
    const double df = static_cast<double>(mv.n - 1);
    const double t = (mv.mean - mu) / se;

    HypothesisTestResult result;
    result.statistic = t;
    result.parameter1 = df;
    result.p_value = t_p_value(t, df, alternative);
    result.estimate1 = mv.mean;
    result.alternative = alternative;
    result.method = "One Sample t-test";
    fill_ci(result, mv.mean, se, df, conf_level, alternative);
    return result;
}

HypothesisTestResult t_test_two_sample(const std::vector<double>& x, const std::vector<double>& y,
                                        bool equal_variance, Alternative alternative, double conf_level) {
    const auto mx = mean_variance(x);
    const auto my = mean_variance(y);
    const double n1 = static_cast<double>(mx.n);
    const double n2 = static_cast<double>(my.n);

    double se, df;
    if (equal_variance) {
        const double pooled_var = ((n1 - 1.0) * mx.variance + (n2 - 1.0) * my.variance) / (n1 + n2 - 2.0);
        se = std::sqrt(pooled_var * (1.0 / n1 + 1.0 / n2));
        df = n1 + n2 - 2.0;
    } else {
        const double v1 = mx.variance / n1;
        const double v2 = my.variance / n2;
        se = std::sqrt(v1 + v2);
        df = (v1 + v2) * (v1 + v2) / (v1 * v1 / (n1 - 1.0) + v2 * v2 / (n2 - 1.0));
    }
    const double estimate = mx.mean - my.mean;
    const double t = estimate / se;

    HypothesisTestResult result;
    result.statistic = t;
    result.parameter1 = df;
    result.p_value = t_p_value(t, df, alternative);
    result.estimate1 = mx.mean;
    result.estimate2 = my.mean;
    result.alternative = alternative;
    result.method = equal_variance ? "Two Sample t-test" : "Welch Two Sample t-test";
    fill_ci(result, estimate, se, df, conf_level, alternative);
    return result;
}

HypothesisTestResult t_test_paired(const std::vector<double>& x, const std::vector<double>& y,
                                    Alternative alternative, double conf_level) {
    if (x.size() != y.size()) throw std::invalid_argument("t_test_paired: x and y must have the same length");
    std::vector<double> diff(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) diff[i] = x[i] - y[i];
    auto result = t_test_one_sample(diff, 0.0, alternative, conf_level);
    result.method = "Paired t-test";
    return result;
}

} // namespace datamunge::stats
