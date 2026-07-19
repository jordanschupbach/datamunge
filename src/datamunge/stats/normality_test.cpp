#include <datamunge/stats/normality_test.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

HypothesisTestResult shapiro_francia_test(const std::vector<double>& x) {
    const auto n = static_cast<double>(x.size());
    if (x.size() < 5) throw std::invalid_argument("shapiro_francia_test: need at least 5 observations");

    std::vector<double> sorted = x;
    std::sort(sorted.begin(), sorted.end());

    std::vector<double> scores(sorted.size());
    for (std::size_t i = 0; i < sorted.size(); ++i)
        scores[i] = random::normal_quantile((static_cast<double>(i + 1) - 0.375) / (n + 0.25));

    double mean_x = 0.0, mean_m = 0.0;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        mean_x += sorted[i];
        mean_m += scores[i];
    }
    mean_x /= n;
    mean_m /= n;

    double cov = 0.0, var_x = 0.0, var_m = 0.0;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        const double dx = sorted[i] - mean_x;
        const double dm = scores[i] - mean_m;
        cov += dx * dm;
        var_x += dx * dx;
        var_m += dm * dm;
    }
    const double r = cov / std::sqrt(var_x * var_m);
    const double w = r * r;

    const double u = std::log(n);
    const double v = std::log(u);
    const double mu = -1.2725 + 1.0521 * (v - u);
    const double sigma = 1.0308 - 0.26758 * (v + 2.0 / u);
    const double z = (std::log(1.0 - w) - mu) / sigma;

    HypothesisTestResult result;
    result.statistic = w;
    result.estimate1 = w;
    result.p_value = 1.0 - random::normal_cdf(z);
    result.method = "Shapiro-Francia normality test";
    return result;
}

} // namespace datamunge::stats
