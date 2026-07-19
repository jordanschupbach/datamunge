#include <datamunge/stats/wilcoxon_test.hpp>

#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/ranking.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

namespace {

double sign(double v) { return (v > 0.0) - (v < 0.0); }

double normal_p_value(double z, Alternative alternative) {
    switch (alternative) {
        case Alternative::Less: return random::normal_cdf(z);
        case Alternative::Greater: return 1.0 - random::normal_cdf(z);
        case Alternative::TwoSided:
        default: return std::min(1.0, 2.0 * std::min(random::normal_cdf(z), 1.0 - random::normal_cdf(z)));
    }
}

} // namespace

HypothesisTestResult wilcoxon_signed_rank_test(const std::vector<double>& x, double mu, Alternative alternative) {
    std::vector<double> diff;
    diff.reserve(x.size());
    for (const double v : x) {
        if (v != mu) diff.push_back(v - mu);
    }
    const auto n = static_cast<double>(diff.size());
    if (n < 1.0) throw std::invalid_argument("wilcoxon_signed_rank_test: no nonzero differences");

    std::vector<double> abs_diff(diff.size());
    for (std::size_t i = 0; i < diff.size(); ++i) abs_diff[i] = std::fabs(diff[i]);
    const auto ranks = detail::rank_with_ties(abs_diff);

    double w_plus = 0.0;
    for (std::size_t i = 0; i < diff.size(); ++i) {
        if (diff[i] > 0.0) w_plus += ranks[i];
    }

    const double mean_w = n * (n + 1.0) / 4.0;
    const double tie_sum = detail::tie_correction_sum(abs_diff);
    const double var_w = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - tie_sum / 48.0;
    const double sigma = std::sqrt(var_w);

    double correction;
    switch (alternative) {
        case Alternative::Greater: correction = 0.5; break;
        case Alternative::Less: correction = -0.5; break;
        case Alternative::TwoSided:
        default: correction = 0.5 * sign(w_plus - mean_w);
    }
    const double z = (w_plus - mean_w - correction) / sigma;

    HypothesisTestResult result;
    result.statistic = w_plus;
    result.p_value = normal_p_value(z, alternative);
    result.estimate1 = mu; // pseudomedian is not estimated here (matches conf.int = FALSE)
    result.alternative = alternative;
    result.method = "Wilcoxon signed rank test with continuity correction";
    return result;
}

HypothesisTestResult wilcoxon_rank_sum_test(const std::vector<double>& x, const std::vector<double>& y,
                                             Alternative alternative) {
    const auto n1 = static_cast<double>(x.size());
    const auto n2 = static_cast<double>(y.size());
    if (n1 < 1.0 || n2 < 1.0) throw std::invalid_argument("wilcoxon_rank_sum_test: both samples must be non-empty");

    std::vector<double> combined;
    combined.reserve(x.size() + y.size());
    combined.insert(combined.end(), x.begin(), x.end());
    combined.insert(combined.end(), y.begin(), y.end());
    const auto ranks = detail::rank_with_ties(combined);

    double rank_sum_x = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) rank_sum_x += ranks[i];
    const double w = rank_sum_x - n1 * (n1 + 1.0) / 2.0;

    const double n = n1 + n2;
    const double mean_w = n1 * n2 / 2.0;
    const double tie_sum = detail::tie_correction_sum(combined);
    const double var_w = (n1 * n2 / 12.0) * ((n + 1.0) - tie_sum / (n * (n - 1.0)));
    const double sigma = std::sqrt(var_w);

    double correction;
    switch (alternative) {
        case Alternative::Greater: correction = 0.5; break;
        case Alternative::Less: correction = -0.5; break;
        case Alternative::TwoSided:
        default: correction = 0.5 * sign(w - mean_w);
    }
    const double z = (w - mean_w - correction) / sigma;

    HypothesisTestResult result;
    result.statistic = w;
    result.p_value = normal_p_value(z, alternative);
    result.alternative = alternative;
    result.method = "Wilcoxon rank sum test with continuity correction";
    return result;
}

} // namespace datamunge::stats
