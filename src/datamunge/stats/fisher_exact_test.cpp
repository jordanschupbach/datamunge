#include <datamunge/stats/fisher_exact_test.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <limits>

namespace datamunge::stats {

HypothesisTestResult fisher_exact_test_2x2(std::size_t a, std::size_t b, std::size_t c, std::size_t d,
                                            Alternative alternative) {
    const std::size_t row1 = a + b;
    const std::size_t col1 = a + c;
    const std::size_t n_total = a + b + c + d;
    const std::size_t lo = (row1 > n_total - col1) ? row1 - (n_total - col1) : 0;
    const std::size_t hi = std::min(row1, col1);

    HypothesisTestResult result;
    result.statistic = static_cast<double>(a);
    result.alternative = alternative;
    result.method = "Fisher's Exact Test for Count Data";
    result.estimate1 = (b == 0 || c == 0) ? std::numeric_limits<double>::infinity()
                                           : (static_cast<double>(a) * static_cast<double>(d)) /
                                                 (static_cast<double>(b) * static_cast<double>(c));

    if (alternative == Alternative::Less) {
        result.p_value = random::hypergeometric_cdf(a, n_total, col1, row1);
    } else if (alternative == Alternative::Greater) {
        result.p_value = (a == 0) ? 1.0 : 1.0 - random::hypergeometric_cdf(a - 1, n_total, col1, row1);
    } else {
        const double observed_pmf = random::hypergeometric_pdf(a, n_total, col1, row1);
        const double threshold = observed_pmf * (1.0 + 1e-7);
        double sum = 0.0;
        for (std::size_t k = lo; k <= hi; ++k) {
            const double pmf = random::hypergeometric_pdf(k, n_total, col1, row1);
            if (pmf <= threshold) sum += pmf;
        }
        result.p_value = std::min(1.0, sum);
    }
    return result;
}

} // namespace datamunge::stats
