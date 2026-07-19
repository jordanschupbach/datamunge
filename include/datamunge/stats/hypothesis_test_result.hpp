#pragma once

#include <string>

namespace datamunge::stats {

/// @brief Which tail(s) of the null distribution the alternative hypothesis covers,
///        matching R's `alternative` argument across t.test/wilcox.test/cor.test/etc.
enum class Alternative { TwoSided, Less, Greater };

/// @brief A generic hypothesis-test result, reused across every test in this module the way
///        R's `htest` S3 class is reused across t.test/wilcox.test/chisq.test/ks.test/etc.
///        Not every field is meaningful for every test -- e.g. chi-squared and KS tests
///        don't report a confidence interval, so conf_int_lower/upper are left at 0; see
///        each test function's doc comment for which fields it actually fills in.
struct HypothesisTestResult {
    double statistic{0.0};    // the test statistic (t, W, D, X-squared, F, H, S, z, ...)
    double parameter1{0.0};   // primary degrees of freedom (or df1, for F-tests)
    double parameter2{0.0};   // secondary degrees of freedom (df2, F-tests only; else 0)
    double p_value{0.0};

    double estimate1{0.0}; // e.g. a sample mean, mean difference, correlation, proportion
    double estimate2{0.0}; // e.g. a second group's mean/proportion (0 when not applicable)

    double conf_int_lower{0.0};
    double conf_int_upper{0.0};
    bool   has_conf_int{false};

    Alternative alternative{Alternative::TwoSided};
    std::string method; // human-readable test name, e.g. "Welch Two Sample t-test"
};

} // namespace datamunge::stats
