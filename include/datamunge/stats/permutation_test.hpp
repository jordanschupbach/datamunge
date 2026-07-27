#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief A statistic of two samples (e.g. the difference in their means). Under the
///        permutation null the group labels are exchangeable, so this is recomputed on many
///        relabelings to build the statistic's null distribution.
using TwoSampleStatistic = std::function<double(const std::vector<double>&, const std::vector<double>&)>;

struct PermutationTestOptions {
    std::size_t num_permutations{5000};
    Alternative alternative{Alternative::TwoSided};
    std::uint64_t seed{42};
};

struct PermutationTestResult {
    double observed{0.0};                ///< the statistic on the original labeling
    double p_value{0.0};                 ///< Monte Carlo p-value, (1 + #extreme) / (B + 1)
    double null_mean{0.0};               ///< mean of the permutation (null) distribution
    double null_sd{0.0};                 ///< standard deviation of the permutation distribution
    std::size_t num_permutations{0};
    std::vector<double> null_distribution;  ///< the B permuted statistic values
};

/// @brief A two-sample permutation (randomization) test. Under the null hypothesis that @p x
///        and @p y come from the same distribution, the group labels are exchangeable, so the
///        sampling distribution of any test statistic is obtained by recomputing it on random
///        relabelings of the pooled data. The p-value is the fraction of permutations whose
///        statistic is at least as extreme as the observed one -- an exact, assumption-light
///        alternative to a t-test that needs no normality and no analytic null distribution.
///
/// @param statistic the two-sample statistic; if empty (default), the difference in means
///        mean(x) - mean(y) is used.
[[nodiscard]] PermutationTestResult permutation_test_two_sample(const std::vector<double>& x,
                                                                const std::vector<double>& y,
                                                                const TwoSampleStatistic& statistic = {},
                                                                PermutationTestOptions options = {});

} // namespace datamunge::stats
