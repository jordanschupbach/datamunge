#pragma once

#include <datamunge/stats/resample_types.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::stats {

struct BootstrapOptions {
    /// @brief Number of bootstrap resamples B.
    std::size_t num_resamples{2000};
    /// @brief Two-sided confidence level for the returned intervals.
    double confidence_level{0.95};
    std::uint64_t seed{42};
};

struct BootstrapResult {
    double estimate{0.0};        ///< the statistic evaluated on the original sample
    double bias{0.0};            ///< mean of the replicates minus the estimate
    double standard_error{0.0};  ///< standard deviation of the B replicates
    double confidence_level{0.95};

    /// @brief The B bootstrap replicates of the statistic (unsorted, in draw order).
    std::vector<double> replicates;

    /// @brief Percentile interval: the (alpha/2, 1-alpha/2) empirical quantiles of the replicates.
    std::pair<double, double> percentile_interval{0.0, 0.0};
    /// @brief Basic (reverse-percentile) interval: reflects the percentile interval through the estimate.
    std::pair<double, double> basic_interval{0.0, 0.0};
    /// @brief Normal-approximation interval: (estimate - bias) +/- z * standard_error.
    std::pair<double, double> normal_interval{0.0, 0.0};
    /// @brief Bias-corrected and accelerated (BCa) interval (Efron 1987): percentile endpoints
    ///        adjusted by a bias-correction z0 and an acceleration a from the jackknife.
    std::pair<double, double> bca_interval{0.0, 0.0};
};

/// @brief The nonparametric bootstrap (Efron 1979). Repeatedly resamples the data with
///        replacement, recomputing a statistic on each resample to build its sampling
///        distribution empirically -- from which it estimates the statistic's standard error
///        and bias and forms several kinds of confidence interval, all without any parametric
///        model or analytic variance formula.
class Bootstrap {
public:
    explicit Bootstrap(BootstrapOptions options = {});

    /// @brief Runs the bootstrap for @p statistic on @p sample.
    [[nodiscard]] BootstrapResult run(const std::vector<double>& sample, const SampleStatistic& statistic) const;

private:
    BootstrapOptions options_;
};

} // namespace datamunge::stats
