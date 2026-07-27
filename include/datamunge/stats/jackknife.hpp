#pragma once

#include <datamunge/stats/resample_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

struct JackknifeResult {
    double estimate{0.0};         ///< the statistic on the full sample
    double bias{0.0};             ///< (n-1) * (mean of leave-one-out values - estimate)
    double bias_corrected{0.0};   ///< estimate - bias (equivalently, the mean pseudo-value)
    double standard_error{0.0};   ///< the delete-1 jackknife standard error
    std::vector<double> leave_one_out;  ///< the n leave-one-out replicates theta_hat_(i)
    std::vector<double> pseudo_values;  ///< the n pseudo-values n*theta_hat - (n-1)*theta_hat_(i)
};

/// @brief The delete-1 (leave-one-out) jackknife (Quenouille 1949; Tukey 1958). Recomputes the
///        statistic on each of the n samples formed by dropping one observation, and from the
///        spread of those replicates estimates the statistic's bias and standard error -- a
///        deterministic, resampling-free precursor of the bootstrap that is exact for linear
///        statistics.
[[nodiscard]] JackknifeResult jackknife(const std::vector<double>& sample, const SampleStatistic& statistic);

struct DeleteDJackknifeOptions {
    /// @brief Number of observations deleted per subsample (retaining n - d). d = 1 reduces to
    ///        the ordinary jackknife. Larger d is needed for non-smooth statistics such as the median.
    std::size_t d{1};
    /// @brief If the number of \(\binom{n}{d}\) subsets exceeds this, a random Monte Carlo sample
    ///        of this many subsets is used instead of enumerating them all.
    std::size_t max_subsets{2000};
    std::uint64_t seed{42};
};

struct DeleteDJackknifeResult {
    double estimate{0.0};
    double standard_error{0.0};
    std::size_t d{1};
    std::size_t num_subsets{0};  ///< subsets actually evaluated
    bool complete{false};        ///< true iff all C(n,d) subsets were enumerated (no Monte Carlo)
};

/// @brief The delete-d jackknife (Shao & Wu 1989). Deletes d observations at a time rather than
///        one, which -- unlike the delete-1 jackknife -- yields a consistent standard error for
///        non-smooth statistics like the median and sample quantiles.
[[nodiscard]] DeleteDJackknifeResult jackknife_delete_d(const std::vector<double>& sample,
                                                        const SampleStatistic& statistic,
                                                        DeleteDJackknifeOptions options = {});

} // namespace datamunge::stats
