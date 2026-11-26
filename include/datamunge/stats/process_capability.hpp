#pragma once

/// @file process_capability.hpp
/// @brief Process capability indices Cp, Cpk, Pp, Ppk and Cpm with confidence intervals and expected
/// nonconforming rates, from subgrouped or individual data.

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Estimator of the within-subgroup (short-term) standard deviation.
/// @par Example
/// @code
/// const auto w = dm::stats::WithinSigma::average_range;
/// std::println("{}", w == dm::stats::WithinSigma::average_range); // true
/// @endcode
enum class WithinSigma {
    pooled_sd,     ///< \f$s_p/c_4(d+1)\f$ with \f$s_p^2=\sum(n_i-1)s_i^2/d\f$, \f$d=\sum(n_i-1)\f$ (Minitab default).
    average_range, ///< Mean of \f$R_i/d_2(n_i)\f$ (the X-bar/R chart estimate).
    average_sd,    ///< Mean of \f$s_i/c_4(n_i)\f$ (the X-bar/S chart estimate).
    moving_range,  ///< Average moving range of span k divided by \f$d_2(k)\f$, over the data in time order.
};

/// @brief Options for process_capability().
/// @par Example
/// @code
/// dm::stats::CapabilityOptions o{.lsl = 9, .usl = 11};
/// std::println("{}", o.usl - o.lsl); // 2
/// @endcode
struct CapabilityOptions {
    double lsl{std::numeric_limits<double>::quiet_NaN()};    ///< Lower specification limit (NaN: none).
    double usl{std::numeric_limits<double>::quiet_NaN()};    ///< Upper specification limit (NaN: none).
    double target{std::numeric_limits<double>::quiet_NaN()}; ///< Target for Cpm (NaN: midpoint of two limits).
    WithinSigma within{WithinSigma::pooled_sd};               ///< Within-subgroup estimator.
    std::size_t moving_range_span{2};                         ///< Span k of the moving range.
    bool unbiased_overall{false};    ///< Divide the overall sample SD by c4(n).
    double confidence{0.95};         ///< Confidence level of the intervals.
};

/// @brief A capability index with a two-sided confidence interval.
/// @par Example
/// @code
/// dm::stats::CapabilityIndex c{.value = 1.33};
/// std::println("{}", c.value); // 1.33
/// @endcode
struct CapabilityIndex {
    double value{std::numeric_limits<double>::quiet_NaN()}; ///< Estimate (NaN when a needed limit is missing).
    double lower{std::numeric_limits<double>::quiet_NaN()}; ///< Lower confidence limit.
    double upper{std::numeric_limits<double>::quiet_NaN()}; ///< Upper confidence limit.
};

/// @brief Result of process_capability().
/// @par Example
/// @code
/// dm::stats::ProcessCapability c;
/// std::println("{}", c.n); // 0
/// @endcode
struct ProcessCapability {
    std::size_t n{};             ///< Observations.
    std::size_t subgroups{};     ///< Subgroups.
    double mean{};               ///< Grand mean.
    double sigma_within{};       ///< Within (short-term) standard deviation.
    double sigma_overall{};      ///< Overall (long-term) standard deviation.
    CapabilityIndex cp;          ///< (USL - LSL) / (6 sigma_within); chi-square interval on n - 1 df.
    CapabilityIndex cpl;         ///< (mean - LSL) / (3 sigma_within); Bissell interval.
    CapabilityIndex cpu;         ///< (USL - mean) / (3 sigma_within); Bissell interval.
    CapabilityIndex cpk;         ///< min(Cpl, Cpu), or the single one-sided index; Bissell interval.
    CapabilityIndex pp;          ///< Cp with sigma_overall.
    CapabilityIndex ppl;         ///< Cpl with sigma_overall.
    CapabilityIndex ppu;         ///< Cpu with sigma_overall.
    CapabilityIndex ppk;         ///< Cpk with sigma_overall.
    CapabilityIndex cpm;         ///< (USL - LSL) / (6 sqrt(sigma_overall^2 + (mean - T)^2)); Boyles' chi-square interval.
    double expected_ppm_within{};  ///< Normal-theory parts per million outside the limits, within sigma.
    double expected_ppm_overall{}; ///< Same with overall sigma.
    double observed_ppm{};         ///< Observed parts per million outside the limits.
};

/// @brief Capability analysis of subgrouped data.
///
/// Cp-type indices use the within-subgroup SD and describe potential capability if between-subgroup shifts
/// were removed; Pp-type indices use the overall SD and describe actual performance. Both assume normal,
/// independent data from a stable process. Cp and Pp intervals use \f$\chi^2_{n-1}\f$; Cpk-type intervals
/// use Bissell's normal approximation \f$\hat C(1\pm z\sqrt{1/(9n\hat C^2)+1/(2(n-1))})\f$, as in `qcc`.
/// @param subgroups Measurements in time order, grouped (sizes may differ; size 1 requires moving_range).
/// @param options Specification limits and estimator choices.
/// @return Indices.
/// @throws std::invalid_argument Without any limit, with LSL >= USL, too few data, nonfinite values, an
/// estimator that needs subgroups of size >= 2, or zero spread.
/// @par Example
/// @code
/// const auto c = dm::stats::process_capability(std::vector<std::vector<double>>{{9.9, 10.1}, {10.0, 10.2}},
///                                              {.lsl = 9, .usl = 11});
/// std::println("{:.2f}", c.cp.value); // 2.09
/// @endcode
ProcessCapability process_capability(const std::vector<std::vector<double>>& subgroups,
                                     const CapabilityOptions& options);

/// @brief Capability analysis of individual measurements in time order (moving-range within SD).
/// @param values Measurements.
/// @param options Options; `within` is replaced by moving_range.
/// @return Indices.
/// @throws std::invalid_argument As for the subgrouped version.
/// @par Example
/// @code
/// const std::vector<double> x{10, 10.2, 9.9, 10.1, 10.0};
/// const auto c = dm::stats::process_capability(x, {.lsl = 9, .usl = 11});
/// std::println("{:.2f}", c.ppk.value); // 2.81
/// @endcode
ProcessCapability process_capability(const std::vector<double>& values, CapabilityOptions options);

/// @brief Expected range of n standard normal observations, \f$d_2(n)=\int(1-\Phi^n-(1-\Phi)^n)\,dx\f$.
/// @param n Sample size (at least 2).
/// @return d2.
/// @throws std::invalid_argument For n < 2.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::range_constant_d2(5)); // 2.3259
/// @endcode
double range_constant_d2(std::size_t n);

/// @brief Bias constant of the sample SD, \f$c_4(n)=\sqrt{2/(n-1)}\,\Gamma(n/2)/\Gamma((n-1)/2)\f$.
/// @param n Sample size (at least 2).
/// @return c4.
/// @throws std::invalid_argument For n < 2.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::sd_constant_c4(5)); // 0.9400
/// @endcode
double sd_constant_c4(std::size_t n);

} // namespace datamunge::stats
