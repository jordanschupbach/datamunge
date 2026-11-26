#pragma once

/// @file dip_test.hpp
/// @brief Hartigan's dip statistic and a simulation-calibrated test of unimodality.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief The dip of a sample and the modal interval found with it.
/// @par Example
/// @code
/// const dm::stats::DipStatistic dip;
/// std::println("{}", dip.statistic); // 0
/// @endcode
struct DipStatistic {
    double statistic{};     ///< Dip \f$D_n\f$: half the sup distance to the closest unimodal distribution function.
    double modal_lower{};   ///< Lower end of the modal interval of the closest unimodal fit.
    double modal_upper{};   ///< Upper end of the modal interval.
};

/// @brief Controls for dip_test().
/// @par Example
/// @code
/// const dm::stats::DipTestOptions options{.replicates = 500};
/// std::println("{}", options.replicates); // 500
/// @endcode
struct DipTestOptions {
    std::size_t replicates{2000}; ///< Number of uniform samples used to calibrate the p-value.
    std::uint64_t seed{1};        ///< Seed for the uniform samples.
};

/// @brief Dip statistic with its simulation-calibrated p-value.
/// @par Example
/// @code
/// const dm::stats::DipTestResult result;
/// std::println("{}", result.replicates); // 0
/// @endcode
struct DipTestResult {
    DipStatistic dip;           ///< Statistic and modal interval of the observed sample.

    /// \f$(1+\#\{D^*\ge D_n\})/(1+B)\f$ over B uniform samples of the same size.
    double p_value{};
    std::size_t replicates{};   ///< B, the number of uniform samples used.
};

/// @brief Hartigan's dip: the maximum distance between the empirical distribution function and the closest unimodal one.
///
/// Implements the algorithm of Hartigan and Hartigan, as corrected in the R diptest
/// package: alternately fit the greatest convex minorant and least concave majorant
/// on a shrinking modal interval and record the largest deviation. The result is
/// divided by \f$2n\f$ so it lies in \f$[1/(2n),1/4]\f$. It equals `diptest::dip(x)` in R.
/// Ties are allowed. \f$O(n)\f$ after sorting.
/// @param x At least one finite observation.
/// @return Dip and modal interval.
/// @throws std::invalid_argument For an empty or nonfinite sample.
/// @par Example
/// @code
/// const auto d = dm::stats::dip_statistic({1, 2, 3, 4, 10, 11, 12, 13});
/// std::println("{:.4f}", d.statistic); // 0.1667
/// @endcode
/// @par References
/// - J. A. Hartigan and P. M. Hartigan (1985). "The dip test of unimodality". Annals of Statistics
///   13(1), 70-84. doi:10.1214/aos/1176346577
/// - P. M. Hartigan (1985). "Algorithm AS 217: computation of the dip statistic to test for
///   unimodality". Applied Statistics 34(3), 320-325. doi:10.2307/2347485
/// - https://cran.r-project.org/package=diptest
/// @see dip_test()
[[nodiscard]] DipStatistic dip_statistic(const std::vector<double>& x);

/// @brief Test of unimodality calibrated by the dip of uniform samples.
///
/// The uniform distribution is the least favorable unimodal null, so the p-value
/// \f$(1+\#\{D^*_b\ge D_n\})/(1+B)\f$ over B uniform samples of size n is conservative for
/// every unimodal law. This is the simulated calibration of `diptest::dip.test`, which
/// uses the plain proportion rather than adding one; a small p-value indicates
/// bimodality or multimodality.
/// @param x At least four finite observations.
/// @param options Number of uniform replicates (at least one) and seed.
/// @return Dip, modal interval, p-value and replicate count.
/// @throws std::invalid_argument For fewer than four or nonfinite observations, or zero replicates.
/// @par Example
/// @code
/// std::vector<double> x;
/// for (int i = 0; i < 40; ++i) x.push_back(i % 2 ? 20 + .1 * i : .1 * i);
/// const auto t = dm::stats::dip_test(x, {.replicates = 500});
/// std::println("{}", t.p_value < .01); // true
/// @endcode
/// @par References
/// J. A. Hartigan and P. M. Hartigan (1985), doi:10.1214/aos/1176346577
/// @see dip_statistic()
[[nodiscard]] DipTestResult dip_test(const std::vector<double>& x, const DipTestOptions& options = {});

} // namespace datamunge::stats
