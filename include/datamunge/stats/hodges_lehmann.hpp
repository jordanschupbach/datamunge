#pragma once

/// @file hodges_lehmann.hpp
/// @brief Hodges–Lehmann location and shift estimates with rank-based confidence intervals.

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Reference distribution used to pick the order statistics bounding a Hodges–Lehmann interval.
/// @par Example
/// @code
/// const auto method = dm::stats::HodgesLehmannInterval::exact;
/// std::println("{}", method == dm::stats::HodgesLehmannInterval::normal); // false
/// @endcode
enum class HodgesLehmannInterval {
    automatic, ///< Exact when every sample has fewer than 50 values and no repeated value, else normal.
    exact,     ///< Exact Wilcoxon signed-rank or Mann–Whitney null distribution (continuous data).
    normal,    ///< Continuity-corrected normal approximation to that distribution, without tie correction.
};

/// @brief Controls for hodges_lehmann_location() and hodges_lehmann_shift().
/// @par Example
/// @code
/// const dm::stats::HodgesLehmannOptions options{.conf_level=.9};
/// std::println("{}", options.conf_level); // 0.9
/// @endcode
struct HodgesLehmannOptions {
    double conf_level{.95};                                   ///< Nominal coverage in (0, 1).
    Alternative alternative{Alternative::TwoSided};           ///< Two-sided, or a one-sided bound.
    HodgesLehmannInterval method{HodgesLehmannInterval::automatic}; ///< Reference distribution.
};

/// @brief A Hodges–Lehmann estimate and its distribution-free confidence interval.
/// @par Example
/// @code
/// const dm::stats::HodgesLehmannResult result;
/// std::println("{}", result.pair_count); // 0
/// @endcode
struct HodgesLehmannResult {
    double estimate{};          ///< Median of the Walsh averages or of the pairwise differences.
    double conf_int_lower{};    ///< Lower bound; negative infinity for `Alternative::Less`.
    double conf_int_upper{};    ///< Upper bound; positive infinity for `Alternative::Greater`.

    /// Coverage the chosen order statistics attain under the reference distribution;
    /// at least `conf_level` for the exact method, apart from the smallest samples.
    double achieved_conf_level{};
    std::size_t pair_count{};   ///< Number of Walsh averages \f$n(n+1)/2\f$ or differences \f$mn\f$.
    std::size_t lower_rank{};   ///< One-based order statistic used as the lower bound (0 when unbounded).
    std::size_t upper_rank{};   ///< One-based order statistic used as the upper bound (0 when unbounded).
    bool exact{};               ///< Whether the exact reference distribution was used.
};

/// @brief One-sample Hodges–Lehmann pseudomedian with a Wilcoxon signed-rank interval.
///
/// The estimate is the median of the \f$n(n+1)/2\f$ Walsh averages
/// \f$(x_i+x_j)/2,\ i\le j\f$, the center of symmetry estimated by inverting the
/// signed-rank test. With \f$q\f$ the smallest value whose signed-rank null tail
/// \f$P(V\le q)\f$ reaches \f$\alpha/2\f$ (or \f$\alpha\f$ one-sided), at least 1, the
/// interval is \f$[W_{(q)},W_{(N-q+1)}]\f$ over the sorted averages. This matches
/// `wilcox.test(x, conf.int = TRUE, exact = TRUE)` in R. The normal method chooses
/// \f$q=\lceil N/2-1/2-z\sigma\rceil\f$ with \f$\sigma^2=n(n+1)(2n+1)/24\f$; R's normal
/// interval instead solves its tie- and continuity-corrected statistic by root finding,
/// so the two can differ by one order statistic. The interval does not depend on a
/// null location, so no value is discarded. Uses \f$O(n^2)\f$ memory and time.
/// @param x At least two finite observations.
/// @param options Coverage, sidedness and reference distribution.
/// @return Estimate, interval, achieved coverage and the bounding ranks.
/// @throws std::invalid_argument For fewer than two or nonfinite values, a coverage outside (0, 1),
/// an invalid alternative, or an exact request with more than 1000 values.
/// @par Example
/// @code
/// const auto hl = dm::stats::hodges_lehmann_location({1.1, 2.3, 0.4, 3.8, 2.9, 1.7});
/// std::println("{:.2f} [{:.2f}, {:.2f}]", hl.estimate, hl.conf_int_lower, hl.conf_int_upper);
/// // 2.00 [0.40, 3.80]
/// @endcode
/// @par References
/// - J. L. Hodges and E. L. Lehmann (1963). "Estimates of location based on rank tests".
///   Annals of Mathematical Statistics 34(2), 598-611. doi:10.1214/aoms/1177704172
/// - D. F. Bauer (1972). "Constructing confidence sets using rank statistics". JASA 67(339),
///   687-690. doi:10.1080/01621459.1972.10481279
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
/// @see hodges_lehmann_shift(), wilcoxon_signed_rank_exact_test()
[[nodiscard]] HodgesLehmannResult hodges_lehmann_location(
    const std::vector<double>& x, const HodgesLehmannOptions& options = {});

/// @brief Two-sample Hodges–Lehmann shift with a Mann–Whitney interval.
///
/// The estimate is the median of the \f$mn\f$ differences \f$x_i-y_j\f$, the location
/// shift of x relative to y under a common-shape shift model. The interval
/// \f$[D_{(q)},D_{(mn-q+1)}]\f$ uses the Mann–Whitney null distribution as in
/// `wilcox.test(x, y, conf.int = TRUE, exact = TRUE)`; the normal method uses mean
/// \f$mn/2\f$ and variance \f$mn(m+n+1)/12\f$ in the same rounding as
/// hodges_lehmann_location(). Uses \f$O(mn)\f$ memory and time.
/// @param x Nonempty finite first sample.
/// @param y Nonempty finite second sample.
/// @param options Coverage, sidedness and reference distribution.
/// @return Estimate, interval, achieved coverage and the bounding ranks.
/// @throws std::invalid_argument For empty or nonfinite samples, a coverage outside (0, 1), an
/// invalid alternative, or an exact request with more than 40,000 differences.
/// @par Example
/// @code
/// const auto hl = dm::stats::hodges_lehmann_shift({5.1, 6.0, 7.2, 5.8}, {4.0, 4.9, 3.6, 5.5, 4.2});
/// std::println("{:.2f} [{:.2f}, {:.2f}]", hl.estimate, hl.conf_int_lower, hl.conf_int_upper);
/// // 1.65 [0.20, 3.20]
/// @endcode
/// @par References
/// - J. L. Hodges and E. L. Lehmann (1963), doi:10.1214/aoms/1177704172
/// - D. F. Bauer (1972), doi:10.1080/01621459.1972.10481279
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
/// @see hodges_lehmann_location(), wilcoxon_rank_sum_test()
[[nodiscard]] HodgesLehmannResult hodges_lehmann_shift(
    const std::vector<double>& x, const std::vector<double>& y, const HodgesLehmannOptions& options = {});

} // namespace datamunge::stats
