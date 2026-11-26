#pragma once

/// @file tolerance_limits.hpp
/// @brief Normal-theory and distribution-free statistical tolerance intervals.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Which tolerance bounds to compute.
/// @par Example
/// @code
/// const auto side = dm::stats::ToleranceSide::upper;
/// std::println("{}", side == dm::stats::ToleranceSide::lower); // false
/// @endcode
enum class ToleranceSide {
    two_sided, ///< An interval containing the central fraction of the population.
    lower,     ///< A lower bound exceeded by the fraction; the upper limit is \f$+\infty\f$.
    upper,     ///< An upper bound not exceeded by the fraction; the lower limit is \f$-\infty\f$.
};

/// @brief Computation of the two-sided normal tolerance factor.
/// @par Example
/// @code
/// const auto method = dm::stats::NormalToleranceMethod::howe;
/// std::println("{}", method == dm::stats::NormalToleranceMethod::exact); // false
/// @endcode
enum class NormalToleranceMethod {
    exact, ///< Numerical integration of the exact coverage probability (Odeh; Krishnamoorthy and Mathew).

    /// Howe's (1969) approximation \f$k=\sqrt{(n-1)(1+1/n)z^2_{(1+P)/2}/\chi^2_{n-1,1-\gamma}}\f$; the
    /// R tolerance package's method "HE" adds a further correction and gives slightly larger factors.
    howe,
};

/// @brief A tolerance interval and the quantities that determine it.
/// @par Example
/// @code
/// const dm::stats::ToleranceInterval interval;
/// std::println("{}", interval.feasible); // true
/// @endcode
struct ToleranceInterval {
    double lower{};      ///< Lower limit; \f$-\infty\f$ for an upper bound or an infeasible distribution-free request.
    double upper{};      ///< Upper limit; \f$+\infty\f$ for a lower bound or an infeasible distribution-free request.
    double factor{};     ///< Normal factor k with limits \f$\bar x\mp ks\f$ (zero for distribution-free limits).

    /// One-based order statistics used by distribution-free limits (0 when that side is unbounded).
    std::size_t lower_rank{};
    std::size_t upper_rank{};     ///< See `lower_rank`.

    /// Confidence actually attained: equal to the request for normal limits and at least the
    /// request for feasible distribution-free limits.
    double achieved_confidence{};
    bool feasible{true};          ///< False when the sample is too small for distribution-free limits.
};

/// @brief Normal tolerance factor k so that \f$\bar x\pm ks\f$ (or one side) contains a fraction P with given confidence.
///
/// One-sided factors are exact: \f$k=t'_{n-1,\gamma}(z_P\sqrt n)/\sqrt n\f$, a quantile of
/// the noncentral t law. The exact two-sided factor solves
/// \f$\gamma=\sqrt{2n/\pi}\int_0^\infty P\{\chi^2_{n-1}>(n-1)r(z)^2/k^2\}e^{-nz^2/2}\,dz\f$, with
/// \f$\Phi(z+r)-\Phi(z-r)=P\f$, by adaptive Simpson integration and bisection in k.
/// @param n Sample size, at least two.
/// @param coverage Population fraction P in (0, 1).
/// @param confidence Confidence level \f$\gamma\f$ in (0, 1).
/// @param side Two-sided interval or a one-sided bound.
/// @param method Exact or Howe's approximation (two-sided only; one-sided is always exact).
/// @return The factor k.
/// @throws std::invalid_argument For n below two or a level outside (0, 1).
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::normal_tolerance_factor(10, .9, .95)); // 2.8563
/// @endcode
/// @par References
/// - R. E. Odeh (1978). "Tables of two-sided tolerance factors for a normal distribution".
///   Communications in Statistics - Simulation and Computation 7(2), 183-201. doi:10.1080/03610917808812070
/// - W. G. Howe (1969). "Two-sided tolerance limits for normal populations - some improvements".
///   JASA 64(326), 610-620. doi:10.1080/01621459.1969.10500999
/// - K. Krishnamoorthy and T. Mathew (2009). Statistical Tolerance Regions. Wiley. doi:10.1002/9780470473900
/// - https://cran.r-project.org/package=tolerance
/// @see normal_tolerance_interval()
[[nodiscard]] double normal_tolerance_factor(std::size_t n, double coverage, double confidence,
                                             ToleranceSide side = ToleranceSide::two_sided,
                                             NormalToleranceMethod method = NormalToleranceMethod::exact);

/// @brief Normal tolerance interval \f$\bar x\pm ks\f$ (or one bound) from a sample.
///
/// Assumes independent observations from a normal population; the coverage
/// guarantee fails for skewed or heavy-tailed data, for which
/// nonparametric_tolerance_interval() is the alternative. s uses divisor n-1.
/// @param x At least two finite observations, not all equal.
/// @param coverage Population fraction P in (0, 1).
/// @param confidence Confidence level in (0, 1).
/// @param side Two-sided interval or a one-sided bound.
/// @param method Exact or Howe's approximation for two-sided factors.
/// @return Limits, factor and the requested confidence.
/// @throws std::invalid_argument For fewer than two or nonfinite values, a zero standard deviation,
/// or a level outside (0, 1).
/// @par Example
/// @code
/// const auto t = dm::stats::normal_tolerance_interval({10.2, 9.8, 11.1, 10.5, 9.4, 10.9}, .9, .95);
/// std::println("{:.3f} {:.3f}", t.lower, t.upper); // 7.893 12.740
/// @endcode
/// @par References
/// K. Krishnamoorthy and T. Mathew (2009), doi:10.1002/9780470473900
/// @see normal_tolerance_factor(), nonparametric_tolerance_interval()
[[nodiscard]] ToleranceInterval normal_tolerance_interval(const std::vector<double>& x, double coverage,
                                                          double confidence,
                                                          ToleranceSide side = ToleranceSide::two_sided,
                                                          NormalToleranceMethod method = NormalToleranceMethod::exact);

/// @brief Distribution-free tolerance limits from order statistics (Wilks).
///
/// For a continuous population, the coverage of \f$[X_{(r)},X_{(s)}]\f$ has a
/// \f$\mathrm{Beta}(s-r,n-s+r+1)\f$ law, so its confidence is
/// \f$P\{\mathrm{Bin}(n,P)\le s-r-1\}\f$. A two-sided interval uses the symmetric pair
/// \f$s=n-r+1\f$ with the largest feasible r; a one-sided bound uses the most extreme
/// rank needed. When even the sample extremes do not attain the confidence, limits
/// are infinite and `feasible` is false; the confidence the extremes attain is still reported.
/// @param x Nonempty finite observations.
/// @param coverage Population fraction P in (0, 1).
/// @param confidence Confidence level in (0, 1).
/// @param side Two-sided interval or a one-sided bound.
/// @return Limits, ranks, attained confidence and feasibility.
/// @throws std::invalid_argument For an empty or nonfinite sample or a level outside (0, 1).
/// @par Example
/// @code
/// std::vector<double> x(60);
/// for (std::size_t i = 0; i < x.size(); ++i) x[i] = double(i);
/// const auto t = dm::stats::nonparametric_tolerance_interval(x, .9, .95, dm::stats::ToleranceSide::upper);
/// std::println("{} {:.4f}", t.upper_rank, t.achieved_confidence); // 59 0.9862
/// @endcode
/// @par References
/// - S. S. Wilks (1941). "Determination of sample sizes for setting tolerance limits". Annals of
///   Mathematical Statistics 12(1), 91-96. doi:10.1214/aoms/1177731788
/// - https://cran.r-project.org/package=tolerance
/// @see normal_tolerance_interval()
[[nodiscard]] ToleranceInterval nonparametric_tolerance_interval(const std::vector<double>& x, double coverage,
                                                                 double confidence,
                                                                 ToleranceSide side = ToleranceSide::two_sided);

} // namespace datamunge::stats
