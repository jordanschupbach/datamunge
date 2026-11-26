#pragma once

/// @file equivalence.hpp
/// @brief Two one-sided tests (TOST) for means and noninferiority tests for a risk difference.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Controls for the TOST procedures.
/// @par Example
/// @code
/// const dm::stats::TostOptions options{.lower_margin = -1, .upper_margin = 1};
/// std::println("{}", options.alpha); // 0.05
/// @endcode
struct TostOptions {
    double lower_margin{};    ///< Lower equivalence bound \f$\Delta_L\f$ on the mean (difference) scale.
    double upper_margin{};    ///< Upper equivalence bound \f$\Delta_U>\Delta_L\f$.
    double alpha{.05};        ///< Level of each one-sided test in (0, 0.5).
    bool equal_variance{false}; ///< Two-sample only: pooled variance instead of Welch.
};

/// @brief Result of a two one-sided tests procedure.
/// @par Example
/// @code
/// const dm::stats::TostResult result;
/// std::println("{}", result.equivalent); // false
/// @endcode
struct TostResult {
    double estimate{};       ///< Mean, mean difference \f$\bar x-\bar y\f$, or mean of paired differences.
    double std_error{};      ///< Standard error of the estimate.
    double df{};             ///< Degrees of freedom (Welch–Satterthwaite when variances are unequal).
    double t_lower{};        ///< \f$(\hat\delta-\Delta_L)/\mathrm{SE}\f$, testing \f$H_0:\delta\le\Delta_L\f$.
    double t_upper{};        ///< \f$(\hat\delta-\Delta_U)/\mathrm{SE}\f$, testing \f$H_0:\delta\ge\Delta_U\f$.
    double p_lower{};        ///< Upper-tail p-value of `t_lower`.
    double p_upper{};        ///< Lower-tail p-value of `t_upper`.
    double p_value{};        ///< \f$\max(p_L,p_U)\f$, the TOST p-value.
    double conf_low{};       ///< Lower limit of the \f$100(1-2\alpha)\%\f$ interval.
    double conf_high{};      ///< Upper limit of the \f$100(1-2\alpha)\%\f$ interval.
    bool equivalent{};       ///< Whether both one-sided tests reject at level alpha.
};

/// @brief One-sample or paired TOST: is the mean within \f$(\Delta_L,\Delta_U)\f$?
///
/// Pass paired differences for a paired design. Rejecting both
/// \f$H_0:\mu\le\Delta_L\f$ and \f$H_0:\mu\ge\Delta_U\f$ at level \f$\alpha\f$ establishes
/// equivalence at level \f$\alpha\f$; equivalently, the \f$100(1-2\alpha)\%\f$ t interval lies
/// inside the margins. Matches `TOSTER::t_TOST(x, eqb = c(L, U))` and two `t.test` calls.
/// @param x At least two finite observations, not all equal.
/// @param options Margins, alpha (variance option unused).
/// @return Estimate, both one-sided tests, the TOST p-value and the interval.
/// @throws std::invalid_argument For fewer than two or nonfinite values, zero variance,
/// nonfinite or unordered margins, or alpha outside (0, 0.5).
/// @par Example
/// @code
/// const auto r = dm::stats::tost_mean({.2, -.1, .4, .1, 0, .3, -.2, .1}, {.lower_margin = -.5, .upper_margin = .5});
/// std::println("{:.4f} {}", r.p_value, r.equivalent); // 0.0004 true
/// @endcode
/// @par References
/// - D. J. Schuirmann (1987). "A comparison of the two one-sided tests procedure and the power
///   approach for assessing the equivalence of average bioavailability". Journal of
///   Pharmacokinetics and Biopharmaceutics 15(6), 657-680. doi:10.1007/BF01068419
/// - D. Lakens (2017). "Equivalence tests: a practical primer for t tests, correlations, and
///   meta-analyses". Social Psychological and Personality Science 8(4), 355-362. doi:10.1177/1948550617697177
/// @see tost_mean_difference()
[[nodiscard]] TostResult tost_mean(const std::vector<double>& x, const TostOptions& options);

/// @brief Two-sample TOST for the difference of means \f$\mu_x-\mu_y\f$.
///
/// Uses the Welch standard error and Satterthwaite degrees of freedom by default,
/// or the pooled variance with \f$n_x+n_y-2\f$ degrees of freedom.
/// @param x First sample, at least two finite values.
/// @param y Second sample, at least two finite values.
/// @param options Margins, alpha and the variance assumption.
/// @return Estimate, both one-sided tests, the TOST p-value and the interval.
/// @throws std::invalid_argument For a sample with fewer than two or nonfinite values, a zero
/// standard error, nonfinite or unordered margins, or alpha outside (0, 0.5).
/// @par Example
/// @code
/// const auto r = dm::stats::tost_mean_difference({5.1, 4.8, 5.3, 5.0, 4.9}, {5.0, 5.2, 4.7, 5.1, 4.9},
///                                                {.lower_margin = -.5, .upper_margin = .5});
/// std::println("{:.2f} [{:.3f}, {:.3f}]", r.estimate, r.conf_low, r.conf_high); // 0.04 [-0.186, 0.266]
/// @endcode
/// @par References
/// D. J. Schuirmann (1987), doi:10.1007/BF01068419
/// @see tost_mean()
[[nodiscard]] TostResult tost_mean_difference(const std::vector<double>& x, const std::vector<double>& y,
                                              const TostOptions& options);

/// @brief Test statistic used by noninferiority_risk_difference().
/// @par Example
/// @code
/// const auto method = dm::stats::RiskDifferenceMethod::wald;
/// std::println("{}", method == dm::stats::RiskDifferenceMethod::farrington_manning); // false
/// @endcode
enum class RiskDifferenceMethod {
    wald,               ///< Unrestricted variance \f$\hat p_1\hat q_1/n_1+\hat p_2\hat q_2/n_2\f$.

    /// Variance at the maximum likelihood estimates restricted to the null boundary
    /// (Farrington and Manning); the interval inverts this score statistic.
    farrington_manning,
};

/// @brief Noninferiority test result for a risk difference.
/// @par Example
/// @code
/// const dm::stats::RiskDifferenceTest result;
/// std::println("{}", result.noninferior); // false
/// @endcode
struct RiskDifferenceTest {
    double estimate{};        ///< \f$\hat p_1-\hat p_2\f$ (treatment minus reference).
    double statistic{};       ///< \f$z=(\hat p_1-\hat p_2+m)/\sqrt{V}\f$.
    double p_value{};         ///< One-sided p-value \f$1-\Phi(z)\f$ for \f$H_0:p_1-p_2\le-m\f$.
    double restricted_p1{};   ///< \f$\tilde p_1\f$ on the null boundary (the observed \f$\hat p_1\f$ for Wald).
    double restricted_p2{};   ///< \f$\tilde p_2=\tilde p_1+m\f$ (the observed \f$\hat p_2\f$ for Wald).
    double conf_low{};        ///< Lower limit of the \f$100(1-2\alpha)\%\f$ interval for \f$p_1-p_2\f$.
    double conf_high{};       ///< Upper limit of that interval.
    bool noninferior{};       ///< Whether \f$H_0\f$ is rejected at level alpha.
};

/// @brief One-sided test that a success probability is not worse than a reference by more than a margin.
///
/// Tests \f$H_0:p_1-p_2\le-m\f$ against \f$H_1:p_1-p_2>-m\f$ for independent binomial
/// counts, where higher probabilities are better. The Farrington–Manning variance uses
/// the closed-form restricted maximum likelihood estimates under \f$p_1-p_2=-m\f$ from
/// the cubic of Miettinen and Nurminen, without their \f$n/(n-1)\f$ factor. Its interval
/// collects every \f$\delta\f$ whose two-sided score test does not reject at level
/// \f$2\alpha\f$, found by bisection; the Wald interval is \f$\hat\delta\pm z_{1-\alpha}\sqrt{V}\f$.
/// Rejection is equivalent to the interval's lower limit exceeding \f$-m\f$.
/// @param successes1 Successes in the treatment group, at most `n1`.
/// @param n1 Treatment group size, positive.
/// @param successes2 Successes in the reference group, at most `n2`.
/// @param n2 Reference group size, positive.
/// @param margin Noninferiority margin \f$m\in(0,1)\f$.
/// @param alpha One-sided level in (0, 0.5).
/// @param method Wald or Farrington–Manning.
/// @return Estimate, statistic, p-value, restricted estimates, interval and decision.
/// @throws std::invalid_argument For counts above sizes, zero sizes, a margin outside (0, 1),
/// alpha outside (0, 0.5), or a zero Wald variance.
/// @par Example
/// @code
/// const auto r = dm::stats::noninferiority_risk_difference(85, 100, 88, 100, .1);
/// std::println("{:.3f} {:.4f}", r.statistic, r.p_value); // 1.419 0.0779
/// @endcode
/// @par References
/// - C. P. Farrington and G. Manning (1990). "Test statistics and sample size formulae for
///   comparative binomial trials with null hypothesis of non-zero risk difference or non-unity
///   relative risk". Statistics in Medicine 9(12), 1447-1454. doi:10.1002/sim.4780091208
/// - O. Miettinen and M. Nurminen (1985). "Comparative analysis of two rates". Statistics in
///   Medicine 4(2), 213-226. doi:10.1002/sim.4780040211
/// @see tost_mean_difference()
[[nodiscard]] RiskDifferenceTest noninferiority_risk_difference(
    std::size_t successes1, std::size_t n1, std::size_t successes2, std::size_t n2, double margin,
    double alpha = .025, RiskDifferenceMethod method = RiskDifferenceMethod::farrington_manning);

} // namespace datamunge::stats
