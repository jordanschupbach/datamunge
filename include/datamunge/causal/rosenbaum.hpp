#pragma once

/// @file rosenbaum.hpp
/// @brief Rosenbaum sensitivity bounds for matched observational studies.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::causal {

/// @brief Bounds on a one-sided p-value when hidden bias of at most gamma is allowed.
///
/// Within a matched set, units with the same observed covariates may differ in their odds of
/// treatment by at most a factor gamma because of an unobserved covariate. For each gamma the
/// randomisation distribution of the test statistic is not known, but its p-value lies between the
/// value under the assignment most favourable to the treatment effect (`p_lower`) and the value
/// under the least favourable one (`p_upper`). At gamma = 1 both equal the ordinary randomisation
/// p-value. The alternative is that treatment raises the response.
/// @par Example
/// @code
/// dm::causal::RosenbaumBound b;
/// b.gamma = 2;
/// b.p_upper = 0.03;
/// std::println("{}", b.p_upper < 0.05); // true
/// @endcode
struct RosenbaumBound {
    double gamma{1};          ///< Sensitivity parameter (at least 1).
    double statistic{};       ///< Observed test statistic.
    double expectation{};     ///< Null expectation of the statistic at the upper bound.
    double variance{};        ///< Null variance of the statistic at the upper bound.
    double deviate{};         ///< (statistic - expectation) / sqrt(variance); NaN for exact tests.
    double p_upper{};         ///< Largest one-sided p-value consistent with gamma.
    double p_lower{};         ///< Smallest one-sided p-value consistent with gamma.
};

/// @brief Sensitivity bounds for Wilcoxon's signed-rank test in matched pairs.
///
/// Zero differences are dropped and ties get average ranks. With \f$T^+\f$ the sum of the ranks
/// of the positive differences and \f$p=\Gamma/(1+\Gamma)\f$, the upper bound uses the normal
/// approximation with \f$E=p\sum r_i\f$ and \f$V=p(1-p)\sum r_i^2\f$; the lower bound uses
/// \f$1/(1+\Gamma)\f$ in place of p. This is the test of R's `rbounds::psens`.
/// @param differences Treated-minus-control response in each pair.
/// @param gamma Sensitivity parameter (at least 1).
/// @return The bounds; the expectation and variance refer to the upper bound.
/// @throws std::invalid_argument For gamma below 1, nonfinite values or no nonzero differences.
/// @par Example
/// @code
/// const std::vector<double> d{1.2, 0.8, -0.3, 2.1, 0.5, 1.7, 0.9, -0.1, 1.4, 0.6};
/// const auto b = dm::causal::rosenbaum_wilcoxon(d, 1.0);
/// std::println("{} {:.1f}", b.statistic, b.expectation); // 52 27.5
/// @endcode
RosenbaumBound rosenbaum_wilcoxon(const std::vector<double>& differences, double gamma);

/// @brief Exact sensitivity bounds for the sign test or McNemar's test on discordant pairs.
///
/// Among the D = `treated_events + control_events` discordant pairs, `treated_events` is the number
/// in which only the treated unit had the event. Under hidden bias gamma it is stochastically at
/// most Binomial(D, gamma/(1+gamma)) and at least Binomial(D, 1/(1+gamma)), so the bounds are exact
/// binomial upper tails \f$P(X\ge T)\f$. The function differs from `rbounds::binarysens`, which
/// reports \f$P(X>\max(T, D-T))\f$.
/// @param treated_events Discordant pairs with the event in the treated unit only.
/// @param control_events Discordant pairs with the event in the control unit only.
/// @param gamma Sensitivity parameter (at least 1).
/// @return The bounds; `deviate` is NaN.
/// @throws std::invalid_argument For gamma below 1 or no discordant pairs.
/// @par Example
/// @code
/// const auto b = dm::causal::rosenbaum_sign_test(12, 3, 1.0);
/// std::println("{:.4f}", b.p_upper); // 0.0176
/// @endcode
RosenbaumBound rosenbaum_sign_test(std::size_t treated_events, std::size_t control_events, double gamma);

/// @brief Huber-type scores of Rosenbaum's m-test.
///
/// Pairwise differences are divided by the `scale_quantile` quantile h of all absolute
/// within-set differences and passed through
/// \f$\psi(x)=\mathrm{sign}(x)\min\{1,\max(0,|x|-\mathrm{inner})/(\mathrm{trim}-\mathrm{inner})\}\f$.
/// Without trimming (trim infinite), \f$\psi(x)=\mathrm{sign}(x)\max(0,|x|-\mathrm{inner})\f$ and, when
/// inner is also 0, no scaling is done. The defaults reproduce `sensitivitymv::senmv`'s method "h".
/// @par Example
/// @code
/// const dm::causal::MTestOptions o{.trim = 3.0};
/// std::println("{}", o.inner); // 0
/// @endcode
struct MTestOptions {
    double inner{0.0};          ///< Scaled differences below this in magnitude score 0.
    double trim{2.5};           ///< Scaled differences above this score +-1 (infinity for no trimming).
    double scale_quantile{0.5}; ///< Quantile of |differences| used as the scale h, in (0, 1).
    double shift{0.0};          ///< Hypothesised additive effect tau subtracted from treated responses.
};

/// @brief Sensitivity bounds for Rosenbaum's m-test in matched sets with one treated unit.
///
/// Row i holds the treated response in column 0 and the controls in the other columns; NaN marks a
/// missing control, so sets may differ in size. Each unit's score is the mean over the other units in
/// its set of \f$\psi((y_j-y_k)/h)\f$, and the statistic is the sum of the treated scores. For each
/// set, the largest null expectation over the assignments allowed by gamma (with its variance, the
/// larger one on ties) is found among the orderings that put the larger scores at odds gamma, and the
/// normal deviate gives `p_upper`. This is the separable approximation of Gastwirth, Krieger and
/// Rosenbaum (2000) and equals `sensitivitymv::senmv`. `p_lower` uses the smallest expectations.
/// @param sets Matched sets, one row per set, treated unit first.
/// @param gamma Sensitivity parameter (at least 1).
/// @param options Score function and hypothesised shift.
/// @return The bounds.
/// @throws std::invalid_argument For gamma below 1, a missing treated value, a zero scale, bad
/// options, or no set with a control.
/// @par Example
/// @code
/// dm::linalg::DenseMatrix<double> y(6, 2);
/// const double t[6]{3.1, 2.4, 4.0, 1.9, 3.3, 2.8}, c[6]{2.0, 2.6, 2.9, 1.1, 2.2, 1.5};
/// for (int i = 0; i < 6; ++i) {
///     y(i, 0) = t[i];
///     y(i, 1) = c[i];
/// }
/// std::println("{}", dm::causal::rosenbaum_m_test(y, 1.0).p_upper < 0.05); // true
/// @endcode
RosenbaumBound rosenbaum_m_test(const linalg::DenseMatrix<double>& sets, double gamma, const MTestOptions& options = {});

/// @brief Sensitivity value: the largest gamma at which an upper-bound p-value stays at or below alpha.
///
/// Bisects on log gamma; the upper-bound p-value increases with gamma.
/// @param upper_p_value Map from gamma to the upper-bound p-value.
/// @param alpha Significance level in (0, 1).
/// @param max_gamma Largest gamma considered.
/// @return The sensitivity value; 1 when the gamma = 1 test is not significant and `max_gamma`
/// when significance survives it.
/// @throws std::invalid_argument For alpha outside (0, 1) or max_gamma below 1.
/// @par Example
/// @code
/// const auto f = [](double g) { return dm::causal::rosenbaum_sign_test(30, 5, g).p_upper; };
/// const double g = dm::causal::rosenbaum_sensitivity_value(f, 0.05);
/// std::println("{}", g > 2 && g < 4); // true
/// @endcode
double rosenbaum_sensitivity_value(const std::function<double(double)>& upper_p_value, double alpha = 0.05,
                                   double max_gamma = 1000.0);

/// @brief Amplification of gamma into the pair (Lambda, Delta) of Rosenbaum and Silber (2009).
///
/// A hidden bias gamma is equivalent to an unobserved covariate that raises the odds of treatment by
/// Lambda and the odds of a higher response by Delta, with \f$\Gamma=(\Lambda\Delta+1)/(\Lambda+\Delta)\f$.
/// The function returns \f$\Delta=(\Gamma\Lambda-1)/(\Lambda-\Gamma)\f$, as `sensitivitymv::amplify`.
/// @param gamma Sensitivity parameter (above 1).
/// @param lambda Effect of the covariate on treatment odds (above gamma).
/// @return The matching Delta.
/// @throws std::invalid_argument When gamma is not above 1 or lambda not above gamma.
/// @par Example
/// @code
/// std::println("{:.2f}", dm::causal::rosenbaum_amplify(2.0, 3.0)); // 5.00
/// @endcode
double rosenbaum_amplify(double gamma, double lambda);

} // namespace datamunge::causal
