#pragma once
/// @file noncentral_distributions.hpp
/// @brief Direct and inverse probabilities for noncentral t, chi-square, and F laws.

namespace datamunge::random {

/// @brief Evaluates a tail of the noncentral t distribution.
///
/// The variate is (Z + noncentrality) / sqrt(V / df), where Z is standard
/// normal and V is independent chi-square with df degrees of freedom.
/// Integrates the selected tail over the denominator distribution.
/// @param x Evaluation point, possibly infinite.
/// @param df Positive finite degrees of freedom.
/// @param noncentrality Finite signed displacement of the numerator.
/// @param lower_tail True for P(T <= x), false for P(T > x).
/// @return Selected tail probability.
/// @throws std::invalid_argument Invalid degrees of freedom, displacement, or NaN point.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::random::noncentral_t_cdf(0, 10, 1)); // 0.159
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/TDist.html
/// - V. Witkovský (2013), https://www.dml.cz/handle/10338.dmlcz/143545
/// @see noncentral_t_quantile(), noncentral_t_noncentrality()
double noncentral_t_cdf(double x, double df, double noncentrality, bool lower_tail = true);

/// @brief Inverts a noncentral t tail for its signed cutoff.
/// @param probability Tail probability in [0,1].
/// @param df Positive finite degrees of freedom.
/// @param noncentrality Finite signed displacement of the numerator.
/// @param lower_tail Interpret probability as the lower (true) or upper tail.
/// @return Signed quantile, with infinite endpoints.
/// @throws std::invalid_argument Invalid distribution parameters or probability.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::random::noncentral_t_quantile(.05, 10, 0, false)); // 1.812
/// @endcode
/// @see noncentral_t_cdf(), noncentral_t_noncentrality()
double noncentral_t_quantile(double probability, double df, double noncentrality, bool lower_tail = true);

/// @brief Inverts a noncentral t tail for its signed numerator displacement.
/// @param probability Tail probability in [0,1].
/// @param x Finite cutoff to hold fixed.
/// @param df Positive finite degrees of freedom.
/// @param lower_tail Interpret probability as the lower (true) or upper tail.
/// @return Signed noncentrality, with infinite limits at the probability endpoints.
/// @throws std::invalid_argument Invalid cutoff, degrees of freedom, or probability.
/// @par Example
/// @code
/// const auto p = dm::random::noncentral_t_cdf(1, 10, -2);
/// std::print("{:.3f}\n", dm::random::noncentral_t_noncentrality(p, 1, 10)); // -2.000
/// @endcode
/// @see noncentral_t_cdf(), noncentral_t_quantile()
double noncentral_t_noncentrality(double probability, double x, double df, bool lower_tail = true);

/// @brief Evaluates either tail of a noncentral chi-square distribution.
///
/// Uses the Poisson mixture of central chi-square laws and a saddlepoint
/// approximation at very large noncentrality. Zero degrees of freedom
/// include an atom at zero of size exp(-noncentrality/2).
/// @param x Evaluation point, possibly infinite.
/// @param df Finite nonnegative degrees of freedom.
/// @param noncentrality Finite nonnegative noncentrality.
/// @param lower_tail True for P(X <= x), false for P(X > x).
/// @return Selected tail probability.
/// @throws std::invalid_argument Invalid degrees of freedom, noncentrality or NaN point.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::random::noncentral_chi_square_cdf(2, 2, 0)); // 0.632
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Chisquare.html
/// - Lugannani and Rice (1980), https://doi.org/10.2307/1426607
/// @see noncentral_chi_square_quantile(), noncentral_chi_square_noncentrality()
double noncentral_chi_square_cdf(double x, double df, double noncentrality, bool lower_tail = true);

/// @brief Inverts either tail of the noncentral chi-square distribution.
/// @param probability Tail probability in [0,1].
/// @param df Finite nonnegative degrees of freedom.
/// @param noncentrality Finite nonnegative noncentrality.
/// @param lower_tail Interpret the probability as the lower (true) or upper tail.
/// @return Nonnegative quantile; probabilities at the atom map to zero.
/// @throws std::invalid_argument Invalid distribution parameter or probability.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::random::noncentral_chi_square_quantile(.5, 2, 0)); // 1.386
/// @endcode
/// @see noncentral_chi_square_cdf(), noncentral_chi_square_noncentrality()
double noncentral_chi_square_quantile(double probability, double df, double noncentrality,
                                       bool lower_tail = true);

/// @brief Solves for the noncentral chi-square noncentrality at a fixed cutoff.
/// @param probability Attainable tail probability in [0,1].
/// @param x Positive finite cutoff, or zero when df is zero.
/// @param df Finite nonnegative degrees of freedom.
/// @param lower_tail Interpret the probability as the lower (true) or upper tail.
/// @return Nonnegative noncentrality, possibly infinity at the limiting probability.
/// @throws std::invalid_argument Invalid parameters or probability outside the range from the central law to the limiting law.
/// @par Example
/// @code
/// const auto p = dm::random::noncentral_chi_square_cdf(4, 3, 2);
/// std::print("{:.3f}\n", dm::random::noncentral_chi_square_noncentrality(p, 4, 3)); // 2.000
/// @endcode
/// @see noncentral_chi_square_cdf(), noncentral_chi_square_quantile()
double noncentral_chi_square_noncentrality(double probability, double x, double df,
                                            bool lower_tail = true);

/// @brief Evaluates either tail of the noncentral F distribution.
///
/// Noncentrality applies to the numerator chi-square. Uses the
/// Poisson mixture of beta tails. At very large noncentrality it uses
/// moment matching when denominator variation dominates and saddlepoint
/// probabilities when both chi-squares concentrate.
/// @param x Evaluation point, possibly infinite.
/// @param numerator_df Finite positive numerator degrees of freedom.
/// @param denominator_df Finite positive denominator degrees of freedom.
/// @param noncentrality Finite nonnegative noncentrality.
/// @param lower_tail True for P(X <= x), false for P(X > x).
/// @return Selected tail probability.
/// @throws std::invalid_argument Invalid distribution parameter or NaN point.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::random::noncentral_f_cdf(1, 2, 2, 0)); // 0.500
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Fdist.html
/// - Lugannani and Rice (1980), https://doi.org/10.2307/1426607
/// @see noncentral_f_quantile(), noncentral_f_noncentrality()
double noncentral_f_cdf(double x, double numerator_df, double denominator_df, double noncentrality,
                        bool lower_tail = true);

/// @brief Inverts either tail of the noncentral F distribution.
/// @param probability Tail probability in [0,1].
/// @param numerator_df Finite positive numerator degrees of freedom.
/// @param denominator_df Finite positive denominator degrees of freedom.
/// @param noncentrality Finite nonnegative noncentrality.
/// @param lower_tail Interpret the probability as the lower (true) or upper tail.
/// @return Nonnegative quantile, with zero and infinity endpoints.
/// @throws std::invalid_argument Invalid distribution parameter or probability.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::random::noncentral_f_quantile(.5, 2, 2, 0)); // 1.000
/// @endcode
/// @see noncentral_f_cdf(), noncentral_f_noncentrality()
double noncentral_f_quantile(double probability, double numerator_df, double denominator_df,
                             double noncentrality, bool lower_tail = true);

/// @brief Solves for the numerator noncentrality at a fixed noncentral F cutoff.
/// @param probability Attainable tail probability in [0,1].
/// @param x Positive finite cutoff.
/// @param numerator_df Finite positive numerator degrees of freedom.
/// @param denominator_df Finite positive denominator degrees of freedom.
/// @param lower_tail Interpret the probability as the lower (true) or upper tail.
/// @return Nonnegative noncentrality, possibly infinity at the limiting probability.
/// @throws std::invalid_argument Invalid parameters or probability outside the range from the central law to the limiting law.
/// @par Example
/// @code
/// const auto p = dm::random::noncentral_f_cdf(2, 4, 12, 3);
/// std::print("{:.3f}\n", dm::random::noncentral_f_noncentrality(p, 2, 4, 12)); // 3.000
/// @endcode
/// @see noncentral_f_cdf(), noncentral_f_quantile()
double noncentral_f_noncentrality(double probability, double x, double numerator_df, double denominator_df,
                                  bool lower_tail = true);

} // namespace datamunge::random
