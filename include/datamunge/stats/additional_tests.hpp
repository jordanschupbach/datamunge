#pragma once

/// @file additional_tests.hpp
/// @brief Known-variance mean, structural break, exact and composite normality tests.
#include <cstddef>
#include <cstdint>
#include <vector>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>

namespace datamunge::stats {

/// @brief Tests a normal mean with known population standard deviation.
///
/// Uses independent observations. The reference is standard normal.
/// @param x Finite nonempty sample.
/// @param sigma Known finite positive population standard deviation.
/// @param mu Finite null mean.
/// @param alternative Tail relative to the signed first-minus-second effect.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::z_test_one_sample({1, 1}, 2, 1).p_value); // 1
/// @endcode
HypothesisTestResult z_test_one_sample(const std::vector<double>& x,
                                       double sigma,
                                       double mu = 0.0,
                                       Alternative alternative = Alternative::TwoSided);

/// @brief Tests a difference of independent normal means with known variances.
///
/// Uses standard error sqrt(sigma_x²/n_x + sigma_y²/n_y).
/// @param x Finite nonempty first sample.
/// @param y Finite nonempty second sample.
/// @param sigma_x Known finite positive standard deviation for x.
/// @param sigma_y Known finite positive standard deviation for y.
/// @param difference Finite null value of mean(x) minus mean(y).
/// @param alternative Tail relative to the signed first-minus-second effect.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::z_test_two_sample({1}, {1}, 2, 2).p_value); // 1
/// @endcode
HypothesisTestResult z_test_two_sample(const std::vector<double>& x,
                                       const std::vector<double>& y,
                                       double sigma_x,
                                       double sigma_y,
                                       double difference = 0.0,
                                       Alternative alternative = Alternative::TwoSided);

/// @brief Performs a probability-ordered exact multinomial goodness-of-fit test.
///
/// Enumerates all feasible count vectors. Probabilities may include zeros and must sum to one within 1e-12. Reports the observed table probability as statistic.
/// @param observed Counts in category order, with a positive total.
/// @param probabilities Null category probabilities.
/// @param max_states Maximum outcomes or assignments to enumerate before throwing.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
/// @throws std::length_error If the enumeration exceeds max_states.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::multinomial_test({3, 0}, {.5, .5}).p_value); // 0.25
/// @endcode
///
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Multinom.html
HypothesisTestResult multinomial_test(const std::vector<std::size_t>& observed,
                                      const std::vector<double>& probabilities,
                                      std::size_t max_states = 1000000);

/// @brief Tests equality of all coefficients across two linear regression groups.
///
/// Fits separate and shared full-rank designs with the same column meanings. Include an intercept column if required. Assumes independent normal errors with a common variance.
/// @param x First design, rows are observations.
/// @param y First responses.
/// @param z Second design with the same number of columns.
/// @param w Second responses.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::chow_test({2, 1, {1, 1}}, {1, 3}, {2, 1, {1, 1}}, {1, 3}).p_value); // 1
/// @endcode
///
/// @par References
/// - https://doi.org/10.2307/1910133
HypothesisTestResult chow_test(const linalg::DenseMatrix<double>& x,
                               const std::vector<double>& y,
                               const linalg::DenseMatrix<double>& z,
                               const std::vector<double>& w);

/// @brief Performs Barnard's unconditional exact test for two binomial proportions.
///
/// Orders tables by the pooled score statistic. Maximizes the rejection probability over the common success probability. Limited to 256 total observations.
/// The returned p-value uses a numerical upper bound from Bernstein subdivision.
/// @param a First sample successes.
/// @param b Second sample successes.
/// @param c First sample failures.
/// @param d Second sample failures.
/// @param alternative Tail relative to the signed first-minus-second effect.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::barnard_exact_test(0, 3, 3, 0).p_value); // 0.03125
/// @endcode
///
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.barnard_exact.html
HypothesisTestResult barnard_exact_test(std::size_t a,
                                        std::size_t b,
                                        std::size_t c,
                                        std::size_t d,
                                        Alternative alternative = Alternative::TwoSided);

/// @brief Performs Boschloo's unconditional exact test for two binomial proportions.
///
/// Orders tables by one-sided Fisher p-values. The two-sided convention doubles the smaller unconditional tail, capped at one. Limited to 256 total observations.
/// Each unconditional tail uses a numerical upper bound from Bernstein subdivision.
/// @param a First sample successes.
/// @param b Second sample successes.
/// @param c First sample failures.
/// @param d Second sample failures.
/// @param alternative Tail relative to the signed first-minus-second effect.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::boschloo_exact_test(0, 3, 3, 0).p_value); // 0.03125
/// @endcode
///
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boschloo_exact.html
HypothesisTestResult boschloo_exact_test(std::size_t a,
                                         std::size_t b,
                                         std::size_t c,
                                         std::size_t d,
                                         Alternative alternative = Alternative::TwoSided);

/// @brief Tests normality with Shapiro–Wilk W.
///
/// Uses Royston's coefficient and tail approximations for 4–5000 observations and the exact three-point null tail.
/// @param sample Finite nonconstant sample with 3 to 5000 observations.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::shapiro_wilk_test({-1, 0, 1}).p_value); // 1
/// @endcode
///
/// @par References
/// - https://doi.org/10.2307/2986146
HypothesisTestResult shapiro_wilk_test(const std::vector<double>& sample);

/// @brief Tests normality by refitting mean and sample variance in each simulated null sample.
///
/// Returns the maximum fitted-normal CDF discrepancy and a seeded plus-one Monte Carlo p-value. Calibration is for independent continuous observations.
/// @param sample At least four finite nonconstant observations.
/// @param draws Positive number of standard-normal null samples.
/// @param seed Seed for the local random engine.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
///
/// @par Example
/// @code
/// const auto fit = dm::stats::lilliefors_test({0, 0, 0, 1}, 19, 42);
/// std::print("{}\n", fit.statistic > 0); // true
/// @endcode
///
/// @par References
/// - https://doi.org/10.1080/01621459.1967.10482916
HypothesisTestResult lilliefors_test(const std::vector<double>& sample,
                                     std::size_t draws = 9999,
                                     std::uint64_t seed = 42);

/// @brief Tests equality of means by exhaustive relabeling of independent groups.
///
/// Uses the between-group sum of squares. Enumerates labelled allocations preserving group sizes under exchangeability.
/// @param groups Two or more nonempty finite groups.
/// @param max_states Maximum outcomes or assignments to enumerate before throwing.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
/// @throws std::length_error If the enumeration exceeds max_states.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::permutation_test_exact({{1, 2}, {3, 4}}).p_value); // 0.333333
/// @endcode
HypothesisTestResult permutation_test_exact(const std::vector<std::vector<double>>& groups,
                                            std::size_t max_states = 1000000);

/// @brief Calibrates the tie-corrected Kruskal–Wallis statistic by exhaustive relabeling.
///
/// Pools midranks before relabeling. Counts observation identities separately even when values tie.
/// @param groups Two or more nonempty finite groups.
/// @param max_states Maximum outcomes or assignments to enumerate before throwing.
/// @return Statistic, null p-value and method label.
/// @throws std::invalid_argument Invalid sample, parameters or design.
/// @throws std::length_error If the enumeration exceeds max_states.
///
/// @par Example
/// @code
/// std::print("{:.6g}\n", dm::stats::kruskal_wallis_test_exact({{1, 2}, {3, 4}}).p_value); // 0.333333
/// @endcode
///
/// @par References
/// - https://doi.org/10.1080/01621459.1952.10483441
HypothesisTestResult kruskal_wallis_test_exact(const std::vector<std::vector<double>>& groups,
                                               std::size_t max_states = 1000000);

/// @brief Fisher–Freeman–Halton exact test for association in a rectangular contingency table.
///
/// Conditions on both margins and sums the probabilities of all tables at most as likely
/// as observed. Removes empty margins. Limited to 256 observations and 64 nonempty cells.
/// @param table Rectangular row-major nonnegative counts.
/// @param max_states Maximum number of feasible tables to visit.
/// @return Observed table probability, conditional p-value and method label.
/// @throws std::invalid_argument If the table or limit is invalid.
/// @throws std::length_error If the feasible support exceeds max_states.
///
/// @par Example
/// @code
/// const auto r = dm::stats::fisher_exact_test({{0, 3}, {3, 0}});
/// std::print("{:.1f}\n", r.p_value); // 0.1
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html
HypothesisTestResult fisher_exact_test(const std::vector<std::vector<std::size_t>>& table,
                                        std::size_t max_states = 1000000);

} // namespace datamunge::stats
