#pragma once
/// @file power.hpp
/// @brief Prospective power and sample size for Gaussian and binomial designs.

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Calculates power for a one-sample t test from a standardized population mean difference.
///
/// The effect is (population mean minus null mean) divided by the population SD.
/// A two-sided alternative counts rejection in either tail.
/// @param n Number of observations, at least two.
/// @param standardized_effect Finite signed standardized mean difference.
/// @param alpha Significance level strictly between zero and one, default .05.
/// @param alternative TwoSided (default), Greater, or Less.
/// @return Probability of rejecting the null under the specified effect.
/// @throws std::invalid_argument Invalid sample size, effect, significance level, or alternative.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::stats::one_sample_t_test_power(20, .5)); // 0.565
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/power.t.test.html
/// @see paired_t_test_power(), one_sample_t_test_sample_size()
double one_sample_t_test_power(std::size_t n, double standardized_effect, double alpha = .05,
                               Alternative alternative = Alternative::TwoSided);

/// @brief Calculates power for a paired t test by treating the differences as one sample.
/// @param pairs Number of independent pairs, at least two.
/// @param standardized_difference Finite signed population mean of the paired differences minus the null,
/// divided by the population SD of the differences.
/// @param alpha Significance level strictly between zero and one, default .05.
/// @param alternative TwoSided (default), Greater, or Less for the signed differences.
/// @return Probability of rejection, counting both tails for a two-sided test.
/// @throws std::invalid_argument Invalid pair count, effect, significance level, or alternative.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::stats::paired_t_test_power(20, .5, .05, dm::stats::Alternative::Greater)); // 0.695
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/power.t.test.html
/// @see one_sample_t_test_power(), paired_t_test_sample_size()
double paired_t_test_power(std::size_t pairs, double standardized_difference, double alpha = .05,
                           Alternative alternative = Alternative::TwoSided);

/// @brief Calculates pooled-variance two-sample t power for two independent Gaussian samples.
/// @param first First group size, at least two.
/// @param second Second group size, at least two.
/// @param standardized_effect Finite signed population mean difference (first minus second, less its null)
/// divided by the common population SD.
/// @param alpha Significance level strictly between zero and one, default .05.
/// @param alternative TwoSided (default), Greater, or Less for the signed difference.
/// @return Probability of rejection, counting both tails for a two-sided test.
/// @throws std::invalid_argument Invalid sizes, effect, significance level, or alternative.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::stats::pooled_t_test_power(20, 20, .5)); // 0.338
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/power.t.test.html
/// @see pooled_t_test_sample_size()
double pooled_t_test_power(std::size_t first, std::size_t second, double standardized_effect,
                           double alpha = .05, Alternative alternative = Alternative::TwoSided);

/// @brief Finds the smallest integer one-sample size that attains a target power.
/// @param standardized_effect Finite signed standardized population effect as in one_sample_t_test_power().
/// @param target_power Desired probability of rejection strictly between zero and one.
/// @param alpha Significance level strictly between zero and one, default .05.
/// @param alternative TwoSided (default), Greater, or Less.
/// @return Required number of observations, at least two.
/// @throws std::invalid_argument Invalid effect, probability, or alternative.
/// @throws std::domain_error The effect cannot attain the target in the specified direction.
/// @throws std::overflow_error The required size is outside the range of std::size_t.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::one_sample_t_test_sample_size(.5, .8)); // 34
/// @endcode
/// @see one_sample_t_test_power(), paired_t_test_sample_size()
std::size_t one_sample_t_test_sample_size(double standardized_effect, double target_power,
                                          double alpha = .05, Alternative alternative = Alternative::TwoSided);

/// @brief Finds the smallest integer number of pairs that attains a target t-test power.
/// @param standardized_difference Finite signed population effect standardized by the SD of the differences.
/// @param target_power Desired probability of rejection strictly between zero and one.
/// @param alpha Significance level strictly between zero and one, default .05.
/// @param alternative TwoSided (default), Greater, or Less.
/// @return Required number of independent pairs, at least two.
/// @throws std::invalid_argument Invalid effect, probability, or alternative.
/// @throws std::domain_error The effect cannot attain the target in the specified direction.
/// @throws std::overflow_error The required pair count is outside the range of std::size_t.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::paired_t_test_sample_size(.5, .8)); // 34
/// @endcode
/// @see paired_t_test_power(), one_sample_t_test_sample_size()
std::size_t paired_t_test_sample_size(double standardized_difference, double target_power,
                                      double alpha = .05, Alternative alternative = Alternative::TwoSided);

/// @brief Finds the smallest integer sizes at a fixed allocation for a pooled two-sample t test.
///
/// The allocation is reduced to coprime integer units. The search admits
/// common multiples of those units with at least two observations in each group.
/// @param standardized_effect Finite signed population effect standardized by the common SD.
/// @param target_power Desired probability of rejection strictly between zero and one.
/// @param allocation Positive first- and second-group allocation units, default equal allocation.
/// @param alpha Significance level strictly between zero and one, default .05.
/// @param alternative TwoSided (default), Greater, or Less.
/// @return Required first and second group sizes.
/// @throws std::invalid_argument Invalid effect, allocation, probability, or alternative.
/// @throws std::domain_error The effect cannot attain the target in the specified direction.
/// @throws std::overflow_error No representable allocation attains the target power.
/// @par Example
/// @code
/// const auto n = dm::stats::pooled_t_test_sample_size(.6, .8, {2, 3});
/// std::print("{} {}\n", n.first, n.second); // 38 57
/// @endcode
/// @see pooled_t_test_power(), one_sample_t_test_sample_size()
std::pair<std::size_t, std::size_t> pooled_t_test_sample_size(
    double standardized_effect, double target_power, std::pair<std::size_t, std::size_t> allocation = {1, 1},
    double alpha = .05, Alternative alternative = Alternative::TwoSided);

/// @brief Calculates fixed-effects one-way ANOVA power for a specified allocation.
///
/// The common-mean null uses a central F cutoff. The alternative
/// noncentrality is the sum of squared deviations of group means from their
/// allocation-weighted mean, weighted by group size and divided by the common variance.
/// @param means Finite population means in group order, with at least two groups.
/// @param group_sizes Positive group sizes in the same order, with positive residual degrees of freedom.
/// @param common_variance Positive finite population variance shared by groups.
/// @param alpha Significance level strictly between zero and one, default .05.
/// @return Upper-tail rejection probability under the specified group means.
/// @throws std::invalid_argument Invalid means, allocation, common variance, or significance level.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::stats::one_way_anova_power({10, 11, 14}, {10, 18, 27}, 16)); // 0.795
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/power.anova.test.html
/// @see pooled_t_test_power()
double one_way_anova_power(const std::vector<double>& means, const std::vector<std::size_t>& group_sizes,
                           double common_variance, double alpha = .05);

/// @brief Computes normal-approximation power for two independent proportions at specified risks.
///
/// The rejection region uses the difference and sampling variance at the
/// supplied null risks. Power uses the difference and sampling variance
/// at the alternative risks. This is a fixed-null-variance calculation
/// with both tails included for a two-sided test. A deterministic
/// alternative is evaluated using its point mass and strict inequalities.
/// @param first First group size, positive.
/// @param second Second group size, positive.
/// @param null_risks First and second Bernoulli probabilities under the null, with positive sampling variance.
/// @param alternative_risks First and second Bernoulli probabilities under the alternative, each in [0,1].
/// @param alpha Nominal significance level in (0,1), default .05.
/// @param alternative TwoSided (default), Greater or Less for first risk minus second risk.
/// @return Approximate probability of rejection under the alternative.
/// @throws std::invalid_argument Invalid sizes, risks, significance level, or direction.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::stats::two_proportion_power(100, 100, {.4, .4}, {.6, .4})); // 0.823
/// @endcode
/// @par References
/// - C. P. Farrington and G. Manning (1990), normal power with distinct null and alternative variances,
///   https://doi.org/10.1002/sim.4780091208
/// @see two_proportion_sample_size()
double two_proportion_power(std::size_t first, std::size_t second,
                            std::pair<double, double> null_risks, std::pair<double, double> alternative_risks,
                            double alpha = .05, Alternative alternative = Alternative::TwoSided);

/// @brief Finds the smallest integer allocation attaining target two-proportion power.
///
/// Uses the specified null and alternative risks and the normal
/// approximation in two_proportion_power(). Admissible sizes are common
/// positive multiples of the coprime allocation units.
/// @param null_risks First and second Bernoulli probabilities under the null, with positive sampling variance.
/// @param alternative_risks First and second Bernoulli probabilities under the alternative, each in [0,1].
/// @param target_power Desired power in (0,1).
/// @param allocation Positive first- and second-group allocation units, default equal allocation.
/// @param alpha Nominal significance level in (0,1), default .05.
/// @param alternative TwoSided (default), Greater or Less for first risk minus second risk.
/// @return First and second group sizes that attain the target.
/// @throws std::invalid_argument Invalid risks, probabilities, allocation or direction.
/// @throws std::domain_error Target power cannot be attained in the selected direction.
/// @throws std::overflow_error Required group sizes exceed std::size_t.
/// @par Example
/// @code
/// const auto n = dm::stats::two_proportion_sample_size({.5, .5}, {1, 0}, .8);
/// std::print("{} {}\n", n.first, n.second); // 2 2
/// @endcode
/// @see two_proportion_power()
std::pair<std::size_t, std::size_t> two_proportion_sample_size(
    std::pair<double, double> null_risks, std::pair<double, double> alternative_risks,
    double target_power, std::pair<std::size_t, std::size_t> allocation = {1, 1},
    double alpha = .05, Alternative alternative = Alternative::TwoSided);

/// @brief Computes Schoenfeld's approximate number of events for a proportional-hazards comparison.
///
/// Rounds up (z critical + z power)^2 divided by allocation_fraction
/// times its complement times log(hazard_ratio)^2. In the two-sided
/// formula the critical quantile uses alpha/2. The event count refers
/// to the proportional-hazards information approximation.
/// @param hazard_ratio Positive finite hazard in the first group divided by that in the second.
/// @param allocation_fraction Planned first-group allocation fraction in (0,1).
/// @param target_power Desired power in (0,1). A target at most alpha returns zero.
/// @param alpha Nominal significance level in (0,1), default .05.
/// @param alternative TwoSided (default), Greater for a ratio above one, or Less for a ratio below one.
/// @return Approximate required total observed event count.
/// @throws std::invalid_argument Invalid ratio, probabilities, or direction.
/// @throws std::domain_error Target above alpha with a null or oppositely oriented ratio.
/// @throws std::overflow_error Required count exceeds std::size_t.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::schoenfeld_event_count(.75, .5, .8)); // 380
/// @endcode
/// @par References
/// - D. A. Schoenfeld (1983), https://pubmed.ncbi.nlm.nih.gov/6354290/
/// @see one_way_anova_power(), two_proportion_sample_size()
std::size_t schoenfeld_event_count(double hazard_ratio, double allocation_fraction, double target_power,
                                   double alpha = .05, Alternative alternative = Alternative::TwoSided);

/// @brief Variance inflation and effective size for an individual-weighted arm mean.
/// @par Example
/// @code
/// const auto arm = dm::stats::cluster_design_effect({10, 10}, .1);
/// std::print("{:.1f}\n", arm.design_effect); // 1.9
/// @endcode
/// @see cluster_design_effect(), ClusterRandomizedPower
struct ClusterDesignEffect {
  double design_effect;          ///< Variance divided by the variance under independent individuals.
  double effective_sample_size; ///< Number of independent individuals giving the same mean variance.
};

/// @brief Computes variance inflation with unequal cluster sizes and an exchangeable correlation.
///
/// Each individual's observation has the same marginal variance, and
/// each pair within a cluster has correlation rho. The design effect
/// is one plus rho times the average number of other individuals in a
/// randomly chosen individual's cluster.
/// @param cluster_sizes Nonempty planned sizes of independent clusters, each positive.
/// @param rho Intracluster correlation in [0,1].
/// @return Design effect and total group size divided by the effect.
/// @throws std::invalid_argument Invalid cluster sizes or correlation.
/// @par Example
/// @code
/// const auto arm = dm::stats::cluster_design_effect({5, 15}, .1);
/// std::print("{:.2f}\n", arm.design_effect); // 2.15
/// @endcode
/// @par References
/// - S. M. Eldridge, D. Ashby and S. Kerry (2006), https://doi.org/10.1093/ije/dyl129
/// @see cluster_randomized_mean_power(), cluster_randomized_proportion_power()
ClusterDesignEffect cluster_design_effect(const std::vector<std::size_t>& cluster_sizes, double rho);

/// @brief Prospective rejection probability and arm-specific effective sizes for a cluster design.
/// @par Example
/// @code
/// const auto fit = dm::stats::cluster_randomized_mean_power({10, 10}, {10, 10}, .1, 0);
/// std::print("{:.2f}\n", fit.power); // 0.05
/// @endcode
/// @see cluster_randomized_mean_power(), cluster_randomized_proportion_power()
struct ClusterRandomizedPower {
  double power;               ///< Normal-reference rejection probability under the alternative.
  ClusterDesignEffect first;  ///< Variance inflation and effective size for the first arm.
  ClusterDesignEffect second; ///< Variance inflation and effective size for the second arm.
};

/// @brief Computes cluster-randomized power for a difference of individual-weighted means.
///
/// Uses a known-variance normal reference with a common individual
/// marginal SD. Each arm's unequal cluster sizes determine its own
/// variance inflation. Two-sided power includes both rejection tails.
/// @param first Cluster sizes in the first arm, each positive.
/// @param second Cluster sizes in the independent second arm, each positive.
/// @param rho Common intracluster correlation in [0,1].
/// @param standardized_effect Finite signed first-minus-second population contrast relative to the null, divided by the marginal SD.
/// @param alpha Nominal significance level in (0,1), default .05.
/// @param alternative TwoSided (default), Greater, or Less for the signed contrast.
/// @return Normal-reference power and each arm's design effect and effective size.
/// @throws std::invalid_argument Invalid cluster sizes, correlation, effect, level, or direction.
/// @par Example
/// @code
/// const auto fit = dm::stats::cluster_randomized_mean_power({5, 15}, {10, 10}, .1, .5);
/// std::print("{:.2f} {:.2f}\n", fit.first.design_effect, fit.second.design_effect); // 2.15 1.90
/// @endcode
/// @see cluster_design_effect(), cluster_randomized_proportion_power()
ClusterRandomizedPower cluster_randomized_mean_power(
    const std::vector<std::size_t>& first, const std::vector<std::size_t>& second,
    double rho, double standardized_effect, double alpha = .05, Alternative alternative = Alternative::TwoSided);

/// @brief Computes cluster-randomized power for two individual-weighted proportions.
///
/// Applies each arm's design effect to the respective Bernoulli variance
/// under both the null and the alternative. Uses the fixed-risk normal
/// rejection rule in two_proportion_power().
/// @param first Cluster sizes in the first arm, each positive.
/// @param second Cluster sizes in the independent second arm, each positive.
/// @param rho Common intracluster correlation in [0,1] under both scenarios.
/// @param null_risks First and second Bernoulli risks under the null, with positive variance.
/// @param alternative_risks First and second Bernoulli risks under the alternative.
/// @param alpha Nominal significance level in (0,1), default .05.
/// @param alternative TwoSided (default), Greater or Less for first risk minus second risk.
/// @return Approximate power and each arm's design effect and effective size.
/// @throws std::invalid_argument Invalid cluster sizes, correlation, risks, level, or direction.
/// @par Example
/// @code
/// const auto fit = dm::stats::cluster_randomized_proportion_power({10, 10}, {10, 10}, .1, {.4, .4}, {.4, .4});
/// std::print("{:.2f}\n", fit.power); // 0.05
/// @endcode
/// @see cluster_design_effect(), two_proportion_power()
ClusterRandomizedPower cluster_randomized_proportion_power(
    const std::vector<std::size_t>& first, const std::vector<std::size_t>& second, double rho,
    std::pair<double, double> null_risks, std::pair<double, double> alternative_risks,
    double alpha = .05, Alternative alternative = Alternative::TwoSided);

} // namespace datamunge::stats
