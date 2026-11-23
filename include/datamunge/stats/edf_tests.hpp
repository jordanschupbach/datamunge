#pragma once
/// @file edf_tests.hpp
/// @brief One- and multiple-sample empirical distribution function tests.

#include <datamunge/random/random.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>

#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief Tests goodness of fit to a fully specified continuous distribution using Anderson–Darling.
///
/// Large statistics indicate lack of fit, with extra weight on the tails.
/// Simulates the finite-sample reference from independent uniform probabilities.
/// The CDF must be fixed independently of the sample. An observed probability
/// of zero or one produces an infinite statistic. Ties are accepted, and the
/// reference assumes continuous observations.
/// @param sample Nonempty finite independent observations.
/// @param cdf Continuous null CDF, nondecreasing at the sorted observations and in [0, 1].
/// @param draws Positive number of independent reference samples.
/// @param rng Caller-owned generator, advanced by the simulation.
/// @return Anderson–Darling statistic, draws in parameter1, upper-tail count in parameter2, and plus-one p-value.
/// @throws std::invalid_argument Invalid sample, CDF, or draw count. Exceptions from the CDF propagate.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(7);
/// const auto fit = dm::stats::anderson_darling_test({.5}, [](double x) { return x; }, 99, rng);
/// std::print("{:.3f}\n", fit.statistic); // 0.386
/// @endcode
/// @par References
/// - T. W. Anderson and D. A. Darling (1954), https://doi.org/10.1080/01621459.1954.10501232
/// @see anderson_darling_normal_test()
HypothesisTestResult anderson_darling_test(const std::vector<double>& sample,
                                           const std::function<double(double)>& cdf,
                                           std::size_t draws, random::RandomGenerator& rng);

/// @brief Tests normality using Anderson–Darling after fitting location and scale.
///
/// Fits the mean and the sample standard deviation (denominator n-1).
/// The statistic is invariant to normal location and scale. The finite-sample
/// reference simulates standard normal observations and refits both parameters
/// for every draw. Ties are accepted under the continuous reference.
/// @param sample At least three finite independent observations, with variation.
/// @param draws Positive number of independent reference samples.
/// @param rng Caller-owned generator, advanced by the simulation.
/// @return Anderson–Darling statistic, draws in parameter1, upper-tail count in parameter2,
/// plus-one p-value, fitted mean in estimate1 and fitted sample SD in estimate2.
/// An SD larger than the finite double range is reported as positive infinity.
/// @throws std::invalid_argument Too few observations, nonfinite or constant sample, or zero draws.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(7);
/// const auto fit = dm::stats::anderson_darling_normal_test({-2, -1, 0, 1, 2}, 99, rng);
/// std::print("{:.3f}\n", fit.estimate2); // 1.581
/// @endcode
/// @par References
/// - M. A. Stephens (1974), https://doi.org/10.1080/01621459.1974.10480196
/// @see anderson_darling_test()
HypothesisTestResult anderson_darling_normal_test(const std::vector<double>& sample,
                                                  std::size_t draws, random::RandomGenerator& rng);

/// @brief Compares several distributions with the midrank k-sample Anderson–Darling statistic.
///
/// Uses the unstandardized Scholz–Stephens statistic with half jumps at
/// tied values. The reference uniformly permutes pooled labels at fixed
/// group sizes and includes equal statistics in its upper tail. If all
/// observations coincide, the statistic is zero and the p-value is one.
/// @param groups At least two nonempty groups of finite independent observations.
/// @param draws Positive number of independently sampled assignments.
/// @param rng Caller-owned generator, advanced by the permutations.
/// @return Statistic, draw count in parameter1, upper-tail count in parameter2, and plus-one p-value.
/// @throws std::invalid_argument Invalid groups or draw count.
/// @throws std::length_error Pooled sample exceeds container capacity.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(7);
/// const auto fit = dm::stats::anderson_darling_k_sample_test({{1, 1}, {1}, {1, 1}}, 99, rng);
/// std::print("{}\n", fit.p_value); // 1
/// @endcode
/// @par References
/// - F. W. Scholz and M. A. Stephens (1987), equation 7, https://doi.org/10.1080/01621459.1987.10478517
/// @see anderson_darling_k_sample_test_exact(), cramer_von_mises_test_two_sample()
HypothesisTestResult anderson_darling_k_sample_test(const std::vector<std::vector<double>>& groups,
                                                    std::size_t draws, random::RandomGenerator& rng);

/// @brief Enumerates the exact fixed-size label reference for midrank k-sample Anderson–Darling.
///
/// Enumerates every assignment, including the observed one. Duplicate
/// observations remain distinct and the upper tail includes ties.
/// @param groups At least two nonempty groups of finite independent observations.
/// @param max_assignments Positive exact enumeration limit, default 100000.
/// @return Statistic, assignment count in parameter1, upper-tail count in parameter2, and exact p-value.
/// @throws std::invalid_argument Invalid groups or assignment limit.
/// @throws std::length_error Pooled sample or number of assignments exceeds capacity.
/// @par Example
/// @code
/// const auto fit = dm::stats::anderson_darling_k_sample_test_exact({{0, 1}, {2}, {3}});
/// std::print("{}\n", fit.parameter1); // 12
/// @endcode
/// @par References
/// - F. W. Scholz and M. A. Stephens (1987), equation 7, https://doi.org/10.1080/01621459.1987.10478517
/// @see anderson_darling_k_sample_test()
HypothesisTestResult anderson_darling_k_sample_test_exact(const std::vector<std::vector<double>>& groups,
                                                          std::size_t max_assignments = 100'000);

/// @brief Tests a fully specified continuous CDF using the one-sample Cramér–von Mises statistic.
///
/// The finite-sample reference simulates independent uniform probabilities.
/// The CDF must be chosen independently of the data. Ties are accepted under
/// the continuous reference.
/// @param sample Nonempty finite independent observations.
/// @param cdf Continuous null CDF, nondecreasing at the sorted observations and in [0, 1].
/// @param draws Positive number of independent reference samples.
/// @param rng Caller-owned random generator, advanced by the simulation.
/// @return Statistic, draws in parameter1, upper-tail count in parameter2, and plus-one p-value.
/// @throws std::invalid_argument Invalid sample, CDF, or draw count. Exceptions from the CDF propagate.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(7);
/// const auto fit = dm::stats::cramer_von_mises_test({.5}, [](double x) { return x; }, 99, rng);
/// std::print("{:.3f}\n", fit.statistic); // 0.083
/// @endcode
/// @par References
/// - T. W. Anderson and D. A. Darling (1952), https://doi.org/10.1214/aoms/1177729437
/// @see anderson_darling_test(), cramer_von_mises_test_two_sample()
HypothesisTestResult cramer_von_mises_test(const std::vector<double>& sample,
                                           const std::function<double(double)>& cdf,
                                           std::size_t draws, random::RandomGenerator& rng);

/// @brief Compares two empirical CDFs with the Cramér–von Mises statistic and simulated labels.
///
/// Integrates squared differences between the right-continuous sample CDFs
/// over the pooled empirical distribution and multiplies by nm/(n+m).
/// Tied values share a single evaluation point with their full multiplicity.
/// The reference holds the sample sizes fixed and uniformly samples labels.
/// @param first Nonempty first sample of finite observations.
/// @param second Nonempty independent second sample of finite observations.
/// @param draws Positive number of independently sampled assignments.
/// @param rng Caller-owned random generator, advanced by the simulation.
/// @return Statistic, draws in parameter1, upper-tail count in parameter2, and plus-one p-value.
/// @throws std::invalid_argument Invalid samples or draw count.
/// @throws std::length_error Pooled sample exceeds capacity.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(7);
/// const auto fit = dm::stats::cramer_von_mises_test_two_sample({1, 2}, {3, 4}, 99, rng);
/// std::print("{}\n", fit.statistic); // 0.375
/// @endcode
/// @par References
/// - T. W. Anderson (1962), https://doi.org/10.1214/aoms/1177704477
/// @see cramer_von_mises_test_two_sample_exact(), cramer_von_mises_test()
HypothesisTestResult cramer_von_mises_test_two_sample(const std::vector<double>& first,
                                                      const std::vector<double>& second,
                                                      std::size_t draws, random::RandomGenerator& rng);

/// @brief Compares two empirical CDFs with the exact pooled-label Cramér–von Mises reference.
///
/// Enumerates every assignment at the observed group sizes, keeping duplicate
/// observations distinct. Uses the squared right-continuous CDF differences as
/// in cramer_von_mises_test_two_sample(). Large statistics indicate lack of fit.
/// @param first Nonempty first sample of finite observations.
/// @param second Nonempty independent second sample of finite observations.
/// @param max_assignments Positive exact enumeration limit, default 100000.
/// @return Statistic, assignment count in parameter1, upper-tail count in parameter2, and exact p-value.
/// @throws std::invalid_argument Invalid samples or assignment limit.
/// @throws std::length_error The pooled sample or number of assignments exceeds capacity.
/// @par Example
/// @code
/// const auto fit = dm::stats::cramer_von_mises_test_two_sample_exact({1, 2}, {3, 4});
/// std::print("{:.3f}\n", fit.p_value); // 0.333
/// @endcode
/// @par References
/// - T. W. Anderson (1962), https://doi.org/10.1214/aoms/1177704477
/// @see cramer_von_mises_test_two_sample()
HypothesisTestResult cramer_von_mises_test_two_sample_exact(const std::vector<double>& first,
                                                            const std::vector<double>& second,
                                                            std::size_t max_assignments = 100'000);

} // namespace datamunge::stats
