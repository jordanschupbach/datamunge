#pragma once
/// @file restricted_randomization.hpp
/// @brief Design-based randomization inference for matched pairs and randomized blocks.

#include <cstddef>
#include <utility>
#include <vector>

#include <datamunge/random/random.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>

namespace datamunge::stats {

/// @brief Fisher randomization tests and confidence intervals for a constant additive effect.
///
/// Within each block the observed number treated is fixed and all assignments
/// of that size have equal probability. Blocks are randomized independently.
/// The statistic weights each block's treated-minus-control mean by block size.
/// Construct once and reuse the reference for any null effect or confidence level.
/// @par Example
/// @code
/// const dm::stats::RestrictedRandomization fit({3, 1, 6, 2},
///     {true, false, true, false}, {1, 1, 2, 2});
/// std::print("{}\n", fit.test(3).p_value); // 1
/// @endcode
/// @par References
/// - P. R. Rosenbaum (2001), https://doi.org/10.1093/biomet/88.1.219
/// @see test(), confidence_interval()
class RestrictedRandomization {
public:
  /// @brief Enumerates all within-block assignments, including the observed one.
  /// @param outcomes Nonempty finite outcomes in row order.
  /// @param treated Observed treatment indicators, aligned with outcomes.
  /// @param blocks Block labels aligned with outcomes. Each block has both arms.
  /// @param max_assignments Positive enumeration limit, default 1000000.
  /// @throws std::invalid_argument Invalid input lengths, outcome, block or limit.
  /// @throws std::length_error The number of assignments exceeds the limit.
  /// @throws std::overflow_error The observed effect is outside the finite double range.
  /// @par Example
  /// @code
  /// const dm::stats::RestrictedRandomization fit({4, 1}, {true, false}, {5, 5});
  /// std::print("{}\n", fit.test(0).parameter1); // 2
  /// @endcode
  /// @see test()
  RestrictedRandomization(const std::vector<double>& outcomes, const std::vector<bool>& treated,
      const std::vector<std::size_t>& blocks, std::size_t max_assignments = 1'000'000);

  /// @brief Samples independent uniform assignments with the observed within-block counts.
  /// @param outcomes Nonempty finite outcomes in row order.
  /// @param treated Observed treatment indicators, aligned with outcomes.
  /// @param blocks Block labels aligned with outcomes. Each block has both arms.
  /// @param draws Positive number of simulated assignments.
  /// @param rng Caller-owned generator, advanced by the draws.
  /// @throws std::invalid_argument Invalid input lengths, outcome, block or draw count.
  /// @throws std::overflow_error The observed effect is outside the finite double range.
  /// @par Example
  /// @code
  /// dm::random::SplitMix64 rng(42);
  /// const dm::stats::RestrictedRandomization fit({4, 1}, {true, false}, {5, 5}, 19, rng);
  /// std::print("{}\n", fit.test(0).p_value); // 1
  /// @endcode
  /// @see test()
  RestrictedRandomization(const std::vector<double>& outcomes, const std::vector<bool>& treated,
      const std::vector<std::size_t>& blocks, std::size_t draws, random::RandomGenerator& rng);

  /// @brief Tests a constant sharp null using the absolute adjusted mean contrast.
  /// @param null_effect Finite hypothesized additive effect, default zero.
  /// @return Adjusted contrast, observed effect in estimate1, number of evaluated
  /// assignments in parameter1, extreme assignment count in parameter2, and p-value.
  /// The simulated p-value uses the plus-one correction.
  /// @throws std::invalid_argument Nonfinite null effect.
  /// @par Example
  /// @code
  /// const dm::stats::RestrictedRandomization fit({4, 1, 3, 2},
  ///     {true, false, true, false}, {0, 0, 1, 1});
  /// std::print("{}\n", fit.test(0).p_value); // 0.5
  /// @endcode
  /// @see confidence_interval()
  HypothesisTestResult test(double null_effect = 0) const;

  /// @brief Inverts the two-sided tests over all possible constant effects.
  /// @param confidence_level Simultaneous test inversion level in (0,1), default .95.
  /// @return Closed interval of effects with p-value above 1-confidence_level.
  /// Endpoints may be infinite for small designs or strict confidence levels.
  /// @throws std::invalid_argument Invalid confidence level.
  /// @par Example
  /// @code
  /// const dm::stats::RestrictedRandomization fit({4, 1, 3, 2},
  ///     {true, false, true, false}, {0, 0, 1, 1});
  /// const auto interval = fit.confidence_interval(.5);
  /// std::print("{} {}\n", interval.first, interval.second); // 1 3
  /// @endcode
  /// @see test()
  std::pair<double, double> confidence_interval(double confidence_level = .95) const;

private:
  /// @brief Builds the selected reference with an optional caller-owned simulation engine.
  /// @param outcomes Observed responses.
  /// @param treated Observed assignment.
  /// @param blocks Block labels.
  /// @param limit Exact enumeration cap or simulated draw count.
  /// @param rng Simulation generator, or null for enumeration.
  RestrictedRandomization(const std::vector<double>& outcomes, const std::vector<bool>& treated,
      const std::vector<std::size_t>& blocks, std::size_t limit, random::RandomGenerator* rng);

  long double estimate_{0};     ///< Size-weighted observed contrast before rounding.
  bool simulated_{false};       ///< Whether the reference uses the simulation correction.
  std::vector<double> starts_;  ///< Sorted inclusive lower boundaries of extreme assignments.
  std::vector<double> ends_;    ///< Sorted inclusive upper boundaries of extreme assignments.
};

} // namespace datamunge::stats
