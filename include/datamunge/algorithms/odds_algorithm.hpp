#pragma once
/// @file odds_algorithm.hpp
/// @brief Checked independent-Bernoulli last-success stopping with actual suffix histories.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned threshold and computed win probability for the independent last-success problem.
/// @par Example
/// @code
/// auto r=dm::algorithms::odds_algorithm({.5,.5});std::print("{} {}\n",r.threshold,r.win_probability); // 2 0.5
/// @endcode
/// @see odds_algorithm_trace()
struct OddsResult {
  std::size_t threshold{0}; ///< One-based first eligible event;0 only for empty input.
  double      win_probability{
      0}; ///< Computed chance of stopping on the last success, not a realized outcome.
};
/// @brief Actual same-producer result, selected suffix state and work.
/// @par Example
/// @code
/// auto r=dm::algorithms::odds_algorithm_run({1.});std::print("{} {}\n",r.result.win_probability,r.certain); // 1 true
/// @endcode
/// @see odds_algorithm_trace()
struct OddsRun {
  OddsResult  result;    ///< Actual threshold and rounded final exactly-one probability.
  std::size_t visits{0}; ///< Actual backwards suffix events inspected after complete validation.
  long double odds_sum{
      0}; ///< Actual finite suffix-odds sum, excluding undefined certain-event odds.
  long double no_success{1};  ///< Actual selected-suffix failure product.
  long double exactly_one{0}; ///< Actual selected-suffix exactly-one recurrence state.
  bool        certain{
      false}; ///< Actual stopping event had p=1, whose odds are flagged rather than divided.
};
/// @brief Actual one-event prepend to the eligible suffix.
/// @par Example
/// @code
/// auto t=dm::algorithms::odds_algorithm_trace({1.});std::print("{}\n",t.steps[0].reached); // true
/// @endcode
/// @see odds_algorithm_trace()
struct OddsStep {
  std::size_t event{0};       ///< Actual one-based original event index.
  double      probability{0}; ///< Actual caller probability.
  long double failure{0};     ///< Actual wide1-p.
  long double odds{0};        ///< Actual p/(1-p),0 placeholder for certain event.
  long double sum_before{0};  ///< Actual finite odds sum before this visit.
  long double sum_after{
      0}; ///< Actual finite sum after visit, unchanged for undefined certain odds.
  long double none_before{1}; ///< Actual empty/no-success suffix probability before prepend.
  long double none_after{1};  ///< Actual no-success suffix probability after prepend.
  long double one_before{0};  ///< Actual exactly-one suffix probability before prepend.
  long double one_after{0};   ///< Actual exactly-one probability after prepend.
  bool        certain{false}; ///< True iff supplied p is exactly1.
  bool        reached{false}; ///< Actual odds sum>=1 or certain-event termination decision.
};
/// @brief Owned original probabilities and actual backwards decision records.
/// Admission counts input n, final result scalars2 and each fixed record1.
/// Fixed diagnostic fields are included in their logical record, not separately.
/// @par Example
/// @code
/// auto t=dm::algorithms::odds_algorithm_trace({.5,.5});std::print("{}\n",t.captured_entries); // 5
/// @endcode
/// @see odds_algorithm_trace()
struct OddsTrace {
  std::vector<double>   probabilities; ///< Owned complete original vector, even unvisited prefix.
  OddsRun               run;           ///< Actual final result and selected suffix counters/state.
  std::vector<OddsStep> steps;         ///< Actual visited suffix records in reverse event order.
  std::size_t           captured_entries{
      0}; ///< Actual logical input n + result2 + visited records, not bytes.
};
/// @brief Computed fixed-threshold suffix probability, produced by a separate complete curve routine.
/// @par Example
/// @code
/// auto c=dm::algorithms::odds_threshold_curve({1.});std::print("{}\n",c[0].win_probability); // 1
/// @endcode
/// @see odds_algorithm_trace()
struct OddsThresholdPoint {
  std::size_t threshold{0};       ///< One-based first eligible event; n+1 denotes never accepting.
  double      win_probability{0}; ///< Computed probability of exactly one eligible suffix success.
  double no_success_probability{1}; ///< Computed probability of zero eligible suffix successes.
};
/// @brief Solve the independent last-success optimal stopping problem.
/// Complete finite probabilities in[0,1] are validated before suffix work,
/// including an unvisited prefix. Events must be mutually independent and all
/// probabilities known in advance; no dependence/unknown-probability guarantee.
/// Traverse from the last event backwards until accumulated odds reach1, or
/// until p=1. Certain odds are flagged rather than divided by zero. The result
/// accepts the first success at/after its one-based threshold; no success loses.
/// With Q=no-success and W=exactly-one for the already visited suffix, prepend p
/// through W=p*Q+(1-p)*W, Q=(1-p)*Q. This recurrence avoids a singular odds-times-failure probability formula.
/// Empty input returns historical{0,0}; all-zero nonempty input returns{1,0}.
/// Exact-arithmetic odds-theorem optimality does not certify rounded comparisons
/// or probability error. No inputs are mutated, referenced or globally cached.
/// O(n) validation plus at most n suffix visits; constant ordinary working state.
/// @param p Complete independent-event success probabilities, finite in[0,1].
/// @return Owned actual OddsResult for the original event order.
/// @throws std::invalid_argument If any complete probability is invalid.
/// @throws std::length_error If complete count geometry or selected history quota fails.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::odds_algorithm({1.});std::print("{} {}\n",r.threshold,r.win_probability); // 1 1
/// @endcode
/// @par References
/// - F. T. Bruss(2000), Sum the odds to one and stop, Annals of Probability28(3),1384–1391, doi:10.1214/aop/1019160340.
/// @see odds_threshold_curve(), OddsStep
OddsResult odds_algorithm(const std::vector<double>& p);
/// @brief Solve the independent last-success optimal stopping problem.
/// Complete finite probabilities in[0,1] are validated before suffix work,
/// including an unvisited prefix. Events must be mutually independent and all
/// probabilities known in advance; no dependence/unknown-probability guarantee.
/// Traverse from the last event backwards until accumulated odds reach1, or
/// until p=1. Certain odds are flagged rather than divided by zero. The result
/// accepts the first success at/after its one-based threshold; no success loses.
/// With Q=no-success and W=exactly-one for the already visited suffix, prepend p
/// through W=p*Q+(1-p)*W, Q=(1-p)*Q. This recurrence avoids a singular odds-times-failure probability formula.
/// Empty input returns historical{0,0}; all-zero nonempty input returns{1,0}.
/// Exact-arithmetic odds-theorem optimality does not certify rounded comparisons
/// or probability error. No inputs are mutated, referenced or globally cached.
/// O(n) validation plus at most n suffix visits; constant ordinary working state.
/// @param p Complete independent-event success probabilities, finite in[0,1].
/// @return Owned actual OddsRun for the original event order.
/// @throws std::invalid_argument If any complete probability is invalid.
/// @throws std::length_error If complete count geometry or selected history quota fails.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::odds_algorithm_run({.5,.5});std::print("{}\n",r.visits); // 1
/// @endcode
/// @par References
/// - F. T. Bruss(2000), Sum the odds to one and stop, Annals of Probability28(3),1384–1391, doi:10.1214/aop/1019160340.
/// @see odds_threshold_curve(), OddsStep
OddsRun odds_algorithm_run(const std::vector<double>& p);
/// @brief Capture actual suffix decisions and owned inputs.
/// Complete finite probabilities in[0,1] are validated before suffix work,
/// including an unvisited prefix. Events must be mutually independent and all
/// probabilities known in advance; no dependence/unknown-probability guarantee.
/// Traverse from the last event backwards until accumulated odds reach1, or
/// until p=1. Certain odds are flagged rather than divided by zero. The result
/// accepts the first success at/after its one-based threshold; no success loses.
/// With Q=no-success and W=exactly-one for the already visited suffix, prepend p
/// through W=p*Q+(1-p)*W, Q=(1-p)*Q. This recurrence avoids a singular odds-times-failure probability formula.
/// Empty input returns historical{0,0}; all-zero nonempty input returns{1,0}.
/// Exact-arithmetic odds-theorem optimality does not certify rounded comparisons
/// or probability error. No inputs are mutated, referenced or globally cached.
/// O(n) validation plus at most n suffix visits; constant ordinary working state.
/// Logical units n+2+visits are admitted before input copy and each record.
/// The default1000000 quota counts records/elements, not memory bytes;0 disables.
/// Owned trace storage is O(n+visits); failure returns no partial history.
/// @param p Complete independent-event success probabilities, finite in[0,1].
/// @param max_entries Actual logical history cap, default1000000;0 disables.
/// @return Owned actual OddsTrace for the original event order.
/// @throws std::invalid_argument If any complete probability is invalid.
/// @throws std::length_error If complete count geometry or selected history quota fails.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::odds_algorithm_trace({1.});std::print("{}\n",r.steps[0].certain); // true
/// @endcode
/// @par References
/// - F. T. Bruss(2000), Sum the odds to one and stop, Annals of Probability28(3),1384–1391, doi:10.1214/aop/1019160340.
/// @see odds_threshold_curve(), OddsStep
OddsTrace odds_algorithm_trace(const std::vector<double>& p, std::size_t max_entries = 1000000);
/// @brief Compute every fixed-threshold probability through one complete suffix recurrence.
/// Same complete probability validation and independence model as odds_algorithm().
/// Computes all suffixes independently of where that stopping producer terminates;
/// these rows are not claimed to be the producer's actual visited history.
/// Includes n+1 never-accept row, probability0. Empty input has only that row1.
/// O(n) work/output, wide Q/W working states; no caller mutation/retained references.
/// @param p Complete finite success probabilities in[0,1].
/// @param max_points Complete n+1 row cap, default1000000, zero disables.
/// @return Owned ascending threshold rows1 through n+1.
/// @throws std::invalid_argument If any complete probability is invalid.
/// @throws std::length_error If complete output geometry or selected quota is exceeded.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::odds_threshold_curve({1.,0.})[0].win_probability); // 1
/// @endcode
/// @see odds_algorithm_trace()
std::vector<OddsThresholdPoint> odds_threshold_curve(
    const std::vector<double>& p,
    std::size_t                max_points = 1000000);
/// @brief Return the original ten-event report fixture, in its original order.
/// @return Fresh owned probabilities[.1,.15,.05,.3,.2,.4,.25,.35,.12,.28].
/// @throws std::bad_alloc If vector allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::odds_algorithm_demo_probabilities().size()); // 10
/// @endcode
/// @see odds_algorithm()
std::vector<double> odds_algorithm_demo_probabilities();
/// @brief Generate record-event probabilities1/i for a random-permutation secretary model.
/// Under the uniform random-permutation model the record indicators are independent;
/// arbitrary observed ranks or other dependent processes are not covered.
/// Complete row count is checked before allocation. O(n) work and output storage.
/// @param n Original event count, zero accepted.
/// @param max_values Complete output-element cap, default1000000, zero disables.
/// @return Fresh probabilities1,1/2,...,1/n, empty when n=0.
/// @throws std::length_error If complete geometry or selected quota is exceeded.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::odds_secretary_probabilities(2)); // [1, 0.5]
/// @endcode
/// @see odds_algorithm()
std::vector<double> odds_secretary_probabilities(std::size_t n, std::size_t max_values = 1000000);
} // namespace datamunge::algorithms
