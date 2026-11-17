#pragma once
/// @file lll.hpp
/// @brief Bounded floating Gram-Schmidt LLL with checked exact64-bit row operations, unimodular transform and actual operation evidence.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual bounded LLL outer-loop stop, separate from floating property observations.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lll_analyze({{1,0},{0,1}});
/// std::print("{}\n",a.stop==dm::algorithms::LLLStop::Completed); // true
/// @endcode
/// @see LLLAnalysis
enum class LLLStop {
  Completed, ///< Actual outer index reached row count.
  StepLimit ///< Explicit actual outer-step budget exhausted; partial equivalent basis, not claimed reduced.
};
/// @brief Actual producing reduction/check/swap event kind.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lll_trace({{1,0},{3,1}});
/// std::print("{}\n",t.events[0].kind==dm::algorithms::LLLEventKind::SizeReduce); // true
/// @endcode
/// @see LLLEvent
enum class LLLEventKind {
  SizeReduce, ///< Actual integer row subtraction with checked multiplier.
  Lovasz,     ///< Actual computed adjacent floating norm inequality.
  Swap        ///< Actual adjacent basis and transform row exchange.
};
/// @brief Actual selected floating Gram-Schmidt, integer row-operation and outer-loop work.
/// Excludes complete guards, copies/allocation/capture, property-summary
/// comparisons, elapsed time and CPU instruction counts.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lll_analyze({{1,0},{3,1}});
/// std::print("{}\n",a.work.size_reductions); // 1
/// @endcode
/// @par References
/// - A. K. Lenstra, H. W. Lenstra Jr. and L. Lovasz(1982), Factoring Polynomials with Rational Coefficients, Mathematische Annalen261,515–534, https://doi.org/10.1007/BF01457454.
/// @see lll_analyze(), lll_trace()
struct LLLWork {
  std::size_t outer_steps{0}; ///< Actual complete size-reduction-plus-Lovasz outer iterations.
  std::size_t gram_schmidt_runs{
      0}; ///< Actual full Gram-Schmidt evaluations, initial and after each real row operation.
  std::size_t dot_terms{
      0}; ///< Actual scalar coordinate products accumulated in Gram-Schmidt dot products.
  std::size_t orthogonal_updates{0}; ///< Actual scalar Gram-Schmidt projection subtractions.
  std::size_t size_reductions{0};    ///< Actual nonzero rounded integer row reductions.
  std::size_t basis_updates{
      0}; ///< Actual checked original-coordinate row writes from size reduction.
  std::size_t transform_updates{
      0}; ///< Actual checked integer transform-coordinate writes from size reduction.
  std::size_t lovasz_tests{0}; ///< Actual producing adjacent Lovasz comparisons.
  std::size_t swaps{0};        ///< Actual adjacent basis/transform row swaps.
  std::size_t row_norm_terms{
      0}; ///< Additional initial/final row-norm observation coordinate terms.
};
/// @brief Owned actual equivalent integer basis, exact row transform and floating reduction observations.
/// Checked row subtraction/swap preserves the exact original integer lattice,
/// including incomplete runs. Decisions use double Gram-Schmidt, so completed
/// status and observed inequalities are not a rigorous exact-arithmetic LLL
/// or shortest-vector certificate. Transform overflow can throw even if the
/// basis row would fit; no arbitrary-precision fallback is implied.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lll_analyze({{1,0},{3,1}});
/// std::print("{} {}\n",a.basis[1][0],a.transform[1][0]); // 0 -3
/// @endcode
/// @par References
/// - A. K. Lenstra, H. W. Lenstra Jr. and L. Lovasz(1982), Factoring Polynomials with Rational Coefficients, Mathematische Annalen261,515–534, https://doi.org/10.1007/BF01457454.
/// @see lll_analyze(), lll_trace()
struct LLLAnalysis {
  std::vector<std::vector<long long>>
      basis; ///< Actual finite64-bit integer row basis after complete or budget-limited operations.
  std::vector<std::vector<long long>>
      transform; ///< Actual exact integral unimodular row transform U, satisfying basis=U*original mathematically.
  std::vector<std::vector<double>>
      mu; ///< Actual final computed Gram-Schmidt projection coefficients.
  std::vector<double>
      orthogonal_squared_norms; ///< Actual final positive finite computed orthogonal squared norms.
  std::vector<long double>
      initial_squared_norms; ///< Actual widened original-row squared norm observations.
  std::vector<long double>
          final_squared_norms;      ///< Actual widened returned-row squared norm observations.
  LLLStop stop{LLLStop::Completed}; ///< Actual complete or explicit outer-step budget status.
  bool    observed_size_reduced{
      true}; ///< Whether actual final computed abs(mu[i][j])<=0.5 for every j<i.
  bool    observed_lovasz{true}; ///< Whether actual final computed adjacent norm inequalities pass.
  LLLWork work;                  ///< Actual selected producing and separate row-norm work.
};
/// @brief Owned actual basis/transform and computed Gram-Schmidt arrays at one real committed row state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lll_trace({{1,0},{3,1}});
/// std::print("{}\n",t.states.front().basis[1][0]); // 3
/// @endcode
/// @par References
/// - A. K. Lenstra, H. W. Lenstra Jr. and L. Lovasz(1982), Factoring Polynomials with Rational Coefficients, Mathematische Annalen261,515–534, https://doi.org/10.1007/BF01457454.
/// @see lll_analyze(), lll_trace()
struct LLLState {
  std::vector<std::vector<long long>> basis; ///< Actual original/integer-updated/swapped row basis.
  std::vector<std::vector<long long>>
      transform; ///< Actual corresponding checked unimodular row transform.
  std::vector<std::vector<double>>
      orthogonal; ///< Actual full computed Gram-Schmidt orthogonal row coordinates.
  std::vector<std::vector<double>> mu;            ///< Actual computed projection coefficients.
  std::vector<double>              squared_norms; ///< Actual computed orthogonal squared norms.
};
/// @brief Actual producing row reduction, Lovasz decision or adjacent swap linked to a real state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lll_trace({{1,0},{3,1}});
/// std::print("{}\n",t.events[0].multiplier); // 3
/// @endcode
/// @par References
/// - A. K. Lenstra, H. W. Lenstra Jr. and L. Lovasz(1982), Factoring Polynomials with Rational Coefficients, Mathematische Annalen261,515–534, https://doi.org/10.1007/BF01457454.
/// @see lll_analyze(), lll_trace()
struct LLLEvent {
  LLLEventKind kind{LLLEventKind::Lovasz}; ///< Actual operation/check kind.
  std::size_t  k{0};                       ///< Actual current target basis row.
  std::size_t  j{0};                       ///< Actual reduction source or adjacent predecessor row.
  long long    multiplier{
      0}; ///< Actual rounded integer reduction q;0 placeholder on non-reduction events.
  double projection{
      0}; ///< Actual pre-operation mu[k][j] used for reduction/check;0 placeholder on swap.
  double      left{0};       ///< Actual computed Lovasz left norm,0 outside that check.
  double      right{0};      ///< Actual computed Lovasz right expression,0 outside that check.
  bool        passed{false}; ///< Actual Lovasz comparison outcome; false placeholder otherwise.
  std::size_t state{
      0}; ///< Actual captured state index after row operations or unchanged for a check.
};
/// @brief Owned complete original basis and every actual bounded row-operation/check/state record.
/// Each actual state counts1+2*n*d+2*n*n+n logical entries; original input
/// counts n*d and each actual event1. Records admitted before snapshot/event
/// copy; unknown future operations are not fabricated. Default million cap,
///0 optional cap off. Excludes scalar fields/row headers/capacities/result/runtime
/// copies and bytes/RAM. Complete bounded trace or exception.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lll_trace({{1,0},{3,1}});
/// std::print("{}\n",t.analysis.observed_size_reduced); // true
/// @endcode
/// @par References
/// - A. K. Lenstra, H. W. Lenstra Jr. and L. Lovasz(1982), Factoring Polynomials with Rational Coefficients, Mathematische Annalen261,515–534, https://doi.org/10.1007/BF01457454.
/// @see lll_analyze(), lll_trace()
struct LLLTrace {
  std::vector<std::vector<long long>> original;   ///< Owned complete original integer rows.
  double                              delta{.75}; ///< Actual finite reduction parameter in(0.25,1).
  std::vector<LLLState>
      states; ///< Actual original GS state then each committed reduction/swap GS state.
  std::vector<LLLEvent>
              events;   ///< Actual chronological reductions/Lovasz tests/swaps, no prefix reruns.
  LLLAnalysis analysis; ///< Actual final or explicit partial result/transform/observations/work.
  std::size_t captured_entries{
      0}; ///< Actual original-coordinate/event/state/basis/transform/orthogonal/mu/norm logical records.
};
/// @brief Return the completed historical floating-decision LLL basis with checked exact row operations.
/// Full guards precede Gram-Schmidt and checked integer arithmetic. Ordinary
/// finite double decisions/round-half-away multipliers preserve the historical
/// defined-range path. Default100000 actual outer steps bounds possible numerical
/// cycling; exhaustion throws rather than returning a silently reduced claim.
/// No shortest basis, per-row norm decrease or exact floating reduction theorem.
/// @param basis Owned by-value original integer rows; argument copy can precede guards.
/// @param delta Finite parameter in(0.25,1), default0.75.
/// @return Owned completed integer basis, same exact original lattice on success.
/// @throws std::invalid_argument On complete ragged/zero-dimensional/too-many-row or delta violations.
/// @throws std::length_error On complete matrix/work geometry.
/// @throws std::domain_error On zero computed orthogonal norm; a floating breakdown, not an exact rank certificate.
/// @throws std::overflow_error On unrepresentable rounded multiplier, exact64-bit basis/transform update or floating arithmetic.
/// @throws std::bad_alloc On runtime/output allocations.
/// @throws std::length_error On default outer-step budget exhaustion.
/// @par Example
/// @code
/// const auto r=dm::algorithms::lll_reduce({{1,0},{3,1}});
/// std::print("{}\n",r[1][0]); // 0
/// @endcode
/// @par References
/// - A. K. Lenstra, H. W. Lenstra Jr. and L. Lovasz(1982), Factoring Polynomials with Rational Coefficients, Mathematische Annalen261,515–534, https://doi.org/10.1007/BF01457454.
/// @see LLLAnalysis, LLLTrace
std::vector<std::vector<long long>> lll_reduce(
    std::vector<std::vector<long long>> basis,
    double                              delta = .75);
/// @brief Return actual bounded LLL basis, exact row transform and computed property observations.
/// No mutation, shared state or optional history. Completed loop status is
/// distinct from exact-arithmetic certification; inspect observed property flags.
/// @param basis Borrowed complete rectangular integer row basis; n<=dimension and positive dimension when nonempty.
/// @param delta Finite Lovasz parameter strictly between0.25 and1, default0.75.
/// @param max_steps Actual outer-iteration budget, default100000;0 returns original equivalent basis with explicit StepLimit when n>=2.
/// @return Owned actual complete/partial equivalent basis with explicit stop and selected work.
/// @throws std::invalid_argument On complete ragged/zero-dimensional/too-many-row or delta violations.
/// @throws std::length_error On complete matrix/work geometry.
/// @throws std::domain_error On zero computed orthogonal norm; a floating breakdown, not an exact rank certificate.
/// @throws std::overflow_error On unrepresentable rounded multiplier, exact64-bit basis/transform update or floating arithmetic.
/// @throws std::bad_alloc On runtime/output allocations.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lll_analyze({{1,0},{3,1}},.75,0);
/// std::print("{}\n",a.stop==dm::algorithms::LLLStop::StepLimit); // true
/// @endcode
/// @par References
/// - A. K. Lenstra, H. W. Lenstra Jr. and L. Lovasz(1982), Factoring Polynomials with Rational Coefficients, Mathematische Annalen261,515–534, https://doi.org/10.1007/BF01457454.
/// @see LLLAnalysis, LLLTrace
LLLAnalysis lll_analyze(
    const std::vector<std::vector<long long>>& basis,
    double                                     delta     = .75,
    std::size_t                                max_steps = 100000);
/// @brief Capture every actual bounded integer row operation, computed check and real GS state once.
/// Same shared producer as ordinary/count-only methods. Exact transform and
/// finite-size checks precede committing a size-reduction row; captured states
/// are actual GS evaluations, never repeated original-prefix reductions.
/// @param basis Borrowed complete rectangular integer row basis; n<=dimension and positive dimension when nonempty.
/// @param delta Finite Lovasz parameter strictly between0.25 and1, default0.75.
/// @param max_steps Actual outer-iteration budget, default100000;0 returns original equivalent basis with explicit StepLimit when n>=2.
/// @param max_entries Actual logical input/event/state cap, default1000000;0 optional cap off.
/// @return Owned complete bounded actual evidence and final/partial equivalent result.
/// @throws std::invalid_argument On complete ragged/zero-dimensional/too-many-row or delta violations.
/// @throws std::length_error On complete matrix/work geometry.
/// @throws std::domain_error On zero computed orthogonal norm; a floating breakdown, not an exact rank certificate.
/// @throws std::overflow_error On unrepresentable rounded multiplier, exact64-bit basis/transform update or floating arithmetic.
/// @throws std::bad_alloc On runtime/output allocations.
/// @throws std::length_error On actual capture count/quota.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lll_trace({{1,0},{3,1}});
/// std::print("{}\n",t.events[0].multiplier); // 3
/// @endcode
/// @par References
/// - A. K. Lenstra, H. W. Lenstra Jr. and L. Lovasz(1982), Factoring Polynomials with Rational Coefficients, Mathematische Annalen261,515–534, https://doi.org/10.1007/BF01457454.
/// @see LLLAnalysis, LLLTrace
LLLTrace lll_trace(
    const std::vector<std::vector<long long>>& basis,
    double                                     delta       = .75,
    std::size_t                                max_steps   = 100000,
    std::size_t                                max_entries = 1000000);
} // namespace datamunge::algorithms
