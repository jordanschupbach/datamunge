#pragma once
/// @file levinson.hpp
/// @brief Checked unpivoted symmetric Toeplitz recursion with actual stages, explicit leading-block breakdown and residual diagnostics.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual selected recurrence work, excluding validation, capture, residuals and elapsed time.
/// @par Example
/// @code
/// const auto a=dm::algorithms::levinson_analyze({2,1},{1,1});
/// std::print("{}\n",a.work.dot_products); // 3
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see levinson_analyze(), levinson_trace()
struct LevinsonWork {
  std::size_t completed_orders{
      0}; ///< Actual completed leading principal solves, including initial order1.
  std::size_t dot_products{
      0}; ///< Actual scalar multiply/add terms across forward/backward/solution errors, three per index.
  std::size_t vector_updates{
      0}; ///< Actual new forward/backward/solution entries, three per successful extension coordinate.
};
/// @brief Actual complete or explicitly partial prefix solution and exact-zero breakdown status.
/// @par Example
/// @code
/// const auto a=dm::algorithms::levinson_analyze({1,1,0},{1,2,3});
/// std::print("{} {}\n",a.solved,a.breakdown_order); // false 2
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see levinson_analyze(), levinson_trace()
struct LevinsonAnalysis {
  std::vector<double>
      solution; ///< Owned actual last completed leading-prefix solution; full result only if solved=true.
  bool solved{
      true}; ///< True only when every requested leading order completes, including valid empty input.
  std::size_t breakdown_order{
      0}; ///< Actual first failed leading order,0 if complete; not a full-matrix rank certificate.
  LevinsonWork work; ///< Actual completed orders and selected producing work.
};
/// @brief Actual three error sums and denominator of one attempted leading-order extension.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levinson_trace({2,1},{1,1});
/// std::print("{}\n",t.attempts[0].denominator); // 0.75
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see levinson_analyze(), levinson_trace()
struct LevinsonAttempt {
  std::size_t order{0};          ///< Actual target leading matrix order, at least2.
  double      forward_error{0};  ///< Actual sum row[s-i]*forward[i].
  double      backward_error{0}; ///< Actual sum row[i+1]*backward[i].
  double      solution_error{0}; ///< Actual sum row[s-i]*solution[i].
  double denominator{0}; ///< Actual1-backward_error*forward_error; exact zero triggers breakdown.
};
/// @brief Owned actual completed leading-order forward/backward basis and solution.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levinson_trace({2,1},{1,1});
/// std::print("{}\n",t.states.back().order); // 2
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see levinson_analyze(), levinson_trace()
struct LevinsonState {
  std::size_t         order{0}; ///< Actual completed leading dimension.
  std::vector<double> forward;  ///< Actual leading inverse-column basis, length order.
  std::vector<double> backward; ///< Actual trailing inverse-column basis, length order.
  std::vector<double> solution; ///< Actual completed prefix solve, length order.
};
/// @brief Owned input and every actual attempted/completed prefix from a single recurrence execution.
/// Complete conservative admission assumes every prefix completes:2*n inputs,
/// sum(k=1..n)(1+3*k) state/vector records and n-1 attempts. Actual returned
/// count can be smaller on breakdown. Fields/capacities/output/runtime copies
/// and bytes/RAM excluded; default million records,0 optional cap off.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levinson_trace({2,1},{1,1});
/// std::print("{} {}\n",t.states.size(),t.attempts.size()); // 2 1
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see levinson_analyze(), levinson_trace()
struct LevinsonTrace {
  std::vector<double> first_row; ///< Owned original symmetric Toeplitz defining row.
  std::vector<double> rhs;       ///< Owned original matched right-hand side.
  std::vector<LevinsonAttempt>
      attempts; ///< Actual ordered extension attempts, including a possible failed denominator.
  std::vector<LevinsonState>
      states; ///< Actual completed prefix states only; no invented post-breakdown state.
  LevinsonAnalysis analysis;       ///< Actual complete/partial result and producing work.
  std::size_t captured_entries{0}; ///< Actual input/state/coordinate/attempt logical record count.
};
/// @brief Independent widened residual observations for a caller-supplied full solution.
/// This is an observed normwise backward-error ratio, not a rigorous rounding
/// bound, conditioning estimate or forward-accuracy certificate.
/// @par Example
/// @code
/// const auto r=dm::algorithms::levinson_residual({2,1},{1,1},{1.0/3,1.0/3});
/// std::print("{}\n",r.infinity_norm<1e-12); // true
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see levinson_analyze(), levinson_trace()
struct LevinsonResidual {
  long double infinity_norm{0}; ///< Actual max absolute row residual, widened arithmetic.
  long double backward_error{
      0}; ///< Residual norm/(matrix infinity norm*x infinity norm+rhs infinity norm),0 if denominator0.
};
/// @brief Solve a finite symmetric Toeplitz system using unpivoted quadratic-time prefix recursion.
/// Preserves historical finite double recurrence order. Every leading block must
/// be nonsingular for this unpivoted recurrence; positive-definite matrices have
/// that property in exact arithmetic, but general invertible matrices can fail.
/// Exact zero initial diagonal or extension denominator throws. No tolerance,
/// pivoting, regularization or condition/forward-error guarantee; small nonzero
/// denominators can be ill-conditioned. O(n squared) time, O(n) runtime/output.
/// No input mutation or shared state.
/// @param first_row Borrowed complete finite defining row, T(i,j)=row[abs(i-j)].
/// @param rhs Borrowed complete matched finite right-hand side; both empty valid.
/// @return Owned full finite computed solution; empty for valid empty inputs.
/// @throws std::invalid_argument On complete shape/nonfinite input violations.
/// @throws std::length_error On full selected-work/container geometry.
/// @throws std::overflow_error On nonfinite produced recurrence arithmetic.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @throws std::domain_error On exact-zero initial diagonal or extension denominator.
/// @par Example
/// @code
/// const auto x=dm::algorithms::levinson_solve({2,1},{1,1});
/// std::print("{}\n",x.size()); // 2
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see LevinsonAnalysis, LevinsonTrace
std::vector<double> levinson_solve(
    const std::vector<double>& first_row,
    const std::vector<double>& rhs);
/// @brief Return actual unpivoted prefix recursion with explicit nonthrowing algebraic breakdown status.
/// Same checked finite recurrence as levinson_solve(); exact-zero breakdown
/// returns solved=false with its actual order. Other guard/overflow errors throw.
/// Count-only performs no state capture or additional residual evaluation.
/// @param first_row Borrowed complete finite defining row, T(i,j)=row[abs(i-j)].
/// @param rhs Borrowed complete matched finite right-hand side; both empty valid.
/// @return Owned last completed prefix and solved/breakdown/work; partial prefix is not full solution.
/// @throws std::invalid_argument On complete shape/nonfinite input violations.
/// @throws std::length_error On full selected-work/container geometry.
/// @throws std::overflow_error On nonfinite produced recurrence arithmetic.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::levinson_analyze({1,1,0},{1,2,3});
/// std::print("{} {}\n",a.solved,a.solution.size()); // false 1
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see LevinsonAnalysis, LevinsonTrace
LevinsonAnalysis levinson_analyze(
    const std::vector<double>& first_row,
    const std::vector<double>& rhs);
/// @brief Capture every actual successful prefix and attempted error/denominator once.
/// No repeated leading-prefix solves reconstruct history. Complete conservative
/// quota is admitted before copying/computation even if breakdown later shortens
/// the actual history. Actual captured_entries uses only produced records.
/// @param first_row Borrowed complete finite defining row, T(i,j)=row[abs(i-j)].
/// @param rhs Borrowed complete matched finite right-hand side; both empty valid.
/// @param max_entries Conservative full-completion logical cap, default1000000;0 disables optional cap.
/// @return Owned inputs, actual states/attempts and complete or explicit breakdown result.
/// @throws std::invalid_argument On complete shape/nonfinite input violations.
/// @throws std::length_error On full selected-work/container geometry.
/// @throws std::overflow_error On nonfinite produced recurrence arithmetic.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levinson_trace({2,1},{1,1});
/// std::print("{}\n",t.analysis.solved); // true
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see LevinsonAnalysis, LevinsonTrace
LevinsonTrace levinson_trace(
    const std::vector<double>& first_row,
    const std::vector<double>& rhs,
    std::size_t                max_entries = 1000000);
/// @brief Materialize the complete finite symmetric Toeplitz matrix for inspection/reference methods.
/// O(n squared) work/storage; no solve or positive-definiteness inference.
/// @param first_row Borrowed complete finite defining row; empty valid.
/// @return Owned n-by-n matrix with entry row[abs(i-j)].
/// @throws std::invalid_argument On nonfinite defining values.
/// @throws std::length_error On complete matrix geometry.
/// @throws std::bad_alloc On matrix allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::symmetric_toeplitz_matrix({2,1});
/// std::print("{}\n",a[1][0]); // 1
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see LevinsonAnalysis, LevinsonTrace
std::vector<std::vector<double>> symmetric_toeplitz_matrix(const std::vector<double>& first_row);
/// @brief Observe the full residual and normwise backward-error ratio with widened arithmetic.
/// O(n squared) additional work and O(1) additional storage; no factorization,
/// replay, conditioning estimate or rigorous error enclosure.
/// @param first_row Borrowed complete finite defining row, T(i,j)=row[abs(i-j)].
/// @param rhs Borrowed complete matched finite right-hand side; both empty valid.
/// @param solution Borrowed complete matched finite candidate solution, not a partial prefix.
/// @return Owned actual finite widened norm/ratio observations; both0 for empty input.
/// @throws std::invalid_argument On full shape/nonfinite input violations.
/// @throws std::overflow_error On nonfinite widened diagnostic arithmetic.
/// @par Example
/// @code
/// const auto r=dm::algorithms::levinson_residual({2},{4},{2});
/// std::print("{}\n",r.infinity_norm); // 0
/// @endcode
/// @par References
/// - N. Levinson, The Wiener(Root Mean Square) Error Criterion in Filter Design and Prediction, Journal of Mathematics and Physics25,261–278, https://doi.org/10.1002/sapm1946251261 (publisher dates volume1946).
/// @see LevinsonAnalysis, LevinsonTrace
LevinsonResidual levinson_residual(
    const std::vector<double>& first_row,
    const std::vector<double>& rhs,
    const std::vector<double>& solution);
} // namespace datamunge::algorithms
