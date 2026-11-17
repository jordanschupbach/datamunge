#pragma once
/// @file lanczos_iteration.hpp
/// @brief Checked symmetric dense Lanczos with one full reorthogonalization pass and actual stage evidence.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned producing tridiagonal coefficients and basis vectors, preserving legacy field layout.
/// V[j] is basis vector j, not matrix row j of Q. The leading m=alpha.size()
/// diagonal and first m-1 beta entries define T. Historical beta may additionally
/// contain the actual terminal residual norm on early stop when requested steps
/// exceeded m; that extra value is not a T off-diagonal. No eigenvalues computed.
/// @par Example
/// @code
/// const auto r=dm::algorithms::lanczos_iteration({{2,0},{0,3}},{1,0},2);
/// std::print("{} {}\n",r.alpha.size(),r.alpha[0]); // 1 2
/// @endcode
/// @see lanczos_trace()
struct LanczosResult {
  std::vector<double> alpha; ///< Actual producing diagonal coefficients, one per basis vector.
  std::vector<double>
      beta; ///< Actual residual norms; first m-1 used in T, optional terminal extra as documented.
  std::vector<std::vector<double>>
      V; ///< Owned complete actual normalized basis vectors, chronological.
};
/// @brief Actual producing reason for stopping; not an eigensolver convergence certificate.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_trace({{2}},{1},3);
/// std::print("{}\n",t.stop==dm::algorithms::LanczosStop::InvariantSubspace); // true
/// @endcode
/// @see LanczosTrace
enum class LanczosStop {
  StepLimit,         ///< Requested nonnegative step ceiling reached before full dimension.
  InvariantSubspace, ///< Actual residual norm below the historical absolute1e-12 threshold.
  ZeroStart,         ///< Actual input start vector has exactly zero norm.
  Empty,             ///< Empty matched matrix/start vector.
  DimensionLimit     ///< Full n-dimensional basis ceiling reached without the residual threshold.
};
/// @brief Selected actual dense Lanczos operations, excluding validation/copies/allocation/time.
/// dot_products counts alpha and reorthogonalization inner products, excluding
/// sum-of-squares norm work. matrix_products counts actual scalar A*v products.
/// Full reorthogonalization is one sequential pass over all current basis vectors.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_trace({{2}},{1},1);
/// std::print("{} {}\n",t.work.matvecs,t.work.reorth_projections); // 1 1
/// @endcode
/// @see LanczosTrace
struct LanczosWork {
  std::size_t matvecs{0};            ///< Actual full dense matrix-vector products.
  std::size_t matrix_products{0};    ///< Actual scalar matrix-entry times vector-entry products.
  std::size_t dot_products{0};       ///< Actual alpha/Gram inner-product calls, excluding norms.
  std::size_t reorth_projections{0}; ///< Actual sequential Gram subtraction projections.
  std::size_t norm_evaluations{
      0}; ///< Actual initial/residual norm calls, including widened fallback as one call.
};
/// @brief Actual same-pass coefficient subtracted during full reorthogonalization.
/// A coefficient is computed on the residual after all earlier projections in
/// this iteration, not on the original three-term residual or a replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_trace({{2}},{1},1);
/// std::print("{} {}\n",t.projections[0].iteration,t.projections[0].basis_index); // 0 0
/// @endcode
/// @see LanczosStep
struct LanczosProjection {
  std::size_t iteration{0};   ///< Actual current basis/iteration index.
  std::size_t basis_index{0}; ///< Actual earlier/current basis vector projected out.
  double      coefficient{0}; ///< Actual rounded inner product used in this subtraction.
};
/// @brief Owned actual matrix-product, three-term and final reorthogonalized residual boundaries.
/// Full vectors are producing snapshots, not repeated prefix runs. alpha and
/// beta_previous are actual recurrence values; residual_norm follows the one
/// full reorthogonalization pass and controls the absolute stopping decision.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_trace({{2}},{1},1);
/// std::print("{} {}\n",t.steps[0].alpha,t.steps[0].residual_norm); // 2 0
/// @endcode
/// @see LanczosTrace
struct LanczosStep {
  std::size_t         iteration{0};     ///< Actual basis/iteration ordinal.
  double              alpha{0};         ///< Actual v dot A*v diagonal coefficient.
  double              beta_previous{0}; ///< Actual previous residual norm, zero initially.
  double              residual_norm{0}; ///< Actual final norm after full reorthogonalization.
  std::vector<double> matrix_product;   ///< Actual A*v before recurrence subtractions.
  std::vector<double>
      three_term; ///< Actual residual after alpha*v+beta_previous*v_previous subtraction.
  std::vector<double>
      residual; ///< Actual residual after all sequential reorthogonalization projections.
};
/// @brief Owned complete original matrix/start, actual output/stages and same-execution selected work.
/// captured_entries counts n*n+n original scalar values, then1+3*n per step and
/// one record per projection. Excludes original row records, scalar fields,
/// result basis/coefficients, runtime copies/capacities/bytes/RAM. Admit complete
/// original geometry before copies, each step before matrix work and each
/// projection before its inner product. No replay or partial returned trace.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_trace({{2}},{1},1,7);
/// std::print("{} {}\n",t.steps.size(),t.captured_entries); // 1 7
/// @endcode
/// @see lanczos_trace()
struct LanczosTrace {
  std::vector<std::vector<double>>
                      matrix; ///< Owned complete original finite exactly symmetric matrix.
  std::vector<double> start;  ///< Owned complete original finite matched start vector.
  LanczosResult       result; ///< Actual returned coefficients and full basis.
  LanczosWork         work;   ///< Actual selected work, no synthetic complexity estimates.
  LanczosStop         stop{LanczosStop::StepLimit}; ///< Actual stopping reason.
  double              initial_norm{
      0}; ///< Actual finite normalized-input divisor, zero if empty/zero-step/zero start.
  std::vector<LanczosStep>       steps;       ///< Actual chronological stage boundaries.
  std::vector<LanczosProjection> projections; ///< Actual chronological one-pass Gram coefficients.
  std::size_t captured_entries{0}; ///< Logical original scalar+step-vector+projection record count.
};
/// @brief Build a symmetric Krylov basis and tridiagonal coefficients with full reorthogonalization.
/// Validate a finite square exactly symmetric A, matched finite b and m>=0 before
/// access/allocation. Empty returns empty; m0 returns empty; exact zero b returns
/// empty. Nonzero tiny starts are normalized using a widened norm fallback instead
/// of the old absolute1e-300 rejection. Iterate at most min(m,n), retaining actual
/// full basis, alpha and historical beta layout. For each v compute A*v, alpha,
/// subtract alpha*v+previous_beta*previous_v, then one Gram pass over all stored
/// basis vectors. Stop at absolute residual norm<1e-12, requested steps or full
/// dimension; absolute tolerance is not scale invariant or spectral convergence.
/// Ordinary finite double sum/dot/norm operation order is retained. If initial/
/// residual sum of squares is nonfinite or underflows to zero for a nonzero vector,
/// use widened hypot; all produced components/coefficients/norms must fit finite
/// double or throw. No guarantee for every finite ill-conditioned/huge input.
/// Dense work O(m*n*n+n*m*m), full-basis O(n*m) output/runtime storage, not a
/// constant-memory three-term-only method. No input mutation/shared state/history
/// quota; rounded reorthogonalization is not an exact orthogonality certificate.
/// @param A Borrowed complete finite exactly symmetric square row matrix.
/// @param b Borrowed complete matched finite nonzero start, zero allowed for empty convention.
/// @param m Nonnegative requested step ceiling, capped at n actual basis vectors.
/// @return Owned actual alpha/beta/full basis; no eigenvalues or convergence guarantee.
/// @throws std::invalid_argument On negative m, shape/nonfinite data or asymmetry.
/// @throws std::length_error On count/container representability limits.
/// @throws std::overflow_error On nonfinite produced component/coefficient/norm.
/// @throws std::bad_alloc On basis/temporary allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::lanczos_iteration({{2}},{1},4).alpha[0]); // 2
/// @endcode
/// @par References
/// - Lanczos(1950), An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators, J.Res.NBS45(4),255–282, https://nvlpubs.nist.gov/nistpubs/jres/045/jresv45n4p255_A1b.pdf.
///   Native code adds one full reorthogonalization pass and an absolute stopping policy.
/// @see lanczos_trace(), LanczosResult
LanczosResult
lanczos_iteration(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int m);
/// @brief Capture original inputs and actual producing Lanczos stages/Gram coefficients once.
/// Same validation/numerical/ordering/stopping contracts as lanczos_iteration().
/// Complete input and online step/projection admission follow LanczosTrace;
/// adds owned O(n*n+n*m+m*m) capture beyond the actual result/runtime storage.
/// @param A Borrowed complete finite exactly symmetric square matrix.
/// @param b Borrowed complete matched finite start vector.
/// @param m Actual nonnegative requested step ceiling.
/// @param max_entries Logical original/stage/projection ceiling, default1000000, zero off.
/// @return Owned complete actual observations, no prefix rerun or partial output.
/// @throws std::invalid_argument On complete input/step violations before copies.
/// @throws std::length_error On enabled capture quota or geometry/count/container limits.
/// @throws std::overflow_error On nonfinite produced arithmetic.
/// @throws std::bad_alloc On native/capture allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lanczos_trace({{2}},{1},1);
/// std::print("{} {}\n",t.steps[0].matrix_product[0],t.steps[0].residual[0]); // 2 0
/// @endcode
/// @par References
/// - Lanczos(1950), An Iteration Method for the Solution of the Eigenvalue Problem of Linear Differential and Integral Operators, https://nvlpubs.nist.gov/nistpubs/jres/045/jresv45n4p255_A1b.pdf.
/// @see LanczosTrace, lanczos_iteration()
LanczosTrace lanczos_trace(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    int                                     m,
    std::size_t                             max_entries = 1000000);
} // namespace datamunge::algorithms
