#pragma once
/// @file lax_wendroff.hpp
/// @brief Checked periodic Lax–Wendroff advection with actual stencil/state evidence and explicit CFL controls.
#include <complex>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual producing arithmetic path used for one periodic advection cell.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lax_wendroff_trace({1,2},1,1,1,1);
/// std::print("{}\n",t.cells[0].kind==dm::algorithms::LaxWendroffCellKind::ShiftRight); // true
/// @endcode
/// @see LaxWendroffCell
enum class LaxWendroffCellKind {
  General,    ///< Actual ordinary/widened three-point Taylor update.
  CopyCenter, ///< Actual exact numeric identity for computed Courant0.
  ShiftRight, ///< Actual periodic left-neighbor copy for computed Courant+1.
  ShiftLeft   ///< Actual periodic right-neighbor copy for computed Courant-1.
};
/// @brief Selected actual stencil/update work, excluding validation, copying, allocation and time.
/// stencil_reads counts the three original operand point identities initialized
/// per cell, not physical machine loads. diagnostics_cells counts additional
/// trace-only state observations; ordinary/count-only calls retain0 there.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lax_wendroff_analyze({0,1,0},1,1,.5,2);
/// std::print("{} {}\n",a.work.cell_updates,a.work.stencil_reads); // 6 18
/// @endcode
/// @see LaxWendroffAnalysis
struct LaxWendroffWork {
  std::size_t steps{0};        ///< Actual complete periodic steps performed.
  std::size_t cell_updates{0}; ///< Actual grid cells updated across all steps.
  std::size_t stencil_reads{
      0}; ///< Actual selected left/center/right operand identities, three per cell.
  std::size_t general_updates{0}; ///< Actual nonzero/nonunit-Courant Taylor updates.
  std::size_t exact_copies{0};    ///< Actual Courant0 or unit-shift cell copies.
  std::size_t widened_updates{
      0}; ///< Actual general updates using overflow-avoiding widened fallback.
  std::size_t diagnostics_cells{
      0}; ///< Actual trace-only input/state diagnostic point observations.
};
/// @brief Owned actual final grid, computed Courant coefficient and selected same-execution work.
/// allow_unstable records explicit opt-in to abs(Courant)>1, not a stability
/// certificate. Ordinary API and default analysis enforce the CFL domain.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lax_wendroff_analyze({0,0,1,0,0},1,1,.5,1);
/// std::print("{} {}\n",a.courant,a.values[1]); // 0.5 -0.125
/// @endcode
/// @see lax_wendroff_trace()
struct LaxWendroffAnalysis {
  std::vector<double> values;     ///< Owned actual final periodic grid, original index order.
  double              courant{0}; ///< Actual finite computed a*dt/dx coefficient used.
  bool                allow_unstable{false}; ///< Actual explicit CFL-enforcement opt-out flag.
  LaxWendroffWork     work;                  ///< Actual selected producing operations.
};
/// @brief Owned actual old-state stencil and producing new value for one periodic cell.
/// General terms give value=center-advective_term+diffusive_term, with actual
/// double terms widened for storage or actual widened fallback terms. Copy paths
/// have unused term placeholders0 and use the identified original operand directly.
/// Every cell reads the complete old state, never a previously updated new value.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lax_wendroff_trace({0,0,1,0,0},1,1,.5,1);
/// std::print("{} {}\n",t.cells[1].index,t.cells[1].value); // 1 -0.125
/// @endcode
/// @see LaxWendroffTrace
struct LaxWendroffCell {
  std::size_t step{0};           ///< Actual zero-based producing time step.
  std::size_t index{0};          ///< Actual original cell index.
  std::size_t left_index{0};     ///< Actual wrapped old-state left neighbor identity.
  std::size_t right_index{0};    ///< Actual wrapped old-state right neighbor identity.
  double      left{0};           ///< Actual old-state left value.
  double      center{0};         ///< Actual old-state center value.
  double      right{0};          ///< Actual old-state right value.
  long double advective_term{0}; ///< Actual subtractive first-difference term,0 on copy paths.
  long double diffusive_term{0}; ///< Actual additive second-difference term,0 on copy paths.
  double      value{0};          ///< Actual finite committed next-state value.
  LaxWendroffCellKind kind{LaxWendroffCellKind::General}; ///< Actual arithmetic/copy path.
  bool widened{false}; ///< Whether nonfinite ordinary intermediates triggered widened fallback.
};
/// @brief Actual complete-state time, mass, L2 and extrema observations.
/// Time/mass/L2 use long double arithmetic and are observations, not exact
/// conservation/stability/positivity certificates. L2=sqrt(dx*sum(u*u)).
/// @par Example
/// @code
/// const auto t=dm::algorithms::lax_wendroff_trace({0,0,1,0,0},1,1,.5,1);
/// std::print("{} {}\n",t.diagnostics.front().mass,t.diagnostics.back().mass); // 1 1
/// @endcode
/// @see LaxWendroffTrace
struct LaxWendroffState {
  std::size_t completed_steps{0}; ///< Actual number of completed full updates.
  long double time{0};    ///< Actual widened completed_steps*dt, no time-integration replay.
  long double mass{0};    ///< Actual widened dx*sum of all periodic grid values.
  long double l2_norm{0}; ///< Actual widened sqrt(dx*sum of squared grid values).
  double      minimum{0}; ///< Actual finite minimum stored grid value.
  double      maximum{0}; ///< Actual finite maximum stored grid value.
};
/// @brief Owned complete original/every actual state, producing stencils and explicit diagnostics.
/// states[0] is original input, followed by every real completed step; no prefix
/// reruns. captured_entries=n+1+steps*(2*n+1): original n values/diagnostic, then
/// n next-state values, one diagnostic and n cell records per step. Excludes
/// fields/capacities/final-result/runtime copies and bytes/RAM. Complete count
/// and quota admitted before copies/evaluation, default million, zero off.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lax_wendroff_trace({1,2},1,1,1,1,false,8);
/// std::print("{} {}\n",t.states.size(),t.captured_entries); // 2 8
/// @endcode
/// @see lax_wendroff_trace()
struct LaxWendroffTrace {
  double              speed{0};    ///< Actual caller advection speed a.
  double              spacing{1};  ///< Actual positive caller grid spacing dx.
  double              timestep{0}; ///< Actual nonnegative caller timestep dt.
  LaxWendroffAnalysis analysis;    ///< Actual final values/Courant/selected work and CFL policy.
  std::vector<std::vector<double>> states; ///< Owned original and every actual full periodic grid.
  std::vector<LaxWendroffState>    diagnostics; ///< Actual widened observations of captured states.
  std::vector<LaxWendroffCell>
              cells; ///< Actual chronological step-then-cell producing operands/terms.
  std::size_t captured_entries{
      0}; ///< Complete original/state/diagnostic/cell-record logical count.
};
/// @brief Compute the periodic Lax–Wendroff Fourier symbol1-i*nu*sin(theta)+nu^2*(cos(theta)-1).
/// Pure analytic reference, not measured solver work or a convergence certificate.
/// Finite coefficients/angle required; unstable nu is allowed for explicit theory
/// controls. abs(G)<=1 for all theta in exact arithmetic when abs(nu)<=1.
/// @param courant Finite actual Courant coefficient nu, outside[-1,1] explicitly allowed here.
/// @param angle Finite Fourier phase in radians.
/// @return Actual finite double complex amplification reference.
/// @throws std::invalid_argument On nonfinite coefficient/angle.
/// @throws std::overflow_error On nonfinite/unrepresentable produced symbol arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::lax_wendroff_amplification(.5,0).real()); // 1
/// @endcode
/// @par References
/// - Lax and Wendroff(1960), Systems of Conservation Laws, Communications on Pure and Applied Mathematics13(2),217–237, https://doi.org/10.1002/cpa.3160130205.
/// @see lax_wendroff_analyze()
std::complex<double> lax_wendroff_amplification(double courant, double angle);
/// @brief Advect a finite nonempty periodic grid through checked second-order Lax–Wendroff steps.
/// Require finite speed, positive finite dx, nonnegative finite dt, nsteps>=0 and
/// computed abs(a*dt/dx)<=1. Complete input/work geometry checked before copies.
/// Each full old-state update is u[j]-.5*nu*(right-left)+.5*nu*nu*(right-2*u[j]+left).
/// Preserve ordinary finite double formula order. If intermediates are nonfinite,
/// evaluate the same terms in long double and require finite double output.
/// Computed nu0 copies center exactly; nu+1 copies wrapped left and nu-1 wrapped
/// right, avoiding cancellation and making numeric unit-Courant shifts exact.
/// Courant multiplication uses widened fallback if ordinary product/ratio is
/// nonfinite. Zero steps returns original after complete validation.
/// Exact periodic mass conservation and L2 stability do not imply positivity,
/// monotonicity, no ringing or exact floating-point conservation. Second-order
/// consistency requires sufficiently smooth solutions; shocks/steps can oscillate.
/// O(n+nsteps*n) work/O(n) runtime/output, no mutation/history/shared state.
/// @param u0 Borrowed complete finite nonempty initial periodic cell values.
/// @param a Finite signed advection speed.
/// @param dx Positive finite uniform grid spacing.
/// @param dt Nonnegative finite time step,0 permits exact identity updates.
/// @param nsteps Nonnegative complete update count.
/// @return Owned actual finite final periodic values.
/// @throws std::invalid_argument On complete input/parameter or enforced CFL violations.
/// @throws std::length_error On complete selected-work/container geometry limits.
/// @throws std::overflow_error On unrepresentable Courant or produced update arithmetic.
/// @throws std::bad_alloc On state/output allocation failure.
/// @par Example
/// @code
/// const auto u=dm::algorithms::lax_wendroff_advection({1,2,3},1,1,1,1);
/// std::print("{} {} {}\n",u[0],u[1],u[2]); // 3 1 2
/// @endcode
/// @par References
/// - Lax and Wendroff(1960), Systems of Conservation Laws, https://doi.org/10.1002/cpa.3160130205.
/// @see lax_wendroff_analyze(), lax_wendroff_trace()
std::vector<double>
lax_wendroff_advection(const std::vector<double>& u0, double a, double dx, double dt, int nsteps);
/// @brief Return actual periodic updates/Courant/work, with explicit optional unstable-control mode.
/// Same finite/geometry/arithmetic contracts as lax_wendroff_advection(). Default
/// enforces CFL; allow_unstable=true explicitly permits abs(nu)>1 for controlled
/// diagnostics, while finite produced values remain mandatory. No input/history
/// copies or state diagnostics, no optional capture quota or prefix reruns.
/// @param u0 Borrowed complete finite nonempty initial periodic values.
/// @param a Actual finite signed speed.
/// @param dx Actual positive finite spacing.
/// @param dt Actual nonnegative finite timestep.
/// @param nsteps Actual nonnegative complete update count.
/// @param allow_unstable Opt out of abs(nu)<=1 only if true, default false.
/// @return Owned actual final values/Courant/policy and producing selected work.
/// @throws std::invalid_argument On complete input/parameter/enforced CFL violations.
/// @throws std::length_error On complete work/container geometry limits.
/// @throws std::overflow_error On nonfinite Courant/update arithmetic.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::lax_wendroff_analyze({1,2},0,1,1,2).work.exact_copies); // 4
/// @endcode
/// @par References
/// - Lax and Wendroff(1960), Systems of Conservation Laws, https://doi.org/10.1002/cpa.3160130205.
/// @see LaxWendroffWork, lax_wendroff_trace()
LaxWendroffAnalysis lax_wendroff_analyze(
    const std::vector<double>& u0,
    double                     a,
    double                     dx,
    double                     dt,
    int                        nsteps,
    bool                       allow_unstable = false);
/// @brief Capture every actual periodic state, stencil decision and widened state diagnostic once.
/// Same shared producer/explicit CFL policy as lax_wendroff_analyze(). Complete
/// n+1+nsteps*(2*n+1) quota admitted before input/history copies or numerical work.
/// Adds O(nsteps*n) owned observation and diagnostic work beyond the runtime state.
/// No prefix replay or partial returned trace; quota counts records, not peak bytes.
/// @param u0 Borrowed complete finite nonempty original periodic grid.
/// @param a Actual finite signed speed.
/// @param dx Actual positive finite spacing.
/// @param dt Actual nonnegative finite timestep.
/// @param nsteps Actual nonnegative complete update count.
/// @param allow_unstable Explicitly opt out of enforced CFL only if true, default false.
/// @param max_entries Complete original/state/diagnostic/cell ceiling, default1000000, zero off.
/// @return Owned complete actual trajectory/stencils/diagnostics and final analysis.
/// @throws std::invalid_argument On complete input/parameter/enforced CFL violations.
/// @throws std::length_error On enabled quota or complete work/container geometry limits.
/// @throws std::overflow_error On nonfinite update/Courant/widened diagnostic arithmetic.
/// @throws std::bad_alloc On state/history allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lax_wendroff_trace({1,2,3},1,1,1,1);
/// std::print("{} {}\n",t.cells.size(),t.analysis.values[0]); // 3 3
/// @endcode
/// @par References
/// - Lax and Wendroff(1960), Systems of Conservation Laws, https://doi.org/10.1002/cpa.3160130205.
/// @see LaxWendroffTrace, lax_wendroff_analyze()
LaxWendroffTrace lax_wendroff_trace(
    const std::vector<double>& u0,
    double                     a,
    double                     dx,
    double                     dt,
    int                        nsteps,
    bool                       allow_unstable = false,
    std::size_t                max_entries    = 1000000);
} // namespace datamunge::algorithms
