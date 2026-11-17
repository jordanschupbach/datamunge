#pragma once
/// @file stone_sip.hpp
/// @brief Checked actual Stone SIP factors, correction sweeps and final residuals.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the StoneResult contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StoneResult r{};
/// (void)r;
/// @endcode
/// @see stone_sip_run()
struct StoneResult {
  std::vector<double> x; ///< Actual final iterate.
  int    iterations{0};  ///< Actual completed correction updates, not the last pre-update index.
  double residual{};     ///< Actual recomputed ||b-A*x|| at this returned iterate.
};
/// @brief Own the StoneOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StoneOptions r{};
/// (void)r;
/// @endcode
/// @see stone_sip_run()
struct StoneOptions {
  double alpha{.92}; ///< Finite SIP fill-cancellation factor in[0,1).
  int    max_iterations{
      2000}; ///< Nonnegative correction cap; zero still checks initial residual and factorization.
  double tolerance{1e-10}; ///< Finite nonnegative inclusive absolute Euclidean residual threshold.
  std::size_t max_cells{1000000}; ///< Complete rectangular grid cell count; zero disables.
  std::size_t max_grid_visits{
      200000000}; ///< Actual factor cells, residual products, triangular cells and correction assignments; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual factor records/scalars and iterate records/copied coordinates; zero disables.
};
/// @brief Own the StoneFactorStep contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StoneFactorStep r{};
/// (void)r;
/// @endcode
/// @see stone_sip_run()
struct StoneFactorStep {
  std::size_t index{};             ///< Actual row-major grid cell factored.
  double      west_denominator{};  ///< Actual1+alpha*Un[west], or1 at west boundary.
  double      south_denominator{}; ///< Actual1+alpha*Ue[south], or1 at south boundary.
  double      first_fill{};        ///< Actual alpha*Lw*Un[west].
  double      second_fill{};       ///< Actual alpha*Ls*Ue[south].
};
/// @brief Own the StoneIteration contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StoneIteration r{};
/// (void)r;
/// @endcode
/// @see stone_sip_run()
struct StoneIteration {
  int                 iteration{}; ///< Actual completed correction count; initial state is0.
  double              residual{};  ///< Actual recomputed ||b-A*x|| for this iterate.
  std::vector<double> x; ///< Actual complete owned iterate after the indicated corrections.
};
/// @brief Own the StoneRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StoneRun r{};
/// (void)r;
/// @endcode
/// @see stone_sip_run()
struct StoneRun {
  StoneResult result;      ///< Actual final iterate, completed corrections and matching residual.
  bool        converged{}; ///< Actual final residual<=tolerance.
  double      initial_residual{};         ///< Actual residual of the zero initial iterate.
  std::vector<double> Lw;                 ///< Actual complete west factor coefficients.
  std::vector<double> Ls;                 ///< Actual complete south factor coefficients.
  std::vector<double> Lp;                 ///< Actual complete nonzero lower diagonal.
  std::vector<double> Ue;                 ///< Actual complete normalized east upper coefficients.
  std::vector<double> Un;                 ///< Actual complete normalized north upper coefficients.
  std::size_t         factorized_cells{}; ///< Actual grid cells processed by factorization.
  std::size_t
      residual_products{}; ///< Actual stencil coefficient-times-iterate products in all residual checks.
  std::size_t triangular_cells{};       ///< Actual forward/back substitution cell visits.
  std::size_t correction_assignments{}; ///< Actual iterate coordinate additions.
  std::vector<StoneFactorStep> factors; ///< Optional actual factor intermediate values.
  std::vector<StoneIteration>
      history; ///< Optional actual initial and post-correction iterates/residuals.
};
/// @brief Own the StoneGrid contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StoneGrid r{};
/// (void)r;
/// @endcode
/// @see stone_sip_run()
struct StoneGrid {
  int                 nx{}; ///< Complete horizontal grid order.
  int                 ny{}; ///< Complete vertical grid order.
  std::vector<double> Ap;   ///< Row-major center coefficients.
  std::vector<double> Aw;   ///< West coefficients, zero at west boundary.
  std::vector<double> Ae;   ///< East coefficients, zero at east boundary.
  std::vector<double> As;   ///< South coefficients, zero at first grid row.
  std::vector<double> An;   ///< North coefficients, zero at final grid row.
  std::vector<double> b;    ///< Actual assembled right-hand side.
  std::vector<double>
      known; ///< Supplied controlled manufactured solution; empty for general caller grids.
};
/// @brief Factor and solve an actual five-point grid with Stone SIP.
/// Row-major p=j*nx+i; all six coefficient/RHS vectors exactly nx*ny, finite. Dimensions nonnegative; any zero dimension requires empty vectors and produces a converged empty result. Coefficients beyond each boundary must be zero. Builds the actual SIP incomplete triangular factors in grid order, checking denominator and pivot breakdown; then solves L*rho=r, U*delta=rho and adds delta. Residual is independently recomputed at the zero initial iterate and after every complete correction, so returned x, iteration count and residual refer to the same state. Positive definiteness/convergence are not assumed verified, and residual need not decrease monotonically for arbitrary inputs. Absolute inclusive stopping; cap returns converged=false if unresolved. O(grid cells) work per correction, linear working/result factor storage, optional copied histories O(corrections*cells). Outputs own values, no shared state.
/// @param grid Complete borrowed finite five-point operator and RHS; known field ignored by solver.
/// @param options Fill factor, correction cap/tolerance and complete/actual work/capture policies.
/// @return Owned actual factors, final matching residual, convergence flag, work and optional producing histories.
/// @throws std::invalid_argument Malformed dimensions/shapes/boundaries, nonfinite values or invalid controls.
/// @throws std::domain_error Actual zero incomplete-factor denominator or pivot.
/// @throws std::overflow_error Actual factor, substitution, update or residual not representable.
/// @throws std::length_error Complete geometry, actual work/history or counter/container capacity exceeded.
/// @par Example
/// @code
/// dm::algorithms::StoneGrid g{1,1,{2},{0},{0},{0},{0},{4},{}};
/// auto r=dm::algorithms::stone_sip_run(g);
/// std::print("{} {}\n",r.result.x[0],r.converged); // 2 true
/// @endcode
/// @par References
/// - H. L. Stone, Iterative Solution of Implicit Approximations of Multidimensional Partial Differential Equations, SIAM J. Numer. Anal.5(3),1968, DOI10.1137/0705044.
/// @see stone_sip()
StoneRun stone_sip_run(const StoneGrid& grid, const StoneOptions& options = {});
/// @brief Factor and solve an actual five-point grid with Stone SIP retaining actual factors/iterates.
/// Row-major p=j*nx+i; all six coefficient/RHS vectors exactly nx*ny, finite. Dimensions nonnegative; any zero dimension requires empty vectors and produces a converged empty result. Coefficients beyond each boundary must be zero. Builds the actual SIP incomplete triangular factors in grid order, checking denominator and pivot breakdown; then solves L*rho=r, U*delta=rho and adds delta. Residual is independently recomputed at the zero initial iterate and after every complete correction, so returned x, iteration count and residual refer to the same state. Positive definiteness/convergence are not assumed verified, and residual need not decrease monotonically for arbitrary inputs. Absolute inclusive stopping; cap returns converged=false if unresolved. O(grid cells) work per correction, linear working/result factor storage, optional copied histories O(corrections*cells). Outputs own values, no shared state.
/// @param grid Complete borrowed finite five-point operator and RHS; known field ignored by solver.
/// @param options Fill factor, correction cap/tolerance and complete/actual work/capture policies.
/// @return Owned actual factors, final matching residual, convergence flag, work and optional producing histories.
/// @throws std::invalid_argument Malformed dimensions/shapes/boundaries, nonfinite values or invalid controls.
/// @throws std::domain_error Actual zero incomplete-factor denominator or pivot.
/// @throws std::overflow_error Actual factor, substitution, update or residual not representable.
/// @throws std::length_error Complete geometry, actual work/history or counter/container capacity exceeded.
/// @par Example
/// @code
/// dm::algorithms::StoneGrid g{1,1,{2},{0},{0},{0},{0},{4},{}};
/// auto r=dm::algorithms::stone_sip_trace(g);
/// std::print("{} {}\n",r.result.x[0],r.converged); // 2 true
/// @endcode
/// @par References
/// - H. L. Stone, Iterative Solution of Implicit Approximations of Multidimensional Partial Differential Equations, SIAM J. Numer. Anal.5(3),1968, DOI10.1137/0705044.
/// @see stone_sip()
StoneRun stone_sip_trace(const StoneGrid& grid, const StoneOptions& options = {});
/// @brief Return the preserved checked Stone SIP result signature.
/// Same actual producing factors/corrections. Returned residual is now recomputed at returned x, and iterations counts completed corrections, fixing the historical pre-update residual/index shortcut. Default policies apply.
/// @param Nx Nonnegative horizontal dimension.
/// @param Ny Nonnegative vertical dimension.
/// @param Ap Complete finite center coefficients.
/// @param Aw Complete finite west coefficients, zero at west boundary.
/// @param Ae Complete finite east coefficients, zero at east boundary.
/// @param As Complete finite south coefficients, zero at first row.
/// @param An Complete finite north coefficients, zero at final row.
/// @param b Complete finite RHS.
/// @param alpha Finite fill factor in[0,1), default.92.
/// @param max_iter Nonnegative correction cap, default2000.
/// @param tol Finite nonnegative absolute residual tolerance, default1e-10.
/// @return Owned actual final iterate, completed corrections and matching residual.
/// @throws std::invalid_argument Malformed inputs/controls.
/// @throws std::domain_error Preconditioner breakdown.
/// @throws std::overflow_error Actual arithmetic not representable.
/// @throws std::length_error Default geometry/work/capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::stone_sip(1,1,{2},{0},{0},{0},{0},{4}).x[0]); // 2
/// @endcode
/// @see stone_sip_run()
StoneResult stone_sip(
    int                        Nx,
    int                        Ny,
    const std::vector<double>& Ap,
    const std::vector<double>& Aw,
    const std::vector<double>& Ae,
    const std::vector<double>& As,
    const std::vector<double>& An,
    const std::vector<double>& b,
    double                     alpha    = .92,
    int                        max_iter = 2000,
    double                     tol      = 1e-10);
/// @brief Generate the original manufactured five-point Poisson fixture.
/// Center4 and valid neighbor-1, with zero missing neighbors. Draw known solution from local mt19937 seed1 by default and uniform_real_distribution[-1,1], then assemble b=A*known in original center,west,east,south,north double arithmetic order. No solve is hidden. Default30x30 preserves original900-point experiment; generated distribution mapping follows standard library implementation.
/// @param nx Positive horizontal order, default30.
/// @param ny Positive vertical order, default30.
/// @param seed Complete local mt19937 seed, default1.
/// @return Owned actual operator, RHS and known solution.
/// @throws std::invalid_argument Nonpositive fixture dimension.
/// @throws std::length_error Complete fixture exceeds one million cells.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::stone_poisson_example(2,3).b.size()); // 6
/// @endcode
/// @see stone_sip_run()
StoneGrid stone_poisson_example(int nx = 30, int ny = 30, std::uint32_t seed = 1);
} // namespace datamunge::algorithms
