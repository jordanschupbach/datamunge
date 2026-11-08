#pragma once
/// @file crank_nicolson.hpp
/// @brief Checked homogeneous heat evolution, actual Thomas traces and sine-mode controls.
#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::algorithms {

/// @brief Validated uniform heat-grid configuration with homogeneous Dirichlet boundaries.
/// nx interior points lie at rounded i*dx for i=1,...,nx; endpoints are exactly 0,L.
/// Rounded interior coordinates must remain strictly inside the domain.
/// r=D*dt/(2*dx*dx). Coefficients and final time must fit finite double storage.
/// @par Example
/// @code
/// const auto t=dm::algorithms::crank_nicolson_heat_trace([](double){return 1.;},1,1,1,.25,1);
/// std::print("{} {}\n",t.grid.dx,t.grid.r); // 0.5 0.5
/// @endcode
/// @see CrankNicolsonHeatTrace
struct HeatGrid {
  double length=1; ///< Positive domain length L.
  double diffusivity=1; ///< Nonnegative diffusivity D.
  int interior_points=1; ///< Positive number nx of interior unknowns.
  double dx=.5; ///< Positive representable spacing L/(nx+1).
  double dt=0; ///< Nonnegative finite time step.
  double r=0; ///< Finite nonnegative half diffusion ratio.
  int steps=0; ///< Nonnegative number of requested complete steps.
  double final_time=0; ///< Finite steps*dt.
};

/// @brief Owned actual state and uneliminated right-hand side at one heat time level.
/// Initial state has index/time zero, an empty rhs and zero scaled_residual.
/// profiles include both exact zero endpoints; rhs holds nx interior entries.
/// RMS uses the nx interior values. scaled_residual is the maximum rowwise
/// |A*u-rhs|/(|rhs|+|A|*|u|), with zero/zero defined as zero. It is a
/// backward-error diagnostic of the stored tridiagonal solve, not a PDE error.
/// @par Example
/// @code
/// const auto t=dm::algorithms::crank_nicolson_heat_trace([](double){return 1.;},1,1,1,.25,1);
/// std::print("{} {}\n",t.states[1].profile,t.states[1].rhs); // [0, 0, 0] [0]
/// @endcode
/// @see CrankNicolsonHeatTrace
struct HeatStep {
  int index=0; ///< Actual completed step number, zero for sampled initial state.
  double time=0; ///< index*dt, without accumulated time drift.
  std::vector<double> profile; ///< Owned nx+2 values, with both endpoints zero.
  std::vector<double> rhs; ///< Owned actual nx-entry right-hand side before Thomas elimination.
  double rms=0; ///< Root mean square of the nx interior values.
  double max_abs=0; ///< Largest absolute interior value.
  double scaled_residual=0; ///< Maximum componentwise normalized linear-system residual.
};

/// @brief Complete owned initial/step states and actual reusable Thomas factors.
/// One kernel drives final-only and traced calls. Factors are prepared once
/// when steps>0; pivots are the actual eliminated diagonal and multipliers[0]=0.
/// States include the initial sample plus all requested steps, never a partial
/// history. Work/storage is O(nx*(steps+1)); final-only storage is O(nx).
/// All vectors are owned and no callbacks/references survive. Independent calls
/// are thread-safe subject to the supplied callback's own behavior.
/// @par Example
/// @code
/// const auto t=dm::algorithms::crank_nicolson_heat_trace([](double){return 1.;},1,0,2,.1,2);
/// std::print("{} {}\n",t.states.size(),t.states.back().profile); // 3 [0, 1, 1, 0]
/// @endcode
/// @par References
/// - J. Crank and P. Nicolson (1947), A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type, 43(1), 50-67. https://doi.org/10.1017/S0305004100023197
/// @see crank_nicolson_heat_trace(), HeatStep
struct CrankNicolsonHeatTrace {
  HeatGrid grid; ///< Validated grid and step metadata.
  std::vector<double> coordinates; ///< Owned nx+2 coordinates, endpoints exactly 0 and L.
  std::vector<double> pivots; ///< Actual length-nx Thomas pivots; empty when no steps requested.
  std::vector<double> multipliers; ///< Actual length-nx elimination multipliers; empty for zero steps.
  std::vector<HeatStep> states; ///< Complete actual initial and successive states.
  std::size_t factorizations=0; ///< One if steps>0, otherwise zero.
  std::size_t solves=0; ///< Actual completed tridiagonal solves, equal to steps.
  std::size_t rhs_entries=0; ///< Actual stencil entries formed, nx*steps.
};

/// @brief Owned unit sine-mode controls separating continuous, spatial and time errors.
/// For mode m=1,...,nx, initial[i]=sin(m*pi*i/(nx+1)). continuous solves the
/// continuous heat equation at the grid points; semidiscrete uses the exact
/// central-difference eigenvalue; discrete applies the ideal CN amplification
/// factor steps times. These are floating evaluations of modal formulas,
/// independent of the Thomas recurrence, not arbitrary-precision certificates.
/// Boundary values are set to exact zero. Negative amplification alternates sign.
/// @par Example
/// @code
/// const auto m=dm::algorithms::heat_sine_mode(1,1,1,1,.25,1);
/// std::print("{:.1f} {:.1f}\n",m.eigenvalue,m.amplification); // -8.0 0.0
/// @endcode
/// @par References
/// - Crank and Nicolson (1947), https://doi.org/10.1017/S0305004100023197
/// @see heat_sine_mode(), crank_nicolson_heat_trace()
struct HeatSineMode {
  HeatGrid grid; ///< Validated evolution configuration.
  int mode=1; ///< Positive sine-mode index, at most nx.
  double eigenvalue=0; ///< Finite nonpositive central-difference heat-operator eigenvalue.
  double amplification=1; ///< Ideal one-step CN modal factor in [-1,1].
  std::vector<double> coordinates; ///< Owned nx+2 spatial positions.
  std::vector<double> initial; ///< Unit-amplitude sine at time zero, endpoints zero.
  std::vector<double> continuous; ///< exp(-D*(mode*pi/L)^2*t) times initial.
  std::vector<double> semidiscrete; ///< exp(eigenvalue*t) times initial.
  std::vector<double> discrete; ///< amplification^steps times initial.
};

/// @brief Advance u_t=D*u_xx on [0,L] with exact zero boundary values.
/// Samples u0 once at each ascending interior grid point, never at endpoints.
/// One shared CN/Thomas kernel averages old/new spatial stencils and factors
/// the constant matrix once. Returns nx+2 owned values; zero steps returns the
/// sampled initial values with zero endpoints. D=0 or dt=0 gives identity steps.
/// O(nx*(nsteps+1)) time and O(nx) auxiliary storage. Ideal l2 stability does
/// not imply positivity, strong stiff-mode damping, or immunity to rounding.
/// Coefficients and arithmetic are checked; no implicit precision guarantee.
/// @param u0 Nonempty callback yielding finite initial values; its exceptions propagate.
/// @param L Finite positive domain length.
/// @param D Finite nonnegative diffusivity.
/// @param nx Positive interior-point count; nx+2 must fit size_t/container limits.
/// @param dt Finite nonnegative time increment.
/// @param nsteps Nonnegative number of steps.
/// @return Owned final nx+2 profile including exact zero endpoints.
/// @throws std::invalid_argument For invalid parameters, empty callback or nonfinite sampled values.
/// @throws std::overflow_error If spacing loses interior coordinates, a positive ratio underflows, coefficients/time do not fit, or step arithmetic becomes nonfinite.
/// @throws std::length_error If a container or dimension/work count exceeds size_t limits.
/// @throws std::bad_alloc If temporary or output storage cannot be allocated.
/// @par Example
/// @code
/// const auto u=dm::algorithms::crank_nicolson_heat([](double){return 1.;},1,0,1,.1,2);
/// std::print("{}\n",u); // [0, 1, 0]
/// @endcode
/// @par References
/// - J. Crank and P. Nicolson (1947), A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type, 43(1), 50-67. https://doi.org/10.1017/S0305004100023197
/// @see crank_nicolson_heat_trace(), heat_sine_mode()
std::vector<double> crank_nicolson_heat(const std::function<double(double)>& u0,double L,double D,int nx,double dt,int nsteps);

/// @brief Retain every actual heat state, RHS and reusable Thomas factor without replay.
/// Same checked kernel and sample order as crank_nicolson_heat(). Validates
/// scalar parameters/coefficient arithmetic before the complete storage budget;
/// budget failure occurs before allocating profiles or invoking u0. Callback
/// results are then fully checked during initial sampling. No partial result
/// is returned. History scalar count is (nx+2)*(nsteps+1)+nx*nsteps; the cap
/// excludes coordinates, factors and metadata. No state is retained after return.
/// @param u0 Finite-valued interior initial-profile callback; exceptions propagate.
/// @param L Finite positive domain length.
/// @param D Finite nonnegative diffusivity.
/// @param nx Positive number of interior values.
/// @param dt Finite nonnegative time increment.
/// @param nsteps Nonnegative number of complete steps.
/// @param max_values Complete history/RHS scalar cap, default 1000000; zero disables only this cap.
/// @return Owned grid, actual factors and exactly nsteps+1 complete states.
/// @throws std::invalid_argument For invalid scalar parameters/callback or nonfinite sampled values.
/// @throws std::overflow_error For lost interior coordinates, unrepresentable spacing/ratio/coefficient/time or nonfinite arithmetic.
/// @throws std::length_error If complete history budget or dimension/container/work limits are exceeded.
/// @throws std::bad_alloc If owned or working storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::crank_nicolson_heat_trace([](double){return 1.;},1,1,1,1,2,11);
/// std::print("{:.2f} {:.2f}\n",t.states[1].profile[1],t.states[2].profile[1]); // -0.60 0.36
/// @endcode
/// @par References
/// - Crank and Nicolson (1947), https://doi.org/10.1017/S0305004100023197
/// @see CrankNicolsonHeatTrace, HeatStep
CrankNicolsonHeatTrace crank_nicolson_heat_trace(const std::function<double(double)>& u0,double L,double D,int nx,double dt,int nsteps,std::size_t max_values=1'000'000);

/// @brief Evaluate independent sine-mode reference profiles for the heat discretization.
/// Uses the same validated grid, with lambda_h=-4*D*sin²(m*pi/(2*(nx+1)))/dx²
/// and g=(1+dt*lambda_h/2)/(1-dt*lambda_h/2). O(nx) time/storage, no Thomas
/// calls or callbacks. Eigenvalue must fit finite double storage; analytic
/// decay may underflow to zero. The ideal formula does not include solve rounding.
/// @param L Finite positive domain length.
/// @param D Finite nonnegative diffusivity.
/// @param nx Positive interior grid count.
/// @param mode Integer sine mode in [1,nx].
/// @param dt Finite nonnegative time step.
/// @param nsteps Nonnegative number of steps.
/// @return Owned initial, continuous, spatially discretized and ideal CN profiles.
/// @throws std::invalid_argument If any scalar domain or mode is invalid.
/// @throws std::overflow_error If interior coordinates are lost or spacing, positive ratio, coefficient, final time or eigenvalue cannot be represented.
/// @throws std::length_error If dimension/container/work limits are exceeded.
/// @throws std::bad_alloc If result storage cannot be allocated.
/// @par Example
/// @code
/// const auto m=dm::algorithms::heat_sine_mode(1,1,1,1,1,2);
/// std::print("{:.2f} {:.2f}\n",m.amplification,m.discrete[1]); // -0.60 0.36
/// @endcode
/// @par References
/// - Crank and Nicolson (1947), https://doi.org/10.1017/S0305004100023197
/// @see HeatSineMode, crank_nicolson_heat_trace()
HeatSineMode heat_sine_mode(double L,double D,int nx,int mode,double dt,int nsteps);

/// @brief Return the largest absolute difference between two finite equal-length profiles.
/// Empty matching profiles have zero error. Inputs are read-only, no references
/// are retained, and O(n) work uses widened differences before checked conversion.
/// @param actual Finite numerical values.
/// @param reference Finite comparison values of identical length.
/// @return Finite maximum absolute difference, or zero for two empty profiles.
/// @throws std::invalid_argument If shapes differ or any value is nonfinite.
/// @throws std::overflow_error If the maximum difference does not fit double.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::heat_profile_error({1,2},{1,2.5})); // 0.5
/// @endcode
/// @see heat_sine_mode()
double heat_profile_error(const std::vector<double>& actual,const std::vector<double>& reference);
} // namespace datamunge::algorithms
