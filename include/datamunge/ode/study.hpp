#pragma once
/// @file study.hpp
/// @brief Native original ODE report fixtures and actual method/error/trajectory summaries.
#include <datamunge/ode/ode_solver.hpp>
#include <string>
#include <vector>
namespace datamunge::ode {
/// @brief Owned actual original-method decay solution and closed-form final error.
/// @par Example
/// @code
/// dm::ode::ODEAccuracyRow r;std::print("{}\n",r.absolute_error); // 0
/// @endcode
/// @see ode_decay_comparison()
struct ODEAccuracyRow {
  std::string name;              ///< Owned compact original-method display name.
  ODEOptions  options;           ///< Actual original numerical controls used.
  ODESolution solution;          ///< Actual complete native forward decay trajectory, no replay.
  double      exact{0};          ///< std::exp(-2) reference for y(0)=1,k=1.
  double      absolute_error{0}; ///< Actual absolute difference of last state from reference.
};
/// @brief Actual original fixed-step Euler/RK4 decay convergence row.
/// @par Example
/// @code
/// dm::ode::ODEOrderRow r;std::print("{}\n",r.step); // 0
/// @endcode
/// @see ode_decay_comparison()
struct ODEOrderRow {
  double      step{0};        ///< Original nominal fixed step.
  double      euler_error{0}; ///< Actual Euler absolute final error versus exp(-2).
  double      rk4_error{0};   ///< Actual classical RK4 absolute final error versus exp(-2).
  std::size_t euler_steps{
      0};                   ///< Actual accepted Euler steps, including a representable final tail.
  std::size_t rk4_steps{0}; ///< Actual accepted RK4 steps, including a representable final tail.
  std::size_t euler_evaluations{0}; ///< Actual Euler derivative calls.
  std::size_t rk4_evaluations{0};   ///< Actual RK4 derivative calls.
};
/// @brief Owned actual original harmonic trajectory and sampled closed-form/energy diagnostics.
/// @par Example
/// @code
/// dm::ode::ODEOscillatorStudy s;std::print("{}\n",s.energy_drift.size()); // 0
/// @endcode
/// @see ode_decay_comparison()
struct ODEOscillatorStudy {
  ODESolution         solution;       ///< Actual omega1,RK4h.01,[1,0],t0..20 native solve.
  std::vector<double> position;       ///< Actual first state component per retained time.
  std::vector<double> velocity;       ///< Actual second state component per retained time.
  std::vector<double> exact_position; ///< Actual cos(t) reference at returned sample times.
  std::vector<double> exact_velocity; ///< Actual-sin(t) reference at returned sample times.
  std::vector<double>
      energy_drift; ///< Actual x*x+v*v-1 at returned times; scaled energy,not exact preservation.
  double final_energy{0};     ///< Actual final x*x+v*v, expectednear1 but not forced.
  double max_energy_drift{0}; ///< Actual max absolute sampled scaled-energy departure.
  double max_state_error{0};  ///< Actual max sampled hypot(x-cos(t),v+sin(t)).
};
/// @brief Owned actual original full-transient Lorenz solution and x-z presentation coordinates.
/// @par Example
/// @code
/// dm::ode::ODELorenzStudy s;std::print("{}\n",s.x.size()); // 0
/// @endcode
/// @see ode_decay_comparison()
struct ODELorenzStudy {
  ODESolution         solution; ///< Actual sigma10,rho28,beta8/3,RK4h.005,[1,1,1],t0..25 solve.
  std::vector<double> x;        ///< Actual first state component for every returned sample.
  std::vector<double> z;        ///< Actual third state component for every returned sample.
};
/// @brief Run all six original decay comparison configurations through canonical builtins.
/// Euler/midpoint/RK4/AB4/PECE4 use h.01; RK45 uses initialh.1,abs/rel1e-8.
/// Every solve uses y0=1,k=1,t0..2 and original default actual-work policies.
/// Work/storage follow actual native solver trajectories; no caller data/references,
/// callbacks, random draws or shared mutable state. Default policies apply.
/// @return Owned actual std::vector<ODEAccuracyRow>.
/// @throws std::length_error If actual solver work/output policies are exceeded.
/// @throws std::overflow_error If actual stage/state/time arithmetic fails.
/// @throws std::bad_alloc If owned solver/study storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::ode::ode_decay_comparison().size()); // 6
/// @endcode
/// @see ODESolver, ODETrace
std::vector<ODEAccuracyRow> ode_decay_comparison();
/// @brief Run the original six-step-size Euler/RK4 final-error comparison.
/// Steps.1,.05,.02,.01,.005,.002 retain original order; decay y0=1,k=1,t0..2.
/// Counts and errors derive from each actual native solve, not report-side recurrences.
/// Work/storage follow actual native solver trajectories; no caller data/references,
/// callbacks, random draws or shared mutable state. Default policies apply.
/// @return Owned actual std::vector<ODEOrderRow>.
/// @throws std::length_error If actual solver work/output policies are exceeded.
/// @throws std::overflow_error If actual stage/state/time arithmetic fails.
/// @throws std::bad_alloc If owned solver/study storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::ode::ode_decay_step_study().size()); // 6
/// @endcode
/// @see ODESolver, ODETrace
std::vector<ODEOrderRow> ode_decay_step_study();
/// @brief Run the original omega1 harmonic oscillator over0..20 with RK4h.01.
/// Returns actual states/projections, sampled cos/sin references and scaled energy drift.
/// This is a numerical diagnostic, not a symplectic/energy-preservation guarantee.
/// Work/storage follow actual native solver trajectories; no caller data/references,
/// callbacks, random draws or shared mutable state. Default policies apply.
/// @return Owned actual ODEOscillatorStudy.
/// @throws std::length_error If actual solver work/output policies are exceeded.
/// @throws std::overflow_error If actual stage/state/time arithmetic fails.
/// @throws std::bad_alloc If owned solver/study storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::ode::ode_harmonic_study().solution.t.back()); // 20
/// @endcode
/// @see ODESolver, ODETrace
ODEOscillatorStudy ode_harmonic_study();
/// @brief Run the original Lorenz parameters/initial state over0..25 with RK4h.005.
/// Retains the complete initial transient and all actual x/z projections.
/// A finite phase portrait is not certified long-time pointwise accuracy or attractor proof.
/// Work/storage follow actual native solver trajectories; no caller data/references,
/// callbacks, random draws or shared mutable state. Default policies apply.
/// @return Owned actual ODELorenzStudy.
/// @throws std::length_error If actual solver work/output policies are exceeded.
/// @throws std::overflow_error If actual stage/state/time arithmetic fails.
/// @throws std::bad_alloc If owned solver/study storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::ode::ode_lorenz_study().solution.t.back()); // 25
/// @endcode
/// @see ODESolver, ODETrace
ODELorenzStudy ode_lorenz_study();
} // namespace datamunge::ode
