#pragma once
/// @file runge_kutta4.hpp
/// @brief Checked classical four-stage RK4 and actual stage-state/derivative capture.
#include <cstddef>
#include <datamunge/algorithms/ode_types.hpp>
#include <vector>
/// @brief Canonical first-order ODE integration.
namespace datamunge::algorithms {
/// @brief Complete intrinsic trajectory and optional actual-stage policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RK4Options value{};
/// std::print("{}\n",value.max_steps); // 1000000
/// @endcode
struct RK4Options {
  std::size_t max_steps{1000000};            ///< Requested positive step count.
  std::size_t max_dimension{1024};           ///< Complete positive state dimension.
  std::size_t max_solution_entries{1000000}; ///< Intrinsic (N+1)*(d+1) time/state slots.
  std::size_t max_trace_entries{
      1000000}; ///< Actual per-step9*d+1 retained state/derivative/header slots.
};
/// @brief One actual completed classical RK4 step and its four callback inputs/outputs.
/// @par Example
/// @code
/// const dm::algorithms::RK4Step value{};
/// std::print("{}\n",value.index); // 0
/// @endcode
struct RK4Step {
  std::size_t         index{};         ///< Actual zero-based step.
  double              time{};          ///< Actual initial grid time.
  double              midpoint_time{}; ///< Actual time passed to both midpoint callbacks.
  double              endpoint_time{}; ///< Actual time passed to the fourth callback.
  std::vector<double> before;          ///< Actual initial state passed to stage1.
  std::vector<double> second;          ///< Actual second callback state.
  std::vector<double> third;           ///< Actual third callback state.
  std::vector<double> fourth;          ///< Actual fourth callback state.
  std::vector<double> k1;              ///< Actual first callback output.
  std::vector<double> k2;              ///< Actual second callback output.
  std::vector<double> k3;              ///< Actual third callback output.
  std::vector<double> k4;              ///< Actual fourth callback output.
  std::vector<double> after;           ///< Actual committed weighted state.
};
/// @brief Owned intrinsic trajectory and actual selected stage work.
/// @par Example
/// @code
/// const dm::algorithms::RK4Run value{};
/// std::print("{}\n",value.rhs_calls); // 0
/// @endcode
struct RK4Run {
  OdeSolution solution;    ///< Actual N+1 time/state samples.
  double      step_size{}; ///< Actual finite nominal uniform step, signed for reverse integration.
  std::size_t rhs_calls{}; ///< Actual derivative callback invocations;4*N on success.
  std::size_t scalar_updates{}; ///< Actual committed state components;N*d on success.
  std::vector<RK4Step> steps;   ///< Actual optional completed-stage records.
};
/// @brief Integrate a checked classical four-stage RK4 trajectory.
/// Validates nonempty callback, positive steps/dimension, finite initial state/endpoints, complete solution geometry and time grid before invoking user code. Equal endpoints are legal with zero step and N actual repeated RK4 steps; reverse integration is supported. A nonzero interval must have finite nonzero h and a strictly advancing finite grid; final stored endpoint is exactly t1. For ordinary finite arithmetic preserve historical k1/k2/k3/k4 ordering and the weighted double update. Every returned derivative must have the original dimension and finite values; all intermediate stage states and committed values must remain finite. No stability/error estimate is implied. O(N*d+actual callback work), O(N*d) intrinsic storage plus optional9*d+1 entries per step, admitted before any step callback. Callback is synchronous and not retained; its exceptions propagate and concurrency follows its captures. Outputs owned, input state copied.
/// @param f Borrowed synchronous derivative callback.
/// @param y0 Owned initial finite state, positive dimension.
/// @param t0 Finite initial time.
/// @param t1 Finite final time, either direction or equal.
/// @param steps Positive integral uniform step count.
/// @param options Complete intrinsic and optional stage capture policies.
/// @return Owned actual trajectory, nominal h, selected callback/update counts and optional stages.
/// @throws std::invalid_argument Invalid input controls, callback output shape or nonfinite derivative.
/// @throws std::length_error Complete geometry or execution/history policy exceeded.
/// @throws std::overflow_error Nonfinite time, stage or update arithmetic.
/// @throws std::underflow_error Nonzero interval has a vanishing step or nonadvancing time grid.
/// @par Example
/// @code
/// auto r=dm::algorithms::runge_kutta4_run([](double,const auto& y){return y;},{1},0,1,1);
/// std::print("{} {}\n",r.rhs_calls,r.solution.y.back()[0]); // 4 2.708333333333333
/// @endcode
/// @par References
/// - Kutta (1901), Beitrag zur näherungsweisen Integration totaler Differentialgleichungen, Zeitschrift für Mathematik und Physik46,435–453.
/// @see runge_kutta4(), euler_method()
RK4Run runge_kutta4_run(
    const OdeSystem&    f,
    std::vector<double> y0,
    double              t0,
    double              t1,
    int                 steps,
    const RK4Options&   options = {});
/// @brief Integrate a checked classical four-stage RK4 trajectory with actual callback histories.
/// Validates nonempty callback, positive steps/dimension, finite initial state/endpoints, complete solution geometry and time grid before invoking user code. Equal endpoints are legal with zero step and N actual repeated RK4 steps; reverse integration is supported. A nonzero interval must have finite nonzero h and a strictly advancing finite grid; final stored endpoint is exactly t1. For ordinary finite arithmetic preserve historical k1/k2/k3/k4 ordering and the weighted double update. Every returned derivative must have the original dimension and finite values; all intermediate stage states and committed values must remain finite. No stability/error estimate is implied. O(N*d+actual callback work), O(N*d) intrinsic storage plus optional9*d+1 entries per step, admitted before any step callback. Callback is synchronous and not retained; its exceptions propagate and concurrency follows its captures. Outputs owned, input state copied.
/// @param f Borrowed synchronous derivative callback.
/// @param y0 Owned initial finite state, positive dimension.
/// @param t0 Finite initial time.
/// @param t1 Finite final time, either direction or equal.
/// @param steps Positive integral uniform step count.
/// @param options Complete intrinsic and optional stage capture policies.
/// @return Owned actual trajectory, nominal h, selected callback/update counts and optional stages.
/// @throws std::invalid_argument Invalid input controls, callback output shape or nonfinite derivative.
/// @throws std::length_error Complete geometry or execution/history policy exceeded.
/// @throws std::overflow_error Nonfinite time, stage or update arithmetic.
/// @throws std::underflow_error Nonzero interval has a vanishing step or nonadvancing time grid.
/// @par Example
/// @code
/// auto r=dm::algorithms::runge_kutta4_trace([](double,const auto& y){return y;},{1},0,1,1);
/// std::print("{} {}\n",r.rhs_calls,r.solution.y.back()[0]); // 4 2.708333333333333
/// @endcode
/// @par References
/// - Kutta (1901), Beitrag zur näherungsweisen Integration totaler Differentialgleichungen, Zeitschrift für Mathematik und Physik46,435–453.
/// @see runge_kutta4(), euler_method()
RK4Run runge_kutta4_trace(
    const OdeSystem&    f,
    std::vector<double> y0,
    double              t0,
    double              t1,
    int                 steps,
    const RK4Options&   options = {});
/// @brief Return checked classical RK4 samples through the preserved signature.
/// Delegates to the same producer/default geometry bounds. Callback shape/value and invalid step checks replace unchecked indexing/division. Produces N+1 actual time/state samples and no optional stage copies.
/// @param f Synchronous derivative callback.
/// @param y0 Owned initial positive-dimensional finite state.
/// @param t0 Finite initial time.
/// @param t1 Finite final time.
/// @param steps Positive uniform step count.
/// @return Owned actual sampled trajectory.
/// @throws std::invalid_argument Invalid inputs or callback result.
/// @throws std::length_error Default geometry policy exceeded.
/// @throws std::overflow_error Nonfinite arithmetic.
/// @throws std::underflow_error Unresolved time advancement.
/// @par Example
/// @code
/// auto s=dm::algorithms::runge_kutta4([](double,const auto& y){return y;},{1},0,1,1);
/// std::print("{}\n",s.y.size()); // 2
/// @endcode
/// @see runge_kutta4_run()
OdeSolution
runge_kutta4(const OdeSystem& f, std::vector<double> y0, double t0, double t1, int steps);
} // namespace datamunge::algorithms
