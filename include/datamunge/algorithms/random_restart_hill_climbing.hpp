#pragma once
/// @file random_restart_hill_climbing.hpp
/// @brief Bounded native coordinate hill climbing with actual restart/proposal histories.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>
/// @brief Canonical optimizers and owned execution records.
namespace datamunge::algorithms {
/// @brief Borrowed-call objective signature; returned values must be finite.
/// @par Example
/// @code
/// dm::algorithms::HillObjective f=[](const auto& x){return x[0];};
/// std::print("{}\n",f({2})); // 2
/// @endcode
using HillObjective = std::function<double(const std::vector<double>&)>;
/// @brief Legacy owned best point/value layout.
/// @par Example
/// @code
/// const dm::algorithms::HillClimbResult value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct HillClimbResult {
  std::vector<double> x;        ///< Best visited point, empty when zero restarts run.
  double              value{0}; ///< Best visited finite objective, zero when no restart runs.
};
/// @brief Native coordinate-search and selected-execution policies.
/// @par Example
/// @code
/// const dm::algorithms::HillClimbOptions value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct HillClimbOptions {
  std::size_t   restarts{50}; ///< Actual random starts to visit, including zero.
  std::uint64_t seed{1};      ///< Historical xorshift seed; zero maps to the golden-ratio seed.
  double step_fraction{0.1};  ///< Positive finite fraction of maximum box width for initial step.
  double minimum_step{1e-9};  ///< Positive finite stopping step threshold.
  std::size_t max_evaluations{1000000}; ///< Actual objective calls; zero disables.
  std::size_t max_trace_entries{
      1000000}; ///< Actual three-vector proposal scalars plus headers; zero disables.
  std::size_t max_restarts{100000}; ///< Input restart policy; zero disables.
  std::size_t max_dimensions{1024}; ///< Complete dimension policy; zero disables.
};
/// @brief One actual initial evaluation or signed coordinate proposal.
/// @par Example
/// @code
/// const dm::algorithms::HillClimbStep value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct HillClimbStep {
  std::size_t restart{};    ///< Zero-based restart id.
  std::size_t coordinate{}; ///< Proposal coordinate; equals dimension for an initial sample.
  double      step{};       ///< Actual nonnegative current coordinate step magnitude.
  bool        initial{};    ///< Whether this is a real initial random-point evaluation.
  bool        evaluated{};  ///< Whether the actual point passed bounds and invoked the objective.
  bool
      accepted{}; ///< Whether the native state adopted this proposal (always true for initial state).
  std::vector<double>
      before; ///< Actual state before the proposal, initial point for initial records.
  std::vector<double>
      candidate; ///< Actual attempted point; an out-of-bounds proposal may contain infinity.
  std::vector<double> after; ///< Actual state after accept/reject.
  double value_before{};    ///< Actual prior objective; equal to initial value for initial records.
  double candidate_value{}; ///< Actual objective return, zero when evaluated is false.
  double value_after{};     ///< Actual retained finite objective.
};
/// @brief Owned best value and actual selected search work.
/// @par Example
/// @code
/// const dm::algorithms::HillClimbRun value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct HillClimbRun {
  HillClimbResult best;             ///< Best actually visited restart result.
  bool            found{};          ///< At least one initial point was evaluated.
  std::size_t     evaluations{};    ///< Actual objective calls, including initial samples.
  std::size_t     proposals{};      ///< Actual signed coordinate proposals, including bound skips.
  std::size_t     accepted{};       ///< Actual strictly improving proposals accepted.
  std::size_t     shrinks{};        ///< Actual full-sweep no-improvement step halvings.
  std::vector<HillClimbStep> steps; ///< Actual optional initial/proposal history.
};
/// @brief Maximize an objective by the shared bounded random-restart coordinate producer.
/// Preserves the historical xorshift stream, coordinate order and +step then-step proposals, accepting strictly improving finite returns immediately. A sweep with no improvement halves the step. Validates complete finite nonempty matching bounds with lower<=upper, finite positive step controls and a nonempty callback before any call. Fixed coordinates are allowed. Zero restarts returns found=false with the legacy empty/zero best sentinel. Caller inputs unchanged; callback effects/exceptions propagate and cannot be rolled back. Search is heuristic, with no global-optimum guarantee; time scales with actual proposals/evaluations and callback cost, O(d) working storage.
/// @param objective Borrowed synchronous callable returning a finite objective.
/// @param lower Finite inclusive lower box bounds.
/// @param upper Finite inclusive upper bounds, matching lower.
/// @param options Restart, step and selected-execution policies.
/// @return Best actually visited point/value and actual selected search counts.
/// @throws std::invalid_argument Invalid bounds, step policy or empty callback.
/// @throws std::length_error Input/evaluation/counter policy exceeded.
/// @throws std::overflow_error Nonfinite objective return or initial step.
/// @par Example
/// @code
/// auto r=dm::algorithms::hill_climb_run([](const auto& x){return -x[0]*x[0];},{0},{0});
/// std::print("{} {}\n",r.found,r.best.value); // true -0
/// @endcode
/// @see hill_climb_trace(), random_restart_hill_climbing()
HillClimbRun hill_climb_run(
    const HillObjective&       objective,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    const HillClimbOptions&    options = {});
/// @brief Capture actual initial points, attempted proposals and accepted/rejected states.
/// Same single producer as hill_climb_run(); each record costs3*d+1 scalar/header entries, admitted before copying or invoking that proposal. Skipped out-of-bounds proposals are explicit evaluated=false records, with no invented objective result. History is independently bounded; actual step shrinks are counted rather than replayed.
/// @param objective Borrowed finite-return callback.
/// @param lower Inclusive finite lower bounds.
/// @param upper Inclusive matching finite upper bounds.
/// @param options Search and actual-history policies.
/// @return Owned actual search histories, best visited point and selected counts.
/// @throws std::invalid_argument Invalid domain/callback/step policy.
/// @throws std::length_error Any execution/history capacity exceeded.
/// @throws std::overflow_error Nonfinite objective/initial step.
/// @par Example
/// @code
/// dm::algorithms::HillClimbOptions o;o.restarts=1;
/// auto r=dm::algorithms::hill_climb_trace([](const auto&){return 1.;},{2},{2},o);
/// std::print("{}\n",r.steps.size()); // 1
/// @endcode
/// @see hill_climb_run()
HillClimbRun hill_climb_trace(
    const HillObjective&       objective,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    const HillClimbOptions&    options = {});
/// @brief Preserved callable-template entry point using the one canonical native hill-climb producer.
/// Owns the by-value callable exactly as before and borrows it synchronously through a reference-capturing adapter; move-only callables remain usable. Negative restarts now reject. Uses default checked execution policies; use hill_climb_run() to configure them. Returns the historical empty/zero sentinel for zero restarts, not a global certificate.
/// @tparam F Synchronous callable accepting a const vector reference and returning double.
/// @param f Owned objective callable.
/// @param lo Finite lower bounds.
/// @param hi Matching finite upper bounds.
/// @param restarts Nonnegative actual starts, default50.
/// @param seed Historical xorshift seed, default1.
/// @return Best actually visited point/value.
/// @throws std::invalid_argument Invalid domain, restart count or callback.
/// @throws std::length_error Default execution policy exceeded.
/// @throws std::overflow_error Nonfinite objective/step.
/// @par Example
/// @code
/// auto r=dm::algorithms::random_restart_hill_climbing([](const auto&){return 2.;},{1},{1},1);
/// std::print("{}\n",r.value); // 2
/// @endcode
/// @see hill_climb_run()
template<class F>
HillClimbResult random_restart_hill_climbing(
    F                          f,
    const std::vector<double>& lo,
    const std::vector<double>& hi,
    int                        restarts = 50,
    std::uint64_t              seed     = 1) {
  if (restarts < 0)
    throw std::invalid_argument("hill climb: negative restarts");
  HillClimbOptions o;
  o.restarts                    = static_cast<std::size_t>(restarts);
  o.seed                        = seed;
  const HillObjective objective = [&](const std::vector<double>& x) { return f(x); };
  return hill_climb_run(objective, lo, hi, o).best;
}
/// @brief Owned samples of a one-dimensional objective and its largest visited grid value.
/// @par Example
/// @code
/// const dm::algorithms::HillGrid value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct HillGrid {
  std::vector<double> x;      ///< Actual sampled grid coordinates.
  std::vector<double> values; ///< Actual finite callback values.
  HillClimbResult maximum; ///< Largest visited grid value, not an exact global optimum certificate.
};
/// @brief Evaluate the retained original one-dimensional multimodal objective.
/// @param x Exactly one finite coordinate.
/// @return sin(x)+sin(2.2*x)+0.3*x.
/// @throws std::invalid_argument Wrong dimension or nonfinite coordinate.
/// @throws std::overflow_error Nonfinite scaled argument/result.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hill_wave_objective({0})); // 0
/// @endcode
/// @see hill_climb_run(), hill_objective_grid()
double hill_wave_objective(const std::vector<double>& x);
/// @brief Sample a finite one-dimensional objective by actual cumulative grid steps.
/// Retains the original report repeated-addition grid, including both bounds when reached. Rejects a step that cannot advance its stored coordinate; maximum is a grid-only reference, not a global proof. O(points) callback work and owned storage.
/// @param objective Borrowed finite-return callable accepting a one-element vector.
/// @param lower Finite inclusive lower coordinate.
/// @param upper Finite inclusive upper coordinate, at least lower.
/// @param step Finite positive increment.
/// @param max_points Actual retained-point bound, default1000000; zero disables.
/// @return Actual sampled curve and best visited grid point.
/// @throws std::invalid_argument Invalid bounds/step/callback.
/// @throws std::length_error Retained point bound exceeded.
/// @throws std::overflow_error Nonfinite return or nonadvancing grid.
/// @par Example
/// @code
/// auto g=dm::algorithms::hill_objective_grid(dm::algorithms::hill_wave_objective,0,1,.5);
/// std::print("{}\n",g.x.size()); // 3
/// @endcode
/// @see hill_wave_objective()
HillGrid hill_objective_grid(
    const HillObjective& objective,
    double               lower,
    double               upper,
    double               step,
    std::size_t          max_points = 1000000);
} // namespace datamunge::algorithms
