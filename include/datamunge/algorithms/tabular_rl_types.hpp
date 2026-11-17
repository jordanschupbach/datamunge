#pragma once
/// @file tabular_rl_types.hpp
/// @brief Shared owned tabular learning values and borrowed synchronous environment callbacks.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
namespace datamunge::algorithms {
/// @brief One actual environment transition.
/// The consuming solver defines index/reward validation and terminal bootstrapping; q_learning validates even terminal next-state ids.
/// @par Example
/// @code
/// const dm::algorithms::StepResult s{1,-1,true};std::print("{}\n",s.terminal); // true
/// @endcode
struct StepResult {
  std::size_t next_state; ///< Actual returned state id.
  double      reward;     ///< Actual immediate reward in environment units.
  bool        terminal;   ///< Whether this transition terminates the episode.
};
/// @brief Finite tabular geometry and synchronous state-transition callbacks.
/// Callbacks may mutate caller-owned environment state and throw; solvers do not roll back external side effects. Caller ensures callback lifetimes and concurrency safety for the duration of each call.
/// @par Example
/// @code
/// dm::algorithms::TabularEnvironment e{2,1,[]{return std::size_t{0};},[](std::size_t,std::size_t){return dm::algorithms::StepResult{1,2,true};}};std::print("{}\n",e.reset()); // 0
/// @endcode
struct TabularEnvironment {
  std::size_t num_states{0};          ///< State count; Q-learning requires positive.
  std::size_t num_actions{0};         ///< Per-state action count; Q-learning requires positive.
  std::function<std::size_t()> reset; ///< Borrowed synchronous callback returning an initial state.
  std::function<StepResult(std::size_t, std::size_t)>
      step; ///< Borrowed synchronous callback of current state/action.
};
/// @brief Shared tabular training parameters; validation follows the consuming solver.
/// Q-learning requires learning_rate in(0,1], discount/epsilon in[0,1], all finite. Counts may be zero; seed controls the solver behavior RNG, not callback randomness.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::RLParameters{}.episodes); // 500
/// @endcode
struct RLParameters {
  double        learning_rate{0.1}; ///< Additive temporal-difference step size, default0.1.
  double        discount{0.99};     ///< Future reward discount, default0.99.
  double        epsilon{0.1};       ///< Constant exploration probability, default0.1.
  std::size_t   episodes{500};      ///< Requested training resets/episodes, default500.
  std::size_t   max_steps{1000};    ///< Per-episode transition cap; zero still permits resets.
  std::uint64_t seed{0};            ///< Local behavior-generator seed, default0.
};
/// @brief Owned tabular control output shared by Q-learning and SARSA.
/// Greedy ties choose the lowest action index. Returns are actual undiscounted sums during training; finite behavior returns do not certify an optimal policy.
/// @par Example
/// @code
/// dm::algorithms::QLearningResult r;std::print("{}\n",r.q.empty()); // true
/// @endcode
struct QLearningResult {
  std::vector<std::vector<double>> q;       ///< Actual final state/action table.
  std::vector<std::size_t> greedy_policy;   ///< Actual lowest-index argmax per final table row.
  std::vector<double>      episode_returns; ///< Actual undiscounted training episode returns.
};
/// @brief Owned TD prediction values and actual episode returns.
/// Validation and policy semantics follow td_prediction().
/// @par Example
/// @code
/// dm::algorithms::TDPredictionResult r;std::print("{}\n",r.value.empty()); // true
/// @endcode
struct TDPredictionResult {
  std::vector<double> value; ///< Owned estimated state values.
  std::vector<double>
      episode_returns; ///< Actual undiscounted returns collected by the consuming solver.
};
} // namespace datamunge::algorithms
