#pragma once
/// @file discrete.hpp
/// @brief Native masked, variable-duration discrete reinforcement-learning pipelines.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/// @brief Native variable-duration, masked discrete reinforcement learning.
namespace datamunge::rl {
/// @brief Discrete controllers sharing observation, masking, discount and checkpoint contracts.
/// @par Example
/// @code
/// auto method=dm::rl::DiscreteMethod::DoubleDqn;
/// @endcode
enum class DiscreteMethod {
  QLambda,          ///< Tabular off-policy Q-learning with replacing Watkins eligibility traces.
  DoubleDqn,        ///< Replay/target-network Double DQN with a two-layer ReLU MLP.
  DuelingDoubleDqn, ///< Double DQN with shared features and separate value/advantage heads.
  Ppo               ///< Clipped categorical PPO with a shared tanh trunk, value head and GAE.
};
/// @brief Owned observation with finite features, a binary feasibility mask and optional table row.
/// Masks have exactly model.actions entries and at least one allowed action,
/// except a true terminal successor may have none. No true class/future state is required.
/// @par Example
/// @code
/// dm::rl::DiscreteObservation observation{{.2f, .5f},{1,1},0};
/// @endcode
struct DiscreteObservation {
  std::vector<float>         features; ///< Exactly model.features finite CPU float32 values.
  std::vector<unsigned char> allowed;  ///< Exactly model.actions zero/one flags.
  std::size_t state{0}; ///< Table row, below model.states for QLambda; ignored by neural methods.
};
/// @brief One actual environment option, preserving terminal versus time-limit semantics.
/// @par Example
/// @code
/// dm::rl::DiscreteTransition transition;
/// transition.next={{0.f},{1,1},0}; transition.seconds=.125;
/// @endcode
struct DiscreteTransition {
  DiscreteObservation next; ///< Actual successor, including the final observation at truncation.
  double              reward{0};  ///< Finite option reward.
  double              seconds{1}; ///< Finite, strictly positive elapsed time, used in discounting.
  bool                terminated{false}; ///< True terminal: bootstrap is zero.
  bool truncated{false}; ///< Time limit: bootstrap from next, but cut rollout/GAE across reset.
};
/// @brief Borrowed single-writer callbacks; their owners must outlive train_discrete().
/// Neither callback may mutate a policy concurrently. Exceptions propagate after
/// any already completed progress. Reset seeds are derived independently of action draws.
/// @par Example
/// @code
/// dm::rl::DiscreteEnvironment environment;
/// environment.reset=[](std::uint64_t){return dm::rl::DiscreteObservation{{0.f},{1,1},0};};
/// environment.step=[](std::size_t action){return dm::rl::DiscreteTransition{
///   {{0.f},{1,1},0}, action==1?1.:0.,1.,true,false};};
/// @endcode
struct DiscreteEnvironment {
  std::function<DiscreteObservation(std::uint64_t)> reset; ///< Start one independent window.
  std::function<DiscreteTransition(std::size_t)>    step;  ///< Execute an allowed action.
};
/// @brief Architecture and initialization for a native CPU policy.
/// @par Example
/// @code
/// dm::rl::DiscreteModel model; model.method=dm::rl::DiscreteMethod::Ppo;
/// @endcode
struct DiscreteModel {
  DiscreteMethod method{DiscreteMethod::DoubleDqn}; ///< Algorithm/head family.
  std::size_t    features{25};                      ///< Input width, 1..4096.
  std::size_t    actions{5};                        ///< Output width, 2..1024.
  std::size_t    hidden{32};   ///< Width of each of two neural hidden layers, 1..1024.
  std::size_t    states{1680}; ///< Table rows for QLambda, 1..1000000.
  std::uint64_t  seed{17};     ///< Independent parameter initialization seed.
};
/// @brief Owned inference policy; copies require clone(), which never shares learnable storage.
/// Neural scores are native nn tensors internally. Inference is single-writer;
/// optimizer/replay/RNG training state is not part of an inference checkpoint.
/// @par Example
/// @code
/// dm::rl::DiscreteModel model; model.features=1; model.actions=2;
/// dm::rl::DiscretePolicy policy(model);
/// auto action=policy.action({{0.f},{1,0},0}); // always zero
/// @endcode
class DiscretePolicy {
 public:
  /// @brief Initialize one validated model, using an independent parameter RNG.
  /// @param model Architecture and initialization; defaults above.
  /// @throws std::invalid_argument Invalid dimensions or method.
  /// @par Example See the class example.
  explicit DiscretePolicy(DiscreteModel model = {});
  /// @brief Release owned tensors/table values.
  ~DiscretePolicy();
  /// @brief Transfer exclusive policy ownership.
  /// @param other Source policy.
  DiscretePolicy(DiscretePolicy&& other) noexcept;
  /// @brief Transfer ownership, releasing the previous model.
  /// @param other Source policy.
  /// @return This policy.
  DiscretePolicy& operator=(DiscretePolicy&& other) noexcept;
  DiscretePolicy(const DiscretePolicy&) = delete; ///< Use clone() for an independent copy.
  DiscretePolicy& operator=(const DiscretePolicy&) =
      delete; ///< Use clone() for independent values.

  /// @brief Choose the first maximizing allowed Q value or categorical-policy logit.
  /// A forced action avoids a neural forward pass. Does not sample or consume RNGs.
  /// @param observation Valid owned/borrowed features, table row and nonempty mask.
  /// @return Allowed zero-based action index.
  /// @throws std::invalid_argument Invalid observation.
  /// @throws std::runtime_error Nonfinite network scores.
  /// @par Example See the class example.
  std::size_t action(const DiscreteObservation& observation) const;
  /// @brief Read owned unmasked Q values or actor logits without consuming RNGs.
  /// @param observation Valid features, row and nonempty feasibility mask.
  /// @return Exactly model.actions finite scores; caller must still apply allowed.
  /// @throws std::invalid_argument Invalid observation.
  /// @throws std::runtime_error Nonfinite network scores.
  /// @par Example
  /// @code
  /// dm::rl::DiscretePolicy policy;
  /// auto scores=policy.scores({std::vector<float>(25),{1,0,0,0,0},0});
  /// @endcode
  std::vector<float> scores(const DiscreteObservation& observation) const;
  /// @brief Read architecture metadata without borrowing internal mutable tensors.
  /// @return Owned model configuration.
  DiscreteModel model() const;
  /// @brief Deep-copy inference parameters, excluding optimizer/replay/RNG state.
  /// @return Independent policy with identical scores and greedy actions.
  /// @par Example
  /// @code
  /// dm::rl::DiscretePolicy policy; auto copy=policy.clone();
  /// @endcode
  DiscretePolicy clone() const;
  /// @brief Save a versioned NPZ inference checkpoint with architecture and all values.
  /// @param path Writable path whose parent exists.
  /// @throws std::runtime_error I/O failure.
  /// @par Example
  /// @code
  /// dm::rl::DiscretePolicy policy; policy.save("policy.npz");
  /// auto restored=dm::rl::DiscretePolicy::load("policy.npz");
  /// @endcode
  void save(const std::string& path) const;
  /// @brief Restore a strict finite inference checkpoint; never enables training resumption.
  /// @param path Existing checkpoint saved by save().
  /// @return Independently owned model.
  /// @throws std::runtime_error Invalid/missing/versioned tensors or I/O failure.
  /// @par Example See save().
  static DiscretePolicy load(const std::string& path);

 private:
  struct Impl;                   ///< Native parameter store or table.
  std::unique_ptr<Impl> impl_;   ///< Exclusive ownership.
  friend struct DiscreteTrainer; ///< Library-owned learning loop.
};
/// @brief Complete measured work for one independent training window.
/// @par Example
/// @code
/// dm::rl::DiscreteWindow window;
/// @endcode
struct DiscreteWindow {
  double seconds{0}; ///< Actual elapsed simulation time; a final option may overshoot the budget.
  double reward{0};  ///< Undiscounted sum of original environment rewards.
  std::size_t decisions{0};         ///< Composite learning decisions.
  std::size_t environment_steps{0}; ///< Original callback steps, including forced waits.
  std::size_t updates{0};           ///< Gradient/TD update count during the window.
  double      loss{0}; ///< Last finite neural loss, or zero before an update/for QLambda.
};
/// @brief Matched-window controls for tabular, replay-based and on-policy learning.
/// Checkpoints occur at powers of two from window 128 and at completion. The
/// trainer bootstraps time-limit/window boundaries, never true terminal states.
/// Forced-action chains can be coalesced with exact composed reward/discount;
/// every underlying environment step still runs and sees new arrivals.
/// @par Example
/// @code
/// dm::rl::DiscreteTrainingOptions options; options.windows=2048;
/// options.model.method=dm::rl::DiscreteMethod::DuelingDoubleDqn;
/// @endcode
struct DiscreteTrainingOptions {
  DiscreteModel model;                     ///< Model family, dimensions and parameter seed.
  std::size_t   windows{2048};             ///< Positive number of independent windows.
  double        window_seconds{120};       ///< Positive minimum time per nonterminal window.
  double        discount_per_second{.995}; ///< Time discount in (0,1].
  double        learning_rate{.0003};      ///< Positive Adam rate for neural models.
  double        table_learning_rate{.08};  ///< QLambda TD update fraction in (0,1].
  double        epsilon_start{.35};        ///< Initial replay/table epsilon in [0,1].
  double        epsilon_end{.01};          ///< Final replay/table epsilon in [0,1].
  double        trace_decay{.8}; ///< QLambda replacing trace decay per composite decision, [0,1].
  std::size_t   batch{64};       ///< Positive replay/PPO minibatch size.
  std::size_t   replay_capacity{32768}; ///< Replay slots, at least batch.
  std::size_t   warmup{512}; ///< Collected replay transitions before updates, at least batch and at
                             ///< most replay_capacity.
  std::size_t train_every{8};    ///< Replay decisions per update, positive.
  std::size_t target_every{128}; ///< Gradient updates between hard target-network copies.
  std::size_t rollout{256};      ///< Maximum PPO rollout length before an update, positive.
  std::size_t epochs{4};         ///< PPO shuffled passes per rollout, positive.
  double      clip{.2};          ///< PPO ratio clipping radius in (0,1).
  double      gae_lambda{.95};   ///< PPO GAE decay per composite decision, [0,1].
  double      entropy_cost{.01}; ///< Nonnegative categorical entropy bonus coefficient.
  double      value_cost{.5};    ///< Nonnegative PPO squared value-loss coefficient.
  double      gradient_clip{10}; ///< Positive global gradient-norm limit.
  std::size_t forced_chain{8};  ///< Maximum original steps coalesced when each has only one choice.
  std::uint64_t seed{20261121}; ///< Training RNG/reset seed; independent of model initialization.
  std::function<void(std::size_t, const DiscreteWindow&, const DiscretePolicy&)> progress;
  ///< Optional callback after each window; policy is borrowed only for that call and must not be
  ///< mutated.
  std::function<bool(std::size_t, const DiscreteWindow&, const DiscretePolicy&)> stop;
  ///< Optional synchronous window-end stopping rule, called after progress; true stops
  ///< between complete windows and retains a final checkpoint. May evaluate a borrowed
  ///< const policy on an independently owned environment; must not mutate/reenter the
  ///< training environment or policy. Callback exceptions propagate without a result.
};
/// @brief Training evidence and independent inference checkpoints, without optimizer resumption.
/// @par Example
/// @code
/// dm::rl::DiscreteTrainingResult result;
/// @endcode
struct DiscreteTrainingResult {
  DiscretePolicy              policy;             ///< Final inference model.
  std::vector<DiscreteWindow> windows;            ///< Every independent window's measured work.
  std::vector<DiscretePolicy> checkpoints;        ///< Independent inference snapshots.
  std::vector<std::size_t>    checkpoint_windows; ///< Completed windows for each snapshot.
};
/// @brief Train masked Q(lambda), Double DQN, dueling Double DQN or clipped PPO in native C++.
/// Uses CPU nn autograd/Adam for neural methods, uniform replay with a hard target
/// network and Huber loss for DQN, and stored behavior log-probabilities, elapsed-
/// time GAE, clipped ratios, value loss and entropy for PPO. Truncations bootstrap
/// from the actual final observation and cut GAE/reset traces. No environment
/// labels or future observations are inferred by the trainer. Single-writer only.
/// @param environment Valid borrowing callbacks returning owned finite observations.
/// @param options Valid model/training controls; defaults above.
/// @return Final model, completed-window histories and independent checkpoints; stop may finish
/// before the maximum budget.
/// @throws std::invalid_argument Invalid options or callback transitions.
/// @throws std::runtime_error Nonfinite scores/loss/gradients or callback failure.
/// @par Example
/// @code
/// dm::rl::DiscreteEnvironment env;
/// env.reset=[](std::uint64_t){return dm::rl::DiscreteObservation{{0.f},{1,1},0};};
/// env.step=[](std::size_t a){return
/// dm::rl::DiscreteTransition{{{0.f},{1,1},0},a?1.:0.,1.,true,false};};
/// dm::rl::DiscreteTrainingOptions o; o.model.features=1;o.model.actions=2;o.windows=4;
/// auto trained=dm::rl::train_discrete(env,o);
/// @endcode
/// @par References
/// - van Hasselt, Guez and Silver (2016), Deep Reinforcement Learning with Double Q-learning:
///   https://arxiv.org/abs/1509.06461.
/// - Wang et al. (2016), Dueling Network Architectures for Deep Reinforcement Learning:
///   https://arxiv.org/abs/1511.06581.
/// - Schulman et al. (2017), Proximal Policy Optimization Algorithms:
/// https://arxiv.org/abs/1707.06347.
/// @see DiscretePolicy, DiscreteTrainingOptions
DiscreteTrainingResult train_discrete(
    DiscreteEnvironment     environment,
    DiscreteTrainingOptions options = {});
} // namespace datamunge::rl
