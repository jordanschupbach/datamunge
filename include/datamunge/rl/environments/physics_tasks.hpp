#pragma once
/// @file physics_tasks.hpp
/// @brief Owning single/vector environment views over canonical native TaskBatch physics.
#include <algorithm>
#include <cmath>
#include <datamunge/rl/environments/vector.hpp>
#include <datamunge/sim/tasks.hpp>
#include <limits>
namespace datamunge::rl::environments {
/// @brief Resolve one authored native task name without silently selecting a fallback.
///
/// @param name Exactly inverted_pendulum or hopper.
/// @return Corresponding canonical Task value.
/// @throws std::invalid_argument The name is unsupported.
/// @par Example
/// @code
/// auto task=dm::rl::environments::NativeTask("hopper");
/// @endcode
/// @see sim::Task, NativeTaskEnv
inline datamunge::sim::Task NativeTask(const std::string& name) {
  if (name == "inverted_pendulum")
    return datamunge::sim::Task::InvertedPendulum;
  if (name == "hopper")
    return datamunge::sim::Task::Hopper;
  throw std::invalid_argument("task must be inverted_pendulum or hopper");
}
/// @brief Read authored task action bounds in native physical units.
///
/// @param task InvertedPendulum or Hopper.
/// @return Owned continuous descriptor: force[-3,3], or three torques[-1,1].
/// @throws std::invalid_argument Unsupported task enum.
/// @par Example
/// @code
/// auto space=dm::rl::environments::NativeSpace(dm::sim::Task::Hopper);
/// @endcode
/// @see sim::Task
inline ActionSpace NativeSpace(datamunge::sim::Task task) {
  if (task != datamunge::sim::Task::InvertedPendulum && task != datamunge::sim::Task::Hopper)
    throw std::invalid_argument("unsupported native task");
  return task == datamunge::sim::Task::InvertedPendulum
             ? ActionSpace::Continuous({-3}, {3})
             : ActionSpace::Continuous({-1, -1, -1}, {1, 1, 1});
}
/// @brief Copy one native double observation row into Float32 environment storage.
///
/// @param values Borrowed flattened row-major native observations.
/// @param row Zero-based row, less than values.size()/width.
/// @param width Positive row width dividing the flattened size.
/// @return Independent Float32 copy of the selected row.
/// @throws std::invalid_argument Width, divisibility or row bounds are invalid.
/// @par Example
/// @code
/// auto obs=dm::rl::environments::NativeObservation({1,2,3,4},1,2);
/// @endcode
/// @see NativeTaskEnv
inline Obs
NativeObservation(const std::vector<double>& values, std::size_t row, std::size_t width) {
  if (!width || row >= values.size() / width || values.size() % width)
    throw std::invalid_argument("native observation row geometry");
  const auto begin = values.begin() + static_cast<std::ptrdiff_t>(row * width);
  return Obs(begin, begin + static_cast<std::ptrdiff_t>(width));
}
/// @brief Own one canonical CPU physical world through the generic Env interface.
///
/// No auto-reset replaces terminal observations. Physics, rewards, termination and continuing RNG
/// belong to TaskBatch, with Float32 copies at this interface. Task definitions are native
/// datamunge protocols, not Jolt/Gym conformance. Calls mutating one instance require
/// synchronization; borrowed simulator output is never exposed.
///
/// @par Example
/// @code
/// dm::rl::environments::NativeTaskEnv env(dm::sim::Task::InvertedPendulum,10);auto
/// obs=env.reset(0);auto next=env.step(dm::rl::environments::Action(std::vector<float>{0}));
/// @endcode
/// @see sim::TaskBatch, NativeVectorTask
class NativeTaskEnv final : public Env {
 public:
  /// @brief Own one native task with a checked time limit.
  ///
  /// @param task Native task definition.
  /// @param limit Positive cap fitting int; default1000.
  /// @throws std::invalid_argument Invalid checked time limit or native simulator configuration.
  /// @par Example
  /// @code
  /// dm::rl::environments::NativeTaskEnv env(dm::sim::Task::InvertedPendulum,10);auto
  /// obs=env.reset(0);auto next=env.step(dm::rl::environments::Action(std::vector<float>{0}));
  /// @endcode
  /// @see sim::TaskBatch
  explicit NativeTaskEnv(datamunge::sim::Task task, std::uint64_t limit = 1000)
      : task_(task),
        limit_(checked_limit(limit)),
        batch_(task, 1, datamunge::compute::Context{}, 0, limit_) {}
  /// @brief Read a name identifying the native authored task.
  ///
  /// @return Native InvertedPendulum or Native Hopper.
  /// @see NativeTaskEnv
  std::string name() const override {
    return task_ == datamunge::sim::Task::InvertedPendulum ? "Native InvertedPendulum"
                                                           : "Native Hopper";
  }
  /// @brief Read simulator observation width.
  ///
  /// @return Number of Float32 coordinates copied per step.
  /// @see NativeTaskEnv
  int obs_size() const override { return static_cast<int>(batch_.observation_size()); }
  /// @brief Read native physical action bounds.
  ///
  /// @return Owning continuous action descriptor.
  /// @see NativeTaskEnv
  ActionSpace action_space() const override { return NativeSpace(task_); }
  /// @brief Advance one real native physical transition without auto-reset.
  ///
  /// @param action Finite continuous physical action with the task width.
  /// @return Owned Float32 observation/reward and source terminal/truncated flags.
  /// @throws std::invalid_argument Action or simulator configuration is invalid.
  /// @par Example
  /// @code
  /// dm::rl::environments::NativeTaskEnv env(dm::sim::Task::InvertedPendulum,10);auto
  /// obs=env.reset(0);auto next=env.step(dm::rl::environments::Action(std::vector<float>{0}));
  /// @endcode
  /// @see sim::TaskBatch::step()
  StepResult step(const Action& action) override {
    ValidateAction(action_space(), action);
    const std::vector<double> values(action.continuous.begin(), action.continuous.end());
    const auto&               out = batch_.step(values);
    return {
        NativeObservation(out.observations, 0, batch_.observation_size()),
        static_cast<float>(out.rewards[0]),
        out.terminated[0] != 0,
        out.truncated[0] != 0};
  }
  /// @brief Restart the native random stream without implicitly resetting state.
  ///
  /// @param seed 64-bit simulator seed.
  /// @see Env::reset()
  void Seed(std::uint64_t seed) override { batch_.seed(seed); }
  /// @brief Read the configured positive time limit.
  ///
  /// @return Episode cap as a signed int.
  /// @see NativeTaskEnv
  int max_steps() const override { return static_cast<int>(limit_); }

 protected:
  /// @brief Reset the native task using its continuing RNG.
  ///
  /// @return Owning initial Float32 observation.
  /// @see NativeTaskEnv
  Obs Reset() override {
    return NativeObservation(batch_.reset().observations, 0, batch_.observation_size());
  }

 private:
  /// @brief Validate a time limit compatible with the generic signed-int accessor.
  ///
  /// @param limit Requested native cap.
  /// @return Unchanged valid limit.
  /// @throws std::invalid_argument Zero or larger than INT_MAX.
  /// @see max_steps()
  static std::uint64_t checked_limit(std::uint64_t limit) {
    if (!limit || limit > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("native example time limit outside int range");
    return limit;
  }
  datamunge::sim::Task      task_;  ///< Authored native task definition.
  std::uint64_t             limit_; ///< Positive episode cap used by shared evaluation helpers.
  datamunge::sim::TaskBatch batch_; ///< Sole physical state and RNG owner.
};
/// @brief Own batched native worlds with masked auto-reset and retained final observations.
///
/// Each step copies independent generic-interface vectors before masked resets.
/// Terminated/truncated flags describe pre-reset transitions; final_obs is populated only for ended
/// rows, while obs is post-reset. Owned simulator/workers, actions and masks are reused. Calls
/// mutating the same instance require synchronization; no collection or learner loop is owned.
///
/// @par Example
/// @code
/// dm::rl::environments::NativeVectorTask env(dm::sim::Task::InvertedPendulum,2,0);auto
/// obs=env.reset();auto
/// next=env.step(std::vector<dm::rl::environments::Action>(2,dm::rl::environments::Action(std::vector<float>{0})));
/// @endcode
/// @see sim::TaskBatch, VecStep
class NativeVectorTask {
 public:
  /// @brief Create independent native CPU worlds and persistent workers.
  ///
  /// @param task Authored task definition.
  /// @param count Positive world count fitting int.
  /// @param seed 64-bit simulator seed for independent world streams.
  /// @param workers Positive worker count, default1.
  /// @param limit Positive simulator time limit, default1000.
  /// @throws std::invalid_argument Nonpositive counts/workers or invalid simulator configuration.
  /// @par Example
  /// @code
  /// dm::rl::environments::NativeVectorTask env(dm::sim::Task::InvertedPendulum,2,0);auto
  /// obs=env.reset();auto
  /// next=env.step(std::vector<dm::rl::environments::Action>(2,dm::rl::environments::Action(std::vector<float>{0})));
  /// @endcode
  /// @see sim::TaskBatch
  NativeVectorTask(
      datamunge::sim::Task task,
      int                  count,
      std::uint64_t        seed,
      int                  workers = 1,
      std::uint64_t        limit   = 1000)
      : task_(task),
        batch_(
            task,
            positive(count),
            datamunge::compute::Context({}, positive(workers)),
            seed,
            limit),
        actions_(batch_.size() * batch_.action_size()),
        reset_(batch_.size()) {}
  /// @brief Read world count.
  ///
  /// @return Positive number of worlds.
  /// @see NativeVectorTask
  int size() const { return static_cast<int>(batch_.size()); }
  /// @brief Read shared observation width.
  ///
  /// @return Coordinates per world.
  /// @see NativeVectorTask
  int obs_size() const { return static_cast<int>(batch_.observation_size()); }
  /// @brief Read physical action bounds shared across worlds.
  ///
  /// @return Owned task action descriptor.
  /// @see NativeVectorTask
  ActionSpace action_space() const { return NativeSpace(task_); }
  /// @brief Explicitly reset all worlds using their continuing RNG streams.
  ///
  /// @return Owned rows of initial Float32 observations.
  /// @par Example
  /// @code
  /// dm::rl::environments::NativeVectorTask env(dm::sim::Task::InvertedPendulum,2,0);auto
  /// obs=env.reset();auto
  /// next=env.step(std::vector<dm::rl::environments::Action>(2,dm::rl::environments::Action(std::vector<float>{0})));
  /// @endcode
  /// @see NativeVectorTask
  std::vector<Obs> reset() {
    batch_.reset();
    return observations();
  }
  /// @brief Advance all worlds, copy final states and mask-reset ended rows.
  ///
  /// Validates all action rows before simulation. Native stepping may propagate simulator errors.
  /// The caller uses final_obs for time-limit bootstrap and cuts traces at either boundary.
  ///
  /// @param actions One finite physical continuous action with correct width per world.
  /// @return Owned vector transition with retained final_obs on ended rows and post-reset obs.
  /// @throws std::invalid_argument Invalid action row count, width, values or simulator input.
  /// @par Example
  /// @code
  /// dm::rl::environments::NativeVectorTask env(dm::sim::Task::InvertedPendulum,2,0);auto
  /// obs=env.reset();auto
  /// next=env.step(std::vector<dm::rl::environments::Action>(2,dm::rl::environments::Action(std::vector<float>{0})));
  /// @endcode
  /// @see VecStep, returns::NativeGae()
  VecStep step(const std::vector<Action>& actions) {
    if (actions.size() != batch_.size())
      throw std::invalid_argument("native vector action row count");
    for (std::size_t i = 0; i < actions.size(); ++i) {
      ValidateAction(action_space(), actions[i]);
      if (actions[i].continuous.size() != batch_.action_size())
        throw std::invalid_argument("native vector action width");
      std::copy(
          actions[i].continuous.begin(),
          actions[i].continuous.end(),
          actions_.begin() + i * batch_.action_size());
    }
    const auto& out = batch_.step(actions_);
    VecStep     result;
    result.reward.assign(out.rewards.begin(), out.rewards.end());
    result.terminated = out.terminated;
    result.truncated  = out.truncated;
    result.final_obs.resize(batch_.size());
    bool ended = false;
    for (std::size_t i = 0; i < batch_.size(); ++i) {
      reset_[i] = out.terminated[i] || out.truncated[i];
      ended     = ended || reset_[i];
      if (reset_[i])
        result.final_obs[i] =
            NativeObservation(out.final_observations, i, batch_.observation_size());
    }
    if (ended)
      batch_.reset(reset_);
    result.obs = observations();
    return result;
  }

 private:
  /// @brief Validate signed example counts before native size conversion.
  ///
  /// @param count Requested world or worker count.
  /// @return Positive native size.
  /// @throws std::invalid_argument The count is nonpositive.
  /// @see NativeVectorTask
  static std::size_t positive(int count) {
    if (count < 1)
      throw std::invalid_argument("native count/workers must be positive");
    return static_cast<std::size_t>(count);
  }
  /// @brief Copy current native observations after any masked resets.
  ///
  /// @return Independent Float32 row vectors.
  /// @see NativeVectorTask
  std::vector<Obs> observations() const {
    std::vector<Obs> result(batch_.size());
    for (std::size_t i = 0; i < batch_.size(); ++i)
      result[i] = NativeObservation(batch_.output().observations, i, batch_.observation_size());
    return result;
  }
  datamunge::sim::Task      task_;  ///< Shared task definition.
  datamunge::sim::TaskBatch batch_; ///< Native batched state, RNG and preallocated solver storage.
  std::vector<double>       actions_; ///< Reused row-major physical actions.
  std::vector<std::uint8_t> reset_;   ///< Reused binary reset mask.
};

} // namespace datamunge::rl::environments
