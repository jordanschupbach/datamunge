#pragma once
/// @file native_tasks.hpp
/// @brief Thin example adapters over canonical native TaskBatch physics and terminal flags.
#include <algorithm>
#include <cmath>
#include <datamunge/sim/tasks.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/rl/env.hpp"
#include "common/rl/vec_env.hpp"
namespace rl {
/// @brief Reject unsupported GPU training requests before the CPU example parser can ignore them.
/// @param argc Argument count including executable name.
/// @param argv Original argument vector; device/backend values must explicitly select CPU.
/// @throws std::invalid_argument A non-CPU backend was requested.
inline void ValidateNativeDeviceArguments(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto        equals   = argument.find('=');
    const auto        name     = argument.substr(0, equals);
    if ((name == "device" || name == "backend") && equals != std::string::npos) {
      const auto value = argument.substr(equals + 1);
      if (value != "cpu" && value != "cpu:0" && value != "/CPU:0")
        throw std::invalid_argument(
            "native PPO/SAC numerical kernels currently support device=cpu only");
    }
  }
}
/// @brief Select one supported native task; misspellings never select a different environment.
inline datamunge::sim::Task NativeTask(const std::string& name) {
  if (name == "inverted_pendulum")
    return datamunge::sim::Task::InvertedPendulum;
  if (name == "hopper")
    return datamunge::sim::Task::Hopper;
  throw std::invalid_argument("task must be inverted_pendulum or hopper");
}
/// @brief Native task action bounds in simulator units, applied at the policy boundary.
inline ActionSpace NativeSpace(datamunge::sim::Task task) {
  return task == datamunge::sim::Task::InvertedPendulum
             ? ActionSpace::Continuous({-3}, {3})
             : ActionSpace::Continuous({-1, -1, -1}, {1, 1, 1});
}
/// @brief Copy one row of native double observations into the examples' float observation type.
inline Obs
NativeObservation(const std::vector<double>& values, std::size_t row, std::size_t width) {
  const auto begin = values.begin() + static_cast<std::ptrdiff_t>(row * width);
  return Obs(begin, begin + static_cast<std::ptrdiff_t>(width));
}
/// @brief Single native physical task implementing the existing evaluation/replay example
/// interface. Episode endings are explicit; no automatic reset replaces the final observation.
class NativeTaskEnv final : public Env {
 public:
  /// @brief Own one canonical CPU task with independent RNG and the requested positive time limit.
  explicit NativeTaskEnv(datamunge::sim::Task task, std::uint64_t limit = 1000)
      : task_(task),
        limit_(checked_limit(limit)),
        batch_(task, 1, datamunge::compute::Context{}, 0, limit_) {}
  /// @brief Human-readable native task name, distinguishing it from source-runtime conformance.
  std::string name() const override {
    return task_ == datamunge::sim::Task::InvertedPendulum ? "Native InvertedPendulum"
                                                           : "Native Hopper";
  }
  /// @brief Float observation width matching the canonical double task output.
  int obs_size() const override { return static_cast<int>(batch_.observation_size()); }
  /// @brief Owned physical action bounds.
  ActionSpace action_space() const override { return NativeSpace(task_); }
  /// @brief Advance one native world; final observations precede any explicit reset.
  StepResult step(const Action& action) override {
    const std::vector<double> values(action.continuous.begin(), action.continuous.end());
    const auto&               out = batch_.step(values);
    return {
        NativeObservation(out.observations, 0, batch_.observation_size()),
        static_cast<float>(out.rewards[0]),
        out.terminated[0] != 0,
        out.truncated[0] != 0};
  }
  /// @brief Restart the native random stream without implicitly resetting an episode.
  void Seed(std::uint64_t seed) override { batch_.seed(seed); }
  /// @brief Positive configured episode time limit.
  int max_steps() const override { return static_cast<int>(limit_); }

 protected:
  /// @brief Reset using the native task's continuing independent random stream.
  Obs Reset() override {
    return NativeObservation(batch_.reset().observations, 0, batch_.observation_size());
  }

 private:
  /// @brief Keep the generic Env interface's signed time-limit representation valid.
  static std::uint64_t checked_limit(std::uint64_t limit) {
    if (!limit || limit > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("native example time limit outside int range");
    return limit;
  }
  datamunge::sim::Task      task_;  ///< Authored native task definition.
  std::uint64_t             limit_; ///< Positive episode cap used by shared evaluation helpers.
  datamunge::sim::TaskBatch batch_; ///< Sole physical state and RNG owner.
};
/// @brief Native batched task adapter with masked automatic reset and retained final observations.
/// Native stepping is preallocated and parallel; returned example vectors are owned copies.
class NativeVectorTask {
 public:
  /// @brief Own independent native worlds on persistent CPU workers.
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
  /// @brief Number of native worlds.
  int size() const { return static_cast<int>(batch_.size()); }
  /// @brief Shared observation width.
  int obs_size() const { return static_cast<int>(batch_.observation_size()); }
  /// @brief Shared physical action bounds.
  ActionSpace action_space() const { return NativeSpace(task_); }
  /// @brief Explicitly reset all worlds, retaining their continuing random streams.
  std::vector<Obs> reset() {
    batch_.reset();
    return observations();
  }
  /// @brief Step all worlds, copy terminal values, then mask-reset completed rows.
  VecStep step(const std::vector<Action>& actions) {
    if (actions.size() != batch_.size())
      throw std::invalid_argument("native vector action row count");
    for (std::size_t i = 0; i < actions.size(); ++i) {
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
  /// @brief Validate signed example counts before converting them to native sizes.
  static std::size_t positive(int count) {
    if (count < 1)
      throw std::invalid_argument("native count/workers must be positive");
    return static_cast<std::size_t>(count);
  }
  /// @brief Copy all current native rows after any masked resets.
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
/// @brief Compute GAE with explicit next-state bootstraps and reset trace boundaries.
/// @param rewards Scaled rewards for one trajectory segment.
/// @param values Value predictions for its pre-action states.
/// @param bootstraps Actual next-state values; physical termination supplies zero.
/// @param done Binary terminal-or-time-limit flags, cutting traces before reset observations.
/// @param gamma Finite discount in [0,1].
/// @param lambda Finite trace decay in [0,1].
/// @return Advantages and value targets; both are owned and have trajectory length.
/// @throws std::invalid_argument Shape, masks, coefficients or values are invalid.
inline std::pair<std::vector<float>, std::vector<float>> NativeGae(
    const std::vector<float>& rewards,
    const std::vector<float>& values,
    const std::vector<float>& bootstraps,
    const std::vector<float>& done,
    float                     gamma,
    float                     lambda) {
  const auto n = rewards.size();
  if (values.size() != n || bootstraps.size() != n || done.size() != n || !std::isfinite(gamma) ||
      gamma < 0 || gamma > 1 || !std::isfinite(lambda) || lambda < 0 || lambda > 1)
    throw std::invalid_argument("native GAE shape or coefficient");
  std::vector<float> advantages(n), targets(n);
  float              advantage = 0;
  for (auto t = n; t-- > 0;) {
    if (!std::isfinite(rewards[t]) || !std::isfinite(values[t]) || !std::isfinite(bootstraps[t]) ||
        (done[t] != 0 && done[t] != 1))
      throw std::invalid_argument("native GAE nonfinite value or nonbinary boundary");
    const float delta = rewards[t] + gamma * bootstraps[t] - values[t];
    advantage         = delta + gamma * lambda * (1 - done[t]) * advantage;
    advantages[t]     = advantage;
    targets[t]        = advantage + values[t];
  }
  return {std::move(advantages), std::move(targets)};
}
} // namespace rl
