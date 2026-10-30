#pragma once
/// @file tasks.hpp
/// @brief Native articulated reinforcement-learning tasks with reusable batched outputs.
#include <datamunge/sim/batch.hpp>
namespace datamunge::sim {
/// @brief Built-in native rigid-body control task, using SI units and Y-up coordinates.
/// These are datamunge task definitions, not numerical replicas of another simulator.
/// @par Example
/// @code
/// const auto task=dm::sim::Task::InvertedPendulum;
/// @endcode
enum class Task {
  InvertedPendulum, ///< Sliding cart and upright hinged pole; one force action in [-3,3].
  Hopper ///< Planar free torso and three articulated leg segments; three torques in [-1,1].
};
/// @brief Preallocated observations, rewards and independent episode-end flags.
/// Observations are row-major [environments,observation_size]. Terminal observations
/// stay in final_observations until the next terminal transition for that index.
/// No auto-reset occurs. Masked-out rows keep all previous values.
/// @par Example
/// @code
/// dm::sim::TaskBatch task(dm::sim::Task::InvertedPendulum,2);
/// const auto& result=task.output();
/// @endcode
struct TaskOutput {
  std::vector<double>       observations; ///< Current observations, including terminal states.
  std::vector<double>       rewards;      ///< Reward from the last selected step, zero after reset.
  std::vector<std::uint8_t> terminated;   ///< Physical failure, requiring explicit reset.
  std::vector<std::uint8_t>
                      truncated; ///< Time limit, requiring explicit reset; bootstrap is valid.
  std::vector<double> final_observations; ///< Last episode's final observations, meaningful when
                                          ///< final_valid is one.
  std::vector<std::uint8_t>
      final_valid; ///< Whether a final observation has ever been recorded for this row.
};
/// @brief Batched native InvertedPendulum or Hopper rollouts without a graphics dependency.
/// CPU worlds use persistent parallel workers. Actions and all output arrays are
/// finite double precision, with no NN dependency. Single-writer, including reset.
/// Invalid actions fail before forces are applied. A native solver error propagates;
/// reset selected worlds before retrying after such an error. Reset uses small
/// independent uniform joint perturbations and never touches another row's RNG.
/// @par Example
/// @code
/// dm::sim::TaskBatch task(dm::sim::Task::InvertedPendulum,4,dm::compute::Context({},2),42);
/// const auto& result=task.step(std::vector<double>(4,0));
/// @endcode
class TaskBatch {
 public:
  /// @brief Build independent articulated task worlds and sample initial states.
  /// Pendulum observations are [cart_x,pole_angle,cart_v,pole_rate], dt=.02,
  /// force=100*action, reward=1, failure |x|>2.4 or |angle|>.2.
  /// Hopper observations are [torso_y,pitch,three_joint_angles,torso_vx,torso_vy,
  /// pitch_rate,three_joint_rates], dt=.032, torques=200*action. Its reward is
  /// forward COM speed+1-.001*sum(action squared); failure y<.65 or |pitch|>.5.
  /// @param task Built-in task enum.
  /// @param count Positive environment count.
  /// @param context CPU execution owner; one lane by default.
  /// @param seed Independent environment RNG seed, default zero.
  /// @param time_limit Positive maximum episode steps, default 1000.
  /// @throws std::invalid_argument Invalid task, count, context or zero time limit.
  /// @par Example
  /// @code
  /// dm::sim::TaskBatch hopper(dm::sim::Task::Hopper,2);
  /// @endcode
  explicit TaskBatch(
      Task             task,
      std::size_t      count,
      compute::Context context    = compute::Context{},
      std::uint64_t    seed       = 0,
      std::uint64_t    time_limit = 1000);
  /// @brief Return the number of independent environments.
  /// @return Positive environment count.
  std::size_t size() const noexcept { return worlds_->size(); }
  /// @brief Read the task's observation width.
  /// @return Four for InvertedPendulum or eleven for Hopper.
  std::size_t observation_size() const noexcept { return task_ == Task::InvertedPendulum ? 4 : 11; }
  /// @brief Read the continuous action width.
  /// @return One for InvertedPendulum or three for Hopper.
  std::size_t action_size() const noexcept { return task_ == Task::InvertedPendulum ? 1 : 3; }
  /// @brief Borrow current rollout outputs without advancing the simulation.
  /// @return Internal reusable arrays, valid until destruction and modified by step/reset.
  const TaskOutput& output() const noexcept { return output_; }
  /// @brief Borrow one world read-only for inspection or rendering.
  /// @param index Environment index in [0,size()).
  /// @return Const world valid until task destruction.
  /// @throws std::out_of_range Invalid index.
  const World& world(std::size_t index) const { return worlds_->world(index); }
  /// @brief Advance selected task rows and retain final observations on episode end.
  /// Selected completed rows must be reset first. An unselected row remains unchanged.
  /// @param actions Row-major [size(),action_size()] values in the documented action box.
  /// @param mask Empty selects all; otherwise size() binary bytes.
  /// @return Borrowed reusable rollout output; no automatic reset.
  /// @throws std::invalid_argument Invalid action shape/value or mask, before mutation.
  /// @throws std::logic_error A selected row already finished.
  /// @throws std::exception Physics failure; explicit reset is required before retrying.
  /// @par Example
  /// @code
  /// dm::sim::TaskBatch task(dm::sim::Task::Hopper,1);
  /// const auto& result=task.step({0,0,0});
  /// @endcode
  const TaskOutput& step(
      const std::vector<double>&       actions,
      const std::vector<std::uint8_t>& mask = {});
  /// @brief Restore and perturb selected rows; retain final observations for rollout collectors.
  /// Clears rewards/end flags for selected rows. Random streams continue between episodes.
  /// @param mask Empty selects all; otherwise size() binary bytes.
  /// @return Borrowed reset observations and current unselected rows.
  /// @throws std::invalid_argument Invalid mask.
  /// @par Example
  /// @code
  /// dm::sim::TaskBatch task(dm::sim::Task::InvertedPendulum,2);task.reset({0,1});
  /// @endcode
  const TaskOutput& reset(const std::vector<std::uint8_t>& mask = {});
  /// @brief Reproduce fresh reset draws after explicit reseeding; physics is unchanged until reset.
  /// @param seed New batch random seed.
  /// @par Example
  /// @code
  /// dm::sim::TaskBatch task(dm::sim::Task::Hopper,1);task.seed(7);task.reset();
  /// @endcode
  void seed(std::uint64_t seed) noexcept { worlds_->seed(seed); }

 private:
  /// @brief Validate a whole-row selection mask.
  /// @param mask Empty or size() binary bytes.
  void validate_mask(const std::vector<std::uint8_t>& mask) const;
  /// @brief Write current native coordinates into one preallocated observation row.
  /// @param index Environment index.
  void                        observe(std::size_t index);
  Task                        task_;       ///< Immutable task selection.
  std::uint64_t               time_limit_; ///< Positive episode horizon.
  std::unique_ptr<BatchWorld> worlds_;     ///< Independent articulated worlds and RNGs.
  TaskOutput                  output_;     ///< Reused row-major rollout arrays.
};
} // namespace datamunge::sim
