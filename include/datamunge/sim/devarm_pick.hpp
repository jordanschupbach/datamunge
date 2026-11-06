#pragma once
/// @file devarm_pick.hpp
/// @brief Seeded DevArm manipulation, tabular Q-learning, checkpoints and rollout frames.
#include <array>
#include <datamunge/cad/robot.hpp>
#include <datamunge/sim/conveyor.hpp>
#include <functional>
#include <string>
#include <vector>

namespace datamunge::sim {
/// @brief Requested placement goal, independent of the payload's ground-truth class.
/// @par Example
/// @code
/// auto goal=dm::sim::PickBin::Yellow;
/// @endcode
enum class PickBin {
  Red,   ///< Bin at CAD Y=-0.42 m.
  Yellow ///< Bin at CAD Y=+0.42 m.
};
/// @brief Cartesian 30 mm servo motions and proximity-gripper commands.
enum class PickAction {
  XPlus,  ///< Move along conveyor by +30 mm.
  XMinus, ///< Move along conveyor by -30 mm.
  YPlus,  ///< Move across conveyor by +30 mm in CAD coordinates.
  YMinus, ///< Move toward the red bin by 30 mm.
  ZPlus,  ///< Raise the grasp point by 30 mm.
  ZMinus, ///< Lower the grasp point by 30 mm.
  Close,  ///< Attempt to close around the fruit; reopen on an empty grasp.
  Open,   ///< Release the fruit, then allow 0.7 s for physical settling.
  Wait    ///< Hold the current arm pose for 0.15 s.
};
/// @brief Owned state and render frame in CAD Z-up metres and URDF radians.
/// @par Example
/// @code
/// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
/// auto frame=env.observation();
/// @endcode
struct PickObservation {
  FruitVariation  variation{FruitVariation::Basic}; ///< Domain protocol of this frame.
  FruitAppearance appearance;                 ///< Recorded shape, skin and material realization.
  Quaternion      fruit_rotation{1, 0, 0, 0}; ///< Actual Y-up collision-body orientation.
  Vec3            fruit_dimensions{};         ///< Body-local collision dimensions, metres.
  ConveyorStage   feed_stage{ConveyorStage::Waiting}; ///< Arrival sensor/workcell status.
  double          belt_travel{0};                     ///< Integrated moving tread distance, metres.

  std::array<double, 6> joints{}; ///< Six actual position-servo coordinates in URDF order.
  double     finger{.05};         ///< Left finger coordinate, metres; right follows its URDF mimic.
  cad::Vec3d tool;                ///< Finger-tip grasp centre in workcell metres.
  cad::Vec3d fruit;               ///< Payload centre in workcell metres.
  FruitKind  kind{
      FruitKind::Apple}; ///< Ground truth for training/scoring, never a camera prediction.
  PickBin     destination{PickBin::Red}; ///< Requested goal, independent of ground truth.
  bool        grasped{false};    ///< Grasp state: ideal legacy latch or established articulated finger contact.
  bool        released{false};   ///< A successful grasp has subsequently been opened.
  bool        success{false};    ///< Released dynamic fruit settled on the requested bin floor.
  bool        terminated{false}; ///< Success, failed release or loss; reset is required.
  bool        truncated{false};  ///< Decision horizon reached; reset is required.
  bool        collision{false};  ///< Last requested motion was rejected by the tool/payload guard.
  std::size_t state{0};          ///< Discrete position/holding state for the tabular policy.
  std::size_t decisions{0};      ///< Agent actions since reset, including gripper commands.
  double      time{0};           ///< Simulated seconds since reset.
  bool measured_joints{false}; ///< Render actual physical encoder coordinates, including small
                               ///< compliant-limit excursions, rather than authored commands.
  double      reward{0};         ///< Last task reward including discounted potential shaping.
};
/// @brief Headless DevArm pick-and-place environment with a caller-selected bin objective.
/// Nine actions operate a bounded Cartesian lattice; numerical IK drives the
/// six real URDF joint coordinates, clamped to limits and limited to 1.5 rad/s.
/// IK accepts up to 12 mm position residual and uses a soft wrist-down preference.
/// The grasp point is end_link local (-.02,0,0) m, between the finger tips.
/// Tool and carried-payload collision guards reject moves through workcell
/// solids. The grasp is an ideal proximity latch with size-dependent jaw aperture and payload
/// orientation stabilization; opening restores dynamic gravity/contact physics.
/// Robot servos are kinematic: this is not torque control, a full mesh-collision
/// robot simulator, or hardware deployment. Observations are privileged state,
/// not image observations. Instances are single-writer and own independent worlds.
/// @par Example
/// @code
/// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
/// env.reset(42);
/// auto state=env.step(dm::sim::PickAction::ZMinus);
/// @endcode
/// @par References
/// - S. R. Buss, Introduction to Inverse Kinematics with Jacobian Transpose,
///   Pseudoinverse and Damped Least Squares methods (author's notes).
///   https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/
class DevArmPickEnvironment {
 public:
  /// @brief Load the actual B601-DM URDF and initialize the workcell.
  /// @param urdf Existing DevArm URDF, with its referenced visual meshes.
  /// @param horizon Positive action limit, default 80 decisions.
  /// @param variation Basic preserves earlier tables; Diverse uses varied payloads and
  /// lateral-state bins.
  /// @throws std::invalid_argument Invalid horizon or unsupported URDF joints/links.
  /// @throws std::runtime_error Asset import or unreachable initial pose.
  /// @par Example
  /// See the class example.
  explicit DevArmPickEnvironment(
      const std::string& urdf,
      std::size_t        horizon   = 80,
      FruitVariation     variation = FruitVariation::Basic);
  /// @brief Release the owned robot, physics and pose cache.
  ~DevArmPickEnvironment();
  /// @brief Copying independent live worlds is disabled.
  /// @param other Environment that cannot be copied.
  DevArmPickEnvironment(const DevArmPickEnvironment& other) = delete;
  /// @brief Assignment of live worlds is disabled.
  /// @param other Environment that cannot be assigned.
  /// @return No value; deleted operation.
  DevArmPickEnvironment& operator=(const DevArmPickEnvironment& other) = delete;
  /// @brief Restart with a seeded apple/banana waiting under the open gripper.
  /// Longitudinal jitter is +/-8 mm; height follows the actual collision proxy.
  /// Basic also jitters laterally by +/-8 mm; Diverse retains its sampled
  /// across-belt position in [-.065,.065] m. This training reset skips feeding;
  /// use reset_feed()/advance_feed() for a physical conveyor arrival.
  /// @param seed Seed independent of policy exploration and evaluation seeds.
  /// @return Initial observation at decision zero.
  /// @throws std::runtime_error Invalid initial IK solution.
  /// @par Example
  /// See the class example.
  PickObservation reset(std::uint64_t seed);
  /// @brief Start the fruit at its sampled belt entry rather than at the pick station.
  /// @param seed Reproducible fruit prototype, appearance, position and orientation seed.
  /// @return Initial feeding observation; call advance_feed() until the sensor stops the belt.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf",80,dm::sim::FruitVariation::Diverse);
  /// auto state=env.reset_feed(42);
  /// while(state.feed_stage==dm::sim::ConveyorStage::Travelling)state=env.advance_feed(1./120);
  /// @endcode
  PickObservation reset_feed(std::uint64_t seed);
  /// @brief Advance automatic feeding/settling before the first arm action.
  /// Emits recorder frames at 24 Hz without consuming agent decisions.
  /// @param seconds Finite elapsed time in (0,1/30].
  /// @return Updated measured fruit/arm state.
  /// @throws std::invalid_argument Invalid delta before mutation.
  /// @throws std::logic_error Agent has acted, or episode ended.
  /// @par Example
  /// See reset_feed().
  PickObservation advance_feed(double seconds);
  /// @brief Select a goal before the first action, without inspecting fruit identity.
  /// reset() preserves this setting; construction defaults to Red.
  /// @param destination Red or Yellow, for example selected by a camera classifier.
  /// @throws std::invalid_argument Unknown enum value.
  /// @throws std::logic_error Episode has actions or has ended.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// env.reset(42); env.set_destination(dm::sim::PickBin::Yellow);
  /// @endcode
  void set_destination(PickBin destination);
  /// @brief Execute a Cartesian/gripper action and return its measured transition.
  /// Reward is -0.05 per decision, -0.15 for a blocked move, +10 for a settled
  /// requested-bin release, or -3 for a failed release/lost fruit, plus
  /// 0.98*Phi(next)-Phi(current). Phi is -1-3*distance(tool,fruit) before grasp
  /// and 1-3*distance(fruit,requested-bin release point) while grasped; terminal Phi=0.
  /// @param action One of the nine PickAction values; no automatic action selection.
  /// @return State with separate termination and time-limit flags.
  /// @throws std::invalid_argument Unknown action, before mutation.
  /// @throws std::logic_error Episode already ended.
  /// @throws std::runtime_error Native solver failure.
  /// @par Example
  /// See the class example.
  PickObservation step(PickAction action);
  /// @brief Copy the current observation, without stepping physics.
  /// @return Owned state in CAD Z-up coordinates.
  PickObservation observation() const;
  /// @brief Set an optional rollout recorder, called at reset and every 1/24 simulated second.
  /// The callback must not modify this environment and should not throw.
  /// @param recorder Receives owned observations; empty disables recording for training.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// std::vector<dm::sim::PickObservation> frames;
  /// env.record([&](const auto& frame){frames.push_back(frame);}); env.reset(42);
  /// @endcode
  void record(std::function<void(const PickObservation&)> recorder);
  /// @brief Select workcell/arm rendering assets without moving the real servo state.
  /// @param style Simple or Realistic; retained across reset().
  /// @throws std::invalid_argument Unknown style.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// env.set_visual_style(dm::sim::WorkcellVisualStyle::Realistic);
  /// @endcode
  void set_visual_style(WorkcellVisualStyle style);
  /// @brief Read the visual profile used by scenes and attached wrist cameras.
  /// @return Current profile; Simple by default.
  WorkcellVisualStyle visual_style() const noexcept;
  /// @brief Copy the stationary belt/bins/pedestal, without robot or fruit.
  /// @return Owned scene in CAD Z-up metres; both bins are available for caller-selected goals.
  cad::Scene static_scene() const;
  /// @brief Build the actual posed arm and fruit for a recorded frame.
  /// @param frame Frame from this environment's current fruit class/reset.
  /// @return Owned moving-actor scene for depth composition with static_scene().
  /// @throws std::invalid_argument Frame has a different fruit class or invalid joints.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// auto frame=env.observation();auto actors=env.dynamic_scene(frame);
  /// dm::cad::render(actors,dm::cad::fit_camera(actors)).save("pick.png");
  /// @endcode
  cad::Scene dynamic_scene(const PickObservation& frame) const;
  /// @brief Build only the moving payload and conveyor marks, without the arm.
  /// @param frame Recorded frame with the current reset's fruit class.
  /// @return Owned Z-up scene, useful for camera background caching.
  /// @throws std::invalid_argument Different fruit class.
  /// @par Example
  /// See dynamic_scene(); this is its payload component.
  cad::Scene fruit_scene(const PickObservation& frame) const;
  /// @brief Build only the actual posed URDF arm at its mounting plate.
  /// @param frame Valid recorded joint/finger coordinates.
  /// @return Owned Z-up arm geometry, without fruit or static scenery.
  /// @throws std::invalid_argument Invalid recorded coordinates.
  /// @par Example
  /// See dynamic_scene(); this is its arm component.
  cad::Scene robot_scene(const PickObservation& frame) const;
  /// @brief Compute a real link frame at a recorded pose, including the mounting transform.
  /// @param frame Valid recorded joint/finger coordinates.
  /// @param link Existing URDF link, default end_link.
  /// @return Local-link to Z-up workcell transform in metres.
  /// @throws std::invalid_argument Unknown link or invalid coordinates.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// const auto wrist=env.link_pose(env.observation());
  /// @endcode
  cad::Mat4d link_pose(const PickObservation& frame, const std::string& link = "end_link") const;
  /// @brief Solve one existing Cartesian lattice target without moving the live arm.
  /// This exposes the same cached real-URDF IK used by step(), for continuing
  /// controllers which own their own world clock and interpolate servo commands.
  /// @param x Longitudinal index -2..2, world X=.10+.03*x metres.
  /// @param y Lateral index -16..16, world Y=.03*y metres.
  /// @param z Height index 0..8, world Z=.76+.03*z metres.
  /// @return Six joint angles in radians, within the URDF limits.
  /// @throws std::invalid_argument Index outside the lattice.
  /// @throws std::runtime_error Target has no IK solution within the 12 mm tolerance.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// auto hover=env.joint_target(0,0,4);
  /// @endcode
  std::array<double, 6> joint_target(int x, int y, int z);
  /// @brief Legacy Basic-protocol lattice/holding capacity, including its absorbing state.
  /// Use state_space_size() for an instance that may select Diverse.
  /// @return 2971; retained for existing Basic checkpoints and source compatibility.
  static std::size_t state_count() noexcept;
  /// @brief Actual state-table capacity for this environment's domain protocol.
  /// Diverse includes five observed fruit-lateral bins before grasp and a shared carrying table.
  /// @return 2971 for Basic or 8911 for Diverse, including one absorbing state.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf",80,dm::sim::FruitVariation::Diverse);
  /// auto states=env.state_space_size(); // 8911
  /// @endcode
  std::size_t state_space_size() const noexcept;
  /// @brief Fixed discrete action-space cardinality.
  /// @return Nine.
  static std::size_t action_count() noexcept { return 9; }

 private:
  struct Impl;                 ///< Owned robot, solver state, IK cache and recorder.
  std::unique_ptr<Impl> impl_; ///< Exclusive mutable task state.
};

/// @brief Frozen tabular Q-learning policy with a portable text checkpoint.
/// Rows match state_space_size() for the training protocol and nine actions.
/// Basic tables have 2971 rows; Diverse tables have 8911 and must be retrained.
/// @par Example
/// @code
/// auto policy=dm::sim::PickPolicy::load("policy.txt");
/// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
/// auto action=policy.action(env.reset(42).state);
/// @endcode
struct PickPolicy {
  std::vector<std::vector<double>> q; ///< Finite learned action values, state-major.

  /// @brief Select a greedy action; ties resolve to the lowest action index.
  /// @param state Valid state index.
  /// @return Action maximizing the frozen Q row.
  /// @throws std::invalid_argument Malformed/nonfinite Q row.
  /// @throws std::out_of_range State outside the policy.
  /// @par Example
  /// See the class example.
  PickAction action(std::size_t state) const;
  /// @brief Save every learned Q value using round-trip decimal precision.
  /// @param path Output checkpoint file; parent directory must exist.
  /// @throws std::invalid_argument Invalid policy shape or nonfinite values.
  /// @throws std::runtime_error File write failure.
  /// @par Example
  /// See train_red_bin() and the class example.
  void save(const std::string& path) const;
  /// @brief Load and validate a versioned checkpoint; no training or policy fallback.
  /// @param path Existing text checkpoint.
  /// @return Owned frozen policy with validated dimensions and finite values.
  /// @throws std::runtime_error Malformed/truncated/trailing-data checkpoint or I/O error.
  /// @par Example
  /// See the class example.
  static PickPolicy load(const std::string& path);
};
/// @brief Per-episode learning evidence and the final, frozen policy.
/// @par Example
/// See train_red_bin().
struct PickTrainingResult {
  PickBin             destination{PickBin::Red}; ///< Pair the policy with this goal at inference.
  PickPolicy          policy;                    ///< Final Q table, without evaluation selection.
  std::vector<double> returns;                   ///< Undiscounted training reward per episode.
  std::vector<unsigned char> successes; ///< One if that exploratory episode placed into red.
};
/// @brief Train model-free, epsilon-greedy tabular Q-learning from real task transitions.
/// Reuses the canonical algorithms::q_learning implementation with alpha=.3,
/// gamma=.98 and epsilon=.25. No demonstrations, scripted policy or action masking
/// are used. Every training reset selects Red; train_pick_bin() also supports Yellow.
/// Evaluation must use separate seeds and the returned frozen policy.
/// @param environment Independent mutable training environment; recorder is disabled.
/// @param episodes Positive number of training episodes, default 4000.
/// @param seed Exploration/initial-state seed, default 20261101.
/// @return Final policy and complete learning history; no success-rate guarantee.
/// @throws std::invalid_argument Zero episodes.
/// @throws std::runtime_error Environment/solver failure.
/// @par Example
/// @code
/// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
/// auto trained=dm::sim::train_red_bin(env,4000,20261101);
/// trained.policy.save("policy.txt");
/// @endcode
/// @par References
/// - Watkins and Dayan (1992), Q-learning. https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf
/// - Ng, Harada and Russell (1999), Policy invariance under reward transformations.
///   https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf
PickTrainingResult train_red_bin(
    DevArmPickEnvironment& environment,
    std::size_t            episodes = 4000,
    std::uint64_t          seed     = 20261101);
/// @brief Train native Q-learning for an explicitly requested bin, independent of fruit class.
/// Pair the returned table with the same set_destination() value at inference;
/// Q checkpoints do not encode goal metadata. See train_red_bin() for algorithm details.
/// @param environment Mutable independent training environment; recorder is disabled.
/// @param destination Requested Red or Yellow bin for every training reset.
/// @param episodes Positive episode count, default 2000.
/// @param seed Exploration/initial-state seed, default 20261102.
/// @return Frozen action values, requested goal and complete learning history.
/// @throws std::invalid_argument Unknown destination or zero episodes.
/// @throws std::runtime_error Native solver failure.
/// @par Example
/// @code
/// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
/// auto yellow=dm::sim::train_pick_bin(env,dm::sim::PickBin::Yellow,20,42);
/// @endcode
PickTrainingResult train_pick_bin(
    DevArmPickEnvironment& environment,
    PickBin                destination,
    std::size_t            episodes = 2000,
    std::uint64_t          seed     = 20261102);
} // namespace datamunge::sim
