#pragma once
/// @file fruit_stream.hpp
/// @brief Continuing Poisson conveyor sorting and duration-aware hierarchical Q-learning.
#include <array>
#include <datamunge/rl/discrete.hpp>
#include <datamunge/sim/physics_backend.hpp>
#include <datamunge/sim/devarm_pick.hpp>
#include <functional>
#include <memory>
#include <vector>

namespace datamunge::sim {
/// @brief Duration-varying interception options for a continuously moving belt.
/// Options use fixed, rate-limited IK servos; Q-learning selects timing/target/gate.
/// @par Example
/// @code
/// auto action=dm::sim::StreamAction::LateFront;
/// @endcode
enum class StreamAction {
  Wait,       ///< Advance 0.125 seconds; optionally watch the infeed when the queue is empty.
  EarlyFront, ///< Intercept the leading fruit at X=.04 m.
  LateFront,  ///< Intercept the leading fruit at X=.16 m.
  EarlyNext,  ///< Intercept the second-leading fruit at X=.04 m, letting an urgent leader pass.
  LateNext    ///< Intercept the second-leading fruit at X=.16 m.
};
/// @brief Physical lifecycle of one fruit; outcome counters never depend on held poses.
/// @par Example
/// @code
/// auto status=dm::sim::StreamFruitStatus::Belt;
/// @endcode
enum class StreamFruitStatus {
  Belt,     ///< Travelling on the live belt.
  Held,     ///< Grasp established; articulated payload remains dynamic and may slip.
  Released, ///< Dynamic fruit falling into a bin.
  Correct,  ///< Settled, grounded in the true class's bin.
  Wrong,    ///< Settled, grounded in the other bin.
  Missed    ///< Passed the pickup window, fell away, or failed release.
};
/// @brief Configuration of an unbounded-time, bounded-memory sorting workcell.
/// Arrivals are never postponed for a busy arm. At capacity, arrivals are counted
/// as overflow misses. Outcomes are removed after two seconds (self-emptying bins).
/// Legacy mode stabilizes orientation and uses ideal latches. Articulated mode
/// integrates free rotation, force-limited servos and physical grasps. Motor geometry
/// remains privileged state; nominal physics is not a hardware calibration.
/// @par Example
/// @code
/// dm::sim::FruitStreamOptions options; options.arrival_rate=.14;
/// options.watch_infeed=true; // point the wrist camera upstream while idle
/// options.synchronized_grasp=true; // align the inspection and descent wrist roll
/// options.articulated_physics=true; // native, no external physics runtime
/// options.physics_timestep=1./960;
/// options.queued_wait_cost=.5; options.miss_cost=6; options.wrong_cost=8;
/// @endcode
struct FruitStreamOptions {
  bool training_proxy{
      false}; ///< Opt-in analytic transport, incompatible with articulated_physics.
  double                arrival_rate{.14};   ///< Poisson intensity, fruit/second, finite 0..5.
  std::size_t           active_capacity{32}; ///< Maximum retained payload bodies, 1..32.
  std::array<double, 2> training_error{
      .015625,
      .1015625}; ///< Synthetic training error rates for apple/banana, each 0..1.
  WorkcellVisualStyle visual_style{
      WorkcellVisualStyle::Simple}; ///< Rendering-only assets and lighting; appended to preserve
                                    ///< legacy aggregate initialization.
  bool watch_infeed{false}; ///< Opt-in idle wrist-camera aim at the belt entry; default preserves
                            ///< existing policies. Uses bounded servos during 0.125-second empty
                            ///< waits, yields to picks, and changes option travel times. Retrain or
                            ///< reevaluate scheduler checkpoints before comparing throughput.
  bool synchronized_grasp{false}; ///< Align approach/descent wrist roll and choose a guarded short
                                  ///< approach route; articulated carrying uses twice-duration quintic references
                                  ///< with zero endpoint velocity/acceleration to reduce contact loss.
                                  ///< Default false retains the original motor options.
  double queued_wait_cost{0}; ///< Finite 0..100 reward cost per second of waiting with an eligible
                              ///< belt fruit; zero retains the original reward.
  double miss_cost{3}; ///< Finite 0..100 reward cost per missed fruit, including capacity overflow.
  double wrong_cost{5}; ///< Finite 0..100 reward cost per settled wrong-bin placement.

  /// @brief Use articulated dynamics and physical finger contacts, default false.
  /// Native is the default backend; optional MuJoCo requires DATAMUNGE_ENABLE_MUJOCO.
  /// Incompatible with training_proxy; no silent fallback.
  /// URDF link inertias, finite-effort servos, rolling fruit and compound collision
  /// proxies are integrated during training and evaluation. Parameters are nominal,
  /// not calibrated hardware data. Rendering uses measured joint/body poses.
  bool articulated_physics{false};

  /// @brief Internal articulated step, seconds; 1/240, 1/480, 1/960 or 1/1920.
  /// Default 1/480. Observations/arrival/option clocks remain at 240 Hz.
  double physics_timestep{1./480};

  ArticulatedBackend physics_backend{ArticulatedBackend::Native}; ///< Explicit articulated engine, native by default.

  /// @brief Articulated jaw-closing trigger: tool-to-fruit horizontal distance, metres.
  /// Finite .005..05; default .012 retains the original skill. The native tracking servo
  /// trails a moving fruit by roughly 13-15 mm, so .02 lets the jaws close on fruit that
  /// the default never starts closing on. Changes option outcomes: re-evaluate policies.
  double grasp_alignment_tolerance{.012};
  /// @brief Stop re-aiming the articulated tool once a finger touches the fruit, default false.
  /// The default re-solves the tracking target every eight ticks, which can jerk a grasp that
  /// is still building contact. Changes option outcomes: re-evaluate policies.
  bool hold_grasp_target{false};
};
/// @brief Cumulative event counts since the one initial reset; unresolved fruit remain explicit.
/// @par Example
/// @code
/// dm::sim::StreamCounters counters;
/// @endcode
struct StreamCounters {
  std::uint64_t arrivals{0};               ///< All Poisson arrivals, including capacity overflow.
  std::uint64_t correct{0};                ///< Actual correct-bin settled placements.
  std::uint64_t wrong{0};                  ///< Actual wrong-bin settled placements.
  std::uint64_t missed{0};                 ///< Escapes, failed releases and capacity overflow.
  std::uint64_t overflow{0};               ///< Subset of misses caused by capacity.
  std::uint64_t attempts{0};               ///< Selected fruit interception attempts.
  std::uint64_t grasped{0};                ///< Established grasps; ideal latches only in legacy mode.
  std::uint64_t inspections{0};            ///< Valid classifier predictions.
  std::uint64_t classification_correct{0}; ///< Correct predictions, for scoring only.
  std::uint64_t unknown{0}; ///< Empty/invalid vision result; never routed using truth.
};
/// @brief Recorded payload geometry, scheduling and latched classifier prediction.
/// @par Example
/// @code
/// dm::sim::StreamFruit fruit;
/// @endcode
struct StreamFruit {
  std::uint64_t       id{0};      ///< Monotonic stream ID; independent of world body IDs.
  std::uint64_t       seed{0};    ///< Appearance/entry replay seed.
  double              arrival{0}; ///< Exact Poisson scheduled arrival, seconds.
  ConveyorObservation body;       ///< Actual physics pose/dimensions and seeded CAD appearance.
  StreamFruitStatus   status{StreamFruitStatus::Belt}; ///< Measured lifecycle.
  int    prediction{-1}; ///< -1 unknown, 0 apple/red, 1 banana/yellow; never corrected from truth.
  double confidence{0};  ///< Confidence returned by the perception callback.
};
/// @brief Owned frame of a continuing world, suitable for deterministic video replay.
/// @par Example
/// @code
/// dm::sim::StreamFrame frame;
/// @endcode
struct StreamFrame {
  bool training_proxy{false}; ///< Whether this is the explicit analytic training model, never used
                              ///< for final physics video.
  double                   time{0}; ///< World time in seconds, never reset between picks.
  PickObservation          arm;   ///< Actual joint/finger/FK pose; legacy episode flags are unused.
  std::vector<StreamFruit> fruit; ///< Retained live/outcome bodies, bounded by capacity.
  StreamCounters           counts;    ///< Cumulative outcomes as of this exact frame.
  std::uint64_t            target{0}; ///< Current attempted fruit ID, zero while idle.
  bool articulated_physics{false}; ///< True only for articulated contact integration.
  ArticulatedBackend physics_backend{ArticulatedBackend::Native}; ///< Actual selected articulated engine.
  double maximum_penetration{0}; ///< Maximum active-contact penetration since reset, metres.
  double maximum_effort_fraction{0}; ///< Maximum actuator effort divided by authored limit.
  double maximum_joint_speed{0}; ///< Maximum actual arm speed across internal steps, rad/s.
  double maximum_finger_speed{0}; ///< Maximum actual finger speed across internal steps, m/s.
  bool                     synthetic_perception{
      true}; ///< True only when the explicitly synthetic training channel is active.
  std::size_t state{
      0}; ///< Finite scheduler abstraction: queue, urgency/gap, arm side, pending releases.
};
/// @brief Image classifier output; an unknown result leaves the fruit unhandled.
/// @par Example
/// @code
/// dm::sim::StreamPrediction prediction{0,.95};
/// @endcode
struct StreamPrediction {
  int    label{-1};     ///< 0 apple/red, 1 banana/yellow, -1 unknown.
  double confidence{0}; ///< Finite probability, 0..1.
};
/// @brief One continuing semi-Markov transition, including all arrivals/misses during a servo
/// option.
/// @par Example
/// @code
/// dm::sim::StreamTransition transition;
/// @endcode
struct StreamTransition {
  StreamFrame frame; ///< Measured state after the option finishes.
  double      reward{
      0}; ///< 5*correct-wrong_cost*wrong-miss_cost*missed-.02*seconds-.25*failed_attempts,
               ///< minus queued_wait_cost*seconds for a Wait with an eligible belt target.
  double seconds{0}; ///< Strictly positive simulated duration of the option.
};
/// @brief Native continuing multi-fruit conveyor, real-URDF servos and asynchronous outcomes.
/// The belt always runs at .22 m/s. Independent exponential inter-arrival times
/// are scheduled even during arm motion, and insertion is quantized by at most
/// one 1/240-second physics tick. Fruit passing X=.30 m is missed. There is no
/// between-pick reset, teleport, stopped belt or hidden queue at the source.
/// An opt-in training_proxy accelerates empty/servo intervals using analytic belt
/// transport and a .4 s bin-settling approximation, omitting fruit-fruit contacts.
/// Arrivals and task rewards are shared, but the proxy is incompatible with articulated
/// dynamics and is not physics evidence. Use articulated_physics for contact-based
/// learning/evaluation; it imports link inertias/collision hulls and simulates finite
/// actuator effort, free payload rotation, slip and physical release momentum.
/// Interception options target the oldest/second-oldest observed fruit, inspect
/// it through a supplied classifier, time a descending grasp, lift, traverse and
/// release. Fixed IK servos are not learned torques; the scheduler is trainable.
/// The default perception channel is explicitly SYNTHETIC and intended for fast
/// RL training, with configured class-error probabilities. Supply set_perception()
/// for actual RGB evaluation; unknown predictions never use true-label fallback.
/// With watch_infeed enabled, reset starts at a camera-aimed entry observation pose.
/// Empty-queue waits return there via guarded, rate-limited waypoints; the next
/// decision can interrupt after 0.125 seconds. This orients the real rigid camera,
/// not a virtual pan. Queue state still uses privileged tracking, not RGB detection.
/// Instances are single-writer; callbacks must not mutate the environment.
/// @par Example
/// @code
/// dm::sim::FruitStreamOptions options; options.watch_infeed=true;
/// dm::sim::FruitStreamEnvironment env("ReBot_Arm_DM.urdf",options);
/// env.reset(42);
/// while(env.observation().time<10)env.step(dm::sim::StreamAction::LateFront);
/// @endcode
class FruitStreamEnvironment {
 public:
  /// @brief Load the real arm and start an empty continuing world.
  /// @param urdf Existing B601-DM URDF with its referenced assets.
  /// @param options Valid arrival/capacity/training-channel settings; defaults above.
  /// @throws std::invalid_argument Invalid configuration.
  /// @throws std::runtime_error Asset/IK/model failure, unavailable backend or engine mismatch.
  /// @par Example
  /// See the class example.
  explicit FruitStreamEnvironment(const std::string& urdf, FruitStreamOptions options = {});
  /// @brief Release the world, servo cache and callbacks.
  ~FruitStreamEnvironment();
  /// @brief Exclusive mutable worlds cannot be copied.
  /// @param other Environment which cannot be copied.
  FruitStreamEnvironment(const FruitStreamEnvironment& other) = delete;
  /// @brief Exclusive mutable worlds cannot be assigned.
  /// @param other Environment which cannot be assigned.
  /// @return No value; deleted operation.
  FruitStreamEnvironment& operator=(const FruitStreamEnvironment& other) = delete;
  /// @brief Reset only at the start of an independent training/evaluation stream.
  /// Arrival/appearance/perception RNG streams are independent of policy actions.
  /// @param seed Reproducibility seed; no fruit is forced at time zero.
  /// @return Empty initial frame, preserving perception/record callbacks.
  /// @par Example
  /// See the class example.
  StreamFrame reset(std::uint64_t seed);
  /// @brief Start an independent stream with an explicit arrival intensity.
  /// @param seed Reproducibility seed.
  /// @param arrival_rate Finite intensity in 0..5 fruit/second.
  /// @return Empty initial frame.
  /// @throws std::invalid_argument Invalid rate, before mutation.
  /// @par Example
  /// See reset(); use reset(42,.2) for twelve arrivals/minute in expectation.
  StreamFrame reset(std::uint64_t seed, double arrival_rate);
  /// @brief Execute one interception/wait option while the entire world advances.
  /// Legacy latches stabilize orientation and use tool/payload fixture guards.
  /// Articulated mode keeps fruit dynamic, requires sustained opposing finger
  /// forces, checks actual bin-floor support and permits slip. Non-adjacent arm
  /// collisions are solved, using convex hulls and decomposed finger proxies.
  /// A solver warning rejects the run; reset before reusing a failed simulation.
  /// Optional empty-queue surveillance shares the wait's 0.125-second budget,
  /// including missing-target actions, and never delays arrivals or active picks.
  /// @param action One of five scheduler options.
  /// @return Measured duration, event reward and new state; never terminates itself.
  /// @throws std::invalid_argument Invalid action.
  /// @throws std::runtime_error Solver or callback failure, possibly after partial progress.
  /// @par Example
  /// See the class example.
  StreamTransition step(StreamAction action);
  /// @brief Report feasible wait/interception options from current tracked geometry and servo pose.
  /// Wait is always allowed. Interceptions require an eligible target, a guarded
  /// approach and enough remaining belt travel for approach, inference and descent.
  /// Uses no true class or future arrivals. Owned masks may be passed to StreamPolicy::action();
  /// cached IK/query results may change, but clocks, joints, counters and RNGs do not.
  /// Instances retain the single-writer contract even for this query.
  /// @return Five flags in StreamAction order; at least Wait is true.
  /// @throws std::runtime_error Native IK failure.
  /// @par Example
  /// @code
  /// dm::sim::FruitStreamEnvironment env("ReBot_Arm_DM.urdf");
  /// auto allowed=env.action_mask(); // an empty queue permits only Wait
  /// @endcode
  std::array<bool, 5> action_mask() const;
  /// @brief Encode the current tracked scene for tabular and neural discrete RL policies.
  /// Returns 25 bounded features: queue/release counts, six joints divided by pi,
  /// aperture, three tool coordinates, two leading targets' presence/X/lateral/age,
  /// and five feasibility flags. The table row is the existing 1680-state encoding.
  /// Excludes fruit labels, dimensions, future arrivals and perception-mode flags.
  /// Query caches may change; clocks, random streams, joints and counters do not.
  /// @return Owned normalized features, current mask and table row.
  /// @throws std::runtime_error Native IK failure while querying feasibility.
  /// @par Example
  /// @code
  /// dm::sim::FruitStreamEnvironment env("ReBot_Arm_DM.urdf");
  /// dm::rl::DiscretePolicy policy;
  /// auto action=policy.action(env.learning_observation());
  /// @endcode
  rl::DiscreteObservation learning_observation() const;
  /// @brief Read an owned snapshot without advancing any clocks.
  /// @return Frame with bounded retained payloads and cumulative counters.
  StreamFrame observation() const;
  /// @brief Replace the synthetic channel with actual image classification.
  /// @param classifier Callback at the actual pre-grasp pose, receiving frame and target ID.
  /// Empty restores the explicitly synthetic training channel. No callback may mutate this world.
  /// @par Example
  /// @code
  /// dm::sim::FruitStreamEnvironment env("ReBot_Arm_DM.urdf");
  /// env.set_perception([](const dm::sim::StreamFrame&,std::uint64_t){return
  /// dm::sim::StreamPrediction{-1,0};});
  /// @endcode
  void set_perception(
      std::function<StreamPrediction(const StreamFrame&, std::uint64_t)> classifier);
  /// @brief Record exact tick-aligned world frames, including all motion and waiting.
  /// @param recorder Called on reset and each sample; empty disables recording.
  /// @param fps Positive divisor of 240, at most 60; default 24.
  /// @throws std::invalid_argument Invalid sample frequency.
  /// @throws std::logic_error Recording requested on the analytic training proxy.
  /// @par Example
  /// @code
  /// dm::sim::FruitStreamEnvironment env("ReBot_Arm_DM.urdf");
  /// std::vector<dm::sim::StreamFrame> frames;
  /// env.record([&](const auto& frame){frames.push_back(frame);},15);
  /// env.reset(42);
  /// @endcode
  void record(std::function<void(const StreamFrame&)> recorder, int fps = 24);
  /// @brief Access the fixed arm geometry owner for camera extrinsics and recorded-pose rendering.
  /// @return Borrowed model, valid for this environment's lifetime; do not step/reset it.
  DevArmPickEnvironment& arm_model();
  /// @brief Build all fruit and moving belt marks at an owned recorded frame.
  /// @param frame Previously recorded stream snapshot.
  /// @return Owned CAD Z-up actor scene, without the arm or fixed fixtures.
  /// @par Example
  /// @code
  /// dm::sim::FruitStreamEnvironment env("ReBot_Arm_DM.urdf");
  /// auto actors=env.fruit_scene(env.observation());
  /// @endcode
  cad::Scene fruit_scene(const StreamFrame& frame) const;
  /// @brief Scheduler table size, including queue/urgency/gap/arm-side/pending-release bins.
  /// @return 1680 rows, with five actions per row.
  static std::size_t state_count() noexcept { return 1680; }

 private:
  struct Impl;                 ///< Exclusive world, RNGs, bounded payloads and servo state.
  std::unique_ptr<Impl> impl_; ///< Non-shared native implementation.
};
/// @brief Report the optional articulated engine version, without creating a simulation.
/// @return MuJoCo version string, or empty when built without DATAMUNGE_ENABLE_MUJOCO.
/// @par Example
/// @code
/// auto version=dm::sim::articulated_physics_version();
/// @endcode
/// @see FruitStreamOptions::articulated_physics
std::string articulated_physics_version();
/// @brief Borrow a continuing workcell as a native discrete learning environment.
/// Independent reset seeds map deterministically to a uniform arrival-rate
/// curriculum, separate from policy/replay RNGs. Keeps perception/recorder callbacks;
/// the caller owns the single-writer environment for the returned callbacks' lifetime.
/// @param environment Borrowed workcell; articulated_physics retains contact dynamics in training.
/// @param rate_min Finite lower rate in 0..5; default .08.
/// @param rate_max Finite upper rate in rate_min..5; default .30.
/// @return Borrowing reset/step callbacks with real elapsed option durations.
/// @throws std::invalid_argument Invalid curriculum, before changing the environment.
/// @par Example
/// @code
/// dm::sim::FruitStreamOptions config;config.training_proxy=true;
/// dm::sim::FruitStreamEnvironment env("ReBot_Arm_DM.urdf",config);
/// dm::rl::DiscreteTrainingOptions options;options.windows=2;
/// auto learned=dm::rl::train_discrete(dm::sim::stream_learning_environment(env),options);
/// @endcode
/// @see rl::train_discrete(), FruitStreamEnvironment::learning_observation()
rl::DiscreteEnvironment stream_learning_environment(
    FruitStreamEnvironment& environment,
    double                  rate_min = .08,
    double                  rate_max = .30);

/// @brief Versioned finite scheduler table; learned choices never select a bin from true labels.
/// @par Example
/// @code
/// dm::sim::StreamPolicy policy;
/// policy.q.resize(dm::sim::FruitStreamEnvironment::state_count());
/// policy.save("stream-policy.txt");
/// auto restored=dm::sim::StreamPolicy::load("stream-policy.txt");
/// auto next=restored.action(0);
/// @endcode
struct StreamPolicy {
  std::vector<std::array<double, 5>> q; ///< Finite action values, one row per continuing state.

  /// @brief Choose the first maximizing learned option, without exploration or fallback.
  /// @param state Valid scheduler row.
  /// @return One of five options.
  /// @throws std::out_of_range Invalid row.
  /// @par Example
  /// See the struct example.
  StreamAction action(std::size_t state) const;
  /// @brief Choose the first maximizing feasible option from a caller-supplied mask.
  /// The fixed table/checkpoint shape is unchanged; masked values are ignored.
  /// @param state Valid scheduler row.
  /// @param allowed Five flags in StreamAction order, with at least one true.
  /// @return Maximizing allowed option; ties choose the lowest allowed index.
  /// @throws std::out_of_range Invalid row.
  /// @throws std::invalid_argument All actions are masked.
  /// @par Example
  /// @code
  /// dm::sim::StreamPolicy policy;
  /// policy.q.resize(dm::sim::FruitStreamEnvironment::state_count());
  /// auto next=policy.action(0,{true,false,false,false,false});
  /// @endcode
  StreamAction action(std::size_t state, const std::array<bool, 5>& allowed) const;
  /// @brief Save a complete finite text checkpoint.
  /// @param path Writable destination; parent must exist.
  /// @throws std::invalid_argument Invalid table.
  /// @throws std::runtime_error I/O failure.
  /// @par Example
  /// See the struct example.
  void save(const std::string& path) const;
  /// @brief Load a versioned scheduler checkpoint, validating every value.
  /// @param path Existing checkpoint.
  /// @return Owned action values.
  /// @throws std::runtime_error Invalid checkpoint or I/O failure.
  /// @par Example
  /// See the struct example.
  static StreamPolicy load(const std::string& path);
};
/// @brief Duration-aware Q-learning controls for independent truncated continuing rollouts.
/// No artificial terminal reward is applied at a rollout boundary; the target
/// bootstraps with discount_per_second raised to the measured option duration.
/// @par Example
/// @code
/// dm::sim::StreamTrainingOptions options; options.rollouts=2048;
/// options.trace_decay=.8; options.mask_actions=true;
/// @endcode
struct StreamTrainingOptions {
  std::size_t rollouts{2048}; ///< Positive number of independent continuing windows.
  double seconds{120};  ///< Minimum positive simulated seconds per window; final option completes.
  double rate_min{.10}; ///< Minimum curriculum intensity, fruit/second.
  double rate_max{.22}; ///< Maximum curriculum intensity, at most 5.
  double learning_rate{.15};        ///< Q update fraction, (0,1].
  double discount_per_second{.995}; ///< Time discount in (0,1].
  double epsilon_start{.5};         ///< Initial exploration probability, 0..1.
  double epsilon_end{.05};          ///< Final exploration probability, 0..1.
  std::uint64_t seed{20261108};     ///< Independent exploration, rate and stream seeds.
  double trace_decay{0}; ///< Watkins replacing state-action eligibility decay per decision, 0..1.
                         ///< Traces also decay by elapsed-time discount and clear at nongreedy
                         ///< actions and independent windows; zero is original one-step learning.
  bool mask_actions{false}; ///< Explore/bootstrap/act only through the environment's feasible mask.
};
/// @brief Per-window learning evidence and final frozen scheduler.
/// @par Example
/// @code
/// dm::sim::StreamTrainingResult result;
/// @endcode
struct StreamTrainingResult {
  StreamPolicy                policy;  ///< Learned duration-aware action values.
  std::vector<StreamCounters> counts;  ///< Every rollout's complete outcome counts.
  std::vector<double>         seconds; ///< Actual simulated duration per window.
  std::vector<double>         returns; ///< Undiscounted sum of event rewards per window.
  std::vector<StreamPolicy>
      snapshots; ///< Power-of-two checkpoints from window 128 onward, plus final.
  std::vector<std::size_t>
      snapshot_windows; ///< Completed rollout count corresponding to each checkpoint.
};
/// @brief Train a continuing hierarchical Q scheduler using measured option transitions.
/// Uses independent randomized stream windows, no termination bootstrap, and
/// fixed low-level IK options. The supplied environment's perception channel is
/// retained; the default is an explicit, calibrated synthetic training channel.
/// Optional replacing Watkins traces propagate delayed placement rewards across
/// earlier options. Next behavior is selected before Q updates; nongreedy behavior
/// clears traces. Decay is discount_per_second^elapsed_seconds times trace_decay
/// per decision, with traces below 1e-8 pruned and independent windows cleared.
/// Optional action masking applies to exploration, greedy choice and bootstrapping.
/// @param environment Mutable independent environment; recorder is disabled.
/// @param options Valid positive rollout, learning and curriculum settings.
/// @return Frozen scheduler and every rollout's measured counts/duration/reward.
/// @throws std::invalid_argument Invalid settings, before training mutation.
/// @throws std::runtime_error Native world or callback failure.
/// @par Example
/// @code
/// dm::sim::FruitStreamEnvironment env("ReBot_Arm_DM.urdf");
/// dm::sim::StreamTrainingOptions options; options.rollouts=2; options.seconds=10;
/// auto learned=dm::sim::train_fruit_stream(env,options);
/// @endcode
/// @par References
/// - Sutton, Precup and Singh (1999), Between MDPs and semi-MDPs: A framework
///   for temporal abstraction in reinforcement learning. Artificial Intelligence 112, 181–211.
/// - Sutton and Barto, Reinforcement Learning: An Introduction, draft section 7.6,
///   Watkins's Q(lambda), adapted here to elapsed-time discounting and replacing traces:
///   https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
StreamTrainingResult train_fruit_stream(
    FruitStreamEnvironment& environment,
    StreamTrainingOptions   options = {});
} // namespace datamunge::sim
