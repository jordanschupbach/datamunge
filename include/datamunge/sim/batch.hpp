#pragma once
/// @file batch.hpp
/// @brief Independent native worlds with reusable parallel stepping and masked resets.
#include <cstdint>
#include <datamunge/sim/world.hpp>
#include <exception>
/// @brief Native rigid-body physics and batched reinforcement-learning rollouts.
namespace datamunge::sim {
/// @brief Fixed-size collection of independent native worlds for CPU rollouts.
/// Construction deep-copies a prototype, including pending forces and solver state.
/// Stepping/reset reuse reserved storage; capture_reset allocates during setup.
/// Each world has an independent deterministic random stream. Streams advance
/// across episode resets; reseeding is explicit. Single-writer: do not mutate
/// worlds concurrently with a batch operation. GPU physics is not yet supported.
/// @par Example
/// @code
/// dm::sim::WorldOptions
/// limits;limits.body_capacity=8;limits.pair_capacity=16;limits.contact_capacity=16; dm::sim::World
/// prototype(limits); dm::sim::BatchWorld batch(prototype,4,dm::compute::Context({},2),42);
/// batch.step(1./60);batch.reset({1,0,0,1});
/// @endcode
class BatchWorld {
 public:
  /// @brief Fork a prototype into count independently resettable worlds.
  /// @param prototype Configured world; its capacities are reserved in each copy.
  /// @param count Positive number of independent environments.
  /// @param context CPU worker context, default one lane; copies share its pool.
  /// @param seed Batch random seed, default zero; streams also depend on stable environment index.
  /// @throws std::invalid_argument Zero count or non-CPU context.
  /// @throws std::bad_alloc Insufficient memory for configured capacities.
  /// @par Example
  /// @code
  /// dm::sim::World prototype;dm::sim::BatchWorld batch(prototype,2);
  /// @endcode
  explicit BatchWorld(
      const World&     prototype,
      std::size_t      count,
      compute::Context context = compute::Context{},
      std::uint64_t    seed    = 0);
  /// @brief Return the fixed number of independent worlds.
  /// @return Positive environment count.
  std::size_t size() const noexcept { return worlds_.size(); }
  /// @brief Access a world between batch operations for setup, actions or queries.
  /// Topology edits require capture_reset before resetting this world again.
  /// @param index Environment index in [0,size()).
  /// @return Borrowed world valid until batch destruction.
  /// @throws std::out_of_range Invalid index.
  /// @par Example
  /// @code
  /// dm::sim::World prototype;dm::sim::BatchWorld batch(prototype,2);
  /// const auto n=batch.world(0).body_count();
  /// @endcode
  World& world(std::size_t index);
  /// @brief Borrow a world read-only; see mutable world() for an example.
  /// @param index Environment index in [0,size()).
  /// @return Borrowed const world valid until batch destruction.
  /// @throws std::out_of_range Invalid index.
  const World& world(std::size_t index) const;
  /// @brief Capture one world's current state as its future episode reset target.
  /// This setup operation allocates a snapshot and clears its episode step count.
  /// @param index Environment index in [0,size()).
  /// @throws std::out_of_range Invalid index.
  /// @par Example
  /// @code
  /// dm::sim::World prototype;dm::sim::BatchWorld batch(prototype,1);
  /// batch.world(0).add_body({});batch.capture_reset(0);batch.reset();
  /// @endcode
  void capture_reset(std::size_t index);
  /// @brief Restore selected worlds and clear their episode step counters.
  /// All selected snapshots are validated before mutation. Random streams retain
  /// their current state. Worlds never reset automatically as a side effect of stepping.
  /// @param mask Empty selects all; otherwise exactly size() bytes, each zero or one.
  /// @throws std::invalid_argument Invalid mask or stale reset snapshot.
  /// @par Example
  /// @code
  /// dm::sim::World prototype;dm::sim::BatchWorld batch(prototype,2);
  /// batch.step(1./60);batch.reset({1,0});
  /// @endcode
  void reset(const std::vector<std::uint8_t>& mask = {});
  /// @brief Advance selected worlds on persistent workers, rolling back the batch on failure.
  /// All selected worlds run once. Errors are collected after every worker finishes;
  /// every successfully advanced world then rewinds, counters remain unchanged, and
  /// the lowest-index failing world's exception is rethrown.
  /// @param delta Finite positive seconds at most 1/30.
  /// @param mask Empty selects all; otherwise exactly size() binary bytes.
  /// @throws std::invalid_argument Invalid delta or mask, before advancement.
  /// @throws std::exception Native solver error after whole-batch rollback.
  /// @par Example
  /// @code
  /// dm::sim::World prototype;dm::sim::BatchWorld batch(prototype,2);
  /// batch.step(1./60,{0,1});
  /// @endcode
  void step(double delta, const std::vector<std::uint8_t>& mask = {});
  /// @brief Read successful steps since the last reset/capture for one world.
  /// @param index Environment index in [0,size()).
  /// @return Episode step count; masked-out or failed steps do not increment it.
  /// @throws std::out_of_range Invalid index.
  std::uint64_t steps(std::size_t index) const;
  /// @brief Reinitialize random streams without changing physics or episode counters.
  /// @param value Batch seed; identical seed and index reproduce identical CPU draws.
  /// @par Example
  /// @code
  /// dm::sim::World prototype;dm::sim::BatchWorld batch(prototype,2);batch.seed(42);
  /// @endcode
  void seed(std::uint64_t value) noexcept;
  /// @brief Draw one IEEE double in [0,1) from one environment's stream.
  /// Draws in other environments and worker scheduling do not affect this stream.
  /// @param index Environment index in [0,size()).
  /// @return Uniform 53-bit sample including zero and excluding one.
  /// @throws std::out_of_range Invalid index.
  /// @par Example
  /// @code
  /// dm::sim::World prototype;dm::sim::BatchWorld batch(prototype,1);
  /// const double u=batch.uniform(0);
  /// @endcode
  /// @par References
  /// - Steele, Lea and Flood (2014), Fast Splittable Pseudorandom Number Generators,
  ///   https://gee.cs.oswego.edu/dl/papers/oopsla14.pdf. This uses fixed-gamma
  ///   SplitMix64 streams seeded by environment index, rather than the paper's split API.
  double uniform(std::size_t index);

 private:
  /// @brief Validate a complete mask before touching any world.
  /// @param mask Empty or size() binary values.
  void                                validate_mask(const std::vector<std::uint8_t>& mask) const;
  compute::Context                    context_;  ///< Persistent CPU execution owner.
  std::vector<std::unique_ptr<World>> worlds_;   ///< Independent solver states.
  std::vector<PhysicsState>           initial_;  ///< Reset targets tied to each world.
  std::vector<std::uint64_t>          steps_;    ///< Successful episode steps.
  std::vector<std::uint64_t>          random_;   ///< Independent SplitMix64 states.
  std::vector<std::uint8_t>           advanced_; ///< Reusable completion flags.
  std::vector<std::exception_ptr>     errors_;   ///< Reusable per-environment errors.
};
} // namespace datamunge::sim
