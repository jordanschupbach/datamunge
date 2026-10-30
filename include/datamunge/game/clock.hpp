#pragma once
/// @file clock.hpp
/// @brief Bounded fixed-step time accumulation independent of windows and rendering.
#include <cstddef>
/// @brief Library-only game scenes, fixed clocks and native behavior lifecycle.
namespace datamunge::game {
/// @brief Result of accumulating one render-frame duration into fixed simulation ticks.
/// @par Example
/// @code
/// const dm::game::TickBatch batch;
/// std::print("{}\n",batch.steps); // 0
/// @endcode
struct TickBatch {
  std::size_t steps{
      0};  ///< Fixed updates to execute, bounded by the clock's capacity.
  double alpha{
      0};  ///< Fraction of one tick left for presentation interpolation, in [0,1).
  double dropped_seconds{
      0};  ///< Incoming duration discarded to prevent an unbounded catch-up loop.
};
/// @brief Fixed-step accumulator; callers execute returned ticks in their own application loop.
/// This library owns no window, process main loop or wall clock. Each instance is
/// single-writer and stores only the unconsumed fraction of a fixed tick.
/// @par Example
/// @code
/// dm::game::FixedClock clock(.01,4);
/// std::print("{}\n",clock.advance(.025).steps); // 2
/// @endcode
class FixedClock {
 public:
  /// @brief Create an empty accumulator.
  /// @param step Fixed seconds in [1e-6,1], default 1/60.
  /// @param max_steps Maximum catch-up ticks per advance, in [1,10000], default eight.
  /// @throws std::invalid_argument Invalid step or capacity.
  /// @par Example
  /// @code
  /// const dm::game::FixedClock clock(.01,4);
  /// std::print("{}\n",clock.step()); // 0.01
  /// @endcode
  explicit FixedClock(double step = 1. / 60, std::size_t max_steps = 8);
  /// @brief Accumulate elapsed time and consume complete fixed ticks.
  /// @param elapsed Finite nonnegative seconds, bounded to step*max_steps before accumulation.
  /// @return Tick count, interpolation fraction and discarded incoming seconds.
  /// @throws std::invalid_argument Negative/nonfinite elapsed; accumulator is unchanged.
  /// @par Example
  /// @code
  /// dm::game::FixedClock clock(.01,4);
  /// const auto batch=clock.advance(.025);
  /// std::print("{} {:.1f}\n",batch.steps,batch.alpha); // 2 0.5
  /// @endcode
  TickBatch advance(double elapsed);
  /// @brief Configured fixed simulation seconds per tick.
  /// @return Positive step supplied to the constructor.
  double step() const noexcept { return step_; }
  /// @brief Clear pending fractional time without changing controls.
  /// @par Example
  /// @code
  /// dm::game::FixedClock clock;clock.advance(.001);clock.reset();
  /// @endcode
  void reset() noexcept { remainder_ = 0; }

 private:
  double step_;            ///< Fixed tick duration.
  std::size_t max_steps_;  ///< Positive catch-up bound.
  double remainder_{0};    ///< Unconsumed sub-tick seconds.
};
}  // namespace datamunge::game
