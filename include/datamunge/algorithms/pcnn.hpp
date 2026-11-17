#pragma once
/// @file pcnn.hpp
/// @brief Synchronous simplified spiking cortical model with actual cell histories.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned complete pulse maps, firing counts and first-fire times.
/// @par Example
/// @code
/// dm::algorithms::PCNNParameters p;p.iterations=0;std::print("{}\n",dm::algorithms::pcnn_run({{1}},p).first_fire[0][0]); // 0
/// @endcode
struct PCNNResult {
  std::vector<std::vector<std::vector<int>>>
                   pulses;         ///< pulses[n] is the binary pulse map at iteration n.
  std::vector<int> time_signature; ///< Number of neurons firing at each iteration.
  std::vector<std::vector<int>>
      first_fire; ///< Iteration (1-based) each pixel first fired (0 = never).
};

/// @brief Synchronous simplified PCNN iteration count and finite dynamics parameters.
/// @par Example
/// @code
/// const dm::algorithms::PCNNParameters p;std::print("{}\n",p.iterations); // 20
/// @endcode
struct PCNNParameters {
  std::size_t iterations      = 20;   ///< Iteration count, representable by int.
  double      beta            = 0.2;  ///< Linking strength.
  double      threshold_decay = 0.8;  ///< Multiplicative threshold decay \f$\alpha_\Theta\f$.
  double      threshold_gain  = 20.0; ///< Refractory threshold jump \f$V_\Theta\f$ on firing.
  double      initial_threshold =
      1.0; ///< Finite nonnegative initial threshold; need not exceed all stimuli.
};

/// @brief Output and actual-cell-history policies, independently disabled by zero.
/// @par Example
/// @code
/// const dm::algorithms::PCNNOptions o;std::print("{}\n",o.max_trace_records); // 1000000
/// @endcode
struct PCNNOptions {
  std::size_t max_output_entries{
      10000000}; ///< Pulse and first-fire cells plus signature entries; not bytes or peak RAM.
  std::size_t max_trace_records{1000000}; ///< Actual per-iteration per-cell record cap.
};
/// @brief Actual synchronous update operands for one pixel and one iteration.
/// @par Example
/// @code
/// dm::algorithms::PCNNParameters p;p.iterations=1;std::print("{}\n",dm::algorithms::pcnn_trace({{.9}},p).cells[0].pulse); // 1
/// @endcode
struct PCNNCell {
  std::size_t iteration; ///< One-based iteration.
  std::size_t row;       ///< Pixel row.
  std::size_t column;    ///< Pixel column.
  double      linking;   ///< Sum of previous iteration's eight-neighbor pulses, no self.
  double      activity;  ///< Stimulus times (1+beta*linking).
  double      before;    ///< Threshold before decay.
  double      decayed;   ///< Threshold after decay, used by strict comparison.
  int         pulse;     ///< One exactly when activity > decayed.
  double      after;     ///< Threshold after optional firing jump.
};
/// @brief Owned ordinary result and actual producing per-cell records.
/// @par Example
/// @code
/// dm::algorithms::PCNNParameters p;p.iterations=1;std::print("{}\n",dm::algorithms::pcnn_trace({{.9}},p).cells.size()); // 1
/// @endcode
struct PCNNTrace {
  PCNNResult            result; ///< Actual pulses, first-fire times and signature.
  std::vector<PCNNCell> cells;  ///< Actual row-major updates, grouped by iteration.
};
/// @brief Run the native simplified synchronous pulse-coupled model.
/// Every iteration uses only previous pulses for linking, decays threshold,
/// compares strictly, then adds gain after firing. Intensities must be finite
/// and nonnegative; values above one are allowed without clipping. Empty or
/// zero-width rectangular images return an empty result after full validation.
/// O(iterations*rows*cols) work and retained pulse storage. Input is unchanged;
/// no shared state or retained references. This is no trained or validated segmenter.
/// @param image Rectangular nonnegative finite stimulus rows.
/// @param params Iterations <= INT_MAX; finite beta/gain/initial threshold >=0;
/// threshold_decay in [0,1]. Initial threshold need not exceed the maximum input.
/// @return Owned actual result; default output-entry cap is ten million.
/// @throws std::invalid_argument If shapes, stimuli or parameters are invalid.
/// @throws std::length_error If index/storage/default output ranges are exceeded.
/// @throws std::overflow_error If finite activity or threshold arithmetic overflows.
/// @par Example
/// @code
/// dm::algorithms::PCNNParameters p;p.iterations=1;std::print("{}\n",dm::algorithms::pcnn_run({{.9}},p).time_signature); // [1]
/// @endcode
/// @par References
/// - J. L. Johnson and M. L. Padgett (1999), PCNN models and applications,
///   IEEE Transactions on Neural Networks 10, 480-498, doi:10.1109/72.761706.
PCNNResult pcnn_run(const std::vector<std::vector<double>>& image, const PCNNParameters& params);
/// @brief Capture each actual update from pcnn_run's shared synchronous kernel.
/// O(iterations*pixel count) extra records; all count policies checked before
/// iterations begin. No replay or independent recurrence populates the history.
/// @param image Complete rectangular nonnegative finite stimulus rows.
/// @param params Same dynamics constraints as pcnn_run().
/// @param options Output-entry and retained-cell policies; zero disables each.
/// @return Owned result and every actual update record.
/// @throws std::invalid_argument If shapes, stimuli or parameters are invalid.
/// @throws std::length_error If representable storage or selected policies are exceeded.
/// @throws std::overflow_error If activity or threshold arithmetic overflows.
/// @par Example
/// @code
/// dm::algorithms::PCNNParameters p;p.iterations=2;std::print("{}\n",dm::algorithms::pcnn_trace({{.9}},p).cells.size()); // 2
/// @endcode
PCNNTrace pcnn_trace(
    const std::vector<std::vector<double>>& image,
    const PCNNParameters&                   params,
    PCNNOptions                             options = {});
} // namespace datamunge::algorithms
