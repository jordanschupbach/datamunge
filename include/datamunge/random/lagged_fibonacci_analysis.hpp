#pragma once
/// @file lagged_fibonacci_analysis.hpp
/// @brief Actual consuming ring traces and nonmutating exact finite parity-state orbit inspection.
#include <cstddef>
#include <cstdint>
#include <datamunge/random/random.hpp>
#include <vector>
namespace datamunge::random {
/// @brief Owned complete before/after physical rings and every actual consumed modular transition.
/// captured_entries=2*k+count counts both full boundary-ring words and scalar
/// transition records, not fields/capacities/bytes/RAM. Allocation/admission of
/// both rings and all records finishes before the caller engine is advanced.
/// No input engine references or replay; normal cache is left intact.
/// @par Example
/// @code
/// dm::random::LaggedFibonacci64 g(2026,2,5);const auto t=dm::random::lagged_fibonacci_trace(g,5);
/// std::print("{} {}\n",t.steps.size(),t.final_index); // 5 0
/// @endcode
/// @see lagged_fibonacci_trace()
struct LaggedFibonacciTrace {
  std::size_t short_lag{0};     ///< Actual configured j, retained by value.
  std::size_t long_lag{0};      ///< Actual k/complete ring word count.
  std::size_t initial_index{0}; ///< Actual oldest-word slot before consuming transitions.
  std::size_t final_index{0};   ///< Actual oldest-word slot after complete consumption.
  std::vector<std::uint64_t>       initial_state; ///< Owned full original physical ring.
  std::vector<std::uint64_t>       final_state;   ///< Owned full actual final physical ring.
  std::vector<LaggedFibonacciStep> steps; ///< Every actual transition, chronological, no replay.
  std::size_t captured_entries{0}; ///< Complete2*k boundary words plus actual transition records.
};
/// @brief Owned normalized oldest-first parity masks until the first repeat or explicit cap.
/// Bit i denotes parity of the i-th oldest ring word, eliminating physical index
/// rotation. This is an exact finite-state projection, not the full64-bit engine
/// period or a randomness certificate. complete=false has no inferred period.
/// All masks are distinct; the repeated closing mask is identified by offset.
/// @par Example
/// @code
/// dm::random::LaggedFibonacci64 g(2026,2,5);const auto o=dm::random::lagged_fibonacci_parity_orbit(g,1);
/// std::print("{}\n",o.masks.size()); // 1
/// @endcode
/// @see lagged_fibonacci_parity_orbit()
struct LaggedFibonacciParityOrbit {
  std::size_t                short_lag{0}; ///< Original configured j.
  std::size_t                long_lag{0};  ///< Original configured k, constrained to<=63 for masks.
  std::vector<std::uint64_t> masks; ///< Owned chronological distinct oldest-first parity states.
  bool                       complete{false}; ///< Whether an actual repeated parity mask was found.
  std::size_t repeat_offset{0};               ///< Repeated mask's earlier index only when complete.
  std::size_t period{0}; ///< Exact observed parity-cycle length if complete, otherwise0.
};
/// @brief Consume exactly count actual ring updates after admitting/allocating complete output.
/// Captures original/final ring and every next_step() record. Caller engine advances
/// count times; no prefix replay or second generation. Known count/quota/allocation
/// failures occur before advancement; subsequent fixed-ring updates/copies do not
/// allocate or throw. O(k+count) time/owned capture. Inherited normal cache untouched.
/// @param generator Borrowed mutable engine, advanced exactly count times on success.
/// @param count Complete raw transition count, zero allowed.
/// @param max_entries Complete2*k+count ceiling, default1000000, zero disables optional cap.
/// @return Owned complete actual observations and boundary snapshots.
/// @throws std::length_error On count/geometry or enabled quota before advancement.
/// @throws std::bad_alloc On complete capture allocation before advancement.
/// @par Example
/// @code
/// dm::random::LaggedFibonacci64 g(1,2,5);const auto t=dm::random::lagged_fibonacci_trace(g,3,13);
/// std::print("{} {}\n",t.captured_entries,g.ring_index()); // 13 3
/// @endcode
/// @par References
/// - Knuth, The Art of Computer Programming, Volume2: Seminumerical Algorithms, third edition(1997), chapter3(Random Numbers).
/// @see LaggedFibonacciTrace, LaggedFibonacci64::next_step()
LaggedFibonacciTrace lagged_fibonacci_trace(
    LaggedFibonacci64& generator,
    std::size_t        count,
    std::size_t        max_entries = 1000000);
/// @brief Inspect the exact parity-state orbit of an independent complete engine copy.
/// Require k<=63 so oldest-first parities fit one uint64 mask. At each actual
/// copied-engine transition inspect the current mask, stop on first repeated mask,
/// or return incomplete before retaining a new mask beyond max_states. Original
/// engine/raw position/distribution cache unchanged. O(k*S) inspection work and
/// O(k+S) storage for S distinct visited masks; does not claim full engine period.
/// @param generator Borrowed immutable engine at arbitrary raw stream position.
/// @param max_states Distinct-mask ceiling, default1000000; zero disables optional cap.
/// @return Owned distinct masks and exact first-repeat cycle if complete.
/// @throws std::invalid_argument On k>63.
/// @throws std::length_error On mask/counter/container representability limits.
/// @throws std::bad_alloc On independent ring/map/output allocation failure.
/// @par Example
/// @code
/// dm::random::LaggedFibonacci64 g(1,2,5);const auto before=g.ring_index();
/// const auto o=dm::random::lagged_fibonacci_parity_orbit(g,1);
/// std::print("{} {}\n",g.ring_index()==before,o.masks.size()); // true 1
/// @endcode
/// @par References
/// - Knuth, The Art of Computer Programming, Volume2: Seminumerical Algorithms, third edition(1997), chapter3(Random Numbers).
/// @see LaggedFibonacciParityOrbit, lagged_fibonacci_trace()
LaggedFibonacciParityOrbit lagged_fibonacci_parity_orbit(
    const LaggedFibonacci64& generator,
    std::size_t              max_states = 1000000);
} // namespace datamunge::random
