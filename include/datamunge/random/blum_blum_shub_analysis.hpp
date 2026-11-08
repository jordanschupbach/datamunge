#pragma once
/// @file blum_blum_shub_analysis.hpp
/// @brief Canonical BBS bit transitions and bounded modular-state orbit inspection.
#include <datamunge/random/random.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
/// @brief Native pseudorandom engines, common samplers and explicit finite-state inspections.
namespace datamunge::random {
/// @brief Owned before/after modular state and low bit for one actual engine transition.
/// The bit belongs to after, not before. No engine or input is borrowed by a record.
/// @par Example
/// @code
/// dm::random::BlumBlumShub rng(3,7,11);
/// const auto trace=dm::random::blum_blum_shub_trace(rng,1);
/// std::print("{} {} {}\n",trace[0].before,trace[0].after,trace[0].bit); // 9 4 0
/// @endcode
/// @see blum_blum_shub_trace(), BlumBlumShub
struct BlumBlumShubStep {
  std::uint64_t before{0}; ///< State immediately before the square.
  std::uint64_t after{0}; ///< State immediately after the square/modulo reduction.
  std::uint64_t bit{0}; ///< Low bit of after, either zero or one.
};
/// @brief Owned bounded state path, optionally closed by an actual repeated modular state.
/// states begins at the inspected engine state. On closure it includes the closing
/// duplicate, so states.size()=transient+period+1. Unknown closure has period and
/// transient zero; this is a budget result, not evidence of an infinite orbit.
/// @par Example
/// @code
/// dm::random::BlumBlumShub rng(3,7,11);
/// const auto orbit=dm::random::blum_blum_shub_orbit(rng,4);
/// std::print("{} {}\n",orbit.period,orbit.states); // 4 [9, 4, 16, 25, 9]
/// @endcode
/// @see blum_blum_shub_orbit()
struct BlumBlumShubOrbit {
  std::vector<std::uint64_t> states; ///< Initial state and every observed transition, including a closing repeat.
  bool closed{false}; ///< Whether a previously observed state was reached within the budget.
  std::size_t transient{0}; ///< Index of the first occurrence of the repeated state, when closed.
  std::size_t period{0}; ///< Positive repeated-state distance, when closed.
};
/// @brief Advance a BBS engine while recording its actual canonical bit transitions.
/// Uses next_bit(), the same transition used by next_u64(). Exactly count squares;
/// output records are owned and leave the normal() cache intact. Reserves storage
/// before advancing, so an allocation/length failure leaves engine state unchanged.
/// O(count) work/storage with fixed-width arithmetic; count zero is an empty no-op.
/// @param engine Mutable engine advanced by count individual bits.
/// @param count Number of transitions to record; zero is permitted.
/// @return Owned chronological before/after/bit records.
/// @throws std::length_error If count exceeds the vector's record capacity.
/// @throws std::bad_alloc If record allocation fails, before state mutation.
/// @par Example
/// @code
/// dm::random::BlumBlumShub rng(3,7,11);
/// const auto trace=dm::random::blum_blum_shub_trace(rng,4);
/// std::print("{} {}\n",trace.size(),rng.state()); // 4 9
/// @endcode
/// @par References
/// - L. Blum, M. Blum and M. Shub (1986), A Simple Unpredictable Pseudo-Random
///   Number Generator, doi:10.1137/0215025; finite uint64 inspection here.
/// @see BlumBlumShub::next_bit(), blum_blum_shub_orbit()
std::vector<BlumBlumShubStep> blum_blum_shub_trace(BlumBlumShub& engine,std::size_t count);
/// @brief Inspect a copy of the engine until its first repeated state or an explicit step budget.
/// The caller's raw state and distribution cache are unchanged. Each step uses
/// next_bit(); a hash table records first visits. With B observed transitions,
/// expected O(B) time and O(B) owned storage; hash collisions may worsen time.
/// Prime-factor assumptions imply purely periodic admissible states, but this
/// inspector also reports transients for structurally accepted composite factors.
/// @param engine Engine whose current state is the initial vertex; not advanced.
/// @param max_steps Maximum transitions to inspect, including a closing step; zero returns only the initial state.
/// @return Owned path with explicit closure, transient and period; open paths have both lengths zero.
/// @par Example
/// @code
/// dm::random::BlumBlumShub rng(3,7,11);
/// const auto orbit=dm::random::blum_blum_shub_orbit(rng,3);
/// std::print("{} {} {}\n",orbit.closed,orbit.states.size(),rng.state()); // false 4 9
/// @endcode
/// @par References
/// - L. Blum, M. Blum and M. Shub (1986), A Simple Unpredictable Pseudo-Random
///   Number Generator, doi:10.1137/0215025. First-visit orbit inspection is deterministic bookkeeping.
/// @see blum_blum_shub_trace(), BlumBlumShubOrbit
BlumBlumShubOrbit blum_blum_shub_orbit(const BlumBlumShub& engine,std::size_t max_steps);
} // namespace datamunge::random
