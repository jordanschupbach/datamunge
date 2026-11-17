#pragma once
/// @file sfc64_analysis.hpp
/// @brief Actual SFC64 raw-word histories and finite sample bit/byte counts.
#include <array>
#include <cstddef>
#include <datamunge/random/random.hpp>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::random {
/// @brief Complete raw sample and optional capture policies; zero disables a size cap.
/// @par Example
/// @code
/// const dm::random::Sfc64Options value{};
/// std::print("{}\n",value.max_draws); // 1000000
/// @endcode
struct Sfc64Options {
  std::size_t max_draws{1000000};         ///< Complete requested raw words.
  std::size_t max_trace_entries{1000000}; ///< Actual retained before/output/after step records.
};
/// @brief Owned actual raw sample and exact finite-prefix count summaries.
/// @par Example
/// @code
/// const dm::random::Sfc64Run value{};
/// std::print("{}\n",value.outputs.size()); // 0
/// @endcode
struct Sfc64Run {
  std::vector<std::uint64_t>  outputs; ///< Actual chronological complete raw words.
  std::array<std::size_t, 64> ones{};  ///< Actual one-bit counts, low bit first.
  std::array<std::size_t, 256>
      low_bytes{}; ///< Actual least-significant-byte counts, including empty bins.
  std::array<std::uint64_t, 4> initial{}; ///< Actual seeded raw state before sampling.
  std::array<std::uint64_t, 4> final{};   ///< Actual raw state after the complete sample.
  std::vector<Sfc64Step>       steps;     ///< Optional actual same-execution raw transitions.
};
/// @brief Sample the historical native SFC64 raw stream.
/// Uses a fresh local Sfc64(seed) and exactly count raw steps. Bit/low-byte counters are incremented from each actually produced word. Empty samples return zero counts and matching initial/final snapshots. Complete output/capture geometry checked before stepping. O(64*count) selected bit-inspection work and O(count) intrinsic output, optional one owned state record per draw. No distribution method consumes additional words. Finite-prefix frequencies and absent detected short periods do not prove uniformity, independence, a full period or cryptographic security. Independent calls share no state.
/// @param seed Complete historical local engine seed.
/// @param count Complete requested raw words.
/// @param options Complete output/capture bounds.
/// @return Owned actual words, exact finite-prefix counts, raw endpoint states and optional producing records.
/// @throws std::length_error Complete sample/capture/container geometry exceeded.
/// @par Example
/// @code
/// auto r=dm::random::sfc64_run(0xC0FFEEULL,1);
/// std::print("{}\n",r.outputs[0]); // 13215376215447164276
/// @endcode
/// @par References
/// - Doty-Humphrey, PractRand SFC family. https://pracrand.sourceforge.net/ ; local historical seeding explicitly preserved.
/// @see Sfc64::next_trace(), bit_periods_of()
Sfc64Run sfc64_run(std::uint64_t seed, std::size_t count, const Sfc64Options& options = {});
/// @brief Sample the historical native SFC64 raw stream with actual producing state transitions.
/// Uses a fresh local Sfc64(seed) and exactly count raw steps. Bit/low-byte counters are incremented from each actually produced word. Empty samples return zero counts and matching initial/final snapshots. Complete output/capture geometry checked before stepping. O(64*count) selected bit-inspection work and O(count) intrinsic output, optional one owned state record per draw. No distribution method consumes additional words. Finite-prefix frequencies and absent detected short periods do not prove uniformity, independence, a full period or cryptographic security. Independent calls share no state.
/// @param seed Complete historical local engine seed.
/// @param count Complete requested raw words.
/// @param options Complete output/capture bounds.
/// @return Owned actual words, exact finite-prefix counts, raw endpoint states and optional producing records.
/// @throws std::length_error Complete sample/capture/container geometry exceeded.
/// @par Example
/// @code
/// auto r=dm::random::sfc64_trace(0xC0FFEEULL,1);
/// std::print("{}\n",r.outputs[0]); // 13215376215447164276
/// @endcode
/// @par References
/// - Doty-Humphrey, PractRand SFC family. https://pracrand.sourceforge.net/ ; local historical seeding explicitly preserved.
/// @see Sfc64::next_trace(), bit_periods_of()
Sfc64Run sfc64_trace(std::uint64_t seed, std::size_t count, const Sfc64Options& options = {});
} // namespace datamunge::random
