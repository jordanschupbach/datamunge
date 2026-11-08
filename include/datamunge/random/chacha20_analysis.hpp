#pragma once
/// @file chacha20_analysis.hpp
/// @brief Reproducible seed expansion and bounded inspection of actual buffered ChaCha20 draws.
#include <datamunge/random/random.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::random {
/// @brief Owned location, capacity and value of one actual ChaCha20 raw draw.
/// The location describes the consumed word, not the next draw. No key or nonce
/// is copied into a record; values are still raw keystream and may be sensitive.
/// @par Example
/// @code
/// dm::random::ChaCha20 rng(std::array<std::uint32_t,8>{});
/// const auto t=dm::random::chacha20_trace(rng,1);
/// std::print("{} {} {:016x}\n",t[0].counter,t[0].word_index,t[0].value); // 0 0 903df1a0ade0b876
/// @endcode
/// @see chacha20_trace(), ChaCha20::remaining_u64()
struct ChaCha20Draw {
  std::uint32_t counter{0}; ///< Actual RFC block counter that supplied this draw.
  std::size_t word_index{0}; ///< uint64 index within the block, from zero through seven.
  std::uint64_t remaining_before{0}; ///< Available raw draws immediately before consumption, including this one.
  std::uint64_t value{0}; ///< Actual next_u64() output, low 32-bit word first.
};
/// @brief Expand a 64-bit simulation seed using four SplitMix64 words, low half first.
/// Matches the ChaCha20 seed constructor exactly. All 64-bit seeds are accepted,
/// but at most 2^64 different keys can result: this adds no entropy. Returns an
/// owned array, does not mutate the caller and uses constant work/storage without
/// allocation. Safe for concurrent independent calls; unsuitable for secret keys.
/// @param seed Reproducible simulation seed, including zero.
/// @return Eight uint32 words in RFC key order.
/// @par Example
/// @code
/// const auto key=dm::random::chacha20_seed_key(0);
/// std::print("{:08x} {:08x}\n",key[0],key[1]); // 7b1dcdaf e220a839
/// @endcode
/// @par References
/// - S. Vigna, SplitMix64 reference, https://prng.di.unimi.it/splitmix64.c
///   Four-output low/high packing follows this library's legacy seed constructor.
/// @see ChaCha20
std::array<std::uint32_t,8> chacha20_seed_key(std::uint64_t seed);
/// @brief Consume raw engine draws while recording their actual buffered stream locations.
/// No cipher is replayed. Before drawing, checks count against remaining capacity
/// and reserves all records; any validation/allocation failure leaves both raw
/// state and normal() cache unchanged. Successful draws preserve the normal cache
/// like next_u64(). Locations follow from remaining capacity: for R>0, the counter
/// is UINT32_MAX-(R-1)/8 and the word index is (8-R%8)%8. Owns all output records;
/// count zero is an empty no-op even at exhaustion. O(count) work/storage.
/// This is a public-test diagnostic, not an independent stream or entropy source.
/// @param engine Mutable ChaCha20 engine advanced by exactly count raw words.
/// @param count Number of raw uint64 draws to record; zero is permitted.
/// @return Owned chronological records containing the actual returned values.
/// @throws std::length_error If count exceeds remaining raw capacity or vector capacity, before mutation.
/// @throws std::bad_alloc If record reservation fails, before mutation.
/// @par Example
/// @code
/// dm::random::ChaCha20 rng(std::array<std::uint32_t,8>{},{},0xffffffffU);
/// const auto t=dm::random::chacha20_trace(rng,8);
/// std::print("{} {}\n",t.back().word_index,rng.remaining_u64()); // 7 0
/// @endcode
/// @see ChaCha20::next_u64(), ChaCha20::remaining_u64(), ChaCha20Draw
std::vector<ChaCha20Draw> chacha20_trace(ChaCha20& engine,std::size_t count);
} // namespace datamunge::random
