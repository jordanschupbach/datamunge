#pragma once
/// @file pearson_hashing.hpp
/// @brief Checked permutation-table byte hashing; an educational noncryptographic hash.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Return the native fixed permutation, in preserved historical byte order.
/// @return Process-lifetime immutable 256-byte table; concurrent reads are safe.
/// @par Example
/// @code
/// std::print("{}\n",int(dm::algorithms::pearson_default_table()[0])); // 98
/// @endcode
const std::array<std::uint8_t, 256>& pearson_default_table();
/// @brief One actual byte lookup in an eight-bit hash.
/// @par Example
/// @code
/// std::print("{}\n",int(dm::algorithms::pearson_hash_trace("a").steps[0].byte)); // 97
/// @endcode
struct PearsonStep {
  std::size_t  index;       ///< Zero-based input byte position.
  std::uint8_t byte;        ///< Unsigned byte, including zeros and high bytes.
  std::uint8_t before;      ///< Accumulator before lookup.
  std::uint8_t table_index; ///< Actual before XOR byte index.
  std::uint8_t after;       ///< Actual table lookup result.
};
/// @brief Owned actual hash byte and optional lookup history.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pearson_hash_trace("a").steps.size()); // 1
/// @endcode
struct PearsonTrace {
  std::uint8_t             hash{0}; ///< Final accumulator, zero for empty input.
  std::vector<PearsonStep> steps;   ///< Actual producing lookups, not replay.
};
/// @brief Hash complete bytes from accumulator zero with a validated permutation.
/// A single fixed-position byte substitution in an equal-length message changes
/// the output: each subsequent fixed-byte step is bijective. Multiple changes
/// can collide; only 256 outputs exist. This is not a cryptographic hash.
/// O(256+n) time, O(256) validation state; borrowed inputs are unchanged.
/// @param data Complete binary string, possibly empty.
/// @param table Permutation of all byte values; default native fixed table.
/// @return Final eight-bit accumulator, zero for empty data.
/// @throws std::invalid_argument If table contains duplicate values.
/// @par Example
/// @code
/// std::print("{}\n",int(dm::algorithms::pearson_hash(""))); // 0
/// @endcode
/// @par References
/// - P. K. Pearson (1990), Fast Hashing of Variable-Length Text Strings,
///   Communications of the ACM 33, 677-680, doi:10.1145/78973.78978.
std::uint8_t pearson_hash(
    const std::string&                   data,
    const std::array<std::uint8_t, 256>& table = pearson_default_table());
/// @brief Retain each actual byte lookup of pearson_hash's producing loop.
/// @param data Complete binary string, unchanged.
/// @param table Complete byte permutation.
/// @param max_steps Retained lookup cap, default one million; zero disables policy.
/// @return Owned final hash and n actual lookup records; O(n) extra history.
/// @throws std::invalid_argument If table is not a permutation.
/// @throws std::length_error If retained lookup policy is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pearson_hash_trace("hash").steps.size()); // 4
/// @endcode
PearsonTrace pearson_hash_trace(
    const std::string&                   data,
    const std::array<std::uint8_t, 256>& table     = pearson_default_table(),
    std::size_t                          max_steps = 1000000);
/// @brief Concatenate width offset-first-byte Pearson lanes in big-endian order.
/// Empty data yields zero in every lane. Lane zero agrees with pearson_hash()
/// for nonempty data. This is a native extension, not a 64-bit security claim.
/// @param data Complete binary string, unchanged.
/// @param width Number of bytes in [1,8].
/// @param table Complete byte permutation.
/// @return Owned width-byte value in the low bits of uint64; O(width*n+256) time.
/// @throws std::invalid_argument If width or table is invalid.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pearson_hash_wide("",8)); // 0
/// @endcode
std::uint64_t pearson_hash_wide(
    const std::string&                   data,
    int                                  width,
    const std::array<std::uint8_t, 256>& table = pearson_default_table());
} // namespace datamunge::algorithms
