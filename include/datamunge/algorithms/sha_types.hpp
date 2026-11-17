#pragma once
/// @file sha_types.hpp
/// @brief Shared checked digest geometry and full-support avalanche summaries.
#include <cstddef>
#include <string>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Complete message and optional native history bounds; zero disables a size policy.
/// @par Example
/// @code
/// const dm::algorithms::SHAOptions value{};
/// std::print("{}\n",value.max_bytes); // 1000000
/// @endcode
struct SHAOptions {
  std::size_t max_bytes{1000000}; ///< Complete original raw message-byte cap.
  std::size_t max_trace_entries{
      1000000}; ///< Captured schedule words, chain words and round records with working-state words.
};
/// @brief Owned actual single-input-bit hash sensitivity summary.
/// @par Example
/// @code
/// const dm::algorithms::HashAvalanche value{};
/// std::print("{}\n",value.hash_calls); // 0
/// @endcode
struct HashAvalanche {
  std::string baseline_hex; ///< Actual unmodified-message digest in lowercase hex.
  std::vector<std::size_t>
      changed_bits; ///< Actual digest Hamming distance for each LSB-first input bit flip.
  std::vector<std::size_t> histogram; ///< Full support 0..digest_bits, without clipped tails.
  double                   mean{};    ///< Actual mean changed bits, zero when there are no flips.
  std::size_t              hash_calls{}; ///< Actual baseline plus one call per input bit.
  std::size_t compression_blocks{}; ///< Actual total producing compression blocks across all calls.
};
} // namespace datamunge::algorithms
