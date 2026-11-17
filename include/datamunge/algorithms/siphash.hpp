#pragma once
/// @file siphash.hpp
/// @brief Actual SipHash-2-4 ARX rounds and keyed bucket fixtures.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief SipHashOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SipHashOptions value{};
/// std::print("{}\n",value.max_bytes); // 1000000
/// @endcode
struct SipHashOptions {
  std::size_t max_bytes{1000000};         ///< Complete message-byte policy; zero disables.
  std::size_t max_round_records{1000000}; ///< Captured actual SipRound records; zero disables.
};
/// @brief SipHashRound stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SipHashRound value{};
/// std::print("{}\n",value.finalization); // false
/// @endcode
struct SipHashRound {
  std::size_t block{};        ///< Absorbed block index, tail included.
  bool        finalization{}; ///< Four-round finalization after v2 XOR ff.
  std::size_t
      round{}; ///< Zero-based round within two-round compression or four-round finalization.
  std::array<std::uint64_t, 4> before{}; ///< Actual words immediately before this SipRound.
  std::array<std::uint64_t, 4> after{};  ///< Actual words immediately after this SipRound.
};
/// @brief SipHashRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SipHashRun value{};
/// std::print("{}\n",value.rounds); // 0
/// @endcode
struct SipHashRun {
  std::uint64_t hash{};   ///< Actual XOR of the four final words.
  std::size_t   blocks{}; ///< Actual eight-byte blocks plus one compulsory tail/length block.
  std::size_t   rounds{}; ///< Actual executed SipRounds, 2*blocks+4.
  std::vector<SipHashRound> history; ///< Optional actual ARX states.
};
/// @brief Compute SipHash-2-4.
/// 128-bit key, little-endian eight-byte loading, modulo2^64 additions, two compression rounds per full/tail block and four final rounds. Top tail byte holds message length modulo256; binary NULs are ordinary bytes. Same producer for trace and ordinary API. O(bytes) time and O(1) working storage without optional O(bytes) history. Complete block/round/capture geometry is checked before hashing. Not a cryptographic collision-resistant content digest: 64-bit output has birthday limits. Secret unpredictable keys mitigate predictable hash-table flooding; these deterministic examples do not establish cryptographic security. No shared state; results own values.
/// @param key Borrowed complete16-byte key; never generated implicitly.
/// @param message Borrowed complete raw byte string.
/// @param options Complete byte and optional round-record policies; zero disables policies.
/// @return Actual hash, blocks/rounds and optional ARX histories.
/// @throws std::length_error Byte, capture, counter or container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::siphash24_run(dm::algorithms::siphash_example_key(),"");
/// std::print("{:016x} {}\n",r.hash,r.rounds); // 726fdb47dd0e0e31 6
/// @endcode
/// @par References
/// - J.-P. Aumasson and D. J. Bernstein, SipHash: a fast short-input PRF, 2012, https://www.aumasson.jp/siphash/siphash.pdf.
/// @see siphash24()
SipHashRun siphash24_run(
    const std::array<std::uint8_t, 16>& key,
    const std::string&                  message,
    const SipHashOptions&               options = {});
/// @brief Compute SipHash-2-4 with actual SipRound word states.
/// 128-bit key, little-endian eight-byte loading, modulo2^64 additions, two compression rounds per full/tail block and four final rounds. Top tail byte holds message length modulo256; binary NULs are ordinary bytes. Same producer for trace and ordinary API. O(bytes) time and O(1) working storage without optional O(bytes) history. Complete block/round/capture geometry is checked before hashing. Not a cryptographic collision-resistant content digest: 64-bit output has birthday limits. Secret unpredictable keys mitigate predictable hash-table flooding; these deterministic examples do not establish cryptographic security. No shared state; results own values.
/// @param key Borrowed complete16-byte key; never generated implicitly.
/// @param message Borrowed complete raw byte string.
/// @param options Complete byte and optional round-record policies; zero disables policies.
/// @return Actual hash, blocks/rounds and optional ARX histories.
/// @throws std::length_error Byte, capture, counter or container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::siphash24_trace(dm::algorithms::siphash_example_key(),"");
/// std::print("{:016x} {}\n",r.hash,r.rounds); // 726fdb47dd0e0e31 6
/// @endcode
/// @par References
/// - J.-P. Aumasson and D. J. Bernstein, SipHash: a fast short-input PRF, 2012, https://www.aumasson.jp/siphash/siphash.pdf.
/// @see siphash24()
SipHashRun siphash24_trace(
    const std::array<std::uint8_t, 16>& key,
    const std::string&                  message,
    const SipHashOptions&               options = {});
/// @brief Return SipHash-2-4 through the preserved arbitrary-length API.
/// Uses the same actual producer without byte policy or retained history; intrinsic counter capacity checks remain.
/// @param key Complete borrowed16-byte key.
/// @param message Complete borrowed raw byte string.
/// @return Actual64-bit keyed hash.
/// @throws std::length_error Intrinsic counter capacity exceeded.
/// @par Example
/// @code
/// std::print("{:016x}\n",dm::algorithms::siphash24(dm::algorithms::siphash_example_key(),"")); // 726fdb47dd0e0e31
/// @endcode
/// @see siphash24_run()
std::uint64_t siphash24(const std::array<std::uint8_t, 16>& key, const std::string& message);
/// @brief Return the original reference key00 through0f.
/// Public deterministic fixture, never a production secret. Owned output, no shared state.
/// @return Sixteen sequential bytes.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::siphash_example_key().back()); // 15
/// @endcode
/// @see siphash24_run()
std::array<std::uint8_t, 16> siphash_example_key();
/// @brief Return a bounded sequential binary reference message.
/// Bytes are i modulo256, matching the SipHash reference vectors for lengths0..63. Owned fixture, no shared state.
/// @param length Byte count up to one million.
/// @return Owned raw bytes including NUL.
/// @throws std::length_error Fixture bound exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::siphash_example_message(15).size()); // 15
/// @endcode
/// @see siphash24_run()
std::string siphash_example_message(std::size_t length);
/// @brief Return the original32-word keyed bucket fixture.
/// Owned deterministic toy inputs; neither random input data nor an adaptive attacker experiment.
/// @return Original complete ordered word list.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::siphash_example_words().size()); // 32
/// @endcode
/// @see siphash_bucket_counts()
std::vector<std::string> siphash_example_words();
/// @brief Count actual keyed hash residues for complete supplied messages.
/// Computes the canonical hash once per message and increments hash modulo bucket_count. O(total bytes+messages+buckets), owned output, no shared state. Not a security/uniformity guarantee.
/// @param key Complete borrowed16-byte key.
/// @param messages Complete borrowed byte strings, at most one million.
/// @param bucket_count Number of output buckets,1..one million.
/// @return Owned complete bucket counts summing to messages.size().
/// @throws std::invalid_argument Zero bucket count.
/// @throws std::length_error Fixture/count/counter capacity exceeded.
/// @par Example
/// @code
/// auto c=dm::algorithms::siphash_bucket_counts(dm::algorithms::siphash_example_key(),{"a","b"},1);
/// std::print("{}\n",c); // [2]
/// @endcode
/// @see siphash24()
std::vector<std::size_t> siphash_bucket_counts(
    const std::array<std::uint8_t, 16>& key,
    const std::vector<std::string>&     messages,
    std::size_t                         bucket_count);
} // namespace datamunge::algorithms
