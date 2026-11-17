#pragma once
/// @file sha1.hpp
/// @brief Checked FIPS 180 digest with actual schedules, working rounds and block feed-forward.
#include <array>
#include <cstdint>
#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/sha_types.hpp>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief One actual producing sha1 working-state round.
/// @par Example
/// @code
/// const dm::algorithms::SHA1Round value{};
/// std::print("{}\n",value.index); // 0
/// @endcode
struct SHA1Round {
  std::size_t                  index{};  ///< Actual zero-based round within its block.
  std::uint32_t                word{};   ///< Actual scheduled message word consumed by this round.
  std::array<std::uint32_t, 5> before{}; ///< Actual working words before mutation.
  std::array<std::uint32_t, 5> after{};  ///< Actual working words after mutation.
};
/// @brief One actual producing sha1 compression block.
/// @par Example
/// @code
/// const dm::algorithms::SHA1Block value{};
/// std::print("{}\n",value.offset); // 0
/// @endcode
struct SHA1Block {
  std::size_t                   offset{};   ///< Actual byte offset in the padded message.
  std::array<std::uint32_t, 80> schedule{}; ///< Actual loaded and expanded schedule.
  std::array<std::uint32_t, 5>  before{};   ///< Actual incoming chaining words.
  std::array<std::uint32_t, 5>  after{};    ///< Actual chaining words after feed-forward.
  std::vector<SHA1Round>        rounds;     ///< Actual optional chronological working rounds.
};
/// @brief Owned actual sha1 digest and selected compression work.
/// @par Example
/// @code
/// const dm::algorithms::SHA1Run value{};
/// std::print("{}\n",value.rounds); // 0
/// @endcode
struct SHA1Run {
  std::array<std::uint8_t, 20> digest{};        ///< Actual big-endian digest bytes.
  std::size_t                  message_bytes{}; ///< Complete original raw byte count.
  std::size_t padded_bytes{};       ///< Actual complete padded byte count, including final length.
  std::size_t compression_blocks{}; ///< Actual processed blocks.
  std::size_t
      schedule_expansions{}; ///< Actual newly generated words beyond the first sixteen per block.
  std::size_t            rounds{}; ///< Actual executed working rounds.
  std::vector<SHA1Block> blocks;   ///< Actual optional same-execution blocks and their rounds.
};
/// @brief Compute checked sha1 on complete raw bytes.
/// Follows FIPS180-4 SHA1 padding, schedule, modulo2^32 round updates and block feed-forward. Empty and embedded-NUL messages are supported. Checks original length fits the FIPS64-bit bit-length field, padded container geometry, selected counters and complete optional capture before allocation or compression. Captured logical units per block are one block header+80 scheduled words+10 chain words+80*(one round header+10 working words+one scheduled-word copy). Does not retain a second padded-message copy in the result. O(bytes) time/intrinsic padded storage; optional history proportional to actual blocks with the stated fixed constants. Borrowed input unchanged; output owns its bytes/history; no shared state. SHA-1 is retained as a historical algorithm, not a collision-resistant security choice. Digest conformance and avalanche diagnostics do not establish resistance to attacks.
/// @param message Borrowed complete raw bytes, including empty/binary strings.
/// @param options Complete raw-byte and optional retained-history bounds.
/// @return Owned actual digest, selected work and optional producing histories.
/// @throws std::length_error Complete FIPS length, padding, counter, container or history policy/capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sha1_run("abc");
/// std::print("{} {}\n",r.compression_blocks,r.rounds); // 1 80
/// @endcode
/// @par References
/// - NIST FIPS180-4, Secure Hash Standard. https://csrc.nist.gov/pubs/fips/180-4/upd1/final
/// @see sha1_hex()
SHA1Run sha1_run(const std::string& message, const SHAOptions& options = {});
/// @brief Compute checked sha1 on complete raw bytes retaining actual schedules and rounds.
/// Follows FIPS180-4 SHA1 padding, schedule, modulo2^32 round updates and block feed-forward. Empty and embedded-NUL messages are supported. Checks original length fits the FIPS64-bit bit-length field, padded container geometry, selected counters and complete optional capture before allocation or compression. Captured logical units per block are one block header+80 scheduled words+10 chain words+80*(one round header+10 working words+one scheduled-word copy). Does not retain a second padded-message copy in the result. O(bytes) time/intrinsic padded storage; optional history proportional to actual blocks with the stated fixed constants. Borrowed input unchanged; output owns its bytes/history; no shared state. SHA-1 is retained as a historical algorithm, not a collision-resistant security choice. Digest conformance and avalanche diagnostics do not establish resistance to attacks.
/// @param message Borrowed complete raw bytes, including empty/binary strings.
/// @param options Complete raw-byte and optional retained-history bounds.
/// @return Owned actual digest, selected work and optional producing histories.
/// @throws std::length_error Complete FIPS length, padding, counter, container or history policy/capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sha1_trace("abc");
/// std::print("{} {}\n",r.compression_blocks,r.rounds); // 1 80
/// @endcode
/// @par References
/// - NIST FIPS180-4, Secure Hash Standard. https://csrc.nist.gov/pubs/fips/180-4/upd1/final
/// @see sha1_hex()
SHA1Run sha1_trace(const std::string& message, const SHAOptions& options = {});
/// @brief Return the checked sha1 raw digest through the preserved signature.
/// Delegates to the same native producer with no optional byte-size policy and no optional history, preserving historical arbitrary-message support within intrinsic FIPS and container bounds.
/// @param message Borrowed complete raw bytes.
/// @return Owned 20 big-endian digest bytes.
/// @throws std::length_error Intrinsic FIPS length or container capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sha1("abc").size()); // 20
/// @endcode
/// @par References
/// - NIST FIPS180-4. https://csrc.nist.gov/pubs/fips/180-4/upd1/final
/// @see sha1_run()
std::array<std::uint8_t, 20> sha1(const std::string& message);
/// @brief Return the checked sha1 digest as lowercase hexadecimal.
/// Uses the canonical native digest and hex encoder without a separate hashing implementation.
/// @param message Borrowed complete raw bytes.
/// @return Owned exactly 40 lowercase hexadecimal characters.
/// @throws std::length_error Intrinsic FIPS length or container capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sha1_hex("abc")); // a9993e364706816aba3e25717850c26c9cd0d89d
/// @endcode
/// @par References
/// - NIST FIPS180-4. https://csrc.nist.gov/pubs/fips/180-4/upd1/final
/// @see sha1()
std::string sha1_hex(const std::string& message);
/// @brief Measure actual sha1 output changes for every single input-bit flip.
/// Hashes the unchanged baseline once, then one same-length message per input bit in byte order and least-significant-bit-first order. Native digest-Hamming popcounts feed a full support histogram, without the historical clipped tail bins. Empty input still hashes the baseline once and returns a zero mean and empty flip history. Complete bit count and requested flip cap checked before any hashing. Retains only current message/digest plus owned O(bits+digest_bits) summary. This fixed-message diagnostic is not a security proof or statistical independence test.
/// @param message Borrowed complete raw baseline message.
/// @param max_flips Complete number of input-bit variants; default4096, zero disables.
/// @param options Per-call complete message policy; trace policy unused because calls retain no rounds.
/// @return Owned baseline digest, actual per-bit distances, full histogram, mean and actual total producing calls/blocks.
/// @throws std::length_error Complete variant, FIPS byte or cumulative count geometry/capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sha1_avalanche("a");
/// std::print("{} {}\n",r.changed_bits.size(),r.hash_calls); // 8 9
/// @endcode
/// @par References
/// - NIST FIPS180-4 digest specification. https://csrc.nist.gov/pubs/fips/180-4/upd1/final
/// @see sha1_run()
HashAvalanche sha1_avalanche(
    const std::string& message,
    std::size_t        max_flips = 4096,
    const SHAOptions&  options   = {});
} // namespace datamunge::algorithms
