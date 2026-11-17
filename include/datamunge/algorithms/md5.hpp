#pragma once

/// @file md5.hpp
/// @brief Canonical md5 API, producing histories and checked native contracts.

#include <array>
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/hex_encoding.hpp>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Sixteen MD5 digest bytes in RFC1321 little-endian output order.
/// @details A fixed checksum representation, not a collision-resistant security primitive.
/// @par Example
/// @code
/// auto d=dm::algorithms::md5("abc");
/// std::print("{}\n",d.size()); // 16
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
using MD5Digest = std::array<std::uint8_t, 16>;

/// @brief Actual digest and message/block accounting from the canonical producer.
/// @details Input length is encoded modulo2^64 bits as specified by RFC1321. Every block performs exactly64 actual compression steps.
/// @par Example
/// @code
/// auto a=dm::algorithms::md5_analyze("abc");
/// std::print("{} {}\n",a.blocks,a.steps); // 1 64
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
struct MD5Analysis {
  MD5Digest     digest{};        ///< Actual sixteen-byte digest.
  std::size_t   input_bytes{0};  ///< Original message byte count.
  std::size_t   padded_bytes{0}; ///< Actual padded message byte count, a positive multiple of64.
  std::uint64_t encoded_bit_length{0}; ///< Original byte length times8 modulo2^64.
  std::size_t   blocks{0};             ///< Actual compression blocks.
  std::size_t   steps{0};              ///< Actual compression iterations,64 per block.
};

/// @brief One actual modulo2^32 compression step, captured before and after assignment.
/// @details State arrays use a,b,c,d order. Boolean function precedes addition; sum is the modular a+function+constant+word.
/// @par Example
/// @code
/// auto t=dm::algorithms::md5_trace("abc");
/// std::print("{}\n",t.steps[0].word_index); // 0
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
struct MD5Step {
  std::size_t                  block{0};      ///< Actual zero-based compression block.
  std::size_t                  index{0};      ///< Actual zero-based step within the block.
  std::array<std::uint32_t, 4> before{};      ///< Actual a,b,c,d before this step.
  std::uint32_t                function{0};   ///< Actual round Boolean function value.
  std::size_t                  word_index{0}; ///< Actual selected message-word index0..15.
  std::uint32_t                word{0};       ///< Actual decoded message word read.
  std::uint32_t                constant{0};   ///< Actual RFC additive constant.
  unsigned                     shift{0};      ///< Actual left rotation amount.
  std::uint32_t                sum{0};        ///< Actual modular input to rotation.
  std::uint32_t                rotated{0};    ///< Actual rotated word.
  std::array<std::uint32_t, 4> after{};       ///< Actual a,b,c,d after this step.
};

/// @brief One actual decoded block and feed-forward result.
/// @details Words are decoded little-endian from the real padded byte buffer. Chain arrays use A,B,C,D order.
/// @par Example
/// @code
/// auto t=dm::algorithms::md5_trace("abc");
/// std::print("{:08x}\n",t.blocks[0].words[0]); // 80636261
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
struct MD5Block {
  std::size_t                   offset{0};       ///< Actual byte offset in the padded message.
  std::array<std::uint32_t, 16> words{};         ///< Actual sixteen decoded little-endian words.
  std::array<std::uint32_t, 4>  before{};        ///< Actual incoming chaining state.
  std::array<std::uint32_t, 4>  working_after{}; ///< Actual working a,b,c,d after64 steps.
  std::array<std::uint32_t, 4>  after{};         ///< Actual modular feed-forward chaining state.
};

/// @brief Owned input/padded bytes and actual complete block/step history.
/// @details Logical quota counts N original bytes, P padded bytes, one block record and64 step records per block, plus16 final digest bytes. Fields, capacities, runtime copies and bytes inside records are not separately counted.
/// @par Example
/// @code
/// auto t=dm::algorithms::md5_trace("abc");
/// std::print("{}\n",t.captured_entries); // 148
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
struct MD5Trace {
  std::string           input;    ///< Owned original arbitrary bytes, including embedded NUL.
  std::string           padded;   ///< Owned actual padded message buffer.
  MD5Analysis           analysis; ///< Actual same-run digest and work.
  std::vector<MD5Step>  steps;    ///< Actual complete compression steps in producing order.
  std::vector<MD5Block> blocks;   ///< Actual decoded block/feed-forward records.
  std::size_t           captured_entries{0}; ///< Complete logical capture count.
};

/// @brief One actual single-input-bit mutation and its observed digest difference.
/// @details Bit order is byte-major, least-significant bit first.
/// @par Example
/// @code
/// auto a=dm::algorithms::md5_avalanche("a");
/// std::print("{}\n",a.flips[0].bit); // 0
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
struct MD5Flip {
  std::size_t bit{0};     ///< Actual flipped input bit index.
  unsigned    changed{0}; ///< Actual Hamming distance between baseline and mutated digest,0..128.
};

/// @brief Actual complete single-bit perturbation study for one supplied message.
/// @details This finite diffusion observation is not a collision-resistance or cryptographic security certificate. Every mutant is hashed by the canonical producer.
/// @par Example
/// @code
/// auto a=dm::algorithms::md5_avalanche(dm::algorithms::md5_demo_message());
/// std::print("{}\n",a.flips.size()); // 72
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
struct MD5Avalanche {
  MD5Digest            baseline{}; ///< Actual digest of unmodified input.
  std::vector<MD5Flip> flips;      ///< Actual one-bit mutation outcomes.
  double   mean_changed{0};    ///< Mean observed changed bits, zero placeholder for empty input.
  unsigned minimum_changed{0}; ///< Minimum observed changed bits, zero for no flips.
  unsigned maximum_changed{0}; ///< Maximum observed changed bits, zero for no flips.
};

/// @brief Compute the canonical RFC1321 MD5 digest of arbitrary bytes.
/// @details Pads with0x80, zeros to byte56 modulo64, then the original bit length modulo2^64 in little-endian order. Compression uses defined uint32 modular additions and rotations. O(N) time and O(N) padded workspace; this is not a streaming API. MD5 is not collision resistant and is unsuitable for adversarial integrity/authentication.
/// @param message Arbitrary bytes; embedded NUL and high bytes are retained.
/// @return Actual sixteen-byte digest in RFC output order.
/// @throws std::length_error Before copying if complete padded geometry cannot be represented.
/// @par Example
/// @code
/// auto d=dm::algorithms::md5("abc");
/// std::print("{:02x}\n",d[0]); // 90
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
MD5Digest md5(const std::string& message);

/// @brief Return the canonical digest as32 lowercase hexadecimal characters.
/// @details Uses the same md5 producer; hexadecimal encoding does not change the checksum objective or security limitations.
/// @param message Arbitrary bytes.
/// @return Lowercase hexadecimal RFC digest.
/// @throws std::length_error On nonrepresentable padded geometry.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::md5_hex("abc")); // 900150983cd24fb0d6963f7d28e17f72
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
std::string md5_hex(const std::string& message);

/// @brief Return the actual digest and same-run padding/block/step accounting.
/// @details Counts come from canonical compression, without rerunning a reference hash.
/// @param message Arbitrary bytes.
/// @return Actual digest and selected work.
/// @throws std::length_error On nonrepresentable padded geometry.
/// @par Example
/// @code
/// auto a=dm::algorithms::md5_analyze("");
/// std::print("{}\n",a.padded_bytes); // 64
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
MD5Analysis md5_analyze(const std::string& message);

/// @brief Capture every actual decoded block, modular step and feed-forward result.
/// @details Complete quota N+P+65B+16, where P is padded bytes and B=P/64, is admitted before copying/padding. Default cap one million; zero disables it. Captured storage is O(N), with a substantial record-size constant not described by the logical count.
/// @param message Arbitrary bytes.
/// @param max_entries Complete logical cap; zero off.
/// @return Owned inputs, actual padded bytes, actual history and same-run digest.
/// @throws std::length_error Before capture if geometry or complete logical quota is exceeded.
/// @par Example
/// @code
/// auto t=dm::algorithms::md5_trace("abc",148);
/// std::print("{}\n",t.steps.size()); // 64
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
MD5Trace md5_trace(const std::string& message, std::size_t max_entries = 1000000);

/// @brief Count exact differing bits between two sixteen-byte digest arrays.
/// @details Uses XOR and popcount on each actual digest byte.
/// @param a First digest.
/// @param b Second digest.
/// @return Exact integer Hamming distance0..128.
/// @par Example
/// @code
/// auto d=dm::algorithms::md5("abc");
/// std::print("{}\n",dm::algorithms::md5_hamming(d,d)); // 0
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
unsigned md5_hamming(const MD5Digest& a, const MD5Digest& b);

/// @brief Count exact differing bits in two four-word MD5 working states.
/// @param a First a,b,c,d state.
/// @param b Second a,b,c,d state.
/// @return Exact XOR-popcount sum0..128.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::md5_state_hamming({0,0,0,0},{1,0,0,0})); // 1
/// @endcode
/// @see MD5Step, md5_hamming
/// @par Reference
/// Hamming distance on the128-bit MD5 compression working state.
unsigned md5_state_hamming(
    const std::array<std::uint32_t, 4>& a,
    const std::array<std::uint32_t, 4>& b);

/// @brief Hash every actual one-bit input mutation and measure its digest Hamming distance.
/// @details Exactly8N mutants. Complete flip geometry and the optional one-million default flip cap are checked before baseline hashing. Zero disables that optional cap. Each mutant copies/hashes N bytes, so total time is O(N^2), workspace O(N), plus O(N) outcome records. Empty input has its real baseline digest and no mutations; zero mean/min/max are placeholders.
/// @param message Arbitrary baseline bytes.
/// @param max_flips Complete mutation-count cap, zero off.
/// @return Actual baseline, every mutation outcome and measured summaries.
/// @throws std::length_error On nonrepresentable geometry, padded size or complete flip cap.
/// @par Example
/// @code
/// auto a=dm::algorithms::md5_avalanche("a",8);
/// std::print("{}\n",a.flips.size()); // 8
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
MD5Avalanche md5_avalanche(const std::string& message, std::size_t max_flips = 1000000);

/// @brief Return the original nine-byte avalanche fixture.
/// @details The word avalanche contains nine bytes and therefore72 single-bit flips, correcting the legacy prose description as eight bytes.
/// @return The byte string avalanche.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::md5_demo_message().size()); // 9
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
std::string md5_demo_message();

/// @brief Return the seven published RFC1321 test messages and expected hexadecimal digests.
/// @details These are published fixed expected values, not outputs manufactured by the producer being tested.
/// @return Owned message/expected-digest pairs in RFC order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::md5_reference_vectors().size()); // 7
/// @endcode
/// @see md5, md5_trace
/// @par Reference
/// Rivest (1992), RFC1321, The MD5 Message-Digest Algorithm; RFC6151 explains its collision-security limitations.
std::vector<std::pair<std::string, std::string>> md5_reference_vectors();

} // namespace datamunge::algorithms
