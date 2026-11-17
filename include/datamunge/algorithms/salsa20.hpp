#pragma once
/// @file salsa20.hpp
/// @brief Salsa20/20 core and checked streams with actual quarter-round/block histories.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
/// @brief Canonical byte-stream cipher primitives and conformance records.
namespace datamunge::algorithms {
/// @brief One actual column/row quarter-round input and output.
/// @par Example
/// @code
/// const dm::algorithms::Salsa20Quarter value{};
/// std::print("{}\n",value.round); // 0
/// @endcode
struct Salsa20Quarter {
  std::size_t                  round{};   ///< Actual round0..19, even column and odd row.
  std::size_t                  quarter{}; ///< Actual quartet0..3 in the specified round order.
  std::array<std::size_t, 4>   indices{}; ///< Actual state word indices in a,b,c,d order.
  std::array<std::uint32_t, 4> before{};  ///< Actual four words before the four ARX updates.
  std::array<std::uint32_t, 4> after{};   ///< Actual four words after those updates.
};
/// @brief Owned actual Salsa20 core input, mixed words, feed-forward and emitted bytes.
/// @par Example
/// @code
/// const dm::algorithms::Salsa20CoreTrace value{};
/// std::print("{}\n",value.quarters.size()); // 0
/// @endcode
struct Salsa20CoreTrace {
  std::array<std::uint32_t, 16> initial{}; ///< Actual little-endian input words.
  std::array<std::uint32_t, 16> mixed{};   ///< Actual state after20 rounds, before feed-forward.
  std::array<std::uint32_t, 16>
      output_words{};                    ///< Actual modulo2^32 sums of mixed and initial words.
  std::array<std::uint8_t, 64> bytes{};  ///< Actual little-endian emitted core output.
  std::vector<Salsa20Quarter>  quarters; ///< Actual80 optional quarter-round records.
};
/// @brief Complete stream-byte and actual captured-block policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::Salsa20Options value{};
/// std::print("{}\n",value.max_bytes); // 1000000
/// @endcode
struct Salsa20Options {
  std::size_t max_bytes{1000000}; ///< Complete input/output byte count.
  std::size_t max_trace_entries{
      1000000}; ///< Actual64 key-stream bytes plus one header per captured block.
};
/// @brief One actual counter block and its key-stream bytes used by XOR.
/// @par Example
/// @code
/// const dm::algorithms::Salsa20BlockRecord value{};
/// std::print("{}\n",value.used); // 0
/// @endcode
struct Salsa20BlockRecord {
  std::uint64_t counter{}; ///< Actual nonwrapped block counter.
  std::size_t   offset{};  ///< Actual first message byte using this block.
  std::size_t   used{};    ///< Actual XOR byte count,1..64.
  std::array<std::uint8_t, 64>
      keystream{}; ///< Actual generated block, including an unused partial-block tail.
};
/// @brief Owned actual stream output and selected native block/XOR work.
/// @par Example
/// @code
/// const dm::algorithms::Salsa20StreamRun value{};
/// std::print("{}\n",value.block_calls); // 0
/// @endcode
struct Salsa20StreamRun {
  std::string output;        ///< Actual XOR output, preserving every raw byte.
  std::size_t block_calls{}; ///< Actual generated counter blocks, including a final partial block.
  std::size_t byte_xors{};   ///< Actual message-byte XOR updates.
  std::vector<Salsa20BlockRecord>
      blocks; ///< Actual optional block records, not regenerated output.
};
/// @brief Apply the canonical Salsa20/20 core to one64-byte state.
/// Decodes16 little-endian words, executes10 column/row double rounds with modulo2^32 additions and specified rotations7,9,13,18, adds the original state, then emits64 little-endian bytes. Fixed work/storage, no allocation or shared mutable state; input unchanged. This bare permutation-plus-feed-forward core is not an authenticated message format.
/// @param in Complete borrowed64 input bytes.
/// @return Owned actual64-byte core result.
/// @par Example
/// @code
/// const auto out=dm::algorithms::salsa20_core({});
/// std::print("{}\n",unsigned(out[0])); // 0
/// @endcode
/// @par References
/// - D. J. Bernstein, Salsa20 specification, https://cr.yp.to/snuffle/spec.pdf.
/// @par References
/// - Bernstein, Salsa20 specification. https://cr.yp.to/snuffle/spec.pdf
/// @see salsa20_core_trace(), salsa20_block()
std::array<std::uint8_t, 64> salsa20_core(const std::array<std::uint8_t, 64>& in);
/// @brief Capture all actual quarter-round inputs/outputs from the same Salsa20 core.
/// The80 records read the four real words immediately before and after each specified quarter; no second execution populates history. Complete optional geometry is known:48 retained initial/mixed/feed-forward words plus80*(8 words+1 header)=768 logical units. Check this before decoding/mutation; output64 bytes are intrinsic and excluded from that optional geometry. Fixed-time core work, owned copied values, input unchanged; zero disables only the quota.
/// @param in Borrowed64 state bytes.
/// @param max_trace_entries Complete optional word/header policy, default1000000; zero disables.
/// @return Owned actual state words,80 real quarter events and emitted bytes.
/// @throws std::length_error Complete768-unit history exceeds the policy.
/// @par Example
/// @code
/// auto t=dm::algorithms::salsa20_core_trace({});
/// std::print("{}\n",t.quarters.size()); // 80
/// @endcode
/// @par References
/// - D. J. Bernstein, Salsa20 specification, https://cr.yp.to/snuffle/spec.pdf.
/// @par References
/// - Bernstein, Salsa20 specification. https://cr.yp.to/snuffle/spec.pdf
/// @see salsa20_core()
Salsa20CoreTrace salsa20_core_trace(
    const std::array<std::uint8_t, 64>& in,
    std::size_t                         max_trace_entries = 1000000);
/// @brief Generate one actual Salsa20/20 block for a256-bit key and64-bit nonce/counter.
/// Uses the standard expand32-byte-k constants and original Salsa word layout: key at1..4/11..14, nonce6..7, counter8..9. Calls the same canonical core once. Fixed work/owned output; copying key/nonce is synchronous and no references survive.
/// @param key Complete32 raw key bytes.
/// @param counter Exact64-bit block counter, any value.
/// @param nonce Complete8 raw nonce bytes.
/// @return Owned actual64 key-stream bytes.
/// @par Example
/// @code
/// const auto b=dm::algorithms::salsa20_block({},0,{});
/// std::print("{}\n",b.size()); // 64
/// @endcode
/// @par References
/// - D. J. Bernstein, Salsa20 specification, https://cr.yp.to/snuffle/spec.pdf.
/// @par References
/// - Bernstein, Salsa20 specification. https://cr.yp.to/snuffle/spec.pdf
/// @see salsa20_stream_run()
std::array<std::uint8_t, 64> salsa20_block(
    const std::array<std::uint8_t, 32>& key,
    std::uint64_t                       counter,
    const std::array<std::uint8_t, 8>&  nonce);
/// @brief XOR a raw message with checked actual consecutive Salsa20 blocks.
/// Validates complete byte length, nonwrapping final counter and known captured geometry before output allocation/core execution. Empty data uses no counter blocks. Generates ceil(bytes/64) actual blocks, then XORs exactly one key-stream byte per message byte. Capture costs65 logical units per generated block, including its full64-byte key stream even for a partial final block. This is raw unauthenticated XOR; caller controls key/nonce/counter uniqueness and message framing. Time O(bytes), owned output O(bytes) plus optional O(blocks) history; input/key/nonce unchanged and no shared mutable state.
/// @param key Complete32 key bytes.
/// @param counter First actual64-bit counter.
/// @param nonce Complete8 nonce bytes.
/// @param data Borrowed complete raw message/ciphertext, including NUL/high bytes.
/// @param options Complete byte and optional captured-block policies.
/// @return Owned actual XOR output, block/XOR counts and optional actual key streams.
/// @throws std::length_error Complete byte/history/container or count capacity exceeded.
/// @throws std::overflow_error The last required block counter would wrap uint64.
/// @par Example
/// @code
/// auto r=dm::algorithms::salsa20_stream_run({},0,{},"abc");
/// std::print("{} {}\n",r.block_calls,r.byte_xors); // 1 3
/// @endcode
/// @par References
/// - D. J. Bernstein, Salsa20 specification, https://cr.yp.to/snuffle/spec.pdf.
/// @par References
/// - Bernstein, Salsa20 specification. https://cr.yp.to/snuffle/spec.pdf
/// @see salsa20_encrypt(), salsa20_core_trace()
Salsa20StreamRun salsa20_stream_run(
    const std::array<std::uint8_t, 32>& key,
    std::uint64_t                       counter,
    const std::array<std::uint8_t, 8>&  nonce,
    const std::string&                  data,
    const Salsa20Options&               options = {});
/// @brief XOR a raw message with checked actual consecutive Salsa20 blocks and retain the used blocks.
/// Validates complete byte length, nonwrapping final counter and known captured geometry before output allocation/core execution. Empty data uses no counter blocks. Generates ceil(bytes/64) actual blocks, then XORs exactly one key-stream byte per message byte. Capture costs65 logical units per generated block, including its full64-byte key stream even for a partial final block. This is raw unauthenticated XOR; caller controls key/nonce/counter uniqueness and message framing. Time O(bytes), owned output O(bytes) plus optional O(blocks) history; input/key/nonce unchanged and no shared mutable state.
/// @param key Complete32 key bytes.
/// @param counter First actual64-bit counter.
/// @param nonce Complete8 nonce bytes.
/// @param data Borrowed complete raw message/ciphertext, including NUL/high bytes.
/// @param options Complete byte and optional captured-block policies.
/// @return Owned actual XOR output, block/XOR counts and optional actual key streams.
/// @throws std::length_error Complete byte/history/container or count capacity exceeded.
/// @throws std::overflow_error The last required block counter would wrap uint64.
/// @par Example
/// @code
/// auto r=dm::algorithms::salsa20_stream_trace({},0,{},"abc");
/// std::print("{} {}\n",r.block_calls,r.byte_xors); // 1 3
/// @endcode
/// @par References
/// - D. J. Bernstein, Salsa20 specification, https://cr.yp.to/snuffle/spec.pdf.
/// @par References
/// - Bernstein, Salsa20 specification. https://cr.yp.to/snuffle/spec.pdf
/// @see salsa20_encrypt(), salsa20_core_trace()
Salsa20StreamRun salsa20_stream_trace(
    const std::array<std::uint8_t, 32>& key,
    std::uint64_t                       counter,
    const std::array<std::uint8_t, 8>&  nonce,
    const std::string&                  data,
    const Salsa20Options&               options = {});
/// @brief Encrypt/decrypt through the preserved native Salsa20 XOR signature.
/// Delegates once to the same producer/default million-byte bound. Preserves valid ordinary outputs; required counter wrap now rejects before generating repeated stream blocks. No authentication/tag format is added.
/// @param key Complete32 key bytes.
/// @param counter First64-bit block counter.
/// @param nonce Complete8 nonce bytes.
/// @param data Complete raw message/ciphertext.
/// @return Owned actual same-length XOR bytes.
/// @throws std::length_error Default byte policy exceeded.
/// @throws std::overflow_error Required counter wrap.
/// @par Example
/// @code
/// auto c=dm::algorithms::salsa20_encrypt({},0,{},"abc");
/// std::print("{}\n",dm::algorithms::salsa20_encrypt({},0,{},c)); // abc
/// @endcode
/// @par References
/// - D. J. Bernstein, Salsa20 specification, https://cr.yp.to/snuffle/spec.pdf.
/// @par References
/// - Bernstein, Salsa20 specification. https://cr.yp.to/snuffle/spec.pdf
/// @see salsa20_stream_run()
std::string salsa20_encrypt(
    const std::array<std::uint8_t, 32>& key,
    std::uint64_t                       counter,
    const std::array<std::uint8_t, 8>&  nonce,
    const std::string&                  data);
/// @brief Owned original documented core vector and stream-report inputs.
/// @par Example
/// @code
/// const dm::algorithms::Salsa20Example value{};
/// std::print("{}\n",value.plaintext.size()); // 0
/// @endcode
struct Salsa20Example {
  std::array<std::uint8_t, 64> input{};    ///< Bernstein documented core input.
  std::array<std::uint8_t, 64> expected{}; ///< Documented complete64-byte core output.
  std::array<std::uint8_t, 32> key{};      ///< Original report key byte i*7+1 modulo256.
  std::array<std::uint8_t, 8>  nonce{};    ///< Original nonce1,...,8.
  std::string plaintext; ///< Original exact message, whose actual block count must be measured.
};
/// @brief Return the original core conformance vector and deterministic report key/message.
/// Owns fixed specification bytes, the original deterministic32-byte key, nonce and exact plaintext. No randomness or encryption is hidden; invoke the producing API explicitly.
/// @return Owned original conformance/report inputs.
/// @par Example
/// @code
/// std::print("{}\n",unsigned(dm::algorithms::salsa20_example().input[0])); // 211
/// @endcode
/// @par References
/// - D. J. Bernstein, Salsa20 specification, https://cr.yp.to/snuffle/spec.pdf.
/// @par References
/// - Bernstein, Salsa20 specification. https://cr.yp.to/snuffle/spec.pdf
/// @see salsa20_core_trace()
Salsa20Example salsa20_example();
/// @brief Actual byte and high-nibble counts across consecutive generated blocks.
/// @par Example
/// @code
/// const dm::algorithms::Salsa20ByteStudy value{};
/// std::print("{}\n",value.bytes); // 0
/// @endcode
struct Salsa20ByteStudy {
  std::array<std::size_t, 256> counts{}; ///< Actual individual byte values observed.
  std::array<std::size_t, 16>
              groups{}; ///< Actual high-nibble bins, each aggregating16 byte values.
  std::size_t blocks{}; ///< Actual generated consecutive counter blocks.
  std::size_t bytes{};  ///< Actual observed key-stream bytes,64*blocks.
};
/// @brief Count actual key-stream bytes from consecutive native blocks.
/// Generates each real block once and counts every output byte into both256 byte bins and16 high-nibble bins. This fixed-key/counter histogram is descriptive, not a test or proof of cryptographic security or independent randomness. Complete requested byte geometry and final counter are checked before block execution.
/// @param key Complete32 key bytes.
/// @param nonce Complete8 nonce bytes.
/// @param counter First actual counter, default0.
/// @param blocks Requested complete blocks, default64.
/// @param max_bytes Complete64*blocks policy, default1000000; zero disables.
/// @return Owned actual observed counts and native block/byte work.
/// @throws std::length_error Complete byte geometry/policy exceeded.
/// @throws std::overflow_error Required final counter wraps uint64.
/// @par Example
/// @code
/// auto r=dm::algorithms::salsa20_byte_study({},{});
/// std::print("{}\n",r.bytes); // 4096
/// @endcode
/// @par References
/// - D. J. Bernstein, Salsa20 specification, https://cr.yp.to/snuffle/spec.pdf.
/// @par References
/// - Bernstein, Salsa20 specification. https://cr.yp.to/snuffle/spec.pdf
/// @see salsa20_block()
Salsa20ByteStudy salsa20_byte_study(
    const std::array<std::uint8_t, 32>& key,
    const std::array<std::uint8_t, 8>&  nonce,
    std::uint64_t                       counter   = 0,
    std::size_t                         blocks    = 64,
    std::size_t                         max_bytes = 1000000);
} // namespace datamunge::algorithms
