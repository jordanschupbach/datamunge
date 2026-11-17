#pragma once
/// @file rc4.hpp
/// @brief Historical RC4 KSA/PRGA with actual permutation decisions and deterministic bias-study inputs.
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
/// @brief Canonical historical byte-cipher demonstrations and owned records.
namespace datamunge::algorithms {
/// @brief One actual key-scheduling swap decision.
/// @par Example
/// @code
/// const dm::algorithms::Rc4KsaStep value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct Rc4KsaStep {
  std::size_t  i{};            ///< Actual KSA position0..255.
  std::uint8_t j_before{};     ///< Actual accumulator before this iteration.
  std::uint8_t j_after{};      ///< Actual accumulator after adding S[i] and the key byte modulo256.
  std::uint8_t key_byte{};     ///< Actual raw unsigned key byte used.
  std::uint8_t left_before{};  ///< Actual S[i] before swapping.
  std::uint8_t right_before{}; ///< Actual S[j_after] before swapping.
};
/// @brief One actual PRGA update, swap and output lookup.
/// @par Example
/// @code
/// const dm::algorithms::Rc4Step value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct Rc4Step {
  std::uint8_t i_before{};     ///< Actual previous i accumulator.
  std::uint8_t j_before{};     ///< Actual previous j accumulator.
  std::uint8_t i_after{};      ///< Actual incremented i accumulator.
  std::uint8_t j_after{};      ///< Actual updated j accumulator.
  std::uint8_t left_before{};  ///< Actual S[i_after] before the swap.
  std::uint8_t right_before{}; ///< Actual S[j_after] before the swap.
  std::uint8_t output_index{}; ///< Actual post-swap modulo256 S[i]+S[j] lookup index.
  std::uint8_t output{};       ///< Actual byte returned from the post-swap permutation.
};
/// @brief Owned raw RC4 state; copying it is not an independent stream.
/// @par Example
/// @code
/// const dm::algorithms::Rc4State value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct Rc4State {
  std::array<std::uint8_t, 256> permutation{}; ///< Actual owned permutation.
  std::uint8_t                  i{};           ///< Actual PRGA i accumulator.
  std::uint8_t                  j{};           ///< Actual PRGA j accumulator.
};
/// @brief Historical RC4 raw-byte stream state with canonical KSA and PRGA.
/// RC4 has known key-stream biases and is prohibited in TLS by RFC7465; this class is for historical/conformance study, not a modern security construction. A fresh nonempty raw byte key initializes the256-byte identity permutation then256 KSA swaps. Copies duplicate raw state; mutating a shared instance needs synchronization. Constant state storage; constructor O(256), each next O(1). Owns no key/history pointer after construction.
/// @par Example
/// @code
/// dm::algorithms::Rc4 stream("Key");
/// std::print("{:02x}\n",unsigned(stream.next())); // eb
/// @endcode
/// @par References
/// - RFC6229 RC4 test vectors, https://www.rfc-editor.org/rfc/rfc6229; RFC7465, https://www.rfc-editor.org/rfc/rfc7465
/// @see rc4_trace(), next_step()
class Rc4 {
 public:
  /// @brief Initialize from raw key bytes and optionally append the actual256 KSA records.
  /// Validates a nonempty key before any modulo/index operation. Embedded NUL and high bytes are supported. For keys longer than256 bytes the classic256-iteration KSA reads only the first256 bytes, as before. Optional caller history is appended synchronously and is never retained by the stream; allocation failure may leave a caller vector partially appended.
  /// @param key Nonempty borrowed raw byte string.
  /// @param scheduling Optional borrowed output vector, default null.
  /// @throws std::invalid_argument Empty key.
  /// @par Example
  /// @code
  /// std::vector<dm::algorithms::Rc4KsaStep> steps;
  /// dm::algorithms::Rc4 stream("Key",&steps);
  /// std::print("{}\n",steps.size()); // 256
  /// @endcode
  /// @see Rc4, state()
  explicit Rc4(const std::string& key, std::vector<Rc4KsaStep>* scheduling = nullptr);
  /// @brief Consume one canonical PRGA byte through the actual-step producer.
  /// Advances i/j, swaps the actual permutation entries and reads one actual output byte. Constant work/storage, no allocation.
  /// @return Next raw keystream byte.
  /// @par Example
  /// @code
  /// dm::algorithms::Rc4 stream("Key");
  /// std::print("{:02x}\n",unsigned(stream.next())); // eb
  /// @endcode
  /// @see next_step()
  std::uint8_t next();
  /// @brief Consume one PRGA byte and return its actual accumulator/swap/lookup record.
  /// This is the same producer used by next(); it never replays a previous transition. The returned scalar record owns its values and consumes exactly one output byte. Constant work/storage, no allocation.
  /// @return Actual before/after indices, swapped values and emitted byte.
  /// @par Example
  /// @code
  /// dm::algorithms::Rc4 stream("Key");
  /// auto step=stream.next_step();
  /// std::print("{:02x}\n",unsigned(step.output)); // eb
  /// @endcode
  /// @see next(), state()
  Rc4Step next_step();
  /// @brief Copy the actual permutation and accumulators without advancing the stream.
  /// Copies256 bytes and two accumulators, with no allocation; a snapshot is raw state, not a fresh key or independent stream.
  /// @return Owned raw state.
  /// @par Example
  /// @code
  /// dm::algorithms::Rc4 stream("Key");
  /// std::print("{}\n",unsigned(stream.state().i)); // 0
  /// @endcode
  /// @see next_step()
  Rc4State state() const;

 private:
  std::array<std::uint8_t, 256> s_{};  ///< Owned raw KSA/PRGA permutation.
  std::uint8_t                  i_{0}; ///< Actual PRGA i accumulator.
  std::uint8_t                  j_{0}; ///< Actual PRGA j accumulator.
};
/// @brief Allocation and actual-record policies; zero disables an optional cap.
/// @par Example
/// @code
/// const dm::algorithms::Rc4Options value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct Rc4Options {
  std::size_t max_bytes{1000000}; ///< Maximum requested output bytes.
  std::size_t max_trace_entries{
      1000000}; ///< Two258-scalar states plus256 KSA records plus output-step records.
};
/// @brief Owned actual key schedule and output transitions.
/// @par Example
/// @code
/// const dm::algorithms::Rc4Trace value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct Rc4Trace {
  Rc4State                  scheduled;  ///< Actual post-KSA state before any PRGA byte.
  Rc4State                  final;      ///< Actual post-output state.
  std::vector<Rc4KsaStep>   scheduling; ///< Actual256 KSA decisions.
  std::vector<Rc4Step>      steps;      ///< Actual requested PRGA records.
  std::vector<std::uint8_t> bytes;      ///< Actual emitted keystream bytes.
};
/// @brief Return the first n actual classic RC4 keystream bytes.
/// Same Rc4 producer as the trace. Nonempty key is validated even for n0; owned output allocation is bounded by the default million-byte policy. Use rc4_trace() to configure capture policies. O(256+n) time and O(n) output.
/// @param key Nonempty raw key bytes.
/// @param n Requested output bytes.
/// @return Owned raw keystream.
/// @throws std::invalid_argument Empty key.
/// @throws std::length_error Default output allocation bound exceeded.
/// @par Example
/// @code
/// auto bytes=dm::algorithms::rc4_keystream("Wiki",1);
/// std::print("{:02x}\n",unsigned(bytes[0])); // 60
/// @endcode
/// @see Rc4, rc4_trace()
std::vector<std::uint8_t> rc4_keystream(const std::string& key, std::size_t n);
/// @brief Capture actual KSA and PRGA decisions from one canonical stream execution.
/// Complete logical capture admission precedes construction:516 state scalars+256 schedule records+n output records. Returned bytes and step output fields come from the same actual transitions. No inline XML/encoding is produced. O(256+n) time and owned storage.
/// @param key Nonempty raw key bytes.
/// @param n Requested output bytes, including zero.
/// @param options Output and complete history admission policies.
/// @return Owned actual schedule, bytes, step records and endpoint states.
/// @throws std::invalid_argument Empty key.
/// @throws std::length_error Output/history/capacity bound exceeded.
/// @par Example
/// @code
/// auto t=dm::algorithms::rc4_trace("Key",3);
/// std::print("{}\n",t.steps.size()); // 3
/// @endcode
/// @see Rc4::next_step()
Rc4Trace rc4_trace(const std::string& key, std::size_t n, const Rc4Options& options = {});
/// @brief XOR raw data with one fresh classic RC4 stream.
/// Encryption and decryption are the same bytewise XOR. Validates a nonempty key even for empty data; embedded NUL/high bytes are preserved. Reusing a key restarts the same keystream; no nonce, authentication or modern security property is supplied. O(256+size) time and owned output, default million-byte allocation policy.
/// @param key Nonempty raw key bytes.
/// @param data Borrowed raw plaintext/ciphertext bytes.
/// @return Owned XOR result with identical byte length.
/// @throws std::invalid_argument Empty key.
/// @throws std::length_error Default output bound exceeded.
/// @par Example
/// @code
/// const auto x=dm::algorithms::rc4_encrypt("Key","Plaintext");
/// std::print("{}\n",dm::algorithms::rc4_encrypt("Key",x)); // Plaintext
/// @endcode
/// @see rc4_keystream()
std::string rc4_encrypt(const std::string& key, const std::string& data);
/// @brief Actual histogram of second bytes from the original deterministic key recipe.
/// @par Example
/// @code
/// const dm::algorithms::Rc4BiasStudy value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct Rc4BiasStudy {
  std::array<std::size_t, 256> counts{}; ///< Actual exact second-byte counts.
  std::array<std::size_t, 16>  groups{}; ///< Actual high-nibble bin counts.
  std::size_t                  keys{};   ///< Actual generated key count, possibly zero.
  std::size_t
      unique_keys{}; ///< Actual distinct key strings; repeated keys are not independent samples.
};
/// @brief Run the original deterministic second-keystream-byte experiment natively.
/// Generates key "k"+decimal(uint32(k)*2654435761 mod2^32 mod1000000) for k0..count-1. This preserves the report recipe including32-bit multiplication wrap. Counts exact second bytes, high-nibble groups and distinct strings; the sequence is not iid random keys and finite histograms do not establish a bias theorem. O(256*count) work plus distinct-key storage.
/// @param count Generated key count, default20000, at most UINT32_MAX.
/// @param max_keys Allocation/work policy, default1000000; zero disables.
/// @return Actual deterministic key counts and byte histogram.
/// @throws std::length_error Key policy or representable recipe range exceeded.
/// @par Example
/// @code
/// auto s=dm::algorithms::rc4_second_byte_study(0);
/// std::print("{}\n",s.keys); // 0
/// @endcode
/// @see rc4_trace()
Rc4BiasStudy rc4_second_byte_study(std::size_t count = 20000, std::size_t max_keys = 1000000);
} // namespace datamunge::algorithms
