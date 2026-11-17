#pragma once
/// @file hmac.hpp
/// @brief HMAC-SHA-256 with checked call-boundary work and actual owned intermediate bytes.
#include <array>
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/sha256.hpp>
#include <limits>
#include <stdexcept>
#include <string>
namespace datamunge::algorithms {
/// @brief Complete 32-byte SHA-256 authentication tag, retaining embedded zero bytes.
/// @par Example
/// @code
/// const dm::algorithms::HmacTag tag{};std::print("{}\n",tag.size()); // 32
/// @endcode
using HmacTag = std::array<std::uint8_t, 32>;
/// @brief Complete 64-byte SHA-256 key/pad block, in original byte order.
/// @par Example
/// @code
/// const dm::algorithms::HmacBlock block{};std::print("{}\n",block.size()); // 64
/// @endcode
using HmacBlock = std::array<std::uint8_t, 64>;
/// @brief Complete combined key/message input policy before copying or hashing.
/// The cap counts input bytes once, not allocations, bytes hashed or peak memory.
/// @par Example
/// @code
/// const dm::algorithms::HmacOptions o;std::print("{}\n",o.max_input_bytes); // 1000000
/// @endcode
struct HmacOptions {
  std::size_t max_input_bytes{1000000}; ///< Combined complete input-byte cap; zero disables policy.
};
/// @brief Selected work recorded at actual SHA-256 invocation boundaries.
/// Block counts use SHA-256 padding length, not a captured round/compression trace.
/// Excludes pad generation, copying, allocation, hex formatting and trace copies.
/// @par Example
/// @code
/// const dm::algorithms::HmacWork w;std::print("{}\n",w.sha_calls); // 0
/// @endcode
struct HmacWork {
  std::size_t sha_calls{0};     ///< Actual SHA-256 calls, two or three for a completed HMAC.
  std::size_t bytes_hashed{0};  ///< Sum of unpadded input bytes supplied to these calls.
  std::size_t padded_blocks{0}; ///< Sum of padded 64-byte block counts derived at those call sites.
};
/// @brief Owned tag and producing call-boundary diagnostics; no input references retained.
/// @par Example
/// @code
/// const dm::algorithms::HmacRun r;std::print("{}\n",r.tag.size()); // 32
/// @endcode
struct HmacRun {
  HmacTag     tag{};             ///< Actual complete final SHA-256 digest.
  bool        key_hashed{false}; ///< True exactly when the supplied key exceeds 64 bytes.
  std::size_t key_bytes{0};      ///< Complete supplied key length, including zeros.
  std::size_t message_bytes{0};  ///< Complete supplied message length, including zeros.
  HmacWork    work;              ///< Actual SHA invocations and selected call-boundary counts.
};
/// @brief Fixed-size owned intermediates captured during the actual producing HMAC.
/// This educational record contains key material; example keys are public fixtures.
/// No replay, message-history copy or input references. Arrays occupy 256 byte
/// values beyond run (three 64-byte blocks plus two 32-byte digests), excluding
/// metadata/padding. key_digest is meaningful only when run.key_hashed; otherwise
/// its zero initialization is a sentinel. This is not secure-erasing storage.
/// @par Example
/// @code
/// const dm::algorithms::HmacTrace t;std::print("{}\n",t.key_block.size()); // 64
/// @endcode
struct HmacTrace {
  HmacRun   run;         ///< Actual producing tag and diagnostics.
  HmacBlock key_block{}; ///< Actual normalized key, prehashed if needed and zero-padded to64 bytes.
  HmacBlock inner_pad{}; ///< Actual key block XOR 0x36, used by the inner SHA call.
  HmacBlock outer_pad{}; ///< Actual key block XOR 0x5c, used by the outer SHA call.
  HmacTag
      key_digest{}; ///< Actual prehash digest only for an overlong key; otherwise zero sentinel.
  HmacTag inner_digest{}; ///< Actual digest of inner pad concatenated with the complete message.
};
/// @brief Internal complete-length checks and shared producing HMAC kernel.
namespace hmac_detail {
/// @brief Add logical byte/work counts without wraparound.
/// @param a Existing units.
/// @param b Additional units.
/// @return Exact sum.
/// @throws std::length_error If the sum cannot be represented.
inline std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("HMAC count overflow");
  return a + b;
}
/// @brief Check SHA-256 bit length and padded string representability before allocation.
/// @param bytes Complete unpadded input length.
/// @return Complete padded 64-byte block count.
/// @throws std::length_error If the bit length or padded string cannot be represented.
inline std::size_t blocks(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::uint64_t>::max() / 8)
    throw std::length_error("HMAC SHA bit length");
  const auto extra  = 64 - (add(bytes, 8) % 64);
  const auto padded = add(add(bytes, 8), extra);
  if (padded > std::string{}.max_size())
    throw std::length_error("HMAC padded string size");
  return padded / 64;
}
/// @brief Validate complete combined input and all required SHA input sizes.
/// @param key_bytes Complete borrowed key length.
/// @param message_bytes Complete borrowed message length.
/// @param options Combined input cap.
/// @throws std::length_error On cap, count or SHA padded-length failure.
inline void validate(std::size_t key_bytes, std::size_t message_bytes, HmacOptions options) {
  const auto total = add(key_bytes, message_bytes);
  if (options.max_input_bytes && total > options.max_input_bytes)
    throw std::length_error("HMAC input cap");
  if (key_bytes > 64)
    (void)blocks(key_bytes);
  (void)blocks(add(64, message_bytes));
  (void)blocks(96);
  (void)add(add(160, message_bytes), key_bytes > 64 ? key_bytes : 0);
}
/// @brief Invoke the existing SHA implementation once and record its call-boundary work.
/// @param input Complete unpadded owned SHA input.
/// @param work Selected actual invocation counters updated in place.
/// @return Actual SHA digest.
/// @throws std::length_error On count or padded-size failure.
/// @throws std::bad_alloc On allocation failure in SHA.
inline HmacTag hash(const std::string& input, HmacWork& work) {
  work.sha_calls     = add(work.sha_calls, 1);
  work.bytes_hashed  = add(work.bytes_hashed, input.size());
  work.padded_blocks = add(work.padded_blocks, blocks(input.size()));
  return sha256(input);
}
/// @brief Execute one shared HMAC, optionally retaining actual pads and digests.
/// @param key Borrowed arbitrary key bytes, including zeros/high bytes.
/// @param message Borrowed arbitrary message bytes, including zeros/high bytes.
/// @param options Complete input policy before allocation or hashing.
/// @param trace Optional owned intermediate destination.
/// @return Actual final digest and call-boundary work.
/// @throws std::length_error On complete input, SHA size or counter limits.
/// @throws std::bad_alloc On allocation failure.
inline HmacRun
execute(const std::string& key, const std::string& message, HmacOptions options, HmacTrace* trace) {
  validate(key.size(), message.size(), options);
  HmacRun run;
  run.key_bytes     = key.size();
  run.message_bytes = message.size();
  run.key_hashed    = key.size() > 64;
  std::string normalized;
  if (run.key_hashed) {
    const auto digest = hash(key, run.work);
    normalized.assign(reinterpret_cast<const char*>(digest.data()), digest.size());
    if (trace)
      trace->key_digest = digest;
  } else
    normalized = key;
  normalized.resize(64, '\0');
  std::string inner_pad(64, '\0'), outer_pad(64, '\0');
  for (std::size_t i = 0; i < 64; ++i) {
    const auto byte = static_cast<std::uint8_t>(normalized[i]);
    inner_pad[i]    = static_cast<char>(byte ^ 0x36);
    outer_pad[i]    = static_cast<char>(byte ^ 0x5c);
    if (trace) {
      trace->key_block[i] = byte;
      trace->inner_pad[i] = static_cast<std::uint8_t>(inner_pad[i]);
      trace->outer_pad[i] = static_cast<std::uint8_t>(outer_pad[i]);
    }
  }
  const auto inner = hash(inner_pad + message, run.work);
  if (trace)
    trace->inner_digest = inner;
  const std::string inner_bytes(reinterpret_cast<const char*>(inner.data()), inner.size());
  run.tag = hash(outer_pad + inner_bytes, run.work);
  return run;
}
} // namespace hmac_detail
/// @brief Compute full HMAC-SHA-256 using the preserved one-shot SHA implementation.
/// Arbitrary byte strings are accepted, including empty and embedded-NUL inputs.
/// Keys longer than 64 bytes are hashed first; other keys are zero-padded to 64.
/// The actual inner digest hashes (keyblock XOR0x36)||message; the final digest
/// hashes (keyblock XOR0x5c)||inner_digest. No tag truncation or verification.
/// O(k+m) time and O(k+m) additional storage for the one-shot copies and SHA
/// padding, not streaming O(1) storage. No optional input policy; representable
/// bit lengths, padded string sizes and counters are checked before hashing.
/// No retained references, shared mutable state or constant-time/secure-erasure
/// guarantee. Functional digest correctness is distinct from security certification.
/// @param key Complete borrowed secret-key byte string.
/// @param message Complete borrowed message byte string.
/// @return Owned complete 32-byte tag.
/// @throws std::length_error On input/count/padded-length representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hmac_sha256("Jefe","what do ya want for nothing?").size()); // 32
/// @endcode
/// @par References
/// - RFC2104, https://www.rfc-editor.org/rfc/rfc2104.html .
/// - RFC4231 test vectors, https://www.rfc-editor.org/rfc/rfc4231.html .
/// @see hmac_sha256_hex(), hmac_sha256_trace()
inline HmacTag hmac_sha256(const std::string& key, const std::string& message) {
  return hmac_detail::execute(key, message, {0}, nullptr).tag;
}
/// @brief Format the preserved complete HMAC tag as 64 lowercase hexadecimal characters.
/// Same byte-domain, complexity, ownership and non-streaming contract as hmac_sha256().
/// @param key Complete borrowed key bytes.
/// @param message Complete borrowed message bytes.
/// @return Complete lowercase hex tag, never truncated.
/// @throws std::length_error On input/count/padded-length representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hmac_sha256_hex("Jefe","what do ya want for nothing?")); // 5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843
/// @endcode
/// @see hmac_sha256()
inline std::string hmac_sha256_hex(const std::string& key, const std::string& message) {
  return to_hex(hmac_sha256(key, message));
}
/// @brief Run the shared HMAC with complete input policy and actual call-boundary counts.
/// Same byte/arithmetic/storage contracts as hmac_sha256(); no intermediate capture.
/// @param key Complete borrowed key bytes.
/// @param message Complete borrowed message bytes.
/// @param options Combined input-byte cap, default 1m; zero disables policy.
/// @return Owned actual tag, key-normalization branch and selected work.
/// @throws std::length_error On complete input policy or representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hmac_sha256_run("k","m");
/// std::print("{} {} {}\n",r.work.sha_calls,r.work.bytes_hashed,r.work.padded_blocks); // 2 161 4
/// @endcode
/// @see HmacRun, HmacOptions
inline HmacRun
hmac_sha256_run(const std::string& key, const std::string& message, HmacOptions options = {}) {
  return hmac_detail::execute(key, message, options, nullptr);
}
/// @brief Capture actual key normalization, pads and inner digest from the producing HMAC.
/// Same complete input policy/storage/domain as hmac_sha256_run(). Fixed owned
/// intermediate arrays contain key material; no recomputation or round history.
/// @param key Complete borrowed key bytes; use public fixtures for displayed traces.
/// @param message Complete borrowed message bytes.
/// @param options Combined input-byte cap, default 1m; zero disables policy.
/// @return Owned actual intermediate bytes, final tag and selected work.
/// @throws std::length_error On complete input policy or representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hmac_sha256_trace("Jefe","x");
/// std::print("{} {} {}\n",t.run.key_hashed,t.key_block[0],t.inner_pad[0]); // false 74 124
/// @endcode
/// @see HmacTrace
inline HmacTrace
hmac_sha256_trace(const std::string& key, const std::string& message, HmacOptions options = {}) {
  HmacTrace t;
  t.run = hmac_detail::execute(key, message, options, &t);
  return t;
}
} // namespace datamunge::algorithms
