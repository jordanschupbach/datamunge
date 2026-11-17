#pragma once

/// @file fnv_hash.hpp
/// @brief FNV-1a integer hashes and an owned trace of their actual byte updates.
/// Non-cryptographic fixed-width arithmetic; no text normalization or terminator.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Actual before/XOR/after state for one consumed byte, in integer hash order.
/// Owns primitive values, not a view into the input. Unsigned word arithmetic wraps.
/// @par Example
/// @code
/// const dm::algorithms::FNVStep step{};
/// const auto first = step.index; // 0
/// @endcode
/// @see FNVTrace, fnv1a_trace()
struct FNVStep {
  std::size_t   index{0};  ///< Zero-based byte position.
  unsigned      byte{0};   ///< Unsigned input byte in [0,255].
  std::uint64_t before{0}; ///< Accumulator before XOR, zero-extended for a 32-bit hash.
  std::uint64_t xored{0};  ///< Actual XOR result before multiplication.
  std::uint64_t after{0};  ///< Actual accumulator after multiplication modulo 2^bits.
};

/// @brief Owned hash value and one complete actual record per consumed byte.
/// Empty input retains zero records and its offset basis. Copying owns independent
/// history; input and ordinary scalar state are excluded from the record quota.
/// This is O(n) retained storage, unlike the ordinary hashes' O(1) state.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::fnv1a_trace("a", 32);
/// const auto n = trace.steps.size(); // 1
/// @endcode
/// @see fnv1a_trace(), FNVStep
struct FNVTrace {
  unsigned             bits{64}; ///< Requested integer word width, 32 or 64 for a completed trace.
  std::uint64_t        value{0}; ///< Final unsigned integer hash; default zero is unevaluated.
  std::vector<FNVStep> steps;    ///< Complete byte records, without replay or truncation.
};

/// @brief Shared byte-update kernel for ordinary hashes and tracing.
namespace fnv_detail {
/// @brief Run the canonical XOR/multiply loop once, optionally recording each update.
/// @tparam UInt Unsigned 32- or 64-bit accumulator type.
/// @param data Borrowed raw bytes, unchanged, including embedded NULs.
/// @param basis Initial accumulator.
/// @param prime FNV prime, multiplied in unsigned 64-bit arithmetic then narrowed.
/// @param steps Optional owned-history sink, already admitted and reserved by the caller.
/// @return Final accumulator; no input/callback/state references are retained.
/// @throws std::bad_alloc If optional record allocation fails.
template<class UInt>
inline UInt
hash(const std::string& data, UInt basis, std::uint64_t prime, std::vector<FNVStep>* steps) {
  UInt        h     = basis;
  std::size_t index = 0;
  for (unsigned char byte : data) {
    const std::uint64_t before = h;
    const std::uint64_t xored  = static_cast<std::uint64_t>(h) ^ byte;
    h                          = static_cast<UInt>(xored * prime);
    if (steps)
      steps->push_back({index, byte, before, xored, static_cast<std::uint64_t>(h)});
    ++index;
  }
  return h;
}
} // namespace fnv_detail

/// @brief 64-bit FNV-1a hash of a byte string, as RFC 9923's integer interface.
/// Starts at 14695981039346656037 and uses prime 1099511628211. Processes every
/// unsigned byte, including NUL, without Unicode decoding, normalization or an
/// appended terminator. Empty input returns the basis. Pure, thread-safe, O(n)
/// selected byte updates and O(1) auxiliary state; not a cryptographic hash.
/// Integer return is distinct from the RFC's little-endian serialized byte vector.
/// @param data Borrowed bytes, unchanged.
/// @return The unsigned integer accumulator modulo 2^64.
/// @par Example
/// @code
/// const auto h = dm::algorithms::fnv1a_64("a");
/// std::print("{:016X}\n", h); // AF63DC4C8601EC8C
/// @endcode
/// @par References
/// - RFC 9923 (2026), Sections 2 and 2.3: https://www.rfc-editor.org/rfc/rfc9923.html
/// @see fnv1a_32(), fnv1a_trace()
inline std::uint64_t fnv1a_64(const std::string& data) {
  return fnv_detail::hash<std::uint64_t>(data, 14695981039346656037ULL, 1099511628211ULL, nullptr);
}

/// @brief 32-bit FNV-1a integer hash, with the same byte semantics as fnv1a_64().
/// Basis 2166136261, prime 16777619; unsigned narrowing after every multiplication
/// implements modulo 2^32. Pure O(n) selected updates, O(1) auxiliary state.
/// @param data Borrowed raw bytes, unchanged, with no implicit terminator.
/// @return Unsigned 32-bit integer hash, including the basis for empty input.
/// @par Example
/// @code
/// std::print("{:08X}\n", dm::algorithms::fnv1a_32("a")); // E40C292C
/// @endcode
/// @par References
/// - RFC 9923 (2026), Section 2: https://www.rfc-editor.org/rfc/rfc9923.html#section-2
/// @see fnv1a_64(), fnv1a_trace()
inline std::uint32_t fnv1a_32(const std::string& data) {
  return fnv_detail::hash<std::uint32_t>(data, 2166136261u, 16777619u, nullptr);
}

/// @brief Trace the same native FNV-1a loop, retaining one actual record per byte.
/// The complete n-record quota is checked before allocation or hashing. Zero
/// disables the policy quota; vector capacity is still checked. No replay,
/// partial return, input copy, or peak-RAM interpretation of the quota. Calls
/// ordinary and traced kernels with identical arithmetic; pure and thread-safe.
/// @param data Borrowed bytes, unchanged, including embedded NULs.
/// @param bits Integer width, 32 or 64; default 64.
/// @param max_records Maximum complete byte records, default 1000000; zero disables this cap.
/// @return Owned final integer and all n byte updates, or no result on failure.
/// @throws std::invalid_argument If bits is neither 32 nor 64, before quota checks.
/// @throws std::length_error If the whole history exceeds quota or vector capacity.
/// @throws std::bad_alloc If admitted storage cannot be allocated.
/// @par Example
/// @code
/// const auto t = dm::algorithms::fnv1a_trace(std::string("a\0", 2), 32, 2);
/// std::print("{} {}\n", t.steps.size(), t.steps[1].byte); // 2 0
/// @endcode
/// @par References
/// - RFC 9923 (2026), Section 2: https://www.rfc-editor.org/rfc/rfc9923.html#section-2
/// @see fnv1a_32(), fnv1a_64(), FNVTrace
inline FNVTrace
fnv1a_trace(const std::string& data, unsigned bits = 64, std::size_t max_records = 1000000) {
  if (bits != 32 && bits != 64)
    throw std::invalid_argument("fnv1a_trace: bits must be 32 or 64");
  FNVTrace trace;
  trace.bits = bits;
  if ((max_records && data.size() > max_records) || data.size() > trace.steps.max_size())
    throw std::length_error("fnv1a_trace: complete byte-record quota");
  trace.steps.reserve(data.size());
  trace.value = bits == 32
                    ? fnv_detail::hash<std::uint32_t>(data, 2166136261u, 16777619u, &trace.steps)
                    : fnv_detail::hash<std::uint64_t>(
                          data, 14695981039346656037ULL, 1099511628211ULL, &trace.steps);
  return trace;
}
} // namespace datamunge::algorithms
