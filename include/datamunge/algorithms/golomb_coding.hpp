#pragma once
/// @file golomb_coding.hpp
/// @brief Checked Golomb/Rice bit-vector coding and actual owned codeword spans.
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::algorithms {
/// @brief Actual complete codeword boundaries and decoded/emitted integer parts.
/// Owns scalars; indices are half-open bit positions in the associated trace.
/// @par Example
/// @code
/// const auto t=dm::algorithms::golomb_encode_trace({8},3);
/// std::print("{} {} {}\n",t.words[0].quotient,t.words[0].remainder,t.words[0].end); // 2 2 5
/// @endcode
/// @see GolombTrace
struct GolombWord {
  std::uint64_t value{0};           ///< Actual original or parsed value.
  std::uint64_t quotient{0};        ///< Unary one count.
  std::uint64_t remainder{0};       ///< Remainder before truncated-binary mapping.
  std::size_t   begin{0};           ///< First unary one, or terminator when quotient is zero.
  std::size_t   remainder_begin{0}; ///< First remainder bit, immediately after the unary zero.
  std::size_t   end{0};             ///< First position after the complete codeword.
};
/// @brief Owned actual bits, complete spans and selected bit-visit counts.
/// Encoding owns emitted bits; decoding owns only the consumed prefix, excluding
/// any unread tail. Each word owns its value; no input references survive.
/// Unary counts include terminators. Remainder counts include actual short/long
/// payload positions. These selected visits exclude preflight, copies and CPU costs.
/// The quota counts words only, not bits, bytes, capacity or peak RAM.
/// @par Example
/// @code
/// const auto t=dm::algorithms::golomb_encode_trace({8},3);
/// std::print("{} {} {}\n",t.bits.size(),t.unary_bits,t.remainder_bits); // 5 3 2
/// @endcode
/// @see golomb_encode_trace(), golomb_decode_trace()
struct GolombTrace {
  std::vector<bool>       bits;  ///< Owned emitted bits or exact consumed input prefix.
  std::vector<GolombWord> words; ///< Owned actual chronological complete codeword records.
  std::size_t unary_bits{0};     ///< Actual unary positions emitted/consumed, including zeros.
  std::size_t remainder_bits{0}; ///< Actual binary payload positions emitted/consumed.
};
/// @brief Private checked parameter geometry and shared emit/parse kernels.
namespace golomb_detail {
/// @brief Validate the positive signed-int Golomb parameter before division or shifts.
/// @param M Common parameter, including on empty inputs.
/// @throws std::invalid_argument If M is not positive.
inline void parameter(int M) {
  if (M <= 0)
    throw std::invalid_argument("Golomb M must be positive");
}
/// @brief Smallest bit width whose unsigned power of two covers M.
/// @param M Valid positive signed-int parameter.
/// @return Width in [0,31] on a 32-bit-int platform.
inline unsigned width(int M) {
  return std::bit_width(static_cast<unsigned>(M) - 1);
}
/// @brief Truncated-binary threshold without signed left shifts.
/// @param M Valid positive parameter.
/// @return 2^ceil(log2(M))-M, representable in uint64.
inline std::uint64_t cutoff(int M) {
  return (std::uint64_t{1} << width(M)) - static_cast<unsigned>(M);
}
/// @brief Checked addition of logical bit counts.
/// @param a Existing count.
/// @param b Increment.
/// @return Nonwrapping sum.
/// @throws std::length_error On size_t overflow.
inline std::size_t add(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("Golomb size overflow");
  return a + b;
}
/// @brief Determine payload width for one already validated value.
/// @param n Nonnegative integer.
/// @param M Valid positive parameter.
/// @return Remainder width, including zero when M=1.
inline unsigned payload(std::uint64_t n, int M) {
  auto b = width(M);
  return n % M < cutoff(M) ? b - 1 : b;
}
/// @brief Complete mathematical word size, checked before allocation or mutation.
/// @param n Nonnegative value.
/// @param M Valid positive parameter.
/// @return Bit count representable by size_t.
/// @throws std::length_error On logical overflow.
inline std::size_t length(std::uint64_t n, int M) {
  const auto q = n / M;
  if (q > SIZE_MAX)
    throw std::length_error("Golomb quotient size overflow");
  return add(add(static_cast<std::size_t>(q), 1), payload(n, M));
}
/// @brief Preflight total emitted size before any output mutation.
/// @param values Borrowed values.
/// @param M Valid parameter.
/// @return Complete bit count within bit-container limits.
/// @throws std::length_error On arithmetic or container limits.
inline std::size_t total(const std::vector<std::uint64_t>& values, int M) {
  std::size_t n = 0;
  for (auto x : values)
    n = add(n, length(x, M));
  if (n > std::vector<bool>{}.max_size())
    throw std::length_error("Golomb bit capacity");
  return n;
}
/// @brief Admit a next whole word before its bit operations.
/// @param words Current history, borrowed only for this check.
/// @param cap Optional word quota; zero disables the policy cap.
/// @throws std::length_error If the next word exceeds policy/container limits.
inline void admit(const std::vector<GolombWord>& words, std::size_t cap) {
  if ((cap && words.size() >= cap) || words.size() >= words.max_size())
    throw std::length_error("Golomb word quota");
}
/// @brief Emit one word into sufficiently reserved storage and capture actual positions.
/// @param n Value to emit.
/// @param M Valid positive parameter.
/// @param out Borrowed output with enough reserved capacity.
/// @return Actual complete word geometry.
inline GolombWord emit(std::uint64_t n, int M, std::vector<bool>& out) {
  const auto begin = out.size();
  const auto q = n / M, r = n % M;
  for (std::uint64_t j = 0; j < q; ++j)
    out.push_back(true);
  out.push_back(false);
  const auto start = out.size();
  const auto c     = cutoff(M);
  const auto b     = payload(n, M);
  const auto v     = r < c ? r : r + c;
  for (unsigned j = b; j > 0; --j)
    out.push_back((v >> (j - 1)) & 1);
  return {n, q, r, begin, start, out.size()};
}
/// @brief Retain a complete native word and update selected visit counters.
/// @param trace Owned capture under construction.
/// @param word Actual completed word.
/// @throws std::bad_alloc If record allocation fails.
inline void record(GolombTrace& trace, const GolombWord& word) {
  trace.words.push_back(word);
  trace.unary_bits += word.remainder_begin - word.begin;
  trace.remainder_bits += word.end - word.remainder_begin;
}
/// @brief Shared sequence emitter, with optional capture of actual emitted words.
/// @param values Borrowed integers.
/// @param M Validated by this kernel even for empty input.
/// @param trace Optional borrowed capture sink; null disables history.
/// @param cap Optional whole-word cap, zero disables it.
/// @return Owned complete bit sequence.
/// @throws std::invalid_argument For invalid M.
/// @throws std::length_error For complete bit size or known word quota violations.
/// @throws std::bad_alloc For allocation failure.
inline std::vector<bool>
encode(const std::vector<std::uint64_t>& values, int M, GolombTrace* trace, std::size_t cap) {
  parameter(M);
  const auto n = total(values, M);
  if (trace && ((cap && values.size() > cap) || values.size() > trace->words.max_size()))
    throw std::length_error("Golomb word quota");
  std::vector<bool> out;
  out.reserve(n);
  if (trace)
    trace->words.reserve(values.size());
  for (auto x : values) {
    auto w = emit(x, M, out);
    if (trace)
      record(*trace, w);
  }
  return out;
}
/// @brief Read one checked input position, advancing only a local parser cursor.
/// @param bits Borrowed input.
/// @param pos Current cursor, incremented after a valid read.
/// @return Actual bit at the old position.
/// @throws std::invalid_argument At an incomplete attempted word.
inline bool read(const std::vector<bool>& bits, std::size_t& pos) {
  if (pos == bits.size())
    throw std::invalid_argument("truncated Golomb word");
  return bits[pos++];
}
/// @brief Reconstruct a scalar value without overflowing quotient multiplication or addition.
/// @param q Parsed quotient.
/// @param r Parsed remainder, less than valid positive M.
/// @param M Valid positive parameter.
/// @return Exact uint64 value when representable.
/// @throws std::overflow_error For an unrepresentable value.
inline std::uint64_t reconstruct(std::uint64_t q, std::uint64_t r, int M) {
  const auto m = static_cast<unsigned>(M);
  if (q > UINT64_MAX / m || r > UINT64_MAX - q * m)
    throw std::overflow_error("Golomb value overflow");
  return q * m + r;
}
/// @brief Shared prefix parser with checked value reconstruction and optional actual capture.
/// @param bits Borrowed input; unread tails are not validated.
/// @param M Common positive parameter, checked even for count zero.
/// @param count Maximum number of complete words; nonnegative.
/// @param trace Optional capture sink, initially empty.
/// @param cap Online complete-word quota; zero disables policy cap.
/// @return Owned decoded values, possibly fewer at a clean end of input.
/// @throws std::invalid_argument For parameter/count or a truncated attempted word.
/// @throws std::overflow_error If a complete attempted value cannot fit uint64.
/// @throws std::length_error For a next history record beyond quota/container capacity.
/// @throws std::bad_alloc On allocation failure.
inline std::vector<std::uint64_t>
decode(const std::vector<bool>& bits, int M, int count, GolombTrace* trace, std::size_t cap) {
  parameter(M);
  if (count < 0)
    throw std::invalid_argument("Golomb count must be nonnegative");
  std::vector<std::uint64_t> out;
  std::size_t                pos = 0;
  const auto                 b   = width(M);
  const auto                 c   = cutoff(M);
  for (int i = 0; i < count && pos < bits.size(); ++i) {
    if (trace)
      admit(trace->words, cap);
    const auto    begin = pos;
    std::uint64_t q     = 0;
    while (read(bits, pos)) {
      if (q == UINT64_MAX / static_cast<unsigned>(M))
        throw std::overflow_error("Golomb quotient overflow");
      ++q;
    }
    const auto    start = pos;
    std::uint64_t r     = 0;
    if (b) {
      for (unsigned j = 0; j < b - 1; ++j)
        r = (r << 1) | read(bits, pos);
      if (r >= c)
        r = ((r << 1) | read(bits, pos)) - c;
    }
    const auto value = reconstruct(q, r, M);
    out.push_back(value);
    if (trace)
      record(*trace, {value, q, r, begin, start, pos});
  }
  if (trace)
    trace->bits.assign(bits.begin(), bits.begin() + static_cast<std::ptrdiff_t>(pos));
  return out;
}
/// @brief Validate Rice order before forming a signed-int-compatible Golomb parameter.
/// @param k Order, in [0,30] on a 32-bit-int platform.
/// @return Positive 2^k representable by int.
/// @throws std::invalid_argument If k is negative or exceeds int value width minus one.
inline int rice_parameter(int k) {
  if (k < 0 || k >= std::numeric_limits<int>::digits)
    throw std::invalid_argument("Rice order outside int parameter range");
  return static_cast<int>(unsigned{1} << k);
}
} // namespace golomb_detail

/// @brief Compute a Golomb word's exact logical bit size without emitting it.
/// O(1) scalar work, no allocation; M=1 gives unary length n+1 when representable.
/// @param n Nonnegative integer.
/// @param M Positive int parameter.
/// @return Complete mathematical bit count, independent of allocator capacity.
/// @throws std::invalid_argument If M<=0.
/// @throws std::length_error If the count cannot fit size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::golomb_code_length(8,3)); // 5
/// @endcode
/// @par References
/// - S. W. Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see golomb_encode_value()
inline std::size_t golomb_code_length(std::uint64_t n, int M) {
  golomb_detail::parameter(M);
  return golomb_detail::length(n, M);
}

/// @brief Append an actual Golomb word after complete size/capacity preflight.
/// Writes q ones, one zero and an MSB-first truncated-binary remainder. M=1 has
/// no remainder. Existing prefix bits are preserved. Parameter/length/allocation
/// failures occur before mutation; work/storage are O(word bits), not O(1).
/// No packed-byte wire format, parameter header or framing is supplied.
/// @param n Nonnegative integer.
/// @param M Positive int parameter, retained historical signature.
/// @param out Borrowed bit vector, appended only after successful reserve.
/// @throws std::invalid_argument If M<=0.
/// @throws std::length_error On logical or container size overflow.
/// @throws std::bad_alloc On reserve failure.
/// @par Example
/// @code
/// std::vector<bool> b{true}; dm::algorithms::golomb_encode_value(0,1,b);
/// std::print("{}\n",b.size()); // 2
/// @endcode
/// @par References
/// - S. W. Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see golomb_encode(), golomb_code_length()
inline void golomb_encode_value(std::uint64_t n, int M, std::vector<bool>& out) {
  auto total = golomb_detail::add(out.size(), golomb_code_length(n, M));
  if (total > out.max_size())
    throw std::length_error("Golomb bit capacity");
  out.reserve(total);
  golomb_detail::emit(n, M, out);
}
/// @brief Encode a sequence with checked complete sizing and no retained word history.
/// Same bit convention as golomb_encode_value(). Empty input returns empty after
/// parameter validation. Input is borrowed only during the call. O(values+bits)
/// selected work, O(bits) output storage. Output is not a byte serialization.
/// @param values Nonnegative integers.
/// @param M Common positive int parameter.
/// @return Owned actual emitted bit vector.
/// @throws std::invalid_argument If M<=0.
/// @throws std::length_error On logical/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto b=dm::algorithms::golomb_encode({8},3);
/// std::print("{}\n",b.size()); // 5
/// @endcode
/// @par References
/// - S. W. Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see golomb_decode(), golomb_encode_trace()
inline std::vector<bool> golomb_encode(const std::vector<std::uint64_t>& values, int M) {
  return golomb_detail::encode(values, M, nullptr, 0);
}
/// @brief Decode at most count complete words using checked bounded input reads.
/// Clean input exhaustion returns fewer values; a partial attempted word throws.
/// Reaching count leaves any tail unread and unvalidated. M and count are external
/// framing; byte-padding is not ignored automatically. No inputs are mutated.
/// O(consumed bits+values) work and O(values) output; no trace allocation.
/// @param bits Borrowed first-bit-first input.
/// @param M Positive int parameter.
/// @param count Nonnegative maximum number of words; zero reads no bits.
/// @return Owned decoded values; no partial result escapes on failure.
/// @throws std::invalid_argument For invalid M/count or incomplete attempted words.
/// @throws std::overflow_error For an unrepresentable decoded uint64 value.
/// @throws std::length_error On output container capacity limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto v=dm::algorithms::golomb_decode({true,true,false,true,true},3,1);
/// std::print("{}\n",v[0]); // 8
/// @endcode
/// @par References
/// - S. W. Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see golomb_decode_trace(), golomb_encode()
inline std::vector<std::uint64_t> golomb_decode(const std::vector<bool>& bits, int M, int count) {
  return golomb_detail::decode(bits, M, count, nullptr, 0);
}
/// @brief Capture actual encoding words in the ordinary emitter, without replay.
/// Known complete word quota and bit sizing are checked before emission/allocation.
/// History has O(values) entries in addition to the ordinary O(bits) output.
/// No input references survive. Zero max_words disables only the policy cap.
/// @param values Borrowed nonnegative integers.
/// @param M Positive int parameter.
/// @param max_words Whole-record quota, default 1000000; zero disables it.
/// @return Owned emitted bits, actual words and selected position counts.
/// @throws std::invalid_argument For M<=0, before sizing/quota.
/// @throws std::length_error On word quota or bit/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::golomb_encode_trace({8},3,1);
/// std::print("{}\n",t.words.size()); // 1
/// @endcode
/// @see GolombTrace, golomb_encode()
inline GolombTrace golomb_encode_trace(
    const std::vector<std::uint64_t>& values,
    int                               M,
    std::size_t                       max_words = 1000000) {
  GolombTrace t;
  t.bits = golomb_detail::encode(values, M, &t, max_words);
  return t;
}
/// @brief Capture complete actual parsed words and own their consumed bit prefix.
/// Online admission occurs before attempting each word. Unread tails are excluded
/// from trace bits and all counters. Malformed/quota failures return no partial
/// history. The quota limits word records, not input/output bit storage or bytes.
/// @param bits Borrowed first-bit-first input, unmodified.
/// @param M Positive int parameter.
/// @param count Nonnegative maximum complete words.
/// @param max_words Whole-record quota, default 1000000; zero disables it.
/// @return Owned consumed bits, actual parsed words and selected visits.
/// @throws std::invalid_argument For invalid M/count or partial attempted words.
/// @throws std::overflow_error On unrepresentable values.
/// @throws std::length_error On next-record quota/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::golomb_decode_trace({true,true,false,true,true},3,1);
/// std::print("{}\n",t.words[0].value); // 8
/// @endcode
/// @see GolombTrace, golomb_decode()
inline GolombTrace golomb_decode_trace(
    const std::vector<bool>& bits,
    int                      M,
    int                      count,
    std::size_t              max_words = 1000000) {
  GolombTrace t;
  (void)golomb_detail::decode(bits, M, count, &t, max_words);
  return t;
}
/// @brief Encode Rice words using the same native Golomb emitter with M=2^k.
/// Same ownership, bit convention, complexity and failures as golomb_encode().
/// @param values Borrowed nonnegative integers.
/// @param k Order in [0, numeric_limits<int>::digits-1], normally [0,30].
/// @return Owned actual bits.
/// @throws std::invalid_argument For invalid k.
/// @throws std::length_error On bit/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto b=dm::algorithms::rice_encode({8},2);
/// std::print("{}\n",b.size()); // 5
/// @endcode
/// @see golomb_encode(), rice_decode()
inline std::vector<bool> rice_encode(const std::vector<std::uint64_t>& values, int k) {
  return golomb_encode(values, golomb_detail::rice_parameter(k));
}
/// @brief Decode Rice words through the same checked native Golomb parser.
/// Same clean exhaustion, attempted-word and unread-tail semantics as golomb_decode().
/// @param bits Borrowed first-bit-first input.
/// @param k Order in [0, numeric_limits<int>::digits-1], normally [0,30].
/// @param count Nonnegative maximum complete words.
/// @return Owned decoded values.
/// @throws std::invalid_argument For invalid k/count or partial attempted words.
/// @throws std::overflow_error On unrepresentable values.
/// @throws std::length_error On output container capacity limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto v=dm::algorithms::rice_decode(dm::algorithms::rice_encode({8},2),2,1);
/// std::print("{}\n",v[0]); // 8
/// @endcode
/// @see golomb_decode(), rice_encode()
inline std::vector<std::uint64_t> rice_decode(const std::vector<bool>& bits, int k, int count) {
  return golomb_decode(bits, golomb_detail::rice_parameter(k), count);
}
/// @brief Copy logical bits into a printable first-bit-first string.
/// No wire format or byte packing is implied. O(bits) work and owned characters.
/// @param bits Borrowed logical bits.
/// @return Owned sequence of ASCII zeros and ones, empty for empty input.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error If string capacity is exceeded.
/// @par Example
/// @code
/// const auto bits=dm::algorithms::golomb_encode({8},3);
/// std::print("{}\n",dm::algorithms::golomb_bit_string(bits)); // 11011
/// @endcode
/// @see golomb_encode()
inline std::string golomb_bit_string(const std::vector<bool>& bits) {
  std::string s;
  s.reserve(bits.size());
  for (bool b : bits)
    s.push_back(b ? '1' : '0');
  return s;
}
} // namespace datamunge::algorithms
