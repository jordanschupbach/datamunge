#pragma once
/// @file exp_golomb.hpp
/// @brief Full-range unsigned Exp-Golomb coding, checked bounded-prefix parsing and actual word spans.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {

/// @brief Logical values and emitted/consumed bits from the actual Exp-Golomb kernel.
/// zero_prefix_bits counts only leading zeros; quotient_bits includes the leading
/// one and remaining quotient-plus-one binary digits. suffix_bits counts the k raw
/// low bits per value. bit_visits is their sum, excluding preflight, allocation,
/// copies, validation and CPU instructions. Unread input tails are not visits.
///
///
/// @par Example
/// @code
/// const auto e=dm::algorithms::exp_golomb_encode_work({10},2);
/// std::print("{} {} {} {}\n",e.work.zero_prefix_bits,e.work.quotient_bits,e.work.suffix_bits,e.work.bit_visits); // 1 2 2 5
/// @endcode
///
/// @see exp_golomb_encode_work(), exp_golomb_decode_work()
struct ExpGolombWork {
    std::size_t values=0; ///< Complete values actually emitted or parsed.
    std::size_t zero_prefix_bits=0; ///< Actual leading zero-run positions.
    std::size_t quotient_bits=0; ///< Actual quotient-plus-one binary payload positions, including its first one.
    std::size_t suffix_bits=0; ///< Actual raw low-bit positions.
    std::size_t bit_visits=0; ///< Actual written/consumed bit positions, equal to the three part counts.
};

/// @brief Owned bit-vector payload and actual encoding work.
/// bits follows vector<bool> specialised storage, with no byte serialization or
/// endianness format defined. Order k and outer framing are supplied by the caller.
/// Empty input produces empty bits and zero work. No optional history/quota is kept.
///
///
/// @par Example
/// @code
/// const auto e=dm::algorithms::exp_golomb_encode_work({10},2);
/// std::print("{} {}\n",dm::algorithms::exp_golomb_bit_string(e.bits),e.work.values); // 01110 1
/// @endcode
///
/// @see exp_golomb_encode_work(), ExpGolombTrace
struct ExpGolombEncoded {
    std::vector<bool> bits; ///< Owned encoded bit sequence, first emitted bit first.
    ExpGolombWork work; ///< Actual logical emission counts.
};

/// @brief Owned complete decoded values, consumed prefix length and actual work.
/// The count argument is an upper limit. A complete frame ending earlier returns
/// fewer values; hitting the limit leaves any tail unread. consumed_bits identifies
/// that exact boundary. A malformed attempted word throws instead of returning
/// partial output. No input, optional span history or capture quota is retained.
///
///
/// @par Example
/// @code
/// const auto d=dm::algorithms::exp_golomb_decode_work({true,false},1);
/// std::print("{} {}\n",d.values[0],d.consumed_bits); // 0 1
/// @endcode
///
/// @see exp_golomb_decode_work(), ExpGolombTrace
struct ExpGolombDecoded {
    std::vector<std::uint64_t> values; ///< Owned complete values in parse order.
    std::size_t consumed_bits=0; ///< Exact bit prefix consumed; can be smaller than input size.
    ExpGolombWork work; ///< Actual successful parse counts.
};

/// @brief Actual emitted/parsed word with zero-run, quotient and low-bit boundaries.
/// All half-open indices address the trace bit vector. [begin,payload_begin) is
/// the zero run; [payload_begin,suffix_begin) is binary(q+1), including its leading
/// one; [suffix_begin,end) contains k raw low bits. At UINT64_MAX,k=0 the quotient
/// payload is a real 65-bit sequence, not a wrapped uint64 value. Records own scalars.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::exp_golomb_encode_trace({10},2);
/// const auto& w=t.words[0];
/// std::print("{} {} {} {}\n",w.begin,w.payload_begin,w.suffix_begin,w.end); // 0 1 3 5
/// @endcode
///
/// @see ExpGolombTrace
struct ExpGolombWord {
    std::uint64_t value=0; ///< Original value emitted or complete value parsed.
    std::size_t begin=0; ///< Start of the zero run.
    std::size_t payload_begin=0; ///< Actual leading-one position.
    std::size_t suffix_begin=0; ///< First low remainder bit, or end for order zero.
    std::size_t end=0; ///< First position after this word.
};

/// @brief Owned actual bit vector, values and spans from the shared Exp-Golomb execution.
/// Encoding owns the complete output bits and copied input values. Decoding owns
/// the entire input vector, including unread tails, plus actual complete values.
/// There is no explanatory replay. consumed_bits equals output size for encoding
/// and actual prefix length for decoding. Logical C=B+2N counts retained bit positions,
/// value entries and whole word records. Encoding reserves known C before copies or
/// emission; decoding reserves whole input B before copying, then two entries online
/// before each value/span append. Default one million; zero disables only quota.
/// Other metadata, runtime/capacity/container headers/allocator bytes are excluded:
/// C is not peak RAM and is not a byte-packing format. Exceptions expose no partial
/// trace; every returned container owns its values. Caller mutation is independent.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::exp_golomb_encode_trace({10},2);
/// std::print("{} {} {}\n",t.order,t.words[0].value,t.captured_entries); // 2 10 7
/// @endcode
///
/// @see exp_golomb_encode_trace(), exp_golomb_decode_trace()
struct ExpGolombTrace {
    int order=0; ///< Validated common order k in [0,63].
    std::vector<bool> bits; ///< Owned complete encoded output or complete decoder input.
    std::vector<std::uint64_t> values; ///< Owned input values for encoding or actual complete decoded values.
    std::vector<ExpGolombWord> words; ///< Owned actual word boundaries, one per value.
    ExpGolombWork work; ///< Actual logical kernel counts.
    std::size_t consumed_bits=0; ///< Actual written or consumed prefix length.
    std::size_t captured_entries=0; ///< Defined logical bit/value/word record capture count.
};

/// @brief Exact coded bits and a positive-common-width binary payload comparison.
/// For nonempty input fixed_width=max(1,bit_width(maximum)); all-zero values thus
/// use one bit per field by this convention. Empty input has zero width/counts and
/// maximum zero. fixed_bits=values*fixed_width. Order/count/width framing and storage
/// overhead are excluded. Computing a count does not promise available output memory.
///
///
/// @par Example
/// @code
/// const auto s=dm::algorithms::exp_golomb_size({0,1,3});
/// std::print("{} {} {} {}\n",s.values,s.bits,s.fixed_width,s.fixed_bits); // 3 9 2 6
/// @endcode
///
/// @see exp_golomb_size(), exp_golomb_code_length()
struct ExpGolombSize {
    std::size_t values=0; ///< Number of supplied nonnegative values.
    std::size_t bits=0; ///< Exact logical Exp-Golomb payload bits.
    std::uint64_t maximum=0; ///< Maximum value, or zero for empty input.
    std::size_t fixed_width=0; ///< Smallest positive common binary field width, or zero for empty input.
    std::size_t fixed_bits=0; ///< Logical common-width payload, excluding framing.
    int order=0; ///< Validated order used by the coded-bit calculation.
};

/// @brief Return the exact Exp-Golomb word length of any uint64 value.
/// Let q=n>>k and m=floor(log2(q+1)) mathematically. Length is 2m+1+k;
/// UINT64_MAX,k=0 uses m=64 and length129. Native arithmetic avoids forming
/// unrepresentable q+1 or shifting by64. Every uint64 including zero is supported.
///
/// @param n Nonnegative value in [0,UINT64_MAX].
/// @param k Common order (default0), restricted to [0,63].
/// @return Exact logical word bits, at most129.
/// @throws std::invalid_argument If k is outside [0,63].
///
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::exp_golomb_code_length(std::numeric_limits<std::uint64_t>::max()),dm::algorithms::exp_golomb_code_length(10,2)); // 129 5
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see exp_golomb_size(), exp_golomb_encode_work()
std::size_t exp_golomb_code_length(std::uint64_t n,int k=0);

/// @brief Measure exact coded size and positive-common-width binary size without encoding.
/// Scans values once, using only scalar counters. No per-word cache/output vector
/// is allocated. The fixed-width convention follows ExpGolombSize; comparison
/// metadata and physical storage overhead are excluded. Inputs are borrowed unchanged.
///
/// @param values Borrowed uint64 values; zero and empty input are valid.
/// @param k Common order (default0), in [0,63].
/// @return Owned scalar counts; empty input returns neutral zero counts.
/// @throws std::invalid_argument If k is outside [0,63], even for empty input.
/// @throws std::length_error On coded/fixed bit-count arithmetic overflow.
///
/// @par Example
/// @code
/// const auto s=dm::algorithms::exp_golomb_size({0,1,3,7,15,100,1000,65535});
/// std::print("{} {}\n",s.bits,s.fixed_bits); // 90 128
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see ExpGolombSize, exp_golomb_code_length()
ExpGolombSize exp_golomb_size(const std::vector<std::uint64_t>& values,int k=0);

/// @brief Render a bool bit sequence as owned literal zero/one debug characters.
/// Does not encode, decode or define byte packing. Allocates one char per input
/// bit, with O(B) time/storage. Empty input returns empty text. No view or input
/// mutation survives; independent calls are safe for unmodified borrowed input.
///
/// @param bits Borrowed specialised vector<bool> bit sequence.
/// @return Owned ASCII zero/one text with length equal to bits.size().
/// @throws std::length_error If text container geometry cannot hold B characters.
/// @throws std::bad_alloc If text storage cannot be allocated.
///
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::exp_golomb_bit_string({false,true,true,true,false})); // 01110
/// @endcode
///
/// @see ExpGolombEncoded
std::string exp_golomb_bit_string(const std::vector<bool>& bits);

/// @brief Append one actual order-k word and return counts for the appended word only.
/// Keeps arbitrary existing prefix bits unchanged and does not parse that prefix.
/// Validates order and complete output geometry, then reserves before any write.
/// Exceptions leave the original logical contents/size unchanged. Reallocation may
/// invalidate existing vector<bool> proxy references/iterators. Runtime O(word bits),
/// no optional history/quota; out must not be concurrently accessed during mutation.
///
/// @param n Any uint64 nonnegative value, including UINT64_MAX.
/// @param out Mutated bit vector; the complete new word is appended after its existing contents.
/// @param k Common order (default0), in [0,63].
/// @return Actual work for this single appended value, excluding prior prefix bits.
/// @throws std::invalid_argument If k is outside [0,63].
/// @throws std::length_error On append count/container geometry limits.
/// @throws std::bad_alloc If reservation fails.
///
/// @par Example
/// @code
/// std::vector<bool> bits{true,false};
/// const auto w=dm::algorithms::exp_golomb_append_work(0,bits);
/// std::print("{} {}\n",dm::algorithms::exp_golomb_bit_string(bits),w.bit_visits); // 101 1
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see exp_golomb0_encode_value(), exp_golomb_encode_work()
ExpGolombWork exp_golomb_append_work(std::uint64_t n,std::vector<bool>& out,int k=0);

/// @brief Encode nonnegative values and count actual zero, quotient and suffix bits.
/// For each n, emits a leading zero run, binary(q+1) for q=n>>k, then k raw
/// low bits, most significant first within each field. Supports the entire uint64
/// range, including the129bit order-zero maximum word. Checks full bit geometry
/// before allocation. Runtime O(N+B), owned output O(B); no optional span/input
/// history or capture quota. k and outer framing must be known by the decoder.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
///
/// @param values Borrowed uint64 sequence; zero and empty input are valid.
/// @param k Common order (default0), in [0,63], even for empty input.
/// @return Owned vector<bool> payload and actual emission counts.
/// @throws std::invalid_argument If k is outside [0,63].
/// @throws std::length_error On bit-counter or output-container limits.
/// @throws std::bad_alloc If required output storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto e=dm::algorithms::exp_golomb_encode_work({0,1,3});
/// std::print("{} {}\n",dm::algorithms::exp_golomb_bit_string(e.bits),e.work.bit_visits); // 101000100 9
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see ExpGolombEncoded, exp_golomb_encode_trace()
ExpGolombEncoded exp_golomb_encode_work(const std::vector<std::uint64_t>& values,int k=0);

/// @brief Decode a bounded complete-word prefix and expose its actual consumed length.
/// Parses complete words until count values are returned or the input ends at a
/// word boundary. count is an UPPER LIMIT: a shorter complete frame returns fewer
/// values; count0 reads nothing. Hitting the limit leaves any tail unread, without
/// validating that tail. consumed_bits reports the exact boundary. Any incomplete
/// attempted zero run, quotient payload or k-bit suffix throws; no partial result
/// is returned on that error. Wider-than-uint64 values fail instead of wrapping.
/// Runtime O(consumed bits), output O(actual values); input is borrowed unchanged,
/// without optional history/quota. Order must match the sender. Independent calls
/// are safe for stable inputs; input mutation during a call is unsafe.
///
/// @param bits Borrowed bool bit vector, first stream bit at index0; empty input is valid.
/// @param count Nonnegative maximum complete values to read; zero consumes no bits.
/// @param k Common order (default0), in [0,63].
/// @return Owned complete values, consumed prefix length and actual parse counts.
/// @throws std::invalid_argument On invalid k/count or an incomplete attempted word.
/// @throws std::overflow_error On an overwide quotient or unrepresentable uint64 value.
/// @throws std::length_error On actual output/counter/container limits.
/// @throws std::bad_alloc If output storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto d=dm::algorithms::exp_golomb_decode_work({true},2);
/// std::print("{} {} {}\n",d.values.size(),d.values[0],d.consumed_bits); // 1 0 1
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see ExpGolombDecoded, exp_golomb_decode_trace()
ExpGolombDecoded exp_golomb_decode_work(const std::vector<bool>& bits,int count,int k=0);

/// @brief Capture actual encoding spans while running the same word emitter.
/// For each n, emits a leading zero run, binary(q+1) for q=n>>k, then k raw
/// low bits, most significant first within each field. Supports the entire uint64
/// range, including the129bit order-zero maximum word. Checks full bit geometry
/// before allocation. Runtime O(N+B), owned output/history O(B+N). k and outer framing must be known by the decoder.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
/// C=B+2N follows ExpGolombTrace. Full known C and word/output geometry are
/// checked before input copies or emission; no partial trace escapes an exception.
///
/// @param values Borrowed uint64 sequence; zero and empty input are valid.
/// @param k Common order (default0), in [0,63], even for empty input.
/// @param max_entries Maximum logical capture entries (default1000000); zero disables only quota.
/// @return Owned actual bits, copied values, spans and counters.
/// @throws std::invalid_argument If k is outside [0,63].
/// @throws std::length_error On bit-counter or output-container limits.
/// @throws std::bad_alloc If required output storage cannot be allocated.
/// @throws std::length_error If known C exceeds an enabled capture limit.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::exp_golomb_encode_trace({10},2,7);
/// std::print("{} {}\n",dm::algorithms::exp_golomb_bit_string(t.bits),t.captured_entries); // 01110 7
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see ExpGolombTrace, ExpGolombWord
ExpGolombTrace exp_golomb_encode_trace(const std::vector<std::uint64_t>& values,int k=0,std::size_t max_entries=1000000);

/// @brief Capture actual bounded-prefix parsing spans with a checked history quota.
/// Parses complete words until count values are returned or the input ends at a
/// word boundary. count is an UPPER LIMIT: a shorter complete frame returns fewer
/// values; count0 reads nothing. Hitting the limit leaves any tail unread, without
/// validating that tail. consumed_bits reports the exact boundary. Any incomplete
/// attempted zero run, quotient payload or k-bit suffix throws; no partial result
/// is returned on that error. Wider-than-uint64 values fail instead of wrapping.
/// Parsing takes O(consumed bits); input is borrowed unchanged. Order must match the sender. Independent calls
/// are safe for stable inputs; input mutation during a call is unsafe.
/// The trace owns whole input B, including unread tails, and reserves B before
/// copying. It reserves two additional entries online before each complete value
/// and word record is appended. Thus C=B+2*actual_values, not the requested upper
/// limit. Capturing input adds O(B) time/storage, even when count0 consumes nothing.
///
/// @param bits Borrowed bool bit vector, first stream bit at index0; empty input is valid.
/// @param count Nonnegative maximum complete values to read; zero consumes no bits.
/// @param k Common order (default0), in [0,63].
/// @param max_entries Maximum logical captured entries (default1000000); zero disables only quota.
/// @return Owned whole input, actual values/spans, consumed boundary and work.
/// @throws std::invalid_argument On invalid k/count or an incomplete attempted word.
/// @throws std::overflow_error On an overwide quotient or unrepresentable uint64 value.
/// @throws std::length_error On actual output/counter/container limits.
/// @throws std::bad_alloc If output storage cannot be allocated.
/// @throws std::length_error If initial B or next complete value/word exceeds enabled quota.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::exp_golomb_decode_trace({true,false},1,0,4);
/// std::print("{} {} {}\n",t.values[0],t.consumed_bits,t.captured_entries); // 0 1 4
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see ExpGolombTrace, exp_golomb_decode_work()
ExpGolombTrace exp_golomb_decode_trace(const std::vector<bool>& bits,int count,int k=0,std::size_t max_entries=1000000);

/// @brief Append the order-zero word through the shared checked compiled emitter.
/// Retains the original inline entry point. Every uint64 is valid, including zero
/// and UINT64_MAX. Existing prefix bits are unchanged; complete geometry is checked
/// and reserved before appending. Exceptions preserve original logical contents/size;
/// reallocation may invalidate proxy references. Runtime O(word length), no history.
///
/// @param n Any nonnegative uint64 value.
/// @param out Mutable bit vector receiving the complete appended word.
/// @throws std::length_error On append bit-count/container limits.
/// @throws std::bad_alloc If output reservation fails.
///
/// @par Example
/// @code
/// std::vector<bool> bits;dm::algorithms::exp_golomb0_encode_value(3,bits);
/// std::print("{}\n",dm::algorithms::exp_golomb_bit_string(bits)); // 00100
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see exp_golomb_append_work()
inline void exp_golomb0_encode_value(std::uint64_t n,std::vector<bool>& out){(void)exp_golomb_append_work(n,out,0);}

/// @brief Encode nonnegative uint64 values, preserving the original inline API.
/// For each n, emits a leading zero run, binary(q+1) for q=n>>k, then k raw
/// low bits, most significant first within each field. Supports the entire uint64
/// range, including the129bit order-zero maximum word. Checks full bit geometry
/// before allocation. Runtime O(N+B), owned output O(B); no optional span/input
/// history or capture quota. k and outer framing must be known by the decoder.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
///
/// @param values Borrowed uint64 sequence; zero and empty input are valid.
/// @param k Common order (default0), in [0,63], even for empty input.
/// @return Owned bit payload, with empty output for empty input.
/// @throws std::invalid_argument If k is outside [0,63].
/// @throws std::length_error On bit-counter or output-container limits.
/// @throws std::bad_alloc If required output storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto bits=dm::algorithms::exp_golomb_encode({0,1,2});
/// std::print("{}\n",dm::algorithms::exp_golomb_bit_string(bits)); // 1010011
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see exp_golomb_encode_work(), exp_golomb_encode_trace()
inline std::vector<bool> exp_golomb_encode(const std::vector<std::uint64_t>& values,int k=0){return exp_golomb_encode_work(values,k).bits;}

/// @brief Decode up to count complete values, preserving original bounded-prefix behavior.
/// Parses complete words until count values are returned or the input ends at a
/// word boundary. count is an UPPER LIMIT: a shorter complete frame returns fewer
/// values; count0 reads nothing. Hitting the limit leaves any tail unread, without
/// validating that tail. consumed_bits reports the exact boundary. Any incomplete
/// attempted zero run, quotient payload or k-bit suffix throws; no partial result
/// is returned on that error. Wider-than-uint64 values fail instead of wrapping.
/// Runtime O(consumed bits), output O(actual values); input is borrowed unchanged,
/// without optional history/quota. Order must match the sender. Independent calls
/// are safe for stable inputs; input mutation during a call is unsafe.
/// Retains the original inline entry point. Use the work/trace result to obtain
/// consumed_bits; exact-count or whole-frame callers check their own framing.
///
/// @param bits Borrowed bool bit vector, first stream bit at index0; empty input is valid.
/// @param count Nonnegative maximum complete values to read; zero consumes no bits.
/// @param k Common order (default0), in [0,63].
/// @return Owned complete values in stream order; may contain fewer than count.
/// @throws std::invalid_argument On invalid k/count or an incomplete attempted word.
/// @throws std::overflow_error On an overwide quotient or unrepresentable uint64 value.
/// @throws std::length_error On actual output/counter/container limits.
/// @throws std::bad_alloc If output storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto values=dm::algorithms::exp_golomb_decode({true,false},1);
/// std::print("{} {}\n",values.size(),values[0]); // 1 0
/// @endcode
///
/// @par References
/// - ITU-T H.274 (05/2022), Versatile supplemental enhancement information messages for coded video bitstreams, section 9.1, equation (72). https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.274-202205-S!!PDF-E&lang=e&type=items
///
/// @see exp_golomb_decode_work(), exp_golomb_decode_trace()
inline std::vector<std::uint64_t> exp_golomb_decode(const std::vector<bool>& bits,int count,int k=0){return exp_golomb_decode_work(bits,count,k).values;}

} // namespace datamunge::algorithms
