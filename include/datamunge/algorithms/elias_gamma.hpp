#pragma once
/// @file elias_gamma.hpp
/// @brief Full-range positive uint64 Elias gamma coding, strict parsing and actual codeword spans.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {

/// @brief Logical value and bit counts from actual gamma emission or parsing.
/// Each successful codeword contributes its zero-prefix and full binary payload
/// lengths. bit_visits counts emitted characters for encoding or consumed characters
/// for decoding; validation, preflight, allocation/copying and CPU instructions are
/// excluded. Full input validation is an additional pass, not hidden in this count.
/// @par Example
/// @code
/// const auto r=dm::algorithms::elias_gamma_encode_work({13});
/// std::print("{} {} {}\n",r.work.zero_prefix_bits,r.work.payload_bits,r.work.bit_visits); // 3 4 7
/// @endcode
/// @see elias_gamma_encode_work(), elias_gamma_decode_work()
struct GammaWork {
    std::size_t values=0; ///< Complete positive values emitted or parsed.
    std::size_t zero_prefix_bits=0; ///< Sum of zero-prefix lengths.
    std::size_t payload_bits=0; ///< Sum of binary lengths, including each leading one.
    std::size_t bit_visits=0; ///< Written/consumed bits, equal to zero_prefix_bits+payload_bits.
};

/// @brief Owned encoded characters and actual emission work, with no optional history.
/// bits uses one char per bit, not packed bytes. No input view survives; ordinary
/// work calls have no optional capture quota. Empty input gives empty bits and zero work.
/// @par Example
/// @code
/// const auto r=dm::algorithms::elias_gamma_encode_work({1,2,5});
/// std::print("{} {}\n",r.bits,r.work.values); // 101000101 3
/// @endcode
/// @see elias_gamma_encode_work()
struct GammaEncoded {
    std::string bits; ///< Concatenated '0'/'1' characters in codeword order.
    GammaWork work; ///< Counts from the actual emitter.
};

/// @brief Owned strictly decoded positive values and actual parse work, without history.
/// Every input character must belong to one complete representable codeword;
/// trailing zeros are not padding. Empty input yields no values. No input views survive.
/// @par Example
/// @code
/// const auto r=dm::algorithms::elias_gamma_decode_work("101000101");
/// std::print("{} {}\n",r.values.size(),r.values[2]); // 3 5
/// @endcode
/// @see elias_gamma_decode_work()
struct GammaDecoded {
    std::vector<std::uint64_t> values; ///< Parsed values in original order.
    GammaWork work; ///< Counts from the actual strict parser.
};

/// @brief Actual half-open bit spans of one emitted or parsed gamma codeword.
/// The prefix occupies [begin,payload_begin), and the binary payload including its
/// leading one occupies [payload_begin,end). Spans refer to GammaTrace::bits,
/// which owns its data; numbers are indices, not borrowed iterators or views.
/// @par Example
/// @code
/// const auto t=dm::algorithms::elias_gamma_encode_trace({13});
/// std::print("{} {} {}\n",t.words[0].begin,t.words[0].payload_begin,t.words[0].end); // 0 3 7
/// @endcode
/// @see GammaTrace
struct GammaWord {
    std::uint64_t value=0; ///< Positive value associated with this complete word.
    std::size_t begin=0; ///< Inclusive prefix start.
    std::size_t payload_begin=0; ///< Index of the leading binary one.
    std::size_t end=0; ///< Exclusive payload/word end.
};

/// @brief Owned values, bit characters and spans captured during the actual codec run.
/// Retained logical entries C=B+2N count B bit characters, N values and N word
/// records. Scalar fields, result/runtime temporaries, string/vector capacity and
/// allocator bytes are excluded: this is not a peak-memory bound. Encoding knows
/// complete C before copies/emission; decoding knows B first and reserves two entries
/// per complete word online. Copies own independent buffers, with no input views.
/// @par Example
/// @code
/// const auto t=dm::algorithms::elias_gamma_decode_trace("0001101");
/// std::print("{} {} {}\n",t.values[0],t.encoding,t.captured_entries); // 13 false 9
/// @endcode
/// @see elias_gamma_encode_trace(), elias_gamma_decode_trace()
struct GammaTrace {
    bool encoding=false; ///< True for emission evidence, false for parsing evidence.
    std::vector<std::uint64_t> values; ///< Exact positive input or decoded values.
    std::string bits; ///< Exact owned encoded output or validated input.
    std::vector<GammaWord> words; ///< Actual complete-word spans, without replay.
    GammaWork work; ///< Logical work of this same run.
    std::size_t captured_entries=0; ///< Exact B+2N retained logical entries.
};

/// @brief Exact gamma and minimal common fixed-width payload sizes for positive values.
/// Empty input has zero sizes and maximum zero. Fixed width is the binary width of
/// the largest value; it does not include framing or a transmitted width/count.
/// These are theoretical bit counts. The string codec stores one char per bit.
/// @par Example
/// @code
/// const auto s=dm::algorithms::elias_gamma_size_summary({1,2,5});
/// std::print("{} {} {}\n",s.gamma_bits,s.fixed_width,s.fixed_bits); // 9 3 9
/// @endcode
/// @see elias_gamma_size_summary()
struct GammaSizeSummary {
    std::size_t values=0; ///< Number of positive values.
    std::uint64_t maximum=0; ///< Largest value, or zero for empty input.
    std::size_t gamma_bits=0; ///< Sum of 2*floor(log2(value))+1.
    std::size_t fixed_width=0; ///< Minimal binary width covering every value; zero if empty.
    std::size_t fixed_bits=0; ///< Checked values*fixed_width, excluding framing.
};

/// @brief Return the exact Elias gamma length of one positive uint64 value.
///
/// Uses integer bit width, with no floating logarithm or shift by 64. The full
/// uint64_t range is supported: lengths are odd and lie in [1,127]. No allocation,
/// input mutation or shared state. Zero is outside the code domain.
/// @param n Positive value, including UINT64_MAX.
/// @return 2*bit_width(n)-1 bits.
/// @throws std::invalid_argument If n is zero.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::elias_gamma_code_length(13)); // 7
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see elias_gamma_size_summary(), elias_gamma_encode_one()
std::size_t elias_gamma_code_length(std::uint64_t n);

/// @brief Summarize gamma payload and minimal common fixed-width bit counts.
///
/// Validates every positive value and checked sums/products before returning.
/// No encoded string or per-value history is allocated; O(N) word operations and
/// O(1) extra storage. Counts exclude framing/width metadata and allocation.
/// Input remains unchanged; independent calls share no mutable state.
/// @param values Positive values in arbitrary order; empty is valid.
/// @return Exact bit counts, width, count and maximum; all zero for empty input.
/// @throws std::invalid_argument If any value is zero.
/// @throws std::length_error On bit-count arithmetic overflow.
/// @par Example
/// @code
/// const auto s=dm::algorithms::elias_gamma_size_summary({1,2,5,3,100,7,42,1000,8,9});
/// std::print("{} {} {}\n",s.gamma_bits,s.fixed_width,s.fixed_bits); // 74 10 100
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see GammaSizeSummary, elias_gamma_code_length()
GammaSizeSummary elias_gamma_size_summary(const std::vector<std::uint64_t>& values);

/// @brief Emit positive uint64 gamma words with actual logical work counters.
///
/// Preflights all values and output size before allocation. For n, writes k zero
/// characters then its k+1 binary characters, k=bit_width(n)-1. All shifts are
/// in [0,63]. Empty input yields an empty string. O(N+B) work, O(B) output and
/// O(1) extra storage; one char per bit, not packed bytes. No optional history or
/// capture quota. Inputs are unchanged and calls share no mutable state.
/// @param values Positive uint64 values; zero is rejected, not silently omitted.
/// @return Owned concatenated bit characters and counters.
/// @throws std::invalid_argument If any value is zero.
/// @throws std::length_error On count arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::elias_gamma_encode_work({13});
/// std::print("{} {}\n",r.bits,r.work.bit_visits); // 0001101 7
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see GammaEncoded, elias_gamma_encode(), elias_gamma_encode_trace()
GammaEncoded elias_gamma_encode_work(const std::vector<std::uint64_t>& values);

/// @brief Strictly parse an entire gamma bit string with actual work counters.
///
/// Validates all characters before parsing. Every character must belong to a
/// complete codeword; trailing zeros are not padding. Prefixes up to 63 zeros
/// and the full uint64_t range are supported. Empty input returns no values.
/// Malformed input exposes no partial result. O(B) validation+parse work, O(N)
/// output and O(1) extra storage. No optional history/quota or input mutation.
/// Earlier permissive acceptance of incomplete/nonbinary streams is corrected.
/// @param bits Complete concatenated codewords, using only literal characters zero and one.
/// @return Owned positive values and consumed-bit counters.
/// @throws std::invalid_argument On nonbinary characters or truncated prefix/payload.
/// @throws std::overflow_error If a zero prefix exceeds 63, requiring a value wider than uint64_t.
/// @throws std::length_error On count arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::elias_gamma_decode_work("0001101");
/// std::print("{} {}\n",r.values[0],r.work.bit_visits); // 13 7
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see GammaDecoded, elias_gamma_decode(), elias_gamma_decode_trace()
GammaDecoded elias_gamma_decode_work(const std::string& bits);

/// @brief Capture actual emission spans after complete positive-input and quota preflight.
///
/// Shares the ordinary emitter; no replay. Computes complete B+2N entry cost and
/// checks bit/word container geometry before owned copies or output allocation.
/// Default one million logical entries, zero disables only quota; arithmetic
/// checks remain. See GammaTrace for exclusions; failures expose no partial trace.
/// @param values Positive values, copied exactly on success.
/// @param max_entries Retained logical-entry cap, default 1000000; zero disables it.
/// @return Owned emitted bits, input values, actual word spans and work.
/// @throws std::invalid_argument If any value is zero.
/// @throws std::length_error On quota, arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::elias_gamma_encode_trace({13},9);
/// std::print("{} {}\n",t.bits,t.captured_entries); // 0001101 9
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see GammaTrace, elias_gamma_encode_work()
GammaTrace elias_gamma_encode_trace(const std::vector<std::uint64_t>& values,std::size_t max_entries=1000000);

/// @brief Capture actual strict parse spans and values with an online retained-history cap.
///
/// Shares the ordinary parser; validates the complete binary alphabet first.
/// Reserves B input entries before owned copying, then two entries per complete
/// value/word record before appending. It may parse a word before knowing its
/// record cost; quota is retained history, not a CPU/peak-memory limit. Default
/// one million entries; zero disables only quota. No partial trace escapes failure.
/// @param bits Complete binary codewords; empty is valid and trailing zeros are rejected.
/// @param max_entries Retained logical-entry cap, default 1000000; zero disables it.
/// @return Owned original bits, parsed values, actual spans and work.
/// @throws std::invalid_argument On nonbinary characters or truncated prefix/payload.
/// @throws std::overflow_error If a zero prefix exceeds 63, requiring a value wider than uint64_t.
/// @throws std::length_error On count arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @throws std::length_error Also on optional retained-history quota exhaustion.
/// @par Example
/// @code
/// const auto t=dm::algorithms::elias_gamma_decode_trace("0001101",9);
/// std::print("{} {}\n",t.values[0],t.captured_entries); // 13 9
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see GammaTrace, elias_gamma_decode_work()
GammaTrace elias_gamma_decode_trace(const std::string& bits,std::size_t max_entries=1000000);

/// @brief Encode one positive integer, retaining the original inline entry point.
///
/// Forwards to the shared checked emitter. Full uint64_t range is supported;
/// zero now throws instead of becoming an empty word. Returns one char per bit,
/// not packed bytes. No optional history/quota; no input mutation or shared state.
/// @param n Positive uint64 value.
/// @return Owned gamma codeword of 1..127 characters.
/// @throws std::invalid_argument If any value is zero.
/// @throws std::length_error On count arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::elias_gamma_encode_one(13)); // 0001101
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see elias_gamma_encode_work(), elias_gamma_code_length()
inline std::string elias_gamma_encode_one(std::uint64_t n){return elias_gamma_encode_work({n}).bits;}

/// @brief Concatenate positive gamma words, retaining the original inline entry point.
///
/// Forwards to the shared checked emitter. Empty input returns empty output;
/// zero is rejected instead of silently disappearing. Valid legacy encodings
/// are unchanged. No optional history/quota; returned characters are not packed.
/// @param values Positive uint64 values, unchanged by the call.
/// @return Owned concatenation of complete words.
/// @throws std::invalid_argument If any value is zero.
/// @throws std::length_error On count arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::elias_gamma_encode({1,2,5})); // 101000101
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see elias_gamma_encode_work(), elias_gamma_encode_one()
inline std::string elias_gamma_encode(const std::vector<std::uint64_t>& values){return elias_gamma_encode_work(values).bits;}

/// @brief Decode complete gamma words, retaining the original inline entry point.
///
/// Forwards to the shared strict parser. Valid legacy streams are unchanged;
/// nonbinary text, truncated words and trailing zeros now fail explicitly.
/// Empty input returns no values. No optional history/quota and no partial result.
/// @param bits Complete binary codewords; unchanged by the call.
/// @return Owned decoded positive uint64 values.
/// @throws std::invalid_argument On nonbinary characters or truncated prefix/payload.
/// @throws std::overflow_error If a zero prefix exceeds 63, requiring a value wider than uint64_t.
/// @throws std::length_error On count arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto v=dm::algorithms::elias_gamma_decode("101000101");
/// std::print("{} {}\n",v.size(),v[2]); // 3 5
/// @endcode
/// @par References
/// - P. Elias (1975), Universal codeword sets and representations of the integers, IEEE Transactions on Information Theory 21(2), 194–203. doi:10.1109/TIT.1975.1055349.
/// @see elias_gamma_decode_work(), elias_gamma_decode_trace()
inline std::vector<std::uint64_t> elias_gamma_decode(const std::string& bits){return elias_gamma_decode_work(bits).values;}

} // namespace datamunge::algorithms
