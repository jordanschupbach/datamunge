#pragma once
/// @file fibonacci_coding.hpp
/// @brief Full-positive-uint64 Fibonacci coding, strict whole-frame parsing and actual spans.
/// Zeckendorf payloads use weights1,2,3,5,... least weight first and append one
/// terminating bit. The final11 is unambiguous from a known frame start. This is
/// not error detection/correction, a byte-packing format or a blanket guarantee
/// of recovery from arbitrary corruption or arbitrary starting offsets.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Logical complete-value and bit counts from the actual Fibonacci codec.
/// Payload_bits counts Zeckendorf digits, including their highest one. Each word
/// adds one terminator bit; bit_visits is their sum. Counts exclude table lookup,
/// preflight, greedy candidate checks, copying, allocation and CPU instructions.
/// Empty inputs give zero counts; scalars are owned without shared state.
/// @par Example
/// @code
/// const auto e=dm::algorithms::fibonacci_encode_work({10});
/// std::print("{} {} {}\n",e.work.payload_bits,e.work.terminator_bits,e.work.bit_visits); // 5 1 6
/// @endcode
/// @see FibonacciEncoded, FibonacciDecoded
struct FibonacciWork {
    std::size_t values=0; ///< Complete positive values actually emitted or parsed.
    std::size_t payload_bits=0; ///< Actual Fibonacci payload positions.
    std::size_t terminator_bits=0; ///< One appended/consumed terminating bit per complete word.
    std::size_t bit_visits=0; ///< Actual emitted/consumed bits; payload plus terminator.
};

/// @brief Owned packed-bool encoded payload and actual emission work.
/// Empty values give empty bits. Output uses vector<bool> storage; no byte order,
/// serialization, padding or external frame metadata is defined. No optional
/// history/capture quota; copies own their buffers independently.
/// @par Example
/// @code
/// const auto e=dm::algorithms::fibonacci_encode_work({1,2});
/// std::print("{} {}\n",e.bits.size(),e.work.values); // 5 2
/// @endcode
/// @see fibonacci_encode_work(), FibonacciTrace
struct FibonacciEncoded {
    std::vector<bool> bits; ///< Owned emitted bit sequence, first bit first.
    FibonacciWork work; ///< Actual same-pass emission counts.
};

/// @brief Owned complete positive values and work from strict whole-frame parsing.
/// Every input bit must belong to a complete representable codeword; unfinished
/// tails are errors, not padding. Empty bits give no values. No borrowed input
/// references survive, and no partial result is returned on a parse error.
/// @par Example
/// @code
/// const auto d=dm::algorithms::fibonacci_decode_work({true,true,false,true,true});
/// std::print("{} {}\n",d.values[1],d.work.bit_visits); // 2 5
/// @endcode
/// @see fibonacci_decode_work(), FibonacciTrace
struct FibonacciDecoded {
    std::vector<std::uint64_t> values; ///< Owned parsed positive values in order.
    FibonacciWork work; ///< Actual same-pass complete-word and bit counts.
};

/// @brief Actual half-open span and overlapping delimiter of one complete word.
/// Begin and end address FibonacciTrace::bits. Payload is [begin,end-1); the
/// two-bit delimiter occupies [delimiter_begin,end) with delimiter_begin=end-2.
/// Its first one is also the highest payload digit. Indices are owned scalars,
/// not iterators. Spans are captured during emission/parsing without replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fibonacci_encode_trace({10});
/// std::print("{} {} {}\n",t.words[0].begin,t.words[0].delimiter_begin,t.words[0].end); // 0 4 6
/// @endcode
/// @see FibonacciTrace, fibonacci_encode_trace()
struct FibonacciWord {
    std::uint64_t value=0; ///< Exact positive value associated with this word.
    std::size_t begin=0; ///< Inclusive first payload bit.
    std::size_t delimiter_begin=0; ///< Highest payload one, followed by the appended terminator one.
    std::size_t end=0; ///< Exclusive end after the terminating one.
};

/// @brief Owned same-pass Fibonacci bits, values, spans and work.
/// Logical capture C=B+2N counts B retained bit positions, N value entries and
/// N whole FibonacciWord records. Scalars, vector capacity/headers, allocators
/// and runtime temporaries are excluded: C is not peak RAM. Encoding admits
/// complete C before copies/emission. Decoding admits B before copying, then
/// two entries online per complete word before its value/span append. Copies
/// are independent; edits do not recompute work/spans. No nested digit history.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fibonacci_decode_trace({true,true});
/// std::print("{} {} {}\n",t.values[0],t.encoding,t.captured_entries); // 1 false 4
/// @endcode
/// @see fibonacci_encode_trace(), fibonacci_decode_trace()
struct FibonacciTrace {
    bool encoding=false; ///< True for emission evidence, false for parsing evidence.
    std::vector<std::uint64_t> values; ///< Owned original or decoded positive values.
    std::vector<bool> bits; ///< Owned complete emitted output or complete input frame.
    std::vector<FibonacciWord> words; ///< Actual complete-word spans in execution order.
    FibonacciWork work; ///< Actual work of the same codec pass.
    std::size_t captured_entries=0; ///< Exact retained logical B+2N entries.
};

/// @brief Return one exact Zeckendorf coding weight from the complete uint64 table.
/// Index0 is1, index1 is2; index91 is12200160415121876738. The next weight
/// exceeds uint64 and is never formed in that type. Constant lookup/storage,
/// no allocation, mutation or shared mutable state.
/// @param index Zero-based coding-weight index in [0,91].
/// @return Exact Fibonacci F(index+2), with conventional F1=F2=1.
/// @throws std::out_of_range If index>=92.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::fibonacci_coding_weight(0),dm::algorithms::fibonacci_coding_weight(91)); // 1 12200160415121876738
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see fibonacci_code_length(), FibonacciWord
std::uint64_t fibonacci_coding_weight(std::size_t index);

/// @brief Return the exact Fibonacci codeword length of a positive uint64.
/// Counts coding weights<=n then adds one terminating bit. Integer-only search
/// over the92-weight table, with lengths2..93; UINT64_MAX has93bits. No floating
/// logarithms, allocation, mutation or shared state.
/// @param n Positive integer, including UINT64_MAX.
/// @return Number of payload bits plus one terminating bit.
/// @throws std::invalid_argument If n==0.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::fibonacci_code_length(10),dm::algorithms::fibonacci_code_length(18446744073709551615ULL)); // 6 93
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see fibonacci_encode_value(), fibonacci_coding_weight()
std::size_t fibonacci_code_length(std::uint64_t n);

/// @brief Append one full-range positive Fibonacci word after arbitrary existing bits.
/// Keeps the original callable signature and prefix unchanged; it does not parse
/// the prefix. Validates n, checked total size and reserves before writing, so
/// domain/length/allocation failures leave out unchanged. Greedy descending
/// Zeckendorf digits are emitted least weight first, then one terminator. At
/// most92digits; O(log(n)+1) work, constant scratch plus appended output.
/// Independent calls are safe on separate outputs.
/// @param n Positive value; zero is rejected before any mutation.
/// @param out Borrowed mutable bit vector; receives one complete word at its end.
/// @throws std::invalid_argument If n==0.
/// @throws std::length_error On checked output-size or container limits.
/// @throws std::bad_alloc If reserve cannot allocate storage.
/// @par Example
/// @code
/// std::vector<bool> bits{false};
/// dm::algorithms::fibonacci_encode_value(1,bits);
/// std::print("{} {}\n",bits.size(),bits.front()); // 3 false
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see fibonacci_encode(), fibonacci_code_length()
void fibonacci_encode_value(std::uint64_t n,std::vector<bool>& out);

/// @brief Emit positive full-range Fibonacci words into an owned result.
/// Preflights every positive value and total output size before emission. Empty
/// input yields empty output. Canonical greedy digits and terminator preserve
/// existing valid codewords; UINT64_MAX is supported. O(N+B) word work and O(B)
/// output, with constant scratch; preflight is excluded from bit_visits.
/// No optional history or capture quota; calls share no mutable state.
/// @param values Borrowed positive uint64 values; zero is invalid.
/// @return Owned emitted bits.
/// @throws std::invalid_argument If any value is zero.
/// @throws std::length_error On checked count/container limits.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// const auto bits=dm::algorithms::fibonacci_encode({1,2,10});
/// std::print("{}\n",bits.size()); // 11
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see FibonacciTrace, FibonacciWork, fibonacci_code_length()
std::vector<bool> fibonacci_encode(const std::vector<std::uint64_t>& values);

/// @brief Emit positive full-range Fibonacci words with actual work counts.
/// Preflights every positive value and total output size before emission. Empty
/// input yields empty output. Canonical greedy digits and terminator preserve
/// existing valid codewords; UINT64_MAX is supported. O(N+B) word work and O(B)
/// output, with constant scratch; preflight is excluded from bit_visits.
/// No optional history or capture quota; calls share no mutable state.
/// @param values Borrowed positive uint64 values; zero is invalid.
/// @return Owned emitted bits and actual work.
/// @throws std::invalid_argument If any value is zero.
/// @throws std::length_error On checked count/container limits.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// const auto e=dm::algorithms::fibonacci_encode_work({1,2,10});
/// std::print("{} {}\n",e.bits.size(),e.work.values); // 11 3
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see FibonacciTrace, FibonacciWork, fibonacci_code_length()
FibonacciEncoded fibonacci_encode_work(const std::vector<std::uint64_t>& values);

/// @brief Emit positive full-range Fibonacci words and capture their actual spans.
/// Preflights every positive value and total output size before emission. Empty
/// input yields empty output. Canonical greedy digits and terminator preserve
/// existing valid codewords; UINT64_MAX is supported. O(N+B) word work and O(B)
/// output, with constant scratch; preflight is excluded from bit_visits.
/// Same-pass capture, no replay. Quota defaults to1m logical B+2N entries;
/// zero disables only quota. Encoding admits all entries before copies/emission;
/// decoding admits input B before copying and two entries per completed word
/// before appends. No partial trace is returned on errors; not a peak-RAM bound.
/// Owned copies are independent and retain no views into caller input.
/// @param values Borrowed positive uint64 values; zero is invalid.
/// @param max_entries Maximum retained logical entries, default1000000; zero disables quota.
/// @return Owned emitted bits, values, actual spans and work.
/// @throws std::invalid_argument If any value is zero.
/// @throws std::length_error On checked count/container limits or capture quota.
/// @throws std::bad_alloc If output or history allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fibonacci_encode_trace({1,2},9);
/// std::print("{} {}\n",t.words.size(),t.captured_entries); // 2 9
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see FibonacciTrace, FibonacciWork, fibonacci_code_length()
FibonacciTrace fibonacci_encode_trace(const std::vector<std::uint64_t>& values,std::size_t max_entries=1000000);

/// @brief Strictly parse every complete positive Fibonacci word into an owned result.
/// Whole-frame parser: empty input is valid, unfinished tails are rejected and
/// each payload fits at most92coding weights. Checks weight bounds and unsigned
/// value sums before indexing/addition; a terminator at the maximum width is
/// recognized before needing a nonexistent next weight. O(B) word work, O(N)
/// output and constant scratch. No partial result on malformed/overflow input.
/// No optional history or capture quota; calls share no mutable state.
/// @param bits Borrowed complete vector<bool> frame; trailing bits are not padding.
/// @return Owned complete values.
/// @throws std::invalid_argument If the final codeword is unfinished.
/// @throws std::overflow_error If a payload exceeds the uint64 weight range or value sum.
/// @throws std::length_error On checked count/container limits.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// const auto v=dm::algorithms::fibonacci_decode({true,true,false,true,true});
/// std::print("{} {}\n",v[0],v[1]); // 1 2
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see FibonacciTrace, FibonacciWork, fibonacci_code_length()
std::vector<std::uint64_t> fibonacci_decode(const std::vector<bool>& bits);

/// @brief Strictly parse every complete positive Fibonacci word with actual work counts.
/// Whole-frame parser: empty input is valid, unfinished tails are rejected and
/// each payload fits at most92coding weights. Checks weight bounds and unsigned
/// value sums before indexing/addition; a terminator at the maximum width is
/// recognized before needing a nonexistent next weight. O(B) word work, O(N)
/// output and constant scratch. No partial result on malformed/overflow input.
/// No optional history or capture quota; calls share no mutable state.
/// @param bits Borrowed complete vector<bool> frame; trailing bits are not padding.
/// @return Owned complete values and actual work.
/// @throws std::invalid_argument If the final codeword is unfinished.
/// @throws std::overflow_error If a payload exceeds the uint64 weight range or value sum.
/// @throws std::length_error On checked count/container limits.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// const auto d=dm::algorithms::fibonacci_decode_work({false,true,false,false,true,true});
/// std::print("{} {}\n",d.values[0],d.work.payload_bits); // 10 5
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see FibonacciTrace, FibonacciWork, fibonacci_code_length()
FibonacciDecoded fibonacci_decode_work(const std::vector<bool>& bits);

/// @brief Strictly parse every complete positive Fibonacci word and capture their actual spans.
/// Whole-frame parser: empty input is valid, unfinished tails are rejected and
/// each payload fits at most92coding weights. Checks weight bounds and unsigned
/// value sums before indexing/addition; a terminator at the maximum width is
/// recognized before needing a nonexistent next weight. O(B) word work, O(N)
/// output and constant scratch. No partial result on malformed/overflow input.
/// Same-pass capture, no replay. Quota defaults to1m logical B+2N entries;
/// zero disables only quota. Encoding admits all entries before copies/emission;
/// decoding admits input B before copying and two entries per completed word
/// before appends. No partial trace is returned on errors; not a peak-RAM bound.
/// Owned copies are independent and retain no views into caller input.
/// @param bits Borrowed complete vector<bool> frame; trailing bits are not padding.
/// @param max_entries Maximum retained logical entries, default1000000; zero disables quota.
/// @return Owned complete values, input bits, actual spans and work.
/// @throws std::invalid_argument If the final codeword is unfinished.
/// @throws std::overflow_error If a payload exceeds the uint64 weight range or value sum.
/// @throws std::length_error On checked count/container limits or capture quota.
/// @throws std::bad_alloc If output or history allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fibonacci_decode_trace({true,true,false,true,true},9);
/// std::print("{} {}\n",t.words[1].begin,t.captured_entries); // 2 9
/// @endcode
/// @par References
/// - A. Apostolico and A. S. Fraenkel, Robust Transmission of Unbounded Strings
///   Using Fibonacci Representations, Purdue CSD-TR-545 (1985), later IEEE IT33(2)
///   (1987),238-245, doi:10.1109/TIT.1987.1057284. https://docs.lib.purdue.edu/cstech/464/
/// @see FibonacciTrace, FibonacciWork, fibonacci_code_length()
FibonacciTrace fibonacci_decode_trace(const std::vector<bool>& bits,std::size_t max_entries=1000000);

} // namespace datamunge::algorithms
