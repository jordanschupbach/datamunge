#pragma once
/// @file levenshtein_coding.hpp
/// @brief Full unsigned64 recursive Levenshtein integer coding with checked prefix parsing and actual word/block spans.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief One actual emitted/read binary block after its implicit leading1.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_encode_trace({10});
/// std::print("{}\n",t.words[0].blocks.back().value); // 10
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see levenshtein_encode_trace(), levenshtein_decode_trace()
struct LevenshteinCodeBlock {
  std::size_t   begin{0}; ///< Actual beginning bit offset in complete owned stream.
  std::size_t   width{0}; ///< Actual number of explicit low bits, in[0,63].
  std::uint64_t value{0}; ///< Actual reconstructed block value including implicit leading1.
};
/// @brief Actual value, stream span, unary count and producing binary blocks.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_encode_trace({10});
/// std::print("{} {}\n",t.words[0].unary_ones,t.words[0].end); // 3 8
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see levenshtein_encode_trace(), levenshtein_decode_trace()
struct LevenshteinCodeword {
  std::uint64_t value{0};    ///< Actual original encoded or parsed value.
  std::size_t   begin{0};    ///< Actual inclusive original word beginning bit offset.
  std::size_t   end{0};      ///< Actual exclusive original word end bit offset.
  std::size_t unary_ones{0}; ///< Actual leading one count before delimiter;0 is the zero codeword.
  std::vector<LevenshteinCodeBlock>
      blocks; ///< Actual emission/parsing order; leading width0 implicit1 record included for positive values.
};
/// @brief Actual selected complete values and written/read bits, excluding validation/copies/allocation.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_encode_trace({10});
/// std::print("{}\n",t.work.bit_visits); // 8
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see levenshtein_encode_trace(), levenshtein_decode_trace()
struct LevenshteinCodingWork {
  std::size_t values{0};       ///< Actual complete values emitted or parsed.
  std::size_t unary_bits{0};   ///< Actual count-prefix ones plus each zero delimiter written/read.
  std::size_t payload_bits{0}; ///< Actual explicit binary block bits written/read.
  std::size_t bit_visits{0};   ///< Actual sum of unary_bits and payload_bits; unread tail excluded.
};
/// @brief Owned full stream, actual word/block evidence and prefix progress from one execution.
/// Complete trace or exception, never a silent partial parse. Capture cap counts
/// full owned stream bits, values, word records and actual block records. Fields,
/// capacities/output/runtime copies and bytes/RAM excluded; default million,0 off.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_encode_trace({0,1});
/// std::print("{}\n",t.bits.size()); // 3
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see levenshtein_encode_trace(), levenshtein_decode_trace()
struct LevenshteinCodingTrace {
  std::vector<bool>
      bits; ///< Owned emitted stream or complete original decoding input, including unread tail.
  std::vector<std::uint64_t>
      values; ///< Owned original encoded values or actual decoded requested prefix.
  std::vector<LevenshteinCodeword> words; ///< Actual chronological producing word/block spans.
  LevenshteinCodingWork            work;  ///< Actual selected written/read bit and value counts.
  std::size_t consumed_bits{0}; ///< Actual emitted size or parsed prefix end; tail is not consumed.
  bool        decoding{false};  ///< True only for decoder capture.
  std::size_t captured_entries{0}; ///< Actual stream-bit/value/word/block logical records.
};
/// @brief Append one full unsigned64 recursive codeword with strong exception safety.
/// Zero emits0. Positive n emits C ones then0, followed by reversed low-bit
/// chunks obtained by replacing n with floor(log2(n)) until1. Leading1 bits
/// are implicit. Complete word is prepared/admitted before one insertion;
/// no partial destination mutation on validation/allocation failure.
/// @param n Original nonnegative integer, full uint64 range.
/// @param out Mutable destination bit vector; original prefix retained.
/// @throws std::length_error On complete destination geometry.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::vector<bool> bits;
/// dm::algorithms::levenshtein_encode_value(0,bits);
/// std::print("{}\n",bits.size()); // 1
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see LevenshteinCodingTrace
void levenshtein_encode_value(std::uint64_t n, std::vector<bool>& out);
/// @brief Encode a complete unsigned64 sequence as concatenated prefix codewords.
/// No input mutation or shared state. Bit length grows like log2(n) plus
/// iterated-log overhead, not merely the iterated logarithm. No optional capture.
/// @param values Borrowed original integer values; empty valid.
/// @return Owned specialized vector<bool> stream; no byte packing/endianness format.
/// @throws std::length_error On complete bit-container geometry.
/// @throws std::bad_alloc On output allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::levenshtein_encode({0,1}).size()); // 3
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see LevenshteinCodingTrace
std::vector<bool> levenshtein_encode(const std::vector<std::uint64_t>& values);
/// @brief Parse exactly count complete unsigned64 words from the original bitstream prefix.
/// Check unary delimiter, every block extent and width<=63 before indexing or
/// shifting. Malformed/unrepresentable requested words throw, including missing
/// input. A requested prefix does not validate or consume any unread tail.
/// No input mutation, shared state, Unicode or byte framing.
/// @param bits Borrowed original complete bit stream; unread trailing bits allowed.
/// @param count Nonnegative requested complete word count;0 consumes nothing.
/// @return Owned exact requested values; no silent short prefix on truncation.
/// @throws std::invalid_argument On negative count, malformed/truncated/unrepresentable requested prefix.
/// @throws std::length_error On output-container geometry.
/// @throws std::bad_alloc On output allocation failure.
/// @par Example
/// @code
/// const auto v=dm::algorithms::levenshtein_decode(dm::algorithms::levenshtein_encode({0,10}),2);
/// std::print("{}\n",v[1]); // 10
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see LevenshteinCodingTrace
std::vector<std::uint64_t> levenshtein_decode(const std::vector<bool>& bits, int count);
/// @brief Capture actual recursive word/block emission and complete owned output once.
/// Shared emitting kernel, no re-encoding to reconstruct history. Actual cap
/// counts stream bits, original values, word records and block records.
/// @param values Borrowed complete original unsigned64 values.
/// @param max_entries Actual complete logical record cap, default1000000;0 disables only cap.
/// @return Owned original values, actual stream/spans/blocks/work.
/// @throws std::length_error On actual logical quota or complete output geometry.
/// @throws std::bad_alloc On capture/output allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_encode_trace({10});
/// std::print("{}\n",t.words[0].end-t.words[0].begin); // 8
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see LevenshteinCodingTrace
LevenshteinCodingTrace levenshtein_encode_trace(
    const std::vector<std::uint64_t>& values,
    std::size_t                       max_entries = 1000000);
/// @brief Capture actual checked requested-prefix parsing and original complete stream ownership.
/// Same parser as ordinary decode. Unread tail retained but not validated/read
/// or counted as bit visits. Complete result or exception; no silent prefix.
/// @param bits Borrowed full stream, including any intentionally unread tail.
/// @param count Nonnegative exact requested complete words.
/// @param max_entries Actual record cap including full input bits, default1000000;0 off.
/// @return Owned full original stream, actual decoded values/spans/blocks/work and consumed offset.
/// @throws std::invalid_argument On negative count or malformed/truncated/unrepresentable requested word.
/// @throws std::length_error On actual complete logical cap/container limits.
/// @throws std::bad_alloc On input/capture/output allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::levenshtein_decode_trace({false,true},1);
/// std::print("{} {}\n",t.consumed_bits,t.bits.size()); // 1 2
/// @endcode
/// @par References
/// - V. I. Levenshtein(1968), On the Redundancy and Delay of Decodable Coding of Natural Numbers, Problems of Cybernetics20,173–179, https://www.compression.ru/download/articles/int/levenstein_1968_on_the_redundancy_and_delay.pdf; recursive integer code convention explicitly documented here.
/// @see LevenshteinCodingTrace
LevenshteinCodingTrace levenshtein_decode_trace(
    const std::vector<bool>& bits,
    int                      count,
    std::size_t              max_entries = 1000000);
} // namespace datamunge::algorithms
