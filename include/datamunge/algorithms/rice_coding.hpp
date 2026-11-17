#pragma once
/// @file rice_coding.hpp
/// @brief Rice power-of-two models using canonical actual Golomb emission and parse histories.
#include <cstddef>
#include <datamunge/algorithms/golomb_coding.hpp>
#include <vector>
/// @brief Canonical integer compression and exact descriptive length studies.
namespace datamunge::algorithms {
/// @brief Capture actual Rice encoding codeword spans from the canonical Golomb producer.
/// Rice is the power-of-two Golomb case: quotient unary ones plus zero, followed by exactly k raw remainder bits. Validates k even for empty input and delegates once to the checked native Golomb producer; no replay, decoded prefix excludes unread trailing bits. Encoding checks complete mathematical bit size before allocation. Decoding bounds the complete input, requires exactly count words, and rejects missing complete words even at an input boundary. Time O(actual bit visits+words), owned actual bits/codewords; all fields and overflow semantics follow GolombTrace. The model and count remain external metadata.
/// @param values Complete borrowed unsigned values.
/// @param k Remainder bit count0..30, giving M=2^k.
/// @param max_bits Complete emitted bit policy, default1000000; zero disables.
/// @return Owned actual codeword values, quotient/remainder spans, bits and selected bit visits.
/// @throws std::invalid_argument Invalid k; decoder also rejects negative count, missing requested words or a truncated attempted word.
/// @throws std::length_error Bit/count/container capacity or policy exceeded.
/// @throws std::overflow_error Unrepresentable decoded integer; decoder only.
/// @par Example
/// @code
/// auto r=dm::algorithms::rice_encode_trace({8},2);
/// std::print("{} {}\n",r.words[0].quotient,r.words[0].remainder); // 2 0
/// @endcode
/// @par References
/// - Rice (1979), Some Practical Universal Noiseless Coding Techniques, JPL79-22, https://ntrs.nasa.gov/citations/19790014634; Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rice_encode(), rice_decode(), GolombTrace
GolombTrace
rice_encode_trace(const std::vector<std::uint64_t>& values, int k, std::size_t max_bits = 1000000);
/// @brief Capture actual Rice decoding codeword spans from the canonical Golomb producer.
/// Rice is the power-of-two Golomb case: quotient unary ones plus zero, followed by exactly k raw remainder bits. Validates k even for empty input and delegates once to the checked native Golomb producer; no replay, decoded prefix excludes unread trailing bits. Encoding checks complete mathematical bit size before allocation. Decoding bounds the complete input, requires exactly count words, and rejects missing complete words even at an input boundary. Time O(actual bit visits+words), owned actual bits/codewords; all fields and overflow semantics follow GolombTrace. The model and count remain external metadata.
/// @param bits Complete borrowed raw bits.
/// @param k Remainder bit count0..30, giving M=2^k.
/// @param count Nonnegative externally supplied message length.
/// @param max_bits Complete input bit policy, default1000000; zero disables.
/// @return Owned actual codeword values, quotient/remainder spans, bits and selected bit visits.
/// @throws std::invalid_argument Invalid k; decoder also rejects negative count, missing requested words or a truncated attempted word.
/// @throws std::length_error Bit/count/container capacity or policy exceeded.
/// @throws std::overflow_error Unrepresentable decoded integer; decoder only.
/// @par Example
/// @code
/// auto r=dm::algorithms::rice_decode_trace(dm::algorithms::rice_encode({8},2),2,1);
/// std::print("{} {}\n",r.words[0].quotient,r.words[0].remainder); // 2 0
/// @endcode
/// @par References
/// - Rice (1979), Some Practical Universal Noiseless Coding Techniques, JPL79-22, https://ntrs.nasa.gov/citations/19790014634; Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rice_encode(), rice_decode(), GolombTrace
GolombTrace
rice_decode_trace(const std::vector<bool>& bits, int k, int count, std::size_t max_bits = 1000000);
/// @brief One exact conditional expected length of actual Rice codewords.
/// @par Example
/// @code
/// const dm::algorithms::RiceLengthRow value{};
/// std::print("{}\n",value.k); // 0
/// @endcode
struct RiceLengthRow {
  int         k{};             ///< Actual remainder bit count.
  int         parameter{};     ///< Actual power-of-two Golomb model parameter.
  double      mean_bits{};     ///< Probability-weighted actual emitted word length.
  double      normalization{}; ///< Sum of the original finite support weights.
  std::size_t emitted_bits{};  ///< Actual complete encoded stream bits visited to form all lengths.
};
/// @brief Reproduce the original finite geometric-weight Rice length study through real native emission.
/// For n=0..largest, use weights2^(-n/half_life), normalize over that finite support, and average each actual emitted codeword span. This is a conditional truncated model expectation, not the full geometric expectation, observed compression of random data, or a bitstream byte serialization. For k=0..max_k each canonical encoding executes once. Time/storage follow actual emitted bits, bounded independently for every model.
/// @param max_k Largest remainder width, default6, valid0..30.
/// @param largest Largest support integer, default299, bounded to1000000.
/// @param half_life Finite positive geometric half-life, default8.
/// @param max_bits Per-model actual encoded bit bound, default1000000; zero disables.
/// @return Owned ascending-k rows of actual weighted lengths and work.
/// @throws std::invalid_argument Invalid k/half-life.
/// @throws std::length_error Support or actual bit/container policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::rice_length_study(0,0);
/// std::print("{}\n",r[0].mean_bits); // 1
/// @endcode
/// @par References
/// - Rice (1979), Some Practical Universal Noiseless Coding Techniques, JPL79-22, https://ntrs.nasa.gov/citations/19790014634; Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rice_encode_trace()
std::vector<RiceLengthRow> rice_length_study(
    int         max_k     = 6,
    std::size_t largest   = 299,
    double      half_life = 8,
    std::size_t max_bits  = 1000000);
} // namespace datamunge::algorithms
