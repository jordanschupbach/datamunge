#pragma once
/// @file range_coding.hpp
/// @brief Checked carry-free32-bit range coding and actual byte-renormalization histories.
#include <cstddef>
#include <cstdint>
#include <vector>
/// @brief Canonical static entropy coding and owned actual records.
namespace datamunge::algorithms {
/// @brief Input, byte-allocation and actual-history policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RangeCodingOptions value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RangeCodingOptions {
  std::size_t max_symbols{1000000}; ///< Requested/encoded symbol count.
  std::size_t max_bytes{1000000};   ///< Complete input/output bytes, including four-byte flush.
  std::size_t max_alphabet{65536};  ///< Complete model entry count.
  std::size_t max_trace_entries{
      1000000}; ///< Actual symbol records plus byte-renormalization records.
};
/// @brief One actual byte shift after selecting a renormalization branch.
/// @par Example
/// @code
/// const dm::algorithms::RangeByteStep value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RangeByteStep {
  std::uint32_t low_before{};     ///< Actual low register before correction/shift.
  std::uint32_t range_before{};   ///< Actual range before optional bottom correction.
  std::uint32_t adjusted_range{}; ///< Actual range used by the byte shift.
  bool          corrected{};      ///< Whether the real bottom-range correction branch ran.
  std::uint8_t  byte{};           ///< Actual emitted top byte or actual consumed input byte.
  std::uint32_t low_after{};      ///< Actual low register after shifting by eight bits.
  std::uint32_t range_after{};    ///< Actual range register after shifting by eight bits.
  std::uint32_t code_before{};    ///< Actual decoder code before shift; zero in encoding records.
  std::uint32_t code_after{}; ///< Actual decoder code after input byte; zero in encoding records.
};
/// @brief One actual symbol subdivision and its byte shifts.
/// @par Example
/// @code
/// const dm::algorithms::RangeSymbolStep value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RangeSymbolStep {
  int           symbol{};       ///< Actual encoded/decoded symbol id.
  std::uint32_t low_before{};   ///< Actual low register before subdivision.
  std::uint32_t range_before{}; ///< Actual range before division by model total.
  std::uint32_t unit{};         ///< Actual floor(range_before/total).
  std::uint32_t low_narrowed{}; ///< Actual cumulative-frequency low update before renormalization.
  std::uint32_t range_narrowed{};  ///< Actual symbol-frequency range before renormalization.
  std::uint32_t low_after{};       ///< Actual post-renormalization low register.
  std::uint32_t range_after{};     ///< Actual post-renormalization range register.
  std::uint32_t code_before{};     ///< Actual decoder code before subdivision, zero for encoding.
  std::uint32_t code_after{};      ///< Actual decoder code after all renormalization bytes.
  std::size_t   position_before{}; ///< Actual emitted/consumed byte position before the symbol.
  std::size_t   position_after{};  ///< Actual emitted/consumed position after this symbol's shifts.
  std::vector<RangeByteStep> shifts; ///< Actual optional byte-shift history for this symbol.
};
/// @brief Owned coding outcome and actual selected register work.
/// @par Example
/// @code
/// const dm::algorithms::RangeCodingResult value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RangeCodingResult {
  std::vector<int> symbols; ///< Owned input symbols in encoding, actual output symbols in decoding.
  std::vector<std::uint8_t> bytes; ///< Actual encoded stream or owned complete decoder input.
  std::vector<std::uint32_t>
      cumulative;                     ///< Actual checked model prefix, starting0 and ending total.
  std::vector<RangeSymbolStep> steps; ///< Actual optional symbol/byte histories.
  bool                         decoding{}; ///< Whether the producer decoded rather than encoded.
  std::size_t
      renormalizations{}; ///< Actual byte shifts, excluding final encoder flush and decoder initialization.
  std::size_t
      stream_position{}; ///< Actual produced bytes or consumed decoder bytes, including framing registers.
  std::uint32_t final_low{};   ///< Actual interval low after the last symbol, before encoder flush.
  std::uint32_t final_range{}; ///< Actual interval range after the last symbol.
  std::uint32_t final_code{};  ///< Actual final decoder code, zero for encoding.
};
/// @brief Encode a static symbol sequence with the checked canonical carry-free range producer.
/// Model frequencies may contain zeros, but their positive total must be at most65536 and every encoded symbol must have positive frequency. Validates all symbols and the complete model before coding. Preserves historical unsigned32-bit wrap, byte normalization and final four top-byte flush on valid inputs. Empty message still produces four bytes. O(model+symbols+actual shifts) time, owned input/output/model storage; no shared mutable state or entropy guarantee on an arbitrary source. This bare stream carries no model, message length, checksum or authentication.
/// @param symbols Complete borrowed integer symbol ids.
/// @param frequencies Static nonnegative counts with total1..65536.
/// @param options Input, output-byte and actual-history policies.
/// @return Owned actual encoded stream, original symbols, model prefix and selected register work.
/// @throws std::invalid_argument Empty/zero/oversized-total model or invalid/zero-frequency symbol.
/// @throws std::length_error Input/alphabet/output/history capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::range_encode_run({0,1},{1,1});
/// std::print("{}\n",r.bytes.size()); // 4
/// @endcode
/// @see range_encode_trace(), range_decode_run()
RangeCodingResult range_encode_run(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& frequencies,
    const RangeCodingOptions&         options = {});
/// @brief Capture actual encoder subdivisions and byte shifts from the same native producer.
/// Each symbol and each renormalization shift costs one retained record, admitted before its state update. The final four flush bytes are actual bytes in the result, outside symbol shift records. Other semantics and complexity match range_encode_run(), plus owned actual histories.
/// @param symbols Complete valid symbol ids.
/// @param frequencies Complete checked static model.
/// @param options Input/output and retained-record policies.
/// @return Actual symbol intervals, correction decisions, byte shifts and encoded stream.
/// @throws std::invalid_argument Invalid model or symbol.
/// @throws std::length_error Any execution/history capacity exceeded.
/// @par Example
/// @code
/// auto t=dm::algorithms::range_encode_trace({0,1},{1,1});
/// std::print("{}\n",t.steps.size()); // 2
/// @endcode
/// @see range_encode_run()
RangeCodingResult range_encode_trace(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& frequencies,
    const RangeCodingOptions&         options = {});
/// @brief Decode an explicitly sized message with the checked canonical carry-free range producer.
/// Requires the same model and externally supplied symbol count. Reads four actual initialization bytes even for count0, rejects truncation instead of silently padding zeros, and rejects a quotient outside the model instead of clamping it. Additional trailing bytes are permitted and not authenticated; stream_position reports exactly what was consumed. Well-formed corruption can decode a different valid message, so this is not an integrity check. Complexity O(model+count*log(alphabet)+actual shifts), owned output/input/model storage.
/// @param bytes Complete borrowed byte stream with at least four bytes.
/// @param frequencies Static model with total1..65536.
/// @param count Requested decoded symbols, including zero.
/// @param options Complete input, output-symbol and actual-history policies.
/// @return Actual decoded symbols, model prefix, consumed position and selected register work.
/// @throws std::invalid_argument Invalid complete frequency model.
/// @throws std::runtime_error Truncated stream or invalid coding quotient.
/// @throws std::length_error Input/output/alphabet/history capacity exceeded.
/// @par Example
/// @code
/// auto bytes=dm::algorithms::range_encode({0,1},{1,1});
/// auto r=dm::algorithms::range_decode_run(bytes,{1,1},2);
/// std::print("{}\n",r.symbols); // [0, 1]
/// @endcode
/// @see range_decode_trace(), range_encode_run()
RangeCodingResult range_decode_run(
    const std::vector<std::uint8_t>&  bytes,
    const std::vector<std::uint32_t>& frequencies,
    std::size_t                       count,
    const RangeCodingOptions&         options = {});
/// @brief Capture actual decoder subdivisions, code registers and consumed bytes.
/// Same single decoder as range_decode_run(); one retained record per actual symbol/byte shift, admitted before its update. Initialization bytes are reflected in the first code/position and result stream_position, not invented as symbol shifts.
/// @param bytes Complete input stream.
/// @param frequencies Checked static model.
/// @param count Requested actual symbols.
/// @param options Input/output and actual-history policies.
/// @return Actual decoded symbols, code/interval histories and byte consumption.
/// @throws std::invalid_argument Invalid model.
/// @throws std::runtime_error Truncation or invalid quotient.
/// @throws std::length_error Any execution/history capacity exceeded.
/// @par Example
/// @code
/// auto t=dm::algorithms::range_decode_trace(dm::algorithms::range_encode({0},{1}),{1},1);
/// std::print("{}\n",t.steps[0].symbol); // 0
/// @endcode
/// @see range_decode_run()
RangeCodingResult range_decode_trace(
    const std::vector<std::uint8_t>&  bytes,
    const std::vector<std::uint32_t>& frequencies,
    std::size_t                       count,
    const RangeCodingOptions&         options = {});
/// @brief Return encoded bytes through the preserved signature and shared checked producer.
/// Uses the checked default policies and preserves valid model byte output. Bare output does not include model/count metadata.
/// @param symbols Complete valid symbol ids.
/// @param frequencies Static model with total1..65536.
/// @return Actual owned encoded bytes including four-byte flush.
/// @throws std::invalid_argument Invalid symbol/model.
/// @throws std::length_error Default policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::range_encode({},{1}).size()); // 4
/// @endcode
/// @see range_encode_run()
std::vector<std::uint8_t> range_encode(
    const std::vector<int>&           symbols,
    const std::vector<std::uint32_t>& frequencies);
/// @brief Return decoded symbols through the preserved externally sized signature.
/// Negative count now rejects; truncated streams no longer silently pad zeros. Uses the same native checked decoder and default policies.
/// @param bytes Complete input bytes, at least four.
/// @param frequencies Matching checked static model.
/// @param count Nonnegative externally supplied symbol count.
/// @return Actual decoded symbols.
/// @throws std::invalid_argument Negative count or invalid model.
/// @throws std::runtime_error Truncation or invalid quotient.
/// @throws std::length_error Default policy exceeded.
/// @par Example
/// @code
/// auto b=dm::algorithms::range_encode({0,0},{1});
/// std::print("{}\n",dm::algorithms::range_decode(b,{1},2)); // [0, 0]
/// @endcode
/// @see range_decode_run()
std::vector<int> range_decode(
    const std::vector<std::uint8_t>&  bytes,
    const std::vector<std::uint32_t>& frequencies,
    int                               count);
/// @brief Generate the original ordered frequency-pattern message as native reusable input.
/// Repeats symbol0 frequency[0] times, symbol1 frequency[1] times, and so on, then repeats that complete pattern. Uses the same checked static-model domain; it is deterministic, not iid sampled data. O(total*repetitions) time and owned storage. Checks the complete requested size before allocation.
/// @param frequencies Checked static model, total1..65536.
/// @param repetitions Number of complete ordered patterns.
/// @param max_symbols Allocation policy, default1000000; zero disables.
/// @return Owned original ordered symbol pattern.
/// @throws std::invalid_argument Invalid model.
/// @throws std::length_error Complete requested size or policy exceeded.
/// @par Example
/// @code
/// auto x=dm::algorithms::range_frequency_pattern({2,1},1);
/// std::print("{}\n",x); // [0, 0, 1]
/// @endcode
/// @see range_encode_run()
std::vector<int> range_frequency_pattern(
    const std::vector<std::uint32_t>& frequencies,
    std::size_t                       repetitions,
    std::size_t                       max_symbols = 1000000);
} // namespace datamunge::algorithms
