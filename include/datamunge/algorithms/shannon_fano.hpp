#pragma once
/// @file shannon_fano.hpp
/// @brief Checked native frequency partitions and complete static prefix-code payloads.
#include <array>
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/prefix_code_types.hpp>
#include <string>
#include <unordered_map>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Owned byte-to-binary-word model; nonempty words must form a prefix-free code.
/// @par Example
/// @code
/// const dm::algorithms::ShannonFanoCode value{};
/// std::print("{}\n",value.codes.size()); // 0
/// @endcode
struct ShannonFanoCode {
  std::unordered_map<char, std::string>
      codes; ///< Owned byte identities and ASCII0/1 words; singleton uses0.
};
/// @brief Complete source/payload, actual split and retained-history bounds; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::ShannonFanoOptions value{};
/// std::print("{}\n",value.max_bytes); // 1000000
/// @endcode
struct ShannonFanoOptions {
  std::size_t max_bytes{1000000};              ///< Complete source or requested decoded byte count.
  std::size_t max_bits{10000000};              ///< Complete total codeword bits or payload bits.
  std::size_t max_nodes{1000000};              ///< Complete validated codec-trie nodes.
  std::size_t max_split_comparisons{20000000}; ///< Actual candidate partition boundaries inspected.
  std::size_t max_trace_entries{
      1000000}; ///< Actual sorted symbol/frequency scalars and split records, or emitted-word records.
};
/// @brief One actual chosen partition of the descending-frequency symbol array.
/// @par Example
/// @code
/// const dm::algorithms::ShannonFanoSplit value{};
/// std::print("{}\n",value.imbalance); // 0
/// @endcode
struct ShannonFanoSplit {
  std::size_t   begin{};     ///< Inclusive actual sorted range address.
  std::size_t   end{};       ///< Exclusive actual sorted range address.
  std::size_t   middle{};    ///< Actual first right-child symbol address.
  std::uint64_t left{};      ///< Actual left group frequency sum.
  std::uint64_t right{};     ///< Actual right group frequency sum.
  std::uint64_t imbalance{}; ///< Actual absolute left-right difference.
};
/// @brief Owned actual byte code, complete frequency summary and observed construction work.
/// @par Example
/// @code
/// const dm::algorithms::ShannonFanoRun value{};
/// std::print("{}\n",value.weighted_bits); // 0
/// @endcode
struct ShannonFanoRun {
  ShannonFanoCode              code;          ///< Actual final prefix code.
  std::array<std::size_t, 256> frequencies{}; ///< Actual unsigned-byte input counts.
  std::size_t weighted_bits{}; ///< Exact checked sum frequency times actual codeword length.
  double      entropy{};       ///< Empirical base2 entropy per byte, zero for empty input.
  double      average_bits{};  ///< Actual weighted_bits/source_bytes, zero for empty input.
  std::size_t
      split_comparisons{}; ///< Actual candidate boundaries compared with the current best imbalance.
  std::size_t code_bit_appends{}; ///< Actual word-bit appends, including singleton0.
  std::vector<unsigned>
      sorted_symbols; ///< Optional actual byte ids in descending frequency then historical char tie order.
  std::vector<std::size_t>
      sorted_frequencies;               ///< Optional actual counts aligned with sorted_symbols.
  std::vector<ShannonFanoSplit> splits; ///< Optional actual preorder partition choices.
};
/// @brief Owned actual static byte encoding or decoding with complete-word boundaries.
/// @par Example
/// @code
/// const dm::algorithms::ShannonFanoCodecRun value{};
/// std::print("{}\n",value.bit_visits); // 0
/// @endcode
struct ShannonFanoCodecRun {
  std::string bits;         ///< Actual encoded ASCII0/1 payload, encoding only.
  std::string text;         ///< Actual raw decoded bytes, decoding only.
  std::size_t bit_visits{}; ///< Actual copied/traversed payload bits.
  std::size_t trie_nodes{}; ///< Actual validated trie nodes.
  std::vector<PrefixWordSpan>
      words; ///< Optional actual emitted words; symbol is unsigned byte0..255.
};
/// @brief Build a checked native top-down byte prefix code.
/// Counts complete raw bytes, sorts descending frequency with historical char-order ties, then chooses the earliest boundary minimizing absolute left-right weight difference. Appends0 to the left and1 to the right and recursively splits both actual groups. Uses overflow-free left/right differences; avoids the historical signed-cast/doubled-sum overflow. Empty input gives an empty model and zero summaries; one distinct byte uses word0. Prefix-free but not generally optimal; compare measured average to Huffman separately. At most256 symbols and depth255; construction O(bytes+alphabet^2), owned code storage O(alphabet^2). Actual candidate comparisons and bit appends are observed, not predicted. Optional history retains sorted symbol/count scalars and one actual record per nonleaf split. Input borrowed unchanged; output owns storage; independent calls share no state.
/// @param text Borrowed complete raw bytes, including NUL/high bytes.
/// @param options Complete source/codeword bounds, actual comparisons and optional history limits.
/// @return Owned actual model/frequencies, entropy, exact weighted length, observed work and optional splits.
/// @throws std::length_error Complete source, generated codeword, weighted-length, actual split/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shannon_fano_build_run("aab");
/// std::print("{} {}\n",r.weighted_bits,r.average_bits); // 3 1
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_encode_run(), huffman_build()
ShannonFanoRun shannon_fano_build_run(
    const std::string&        text,
    const ShannonFanoOptions& options = {});
/// @brief Build a checked native top-down byte prefix code retaining actual partition choices.
/// Counts complete raw bytes, sorts descending frequency with historical char-order ties, then chooses the earliest boundary minimizing absolute left-right weight difference. Appends0 to the left and1 to the right and recursively splits both actual groups. Uses overflow-free left/right differences; avoids the historical signed-cast/doubled-sum overflow. Empty input gives an empty model and zero summaries; one distinct byte uses word0. Prefix-free but not generally optimal; compare measured average to Huffman separately. At most256 symbols and depth255; construction O(bytes+alphabet^2), owned code storage O(alphabet^2). Actual candidate comparisons and bit appends are observed, not predicted. Optional history retains sorted symbol/count scalars and one actual record per nonleaf split. Input borrowed unchanged; output owns storage; independent calls share no state.
/// @param text Borrowed complete raw bytes, including NUL/high bytes.
/// @param options Complete source/codeword bounds, actual comparisons and optional history limits.
/// @return Owned actual model/frequencies, entropy, exact weighted length, observed work and optional splits.
/// @throws std::length_error Complete source, generated codeword, weighted-length, actual split/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shannon_fano_build_trace("aab");
/// std::print("{} {}\n",r.weighted_bits,r.average_bits); // 3 1
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_encode_run(), huffman_build()
ShannonFanoRun shannon_fano_build_trace(
    const std::string&        text,
    const ShannonFanoOptions& options = {});
/// @brief Encode every actual raw byte prefix word.
/// Validates the complete supplied model through the same private prefix trie shared by Shannon-Fano-Elias; duplicate/prefix/nonbinary/empty words reject. Encoder validates complete source membership and payload length before output. Decoder traverses ALL supplied ASCII0/1 bits and requires exactly count complete words; absent branches, truncation, extra words and wrong counts reject. Empty model supports empty input/count0. Model/source borrowed unchanged; output owns storage and actual optional per-word spans. O(model bits+payload bits) time and trie/output storage plus optional emitted records; no separate codec replay.
/// @param code Borrowed prefix-free nonempty binary words for active raw bytes.
/// @param text Borrowed complete raw source bytes.
/// @param options Complete source/payload/trie/history bounds.
/// @return Owned actual encoded bits or raw decoded bytes, observed bit visits/trie nodes and optional emitted spans.
/// @throws std::invalid_argument Malformed model, absent source byte, nonbinary/truncated/extra/wrong-count payload.
/// @throws std::length_error Complete source/model/payload/trie/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto c=dm::algorithms::shannon_fano_build("aab");
/// auto r=dm::algorithms::shannon_fano_encode_run(c,"aab");
/// std::print("{}\n",r.bits.size()); // 3
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_build_run()
ShannonFanoCodecRun shannon_fano_encode_run(
    const ShannonFanoCode&    code,
    const std::string&        text,
    const ShannonFanoOptions& options = {});
/// @brief Encode every actual raw byte prefix word with actual emitted boundaries.
/// Validates the complete supplied model through the same private prefix trie shared by Shannon-Fano-Elias; duplicate/prefix/nonbinary/empty words reject. Encoder validates complete source membership and payload length before output. Decoder traverses ALL supplied ASCII0/1 bits and requires exactly count complete words; absent branches, truncation, extra words and wrong counts reject. Empty model supports empty input/count0. Model/source borrowed unchanged; output owns storage and actual optional per-word spans. O(model bits+payload bits) time and trie/output storage plus optional emitted records; no separate codec replay.
/// @param code Borrowed prefix-free nonempty binary words for active raw bytes.
/// @param text Borrowed complete raw source bytes.
/// @param options Complete source/payload/trie/history bounds.
/// @return Owned actual encoded bits or raw decoded bytes, observed bit visits/trie nodes and optional emitted spans.
/// @throws std::invalid_argument Malformed model, absent source byte, nonbinary/truncated/extra/wrong-count payload.
/// @throws std::length_error Complete source/model/payload/trie/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto c=dm::algorithms::shannon_fano_build("aab");
/// auto r=dm::algorithms::shannon_fano_encode_trace(c,"aab");
/// std::print("{}\n",r.bits.size()); // 3
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_build_run()
ShannonFanoCodecRun shannon_fano_encode_trace(
    const ShannonFanoCode&    code,
    const std::string&        text,
    const ShannonFanoOptions& options = {});
/// @brief Decode every actual byte prefix word.
/// Validates the complete supplied model through the same private prefix trie shared by Shannon-Fano-Elias; duplicate/prefix/nonbinary/empty words reject. Encoder validates complete source membership and payload length before output. Decoder traverses ALL supplied ASCII0/1 bits and requires exactly count complete words; absent branches, truncation, extra words and wrong counts reject. Empty model supports empty input/count0. Model/source borrowed unchanged; output owns storage and actual optional per-word spans. O(model bits+payload bits) time and trie/output storage plus optional emitted records; no separate codec replay.
/// @param code Borrowed prefix-free nonempty binary words for active raw bytes.
/// @param bits Borrowed complete ASCII binary payload, with no ignored tail.
/// @param count Required exact decoded byte count, including zero.
/// @param options Complete source/payload/trie/history bounds.
/// @return Owned actual encoded bits or raw decoded bytes, observed bit visits/trie nodes and optional emitted spans.
/// @throws std::invalid_argument Malformed model, absent source byte, nonbinary/truncated/extra/wrong-count payload.
/// @throws std::length_error Complete source/model/payload/trie/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto c=dm::algorithms::shannon_fano_build("aab");
/// auto r=dm::algorithms::shannon_fano_decode_run(c,dm::algorithms::shannon_fano_encode(c,"aab"),3);
/// std::print("{}\n",r.text); // aab
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_build_run()
ShannonFanoCodecRun shannon_fano_decode_run(
    const ShannonFanoCode&    code,
    const std::string&        bits,
    std::size_t               count,
    const ShannonFanoOptions& options = {});
/// @brief Decode every actual byte prefix word with actual emitted boundaries.
/// Validates the complete supplied model through the same private prefix trie shared by Shannon-Fano-Elias; duplicate/prefix/nonbinary/empty words reject. Encoder validates complete source membership and payload length before output. Decoder traverses ALL supplied ASCII0/1 bits and requires exactly count complete words; absent branches, truncation, extra words and wrong counts reject. Empty model supports empty input/count0. Model/source borrowed unchanged; output owns storage and actual optional per-word spans. O(model bits+payload bits) time and trie/output storage plus optional emitted records; no separate codec replay.
/// @param code Borrowed prefix-free nonempty binary words for active raw bytes.
/// @param bits Borrowed complete ASCII binary payload, with no ignored tail.
/// @param count Required exact decoded byte count, including zero.
/// @param options Complete source/payload/trie/history bounds.
/// @return Owned actual encoded bits or raw decoded bytes, observed bit visits/trie nodes and optional emitted spans.
/// @throws std::invalid_argument Malformed model, absent source byte, nonbinary/truncated/extra/wrong-count payload.
/// @throws std::length_error Complete source/model/payload/trie/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto c=dm::algorithms::shannon_fano_build("aab");
/// auto r=dm::algorithms::shannon_fano_decode_trace(c,dm::algorithms::shannon_fano_encode(c,"aab"),3);
/// std::print("{}\n",r.text); // aab
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_build_run()
ShannonFanoCodecRun shannon_fano_decode_trace(
    const ShannonFanoCode&    code,
    const std::string&        bits,
    std::size_t               count,
    const ShannonFanoOptions& options = {});
/// @brief Build the checked code through the preserved signature.
/// Same native producer/default bounds without optional history.
/// @param text Complete raw bytes.
/// @return Owned actual byte model.
/// @throws std::length_error Default source/work/history capacity/policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shannon_fano_build("aab").codes.size()); // 2
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_build_run()
ShannonFanoCode shannon_fano_build(const std::string& text);
/// @brief Encode the complete byte source through the preserved signature.
/// Same shared checked prefix producer/default bounds.
/// @param code Borrowed valid byte codebook.
/// @param text Complete raw bytes present in the model.
/// @return Owned actual ASCII binary payload.
/// @throws std::invalid_argument Malformed code or absent source byte.
/// @throws std::length_error Default complete source/payload/trie capacity/policy exceeded.
/// @par Example
/// @code
/// auto c=dm::algorithms::shannon_fano_build("aab");
/// std::print("{}\n",dm::algorithms::shannon_fano_encode(c,"aab").size()); // 3
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_encode_run()
std::string shannon_fano_encode(const ShannonFanoCode& code, const std::string& text);
/// @brief Compute complete logical codeword bits per source byte without constructing a payload.
/// Validates the entire model and source membership, then accumulates exact checked lengths; empty text returns zero after model validation. Same default source/bit/trie limits as the codec. Borrowed inputs unchanged, no shared state.
/// @param code Borrowed complete prefix-free byte model.
/// @param text Complete raw source.
/// @return Actual mean codeword length per source byte, zero for empty text.
/// @throws std::invalid_argument Malformed model or absent source byte.
/// @throws std::length_error Default source/bit/trie or sum capacity/policy exceeded.
/// @par Example
/// @code
/// auto c=dm::algorithms::shannon_fano_build("aab");
/// std::print("{}\n",dm::algorithms::shannon_fano_average_bits(c,"aab")); // 1
/// @endcode
/// @par References
/// - Shannon (1948), A Mathematical Theory of Communication, Bell System Technical Journal27; Fano (1949), The Transmission of Information, MIT Research Laboratory of Electronics Technical Report65.
/// @see shannon_fano_encode_run()
double shannon_fano_average_bits(const ShannonFanoCode& code, const std::string& text);
/// @brief Return the exact original pangram comparison fixture.
/// @return Owned original 85-byte sentence pair, including punctuation and spaces.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shannon_fano_example_text().empty()); // false
/// @endcode
/// @see shannon_fano_build_run()
std::string shannon_fano_example_text();
/// @brief Construct controlled raw-byte frequencies in increasing unsigned-byte identity order.
/// @param counts Complete frequencies for ids0..counts.size()-1, at most256 ids; zeros allowed.
/// @param max_bytes Complete total-byte cap, default1000000; zero disables.
/// @return Owned raw bytes with exactly the requested counts.
/// @throws std::invalid_argument More than256 byte ids.
/// @throws std::length_error Complete total exceeds capacity or policy.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shannon_fano_frequency_text({2,3}).size()); // 5
/// @endcode
/// @see shannon_fano_build_run()
std::string shannon_fano_frequency_text(
    const std::vector<std::size_t>& counts,
    std::size_t                     max_bytes = 1000000);
} // namespace datamunge::algorithms
