#pragma once
/// @file lz77.hpp
/// @brief Checked greedy overlapping-window byte LZ77 with actual search/emission/copy evidence.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief One native tuple token: overlapping backward copy followed by an optional literal byte.
/// @par Example
/// @code
/// auto tokens=dm::algorithms::lz77_encode("aaaaaa");
/// std::print("{}\n",tokens.back().length); // 5
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77Token {
  int offset; ///< Distance back from token start; zero only for length0 pure literals.
  int length; ///< Nonnegative number of copied bytes; positive requires an available positive offset.
  char next; ///< Literal byte, including NUL; ignored when literal is false.
  bool literal{
      true}; ///< Whether to append next after copying; encoder uses false only for a final match.
};
/// @brief Actual selected greedy encoder work, excluding guards, copies and allocation.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_encode_trace("aaaaaa");
/// std::print("{}\n",t.analysis.work.emissions); // 2
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77EncodeWork {
  std::size_t candidates{
      0}; ///< Actual prior-window candidate starts visited, even for a zero match cap.
  std::size_t comparisons{
      0}; ///< Actual byte comparisons, including mismatching termination comparisons.
  std::size_t best_updates{0}; ///< Actual strict best-length improvements within token searches.
  std::size_t emissions{0};    ///< Actual emitted tokens.
};
/// @brief Owned actual greedy tokens and producing search counts.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_encode_trace("aaaaaa");
/// std::print("{}\n",t.analysis.tokens.size()); // 2
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77EncodeAnalysis {
  std::vector<LZ77Token> tokens; ///< Actual original-order greedy tuples.
  LZ77EncodeWork         work;   ///< Actual selected same-run work.
};
/// @brief One actual byte comparison made while testing a prior-window match start.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_encode_trace("aaaaaa");
/// std::print("{}\n",t.comparisons.front().equal); // true
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77Comparison {
  std::size_t input;     ///< Actual token-start input index.
  std::size_t candidate; ///< Actual earlier candidate start.
  std::size_t length;    ///< Actual matched-prefix length before this comparison.
  unsigned    left;      ///< Actual candidate byte interpreted unsigned0..255.
  unsigned    right;     ///< Actual next-input byte interpreted unsigned0..255.
  bool        equal;     ///< Actual comparison result.
};
/// @brief One actual completed candidate evaluation, preserving strict longest/oldest-start ties.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_encode_trace("aaaaaa");
/// std::print("{}\n",t.candidates.front().matched); // 5
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77Candidate {
  std::size_t input;     ///< Actual token-start input index.
  std::size_t candidate; ///< Actual earlier candidate start, increasing within the window.
  std::size_t matched;   ///< Actual matching byte length, possibly overlapping token start.
  bool
      new_best; ///< Actual strict longest-length update; ties keep the oldest candidate/largest offset.
};
/// @brief One real greedy token emission and actual input advancement.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_encode_trace("aaaaaa");
/// std::print("{}\n",t.emissions.front().next); // 1
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77Emission {
  std::size_t input; ///< Actual original input token start.
  std::size_t next;  ///< Actual one-past consumed input position.
  std::size_t token; ///< Actual token index in the returned output.
};
/// @brief Owned actual encoder input, candidate comparisons, strict updates and token emissions.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_encode_trace("aaaaaa");
/// std::print("{}\n",t.analysis.tokens.size()); // 2
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77EncodeTrace {
  std::string input; ///< Owned arbitrary original bytes, preserving embedded NULs.
  std::vector<LZ77Comparison>
      comparisons; ///< Actual evaluated byte comparisons, without rescanning for display.
  std::vector<LZ77Candidate>
      candidates; ///< Actual completed window candidates, including zero-length matches.
  std::vector<LZ77Emission> emissions; ///< Actual producing input advancement/token indices.
  LZ77EncodeAnalysis        analysis;  ///< Actual same-run greedy output/work.
  std::size_t               captured_entries{
      0}; ///< Input bytes plus each actual comparison/candidate/emission record.
};
/// @brief Actual decoded byte string and selected real token/copy work.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_decode_trace(dm::algorithms::lz77_encode("aaaaaa"));
/// std::print("{}\n",t.analysis.overlapping_reads); // 4
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77DecodeAnalysis {
  std::string data;        ///< Actual decoded arbitrary bytes, independently owned.
  std::size_t tokens{0};   ///< Actual processed input tokens.
  std::size_t copied{0};   ///< Actual backward-copy bytes emitted.
  std::size_t literals{0}; ///< Actual literal bytes emitted, including NULs.
  std::size_t overlapping_reads{
      0}; ///< Actual copy reads from output produced earlier within the same token.
};
/// @brief One actual producing decoder byte emission, from a literal or an available earlier output byte.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_decode_trace(dm::algorithms::lz77_encode("aaaaaa"));
/// std::print("{}\n",t.bytes.front().value); // 97
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77Byte {
  std::size_t token;    ///< Actual input token index.
  std::size_t position; ///< Actual next output byte index.
  std::size_t source;   ///< Actual backward-copy read index, SIZE_MAX for a literal.
  unsigned    value;    ///< Actual emitted unsigned byte0..255.
};
/// @brief Owned original token records, actual emitted-byte decisions and same-run decoded output.
/// @par Example
/// @code
/// auto t=dm::algorithms::lz77_decode_trace(dm::algorithms::lz77_encode("aaaaaa"));
/// std::print("{}\n",t.analysis.overlapping_reads); // 4
/// @endcode
/// @par References
/// - Ziv and Lempel(1977), A Universal Algorithm for Sequential Data Compression, https://doi.org/10.1109/TIT.1977.1055714; this native tuple stream specifies no packed wire format.
/// @see lz77_encode_trace(), lz77_decode_trace()
struct LZ77DecodeTrace {
  std::vector<LZ77Token>
      input; ///< Owned original tuples; invalid streams are rejected completely before capture.
  std::vector<LZ77Byte>
      bytes; ///< Actual chronological output-byte writes, without re-decoding prefixes.
  LZ77DecodeAnalysis analysis;     ///< Actual same-run decoded string and counts.
  std::size_t captured_entries{0}; ///< Original token records plus actual emitted-byte records.
};
/// @brief Greedily encode exact bytes, preserving oldest-start ties and overlap-capable input matching.
/// @param data Arbitrary byte string, including NUL/high bytes; at most INT_MAX bytes for native int tuple fields.
/// @param window Nonnegative maximum offset, default4096; zero gives literal-only tokens.
/// @param max_match Nonnegative copied-byte limit, default255; zero gives literal-only tokens.
/// @return Actual greedy tuple stream, empty for empty input.
/// @throws std::invalid_argument For negative window/match settings.
/// @throws std::length_error For unrepresentable native input/count geometry.
/// @note Naive candidate scans cost O(n*(1+window*(1+max_match))) as a conservative bound, with actual skipping often lower.
/// Token count is not a compressed byte size: this API defines no serialized bit packing, framing or entropy coding.
/// @code
/// std::print("{}\n",dm::algorithms::lz77_encode("aaaaaa").size()); // 2
/// @endcode
std::vector<LZ77Token> lz77_encode(const std::string& data, int window = 4096, int max_match = 255);
/// @brief Return actual greedy tokens and selected producing search work without history.
/// @param data Complete arbitrary byte input.
/// @param window Nonnegative maximum offset.
/// @param max_match Nonnegative maximum copied length.
/// @return Same-run greedy tuples and actual selected work.
/// @throws std::invalid_argument For negative settings.
/// @throws std::length_error For input/count geometry limits.
LZ77EncodeAnalysis
lz77_encode_analyze(const std::string& data, int window = 4096, int max_match = 255);
/// @brief Capture the real greedy candidate/comparison/emission loop, not a reconstructed token explanation.
/// @param data Complete arbitrary byte input.
/// @param window Nonnegative maximum offset.
/// @param max_match Nonnegative maximum copied length.
/// @param max_entries Optional logical capture cap, default one million; zero disables it.
/// @return Owned actual input and all actual selected search/emission records.
/// @throws std::invalid_argument For negative settings.
/// @throws std::length_error Before the next capture copy if actual logical input/record counts exceed the cap.
/// @note Unknown future comparisons are not fabricated. Excludes scalar fields, capacities, output/runtime copies and bytes.
LZ77EncodeTrace lz77_encode_trace(
    const std::string& data,
    int                window      = 4096,
    int                max_match   = 255,
    std::size_t        max_entries = 1000000);
/// @brief Completely validate and decode backward copies, including reads from bytes just produced by the same token.
/// @param tokens Native int tuples; pure literal requires offset=length=0; a match requires length>0 and1<=offset<=current size.
/// @return Actual decoded arbitrary bytes; empty stream returns empty.
/// @throws std::invalid_argument For any negative field, unavailable offset, noncanonical zero-length offset or empty nonliteral token.
/// @throws std::length_error For unrepresentable complete output/container geometry.
/// @note Complete structural/output-size validation precedes emitting any byte. Nonfinal match-only tuples are allowed.
/// @code
/// auto s=dm::algorithms::lz77_decode(dm::algorithms::lz77_encode("abcabc"));
/// std::print("{}\n",s); // abcabc
/// @endcode
std::string lz77_decode(const std::vector<LZ77Token>& tokens);
/// @brief Decode once and retain actual copied/literal/overlapping-read counts.
/// @param tokens Complete native tuple stream.
/// @param max_output_bytes Optional complete output cap; zero disables it, with container limits still enforced.
/// @return Actual owned decoded output and selected same-run counts.
/// @throws std::invalid_argument For any invalid token, including later invalid tokens after a valid prefix.
/// @throws std::length_error Before output allocation if full decoded size exceeds the explicit cap/container geometry.
LZ77DecodeAnalysis lz77_decode_analyze(
    const std::vector<LZ77Token>& tokens,
    std::size_t                   max_output_bytes = 0);
/// @brief Capture every real literal/copy byte decision from one fully validated decoder execution.
/// @param tokens Complete original tuple stream, copied only after full guards/admission.
/// @param max_output_bytes Optional complete output-size cap; zero disables it.
/// @param max_entries Optional complete logical capture cap; default one million, zero disables it.
/// @return Owned original tuples, actual source/output byte decisions and same-run output/work.
/// @throws std::invalid_argument For invalid complete tokens.
/// @throws std::length_error Before capture if token count plus decoded byte count exceeds quota or geometry.
/// @note Excludes scalar fields, capacities, returned/runtime output copies and physical bytes; all actual decoded writes are retained.
LZ77DecodeTrace lz77_decode_trace(
    const std::vector<LZ77Token>& tokens,
    std::size_t                   max_output_bytes = 0,
    std::size_t                   max_entries      = 1000000);
/// @brief Own the original repeated-pangram fixture, exactly89 ASCII bytes including the inter-sentence space.
/// @return The original two-sentence LZ77 report input, without a trailing newline.
std::string lz77_demo_text();
} // namespace datamunge::algorithms
