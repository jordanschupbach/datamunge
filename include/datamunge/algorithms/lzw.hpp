#pragma once
/// @file lzw.hpp
/// @brief Checked byte-seeded integer-code LZW with actual encoder decisions and decoder dictionary reconstruction.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Selected actual string-dictionary encoder work, not a packed-size/time estimate.
/// @par Example
/// @code
/// auto t=dm::algorithms::lzw_encode_trace("AAA");
/// std::print("{}\n",t.analysis.work.emissions); // 2
/// @endcode
/// @par References
/// - Welch(1984), A Technique for High-Performance Data Compression, https://doi.org/10.1109/MC.1984.1659158; byte-seeded integer codes here, without a packed/reset protocol.
/// @see lzw_encode_trace(), lzw_decode_trace()
struct LZWEncodeWork {
  std::size_t input_bytes{0};     ///< Actual processed input bytes.
  std::size_t lookups{0};         ///< Actual candidate lookups plus optional final-prefix lookup.
  std::size_t candidate_bytes{0}; ///< Sum of actual constructed prefix-plus-byte string lengths.
  std::size_t insertions{0};      ///< Actual learned entries beyond the256 initial bytes.
  std::size_t dictionary_bytes{
      256}; ///< Conceptual sum of stored dictionary string lengths, including256 seed bytes.
  std::size_t emissions{0}; ///< Actual emitted integer codes.
};
/// @brief Owned actual integer-code stream and same-run string-dictionary work.
/// @par Example
/// @code
/// auto t=dm::algorithms::lzw_encode_trace("AAA");
/// std::print("{}\n",t.analysis.codes.back()); // 256
/// @endcode
/// @par References
/// - Welch(1984), A Technique for High-Performance Data Compression, https://doi.org/10.1109/MC.1984.1659158; byte-seeded integer codes here, without a packed/reset protocol.
/// @see lzw_encode_trace(), lzw_decode_trace()
struct LZWEncodeAnalysis {
  std::vector<int> codes; ///< Actual ordered emitted codes, seeded byte codes0..255.
  LZWEncodeWork    work;  ///< Actual producing selected work.
};
/// @brief One actual streaming dictionary lookup/learn/emit decision or final pending-prefix flush.
/// @par Example
/// @code
/// auto t=dm::algorithms::lzw_encode_trace("AAA");
/// std::print("{}\n",t.steps[1].learned_code); // 256
/// @endcode
/// @par References
/// - Welch(1984), A Technique for High-Performance Data Compression, https://doi.org/10.1109/MC.1984.1659158; byte-seeded integer codes here, without a packed/reset protocol.
/// @see lzw_encode_trace(), lzw_decode_trace()
struct LZWEncodeStep {
  std::size_t input_index;  ///< Actual input byte position, or one-past input on final flush.
  unsigned    byte;         ///< Actual unsigned input byte, placeholder0 on flush.
  bool        flush;        ///< True only for the final actual pending-prefix emission.
  std::string prefix;       ///< Actual pending dictionary prefix before this decision.
  std::string candidate;    ///< Actual prefix-plus-byte lookup string, empty on flush.
  bool        known;        ///< Actual candidate membership result, false placeholder on flush.
  int         emitted_code; ///< Actual emitted code, minus1 when this decision emits none.
  int learned_code; ///< Actual new code assigned to candidate, minus1 when no entry is learned.
};
/// @brief Owned actual input and complete streaming encoder decisions, without dictionary replay.
/// @par Example
/// @code
/// auto t=dm::algorithms::lzw_encode_trace("AAA");
/// std::print("{}\n",t.analysis.codes.size()); // 2
/// @endcode
/// @par References
/// - Welch(1984), A Technique for High-Performance Data Compression, https://doi.org/10.1109/MC.1984.1659158; byte-seeded integer codes here, without a packed/reset protocol.
/// @see lzw_encode_trace(), lzw_decode_trace()
struct LZWEncodeTrace {
  std::string input; ///< Independently owned original arbitrary bytes.
  std::vector<LZWEncodeStep>
                    steps;    ///< Actual input-byte decisions followed by an optional real flush.
  LZWEncodeAnalysis analysis; ///< Actual same-run code output/work.
  std::size_t       captured_entries{
      0}; ///< Input bytes plus each actual step record and its prefix/candidate bytes.
};
/// @brief Selected actual decoder dictionary/output work.
/// @par Example
/// @code
/// auto t=dm::algorithms::lzw_decode_trace({65,256});
/// std::print("{}\n",t.analysis.work.special_cases); // 1
/// @endcode
/// @par References
/// - Welch(1984), A Technique for High-Performance Data Compression, https://doi.org/10.1109/MC.1984.1659158; byte-seeded integer codes here, without a packed/reset protocol.
/// @see lzw_encode_trace(), lzw_decode_trace()
struct LZWDecodeWork {
  std::size_t codes{0};      ///< Actual processed integer codes.
  std::size_t insertions{0}; ///< Actual learned dictionary entries, one after each noninitial code.
  std::size_t special_cases{
      0}; ///< Actual code-equals-next-code cases, reconstructed as previous+previous.first.
  std::size_t dictionary_bytes{
      256}; ///< Actual conceptual dictionary string bytes, zero on empty input; otherwise includes256 seed bytes.
  std::size_t emitted_bytes{0}; ///< Actual appended decoded bytes.
};
/// @brief Owned actual decoded string and producing decoder work.
/// @par Example
/// @code
/// auto t=dm::algorithms::lzw_decode_trace({65,256});
/// std::print("{}\n",t.analysis.data); // AAA
/// @endcode
/// @par References
/// - Welch(1984), A Technique for High-Performance Data Compression, https://doi.org/10.1109/MC.1984.1659158; byte-seeded integer codes here, without a packed/reset protocol.
/// @see lzw_encode_trace(), lzw_decode_trace()
struct LZWDecodeAnalysis {
  std::string   data; ///< Actual decoded arbitrary bytes, not a silently returned corrupt prefix.
  LZWDecodeWork work; ///< Actual dictionary/output operations from the same run.
};
/// @brief One actual decoded code word and dictionary entry learned from the previous word.
/// @par Example
/// @code
/// auto t=dm::algorithms::lzw_decode_trace({65,256});
/// std::print("{}\n",t.steps.back().entry); // AA
/// @endcode
/// @par References
/// - Welch(1984), A Technique for High-Performance Data Compression, https://doi.org/10.1109/MC.1984.1659158; byte-seeded integer codes here, without a packed/reset protocol.
/// @see lzw_encode_trace(), lzw_decode_trace()
struct LZWDecodeStep {
  std::size_t input_index;  ///< Actual input code index.
  int         code;         ///< Actual supplied integer code.
  bool        special;      ///< Actual missing-but-exactly-next-code reconstruction case.
  std::string entry;        ///< Actual decoded nonempty dictionary word for this code.
  int         learned_code; ///< Actual newly learned dictionary code, minus1 for the initial code.
  std::string
      learned_entry; ///< Actual previous word plus this entry’s first byte, empty on the initial code.
  std::size_t output_begin; ///< Actual output byte count before appending this word.
  std::size_t output_end;   ///< Actual output byte count after appending this word.
};
/// @brief Owned actual code stream and decoder word/learning records from one producing execution.
/// @par Example
/// @code
/// auto t=dm::algorithms::lzw_decode_trace({65,256});
/// std::print("{} {}\n",t.analysis.data,t.steps.back().special); // AAA true
/// @endcode
/// @par References
/// - Welch(1984), A Technique for High-Performance Data Compression, https://doi.org/10.1109/MC.1984.1659158; byte-seeded integer codes here, without a packed/reset protocol.
/// @see lzw_encode_trace(), lzw_decode_trace()
struct LZWDecodeTrace {
  std::vector<int>
      input; ///< Original complete integer codes, validated before dictionary construction/capture.
  std::vector<LZWDecodeStep>
      steps; ///< Actual decoded word/learned-entry records, no prefix decoding replay.
  LZWDecodeAnalysis analysis; ///< Actual same-run decoded output/work.
  std::size_t       captured_entries{
      0}; ///< Input code records plus each actual step and its entry/learned-string bytes.
};
/// @brief Encode arbitrary bytes using the historical256-byte seeded string dictionary and unbounded native integer codes.
/// @param data Arbitrary exact bytes, including NUL/high bytes; empty yields no codes.
/// @return Actual greedy longest-current-word integer codes, preserving ordinary historical output.
/// @throws std::length_error If learned codes exceed INT_MAX or selected geometry/counts exceed container limits.
/// @note This defines no bit packing, code-width schedule, dictionary reset, clear/end codes or GIF/TIFF/compress wire format.
/// String concatenation/hashing/storage cost depends on total constructed/learned word lengths; a linear byte-time claim is unwarranted.
/// @code
/// auto c=dm::algorithms::lzw_encode("AAA");
/// std::print("{} {}\n",c[0],c[1]); // 65 256
/// @endcode
std::vector<int> lzw_encode(const std::string& data);
/// @brief Return actual code output and selected producing string-dictionary counts without history.
/// @param data Complete arbitrary byte input.
/// @return Owned same-run output and work.
/// @throws std::length_error For code/count/container overflow.
LZWEncodeAnalysis lzw_encode_analyze(const std::string& data);
/// @brief Capture actual input-byte membership decisions and the real final pending-prefix flush.
/// @param data Complete arbitrary input bytes.
/// @param max_entries Optional logical input/record/string-byte cap, default one million; zero disables it.
/// @return Owned actual input, decisions and same-run codes/work.
/// @throws std::length_error Before the next capture copy if actual original bytes plus step/prefix/candidate records exceed the cap.
/// @note Excludes fields, capacities, returned/runtime dictionaries/codes and physical storage bytes; future decisions are not fabricated.
LZWEncodeTrace lzw_encode_trace(const std::string& data, std::size_t max_entries = 1000000);
/// @brief Completely validate integer code structure, then reconstruct exact words, including the next-code special case.
/// @param codes First code must be0..255; subsequent index k must be0..255+k, with complete representable dictionary geometry.
/// @return Actual decoded arbitrary bytes; empty stream yields empty output.
/// @throws std::invalid_argument For negative, invalid first or unavailable future codes, even after a valid prefix.
/// @throws std::length_error For code, output or container geometry overflow.
/// @note Invalid streams throw rather than the historical silent prefix return; numeric structure is checked before output processing.
/// @code
/// std::print("{}\n",dm::algorithms::lzw_decode({65,256})); // AAA
/// @endcode
std::string lzw_decode(const std::vector<int>& codes);
/// @brief Decode once and return actual word/output counts with an optional incremental complete-or-error output limit.
/// @param codes Complete validated integer code stream.
/// @param max_output_bytes Optional maximum decoded bytes; zero disables it, retaining container checks.
/// @return Actual decoded bytes and same-run work; no partial corrupt or capped prefix is returned.
/// @throws std::invalid_argument For invalid complete numeric code structure.
/// @throws std::length_error Before the next output append exceeding cap/container geometry.
LZWDecodeAnalysis lzw_decode_analyze(
    const std::vector<int>& codes,
    std::size_t             max_output_bytes = 0);
/// @brief Capture actual decoded words, special cases and newly reconstructed dictionary entries without replay.
/// @param codes Complete original integer codes.
/// @param max_output_bytes Optional actual output-byte cap, zero disables it.
/// @param max_entries Optional actual logical capture cap, default one million; zero disables it.
/// @return Owned original codes, actual word/learning records and same-run output/work.
/// @throws std::invalid_argument Before output/capture on invalid complete numeric code structure.
/// @throws std::length_error Before the next output/capture copy exceeds its geometry or selected cap.
/// @note Counts original code records, one per actual decoded step and its decoded/learned word bytes; fields,
/// capacities, model/runtime copies and returned output copies are excluded. Complete actual trace or exception.
LZWDecodeTrace lzw_decode_trace(
    const std::vector<int>& codes,
    std::size_t             max_output_bytes = 0,
    std::size_t             max_entries      = 1000000);
/// @brief Own the original classic24-byte TOBEORNOTTOBEORTOBEORNOT fixture.
/// @return Exact original classic uppercase input, without newline.
std::string lzw_classic_text();
} // namespace datamunge::algorithms
