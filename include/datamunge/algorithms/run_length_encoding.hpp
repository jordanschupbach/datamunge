#pragma once
/// @file run_length_encoding.hpp
/// @brief Checked raw-byte run encoding, explicit records and actual boundary comparisons.
#include <cstddef>
#include <string>
#include <vector>
/// @brief Canonical byte encoders and owned producing records.
namespace datamunge::algorithms {
/// @brief One owned raw byte and its positive encoder run length; decoders also permit zero lengths.
/// @par Example
/// @code
/// const dm::algorithms::Run value{};
/// std::print("{}\n",value.count); // 0
/// @endcode
struct Run {
  char        symbol{}; ///< Raw byte, preserving NUL/high bytes without text interpretation.
  std::size_t count{};  ///< Repetition count; encoder runs are always positive.
};
/// @brief Complete input/output and retained-run policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RLEOptions value{};
/// std::print("{}\n",value.max_bytes); // 1000000
/// @endcode
struct RLEOptions {
  std::size_t max_bytes{1000000};         ///< Complete source or decoded byte count.
  std::size_t max_runs{1000000};          ///< Complete input/output run count.
  std::size_t max_trace_entries{1000000}; ///< One actual span record per run.
};
/// @brief One actual encoded run span or decoder append span.
/// @par Example
/// @code
/// const dm::algorithms::RLEStep value{};
/// std::print("{}\n",value.begin); // 0
/// @endcode
struct RLEStep {
  std::size_t run_index{}; ///< Actual chronological run id.
  std::size_t begin{};     ///< First source/decoded byte position.
  std::size_t end{};       ///< First byte after the actual run.
  char        symbol{};    ///< Actual raw byte.
  std::size_t
      comparisons{}; ///< Actual encoder equality tests for this run, including a failing boundary; zero when decoding.
};
/// @brief Owned canonical output and actual selected byte/run work.
/// @par Example
/// @code
/// const dm::algorithms::RLERun value{};
/// std::print("{}\n",value.comparisons); // 0
/// @endcode
struct RLERun {
  std::string      text;       ///< Owned original input or actual decoded bytes.
  std::vector<Run> runs;       ///< Actual encoded runs or owned complete decoder input.
  bool             decoding{}; ///< Whether the producing execution decoded rather than encoded.
  std::size_t comparisons{};   ///< Actual encoder symbol equality tests, including boundary tests.
  std::size_t emitted_bytes{}; ///< Actual decoded bytes; zero during encoding.
  std::vector<RLEStep> steps;  ///< Actual optional chronological source/append spans.
};
/// @brief Execute the canonical byte run encoder.
/// Encoding scans maximal equal-byte runs without interpreting Unicode; empty input has no runs. Each encoder equality comparison is counted at the actual branch, including the first unequal boundary. Decoding validates the complete aggregate length and all policies before appending, preserving all bytes; zero-length records append nothing. A run is one logical symbol/count record, not a serialized wire byte or a claim of compression. No byte format is invented. Time O(bytes+runs), O(bytes+runs) owned result and O(actual runs) optional spans. Inputs unchanged, no retained pointers/shared state. One history entry per actual encoded run or decoder input record, admitted before capture/append.
/// @param text Complete borrowed byte string, including embedded NUL.
/// @param options Complete byte/run/history bounds, default bounded.
/// @return Owned actual input/output and optional same-pass span records.
/// @throws std::length_error Aggregate count overflow, string/container capacity or any execution/history policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::rle_encode_run("AAABB");
/// std::print("{} {}\n",r.text.size(),r.runs.size()); // 5 2
/// @endcode
/// @par References
/// - Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rle_encode(), rle_decode()
RLERun rle_encode_run(const std::string& text, const RLEOptions& options = {});
/// @brief Execute the canonical byte run encoder with actual spans.
/// Encoding scans maximal equal-byte runs without interpreting Unicode; empty input has no runs. Each encoder equality comparison is counted at the actual branch, including the first unequal boundary. Decoding validates the complete aggregate length and all policies before appending, preserving all bytes; zero-length records append nothing. A run is one logical symbol/count record, not a serialized wire byte or a claim of compression. No byte format is invented. Time O(bytes+runs), O(bytes+runs) owned result and O(actual runs) optional spans. Inputs unchanged, no retained pointers/shared state. One history entry per actual encoded run or decoder input record, admitted before capture/append.
/// @param text Complete borrowed byte string, including embedded NUL.
/// @param options Complete byte/run/history bounds, default bounded.
/// @return Owned actual input/output and optional same-pass span records.
/// @throws std::length_error Aggregate count overflow, string/container capacity or any execution/history policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::rle_encode_trace("AAABB");
/// std::print("{} {}\n",r.text.size(),r.runs.size()); // 5 2
/// @endcode
/// @par References
/// - Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rle_encode(), rle_decode()
RLERun rle_encode_trace(const std::string& text, const RLEOptions& options = {});
/// @brief Execute the canonical byte run decoder.
/// Encoding scans maximal equal-byte runs without interpreting Unicode; empty input has no runs. Each encoder equality comparison is counted at the actual branch, including the first unequal boundary. Decoding validates the complete aggregate length and all policies before appending, preserving all bytes; zero-length records append nothing. A run is one logical symbol/count record, not a serialized wire byte or a claim of compression. No byte format is invented. Time O(bytes+runs), O(bytes+runs) owned result and O(actual runs) optional spans. Inputs unchanged, no retained pointers/shared state. One history entry per actual encoded run or decoder input record, admitted before capture/append.
/// @param runs Complete borrowed records; zero lengths and adjacent equal symbols are permitted.
/// @param options Complete byte/run/history bounds, default bounded.
/// @return Owned actual input/output and optional same-pass span records.
/// @throws std::length_error Aggregate count overflow, string/container capacity or any execution/history policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::rle_decode_run({{'A',3},{'B',2}});
/// std::print("{} {}\n",r.text.size(),r.runs.size()); // 5 2
/// @endcode
/// @par References
/// - Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rle_encode(), rle_decode()
RLERun rle_decode_run(const std::vector<Run>& runs, const RLEOptions& options = {});
/// @brief Execute the canonical byte run decoder with actual spans.
/// Encoding scans maximal equal-byte runs without interpreting Unicode; empty input has no runs. Each encoder equality comparison is counted at the actual branch, including the first unequal boundary. Decoding validates the complete aggregate length and all policies before appending, preserving all bytes; zero-length records append nothing. A run is one logical symbol/count record, not a serialized wire byte or a claim of compression. No byte format is invented. Time O(bytes+runs), O(bytes+runs) owned result and O(actual runs) optional spans. Inputs unchanged, no retained pointers/shared state. One history entry per actual encoded run or decoder input record, admitted before capture/append.
/// @param runs Complete borrowed records; zero lengths and adjacent equal symbols are permitted.
/// @param options Complete byte/run/history bounds, default bounded.
/// @return Owned actual input/output and optional same-pass span records.
/// @throws std::length_error Aggregate count overflow, string/container capacity or any execution/history policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::rle_decode_trace({{'A',3},{'B',2}});
/// std::print("{} {}\n",r.text.size(),r.runs.size()); // 5 2
/// @endcode
/// @par References
/// - Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rle_encode(), rle_decode()
RLERun rle_decode_trace(const std::vector<Run>& runs, const RLEOptions& options = {});
/// @brief Return canonical checked runs through the preserved signature.
/// Delegates to the shared producer with default million-byte/run policies. Valid ordinary outputs and raw-byte behavior are preserved.
/// @param text Complete raw input bytes.
/// @return Owned positive maximal runs.
/// @throws std::length_error Default byte/run or container capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rle_encode("AAABB").size()); // 2
/// @endcode
/// @par References
/// - Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rle_encode_run()
std::vector<Run> rle_encode(const std::string& text);
/// @brief Return canonical checked decoded bytes through the preserved signature.
/// Delegates to the shared producer with default million-byte/run policies. Valid ordinary outputs and raw-byte behavior are preserved.
/// @param runs Complete raw symbol/count records, zero lengths allowed.
/// @return Owned decoded raw bytes.
/// @throws std::length_error Default byte/run or container capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rle_decode({{'A',3}})); // AAA
/// @endcode
/// @par References
/// - Golomb (1966), Run-Length Encodings, doi:10.1109/TIT.1966.1053907.
/// @see rle_decode_run()
std::string rle_decode(const std::vector<Run>& runs);
/// @brief Count logical symbol/count records, preserving the legacy meaning of cells.
/// This is runs.size(), not serialized bytes, fields or compression ratio; zero-count input records still occupy one logical record. O(1), inputs unchanged.
/// @param runs Borrowed complete record container.
/// @return Number of logical records.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rle_encoded_cells({{'A',20}})); // 1
/// @endcode
/// @see rle_encode()
std::size_t rle_encoded_cells(const std::vector<Run>& runs);
/// @brief Generate a controlled constant or alternating native raw-byte input.
/// Returns exactly size bytes, all A when alternating=false and A,B,... otherwise.
/// This is a synthetic run-structure control, not a random or serialized dataset.
/// @param size Requested byte count, including zero.
/// @param alternating Select alternating A/B rather than constant A.
/// @param max_bytes Complete allocation bound, default1000000; zero disables.
/// @return Owned controlled raw bytes.
/// @throws std::length_error Requested count exceeds policy or string capacity.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rle_example_pattern(4,true)); // ABAB
/// @endcode
/// @see rle_encode_run()
std::string
rle_example_pattern(std::size_t size, bool alternating = false, std::size_t max_bytes = 1000000);
} // namespace datamunge::algorithms
