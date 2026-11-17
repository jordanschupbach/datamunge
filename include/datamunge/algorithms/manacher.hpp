#pragma once
/// @file manacher.hpp
/// @brief Safe tagged-byte Manacher radii and actual mirror/expansion/frontier evidence.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Leftmost longest contiguous byte palindrome, with zero start/length for empty input.
/// @par Example
/// @code
/// auto p=dm::algorithms::longest_palindrome("babad");
/// std::print("{} {}\n",p.start,p.length); // 0 3
/// @endcode
/// @par References
/// - Manacher(1975), A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String, https://doi.org/10.1145/321892.321896; full-radius tagged-byte adaptation here.
/// @see manacher_analyze(), manacher_trace()
struct PalindromeResult {
  std::size_t start{0};  ///< Actual zero-based byte start, zero for empty result.
  std::size_t length{0}; ///< Actual contiguous byte length, zero for empty input.
};
/// @brief Actual selected mirror-scan work, excluding guards, allocation and copies.
/// @par Example
/// @code
/// auto t=dm::algorithms::manacher_trace("aba");
/// std::print("{}\n",t.analysis.work.centers); // 7
/// @endcode
/// @par References
/// - Manacher(1975), A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String, https://doi.org/10.1145/321892.321896; full-radius tagged-byte adaptation here.
/// @see manacher_analyze(), manacher_trace()
struct ManacherWork {
  std::size_t centers{0}; ///< Actual nonsentinel transformed centers visited,2*n+1.
  std::size_t mirror_reuses{
      0}; ///< Actual centers inside the known right frontier using a capped mirror radius.
  std::size_t comparisons{
      0}; ///< Actual tagged-symbol comparisons, including terminating mismatches.
  std::size_t expansions{0}; ///< Actual successful equality expansions.
  std::size_t frontier_updates{
      0}; ///< Actual strict advances of the stored rightmost palindrome endpoint.
  std::size_t best_updates{
      0}; ///< Actual strict longest-radius improvements, preserving leftmost ties.
};
/// @brief Owned actual radii, selected byte palindrome and same-run scan counts.
/// @par Example
/// @code
/// auto t=dm::algorithms::manacher_trace("aba");
/// std::print("{}\n",t.analysis.best_center); // 4
/// @endcode
/// @par References
/// - Manacher(1975), A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String, https://doi.org/10.1145/321892.321896; full-radius tagged-byte adaptation here.
/// @see manacher_analyze(), manacher_trace()
struct ManacherAnalysis {
  std::vector<std::size_t>
                   radii;   ///< Actual transformed radii,2*n+3 slots with zero sentinel radii.
  PalindromeResult longest; ///< Actual first strict maximum converted to original byte coordinates.
  std::size_t      best_center{
      0};       ///< Actual transformed maximum center, zero when no positive radius exists.
  ManacherWork work; ///< Actual selected producing operations.
};
/// @brief One actual tagged-symbol equality test during center expansion.
/// @par Example
/// @code
/// auto t=dm::algorithms::manacher_trace("aba");
/// std::print("{}\n",t.comparisons.front().equal); // false
/// @endcode
/// @par References
/// - Manacher(1975), A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String, https://doi.org/10.1145/321892.321896; full-radius tagged-byte adaptation here.
/// @see manacher_analyze(), manacher_trace()
struct ManacherComparison {
  std::size_t center;       ///< Actual currently scanned transformed center.
  std::size_t radius;       ///< Actual radius before this test.
  std::size_t left;         ///< Actual left transformed index compared.
  std::size_t right;        ///< Actual right transformed index compared.
  int         left_symbol;  ///< Actual original-byte or distinct tagged separator/guard symbol.
  int         right_symbol; ///< Actual compared original-byte or tagged symbol.
  bool        equal;        ///< Actual equality outcome; failed tests do not advance the frontier.
};
/// @brief One actual mirror initialization, expansion result and frontier/best decision.
/// @par Example
/// @code
/// auto t=dm::algorithms::manacher_trace("aba");
/// std::print("{}\n",t.centers.front().index); // 1
/// @endcode
/// @par References
/// - Manacher(1975), A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String, https://doi.org/10.1145/321892.321896; full-radius tagged-byte adaptation here.
/// @see manacher_analyze(), manacher_trace()
struct ManacherCenter {
  std::size_t index;         ///< Actual transformed center visited in increasing order.
  std::size_t center_before; ///< Actual stored frontier center before this visit.
  std::size_t right_before;  ///< Actual inclusive rightmost palindrome endpoint before this visit.
  bool        reused;        ///< Actual index<right_before branch decision.
  std::size_t mirror;        ///< Actual mirrored index, SIZE_MAX when not reused.
  std::size_t mirror_radius; ///< Actual prior mirror radius read, zero placeholder when not reused.
  std::size_t bound;   ///< Actual right_before-index mirror cap, zero placeholder when not reused.
  std::size_t initial; ///< Actual selected initial radius before testing symbols.
  std::size_t final;   ///< Actual radius after real equality tests terminate.
  bool        frontier_updated; ///< Actual strict index+final>right_before decision.
  std::size_t center_after;     ///< Actual stored center after this visit.
  std::size_t right_after;      ///< Actual inclusive endpoint after this visit.
  bool
      new_best; ///< Actual strict best-length update; ties retain the earlier center/leftmost start.
};
/// @brief Owned original bytes, actual tagged transform and complete real mirror/expansion records.
/// @par Example
/// @code
/// auto t=dm::algorithms::manacher_trace("abacaba");
/// std::print("{} {}\n",t.analysis.best_center,t.analysis.longest.length); // 8 7
/// @endcode
/// @par References
/// - Manacher(1975), A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String, https://doi.org/10.1145/321892.321896; full-radius tagged-byte adaptation here.
/// @see manacher_analyze(), manacher_trace()
struct ManacherTrace {
  std::string
      input; ///< Independently owned original arbitrary bytes, including NUL and sentinel-like bytes.
  std::vector<int>
      symbols; ///< Actual transform: left guard257, separator256, bytes0..255, right guard258.
  std::vector<ManacherCenter>
      centers; ///< Actual initialization/frontier decisions from one producer execution.
  std::vector<ManacherComparison>
                   comparisons; ///< Every actual symbol test, without palindrome replay.
  ManacherAnalysis analysis;    ///< Actual same-run final radii, selected palindrome and work.
  std::size_t      captured_entries{
      0}; ///< Original bytes, transform/final-radius slots and actual center/comparison records.
};
/// @brief Return the leftmost longest byte palindrome using safe distinct tagged separators/guards.
/// @param s Complete arbitrary bytes, not Unicode codepoints or grapheme clusters.
/// @return Original byte start/length, zero/zero when empty; strict equal-length ties keep the leftmost.
/// @throws std::length_error Before allocation if2*n+3 transform/radius geometry is unrepresentable.
/// @note O(n) time/space. Literal '^','#','$', NUL and high bytes cannot collide with tagged control symbols.
/// This fixes historical sentinel-collision out-of-bounds/incorrect-radius behavior without changing ordinary safe ASCII results.
/// @code
/// auto p=dm::algorithms::longest_palindrome("abba");
/// std::print("{} {}\n",p.start,p.length); // 0 4
/// @endcode
PalindromeResult longest_palindrome(const std::string& s);
/// @brief Return actual radii over the tagged2*n+3-symbol transform, with zero sentinel endpoints.
/// @param s Original arbitrary bytes.
/// @return Owned radius per transformed position; each nonsentinel radius equals its original-byte palindrome length.
/// @throws std::length_error For unrepresentable complete transform/radius geometry.
/// @note Original byte start for center i and radius r is floor((i-r)/2); only positive maxima are selected.
/// @code
/// auto p=dm::algorithms::palindrome_radii("abacaba");
/// std::print("{} {}\n",p.size(),p[8]); // 17 7
/// @endcode
std::vector<std::size_t> palindrome_radii(const std::string& s);
/// @brief Run the same tagged Manacher producer and retain actual selected work/result without history.
/// @param s Complete original arbitrary bytes.
/// @return Owned actual radii, first strict maximum and same-run mirror/expansion work.
/// @throws std::length_error For unrepresentable complete geometry.
/// @code
/// auto a=dm::algorithms::manacher_analyze("aba");
/// std::print("{}\n",a.longest.length); // 3
/// @endcode
ManacherAnalysis manacher_analyze(const std::string& s);
/// @brief Capture every real mirror initialization and equality test from one safe tagged producer execution.
/// @param s Complete original arbitrary bytes.
/// @param max_entries Optional logical capture cap, default one million; zero disables it.
/// @return Owned original bytes, actual tagged transform/radii and actual center/comparison records; never prefix replay.
/// @throws std::length_error Before copies if base n+2*(2*n+3) slots or the next actual record exceeds quota/geometry.
/// @note Counts logical elements/records, excluding fields, capacities, result/runtime copies and physical bytes.
/// Unknown future tests are not fabricated; capture is complete or throws. Empty input still has its actual separator center.
ManacherTrace manacher_trace(const std::string& s, std::size_t max_entries = 1000000);
/// @brief Label one actual tagged symbol without confusing original punctuation bytes with controls.
/// @param symbol Byte value0..255, separator256, left guard257 or right guard258.
/// @return ASCII label byte0xHH, separator, left guard or right guard.
/// @throws std::invalid_argument For a value outside0..258.
/// @code
/// std::print("{}\n",dm::algorithms::manacher_symbol_label(94)); // byte0x5E
/// @endcode
std::string manacher_symbol_label(int symbol);
} // namespace datamunge::algorithms
