#pragma once

/// @file match_rating.hpp
/// @brief Canonical match rating API, producing histories and checked native contracts.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Legacy MRA comparability, similarity, threshold and heuristic verdict.
/// @details Empty codex pairs preserve the native convention comparable=true, similarity6, minimum5 and match=true; this does not establish valid names or identity.
/// @par Example
/// @code
/// auto r=dm::algorithms::match_rating_comparison("Smith","Smyth");
/// std::print("{}\n",r.comparable); // true
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
struct MatchRatingResult {
  bool comparable{
      false}; ///< False when codex lengths differ by at least3; other fields stay at defaults.
  int  similarity{0}; ///< Six minus the larger final unmatched count.
  int  min_rating{0}; ///< Actual minimum from combined codex lengths,5/4/3/2.
  bool match{false};  ///< Comparable and similarity reaches the threshold; heuristic only.
};

/// @brief Actual normalization, kept-letter stream and final bounded codex.
/// @details Byte ctype follows the active C locale; it does not normalize UTF-8 accents. A dropped vowel still updates the previous alphabetic letter used by duplicate suppression.
/// @par Example
/// @code
/// auto a=dm::algorithms::match_rating_codex_analyze("Babab");
/// std::print("{}\n",a.codex); // BBB
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
struct MatchRatingCodexAnalysis {
  std::string normalized; ///< Actual uppercase alphabetic bytes, with other bytes omitted.
  std::string stripped;   ///< Actual kept first-letter/consonant stream before trimming.
  std::string codex;      ///< Actual full stream if at most6 letters, otherwise first3 plus last3.
};

/// @brief One actual raw-byte normalization/filter decision.
/// @details Nonalphabetic input has upper0 and cannot update previous or first-letter state.
/// @par Example
/// @code
/// auto t=dm::algorithms::match_rating_codex_trace("A1");
/// std::print("{}\n",t.steps[1].alphabetic); // false
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
struct MatchRatingCodexStep {
  std::size_t   index{0};          ///< Actual raw byte position.
  unsigned char raw{0};            ///< Actual original unsigned byte.
  bool          alphabetic{false}; ///< Actual raw-space exclusion and ctype alphabetic test.
  unsigned char upper{0};    ///< Actual uppercase alphabetic byte, zero for omitted nonletters.
  unsigned char previous{0}; ///< Previous actual alphabetic uppercase byte before this input.
  bool          first{true}; ///< Whether no previous alphabetic byte existed.
  bool          kept{false}; ///< Actual kept-letter decision.
  std::size_t   before{0};   ///< Actual stripped stream length before this byte.
  std::size_t   after{0};    ///< Actual stripped stream length after this byte.
};

/// @brief Owned raw input and actual byte decisions/phase strings from one producer.
/// @details Logical quota counts raw input bytes, actual decision records, actual normalized/stripped characters and final codex characters. Fields, capacities, runtime copies and bytes are excluded.
/// @par Example
/// @code
/// auto t=dm::algorithms::match_rating_codex_trace("A");
/// std::print("{}\n",t.captured_entries); // 5
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
struct MatchRatingCodexTrace {
  std::string                       input;    ///< Owned original arbitrary bytes.
  MatchRatingCodexAnalysis          analysis; ///< Actual same-run phase strings.
  std::vector<MatchRatingCodexStep> steps; ///< Actual raw-byte decisions, including skipped bytes.
  std::size_t captured_entries{0};         ///< Complete logical count for this codex execution.
};

/// @brief Actual pair-comparison phases and legacy result from one canonical execution.
/// @details Codex lengths are at most6. Forward unequal positions populate residual strings; reverse unequal positions count the final unmatched letters.
/// @par Example
/// @code
/// auto a=dm::algorithms::match_rating_analyze("A","A");
/// std::print("{}\n",a.result.similarity); // 6
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
struct MatchRatingAnalysis {
  MatchRatingResult result;  ///< Actual legacy verdict fields.
  std::string       codex_a; ///< Actual first final codex.
  std::string       codex_b; ///< Actual second final codex.
  std::string
      residual_a; ///< Actual first residual after forward cancellation; empty if not comparable.
  std::string
      residual_b; ///< Actual second residual after forward cancellation; empty if not comparable.
  int unmatched_a{0}; ///< Actual first unmatched count after reverse pass.
  int unmatched_b{0}; ///< Actual second unmatched count after reverse pass.
};

/// @brief One actual aligned cancellation position in the forward or reverse pass.
/// @details Zero character denotes absent input at this aligned position. Counts are residual lengths in the forward pass and final unmatched counts in reverse.
/// @par Example
/// @code
/// auto t=dm::algorithms::match_rating_trace("A","B");
/// std::print("{}\n",t.cancellations[0].equal); // false
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
struct MatchRatingCancellation {
  bool reverse{false}; ///< False for forward codex alignment, true for reverse residual alignment.
  std::size_t   index{0};     ///< Actual offset from the active scanning end.
  unsigned char a{0};         ///< Actual first compared byte, zero for absent.
  unsigned char b{0};         ///< Actual second compared byte, zero for absent.
  bool          equal{false}; ///< Actual equality decision.
  std::size_t   before_a{0};  ///< Actual first count before this position.
  std::size_t   before_b{0};  ///< Actual second count before this position.
  std::size_t   after_a{0};   ///< Actual first count after this position.
  std::size_t   after_b{0};   ///< Actual second count after this position.
};

/// @brief Owned codex histories and actual two-pass pair cancellation from one execution.
/// @details One shared quota covers both codex captures, every actual cancellation record and each actual forward residual character. Returned phase copies are not counted again; fields/capacities/runtime storage and bytes are excluded.
/// @par Example
/// @code
/// auto t=dm::algorithms::match_rating_trace("A","A");
/// std::print("{}\n",t.captured_entries); // 11
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
struct MatchRatingTrace {
  MatchRatingCodexTrace                a;             ///< Actual first codex execution.
  MatchRatingCodexTrace                b;             ///< Actual second codex execution.
  MatchRatingAnalysis                  analysis;      ///< Same-run actual comparison phases/result.
  std::vector<MatchRatingCancellation> cancellations; ///< Actual forward then reverse decisions.
  std::size_t                          captured_entries{0}; ///< Complete shared logical count.
};

/// @brief Encode a name with the exact native MRA byte rules.
/// @details Keep the first alphabetic letter, then a consonant only if different from the preceding alphabetic source letter, including omitted vowels. Ignore other bytes; ctype case/letter classification uses the current C locale. If more than6 kept letters remain, retain first3 and last3. O(N) time/storage. Empty or nonletter input gives an empty codex. Not Unicode accent normalization.
/// @param name Original arbitrary byte string.
/// @return Native codex of at most6 letters.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::match_rating_codex("Ashcraft")); // ASHRFT
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
std::string match_rating_codex(const std::string& name);

/// @brief Return the actual normalization/filter/trim phases from the canonical codex producer.
/// @details No independent encoder is used to explain the result.
/// @param name Original arbitrary bytes.
/// @return Owned actual normalized, stripped and final codex strings.
/// @par Example
/// @code
/// auto a=dm::algorithms::match_rating_codex_analyze("Ashcraft");
/// std::print("{}\n",a.stripped); // ASHCRFT
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
MatchRatingCodexAnalysis match_rating_codex_analyze(const std::string& name);

/// @brief Capture every actual raw-byte codex decision and owned phase string.
/// @details Admit original bytes first, then actual decision/character records before their operations. Default complete logical cap one million; zero off. Cap failure returns no prefix.
/// @param name Original arbitrary bytes.
/// @param max_entries Complete logical quota, zero off.
/// @return Owned input and complete actual codex history.
/// @throws std::length_error On complete actual capture/count admission overflow.
/// @par Example
/// @code
/// auto t=dm::algorithms::match_rating_codex_trace("A",5);
/// std::print("{}\n",t.analysis.codex); // A
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
MatchRatingCodexTrace match_rating_codex_trace(
    const std::string& name,
    std::size_t        max_entries = 1000000);

/// @brief Compare native MRA codexes with actual aligned forward/reverse cancellation.
/// @details A length gap of at least3 is not comparable. Otherwise threshold is5 if combined length<=4,4 if<=7,3 if<=11,else2. Forward aligned equal letters cancel; align residuals from their right ends and count unequal letters on each side. Similarity6-max(unmatched) reaches threshold for a heuristic match. Not an identity probability, equivalence relation or labelled accuracy guarantee. O(|a|+|b|) time/storage. Empty-codex matching convention is retained explicitly.
/// @param a First original name bytes.
/// @param b Second original name bytes.
/// @return Legacy comparability/score/threshold/verdict.
/// @par Example
/// @code
/// auto r=dm::algorithms::match_rating_comparison("A","A");
/// std::print("{} {}\n",r.similarity,r.min_rating); // 6 5
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
MatchRatingResult match_rating_comparison(const std::string& a, const std::string& b);

/// @brief Return actual codex/residual/unmatched comparison phases and legacy verdict.
/// @details Uses the same canonical codex and cancellation producers as the ordinary function.
/// @param a First name bytes.
/// @param b Second name bytes.
/// @return Owned actual phase strings and same-run result.
/// @par Example
/// @code
/// auto a=dm::algorithms::match_rating_analyze("A","B");
/// std::print("{}\n",a.unmatched_a); // 1
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
MatchRatingAnalysis match_rating_analyze(const std::string& a, const std::string& b);

/// @brief Capture both actual codex executions and every actual cancellation under one shared quota.
/// @details Complete logical count adds both codex capture counts, actual cancellation records and actual residual characters. Default cap one million; zero off. Not-comparable results have no cancellation records. Admission failures throw without returning a partial result.
/// @param a First name bytes.
/// @param b Second name bytes.
/// @param max_entries Shared complete logical cap, zero off.
/// @return Owned complete actual pair history and same-run phases/result.
/// @throws std::length_error On complete actual capture/count admission overflow.
/// @par Example
/// @code
/// auto t=dm::algorithms::match_rating_trace("A","A",11);
/// std::print("{}\n",t.cancellations.size()); // 1
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
MatchRatingTrace
match_rating_trace(const std::string& a, const std::string& b, std::size_t max_entries = 1000000);

/// @brief Return the original twelve name-pair comparison fixtures.
/// @details These pairs carry no ground-truth identity labels; they are mechanism examples, not a validation dataset.
/// @return Owned original twelve pairs.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::match_rating_demo_pairs().size()); // 12
/// @endcode
/// @see match_rating_codex, match_rating_comparison, match_rating_trace
/// @par Reference
/// Western Airlines (1977), Match Rating Approach; Apache Commons Codec MRA API documents the method attribution, while native preprocessing conventions are specified here.
std::vector<std::pair<std::string, std::string>> match_rating_demo_pairs();

} // namespace datamunge::algorithms
