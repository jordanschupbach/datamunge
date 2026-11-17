#pragma once
/// @file nysiis.hpp
/// @brief Native legacy NYSIIS byte-rule variant with actual prefix/token/cleanup histories.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual untruncated native key and selected interior-token work.
/// @par Example
/// @code
/// auto a=dm::algorithms::nysiis_analyze("MacDonald");std::print("{}\n",a.key); // MCDANALD
/// @endcode
/// @see nysiis_trace()
struct NysiisAnalysis {
  std::string key;                ///< Owned actual native byte key, empty only for empty input.
  std::size_t input_bytes{0};     ///< Original byte count, not Unicode characters/phonemes.
  std::size_t interior_tokens{0}; ///< Actual interior rewrite tokens, excluding the first key byte.
  std::size_t suppressed_tokens{
      0}; ///< Actual whole tokens suppressed because token.back()==key.back().
};
/// @brief Actual prefix/suffix/first/interior/cleanup rewrite and owned states.
/// @par Example
/// @code
/// auto t=dm::algorithms::nysiis_trace("MAC");std::print("{}\n",t.steps[0].after); // MCC
/// @endcode
/// @see nysiis_trace()
struct NysiisStep {
  std::string stage; ///< Actual stage: prefix,suffix,first,interior or cleanup.
  std::string rule;  ///< Actual matched branch label, or none/literal for no substitution.
  std::size_t begin{
      0}; ///< Actual rewritten-string start index for first/interior,0 placeholder for other stages.
  std::size_t end{
      0}; ///< Actual exclusive consumed rewritten-string index for first/interior,0 placeholder otherwise.
  std::string
      before; ///< Actual whole working string for prefix/suffix, or whole key for first/interior/cleanup.
  std::string after; ///< Actual same-state string after this producer decision.
  std::string
      token; ///< Actual prefix/suffix replacement, first byte or interior token; empty for cleanup/no match.
  bool appended{false}; ///< Actual first/interior token append; false/ignored for other stages.
};
/// @brief Owned original bytes, actual working rewrites, complete decisions and final native key.
/// Counts input/uppercase/rewritten/final-key bytes, plus per record1 and all
/// retained stage/rule/before/after/token bytes. Capture can be quadratic in
/// input length because actual full key snapshots are owned; ordinary encoding
/// is linear. Empty input has no stages, empty strings and0 capture units.
/// @par Example
/// @code
/// auto t=dm::algorithms::nysiis_trace("MAC");std::print("{}\n",t.analysis.key); // MC
/// @endcode
/// @see nysiis_trace()
struct NysiisTrace {
  std::string    input;     ///< Owned original bytes, including any punctuation/NUL/high bytes.
  std::string    uppercase; ///< Actual same-length C-locale uppercase-byte working string.
  std::string    rewritten; ///< Actual string after prefix and suffix rewriting.
  NysiisAnalysis analysis;  ///< Actual same-producer final key and token counters.
  std::vector<NysiisStep> steps; ///< Actual complete producer decisions, no replay.
  std::size_t             captured_entries{
      0}; ///< Actual retained string-byte units plus one per decision record.
};
/// @brief Encode the original native untruncated NYSIIS byte-rule variant.
/// Preserves the legacy producer exactly: current C-locale toupper with unsigned-byte
/// safety; no filtering, Unicode normalization or six-character truncation.
/// Prefix MAC->MCC,KN->N,K->C,PH/PF->FF,SCH->SSS; suffix IE/EE->Y,
/// DT/RT/RD/NT/ND->D. First rewritten byte is retained verbatim. Interior EV->AF,
/// vowels->A,Q->G,Z->S,M->N,KN->N,K->C,SCH->SS,PH->F and contextual H/W
/// rules use the original rewritten predecessor bytes. Whole replacement tokens
/// are suppressed by token.back()==key.back(), not per-character compression.
/// Consequently internal SS tokens/raw-predecessor W rules can differ from
/// other NYSIIS implementations; no Jellyfish/standard-conformance claim.
/// Trim one trailing S, fold AY->Y, trim one trailing A, in that order, retaining
/// a one-byte key. Empty input returns empty. All byte strings are accepted;
/// punctuation/high bytes/NUL can survive, so output need not be alphabetic.
/// O(N) ordinary work/storage; optional full snapshots cost actual captured
/// bytes, potentially O(N squared). No mutation/references retained; results
/// depend on the process C locale, which this function does not change.
/// @param name Complete original byte string, case conversion as described.
/// @return Owned actual std::string key/result with no identity-equivalence guarantee.
/// @throws std::length_error If count/string geometry is exceeded.
/// @throws std::bad_alloc If encoding/history allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::nysiis("Catherine");std::print("{}\n",r); // CATARAN
/// @endcode
/// @see NysiisAnalysis, NysiisTrace
std::string nysiis(const std::string& name);
/// @brief Encode the original native untruncated NYSIIS byte-rule variant.
/// Preserves the legacy producer exactly: current C-locale toupper with unsigned-byte
/// safety; no filtering, Unicode normalization or six-character truncation.
/// Prefix MAC->MCC,KN->N,K->C,PH/PF->FF,SCH->SSS; suffix IE/EE->Y,
/// DT/RT/RD/NT/ND->D. First rewritten byte is retained verbatim. Interior EV->AF,
/// vowels->A,Q->G,Z->S,M->N,KN->N,K->C,SCH->SS,PH->F and contextual H/W
/// rules use the original rewritten predecessor bytes. Whole replacement tokens
/// are suppressed by token.back()==key.back(), not per-character compression.
/// Consequently internal SS tokens/raw-predecessor W rules can differ from
/// other NYSIIS implementations; no Jellyfish/standard-conformance claim.
/// Trim one trailing S, fold AY->Y, trim one trailing A, in that order, retaining
/// a one-byte key. Empty input returns empty. All byte strings are accepted;
/// punctuation/high bytes/NUL can survive, so output need not be alphabetic.
/// O(N) ordinary work/storage; optional full snapshots cost actual captured
/// bytes, potentially O(N squared). No mutation/references retained; results
/// depend on the process C locale, which this function does not change.
/// @param name Complete original byte string, case conversion as described.
/// @return Owned actual NysiisAnalysis key/result with no identity-equivalence guarantee.
/// @throws std::length_error If count/string geometry is exceeded.
/// @throws std::bad_alloc If encoding/history allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::nysiis_analyze("Catherine");std::print("{}\n",r.key); // CATARAN
/// @endcode
/// @see NysiisAnalysis, NysiisTrace
NysiisAnalysis nysiis_analyze(const std::string& name);
/// @brief Capture actual native NYSIIS byte-rule decisions.
/// Preserves the legacy producer exactly: current C-locale toupper with unsigned-byte
/// safety; no filtering, Unicode normalization or six-character truncation.
/// Prefix MAC->MCC,KN->N,K->C,PH/PF->FF,SCH->SSS; suffix IE/EE->Y,
/// DT/RT/RD/NT/ND->D. First rewritten byte is retained verbatim. Interior EV->AF,
/// vowels->A,Q->G,Z->S,M->N,KN->N,K->C,SCH->SS,PH->F and contextual H/W
/// rules use the original rewritten predecessor bytes. Whole replacement tokens
/// are suppressed by token.back()==key.back(), not per-character compression.
/// Consequently internal SS tokens/raw-predecessor W rules can differ from
/// other NYSIIS implementations; no Jellyfish/standard-conformance claim.
/// Trim one trailing S, fold AY->Y, trim one trailing A, in that order, retaining
/// a one-byte key. Empty input returns empty. All byte strings are accepted;
/// punctuation/high bytes/NUL can survive, so output need not be alphabetic.
/// O(N) ordinary work/storage; optional full snapshots cost actual captured
/// bytes, potentially O(N squared). No mutation/references retained; results
/// depend on the process C locale, which this function does not change.
/// @param name Complete original byte string, case conversion as described.
/// @param max_entries Actual owned-byte/record quota, default1000000, zero disables; admitted before each retained copy.
/// @return Owned complete actual rewrite history and final key.
/// @throws std::length_error If count/string geometry or actual capture quota is exceeded.
/// @throws std::bad_alloc If encoding/history allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::nysiis_trace("Catherine");std::print("{}\n",r.analysis.key); // CATARAN
/// @endcode
/// @see NysiisAnalysis, NysiisTrace
NysiisTrace nysiis_trace(const std::string& name, std::size_t max_entries = 1000000);
/// @brief Actual descriptive keys and collision counts for an owned input list, without pronunciation labels.
/// @par Example
/// @code
/// auto c=dm::algorithms::nysiis_compare_names({"Miller","Mueller"});std::print("{}\n",c.distinct_nysiis); // 1
/// @endcode
/// @see nysiis_trace()
struct NysiisComparison {
  std::vector<std::string> names;        ///< Owned original name strings in supplied order.
  std::vector<std::string> soundex_keys; ///< Actual canonical Soundex outputs in supplied order.
  std::vector<std::string>
              nysiis_keys; ///< Actual canonical native NYSIIS-variant outputs in supplied order.
  std::size_t distinct_soundex{
      0}; ///< Actual distinct Soundex strings, including empty if supplied.
  std::size_t   distinct_nysiis{0}; ///< Actual distinct NYSIIS-variant strings.
  std::uint64_t equal_soundex_pairs{
      0}; ///< Actual unordered equal-code row pairs; duplicates in input remain separate rows.
  std::uint64_t equal_nysiis_pairs{
      0}; ///< Actual unordered equal-key row pairs, not labeled false matches.
};
/// @brief Compute actual Soundex/NYSIIS keys and descriptive equality groups for a supplied cohort.
/// Ordinary producers determine every key; no phonetic or identity ground truth
/// is inferred from more distinct strings or fewer equal-code pairs. O(total
/// byte encoding work) plus O(N log N) variable-length key comparisons, owned input/key storage; no caller mutation.
/// @param names Original complete byte strings, including empty/duplicate rows.
/// @param max_names Complete row cap, default1000000, zero disables.
/// @return Owned original-order keys, distinct counts and checked unordered equal-row-pair counts.
/// @throws std::length_error If complete row policy or pair-count geometry is exceeded.
/// @par Example
/// @code
/// auto c=dm::algorithms::nysiis_compare_names({"Miller","Mueller"});std::print("{}\n",c.equal_nysiis_pairs); // 1
/// @endcode
/// @see NysiisComparison
NysiisComparison nysiis_compare_names(
    const std::vector<std::string>& names,
    std::size_t                     max_names = 1000000);
/// @brief Return the original32-surname illustrative cohort, with no external labels or corpus claim.
/// @return Owned original ordered name list Anderson through Robinson.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nysiis_demo_names().size()); // 32
/// @endcode
/// @see nysiis_compare_names()
std::vector<std::string> nysiis_demo_names();
/// @brief Return the original six explicitly displayed names.
/// @return MacDonald,Catherine,Ashcraft,Ashcroft,Mueller,Miller in original order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nysiis_demo_examples().size()); // 6
/// @endcode
/// @see nysiis_demo_names()
std::vector<std::string> nysiis_demo_examples();
} // namespace datamunge::algorithms
