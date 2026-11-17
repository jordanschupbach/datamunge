#pragma once
/// @file soundex.hpp
/// @brief Actual deterministic American-Soundex byte transitions and cohort counts.
#include <cstddef>
#include <map>
#include <string>
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief SoundexOptions owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SoundexOptions value{};
/// std::print("{}\n",value.max_bytes); // 1000000
/// @endcode
struct SoundexOptions {
  std::size_t max_bytes{1000000}; ///< Complete input byte count, even unused suffix; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual captured records plus copied partial-key/action bytes; zero disables.
};
/// @brief SoundexStep owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SoundexStep value{};
/// std::print("{}\n",value.offset); // 0
/// @endcode
struct SoundexStep {
  std::size_t
      offset{}; ///< Actual consumed byte position; pad record uses number of bytes actually consumed.
  unsigned char original{};  ///< Actual byte; zero is a pad marker when action=pad.
  char          uppercase{}; ///< Actual ASCII uppercase byte; non-ASCII bytes unchanged.
  char          group{};     ///< Actual consonant class1..6 or0.
  char          previous{};  ///< Actual previous consonant class before this transition.
  std::string action;  ///< Actual retain-first, append, suppress, transparent, reset or pad branch.
  std::string partial; ///< Actual key after this transition.
};
/// @brief SoundexRun owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SoundexRun value{};
/// std::print("{}\n",value.scanned_bytes); // 0
/// @endcode
struct SoundexRun {
  std::string code;             ///< Actual four-byte key, or empty for empty input.
  std::size_t scanned_bytes{};  ///< Actual consumed prefix bytes; stops once key has four bytes.
  std::size_t emitted_digits{}; ///< Actual appended consonant digits, excluding zero padding.
  std::size_t padding{};        ///< Actual final zero padding count.
  std::vector<SoundexStep> history; ///< Optional actual byte transitions and final pad state.
};
/// @brief Encode the actual American-Soundex byte rules.
/// ASCII letters case-insensitive. Preserve uppercase first byte, collapse repeated consonant classes, let H/W be transparent, and reset the preceding class on vowels/Y and other uncoded bytes. Groups BFPV1,CGJKQSXZ2,DT3,L4,MN5,R6. Stop after three emitted digits and pad zeros. Empty input gives empty key. Preserves historical byte behavior for punctuation/digits: first byte is retained even if nonalphabetic; later uncoded bytes reset. This is not letter filtering, transliteration, Unicode normalization or a pronunciation guarantee. Non-ASCII bytes remain unchanged and uncoded, independent of locale; language interpretation is specified only for ASCII names. O(consumed prefix) processing after complete size check, constant working state without optional histories; outputs own strings, no shared state.
/// @param name Complete borrowed raw surname bytes.
/// @param options Complete byte and optional actual capture policies; zero disables policies.
/// @return Actual owned key, consumed-byte/digit/padding counts and optional branches.
/// @throws std::length_error Complete byte/history policy or counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::soundex_run("Ashcraft");
/// std::print("{}\n",r.code); // A261
/// @endcode
/// @par References
/// - US National Archives, The Soundex Indexing System, https://www.archives.gov/research/census/soundex.
/// @see soundex()
SoundexRun soundex_run(const std::string& name, const SoundexOptions& options = {});
/// @brief Encode the actual American-Soundex byte rules with native transition states.
/// ASCII letters case-insensitive. Preserve uppercase first byte, collapse repeated consonant classes, let H/W be transparent, and reset the preceding class on vowels/Y and other uncoded bytes. Groups BFPV1,CGJKQSXZ2,DT3,L4,MN5,R6. Stop after three emitted digits and pad zeros. Empty input gives empty key. Preserves historical byte behavior for punctuation/digits: first byte is retained even if nonalphabetic; later uncoded bytes reset. This is not letter filtering, transliteration, Unicode normalization or a pronunciation guarantee. Non-ASCII bytes remain unchanged and uncoded, independent of locale; language interpretation is specified only for ASCII names. O(consumed prefix) processing after complete size check, constant working state without optional histories; outputs own strings, no shared state.
/// @param name Complete borrowed raw surname bytes.
/// @param options Complete byte and optional actual capture policies; zero disables policies.
/// @return Actual owned key, consumed-byte/digit/padding counts and optional branches.
/// @throws std::length_error Complete byte/history policy or counter/container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::soundex_trace("Ashcraft");
/// std::print("{}\n",r.code); // A261
/// @endcode
/// @par References
/// - US National Archives, The Soundex Indexing System, https://www.archives.gov/research/census/soundex.
/// @see soundex()
SoundexRun soundex_trace(const std::string& name, const SoundexOptions& options = {});
/// @brief Return the preserved American-Soundex byte key.
/// Same actual producer without byte policy or retained history. Deterministic ASCII uppercase replaces locale-dependent non-ASCII uppercasing. No letter filtering; raw nonletter behavior is explicitly documented in soundex_run.
/// @param name Complete borrowed raw surname bytes.
/// @return Owned four-byte key or empty for empty input.
/// @throws std::length_error Intrinsic counter/container capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::soundex("Robert")); // R163
/// @endcode
/// @see soundex_run()
std::string soundex(const std::string& name);
/// @brief Return the original twenty surname spellings.
/// Owned deterministic demonstration names, not a representative population or recorded4000-word reference corpus.
/// @return Complete original ordered surname list.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::soundex_example_names().size()); // 20
/// @endcode
/// @see soundex_groups()
std::vector<std::string> soundex_example_names();
/// @brief Group complete supplied names by actual canonical Soundex keys.
/// Applies the same actual encoder to each name and appends it to its key bucket. Keys sort lexicographically; names retain input order, duplicates remain. At most one million rows. Output owns names; no borrowed references or shared state.
/// @param names Complete borrowed names to group.
/// @return Owned map from actual key to original name rows.
/// @throws std::length_error Input-row or container capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::soundex_groups({"Robert","Rupert"}).size()); // 1
/// @endcode
/// @see soundex()
std::map<std::string, std::vector<std::string>> soundex_groups(
    const std::vector<std::string>& names);
} // namespace datamunge::algorithms
