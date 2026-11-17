#pragma once

/// @file metaphone.hpp
/// @brief Canonical metaphone API, producing histories and checked native contracts.

#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual normalized byte stream, final native key and producing counts.
/// @details Normalization lowercases bytes with active C-locale ctype and drops the first byte of a recognized initial kn/gn/pn/wr/ae cluster. This is not UTF-8 accent normalization. The native finite rule set is retained, without claiming every historical Metaphone variant agrees.
/// @par Example
/// @code
/// auto a=dm::algorithms::metaphone_analyze("Knight");
/// std::print("{}\n",a.key); // NT
/// @endcode
/// @see metaphone, metaphone_analyze, metaphone_trace
/// @par Reference
/// Philips (1990), Hanging on the Metaphone, Computer Language7(12), December1990; native rules/conventions are specified separately.
struct MetaphoneAnalysis {
  std::string normalized; ///< Actual lowercased stream after an optional one-byte onset removal.
  std::string key;        ///< Actual uppercased final output.
  std::size_t onset_removed{0}; ///< Actual initial-byte removal, zero or one.
  std::size_t iterations{0}; ///< Actual outer context-rule evaluations, including duplicate skips.
  std::size_t consumed{0};   ///< Actual normalized input bytes consumed by those iterations.
  std::size_t emitted{0};    ///< Actual output bytes before/after final case conversion.
};

/// @brief One actual context-rule branch and its consumed/emitted spans.
/// @details Rule names identify the entered top-level branch; exact context and emitted bytes distinguish its internal cases. Emitted bytes are actual lowercase working output before final uppercase conversion.
/// @par Example
/// @code
/// auto t=dm::algorithms::metaphone_trace("ph");
/// std::print("{} {}\n",t.steps[0].end,t.steps[0].emitted); // 2 f
/// @endcode
/// @see metaphone, metaphone_analyze, metaphone_trace
/// @par Reference
/// Philips (1990), Hanging on the Metaphone, Computer Language7(12), December1990; native rules/conventions are specified separately.
struct MetaphoneStep {
  std::size_t   begin{0};        ///< Actual start in normalized stream, after onset removal.
  std::size_t   end{0};          ///< Actual past-last consumed normalized position.
  unsigned char current{0};      ///< Actual current byte.
  unsigned char next{0};         ///< Actual next byte, zero when absent or a literal NUL.
  unsigned char next2{0};        ///< Actual second lookahead byte, zero when absent or literal NUL.
  bool          has_next{false}; ///< Whether next position actually exists.
  bool          has_next2{false}; ///< Whether second lookahead position exists.
  std::string   rule;  ///< Actual entered top-level context branch or duplicate/unhandled branch.
  std::string emitted; ///< Actual working-output suffix appended by this iteration, possibly empty.
  std::size_t output_before{0}; ///< Actual output length before the branch.
  std::size_t output_after{0};  ///< Actual output length after the branch.
};

/// @brief Owned raw input and complete actual context decisions from one canonical execution.
/// @details Logical quota counts original bytes, actual normalized bytes, each step record, actual step-emitted bytes and final key bytes. Rule label characters, fields, capacities, runtime copies and bytes are not separately counted.
/// @par Example
/// @code
/// auto t=dm::algorithms::metaphone_trace("ph");
/// std::print("{}\n",t.captured_entries); // 7
/// @endcode
/// @see metaphone, metaphone_analyze, metaphone_trace
/// @par Reference
/// Philips (1990), Hanging on the Metaphone, Computer Language7(12), December1990; native rules/conventions are specified separately.
struct MetaphoneTrace {
  std::string       input;    ///< Owned original arbitrary bytes.
  MetaphoneAnalysis analysis; ///< Actual same-run normalized input, key and work.
  std::vector<MetaphoneStep>
              steps; ///< Actual producing rule evaluations, including empty-output branches.
  std::size_t captured_entries{0}; ///< Complete logical entry count.
};

/// @brief Encode bytes using datamunge's retained native Metaphone context rules.
/// @details Lowercase with active C-locale ctype, handle initial silent-onset pairs, then apply the finite native branch rules and uppercase output. Duplicate detection compares the current and next byte, except cc; some context branches consume additional bytes. Spaces have explicit handling, other unhandled bytes emit nothing but still affect context. O(N) time/storage with at most two emitted bytes per processed input byte. This is not Unicode normalization, a pronunciation oracle or a guarantee of agreement with every Metaphone variant.
/// @param name Original arbitrary bytes; ASCII English-word examples are the intended domain.
/// @return Actual native uppercase key.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::metaphone("Knight")); // NT
/// @endcode
/// @see metaphone, metaphone_analyze, metaphone_trace
/// @par Reference
/// Philips (1990), Hanging on the Metaphone, Computer Language7(12), December1990; native rules/conventions are specified separately.
std::string metaphone(const std::string& name);

/// @brief Return actual normalized bytes, final key and work from the canonical context producer.
/// @details No separate explanatory encoding is run. Retains exact ordinary byte/context conventions.
/// @param name Original arbitrary bytes.
/// @return Owned normalized stream, actual key and producing counts.
/// @par Example
/// @code
/// auto a=dm::algorithms::metaphone_analyze("ph");
/// std::print("{} {}\n",a.iterations,a.consumed); // 1 2
/// @endcode
/// @see metaphone, metaphone_analyze, metaphone_trace
/// @par Reference
/// Philips (1990), Hanging on the Metaphone, Computer Language7(12), December1990; native rules/conventions are specified separately.
MetaphoneAnalysis metaphone_analyze(const std::string& name);

/// @brief Capture every actual context branch, consumed span and appended working-output suffix.
/// @details Default complete logical cap one million; zero disables it. Admit original/normalized owned characters and each actual record/suffix before capture copies, after its producer branch is known. Final key characters are admitted before uppercase output conversion. No prefix is returned on quota failure.
/// @param name Original arbitrary bytes.
/// @param max_entries Complete logical cap, zero off.
/// @return Owned raw/normalized input, actual branch history and same-run key/counts.
/// @throws std::length_error On count geometry or complete actual capture admission failure.
/// @par Example
/// @code
/// auto t=dm::algorithms::metaphone_trace("ph",7);
/// std::print("{}\n",t.analysis.key); // F
/// @endcode
/// @see metaphone, metaphone_analyze, metaphone_trace
/// @par Reference
/// Philips (1990), Hanging on the Metaphone, Computer Language7(12), December1990; native rules/conventions are specified separately.
MetaphoneTrace metaphone_trace(const std::string& name, std::size_t max_entries = 1000000);

/// @brief Return the original twelve silent-letter/digraph-rich word fixtures.
/// @details These words have no independently labelled pronunciation truth; the fixture shows native outputs rather than recognition accuracy.
/// @return Owned original words in report order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::metaphone_demo_words().size()); // 12
/// @endcode
/// @see metaphone, metaphone_analyze, metaphone_trace
/// @par Reference
/// Philips (1990), Hanging on the Metaphone, Computer Language7(12), December1990; native rules/conventions are specified separately.
std::vector<std::string> metaphone_demo_words();

} // namespace datamunge::algorithms
