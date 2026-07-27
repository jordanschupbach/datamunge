#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief One reported occurrence: pattern @ref pattern_index occurs in the text starting at
///        @ref position (a zero-based index into the searched text).
struct AhoCorasickMatch {
    /// @brief Start index in the text of this occurrence.
    std::size_t position;
    /// @brief Index into the @p patterns vector of the pattern that occurs here.
    std::size_t pattern_index;
};

/// @brief The Aho-Corasick multi-pattern string-matching algorithm (Aho & Corasick 1975). Given a
///        @p text and a whole *set* of @p patterns, it finds *every* occurrence of *every* pattern
///        in a single left-to-right pass over the text, rather than running a separate single-pattern
///        search per pattern (which would cost O(|text| * #patterns)).
///
///        It builds a finite-state automaton from the patterns. The states are the nodes of a
///        *trie* (prefix tree) of all patterns; the *goto* function is the trie's child edges. To
///        each state it adds a *failure link* -- pointing to the state for the longest proper suffix
///        of the current matched prefix that is itself a trie node -- exactly the KMP prefix
///        function generalized from a single string to a trie. It also adds *output* (a.k.a.
///        dictionary-suffix) links, which chain a state to the nearest failure-reachable state that
///        completes some pattern, so that patterns which are suffixes of the current match (and of
///        one another) are all reported. Scanning the text then follows goto edges where possible
///        and failure links where not, emitting a match for every pattern that ends at the current
///        character.
///
///        Preprocessing is O(sum of pattern lengths); the scan is O(|text| + #matches). All
///        occurrences are reported, including overlapping ones and patterns that are suffixes of
///        other patterns.
///
/// @param text the string to search within.
/// @param patterns the dictionary of patterns to search for; the @ref AhoCorasickMatch::pattern_index
///        fields in the result index into this vector.
/// @return every occurrence as an @ref AhoCorasickMatch. Matches are produced in order of increasing
///         end position (i.e. by the character at which they complete), and within a single end
///         position from the longest pattern down its suffix chain to the shortest. Duplicate
///         patterns (equal strings at different indices) are each reported under their own index. By
///         convention -- mirroring the library's other string searchers -- an empty pattern matches
///         nowhere and is skipped; if @p patterns is empty the result is empty.
std::vector<AhoCorasickMatch> aho_corasick_search(const std::string&              text,
                                                  const std::vector<std::string>& patterns);

} // namespace datamunge::algorithms
