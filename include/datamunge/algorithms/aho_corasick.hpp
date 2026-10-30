#pragma once

/// @file aho_corasick.hpp
/// @brief Aho-Corasick dictionary matching: a reusable automaton (AhoCorasick) with its states,
///        per-pattern counts and scan statistics, plus the one-shot aho_corasick_search().

#include <datamunge/dstruct/dataframe.hpp>

#include <cstddef>
#include <map>
#include <string>
#include <string_view>
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

/// @brief Work done by one scan of AhoCorasick::scan_stats(): characters read, goto (trie) edges
///        taken, failure links followed, dictionary links visited and matches reported.
///
/// Each goto step deepens the current state by one and each failure step makes it shallower, so
/// `failure_steps <= goto_steps <= characters`: the scan is linear in the text.
struct AhoCorasickScanStats {
    std::size_t characters{0};    ///< Characters read.
    std::size_t goto_steps{0};    ///< Trie edges taken.
    std::size_t failure_steps{0}; ///< Failure links followed.
    std::size_t matches{0};       ///< Occurrences reported.
};

/// @brief An Aho-Corasick automaton compiled once from a dictionary and reused for many texts.
///
/// The states are the nodes of the trie of the patterns (state 0 is the root, the empty prefix).
/// Each state @f$v@f$ has a failure link to the longest proper suffix of its prefix that is also
/// a state, and a dictionary link to the nearest state on its failure chain that completes a
/// pattern. Construction is @f$O(m)@f$ in the total pattern length; a search is @f$O(n + z)@f$ in the
/// text length and number of matches, whatever the number of patterns. Empty patterns match
/// nowhere.
///
/// @par Example
/// @code
/// const dm::algorithms::AhoCorasick ac({"he", "she", "his", "hers"});
/// for (const auto& m : ac.search("ushers")) std::print("{} at {}\n", ac.patterns()[m.pattern_index], m.position);
/// @endcode
class AhoCorasick {
public:
    /// @brief Builds the trie, failure links and dictionary links for @p patterns.
    explicit AhoCorasick(std::vector<std::string> patterns);

    /// @brief The dictionary, in the order given; match pattern indices refer to it.
    [[nodiscard]] const std::vector<std::string>& patterns() const noexcept { return patterns_; }
    /// @brief Number of automaton states, including the root (at most 1 + total pattern length).
    [[nodiscard]] std::size_t state_count() const noexcept { return nodes_.size(); }

    /// @brief Every occurrence of every pattern in @p text, ordered by end position and, at one end
    ///        position, from the longest pattern down its suffix chain.
    [[nodiscard]] std::vector<AhoCorasickMatch> search(std::string_view text) const;
    /// @brief Number of occurrences of each pattern in @p text (one entry per pattern).
    [[nodiscard]] std::vector<std::size_t> count(std::string_view text) const;
    /// @brief For each start position in @p text where a pattern occurs, the longest such pattern's
    ///        index: the leftmost-longest reading used to tag text with dictionary terms.
    [[nodiscard]] std::map<std::size_t, std::size_t> longest_at(std::string_view text) const;
    /// @brief Counts the work of one scan over @p text without storing the matches.
    [[nodiscard]] AhoCorasickScanStats scan_stats(std::string_view text) const;

    /// @brief The automaton as a table, one row per state in breadth-first order: `state`,
    ///        `prefix`, `depth`, `fail`, `fail_prefix`, `dict`, `outputs` (pattern indices ending
    ///        here, comma separated) and `edges` (the goto labels, in byte order).
    [[nodiscard]] dstruct::DataFrame automaton() const;

private:
    /// @brief One state: trie edges, failure and dictionary links, and the patterns ending here.
    struct Node {
        std::map<char, std::size_t> next; ///< Goto edges, by byte.
        std::size_t fail{0};              ///< Failure link.
        std::size_t dict{0};              ///< Dictionary link (0 when none).
        std::size_t depth{0};             ///< Length of the state's prefix.
        std::size_t parent{0};            ///< Trie parent, for rebuilding the prefix.
        char label{0};                    ///< Edge label from the parent.
        std::vector<std::size_t> output;  ///< Patterns ending exactly here.
    };
    /// @brief One transition on @p ch from @p state (failure links, then a goto edge), counting
    ///        the steps into @p stats when it is not null.
    [[nodiscard]] std::size_t step(std::size_t state, char ch, AhoCorasickScanStats* stats) const;
    /// @brief The prefix a state stands for.
    [[nodiscard]] std::string prefix(std::size_t state) const;

    std::vector<std::string> patterns_;   ///< The dictionary.
    std::vector<Node> nodes_;             ///< States; 0 is the root.
    std::vector<std::size_t> bfs_order_;  ///< States in breadth-first order.
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
///
/// Equivalent to `AhoCorasick(patterns).search(text)`; build an AhoCorasick to search many texts.
std::vector<AhoCorasickMatch> aho_corasick_search(const std::string&              text,
                                                  const std::vector<std::string>& patterns);

} // namespace datamunge::algorithms
