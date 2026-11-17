#include <datamunge/plot/plot.hpp>
#pragma once
#include <cstddef>
#include <datamunge/algorithms/kadane.hpp>
#include <datamunge/algorithms/longest_common_substring.hpp>
#include <map>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Case-sensitive whole-string byte matching with only star and question-mark metacharacters.
///
/// '*' matches any byte sequence, including empty; '?' matches exactly one
/// byte. Other bytes, including slash, dot, brackets and backslash, are literal.
/// There is no escaping, bracket class, path flag, case folding or Unicode
/// character decoding. Embedded NULs are supported. A pattern star remains a
/// wildcard even when the current text byte is itself '*'. Empty text matches
/// only an empty pattern or a sequence of stars. Inputs are borrowed and unchanged.
/// The shared last-star scan uses O(1) auxiliary storage and O((n+1)*(m+1))
/// worst-case time for n text bytes and m pattern bytes; it can rescan suffixes.
/// Concurrent calls share no state.
/// @param text Byte string to match in its entirety.
/// @param pattern Restricted glob pattern; no substring search is implied.
/// @return True exactly when the complete text is accepted.
/// @par Example
/// @code
/// const bool match=dm::algorithms::wildcard_match("*ba","*a"); // true
/// const bool anchored=dm::algorithms::wildcard_match("surname, Mr. Name","Mr.*"); // false
/// @endcode
/// @par References
/// - K. J. Krauss, Matching Wildcards: An Improved Algorithm for Big Data,
///   https://developforperformance.com/MatchingWildcards_AnImprovedAlgorithmForBigData.html
///   (non-recursive matching family; this is a simpler length-aware last-star scan).
/// @see wildcard_match_stats(), wildcard_match_trace(), wildcard_dp_table()
bool wildcard_match(const std::string& text, const std::string& pattern);

/// @brief One node of a @ref SuffixTree. An edge into this node spans text positions
///        @c [start, end] inclusive; an @ref end of -1 is the sentinel "current leaf end", i.e. the
///        edge runs to the end of the text (a growing leaf). @ref suffix_link is the Ukkonen suffix
///        link (or -1), and @ref children maps a character code (0..255, or 256 for the virtual
///        terminal) to a child node index.
struct SuffixTreeNode {
  int                start{-1};
  int                end{-1};
  int                suffix_link{-1};
  std::map<int, int> children;
};

/// @brief A suffix tree over a text: the compressed trie of all its suffixes. @ref text is the
///        original string (a unique terminal is handled virtually at index @c text.size()), and
///        @ref nodes holds the tree with @c nodes[0] the root. Build it with @ref build_suffix_tree.
struct SuffixTree {
  std::string                 text;
  std::vector<SuffixTreeNode> nodes;
};

/// @brief Actual Ukkonen construction operation counters.
/// @par Example
/// @code
/// dm::algorithms::SuffixTreeWork r;
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
struct SuffixTreeWork {
  SuffixTree               tree;              ///< Owned final compressed suffix tree.
  std::size_t              phases{0};         ///< Actual phases, including virtual terminal.
  std::size_t              extensions{0};     ///< Actual active-point loop iterations.
  std::size_t              splits{0};         ///< Actual edge splits.
  std::size_t              walk_downs{0};     ///< Actual skip/count hops.
  std::vector<std::size_t> nodes_after_phase; ///< Actual node count at each phase end.
};
/// @brief Build a byte suffix tree and expose actual construction work.
/// Virtual terminal256 is outside byte alphabet. std::map transitions give O(n log alphabet) construction and O(m log alphabet) lookup; alphabet bounded257. No recursion or shared mutable state.
/// @param text Borrowed arbitrary byte string; at most1000000 bytes.
/// @return Owned final tree and actual counters.
/// @throws std::length_error More than1000000 text bytes.
/// @par Example
/// @code
/// auto r=dm::algorithms::build_suffix_tree_work("banana"); //15 distinct substrings
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
SuffixTreeWork build_suffix_tree_work(const std::string& text);
/// @brief *Ukkonen's algorithm* (Ukkonen 1995): constructs the @ref SuffixTree of @p text in
///        *O(n log alphabet) time* with ordered transitions, *online* (processing the text left to right, one character at a time).
///        It maintains an "active point" and suffix links so that all suffix insertions for a new
///        character are performed in amortized constant time, using the tricks of a global growing
///        leaf end, skip/count edge walking, and deferred suffix-link creation. The suffix tree
///        supports \(O(m)\) substring queries and linear-time computation of many string
///        statistics. A unique terminal symbol is appended virtually so that every suffix ends at a
///        distinct leaf.
///
/// @param text the string to index.
/// @return the suffix tree of @p text.
SuffixTree build_suffix_tree(const std::string& text);

/// @brief Tests whether @p pattern occurs as a substring of the text indexed by @p tree, in
///        \(O(|pattern|)\) time, by walking @p pattern down the edges from the root.
///
/// @param tree    a suffix tree from @ref build_suffix_tree.
/// @param pattern the string to look for.
/// @return true iff @p pattern is a (contiguous) substring of the tree's text (the empty pattern is
///         trivially present).
bool suffix_tree_contains(const SuffixTree& tree, const std::string& pattern);

/// @brief Counts the number of *distinct non-empty substrings* of the tree's text, in time linear
///        in the tree size. It sums the character lengths of all edges (each root-to-position path
///        is a distinct substring) and subtracts the one terminal character carried by each leaf
///        edge.
///
/// @param tree a suffix tree from @ref build_suffix_tree.
/// @return the number of distinct non-empty substrings of the text.
std::size_t distinct_substring_count(const SuffixTree& tree);

/// @brief Draw an actual canonical suffix tree without reconstructing it.
/// Iterative layout uses leaf order and descendant means. Byte labels suit printable input; use numeric diagnostics for binary strings. O(nodes+total displayed edge bytes) output; caller saves figure.
/// @param tree Unmodified result of build_suffix_tree; empty text accepted.
/// @return Owned plot with actual edge substrings; virtual terminal rendered as $.
/// @par Example
/// @code
/// auto p=dm::algorithms::suffix_tree_plot(dm::algorithms::build_suffix_tree("banana"));
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
graphics::RPlot suffix_tree_plot(const SuffixTree& tree);
} // namespace datamunge::algorithms
