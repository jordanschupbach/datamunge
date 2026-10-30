#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief The result of @ref kadane: the maximum sum over any non-empty contiguous subarray and the
///        half-open index range @c [begin, end) achieving it.
struct MaxSubarray {
    double      sum{0.0}; ///< the maximum contiguous-subarray sum.
    std::size_t begin{0}; ///< first index of the best subarray (inclusive).
    std::size_t end{0};   ///< one past the last index of the best subarray (exclusive).
};

/// @brief *Kadane's algorithm* (Bentley 1984, after Kadane): finds the contiguous subarray of @p a
///        with the largest sum, in a single \(O(n)\) pass. It maintains the best sum of a subarray
///        *ending at the current position* -- either extend the previous best or start fresh here --
///        and tracks the global maximum. Correct with negative values: for an all-negative input it
///        returns the single least-negative element (the best non-empty subarray). An empty input
///        yields sum 0 over the empty range @c [0,0).
///
/// @param a the input values (may be negative).
/// @return the maximum-sum subarray and its index range.
MaxSubarray kadane(const std::vector<int>& a);

/// @brief The *longest common substring* of two sequences @p a and @p b: a longest run of elements
///        that occurs *contiguously* in both (unlike a subsequence, which may skip elements).
///        Computed by the dynamic program @c L(i,j)=L(i-1,j-1)+1 when @c a_i=b_j and 0 otherwise --
///        the length of the common suffix of the two prefixes -- tracking the largest value seen.
///        Runs in \(O(|a||b|)\) time and space. When several longest common substrings exist, one
///        is returned.
///
/// @param a the first sequence.
/// @param b the second sequence.
/// @return one longest common (contiguous) substring; empty if the sequences share no element.
std::vector<int> longest_common_substring(const std::vector<int>& a, const std::vector<int>& b);

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
bool wildcard_match(const std::string& text,const std::string& pattern);

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
    std::string                text;
    std::vector<SuffixTreeNode> nodes;
};

/// @brief *Ukkonen's algorithm* (Ukkonen 1995): constructs the @ref SuffixTree of @p text in
///        *linear time*, *online* (processing the text left to right, one character at a time).
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

} // namespace datamunge::algorithms
