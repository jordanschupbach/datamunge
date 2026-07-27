#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief The *Fisher-Yates shuffle* (Fisher & Yates 1938; Durstenfeld's modern in-place form 1964,
///        popularized by Knuth), which returns a *uniformly random* permutation of @p items: every
///        one of the @c n! orderings is equally likely. Iterating @c i from the last index down to
///        1, it swaps element @c i with a uniformly chosen element in @c [0, i]; because each swap
///        draws from an interval that shrinks by one, the map from the @c n! possible draw sequences
///        to permutations is a bijection, which is exactly what makes the result unbiased. Runs in
///        @c O(n) time using a single pass and the library's seedable =random::SplitMix64= engine,
///        so a given @p seed reproduces the same permutation.
///
/// @param items the elements to shuffle (any values; duplicates are permitted and preserved).
/// @param seed  the seed for the internal SplitMix64 generator; equal seeds give equal results.
/// @return a permutation of @p items.
std::vector<int> fisher_yates_shuffle(const std::vector<int>& items, std::uint64_t seed);

/// @brief *Heap's algorithm* (B. R. Heap, 1963): generates *all* @c n! permutations of @p items,
///        each obtained from the previous by a *single swap of two elements*, so the total number of
///        swaps is @c n!-1 -- the minimum possible. The returned list is in Heap's generation order.
///        Runs in @c O(n!) time (and returns @c O(n * n!) data); intended for small @c n.
///
/// @param items the elements to permute.
/// @return all @c n! permutations of @p items in Heap's order (a single permutation, itself, if
///         @p items has 0 or 1 element).
std::vector<std::vector<int>> heap_permutations(const std::vector<int>& items);

/// @brief The *Steinhaus-Johnson-Trotter algorithm*: generates all @c n! permutations of
///        @c {1,2,...,n} such that *consecutive permutations differ by a single transposition of two
///        adjacent entries* -- a combinatorial Gray code for permutations. Each element carries a
///        direction; the algorithm repeatedly moves the largest "mobile" element (one whose
///        direction points at a smaller neighbour) and reverses the direction of every larger
///        element. Runs in @c O(n!) time (returning @c O(n * n!) data); intended for small @c n.
///
/// @param n the size of the permuted set @c {1,...,n}.
/// @return all @c n! permutations of @c {1,...,n} in Steinhaus-Johnson-Trotter (adjacent-transposition)
///         order; a single empty permutation when @p n is 0.
std::vector<std::vector<int>> sjt_permutations(std::size_t n);

/// @brief A pair of Young tableaux: the *insertion* tableau @ref p and the *recording* tableau
///        @ref q, produced by the Robinson-Schensted correspondence. Both are lists of rows of
///        weakly decreasing length; @ref p holds the permutation's values arranged so that rows
///        increase left to right and columns increase top to bottom, and @ref q (a standard tableau
///        on @c 1..n) records the step at which each box was added, so the two always have the same
///        shape.
struct YoungTableaux {
    std::vector<std::vector<int>> p; ///< insertion tableau (the permutation's values).
    std::vector<std::vector<int>> q; ///< recording tableau (the insertion step of each box).
};

/// @brief The *Schensted algorithm* / Robinson-Schensted correspondence: row-inserts the entries of
///        a @p permutation one by one to build the insertion tableau @c P, recording each new box's
///        birth step in the recording tableau @c Q. Inserting @c x into a row replaces the leftmost
///        entry strictly greater than @c x (which is "bumped" into the next row); if no entry is
///        greater, @c x is appended and a new box is recorded in @c Q. The correspondence is a
///        bijection between permutations of @c {1,...,n} and pairs of standard Young tableaux of the
///        same shape, and (Schensted's theorem) the length of @c P's first row equals the length of
///        the longest increasing subsequence of @p permutation, while the number of rows equals the
///        longest decreasing subsequence. Runs in @c O(n^2) time (naive row scans).
///
/// @param permutation a sequence of distinct values (typically a permutation of @c {1,...,n}).
/// @return the insertion/recording tableau pair.
YoungTableaux rsk_insert(const std::vector<int>& permutation);

} // namespace datamunge::algorithms
