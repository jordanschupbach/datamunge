#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief The *longest common subsequence* (LCS) of two sequences @p a and @p b: a longest sequence
///        that appears in both as a subsequence (its elements occur in order, but not necessarily
///        contiguously). It measures similarity ignoring insertions on either side and underlies the
///        Unix @c diff utility and version-control merges. Computed by the classic Wagner-Fischer
///        dynamic program @c C(i,j)=C(i-1,j-1)+1 when @c a_i=b_j, else @c max(C(i-1,j),C(i,j-1)),
///        with a traceback reconstructing one optimal subsequence. Runs in @c O(|a|*|b|) time and
///        space. When several LCSs of equal length exist, one of them is returned.
///
/// @param a the first sequence.
/// @param b the second sequence.
/// @return one longest common subsequence (empty if the sequences share no element in order).
std::vector<int> longest_common_subsequence(const std::vector<int>& a, const std::vector<int>& b);

/// @brief The *longest strictly increasing subsequence* (LIS) of @p a: a longest subsequence whose
///        values strictly increase. Computed by patience sorting -- maintaining, for each length
///        @c k, the smallest possible tail value of an increasing subsequence of that length, and
///        binary-searching each element into that structure -- with predecessor links so an actual
///        subsequence (not just its length) is reconstructed. Runs in @c O(|a| log |a|) time. When
///        several LISs of equal length exist, one of them is returned.
///
/// @param a the sequence.
/// @return one longest strictly increasing subsequence.
std::vector<int> longest_increasing_subsequence(const std::vector<int>& a);

/// @brief The *shortest common supersequence* (SCS) of two sequences @p a and @p b: a shortest
///        sequence that contains both @p a and @p b as subsequences. Its length is exactly
///        @c |a|+|b|-|LCS(a,b)|, and it is built by walking the LCS dynamic-programming table,
///        emitting shared elements once and the private elements of each side in order. Runs in
///        @c O(|a|*|b|) time and space. When several SCSs of equal length exist, one is returned.
///
/// @param a the first sequence.
/// @param b the second sequence.
/// @return one shortest common supersequence (contains both @p a and @p b as subsequences).
std::vector<int> shortest_common_supersequence(const std::vector<int>& a, const std::vector<int>& b);

/// @brief One maximal-scoring segment reported by @ref ruzzo_tompa: the half-open index range
///        @c [begin, end) into the input and its total @ref score (which is strictly positive).
struct ScoringSegment {
    std::size_t begin; ///< first index of the segment (inclusive).
    std::size_t end;   ///< one past the last index of the segment (exclusive).
    double      score; ///< total score of the segment (sum of its values; always > 0).
};

/// @brief The *Ruzzo-Tompa algorithm* (Ruzzo & Tompa, 1999): given a sequence of real @p scores
///        (typically a mix of positive and negative values), find *all* maximal-scoring contiguous
///        subsequences in a single linear-time pass. A subsequence is *maximal scoring* if it has
///        positive total score, no proper sub-subsequence scores higher, and it is not contained in
///        a longer subsequence with those same properties -- so the result is a set of disjoint
///        "islands" of signal, the tool of choice for finding biologically significant regions in a
///        scored genomic sequence. The algorithm incrementally integrates each positive score into
///        an ordered list of candidate segments, merging leftward whenever doing so raises the
///        score. Runs in @c O(n) time.
///
/// @param scores the per-position scores (may be negative).
/// @return the maximal-scoring segments, in left-to-right order; empty if no positive-scoring
///         subsequence exists.
std::vector<ScoringSegment> ruzzo_tompa(const std::vector<double>& scores);

} // namespace datamunge::algorithms
