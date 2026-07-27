#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief A linear (affine-free) scoring scheme for character alignment: each aligned pair of equal
///        characters scores @ref match, each aligned pair of unequal characters scores @ref
///        mismatch, and each character aligned against a gap scores @ref gap. By convention @ref gap
///        (and usually @ref mismatch) are negative penalties. The defaults (+2 / -1 / -2) are a
///        common general-purpose choice.
struct AlignmentScoring {
    int match{2};     ///< score added for aligning two equal characters.
    int mismatch{-1}; ///< score added for aligning two unequal characters.
    int gap{-2};      ///< score added for aligning a character against a gap ('-').
};

/// @brief The result of a pairwise character alignment: the optimal @ref score under the scoring
///        scheme, and the two input strings written with '-' gap characters inserted so that they
///        have equal length and can be read column by column. Removing the gaps from @ref a_aligned
///        recovers (a substring of, for local alignment) the first input, and likewise for
///        @ref b_aligned.
struct Alignment {
    int         score{0};   ///< optimal alignment score.
    std::string a_aligned;  ///< first string with gap characters inserted.
    std::string b_aligned;  ///< second string with gap characters inserted.
};

/// @brief The Needleman-Wunsch algorithm (Needleman & Wunsch, 1970) for optimal *global* alignment:
///        it aligns the *entire* strings @p a and @p b end to end, maximizing the total score under
///        @p scoring. It fills an @c (|a|+1) x (|b|+1) dynamic-programming table @c H where
///        @c H(i,j) is the best score aligning the length-@c i prefix of @p a with the length-@c j
///        prefix of @p b, using the recurrence
///        @c H(i,j)=max(H(i-1,j-1)+s(a_i,b_j), H(i-1,j)+gap, H(i,j-1)+gap) with fully-penalized
///        boundaries @c H(i,0)=i*gap, @c H(0,j)=j*gap; a traceback from the bottom-right corner
///        reconstructs the alignment. Runs in @c O(|a|*|b|) time and space.
///
/// @param a the first string.
/// @param b the second string.
/// @param scoring the match/mismatch/gap scoring scheme.
/// @return the optimal global @ref Alignment.
Alignment needleman_wunsch(const std::string& a, const std::string& b, AlignmentScoring scoring = {});

/// @brief The Smith-Waterman algorithm (Smith & Waterman, 1981) for optimal *local* alignment: it
///        finds the highest-scoring pair of *substrings* of @p a and @p b, rather than aligning the
///        strings in full. It uses the Needleman-Wunsch recurrence with one change -- every cell is
///        floored at zero, @c H(i,j)=max(0, H(i-1,j-1)+s, H(i-1,j)+gap, H(i,j-1)+gap) -- so an
///        alignment may start afresh anywhere; the traceback begins at the maximum cell and stops at
///        the first zero. Non-matching flanks are simply excluded (never penalized), which is why a
///        local score can exceed the global score of the same strings. Runs in @c O(|a|*|b|) time
///        and space.
///
///        If no positive-scoring alignment exists the result has @c score 0 and empty aligned
///        strings.
///
/// @param a the first string.
/// @param b the second string.
/// @param scoring the match/mismatch/gap scoring scheme.
/// @return the optimal local @ref Alignment (aligned fields hold the matching substrings).
Alignment smith_waterman(const std::string& a, const std::string& b, AlignmentScoring scoring = {});

/// @brief Hirschberg's algorithm (Hirschberg, 1975): computes the *same* optimal global alignment as
///        Needleman-Wunsch but in @c O(min(|a|,|b|)) space instead of @c O(|a|*|b|), by divide and
///        conquer. It splits @p a at its midpoint, uses the linear-space "score-only" forward and
///        reverse Needleman-Wunsch passes to find the column of @p b at which an optimal alignment
///        crosses that midpoint, then recurses on the two halves. The running time stays
///        @c O(|a|*|b|). The returned score is identical to @ref needleman_wunsch; the reconstructed
///        alignment is an optimal one (which specific optimum may differ when ties exist).
///
/// @param a the first string.
/// @param b the second string.
/// @param scoring the match/mismatch/gap scoring scheme.
/// @return an optimal global @ref Alignment, computed in linear space.
Alignment hirschberg(const std::string& a, const std::string& b, AlignmentScoring scoring = {});

// Note: Dynamic Time Warping, the fourth member of the "sequence alignment" family, lives in the
// geometry module as datamunge::geometry::dynamic_time_warping (it aligns numeric series, not
// character strings). See include/datamunge/geometry/dtw.hpp.

} // namespace datamunge::algorithms
