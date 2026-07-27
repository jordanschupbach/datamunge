#pragma once

#include <cstddef>
#include <string>

namespace datamunge::algorithms {

/// @brief The Levenshtein edit distance (Levenshtein 1965) between two byte strings @p a and @p b:
///        the minimum number of single-character *insertions*, *deletions*, and *substitutions*
///        that transform @p a into @p b. Equivalently it is the shortest-path cost in the edit
///        graph whose diagonal moves (matching characters) are free and whose horizontal, vertical,
///        and mismatching-diagonal moves cost one each.
///
///        Computed by the classic Wagner-Fischer dynamic program. Let @c D(i,j) be the distance
///        between the length-@c i prefix of @p a and the length-@c j prefix of @p b. Then
///        @c D(i,0)=i, @c D(0,j)=j, and for @c i,j>0
///        @c D(i,j)=min(D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+[a_i!=b_j]). This implementation keeps
///        only the two most recent rows, so it runs in @c O(|a|*|b|) time and @c O(min(|a|,|b|))
///        space. The result is a true metric: it is symmetric, zero iff the strings are equal, and
///        satisfies the triangle inequality.
///
/// @param a the first string.
/// @param b the second string.
/// @return the Levenshtein distance between @p a and @p b (0 iff they are equal).
std::size_t levenshtein_distance(const std::string& a, const std::string& b);

/// @brief The (true, unrestricted) Damerau-Levenshtein distance between @p a and @p b: the minimum
///        number of insertions, deletions, substitutions, *and transpositions of two adjacent
///        characters* that transform @p a into @p b. Adding the transposition operation makes
///        neighbouring-letter typos (e.g. "ba" -> "ab") cost one instead of two, which is why the
///        measure is favoured for spell-checking and record linkage.
///
///        This is the Lowrance-Wagner algorithm for the *unrestricted* distance -- a full metric in
///        which a substring may take part in more than one transposition -- not the simpler
///        "optimal string alignment" (OSA) restriction, which forbids editing a substring twice and
///        can therefore report a larger value (its classic witness is @c CA <-> @c ABC: this
///        function returns 2, OSA returns 3). It runs in @c O(|a|*|b|) time using a full
///        @c (|a|+2)x(|b|+2) table plus a 256-entry "last row at which each byte occurred" array.
///
/// @param a the first string.
/// @param b the second string.
/// @return the unrestricted Damerau-Levenshtein distance between @p a and @p b.
std::size_t damerau_levenshtein_distance(const std::string& a, const std::string& b);

/// @brief The Hamming distance between two byte strings of *equal length*: the number of positions
///        at which the corresponding characters differ. Unlike edit distance it counts only
///        substitutions -- there is no insertion or deletion -- so it is defined only when
///        @c |a|==|b|.
///
/// @param a the first string.
/// @param b the second string; must have the same length as @p a.
/// @return the number of positions where @p a and @p b differ.
/// @throws std::invalid_argument if @p a and @p b have different lengths.
std::size_t hamming_distance(const std::string& a, const std::string& b);

/// @brief The Jaro similarity (Jaro 1989) between @p a and @p b, a value in @c [0,1] where 1 means
///        the strings are identical and 0 that they share no characters. It is defined as
///        @c 0 when there are no matching characters, and otherwise
///        @c (m/|a| + m/|b| + (m-t)/m)/3, where @c m is the number of *matching* characters -- equal
///        characters no farther apart than @c floor(max(|a|,|b|)/2)-1 positions, each matched at
///        most once -- and @c t is *half* the number of matched characters that occur out of order
///        (the number of transpositions). Both empty strings are defined to have similarity 1.
///
/// @param a the first string.
/// @param b the second string.
/// @return the Jaro similarity in @c [0,1].
double jaro_similarity(const std::string& a, const std::string& b);

/// @brief The Jaro-Winkler similarity (Winkler 1990): the Jaro similarity boosted for strings that
///        share a common prefix, which empirically improves matching of human names. With Jaro
///        score @c j, common-prefix length @c l capped at 4, and scaling factor @p p (the standard
///        value 0.1), it returns @c j + l*p*(1-j). The boost is applied unconditionally here (some
///        variants apply it only when @c j exceeds a "boost threshold"); the result stays within
///        @c [0,1] as long as @c l*p<=1, which holds for the default @c p=0.1.
///
/// @param a the first string.
/// @param b the second string.
/// @param prefix_scale the scaling factor @c p for the prefix bonus (default 0.1).
/// @return the Jaro-Winkler similarity in @c [0,1].
double jaro_winkler_similarity(const std::string& a, const std::string& b, double prefix_scale = 0.1);

/// @brief The Sorensen-Dice coefficient over character *bigrams*: a similarity in @c [0,1] equal to
///        @c 2|A∩B| / (|A|+|B|), where @c A and @c B are the multisets of adjacent character pairs
///        (bigrams) of @p a and @p b and the intersection counts shared bigrams with multiplicity.
///        It is the harmonic-mean-flavoured twin of the Jaccard index and is popular for fuzzy
///        string search because bigrams capture local letter order while tolerating edits elsewhere.
///
///        Following Dice's original word-similarity measure, strings shorter than two characters
///        have no bigrams: two identical strings of length < 2 are defined to have coefficient 1,
///        and a length-<2 string compared against any different string has coefficient 0.
///
/// @param a the first string.
/// @param b the second string.
/// @return the Dice coefficient in @c [0,1] (1 iff the bigram multisets are equal).
double dice_coefficient(const std::string& a, const std::string& b);

} // namespace datamunge::algorithms
