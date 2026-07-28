#pragma once

#include <string>

namespace datamunge::algorithms {

/// @brief *Soundex* (American Soundex): the classic phonetic index that maps a name to its first
///        letter followed by three digits encoding the sounds of the remaining consonants, so that
///        names that sound alike collapse to the same code (@c Robert and @c Rupert both give
///        @c R163). Consonants are grouped by articulation (labials @c b,f,p,v -> 1; gutturals
///        @c c,g,j,k,q,s,x,z -> 2; etc.); vowels and @c y are not coded but *separate* equal-coded
///        consonants, while @c h and @c w are transparent (they do not). Adjacent equal codes are
///        collapsed. The result is always one letter plus three digits, zero-padded or truncated.
///
/// @param name the input string (letters only are considered; case-insensitive).
/// @return the four-character Soundex code, or the empty string for empty input.
std::string soundex(const std::string& name);

/// @brief *NYSIIS* (New York State Identification and Intelligence System): a phonetic encoder that
///        improves on Soundex by keeping an alphabetic key (not digits) and applying a richer set of
///        context-sensitive rewrite rules to prefixes (@c MAC->MCC, @c KN->N, @c PH->FF, ...),
///        suffixes (@c EE,IE->Y, @c DT,RT,RD,NT,ND->D), and interior letters (vowels->A, @c Q->G,
///        @c Z->S, @c M->N, @c EV->AF, silent @c H/W folded into neighbours). Consecutive duplicate
///        code letters are collapsed and a trailing @c S / @c A is trimmed. It captures far more of a
///        name's structure than Soundex's three digits.
///
/// @param name the input string (case-insensitive).
/// @return the NYSIIS key (variable length), or the empty string for empty input.
std::string nysiis(const std::string& name);

/// @brief *Metaphone* (Lawrence Philips, 1990): a phonetic algorithm that transcribes a word into a
///        consonant "sound key" using 16 context rules that model English pronunciation far more
///        faithfully than Soundex -- silent letters are dropped (@c Knight -> @c NT), @c TH becomes
///        @c 0 (theta), @c SH/@c -TIA-/@c -CH- become @c X, soft @c C/@c G become @c S/@c J, and so
///        on. Vowels are kept only when they begin the word. The output alphabet is
///        @c {B,X,S,K,J,T,F,H,L,M,N,P,R,0,W,Y} plus the retained leading vowel.
///
/// @param name the input word (case-insensitive; accents are normalised away).
/// @return the Metaphone key (variable length).
std::string metaphone(const std::string& name);

/// @brief The *Match Rating Approach* codex (G. B. Moore et al., Western Airlines, 1977): compresses
///        a name by deleting vowels (except a leading one) and collapsing doubled consonants, then --
///        if more than six letters remain -- keeping only the first three and last three. This codex
///        is the input to @ref match_rating_comparison.
///
/// @param name the input string (case-insensitive; spaces ignored).
/// @return the MRA codex (at most six letters).
std::string match_rating_codex(const std::string& name);

/// @brief The result of a *Match Rating Approach* comparison of two names: whether they were even
///        @ref comparable (their codex lengths must differ by less than 3), the resulting
///        @ref similarity score (6 minus the larger count of unmatched codex letters after stripping
///        common letters from both ends), the length-dependent @ref min_rating threshold, and whether
///        the pair @ref match (similarity >= threshold).
struct MatchRatingResult {
    bool comparable{false}; ///< false if the codex lengths differ by 3 or more (no verdict).
    int  similarity{0};     ///< 6 minus the larger unmatched-letter count.
    int  min_rating{0};     ///< the minimum similarity required, set by the summed codex lengths.
    bool match{false};      ///< whether @ref similarity meets @ref min_rating (and @ref comparable).
};

/// @brief The *Match Rating Approach* comparison: encode both names with @ref match_rating_codex,
///        reject the pair outright if the codex lengths differ by 3 or more, otherwise strip letters
///        common to both codexes (scanning once from the left, once from the right), count the
///        leftover letters, and declare a match when @c 6 minus the larger leftover count meets a
///        threshold that itself relaxes as the names get longer. It is a fast, forgiving name matcher
///        tolerant of vowel changes, transpositions, and minor spelling drift.
///
/// @param a,b the two names to compare (case-insensitive).
/// @return a @ref MatchRatingResult describing comparability, score, threshold, and verdict.
MatchRatingResult match_rating_comparison(const std::string& a, const std::string& b);

} // namespace datamunge::algorithms
