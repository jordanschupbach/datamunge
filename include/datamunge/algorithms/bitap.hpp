#pragma once

#include <string>

namespace datamunge::algorithms {

/// @brief The *Bitap* (a.k.a. *shift-and* / Baeza-Yates-Gonnet) *exact* string search: it keeps the
///        set of pattern-prefix lengths that currently match the text as a *bitmask* in a single
///        machine word, and advances all of them at once with one shift and one AND per text
///        character. This *bit-parallelism* makes it @c O(n) with a tiny constant, and -- unlike
///        Boyer-Moore or KMP -- it extends naturally to *approximate* matching (see
///        @ref bitap_fuzzy_search). The pattern length must fit in a 64-bit word (<= 63).
///
/// @param text the text to search.
/// @param pattern the pattern to find (length 1..63; empty pattern matches at 0).
/// @return the 0-based index of the first exact occurrence, or -1 if the pattern does not occur.
long bitap_search(const std::string& text, const std::string& pattern);

/// @brief *Fuzzy Bitap* (the *k-mismatches* variant): approximate string search allowing up to
///        @p max_errors *substitutions* -- i.e. it finds a length-@c m window of the text whose
///        *Hamming distance* to the pattern is at most @p max_errors. It carries @c max_errors+1
///        coupled bitmasks (one per error level), advancing them all with a shift and an AND per text
///        character, so it runs in @c O(n·k) while staying almost entirely inside the CPU word. It is
///        the no-indel mode of tools like =agrep=; the same skeleton extends to full Levenshtein
///        matching by adding insertion/deletion transitions between the error levels.
///
/// @param text the text to search.
/// @param pattern the pattern to find (length 1..62).
/// @param max_errors the maximum number of substitutions allowed (0 reduces to exact matching).
/// @return the 0-based *ending* index of the first length-@c m window within @p max_errors
///         substitutions of @p pattern, or -1 if no such window exists.
long bitap_fuzzy_search(const std::string& text, const std::string& pattern, int max_errors);

} // namespace datamunge::algorithms
