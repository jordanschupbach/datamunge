/// @file trigram_search.hpp
/// @brief Deterministic padded-byte trigram sets and stable Jaccard rankings.
#pragma once

// Trigram search: fuzzy string matching for when the exact spelling is unknown.
// A string is represented by the set of its 3-character substrings (trigrams);
// two strings are similar when their trigram sets overlap. The Jaccard overlap
// gives a similarity score robust to typos, insertions, and small edits.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Build padded ASCII-case-folded distinct byte trigrams.
/// Two leading spaces and one trailing space; ASCII A-Z only are folded. O(n log n) comparisons, O(n) storage; calls independent.
/// @param s Arbitrary bytes, including embedded NUL; empty yields empty.
/// @return Owned trigram set.
/// @par Example
/// @code
/// auto grams=dm::algorithms::trigrams("A"); //two trigrams
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline std::set<std::string> trigrams(const std::string& s) {
  std::string t = "  ";
  for (char c : s)
    t.push_back(
        static_cast<char>(
            (c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : static_cast<unsigned char>(c))));
  t.push_back(' ');
  std::set<std::string> out;
  for (std::size_t i = 0; i + 3 <= t.size(); ++i)
    out.insert(t.substr(i, 3));
  return out;
}

/// @brief Compute padded-byte trigram Jaccard similarity.
/// @param a First string.
/// @param b Second string.
/// @return Intersection/union, or1 for two empty sets.
/// @par Example
/// @code
/// double score=dm::algorithms::trigram_similarity("cat","cat"); //1
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline double trigram_similarity(const std::string& a, const std::string& b) {
  const std::set<std::string> A = trigrams(a), B = trigrams(b);
  if (A.empty() && B.empty())
    return 1.0;
  std::size_t inter = 0;
  for (const auto& g : A)
    if (B.count(g))
      ++inter;
  const std::size_t uni = A.size() + B.size() - inter;
  return uni == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(uni);
}

/// @brief Rank candidates by descending trigram Jaccard similarity.
/// @param candidates Borrowed strings; empty accepted.
/// @param query Borrowed query string.
/// @return Owned score,index pairs; ties retain input order.
/// @par Example
/// @code
/// auto ranked=dm::algorithms::trigram_search({"cat","bat"},"cat");
/// @endcode
/// @par References
/// - Actual canonical producer; report derives the recurrence.
/// @see Related legacy convenience overloads in this header.
inline std::vector<std::pair<double, std::size_t>> trigram_search(
    const std::vector<std::string>& candidates,
    const std::string&              query) {
  std::vector<std::pair<double, std::size_t>> scored;
  scored.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i)
    scored.emplace_back(trigram_similarity(candidates[i], query), i);
  std::stable_sort(
      scored.begin(), scored.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
  return scored;
}

} // namespace datamunge::algorithms
