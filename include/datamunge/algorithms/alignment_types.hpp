#pragma once
/// @file alignment_types.hpp
/// @brief Shared byte-alignment scoring and historical gapped-row result types.
#include <string>
namespace datamunge::algorithms {
/// @brief Symmetric linear-gap integer scores, with no affine gap-open term.
/// match scores equal bytes, mismatch unequal bytes, gap one byte against a gap.
/// Defaults2,-1,-2. Specific algorithms validate their own arithmetic/input domains.
/// Positive gap scores are representable but may favor opposing inserted gaps.
/// @par Example
/// @code
/// const dm::algorithms::AlignmentScoring s;
/// std::print("{} {} {}\n",s.match,s.mismatch,s.gap); // 2 -1 -2
/// @endcode
/// @see Alignment, hirschberg()
struct AlignmentScoring {
  int match{2};     ///< Score for equal aligned bytes.
  int mismatch{-1}; ///< Score for unequal aligned bytes.
  int gap{-2};      ///< Score for one byte aligned against an inserted gap.
};
/// @brief Owned score and equal-length gapped rows from a pairwise byte alignment.
/// A '-' denotes an inserted gap; to recover inputs unambiguously their literal
/// alphabet must exclude '-'. Global results cover whole inputs; local results
/// may cover substrings. No input or callable references are retained.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hirschberg("A","A");
/// std::print("{} {}\n",r.score,r.a_aligned); // 2 A
/// @endcode
/// @see AlignmentScoring, needleman_wunsch(), hirschberg()
struct Alignment {
  int         score{0};  ///< Producing algorithm's optimal score under its stated contract.
  std::string a_aligned; ///< Owned first row with '-' inserted for gaps.
  std::string b_aligned; ///< Owned second row with '-' inserted for gaps.
};
} // namespace datamunge::algorithms
