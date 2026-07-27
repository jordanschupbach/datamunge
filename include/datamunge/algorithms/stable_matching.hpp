#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief Sentinel for an unmatched agent in a StableMatchingResult.
inline constexpr std::size_t kUnmatched = static_cast<std::size_t>(-1);

struct StableMatchingResult {
    /// @brief proposer_match[i] is the reviewer matched to proposer i (or kUnmatched).
    std::vector<std::size_t> proposer_match;
    /// @brief reviewer_match[j] is the proposer matched to reviewer j (or kUnmatched).
    std::vector<std::size_t> reviewer_match;
    /// @brief Total proposals made -- a cost measure, at most n^2.
    std::size_t proposals{0};
};

/// @brief The Gale-Shapley algorithm (Gale & Shapley 1962), a.k.a. deferred acceptance. Given
///        two equal-sized sides -- "proposers" and "reviewers" -- each with a complete strict
///        ranking of the other side, it produces a *stable* one-to-one matching: one with no
///        *blocking pair*, i.e. no proposer and reviewer who would both rather be matched to
///        each other than to their assigned partners. Free proposers propose down their
///        preference lists; each reviewer tentatively holds their best proposer so far and
///        rejects the rest, "deferring acceptance" until no proposer is left free. The result
///        is *proposer-optimal* (every proposer gets the best partner they attain in any stable
///        matching) and simultaneously *reviewer-pessimal*. It underlies the medical residency
///        match and many school-choice systems.
///
/// @param proposer_preferences proposer_preferences[i] is proposer i's ranking of reviewer
///        indices, most preferred first (a permutation of 0..n-1).
/// @param reviewer_preferences reviewer_preferences[j] is reviewer j's ranking of proposer
///        indices, most preferred first.
/// @return the proposer-optimal stable matching.
StableMatchingResult gale_shapley(const std::vector<std::vector<std::size_t>>& proposer_preferences,
                                  const std::vector<std::vector<std::size_t>>& reviewer_preferences);

/// @brief Checks whether @p matching (in proposer orientation) is stable under the given
///        preferences: returns true iff it has no blocking pair.
[[nodiscard]] bool is_stable(const StableMatchingResult& matching,
                             const std::vector<std::vector<std::size_t>>& proposer_preferences,
                             const std::vector<std::vector<std::size_t>>& reviewer_preferences);

} // namespace datamunge::algorithms
