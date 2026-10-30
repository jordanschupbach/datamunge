#pragma once

/// @file arc_consistency.hpp
/// @brief AC-3 arc-consistency propagation for binary constraint-satisfaction problems.

#include <functional>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief The result of running AC-3: the (possibly reduced) variable @ref domains and whether the
///        network is still @ref consistent (no domain was emptied).
struct Ac3Result {
    std::vector<std::vector<int>> domains;         ///< domains[v] = the surviving values of variable v.
    bool                          consistent{true}; ///< false if some domain became empty (unsatisfiable).
    long long                     revisions{0};    ///< Arcs taken off the worklist and revised.
    long long                     removals{0};     ///< Values deleted from domains, in total.
    long long                     checks{0};       ///< Calls of the `compatible` predicate.
    std::vector<long long>        candidates;      ///< Total domain size at the start and after every
                                                   ///< revision that deleted a value: front() is the
                                                   ///< initial total, back() the final one.
};

/// @brief The *AC-3 algorithm* (Mackworth, 1977) for enforcing *arc consistency* in a binary
///        constraint-satisfaction problem. A directed arc @c (i,j) is consistent when *every* value in
///        variable @c i's domain has at least one *supporting* value in @c j's domain that satisfies
///        the constraint. AC-3 keeps a worklist of arcs; it repeatedly *revises* an arc, deleting
///        unsupported values from @c i's domain, and whenever a domain shrinks it re-queues the arcs
///        pointing *into* @c i (their support may have vanished). It runs in @c O(e·d^3) and is the
///        standard constraint-propagation step inside CSP solvers -- pruning the search space before
///        (and during) backtracking, sometimes solving the problem outright.
///
///        The worklist holds each arc at most once (an arc already waiting is not queued again),
///        and the arcs into each variable are indexed up front, so re-queuing costs time
///        proportional to that variable's in-degree. When revising @c (i,j) shrinks @c i, every
///        arc @c (k,i) with @c k != j is re-queued. The reverse arc @c (j,i) is skipped: a value
///        deleted from @c i had no compatible partner in @c j, so it supported nothing there.
///        That relies on @p compatible treating @c (j,i) as the converse of @c (i,j) (true of
///        any binary constraint listed in both directions). The result is the unique largest
///        arc-consistent sub-domain of the input, whatever order the arcs are processed in.
///
/// @param domains the initial domain of each variable (index = variable id).
/// @param arcs the directed constraint arcs @c (i,j) to enforce (include both directions for a
///        symmetric constraint).
/// @param compatible predicate: @c compatible(i,a,j,b) is true when value @c a for variable @c i is
///        consistent with value @c b for variable @c j.
/// @return the reduced @ref Ac3Result with work counters; @c consistent is false if any domain
///         was emptied (the remaining domains are then partial and should not be used).
///
/// @par Example
/// @code
/// // x0 < x1 < x2 over 1..5: arc consistency narrows x0 to {1,2,3} and x2 to {3,4,5}.
/// std::vector<std::vector<int>> d(3, {1, 2, 3, 4, 5});
/// const auto r = dm::algorithms::ac3(d, {{0, 1}, {1, 0}, {1, 2}, {2, 1}},
///     [](int i, int a, int j, int b) { return i < j ? a < b : a > b; });
/// @endcode
Ac3Result ac3(std::vector<std::vector<int>>              domains,
              const std::vector<std::pair<int, int>>&    arcs,
              const std::function<bool(int, int, int, int)>& compatible);

} // namespace datamunge::algorithms
