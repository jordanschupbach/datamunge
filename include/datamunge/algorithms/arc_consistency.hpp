#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief The result of running AC-3: the (possibly reduced) variable @ref domains and whether the
///        network is still @ref consistent (no domain was emptied).
struct Ac3Result {
    std::vector<std::vector<int>> domains;         ///< domains[v] = the surviving values of variable v.
    bool                          consistent{true}; ///< false if some domain became empty (unsatisfiable).
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
/// @param domains the initial domain of each variable (index = variable id).
/// @param arcs the directed constraint arcs @c (i,j) to enforce (include both directions for a
///        symmetric constraint).
/// @param compatible predicate: @c compatible(i,a,j,b) is true when value @c a for variable @c i is
///        consistent with value @c b for variable @c j.
/// @return the reduced @ref Ac3Result; @c consistent is false if any domain was emptied.
Ac3Result ac3(std::vector<std::vector<int>>              domains,
              const std::vector<std::pair<int, int>>&    arcs,
              const std::function<bool(int, int, int, int)>& compatible);

} // namespace datamunge::algorithms
