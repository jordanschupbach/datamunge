#pragma once

/// \file mark_compact.hpp
/// \brief Mark-compact garbage collection.
///
/// Mark-compact collectors mark the live objects like mark-sweep, but then *slide* the
/// survivors to one end of the heap, squeezing out the gaps left by dead objects and updating
/// every reference to the new addresses. The result is a compacted heap with no
/// fragmentation, allocated by a simple bump pointer, and (unlike a copying collector) it uses
/// no extra semispace -- at the cost of extra passes to compute and apply the new addresses.
/// This module computes the live set, the sliding relocation (address for each survivor,
/// preserving heap order), and the updated reference graph.

#include <datamunge/algorithms/object_graph.hpp>

#include <vector>

namespace datamunge::algorithms {

/// Result of a mark-compact collection.
struct MarkCompactResult {
    std::vector<char>             marked;       ///< marked[i] != 0 iff object i is live.
    std::vector<int>              new_address;  ///< new_address[i] = compacted slot, or -1 if dead.
    std::vector<int>              order;        ///< Live object ids in their compacted order.
    std::vector<std::vector<int>> updated_refs; ///< order[k]'s refs, expressed as new addresses.
};

/// \brief Run mark-compact on an object graph (sliding/order-preserving compaction).
inline MarkCompactResult mark_compact(const ObjectGraph& g) {
    MarkCompactResult r;
    r.marked = reachable_from_roots(g);
    r.new_address.assign(g.num_objects, -1);

    int slot = 0;                                          // bump pointer over survivors
    for (int i = 0; i < g.num_objects; ++i)
        if (r.marked[i]) { r.new_address[i] = slot++; r.order.push_back(i); }

    // Rewrite each survivor's references to point at the compacted addresses.
    for (int id : r.order) {
        std::vector<int> nr;
        for (int child : g.refs[id])
            if (r.marked[child]) nr.push_back(r.new_address[child]);
        r.updated_refs.push_back(std::move(nr));
    }
    return r;
}

}  // namespace datamunge::algorithms
