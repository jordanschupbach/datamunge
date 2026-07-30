#pragma once

/// \file mark_sweep.hpp
/// \brief Mark-and-sweep garbage collection.
///
/// Mark-and-sweep (McCarthy, 1960) is the original tracing collector. In the *mark* phase it
/// traverses the object graph from the roots, marking every reachable object live; in the
/// *sweep* phase it scans the whole heap and reclaims every unmarked object. Unlike reference
/// counting it collects cyclic garbage, at the cost of a stop-the-world pass over the heap. It
/// does not move objects, so it is simple but can fragment memory. This module computes the
/// live set and the reclaimed set for an object graph.

#include <datamunge/algorithms/object_graph.hpp>

#include <vector>

namespace datamunge::algorithms {

/// Result of a mark-and-sweep collection.
struct MarkSweepResult {
    std::vector<char> marked;  ///< marked[i] != 0 iff object i is live.
    std::vector<int>  live;    ///< Ids of surviving objects.
    std::vector<int>  freed;   ///< Ids of reclaimed objects.
};

/// \brief Run mark-and-sweep on an object graph.
inline MarkSweepResult mark_sweep(const ObjectGraph& g) {
    MarkSweepResult r;
    r.marked = reachable_from_roots(g);           // mark phase
    for (int i = 0; i < g.num_objects; ++i)       // sweep phase
        (r.marked[i] ? r.live : r.freed).push_back(i);
    return r;
}

}  // namespace datamunge::algorithms
