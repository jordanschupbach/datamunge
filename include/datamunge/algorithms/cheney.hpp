#pragma once

/// \file cheney.hpp
/// \brief Cheney's algorithm: iterative copying (semi-space) garbage collection.
///
/// A *copying* collector divides the heap into two equal semispaces. Allocation happens in
/// *from-space*; when it fills, the collector copies every live object into *to-space* and
/// swaps the roles, reclaiming all garbage at once by simply abandoning from-space. Cheney's
/// insight (1970) is to do the copy *iteratively with no auxiliary stack*: a *scan* pointer
/// chases an *allocation* pointer through to-space, and the gap between them is exactly the
/// breadth-first queue of copied-but-not-yet-scanned objects. Copying is proportional to the
/// *live* data only, and the result is compacted with zero fragmentation. This module computes
/// the to-space layout and the forwarding addresses.

#include <datamunge/algorithms/object_graph.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Result of a Cheney copy.
struct CheneyResult {
    std::vector<int> to_space_order;  ///< Old ids in to-space order (index = new address).
    std::vector<int> forwarding;      ///< forwarding[old] = new address, or -1 if garbage.
};

/// \brief Run Cheney's iterative copying collection on an object graph.
///
/// Copies the roots into to-space, then advances a scan pointer: for each scanned object it
/// forwards any not-yet-copied child to the end of to-space. When scan catches the allocation
/// pointer, all live objects have been copied breadth-first.
inline CheneyResult cheney_copy(const ObjectGraph& g) {
    CheneyResult r;
    r.forwarding.assign(g.num_objects, -1);
    std::vector<int>& to = r.to_space_order;

    auto forward = [&](int obj) {
        if (r.forwarding[obj] < 0) {                       // not yet copied
            r.forwarding[obj] = static_cast<int>(to.size());
            to.push_back(obj);
        }
    };

    for (int root : g.roots) forward(root);                // copy the roots first
    for (std::size_t scan = 0; scan < to.size(); ++scan)   // scan pointer chases allocation pointer
        for (int child : g.refs[to[scan]]) forward(child);
    return r;
}

}  // namespace datamunge::algorithms
