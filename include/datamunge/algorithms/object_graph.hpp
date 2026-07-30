#pragma once

/// \file object_graph.hpp
/// \brief A tiny object-reference graph shared by the garbage-collection algorithms.
///
/// The tracing collectors (mark-sweep, mark-compact, Cheney, generational) all view the heap
/// abstractly: a set of objects, each holding references (edges) to other objects, plus a set
/// of *roots* (registers, stack slots, globals) from which liveness is defined. An object is
/// *live* iff it is reachable from some root; everything else is garbage. This header gives
/// that representation and the reachability sweep the collectors share.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// A heap as an object-reference graph.
struct ObjectGraph {
    int                           num_objects;  ///< Objects 0..num_objects-1.
    std::vector<std::vector<int>> refs;         ///< refs[i] = objects that object i points to.
    std::vector<int>              roots;        ///< Root object ids (directly reachable).
    std::vector<int>              sizes;        ///< Optional per-object sizes (default 1 each).
};

/// \brief Mark all objects reachable from the roots (depth-first).
///
/// \return a vector where entry i is nonzero iff object i is live.
inline std::vector<char> reachable_from_roots(const ObjectGraph& g) {
    std::vector<char> mark(g.num_objects, 0);
    std::vector<int>  stack;
    for (int r : g.roots)
        if (r >= 0 && !mark[r]) { mark[r] = 1; stack.push_back(r); }
    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        for (int v : g.refs[u])
            if (!mark[v]) { mark[v] = 1; stack.push_back(v); }
    }
    return mark;
}

/// \brief Size of object i (its \c sizes entry, or 1 if sizes is unset).
inline int object_size(const ObjectGraph& g, int i) {
    return (static_cast<int>(g.sizes.size()) > i) ? g.sizes[i] : 1;
}

}  // namespace datamunge::algorithms
