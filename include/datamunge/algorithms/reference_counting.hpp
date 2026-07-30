#pragma once

/// \file reference_counting.hpp
/// \brief Reference-counting garbage collection (and its cyclic-garbage blind spot).
///
/// Reference counting keeps, with every object, a count of how many references point at it.
/// When a reference is created the count goes up; when one is dropped it goes down, and an
/// object whose count reaches zero is freed *immediately*, its own outgoing references
/// decremented in turn (a cascade). This gives prompt, incremental reclamation with no
/// stop-the-world pause -- but it *cannot* collect *cycles*: a group of objects that reference
/// each other keeps every count above zero even when the whole group is unreachable. This
/// module computes the counts, the objects reclaimed by cascading to zero, and the cyclic
/// garbage that leaks.

#include <datamunge/algorithms/object_graph.hpp>

#include <vector>

namespace datamunge::algorithms {

/// Result of a reference-counting collection.
struct RefCountResult {
    std::vector<int>  refcount;  ///< Incoming-reference count of each object (roots + objects).
    std::vector<char> freed;     ///< freed[i] != 0 iff reclaimed by reaching count zero.
    std::vector<int>  leaked;    ///< Unreachable objects that reference counting fails to free.
};

/// \brief Simulate reference-counting collection on an object graph.
///
/// The count of an object is the number of references to it from roots plus other objects.
/// Objects whose count is (or cascades to) zero are freed; unreachable objects still held by a
/// cycle are reported as leaked.
inline RefCountResult reference_counting_collect(const ObjectGraph& g) {
    RefCountResult r;
    std::vector<int> rc(g.num_objects, 0);
    for (int root : g.roots)
        if (root >= 0) rc[root]++;
    for (int i = 0; i < g.num_objects; ++i)
        for (int c : g.refs[i]) rc[c]++;
    r.refcount = rc;                                  // record the initial counts
    r.freed.assign(g.num_objects, 0);

    std::vector<int> work;                            // objects whose count is zero
    for (int i = 0; i < g.num_objects; ++i)
        if (rc[i] == 0) work.push_back(i);
    while (!work.empty()) {
        int u = work.back();
        work.pop_back();
        if (r.freed[u]) continue;
        r.freed[u] = 1;
        for (int c : g.refs[u])                       // dropping u's references cascades
            if (--rc[c] == 0 && !r.freed[c]) work.push_back(c);
    }

    // Cyclic garbage: unreachable from the roots, yet never freed.
    std::vector<char> reach = reachable_from_roots(g);
    for (int i = 0; i < g.num_objects; ++i)
        if (!reach[i] && !r.freed[i]) r.leaked.push_back(i);
    return r;
}

}  // namespace datamunge::algorithms
