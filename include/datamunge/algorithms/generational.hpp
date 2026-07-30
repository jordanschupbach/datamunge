#pragma once

/// \file generational.hpp
/// \brief Generational garbage collection: collect the young, spare the old.
///
/// The *generational hypothesis* -- "most objects die young" -- motivates segregating objects
/// by age. New objects go in a *young* generation collected frequently; survivors are
/// *promoted* to an *old* generation collected rarely. A *minor* collection traces only the
/// young generation, treating old objects as live boundaries, so it scans far less than a full
/// heap trace. The one subtlety is *old-to-young* references: an old object pointing at a young
/// one keeps it alive, so those pointers are tracked in a *remembered set* and used as extra
/// roots for the minor collection. This module runs a minor collection and reports what it
/// scans, frees, and promotes.

#include <datamunge/algorithms/object_graph.hpp>

#include <vector>

namespace datamunge::algorithms {

/// Result of a minor (young-generation) collection.
struct GenerationalResult {
    std::vector<char> promoted;        ///< promoted[i] != 0 iff young object i survived (-> old).
    std::vector<int>  young_freed;     ///< Young objects reclaimed.
    std::vector<int>  remembered_set;  ///< Old objects with references into the young generation.
    int               minor_scanned;   ///< Young objects visited by the minor collection.
    int               full_scanned;    ///< Objects a full-heap trace would visit (for comparison).
};

/// \brief Run a minor collection given a per-object generation (0 = young, 1 = old).
inline GenerationalResult minor_gc(const ObjectGraph& g, const std::vector<int>& gen) {
    GenerationalResult res;

    // Minor roots: young objects reachable directly from roots, or from old objects
    // (the remembered set) that reference the young generation.
    std::vector<int> minor_roots;
    for (int r : g.roots)
        if (r >= 0 && gen[r] == 0) minor_roots.push_back(r);
    for (int i = 0; i < g.num_objects; ++i)
        if (gen[i] == 1) {
            bool remembered = false;
            for (int c : g.refs[i])
                if (gen[c] == 0) { minor_roots.push_back(c); remembered = true; }
            if (remembered) res.remembered_set.push_back(i);
        }

    // Trace the young generation only (stop at old objects).
    std::vector<char> mark(g.num_objects, 0);
    std::vector<int>  stack;
    for (int r : minor_roots)
        if (!mark[r]) { mark[r] = 1; stack.push_back(r); }
    int scanned = 0;
    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        ++scanned;
        for (int c : g.refs[u])
            if (gen[c] == 0 && !mark[c]) { mark[c] = 1; stack.push_back(c); }
    }

    res.minor_scanned = scanned;
    res.promoted.assign(g.num_objects, 0);
    for (int i = 0; i < g.num_objects; ++i)
        if (gen[i] == 0) {
            if (mark[i]) res.promoted[i] = 1;      // survivor -> promoted to old
            else         res.young_freed.push_back(i);
        }

    std::vector<char> full = reachable_from_roots(g);
    res.full_scanned = 0;
    for (int i = 0; i < g.num_objects; ++i)
        if (full[i]) ++res.full_scanned;
    return res;
}

}  // namespace datamunge::algorithms
