#pragma once

// Gabow's path-based strong-component algorithm (1999): find the strongly
// connected components of a directed graph in a single depth-first search, in
// linear time. Where Tarjan's algorithm tracks a numeric "low-link" per vertex,
// Gabow's keeps two stacks -- one of all vertices on the current DFS path, and one
// of *candidate component roots*. A back edge to an already-open vertex collapses
// the candidate-root stack down to that vertex, merging the cycle; when a vertex
// finishes and is still its own candidate root, it and everything above it on the
// path form one component. It is arguably the simplest linear-time SCC method.

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

// Strongly connected components of an n-vertex digraph given by adjacency lists.
// Returns a component id per vertex (0-based, contiguous); two vertices share an
// id iff they are mutually reachable.
inline std::vector<int> gabow_scc(int n, const std::vector<std::vector<int>>& adj) {
    std::vector<int> comp(n, -1);   // final component id, -1 until assigned
    std::vector<int> preorder(n, 0); // 1-based DFS preorder number, 0 = unvisited
    std::vector<int> path;           // stack S: all vertices on the current DFS path
    std::vector<int> roots;          // stack B: candidate component roots
    int              counter = 0;    // preorder counter
    int              ncomp   = 0;

    // Iterative DFS to avoid recursion limits. Each frame is (vertex, next child index).
    std::vector<std::pair<int, std::size_t>> stack;
    for (int start = 0; start < n; ++start) {
        if (preorder[start] != 0) continue;
        stack.push_back({start, 0});
        while (!stack.empty()) {
            auto& [u, ci] = stack.back();
            if (ci == 0) { // first visit
                preorder[u] = ++counter;
                path.push_back(u);
                roots.push_back(u);
            }
            if (ci < adj[u].size()) {
                const int v = adj[u][ci];
                ++ci;
                if (preorder[v] == 0) {
                    stack.push_back({v, 0});
                } else if (comp[v] == -1) {
                    // v is on the current path: pop candidate roots deeper than v.
                    while (!roots.empty() && preorder[roots.back()] > preorder[v]) roots.pop_back();
                }
            } else {
                // Finished u. If u is still a candidate root, close a component.
                if (!roots.empty() && roots.back() == u) {
                    roots.pop_back();
                    int w;
                    do {
                        w = path.back();
                        path.pop_back();
                        comp[w] = ncomp;
                    } while (w != u);
                    ++ncomp;
                }
                stack.pop_back();
            }
        }
    }
    return comp;
}

} // namespace datamunge::algorithms
