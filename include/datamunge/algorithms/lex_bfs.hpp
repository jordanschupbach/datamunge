#pragma once

// Lexicographic breadth-first search (Lex-BFS): a linear-time vertex ordering of
// an undirected graph by partition refinement. Repeatedly output a vertex from
// the first set of an ordered partition, then split every remaining set so that
// the just-output vertex's neighbours come first. The reverse of a Lex-BFS order
// is a perfect elimination ordering when the graph is chordal, which is how
// Lex-BFS recognises chordal and interval graphs.

#include <list>
#include <vector>

namespace datamunge::algorithms {

// Returns a Lex-BFS ordering of vertices 0..n-1 (starting from vertex 0's set).
inline std::vector<int> lex_bfs(int n, const std::vector<std::vector<int>>& adj) {
    std::vector<std::vector<char>> nb(n, std::vector<char>(n, 0));
    for (int u = 0; u < n; ++u)
        for (int v : adj[u])
            if (u != v) { nb[u][v] = 1; nb[v][u] = 1; }

    std::list<std::vector<int>> parts;
    std::vector<int>            all(n);
    for (int i = 0; i < n; ++i) all[i] = i;
    if (n > 0) parts.push_back(all);

    std::vector<int> order;
    order.reserve(n);
    while (!parts.empty()) {
        // Take a vertex from the first set.
        const int v = parts.front().back();
        parts.front().pop_back();
        if (parts.front().empty()) parts.pop_front();
        order.push_back(v);

        // Refine: split each set into (neighbours of v) before (non-neighbours).
        std::list<std::vector<int>> next;
        for (auto& S : parts) {
            std::vector<int> in, out;
            for (int u : S) (nb[v][u] ? in : out).push_back(u);
            if (!in.empty()) next.push_back(std::move(in));
            if (!out.empty()) next.push_back(std::move(out));
        }
        parts.swap(next);
    }
    return order;
}

} // namespace datamunge::algorithms
