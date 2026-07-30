#pragma once

/// \file velvet.hpp
/// \brief Velvet-style genome assembly over a de Bruijn graph.
///
/// Short-read genome assemblers such as *Velvet* (Zerbino & Birney, 2008) reconstruct a
/// sequence from many overlapping reads using a *de Bruijn graph*. Each read is chopped into
/// \f$k\f$-mers; every \f$k\f$-mer is an edge from its length-\f$(k-1)\f$ prefix node to its
/// length-\f$(k-1)\f$ suffix node. Assembling the sequence then amounts to finding an
/// *Eulerian path* through the graph -- a walk using every edge once -- which spells out the
/// original sequence. This module builds the de Bruijn graph and reconstructs a contig by
/// Hierholzer's Eulerian-path algorithm.

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

/// \brief Build the de Bruijn graph of the reads' k-mers and reconstruct a contig via an
///        Eulerian path.
///
/// \param reads  input reads (each is broken into overlapping k-mers).
/// \param k      k-mer length (nodes are (k-1)-mers, edges are k-mers).
/// \return the reconstructed sequence, or empty if no Eulerian path exists.
inline std::string velvet_assemble(const std::vector<std::string>& reads, int k) {
    // Build adjacency: node ((k-1)-mer) -> list of successor nodes (one per k-mer edge).
    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::unordered_map<std::string, int>                      outdeg, indeg;

    for (const auto& read : reads)
        for (int i = 0; i + k <= (int)read.size(); ++i) {
            std::string from = read.substr(i, k - 1);
            std::string to   = read.substr(i + 1, k - 1);
            adj[from].push_back(to);
            ++outdeg[from];
            ++indeg[to];
            if (!indeg.count(from)) indeg[from] += 0;
            if (!outdeg.count(to)) outdeg[to] += 0;
        }
    if (adj.empty()) return "";

    // Sort successors for a deterministic reconstruction.
    for (auto& [node, succ] : adj) std::sort(succ.begin(), succ.end());

    // Eulerian-path start: a node with outdeg - indeg == 1, else any node with an out-edge.
    std::string start;
    for (const auto& [node, od] : outdeg) {
        int id = indeg.count(node) ? indeg[node] : 0;
        if (od - id == 1) { start = node; break; }
    }
    if (start.empty()) start = adj.begin()->first;

    // Hierholzer's algorithm (iterative) over the multigraph.
    std::unordered_map<std::string, std::size_t> next_edge;   // consumed-edge cursor per node
    std::vector<std::string>                     stack{start}, circuit;
    while (!stack.empty()) {
        std::string v  = stack.back();
        auto        it = adj.find(v);
        if (it != adj.end() && next_edge[v] < it->second.size()) {
            stack.push_back(it->second[next_edge[v]++]);
        } else {
            circuit.push_back(v);
            stack.pop_back();
        }
    }
    std::reverse(circuit.begin(), circuit.end());

    // Spell the sequence: first node in full, then each subsequent node's last character.
    if (circuit.empty()) return "";
    std::string seq = circuit.front();
    for (std::size_t i = 1; i < circuit.size(); ++i) seq += circuit[i].back();
    return seq;
}

}  // namespace datamunge::algorithms
