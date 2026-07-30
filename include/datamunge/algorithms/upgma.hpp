#pragma once

/// \file upgma.hpp
/// \brief UPGMA: unweighted-pair-group phylogenetic tree from a distance matrix.
///
/// UPGMA (Sokal & Michener, 1958) builds a rooted tree from a matrix of pairwise distances by
/// *agglomerative clustering*. It repeatedly joins the two closest clusters, placing the new
/// internal node at height equal to half their distance, and updates distances to the merged
/// cluster as the size-weighted average of its parts. The result is an *ultrametric* tree -- all
/// leaves equidistant from the root -- which corresponds to a molecular clock (constant
/// evolutionary rate). This module returns the sequence of merges with their heights.

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::algorithms {

/// One agglomeration step: the two joined cluster ids and the node height.
struct UPGMAMerge {
    int    left;    ///< Child cluster id (leaf id < n, or internal id >= n).
    int    right;   ///< Child cluster id.
    double height;  ///< Height of the new internal node (= half the join distance).
    int    id;      ///< Id assigned to the merged cluster.
};

/// Result of UPGMA: the merges in the order they were made (last is the root).
struct UPGMAResult {
    std::vector<UPGMAMerge> merges;
};

/// \brief Build a UPGMA tree from an \c n x n symmetric distance matrix.
///
/// Leaves are ids 0..n-1; internal nodes get ids n, n+1, .... Returns the n-1 merges.
inline UPGMAResult upgma(std::vector<std::vector<double>> dist) {
    const int n = static_cast<int>(dist.size());
    UPGMAResult result;

    std::vector<int> id(n), size(n, 1);   // active clusters, parallel arrays over the matrix rows
    for (int i = 0; i < n; ++i) id[i] = i;
    int next_id = n;

    while (static_cast<int>(id.size()) > 1) {
        // Find the closest pair of active clusters.
        int    bi = 0, bj = 1;
        double best = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < id.size(); ++i)
            for (std::size_t j = i + 1; j < id.size(); ++j)
                if (dist[i][j] < best) { best = dist[i][j]; bi = (int)i; bj = (int)j; }

        int si = size[bi], sj = size[bj];
        result.merges.push_back({id[bi], id[bj], best / 2.0, next_id});

        // Weighted-average distances from the new cluster to every other active cluster.
        std::vector<double> newrow(id.size(), 0.0);
        for (std::size_t k = 0; k < id.size(); ++k)
            if ((int)k != bi && (int)k != bj)
                newrow[k] = (si * dist[bi][k] + sj * dist[bj][k]) / (si + sj);

        // Rebuild the active set: drop bi, bj; append the merged cluster.
        std::vector<int>    nid;
        std::vector<int>    nsize;
        std::vector<double> kept;   // distances of survivors to the new cluster
        for (std::size_t k = 0; k < id.size(); ++k)
            if ((int)k != bi && (int)k != bj) { nid.push_back(id[k]); nsize.push_back(size[k]); kept.push_back(newrow[k]); }

        const int m = static_cast<int>(nid.size());
        std::vector<std::vector<double>> nd(m + 1, std::vector<double>(m + 1, 0.0));
        // Copy surviving-vs-surviving distances.
        std::vector<int> survivors;
        for (std::size_t k = 0; k < id.size(); ++k)
            if ((int)k != bi && (int)k != bj) survivors.push_back((int)k);
        for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b) nd[a][b] = dist[survivors[a]][survivors[b]];
        for (int a = 0; a < m; ++a) { nd[a][m] = kept[a]; nd[m][a] = kept[a]; }

        nid.push_back(next_id++);
        nsize.push_back(si + sj);
        id   = std::move(nid);
        size = std::move(nsize);
        dist = std::move(nd);
    }
    return result;
}

}  // namespace datamunge::algorithms
