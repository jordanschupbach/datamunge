#pragma once

/// \file connected_components.hpp
/// \brief Connected-component labeling: assign each foreground cell of a binary image a
///        label identifying the connected blob it belongs to.
///
/// Given a binary image (foreground vs background), connected-component labeling groups
/// foreground cells into *components* -- maximal sets connected through neighboring
/// foreground cells -- and gives every cell of a component the same integer label. It is
/// the flood-fill idea applied exhaustively, and the foundational operation of blob
/// analysis: counting objects, measuring their sizes, and extracting shapes. This
/// implementation uses a *union-find* (disjoint-set) two-pass scheme -- provisional labels
/// in the first pass, merged by equivalences and renumbered in the second -- supporting 4-
/// or 8-connectivity.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Result of labeling: the label image and the number of components found.
struct ConnectedComponents {
    std::vector<std::vector<int>> labels;      ///< 0 = background; 1..count = component ids.
    int                           count = 0;   ///< Number of components.
};

namespace detail {
struct DSU {
    std::vector<int> parent;
    int              make() { parent.push_back(static_cast<int>(parent.size())); return static_cast<int>(parent.size()) - 1; }
    int              find(int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
    void             unite(int a, int b) { parent[find(a)] = find(b); }
};
}  // namespace detail

/// \brief Label the connected components of a binary image (nonzero = foreground).
///
/// \param image           Binary image (nonzero cells are foreground).
/// \param eight_connected Use 8-connectivity if true, else 4-connectivity.
inline ConnectedComponents connected_components(const std::vector<std::vector<int>>& image,
                                                bool eight_connected = false) {
    ConnectedComponents out;
    const std::size_t   rows = image.size();
    const std::size_t   cols = rows ? image.front().size() : 0;
    out.labels.assign(rows, std::vector<int>(cols, 0));
    if (rows == 0 || cols == 0) return out;

    detail::DSU dsu;
    dsu.make();  // dummy 0 = background
    // First pass: assign provisional labels, record equivalences with already-labeled neighbors.
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            if (image[i][j] == 0) continue;
            std::vector<int> nbr;
            auto             consider = [&](int r, int c) {
                if (r >= 0 && c >= 0 && r < static_cast<int>(rows) && c < static_cast<int>(cols) &&
                    out.labels[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] != 0)
                    nbr.push_back(out.labels[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]);
            };
            consider(static_cast<int>(i) - 1, static_cast<int>(j));      // up
            consider(static_cast<int>(i), static_cast<int>(j) - 1);      // left
            if (eight_connected) {
                consider(static_cast<int>(i) - 1, static_cast<int>(j) - 1);
                consider(static_cast<int>(i) - 1, static_cast<int>(j) + 1);
            }
            if (nbr.empty()) {
                out.labels[i][j] = dsu.make();  // new provisional label
            } else {
                int m = nbr[0];
                for (int l : nbr) m = std::min(m, l);
                out.labels[i][j] = m;
                for (int l : nbr) dsu.unite(m, l);
            }
        }

    // Second pass: resolve equivalences and renumber to 1..count.
    std::vector<int> remap(dsu.parent.size(), 0);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            if (out.labels[i][j] == 0) continue;
            const int root = dsu.find(out.labels[i][j]);
            if (remap[root] == 0) remap[root] = ++out.count;
            out.labels[i][j] = remap[root];
        }
    return out;
}

}  // namespace datamunge::algorithms
