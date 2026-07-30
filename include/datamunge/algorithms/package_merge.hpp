#pragma once

/// \file package_merge.hpp
/// \brief The package-merge algorithm for length-limited Huffman codes.
///
/// A plain Huffman code minimizes the expected codeword length, but its longest codeword can be
/// long -- a problem for hardware decoders and formats (JPEG, DEFLATE) that cap code length. The
/// *package-merge* algorithm (Larmore & Hirschberg, 1990) finds the optimal prefix code subject
/// to a maximum length \f$L\f$. It reduces the problem to the *coin collector's problem*: each
/// symbol offers a "coin" at every width \f$2^{-1},\dots,2^{-L}\f$ with numismatic value equal
/// to its frequency, and we buy total width \f$n-1\f$ as cheaply as possible. Building solutions
/// from the finest width up -- *packaging* pairs and *merging* them with the next level's coins --
/// yields code lengths that minimize \f$\sum_i w_i\ell_i\f$ with all \f$\ell_i\le L\f$. This
/// module returns those optimal length-limited code lengths.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
/// A "coin": a weight and the set of symbols whose lengths it increments if selected.
struct PMCoin {
    std::uint64_t    weight;
    std::vector<int> symbols;
};
}  // namespace detail

/// \brief Optimal code lengths minimizing sum(w_i * len_i) with every len_i <= max_len.
///
/// \param weights  symbol frequencies/weights (all > 0).
/// \param max_len  maximum allowed codeword length (must satisfy 2^max_len >= n).
/// \return a length per symbol; the lengths satisfy the Kraft equality sum 2^{-len} = 1.
inline std::vector<int> package_merge(const std::vector<std::uint64_t>& weights, int max_len) {
    using detail::PMCoin;
    const int n = static_cast<int>(weights.size());
    std::vector<int> lengths(n, 0);
    if (n == 0) return lengths;
    if (n == 1) { lengths[0] = 1; return lengths; }

    // Singleton coins, sorted by weight (kept for re-use at every level).
    std::vector<PMCoin> singletons(n);
    for (int i = 0; i < n; ++i) singletons[i] = {weights[i], {i}};
    std::sort(singletons.begin(), singletons.end(),
              [](const PMCoin& a, const PMCoin& b) { return a.weight < b.weight; });

    std::vector<PMCoin> current = singletons;   // level of width 2^-max_len
    for (int level = 0; level < max_len - 1; ++level) {
        // Package: merge consecutive pairs (drop a trailing unpaired coin).
        std::vector<PMCoin> packages;
        for (std::size_t j = 0; j + 1 < current.size(); j += 2) {
            PMCoin p;
            p.weight  = current[j].weight + current[j + 1].weight;
            p.symbols = current[j].symbols;
            p.symbols.insert(p.symbols.end(), current[j + 1].symbols.begin(), current[j + 1].symbols.end());
            packages.push_back(std::move(p));
        }
        // Merge the fresh singletons with the packages, sorted by weight.
        current.clear();
        current.reserve(singletons.size() + packages.size());
        current.insert(current.end(), singletons.begin(), singletons.end());
        current.insert(current.end(), packages.begin(), packages.end());
        std::stable_sort(current.begin(), current.end(),
                         [](const PMCoin& a, const PMCoin& b) { return a.weight < b.weight; });
    }

    // Select the 2n-2 cheapest coins; each symbol's length is how many contain it.
    int take = 2 * n - 2;
    for (int i = 0; i < take && i < (int)current.size(); ++i)
        for (int sym : current[i].symbols) ++lengths[sym];
    return lengths;
}

}  // namespace datamunge::algorithms
