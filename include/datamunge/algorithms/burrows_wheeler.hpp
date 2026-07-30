#pragma once

/// \file burrows_wheeler.hpp
/// \brief The Burrows-Wheeler transform (BWT): a reversible permutation of a string that
///        groups similar contexts together, boosting downstream compression (Burrows &
///        Wheeler 1994).
///
/// The BWT does not itself compress -- it *reversibly reorders* the bytes so that
/// characters preceded by similar contexts end up adjacent, producing long runs that
/// move-to-front + run-length + entropy coding then squeeze (this is exactly the bzip2
/// pipeline). The forward transform sorts all cyclic rotations of the string and outputs
/// the *last column* plus the row index of the original string; remarkably, that column
/// and index are enough to invert the transform exactly, via the "LF-mapping" that walks
/// the last-to-first correspondence. It is one of the most elegant results in data
/// compression: a sort that is undoable.

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// The result of a forward BWT: the transformed string and the index of the original row.
struct BWTResult {
    std::string transformed;  ///< The last column of the sorted-rotation matrix.
    std::size_t index;        ///< Row where the original string landed after sorting.
};

/// \brief Forward Burrows-Wheeler transform of \p s (via sorted cyclic rotations).
inline BWTResult bwt_transform(const std::string& s) {
    const std::size_t n = s.size();
    BWTResult         r;
    if (n == 0) { r.index = 0; return r; }
    // Sort rotation start indices by the rotation they represent.
    std::vector<std::size_t> rot(n);
    std::iota(rot.begin(), rot.end(), std::size_t{0});
    std::sort(rot.begin(), rot.end(), [&](std::size_t a, std::size_t b) {
        for (std::size_t k = 0; k < n; ++k) {
            const char ca = s[(a + k) % n], cb = s[(b + k) % n];
            if (ca != cb) return ca < cb;
        }
        return a < b;
    });
    r.transformed.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        r.transformed[i] = s[(rot[i] + n - 1) % n];  // last char of rotation i
        if (rot[i] == 0) r.index = i;                // where the original row sorted to
    }
    return r;
}

/// \brief Invert the BWT, recovering the original string from (last column, index).
inline std::string bwt_inverse(const BWTResult& r) {
    const std::size_t n = r.transformed.size();
    if (n == 0) return "";
    // First-column offsets: starts[c] = number of characters < c in the string.
    std::vector<int> counts(256, 0);
    for (unsigned char c : r.transformed) ++counts[c];
    std::vector<int> starts(256, 0);
    for (int c = 1; c < 256; ++c) starts[c] = starts[c - 1] + counts[c - 1];

    // LF-mapping: LF[i] = starts[L[i]] + (# earlier occurrences of L[i] in L).
    std::vector<std::size_t> lf(n);
    std::vector<int>         occ(256, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const unsigned char c = static_cast<unsigned char>(r.transformed[i]);
        lf[i]                 = static_cast<std::size_t>(starts[c]) + static_cast<std::size_t>(occ[c]);
        ++occ[c];
    }
    // Walk the LF-mapping from the original row, prepending each recovered character.
    std::string out(n, '\0');
    std::size_t p = r.index;
    for (std::size_t k = 0; k < n; ++k) {
        out[n - 1 - k] = r.transformed[p];
        p              = lf[p];
    }
    return out;
}

}  // namespace datamunge::algorithms
