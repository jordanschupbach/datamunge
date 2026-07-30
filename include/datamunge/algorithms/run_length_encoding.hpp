#pragma once

/// \file run_length_encoding.hpp
/// \brief Run-length encoding (RLE): the simplest lossless compression, replacing
///        runs of a repeated symbol with a (symbol, count) pair.
///
/// Run-length encoding exploits *runs* -- maximal stretches of the same symbol. Each
/// run \f$c\,c\,\dots\,c\f$ of length \f$\ell\f$ is stored as the pair \f$(c,\ell)\f$,
/// so data with long uniform stretches (scanned images, simple graphics, sensor
/// plateaus, the middle stage of fax and BWT-based compressors) shrinks dramatically,
/// while data with no runs *expands* (two entries per symbol). It is exact and its
/// decode is trivial -- emit each symbol \f$\ell\f$ times. RLE is the textbook example
/// of exploiting redundancy and a building block inside PackBits, fax Group 3/4, and
/// the back end of the Burrows-Wheeler transform.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A single run: a symbol and how many times it repeats.
struct Run {
    char        symbol;
    std::size_t count;
};

/// \brief Encode \p text into a list of runs.
inline std::vector<Run> rle_encode(const std::string& text) {
    std::vector<Run> runs;
    for (std::size_t i = 0; i < text.size();) {
        std::size_t j = i;
        while (j < text.size() && text[j] == text[i]) ++j;
        runs.push_back({text[i], j - i});
        i = j;
    }
    return runs;
}

/// \brief Decode a list of runs back into the original string.
inline std::string rle_decode(const std::vector<Run>& runs) {
    std::string out;
    for (const Run& r : runs) out.append(r.count, r.symbol);
    return out;
}

/// Encoded size in "cells" (one per run: a symbol plus a count) -- the compressed length.
inline std::size_t rle_encoded_cells(const std::vector<Run>& runs) { return runs.size(); }

}  // namespace datamunge::algorithms
