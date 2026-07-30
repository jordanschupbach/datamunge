#pragma once

/// \file move_to_front.hpp
/// \brief The move-to-front (MTF) transform: recode a byte stream so that
///        recently-used symbols get small indices.
///
/// MTF keeps a list of the alphabet and, for each input byte, outputs its *current
/// position* in the list, then moves that byte to the front. Data with local repetition
/// (especially the output of a Burrows-Wheeler transform, where identical characters
/// cluster) therefore turns into a stream dominated by *small numbers* -- long runs
/// become runs of zeros -- which run-length and entropy coders then compress well. It is
/// exactly reversible: the decoder maintains the same list and reverses each move. MTF is
/// the middle stage of the bzip2 pipeline (BWT -> MTF -> RLE -> Huffman).

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// \brief Encode a byte string to move-to-front indices (alphabet 0-255).
inline std::vector<int> mtf_encode(const std::string& data) {
    std::array<unsigned char, 256> list;
    for (int i = 0; i < 256; ++i) list[i] = static_cast<unsigned char>(i);
    std::vector<int> out;
    out.reserve(data.size());
    for (unsigned char c : data) {
        int pos = 0;
        while (list[pos] != c) ++pos;   // find the byte's current position
        out.push_back(pos);
        for (int k = pos; k > 0; --k) list[k] = list[k - 1];  // shift up
        list[0] = c;                    // move to front
    }
    return out;
}

/// \brief Decode move-to-front indices back to the original byte string.
inline std::string mtf_decode(const std::vector<int>& indices) {
    std::array<unsigned char, 256> list;
    for (int i = 0; i < 256; ++i) list[i] = static_cast<unsigned char>(i);
    std::string out;
    out.reserve(indices.size());
    for (int pos : indices) {
        const unsigned char c = list[pos];
        out.push_back(static_cast<char>(c));
        for (int k = pos; k > 0; --k) list[k] = list[k - 1];
        list[0] = c;
    }
    return out;
}

}  // namespace datamunge::algorithms
