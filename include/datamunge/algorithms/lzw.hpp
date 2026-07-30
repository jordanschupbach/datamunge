#pragma once

/// \file lzw.hpp
/// \brief LZW (Lempel-Ziv-Welch) dictionary compression (Welch 1984).
///
/// LZW builds a dictionary of strings *on the fly*, so -- unlike LZ77 -- it never has to
/// transmit match offsets or lengths, only integer codes. The dictionary starts with the
/// 256 single bytes. The encoder reads the longest current string \c w that is already in
/// the dictionary, outputs its code, then adds \c w+c (the string plus the next byte) as a
/// new dictionary entry, and continues from \c c. The decoder rebuilds the identical
/// dictionary as it goes, so nothing about it need be transmitted. LZW is the compression
/// behind GIF and TIFF and the classic Unix =compress=; it excels on data with many
/// repeated substrings and needs only one pass and a hash map.

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

/// \brief Compress \p data to a sequence of integer LZW codes (dictionary seeded with 0-255).
inline std::vector<int> lzw_encode(const std::string& data) {
    std::unordered_map<std::string, int> dict;
    for (int i = 0; i < 256; ++i) dict[std::string(1, static_cast<char>(i))] = i;
    int              next_code = 256;
    std::vector<int> out;
    std::string      w;
    for (char cch : data) {
        std::string wc = w + cch;
        if (dict.count(wc)) {
            w = wc;
        } else {
            out.push_back(dict[w]);       // emit code for the longest known prefix
            dict[wc] = next_code++;       // learn the new string w+c
            w        = std::string(1, cch);
        }
    }
    if (!w.empty()) out.push_back(dict[w]);
    return out;
}

/// \brief Decompress a sequence of LZW codes back to the original bytes.
inline std::string lzw_decode(const std::vector<int>& codes) {
    if (codes.empty()) return "";
    std::unordered_map<int, std::string> dict;
    for (int i = 0; i < 256; ++i) dict[i] = std::string(1, static_cast<char>(i));
    int         next_code = 256;
    std::string w         = dict[codes[0]];
    std::string out       = w;
    for (std::size_t k = 1; k < codes.size(); ++k) {
        const int   code = codes[k];
        std::string entry;
        if (dict.count(code)) {
            entry = dict[code];
        } else if (code == next_code) {
            entry = w + w[0];  // the special "code not yet in dictionary" case
        } else {
            return out;  // corrupt stream
        }
        out += entry;
        dict[next_code++] = w + entry[0];  // rebuild the same dictionary entry the encoder made
        w                 = entry;
    }
    return out;
}

}  // namespace datamunge::algorithms
