#pragma once

/// \file lz77.hpp
/// \brief LZ77 sliding-window dictionary compression (Ziv & Lempel 1977).
///
/// LZ77 compresses by replacing repeated substrings with *back-references* into a
/// sliding window of recently seen text. At each position it finds the longest match
/// starting there that also occurs within the preceding window, and emits a token
/// \f$(\text{offset}, \text{length}, \text{next})\f$: go back \c offset characters, copy
/// \c length of them, then append the literal \c next. When there is no useful match it
/// emits \f$(0,0,c)\f$ for the literal \c c. Decoding just replays the copies. Because a
/// back-reference can point at text it is *currently producing*, LZ77 naturally encodes
/// runs and periodic data. It is the ancestor of DEFLATE (zip, gzip, PNG), LZMA, and
/// most modern general-purpose compressors.

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// One LZ77 token: copy \c length bytes from \c offset back, then (if \c literal) emit \c next.
struct LZ77Token {
    int  offset;         ///< Distance back into the window (0 for a pure literal).
    int  length;         ///< Match length (0 for a pure literal).
    char next;           ///< Literal byte following the match (valid only if \c literal).
    bool literal = true; ///< False when a match runs to the end of input (no trailing literal).
};

/// \brief Compress \p data with LZ77 using a sliding window of \p window bytes and a
///        maximum match length of \p max_match.
inline std::vector<LZ77Token> lz77_encode(const std::string& data, int window = 4096, int max_match = 255) {
    std::vector<LZ77Token> out;
    const int              n = static_cast<int>(data.size());
    int                    i = 0;
    while (i < n) {
        int       best_len = 0, best_off = 0;
        const int start = std::max(0, i - window);
        for (int j = start; j < i; ++j) {  // candidate match starting at j
            int len = 0;
            while (len < max_match && i + len < n && data[j + len] == data[i + len]) ++len;
            if (len > best_len) {
                best_len = len;
                best_off = i - j;
            }
        }
        if (i + best_len < n) {
            out.push_back({best_off, best_len, data[i + best_len], true});
            i += best_len + 1;
        } else {
            // Match (or literal) reaches the end; emit with no trailing literal.
            out.push_back({best_off, best_len, '\0', false});
            i += (best_len > 0 ? best_len : 1);
        }
    }
    return out;
}

/// \brief Decompress an LZ77 token stream back to the original bytes.
inline std::string lz77_decode(const std::vector<LZ77Token>& tokens) {
    std::string out;
    for (const LZ77Token& t : tokens) {
        if (t.length > 0) {
            const std::size_t startpos = out.size() - static_cast<std::size_t>(t.offset);
            for (int k = 0; k < t.length; ++k) out.push_back(out[startpos + k]);  // may overlap into new output
        }
        if (t.literal) out.push_back(t.next);
    }
    return out;
}

}  // namespace datamunge::algorithms
