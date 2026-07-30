#pragma once

/// \file shannon_fano.hpp
/// \brief Shannon-Fano coding: a top-down prefix code (Shannon 1948; Fano 1949).
///
/// Shannon-Fano coding, the predecessor of Huffman coding, builds a prefix code by
/// *recursive top-down splitting*: sort the symbols by frequency, split them into two
/// groups of as-nearly-equal total probability as possible, assign 0 to one group and 1
/// to the other, and recurse within each group. The result is a prefix-free code whose
/// average length is close to the entropy -- though, unlike Huffman's bottom-up merge, it
/// is *not always optimal* (the greedy split can be slightly worse). This report's
/// implementation reproduces that classic top-down construction. Shannon-Fano is of
/// mainly historical and pedagogical importance, showing both the power of prefix codes
/// and why Huffman's optimality was the improvement that stuck.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A built Shannon-Fano code.
struct ShannonFanoCode {
    std::unordered_map<char, std::string> codes;  ///< symbol -> binary codeword.
};

namespace detail {
inline void shannon_fano_split(std::vector<std::pair<char, std::size_t>>& syms, std::size_t lo, std::size_t hi,
                               ShannonFanoCode& code) {
    if (hi - lo <= 1) return;  // single symbol: its code is already assigned by the parent
    // Total frequency in [lo, hi) and the split point minimizing |left - right|.
    std::size_t total = 0;
    for (std::size_t i = lo; i < hi; ++i) total += syms[i].second;
    std::size_t acc = 0, best_split = lo + 1;
    long        best_diff = -1;
    for (std::size_t i = lo; i + 1 < hi; ++i) {
        acc += syms[i].second;
        const long diff = std::labs(static_cast<long>(2 * acc) - static_cast<long>(total));
        if (best_diff < 0 || diff < best_diff) {
            best_diff  = diff;
            best_split = i + 1;
        }
    }
    for (std::size_t i = lo; i < best_split; ++i) code.codes[syms[i].first] += '0';
    for (std::size_t i = best_split; i < hi; ++i) code.codes[syms[i].first] += '1';
    shannon_fano_split(syms, lo, best_split, code);
    shannon_fano_split(syms, best_split, hi, code);
}
}  // namespace detail

/// \brief Build a Shannon-Fano code from the symbol frequencies of \p text.
inline ShannonFanoCode shannon_fano_build(const std::string& text) {
    ShannonFanoCode           code;
    std::map<char, std::size_t> freq;
    for (char c : text) ++freq[c];
    if (freq.empty()) return code;
    if (freq.size() == 1) { code.codes[freq.begin()->first] = "0"; return code; }
    // Symbols sorted by descending frequency.
    std::vector<std::pair<char, std::size_t>> syms(freq.begin(), freq.end());
    std::sort(syms.begin(), syms.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    for (auto& s : syms) code.codes[s.first] = "";
    detail::shannon_fano_split(syms, 0, syms.size(), code);
    return code;
}

/// Encode \p text with a built Shannon-Fano code.
inline std::string shannon_fano_encode(const ShannonFanoCode& code, const std::string& text) {
    std::string out;
    for (char c : text) out += code.codes.at(c);
    return out;
}

/// Average code length in bits per symbol for \p text under a built code.
inline double shannon_fano_average_bits(const ShannonFanoCode& code, const std::string& text) {
    if (text.empty()) return 0.0;
    std::size_t total = 0;
    for (char c : text) total += code.codes.at(c).size();
    return static_cast<double>(total) / static_cast<double>(text.size());
}

}  // namespace datamunge::algorithms
