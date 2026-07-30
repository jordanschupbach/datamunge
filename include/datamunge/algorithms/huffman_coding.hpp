#pragma once

/// \file huffman_coding.hpp
/// \brief Huffman coding: optimal prefix-free codes for lossless compression
///        (Huffman 1952).
///
/// To compress a stream of symbols one assigns each a binary codeword; to decode
/// unambiguously the codewords must be *prefix-free* (no codeword is a prefix of
/// another). Among all prefix codes, *Huffman's* is provably *optimal* -- it minimizes
/// the expected code length \f$\sum_s p_s \ell_s\f$ -- and it is built by a beautiful
/// greedy rule: repeatedly take the two least-frequent symbols (or subtrees) and merge
/// them under a new parent whose frequency is their sum, until one tree remains. The
/// path from the root to each leaf (left = 0, right = 1) is that symbol's codeword, so
/// frequent symbols end up near the root with short codes. The average code length
/// lands within one bit of the source entropy \f$H=-\sum_s p_s\log_2 p_s\f$, the
/// information-theoretic lower bound.

#include <cstddef>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A built Huffman code: the per-symbol codewords and the tree used to decode.
struct HuffmanCode {
    std::unordered_map<char, std::string> codes;  ///< symbol -> binary codeword.

    struct Node {
        char        symbol = 0;    ///< Valid only at leaves.
        bool        leaf   = false;
        int         left = -1, right = -1;
    };
    std::vector<Node> tree;  ///< Decoding tree; root is the last element (or index root_).
    int               root = -1;
};

/// \brief Build an optimal Huffman code from symbol frequencies in \p text.
inline HuffmanCode huffman_build(const std::string& text) {
    HuffmanCode hc;
    std::unordered_map<char, std::size_t> freq;
    for (char ch : text) ++freq[ch];
    if (freq.empty()) return hc;

    // Min-heap of (frequency, node index), tie-broken by index for determinism.
    using Item = std::pair<std::size_t, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    for (const auto& [ch, f] : freq) {
        const int idx = static_cast<int>(hc.tree.size());
        hc.tree.push_back({ch, true, -1, -1});
        pq.push({f, idx});
    }
    // Single-symbol input: still give it a 1-bit code.
    if (pq.size() == 1) {
        const int only = pq.top().second;
        hc.root        = only;
        hc.codes[hc.tree[only].symbol] = "0";
        return hc;
    }
    while (pq.size() > 1) {
        const auto [fa, ia] = pq.top(); pq.pop();
        const auto [fb, ib] = pq.top(); pq.pop();
        const int  parent = static_cast<int>(hc.tree.size());
        hc.tree.push_back({0, false, ia, ib});
        pq.push({fa + fb, parent});
    }
    hc.root = pq.top().second;

    // Assign codewords by DFS (left = 0, right = 1).
    std::vector<std::pair<int, std::string>> stack = {{hc.root, ""}};
    while (!stack.empty()) {
        auto [idx, code] = stack.back();
        stack.pop_back();
        const auto& nd = hc.tree[idx];
        if (nd.leaf) {
            hc.codes[nd.symbol] = code;
        } else {
            stack.push_back({nd.left, code + "0"});
            stack.push_back({nd.right, code + "1"});
        }
    }
    return hc;
}

/// Encode \p text to a bit string using a built Huffman code.
inline std::string huffman_encode(const HuffmanCode& hc, const std::string& text) {
    std::string out;
    for (char ch : text) out += hc.codes.at(ch);
    return out;
}

/// Decode a bit string back to the original text by walking the tree.
inline std::string huffman_decode(const HuffmanCode& hc, const std::string& bits) {
    std::string out;
    if (hc.root < 0) return out;
    // Degenerate single-symbol tree: every bit is that symbol.
    if (hc.tree[hc.root].leaf) {
        for (std::size_t i = 0; i < bits.size(); ++i) out += hc.tree[hc.root].symbol;
        return out;
    }
    int idx = hc.root;
    for (char b : bits) {
        idx = (b == '0') ? hc.tree[idx].left : hc.tree[idx].right;
        if (hc.tree[idx].leaf) {
            out += hc.tree[idx].symbol;
            idx = hc.root;
        }
    }
    return out;
}

/// Average code length in bits per symbol for \p text under a built code.
inline double huffman_average_bits(const HuffmanCode& hc, const std::string& text) {
    if (text.empty()) return 0.0;
    std::size_t total = 0;
    for (char ch : text) total += hc.codes.at(ch).size();
    return static_cast<double>(total) / static_cast<double>(text.size());
}

}  // namespace datamunge::algorithms
