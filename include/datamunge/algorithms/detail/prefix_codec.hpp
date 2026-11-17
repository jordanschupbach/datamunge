#pragma once
/// @file prefix_codec.hpp
/// @brief Private shared checked trie and actual word-emission producers for static prefix codes.
#include <array>
#include <datamunge/algorithms/prefix_code_types.hpp>
#include <limits>
#include <stdexcept>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms::prefix_detail {
/// @brief Missing child or symbol marker.
inline constexpr std::size_t missing = std::numeric_limits<std::size_t>::max();
/// @brief Owned trie node with optional emitted symbol.
struct Node {
  std::array<std::size_t, 2> child{missing, missing}; ///< Actual zero and one child addresses.
  std::size_t                symbol{missing};         ///< Actual leaf symbol or missing.
};
/// @brief Checked incremental size admission.
/// @param used Existing units.
/// @param count Next units.
/// @param cap Zero or bound.
inline void admit(std::size_t used, std::size_t count, std::size_t cap) {
  if (count > missing - used || (cap && (used > cap || count > cap - used)))
    throw std::length_error("prefix codec: capacity/policy");
}
/// @brief Validate all nonempty words as a prefix-free code and build its actual trie.
/// @param words Borrowed codewords; empty means inactive symbol.
/// @param max_nodes Complete actual node cap.
/// @param max_bits Complete sum of codeword-length cap.
/// @return Owned validated trie, root at zero even for an empty code.
inline std::vector<Node>
tree(const std::vector<std::vector<bool>>& words, std::size_t max_nodes, std::size_t max_bits) {
  std::vector<Node> nodes(1);
  admit(0, 1, max_nodes);
  std::size_t bits = 0;
  for (std::size_t symbol = 0; symbol < words.size(); ++symbol) {
    if (words[symbol].empty())
      continue;
    admit(bits, words[symbol].size(), max_bits);
    bits += words[symbol].size();
    std::size_t node = 0;
    for (bool bit : words[symbol]) {
      if (nodes[node].symbol != missing)
        throw std::invalid_argument("prefix codec: prefix conflict");
      auto next = nodes[node].child[bit];
      if (next == missing) {
        admit(nodes.size(), 1, max_nodes);
        next                   = nodes.size();
        nodes[node].child[bit] = next;
        nodes.emplace_back();
      }
      node = next;
    }
    if (nodes[node].symbol != missing || nodes[node].child[0] != missing ||
        nodes[node].child[1] != missing)
      throw std::invalid_argument("prefix codec: duplicate/prefix word");
    nodes[node].symbol = symbol;
  }
  return nodes;
}
/// @brief Owned actual payload and emitted-word records from one static codec pass.
struct Result {
  std::vector<bool>           bits;         ///< Actual encoded bits, encoding only.
  std::vector<int>            symbols;      ///< Actual decoded symbols, decoding only.
  std::vector<PrefixWordSpan> spans;        ///< Actual optional complete-word spans.
  std::size_t                 bit_visits{}; ///< Actual copied or traversed bit count.
  std::size_t                 nodes{};      ///< Actual validated decoding trie size.
};
/// @brief Validate complete geometry and encode actual source symbols once.
/// @param words Borrowed prefix codewords.
/// @param symbols Borrowed source ids.
/// @param max_bits Complete codebook/payload bit cap.
/// @param max_symbols Complete source symbol cap.
/// @param max_nodes Complete trie cap.
/// @param capture Retain actual emitted spans.
/// @param max_trace Actual emitted-word record cap.
/// @return Owned actual encoding/work/spans.
inline Result encode(
    const std::vector<std::vector<bool>>& words,
    const std::vector<int>&               symbols,
    std::size_t                           max_bits,
    std::size_t                           max_symbols,
    std::size_t                           max_nodes,
    bool                                  capture,
    std::size_t                           max_trace) {
  admit(0, symbols.size(), max_symbols);
  if (capture)
    admit(0, symbols.size(), max_trace);
  const auto  nodes = tree(words, max_nodes, max_bits);
  std::size_t size  = 0;
  for (int symbol : symbols) {
    if (symbol < 0 || static_cast<std::size_t>(symbol) >= words.size() || words[symbol].empty())
      throw std::invalid_argument("prefix codec: absent source symbol");
    admit(size, words[symbol].size(), max_bits);
    size += words[symbol].size();
  }
  Result out;
  out.nodes = nodes.size();
  out.bits.reserve(size);
  for (int symbol : symbols) {
    const auto begin = out.bits.size();
    for (bool bit : words[symbol]) {
      out.bits.push_back(bit);
      ++out.bit_visits;
    }
    if (capture)
      out.spans.push_back({static_cast<std::size_t>(symbol), begin, out.bits.size()});
  }
  return out;
}
/// @brief Traverse the complete actual bit stream and require exactly the requested number of complete words.
/// @param words Borrowed prefix codewords.
/// @param bits Borrowed complete payload, no ignored tail.
/// @param count Required complete decoded symbols.
/// @param max_bits Complete codebook/payload bit cap.
/// @param max_symbols Complete decoded symbol cap.
/// @param max_nodes Complete trie cap.
/// @param capture Retain actual emitted spans.
/// @param max_trace Actual emitted-word record cap.
/// @return Owned actual decoded symbols/work/spans.
inline Result decode(
    const std::vector<std::vector<bool>>& words,
    const std::vector<bool>&              bits,
    std::size_t                           count,
    std::size_t                           max_bits,
    std::size_t                           max_symbols,
    std::size_t                           max_nodes,
    bool                                  capture,
    std::size_t                           max_trace) {
  admit(0, bits.size(), max_bits);
  admit(0, count, max_symbols);
  if (capture)
    admit(0, count, max_trace);
  const auto nodes = tree(words, max_nodes, max_bits);
  Result     out;
  out.nodes = nodes.size();
  out.symbols.reserve(count);
  std::size_t node = 0, begin = 0;
  for (std::size_t i = 0; i < bits.size(); ++i) {
    node = nodes[node].child[bits[i]];
    ++out.bit_visits;
    if (node == missing)
      throw std::invalid_argument("prefix codec: absent branch");
    if (nodes[node].symbol != missing) {
      if (out.symbols.size() >= count)
        throw std::invalid_argument("prefix codec: extra complete payload");
      if (nodes[node].symbol > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("prefix codec: int symbol capacity");
      out.symbols.push_back(static_cast<int>(nodes[node].symbol));
      if (capture)
        out.spans.push_back({nodes[node].symbol, begin, i + 1});
      begin = i + 1;
      node  = 0;
    }
  }
  if (node || out.symbols.size() != count)
    throw std::invalid_argument("prefix codec: truncated/wrong-count payload");
  return out;
}
} // namespace datamunge::algorithms::prefix_detail
