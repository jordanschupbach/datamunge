#pragma once
/// @file huffman_analysis.hpp
/// @brief Native Huffman fixture, numerical summaries and bounded actual-tree diagrams.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/huffman_coding.hpp>
#include <datamunge/plot/diagram.hpp>
#include <datamunge/plot/plot.hpp>
#include <limits>
#include <set>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Return the original two-sentence89-byte source, including its joining space.
/// Owned ASCII bytes, no external data, shared state or retained references.
/// @return Original fixed text with all26 letters, spaces and periods.
/// @throws std::bad_alloc On string allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::huffman_example_text().size()); // 89
/// @endcode
/// @see huffman_build_trace()
inline std::string huffman_example_text() {
  return "the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy "
         "dog.";
}
/// @brief Computed entropy, weighted mean and Kraft diagnostics for supplied counts/lengths.
/// Values use long-double accumulation then double conversion, not exact
/// certificates or proof of optimality/prefix validity. Empty support gives zeros.
/// @par Example
/// @code
/// const dm::algorithms::HuffmanSummary s;std::print("{}\n",s.active_symbols); // 0
/// @endcode
struct HuffmanSummary {
  std::size_t active_symbols{0}; ///< Actual positive-count symbols.
  double entropy_bits{0}; ///< Computed empirical -sum p*log2(p), zero for empty/singleton support.
  double mean_bits{0};    ///< Computed sum p*length, singleton positive convention yields1.
  double kraft_sum{0};    ///< Computed sum2^-length over positive symbols; can round/underflow.
};
/// @brief Summarize complete positive-count lengths without rerunning construction.
/// Requires equal dimensions, zero length exactly for zero count and positive
/// lengths for positive counts. Does not assert those lengths form a prefix
/// code or are optimal. Supports every uint64 input through finite long-double
/// totals; O(K) work, O(1) storage, no mutation/retained references. Numerical
/// diagnostics can round, especially highly unbalanced or very deep inputs.
/// @param counts Borrowed complete original-index uint64 counts.
/// @param lengths Borrowed corresponding positive/absent lengths.
/// @return Computed support/entropy/weighted mean/Kraft diagnostics.
/// @throws std::invalid_argument On dimension or support/length-domain mismatch.
/// @par Example
/// @code
/// const auto s=dm::algorithms::huffman_summary({1,1},{1,1});
/// std::print("{} {}\n",s.entropy_bits,s.mean_bits); // 1 1
/// @endcode
/// @see huffman_counts_trace(), HuffmanSummary
inline HuffmanSummary huffman_summary(
    const std::vector<std::uint64_t>& counts,
    const std::vector<int>&           lengths) {
  if (counts.size() != lengths.size())
    throw std::invalid_argument("Huffman summary dimensions");
  long double    total = 0;
  HuffmanSummary out;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if ((counts[i] == 0 && lengths[i] != 0) || (counts[i] > 0 && lengths[i] <= 0))
      throw std::invalid_argument("Huffman summary support/length");
    total += static_cast<long double>(counts[i]);
    out.active_symbols += counts[i] > 0;
  }
  if (out.active_symbols == 0)
    return out;
  long double entropy = 0, mean = 0, kraft = 0;
  for (std::size_t i = 0; i < counts.size(); ++i)
    if (counts[i]) {
      const long double p = static_cast<long double>(counts[i]) / total;
      entropy -= p * std::log2(p);
      mean += p * lengths[i];
      kraft += std::ldexp(1.L, -lengths[i]);
    }
  out.entropy_bits = static_cast<double>(entropy);
  out.mean_bits    = static_cast<double>(mean);
  out.kraft_sum    = static_cast<double>(kraft);
  return out;
}
/// @brief Draw actual producing count-tree topology and exact-weight labels in a native diagram.
/// Checks dimensions/basic nodes, admits the actual complete leaf count before
/// layout allocation, then verifies connectivity/unique positive leaves/depths
/// and merge/topology agreement. It does not independently prove heap minimality
/// for manually modified trace data. Labels longer than12 digits are abbreviated;
/// full exact weights remain in records. Empty support gets a labelled empty node.
/// O(K+nodes) checking/layout work beyond decimal-label lengths, O(nodes) owned
/// layout/render storage. Leaf x positions follow actual left-to-right DFS order;
/// each parent is centred between its children, with y from actual depth.
/// @param trace Borrowed actual complete count trace, never mutated or retained.
/// @param max_leaves Complete displayed leaf cap, default16; zero disables policy.
/// @return Owned native Diagram; editable report code chooses the output file.
/// @throws std::invalid_argument On malformed shape/topology/merge/length metadata.
/// @throws std::length_error On complete display quota or index/count limits.
/// @throws std::bad_alloc On layout/diagram allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::huffman_counts_trace({5,2,1});
/// const auto d=dm::algorithms::huffman_tree_diagram(t);
/// std::print("{}\n",t.merges.size()); // 2
/// @endcode
/// @see graphics::Diagram, huffman_counts_trace()
inline graphics::Diagram huffman_tree_diagram(
    const HuffmanCountsTrace& trace,
    std::size_t               max_leaves = 16) {
  const auto& r = trace.run;
  const auto  n = r.nodes.size();
  if (r.lengths.size() != trace.counts.size())
    throw std::invalid_argument("Huffman diagram dimensions");
  std::size_t positive = 0;
  for (auto c : trace.counts)
    positive += c > 0;
  if (positive != r.active_symbols)
    throw std::invalid_argument("Huffman diagram support");
  graphics::Diagram diagram(65, 1100);
  if (positive == 0) {
    if (n || r.root != kHuffmanNoNode || !trace.merges.empty())
      throw std::invalid_argument("Huffman diagram empty metadata");
    diagram.node("empty", {0, 0}, "Empty alphabet");
    return diagram;
  }
  if (positive > std::numeric_limits<std::size_t>::max() / 2 || n != 2 * positive - 1 ||
      r.root >= n || trace.merges.size() != positive - 1)
    throw std::invalid_argument("Huffman diagram topology size");
  for (const auto& node : r.nodes)
    if (node.leaf) {
      if (node.symbol >= trace.counts.size() || !trace.counts[node.symbol] ||
          node.left != kHuffmanNoNode || node.right != kHuffmanNoNode)
        throw std::invalid_argument("Huffman diagram leaf");
    } else if (node.left >= n || node.right >= n || node.left == node.right)
      throw std::invalid_argument("Huffman diagram children");
  if (max_leaves && positive > max_leaves)
    throw std::length_error("Huffman diagram complete leaf cap");
  std::vector<bool>        seen(n);
  std::vector<std::size_t> depth(n), order;
  std::vector<double>      x(n);
  std::vector<std::size_t> stack{r.root};
  std::set<std::size_t>    symbols;
  double                   next = 0;
  while (!stack.empty()) {
    const auto id = stack.back();
    stack.pop_back();
    if (seen[id])
      throw std::invalid_argument("Huffman diagram repeated node");
    seen[id] = true;
    order.push_back(id);
    const auto& node = r.nodes[id];
    if (node.leaf) {
      if (!symbols.insert(node.symbol).second ||
          r.lengths[node.symbol] != static_cast<int>(depth[id] ? depth[id] : 1))
        throw std::invalid_argument("Huffman diagram leaf/depth");
      x[id] = next;
      next += 4.;
    } else {
      if (depth[id] == std::numeric_limits<std::size_t>::max())
        throw std::length_error("Huffman diagram depth");
      depth[node.left] = depth[node.right] = depth[id] + 1;
      stack.push_back(node.right);
      stack.push_back(node.left);
    }
  }
  if (order.size() != n)
    throw std::invalid_argument("Huffman diagram unreachable nodes");
  for (auto it = order.rbegin(); it != order.rend(); ++it)
    if (!r.nodes[*it].leaf)
      x[*it] = (x[r.nodes[*it].left] + x[r.nodes[*it].right]) / 2;
  std::vector<std::string> weight(n);
  for (std::size_t i = 0; i < n; ++i)
    if (r.nodes[i].leaf)
      weight[i] = std::to_string(trace.counts[r.nodes[i].symbol]);
  for (const auto& m : trace.merges) {
    if (m.parent >= n || r.nodes[m.parent].leaf || r.nodes[m.parent].left != m.left ||
        r.nodes[m.parent].right != m.right || !weight[m.parent].empty() ||
        m.parent_weight.empty() ||
        !std::all_of(m.parent_weight.begin(), m.parent_weight.end(), [](char c) {
          return c >= '0' && c <= '9';
        }))
      throw std::invalid_argument("Huffman diagram merge metadata");
    weight[m.parent] = m.parent_weight;
  }
  graphics::DiagramNodeStyle style;
  style.width     = 2.8;
  style.height    = 1.9;
  style.font_size = 20;
  for (std::size_t i = 0; i < n; ++i) {
    if (weight[i].empty())
      throw std::invalid_argument("Huffman diagram missing weight");
    const auto label = weight[i].size() > 12 ? weight[i].substr(0, 9) + "..." : weight[i];
    const auto text  = r.nodes[i].leaf
                           ? "symbol " + std::to_string(r.nodes[i].symbol) + "\nw=" + label +
                                "; len=" + std::to_string(r.lengths[r.nodes[i].symbol])
                           : "node " + std::to_string(i) + "\nw=" + label;
    diagram.node(std::to_string(i), {x[i], -3.5 * depth[i]}, text, style);
  }
  for (std::size_t i = 0; i < n; ++i)
    if (!r.nodes[i].leaf)
      diagram.edge(std::to_string(i), std::to_string(r.nodes[i].left), "0")
          .edge(std::to_string(i), std::to_string(r.nodes[i].right), "1");
  return diagram;
}
} // namespace datamunge::algorithms
