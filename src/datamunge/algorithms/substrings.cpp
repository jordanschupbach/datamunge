/// @file substrings.cpp
/// @brief Canonical producing kernels and native report diagnostics.
#include <cstddef>
#include <datamunge/algorithms/substrings.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Implement the checked canonical build_suffix_tree_work producer; contracts are declared in the matching header.
SuffixTreeWork build_suffix_tree_work(const std::string& text) {
  if (text.size() > 1000000)
    throw std::length_error("suffix tree: text limit");
  SuffixTreeWork work;
  auto&          tree = work.tree;
  tree.text           = text;
  const int n         = static_cast<int>(text.size());
  auto&     nds       = tree.nodes;

  // The character at position i; position n is a unique virtual terminal (code 256).
  auto char_at = [&](int i) -> int {
    return i < n ? static_cast<int>(static_cast<unsigned char>(text[i])) : 256;
  };
  auto new_node = [&](int start, int end) -> int {
    nds.push_back(SuffixTreeNode{start, end, -1, {}});
    return static_cast<int>(nds.size()) - 1;
  };

  const int root        = new_node(-1, -1);
  int       active_node = root;
  int       active_edge = 0;
  int       active_len  = 0;
  int       remaining   = 0;
  int       leaf_end    = -1;
  int       need_link   = -1;
  auto      add_link    = [&](int node) {
    if (need_link != -1)
      nds[need_link].suffix_link = node;
    need_link = node;
  };
  auto edge_len = [&](int node) -> int {
    const int e = nds[node].end == -1 ? leaf_end : nds[node].end;
    return e - nds[node].start + 1;
  };

  for (int pos = 0; pos <= n; ++pos) { // position n is the virtual terminal
    ++work.phases;
    leaf_end = pos;
    remaining += 1;
    need_link = -1;
    while (remaining > 0) {
      ++work.extensions;
      if (active_len == 0)
        active_edge = pos;
      const int c  = char_at(active_edge);
      auto      it = nds[active_node].children.find(c);
      if (it == nds[active_node].children.end()) {
        const int leaf               = new_node(pos, -1); // rule 2: new leaf edge
        nds[active_node].children[c] = leaf;
        add_link(active_node);
      } else {
        const int nxt = it->second;
        if (active_len >= edge_len(nxt)) { // walk down: hop to the next node
          ++work.walk_downs;
          active_edge += edge_len(nxt);
          active_len -= edge_len(nxt);
          active_node = nxt;
          continue;
        }
        if (char_at(nds[nxt].start + active_len) == char_at(pos)) { // rule 3: already present
          active_len += 1;
          add_link(active_node);
          break;
        }
        // Split the edge at active_len and hang a new leaf.
        ++work.splits;
        const int split              = new_node(nds[nxt].start, nds[nxt].start + active_len - 1);
        nds[active_node].children[c] = split;
        const int leaf               = new_node(pos, -1);
        nds[split].children[char_at(pos)] = leaf;
        nds[nxt].start += active_len;
        nds[split].children[char_at(nds[nxt].start)] = nxt;
        add_link(split);
      }
      remaining -= 1;
      if (active_node == root && active_len > 0) {
        active_len -= 1;
        active_edge = pos - remaining + 1;
      } else if (active_node != root) {
        active_node = nds[active_node].suffix_link != -1 ? nds[active_node].suffix_link : root;
      }
    }
    work.nodes_after_phase.push_back(nds.size());
  }
  return work;
}
/// @brief Implement the checked canonical build_suffix_tree producer; contracts are declared in the matching header.
SuffixTree build_suffix_tree(const std::string& text) {
  return build_suffix_tree_work(text).tree;
}

bool suffix_tree_contains(const SuffixTree& tree, const std::string& pattern) {
  if (pattern.empty())
    return true;
  const int n       = static_cast<int>(tree.text.size());
  auto      char_at = [&](int i) -> int {
    return i < n ? static_cast<int>(static_cast<unsigned char>(tree.text[i])) : 256;
  };
  int         node = 0;
  std::size_t ip   = 0;
  while (ip < pattern.size()) {
    const int c  = static_cast<int>(static_cast<unsigned char>(pattern[ip]));
    auto      it = tree.nodes[static_cast<std::size_t>(node)].children.find(c);
    if (it == tree.nodes[static_cast<std::size_t>(node)].children.end())
      return false;
    const int e    = it->second;
    int       k    = tree.nodes[static_cast<std::size_t>(e)].start;
    const int eend = tree.nodes[static_cast<std::size_t>(e)].end == -1
                         ? n
                         : tree.nodes[static_cast<std::size_t>(e)].end;
    while (k <= eend && ip < pattern.size()) {
      if (char_at(k) != static_cast<int>(static_cast<unsigned char>(pattern[ip])))
        return false;
      ++k;
      ++ip;
    }
    node = e;
  }
  return true;
}

std::size_t distinct_substring_count(const SuffixTree& tree) {
  const int n      = static_cast<int>(tree.text.size());
  long long sum    = 0;
  long long leaves = 0;
  for (std::size_t idx = 1; idx < tree.nodes.size(); ++idx) { // skip the root
    const SuffixTreeNode& nd = tree.nodes[idx];
    const int             e  = nd.end == -1 ? n : nd.end;
    sum += (e - nd.start + 1);
    if (nd.children.empty())
      ++leaves;
  }
  return static_cast<std::size_t>(sum - leaves);
}

/// @brief Implement the checked canonical suffix_tree_plot producer; contracts are declared in the matching header.
graphics::RPlot suffix_tree_plot(const SuffixTree& tree) {
  std::size_t         n = tree.nodes.size();
  std::vector<double> x(n), y(n);
  std::vector<int>    parent(n, -1), order, pending{0};
  // Depth-first order keeps every subtree's leaves contiguous in the figure.
  while (!pending.empty()) {
    const int u = pending.back();
    pending.pop_back();
    order.push_back(u);
    for (auto it = tree.nodes[u].children.rbegin(); it != tree.nodes[u].children.rend(); ++it) {
      const int v = it->second;
      parent[v]   = u;
      y[v]        = y[u] - 1;
      pending.push_back(v);
    }
  }
  double leaf = 0;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    int u = *it;
    if (tree.nodes[u].children.empty())
      x[u] = leaf++;
    else {
      double sum = 0;
      for (auto [c, v] : tree.nodes[u].children)
        sum += x[v];
      x[u] = sum / tree.nodes[u].children.size();
    }
  }
  auto p = graphics::RPlot::create();
  for (std::size_t i = 1; i < n; ++i) {
    int u = parent[i];
    p.line({x[u], x[i]}, {y[u], y[i]}, "", {148, 163, 184});
    auto        node = tree.nodes[i];
    std::string label;
    int         end = node.end < 0 ? int(tree.text.size()) : node.end;
    for (int j = node.start; j <= end; ++j)
      label += j == int(tree.text.size()) ? '$' : tree.text[j];
    p.text((x[u] + x[i]) / 2, (y[u] + y[i]) / 2 + .1, label, {161, 98, 7}, 13);
  }
  p.points(x, y, "actual nodes", {37, 99, 235}, 8);
  p.title("Actual suffix tree; $ is virtual terminal")
      .x_label("leaf layout")
      .y_label("negative edge depth");
  return p;
}
} // namespace datamunge::algorithms
