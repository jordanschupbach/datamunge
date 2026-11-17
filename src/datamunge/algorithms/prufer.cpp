/// @file prufer.cpp
/// @brief Checked tree geometry and one shared smallest-leaf producer per coding direction.
#include <datamunge/algorithms/prufer.hpp>
#include <algorithm>
#include <set>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Local geometry, exact ordered leaves and optional producing capture.
namespace prufer_detail {
/// @brief Check complete requested vertex storage before allocation.
/// @param n Requested vertex count, at least2.
/// @param options Vertex/history policies.
/// @throws std::invalid_argument If vertices/edges/code are malformed, the encoding graph is not a tree, or a code label is out of range.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
void vertices(std::size_t n, PruferOptions options) {
  if (n < 2)
    throw std::invalid_argument("prufer needs two vertices");
  if ((options.max_vertices && n > options.max_vertices) ||
      n > std::vector<std::size_t>{}.max_size())
    throw std::length_error("prufer vertex cap");
}
/// @brief Admit actual leaf-set record copies before allocation.
/// @param used Mutable retained entry count.
/// @param amount Next actual entries.
/// @param cap Positive cap or zero.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
void retain(std::size_t& used, std::size_t amount, std::size_t cap) {
  if (amount > std::numeric_limits<std::size_t>::max() - used ||
      (cap && (used > cap || amount > cap - used)))
    throw std::length_error("prufer history cap");
  used += amount;
}
/// @brief Own an actual ordered leaf-set snapshot under the shared history cap.
/// @param leaves Actual producer set.
/// @param used Mutable admitted entry count.
/// @param cap Positive policy or zero.
/// @return Actual ascending leaf ids.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
std::vector<std::size_t>
snapshot(const std::set<std::size_t>& leaves, std::size_t& used, std::size_t cap) {
  retain(used, leaves.size(), cap);
  return {leaves.begin(), leaves.end()};
}
/// @brief Validate connected tree geometry then execute the single encoding producer.
/// @param n Vertex count.
/// @param edges Borrowed complete undirected tree.
/// @param options Policies.
/// @param capture Whether actual leaf sets are retained.
/// @return Owned actual code/survivors and optional history.
/// @throws std::invalid_argument If vertices/edges/code are malformed, the encoding graph is not a tree, or a code label is out of range.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
/// @throws std::runtime_error If a checked internal leaf invariant fails.
PruferEncoding encode(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    PruferOptions                                           options,
    bool                                                    capture) {
  vertices(n, options);
  if (edges.size() != n - 1)
    throw std::invalid_argument("prufer edge count");
  for (auto [u, v] : edges)
    if (u >= n || v >= n || u == v)
      throw std::invalid_argument("prufer endpoint or self loop");
  std::vector<std::vector<std::size_t>> adj(n);
  std::vector<std::size_t>              degree(n);
  for (auto [u, v] : edges) {
    adj[u].push_back(v);
    adj[v].push_back(u);
    ++degree[u];
    ++degree[v];
  }
  std::vector<char>        seen(n);
  std::vector<std::size_t> queue{0};
  seen[0] = 1;
  for (std::size_t i = 0; i < queue.size(); ++i)
    for (auto v : adj[queue[i]])
      if (!seen[v]) {
        seen[v] = 1;
        queue.push_back(v);
      }
  if (queue.size() != n)
    throw std::invalid_argument("prufer graph not a tree");
  std::set<std::size_t> leaves;
  for (std::size_t i = 0; i < n; ++i)
    if (degree[i] == 1)
      leaves.insert(i);
  PruferEncoding out;
  std::size_t    entries = 0;
  if (capture)
    out.initial_leaves = snapshot(leaves, entries, options.max_trace_entries);
  std::vector<char> removed(n);
  out.sequence.reserve(n - 2);
  for (std::size_t step = 0; step < n - 2; ++step) {
    if (leaves.empty())
      throw std::runtime_error("prufer missing leaf");
    auto       leaf = *leaves.begin();
    PruferStep record{};
    if (capture) {
      retain(entries, 1, options.max_trace_entries);
      record.leaves_before = snapshot(leaves, entries, options.max_trace_entries);
    }
    leaves.erase(leaves.begin());
    removed[leaf]        = 1;
    std::size_t neighbor = n;
    for (auto v : adj[leaf]) {
      ++out.adjacency_entries_examined;
      if (!removed[v]) {
        neighbor = v;
        break;
      }
    }
    if (neighbor == n || degree[neighbor] == 0)
      throw std::runtime_error("prufer missing neighbor");
    auto before = degree[neighbor];
    out.sequence.push_back(neighbor);
    degree[leaf] = 0;
    if (--degree[neighbor] == 1)
      leaves.insert(neighbor);
    if (capture) {
      record.leaf                   = leaf;
      record.neighbor               = neighbor;
      record.neighbor_degree_before = before;
      record.neighbor_degree_after  = degree[neighbor];
      record.final_edge             = false;
      record.leaves_after           = snapshot(leaves, entries, options.max_trace_entries);
      out.steps.push_back(std::move(record));
    }
  }
  for (std::size_t v = 0; v < n; ++v)
    if (!removed[v])
      out.survivors.push_back(v);
  return out;
}
/// @brief Validate the complete code then execute the single decoding producer.
/// @param sequence Borrowed source code.
/// @param options Policies.
/// @param capture Whether actual leaf sets are retained.
/// @return Owned actual decoded edges and optional histories.
/// @throws std::invalid_argument If vertices/edges/code are malformed, the encoding graph is not a tree, or a code label is out of range.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
/// @throws std::runtime_error If a checked internal leaf invariant fails.
PruferDecoding
decode(const std::vector<std::size_t>& sequence, PruferOptions options, bool capture) {
  if (sequence.size() > std::numeric_limits<std::size_t>::max() - 2)
    throw std::length_error("prufer length overflow");
  auto n = sequence.size() + 2;
  vertices(n, options);
  for (auto v : sequence)
    if (v >= n)
      throw std::invalid_argument("prufer code label");
  std::vector<std::size_t> degree(n, 1);
  for (auto v : sequence)
    ++degree[v];
  std::set<std::size_t> leaves;
  for (std::size_t v = 0; v < n; ++v)
    if (degree[v] == 1)
      leaves.insert(v);
  PruferDecoding out;
  std::size_t    entries = 0;
  if (capture)
    out.initial_leaves = snapshot(leaves, entries, options.max_trace_entries);
  out.edges.reserve(n - 1);
  for (auto neighbor : sequence) {
    if (leaves.empty())
      throw std::runtime_error("prufer decode missing leaf");
    auto       leaf = *leaves.begin();
    PruferStep record{};
    if (capture) {
      retain(entries, 1, options.max_trace_entries);
      record.leaves_before = snapshot(leaves, entries, options.max_trace_entries);
    }
    leaves.erase(leaves.begin());
    auto before = degree[neighbor];
    out.edges.emplace_back(leaf, neighbor);
    --degree[leaf];
    if (--degree[neighbor] == 1)
      leaves.insert(neighbor);
    if (capture) {
      record.leaf                   = leaf;
      record.neighbor               = neighbor;
      record.neighbor_degree_before = before;
      record.neighbor_degree_after  = degree[neighbor];
      record.final_edge             = false;
      record.leaves_after           = snapshot(leaves, entries, options.max_trace_entries);
      out.steps.push_back(std::move(record));
    }
  }
  if (leaves.size() != 2)
    throw std::runtime_error("prufer final leaf invariant");
  PruferStep record{};
  if (capture) {
    retain(entries, 1, options.max_trace_entries);
    record.leaves_before = snapshot(leaves, entries, options.max_trace_entries);
  }
  auto leaf = *leaves.begin();
  leaves.erase(leaves.begin());
  auto neighbor = *leaves.begin();
  leaves.erase(leaves.begin());
  out.edges.emplace_back(leaf, neighbor);
  --degree[leaf];
  auto before = degree[neighbor];
  --degree[neighbor];
  if (capture) {
    record.leaf                   = leaf;
    record.neighbor               = neighbor;
    record.neighbor_degree_before = before;
    record.neighbor_degree_after  = degree[neighbor];
    record.final_edge             = true;
    record.leaves_after           = snapshot(leaves, entries, options.max_trace_entries);
    out.steps.push_back(std::move(record));
  }
  return out;
}
} // namespace prufer_detail
std::vector<std::size_t> tree_to_prufer(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges) {
  return prufer_detail::encode(n, tree_edges, {}, false).sequence;
}
PruferEncoding tree_to_prufer_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges,
    PruferOptions                                           options) {
  return prufer_detail::encode(n, tree_edges, options, true);
}
std::vector<std::pair<std::size_t, std::size_t>> prufer_to_tree(
    const std::vector<std::size_t>& sequence) {
  return prufer_detail::decode(sequence, {}, false).edges;
}
PruferDecoding prufer_to_tree_trace(
    const std::vector<std::size_t>& sequence,
    PruferOptions                   options) {
  return prufer_detail::decode(sequence, options, true);
}
PruferEnumeration prufer_enumeration(std::size_t n, std::size_t max_sequences) {
  prufer_detail::vertices(n, {});
  std::size_t expected = 1;
  for (std::size_t i = 0; i < n - 2; ++i) {
    if (expected > std::numeric_limits<std::size_t>::max() / n)
      throw std::length_error("Cayley count overflow");
    expected *= n;
    if (max_sequences && expected > max_sequences)
      throw std::length_error("Cayley enumeration cap");
  }
  if (max_sequences && expected > max_sequences)
    throw std::length_error("Cayley enumeration cap");
  std::set<std::vector<std::pair<std::size_t, std::size_t>>> trees;
  std::vector<std::size_t>                                   code(n - 2, 0);
  std::size_t                                                visited = 0;
  for (std::size_t i = 0; i < expected; ++i) {
    auto edges = prufer_to_tree(code);
    for (auto& [u, v] : edges)
      if (v < u)
        std::swap(u, v);
    std::sort(edges.begin(), edges.end());
    trees.insert(std::move(edges));
    ++visited;
    for (std::size_t j = code.size(); j > 0; --j) {
      if (++code[j - 1] < n)
        break;
      code[j - 1] = 0;
    }
  }
  return {n, visited, trees.size(), expected};
}
} // namespace datamunge::algorithms
