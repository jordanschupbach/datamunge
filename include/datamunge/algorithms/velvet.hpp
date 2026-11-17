#pragma once
/// @file velvet.hpp
/// @brief Checked native velvet and actual producing diagnostics.
#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual de Bruijn multigraph and its Euler walk.
/// @par Example
/// @code
/// auto r=dm::algorithms::velvet_work({"AT","TC"},2);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VelvetWork {
  std::string              sequence; ///< Actual spelling when complete; empty if unresolved.
  std::vector<std::string> walk;     ///< Actual reversed Hierholzer circuit/walk.
  std::vector<std::string> vertices; ///< Actual (k-1)-mer vertices in insertion-ID order.
  std::vector<std::pair<std::size_t, std::size_t>>
              edges;           ///< Actual k-mer multiedges, duplicates retained.
  bool        eulerian{false}; ///< Degree checks and complete edge consumption passed.
  std::size_t consumed{0};     ///< Actual traversed edges.
};
/// @brief Assemble one exact Eulerian contig from the read k-mer multiset.
/// At most10000000 input bytes and1000000 edges. Every observed k-mer occurrence is a separate edge. Successors and start use byte-lexicographic ordering for deterministic output. This toy lacks Velvet error correction, coverage handling and contig decomposition; no biological uniqueness claim. O(input*k+edges log edges) construction and O(edges) traversal; no shared state.
/// @param reads Borrowed byte strings; reads shorter than k add no edges.
/// @param k At least2, at most1000 bytes per k-mer.
/// @return Owned actual graph/walk; eulerian=false on degree/connectivity failure.
/// @throws std::invalid_argument k outside[2,1000].
/// @throws std::length_error Input or edge budget exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::velvet_work({"AT","TC"},2); //ATC
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline VelvetWork velvet_work(const std::vector<std::string>& reads, int k) {
  if (k < 2 || k > 1000)
    throw std::invalid_argument("velvet k");
  std::size_t bytes = 0;
  for (auto& read : reads) {
    if (read.size() > 10000000 - bytes)
      throw std::length_error("velvet bytes");
    bytes += read.size();
  }
  VelvetWork                         r;
  std::map<std::string, std::size_t> ids;
  // Intern actual vertex strings once; no report reconstruction supplies the graph.
  auto vertex = [&](std::string text) {
    auto [it, new_value] = ids.emplace(text, ids.size());
    if (new_value)
      r.vertices.push_back(std::move(text));
    return it->second;
  };
  for (auto& read : reads)
    if (read.size() >= static_cast<std::size_t>(k))
      for (std::size_t i = 0; i <= read.size() - k; ++i) {
        if (r.edges.size() == 1000000)
          throw std::length_error("velvet edges");
        auto u = vertex(read.substr(i, k - 1));
        auto v = vertex(read.substr(i + 1, k - 1));
        r.edges.emplace_back(u, v);
      }
  if (r.edges.empty()) {
    r.eulerian = true;
    return r;
  }
  std::vector<std::vector<std::size_t>> adj(ids.size());
  std::vector<long long>                balance(ids.size());
  for (auto [u, v] : r.edges) {
    adj[u].push_back(v);
    ++balance[u];
    --balance[v];
  }
  for (auto& row : adj)
    std::sort(
        row.begin(), row.end(), [&](auto a, auto b) { return r.vertices[a] < r.vertices[b]; });
  std::size_t start    = ids.size();
  int         positive = 0, negative = 0;
  for (auto [name, id] : ids) {
    if (balance[id] == 1) {
      start = id;
      ++positive;
    } else if (balance[id] == -1)
      ++negative;
    else if (balance[id] != 0)
      return r;
  }
  if (!((positive == 1 && negative == 1) || (positive == 0 && negative == 0)))
    return r;
  if (!positive)
    for (auto [name, id] : ids)
      if (!adj[id].empty()) {
        start = id;
        break;
      }
  std::vector<std::size_t> cursor(ids.size()), stack{start}, circuit;
  while (!stack.empty()) {
    auto u = stack.back();
    if (cursor[u] < adj[u].size()) {
      stack.push_back(adj[u][cursor[u]++]);
      ++r.consumed;
    } else {
      circuit.push_back(u);
      stack.pop_back();
    }
  }
  if (r.consumed != r.edges.size())
    return r;
  std::reverse(circuit.begin(), circuit.end());
  for (auto id : circuit)
    r.walk.push_back(r.vertices[id]);
  r.sequence = r.walk.front();
  for (std::size_t i = 1; i < r.walk.size(); ++i)
    r.sequence += r.walk[i].back();
  r.eulerian = true;
  return r;
}
/// @brief Return a complete Eulerian toy contig, or empty on failure.
/// @param reads Read k-mer multiset.
/// @param k Bytes per k-mer, in[2,1000].
/// @return Actual complete sequence or empty.
/// @throws std::invalid_argument Invalid k.
/// @throws std::length_error Input budget exceeded.
/// @par Example
/// @code
/// auto seq=dm::algorithms::velvet_assemble({"AT","TC"},2);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline std::string velvet_assemble(const std::vector<std::string>& reads, int k) {
  return velvet_work(reads, k).sequence;
}

} // namespace datamunge::algorithms
