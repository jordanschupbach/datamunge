/// @file lex_bfs.cpp
/// @brief Shared checked dense Lex-BFS producer and separate explicit perfect-elimination-order check.
#include <datamunge/algorithms/lex_bfs.hpp>
#include <limits>
#include <list>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete graph geometry and actual dense ordered-partition producer.
namespace lex_detail {
/// @brief Add an actual/conservative logical count without wrap.
/// @param value Mutable count.
/// @param amount Additional count.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& value, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - value)
    throw std::length_error("Lex-BFS count geometry");
  value += amount;
}
/// @brief Validate every original endpoint/row and complete dense/selected-work geometry before allocation.
/// @param n Original signed vertex count.
/// @param adj Borrowed complete original rows.
/// @return Actual original adjacency entry count, including loops/duplicates.
/// @throws std::invalid_argument On count/shape/endpoint violations.
/// @throws std::length_error On complete matrix/work/container geometry.
std::size_t validate(int n, const std::vector<std::vector<int>>& adj) {
  if (n < 0 || adj.size() != static_cast<std::size_t>(n))
    throw std::invalid_argument("Lex-BFS complete graph shape");
  const auto size = adj.size(), max = std::numeric_limits<std::size_t>::max();
  if (size && (size > max / size || size > std::vector<char>{}.max_size() / size))
    throw std::length_error("Lex-BFS complete dense geometry");
  std::size_t edges = 0;
  for (auto& row : adj) {
    add(edges, row.size());
    for (int v : row)
      if (v < 0 || v >= n)
        throw std::invalid_argument("Lex-BFS endpoint range");
  }
  return edges;
}
/// @brief Construct the actual normalized symmetric simple dense matrix from validated input.
/// @param adj Validated original complete adjacency.
/// @param work Mutable actual selected construction counts.
/// @return Owned actual matrix used by the producer/checker.
/// @throws std::bad_alloc On dense allocation failure.
std::vector<std::vector<char>> matrix(const std::vector<std::vector<int>>& adj, LexBFSWork& work) {
  const auto                     n = adj.size();
  std::vector<std::vector<char>> out(n, std::vector<char>(n, 0));
  work.matrix_cells = n * n;
  for (std::size_t u = 0; u < n; ++u)
    for (int v : adj[u]) {
      ++work.adjacency_entries;
      if (u != static_cast<std::size_t>(v)) {
        out[u][v] = 1;
        out[v][u] = 1;
      }
    }
  return out;
}
/// @brief Produce the historical deterministic selection/splitting once with optional preadmitted capture.
/// @param adj Validated complete original graph.
/// @param trace Optional completely admitted owned capture.
/// @return Owned actual forward order and selected work.
/// @throws std::bad_alloc On runtime/capture/output allocation failure.
LexBFSAnalysis run(const std::vector<std::vector<int>>& adj, LexBFSTrace* trace) {
  LexBFSAnalysis              out;
  auto                        nb = matrix(adj, out.work);
  std::list<std::vector<int>> parts;
  std::vector<int>            all(adj.size());
  for (std::size_t i = 0; i < all.size(); ++i)
    all[i] = static_cast<int>(i);
  if (!all.empty())
    parts.push_back(all);
  if (trace)
    trace->stages.push_back({-1, {parts.begin(), parts.end()}});
  out.order.reserve(adj.size());
  while (!parts.empty()) {
    const int v = parts.front().back();
    parts.front().pop_back();
    if (parts.front().empty())
      parts.pop_front();
    const auto step = out.order.size();
    out.order.push_back(v);
    ++out.work.emitted;
    std::list<std::vector<int>> next;
    std::size_t                 block = 0;
    for (auto& s : parts) {
      ++out.work.blocks_scanned;
      std::vector<int> in, rest;
      for (int u : s) {
        const bool neighbor = nb[v][u];
        ++out.work.neighbor_tests;
        (neighbor ? in : rest).push_back(u);
        if (trace)
          trace->classifications.push_back({step, block, v, u, neighbor});
      }
      if (!in.empty())
        next.push_back(std::move(in));
      if (!rest.empty())
        next.push_back(std::move(rest));
      ++block;
    }
    parts.swap(next);
    if (trace)
      trace->stages.push_back({v, {parts.begin(), parts.end()}});
  }
  if (trace)
    trace->matrix = std::move(nb);
  return out;
}
} // namespace lex_detail
std::vector<int> lex_bfs(int n, const std::vector<std::vector<int>>& adj) {
  return lex_bfs_analyze(n, adj).order;
}
LexBFSAnalysis lex_bfs_analyze(int n, const std::vector<std::vector<int>>& adj) {
  lex_detail::validate(n, adj);
  return lex_detail::run(adj, nullptr);
}
LexBFSTrace
lex_bfs_trace(int n, const std::vector<std::vector<int>>& adj, std::size_t max_entries) {
  auto       capture = lex_detail::validate(n, adj);
  const auto size = adj.size(), cells = size * size;
  lex_detail::add(capture, cells);
  lex_detail::add(capture, cells);
  lex_detail::add(capture, size);
  lex_detail::add(capture, 1);
  if (max_entries && capture > max_entries)
    throw std::length_error("Lex-BFS complete capture quota");
  LexBFSTrace out;
  out.captured_entries = capture;
  out.input_adjacency  = adj;
  out.stages.reserve(size + 1);
  out.classifications.reserve(size ? size * (size - 1) / 2 : 0);
  out.analysis = lex_detail::run(adj, &out);
  return out;
}
PerfectEliminationCheck check_perfect_elimination_order(
    int                                  n,
    const std::vector<std::vector<int>>& adj,
    const std::vector<int>&              order) {
  lex_detail::validate(n, adj);
  if (order.size() != adj.size())
    throw std::invalid_argument("PEO complete permutation shape");
  std::vector<char> seen(adj.size());
  for (int v : order) {
    if (v < 0 || v >= n || seen[v])
      throw std::invalid_argument("PEO complete permutation");
    seen[v] = 1;
  }
  LexBFSWork              work;
  const auto              nb = lex_detail::matrix(adj, work);
  PerfectEliminationCheck out;
  for (std::size_t i = 0; i < order.size(); ++i) {
    const int v = order[i];
    for (std::size_t j = i + 1; j < order.size(); ++j)
      if (nb[v][order[j]])
        for (std::size_t k = j + 1; k < order.size(); ++k)
          if (nb[v][order[k]]) {
            lex_detail::add(out.neighbor_pairs, 1);
            if (!nb[order[j]][order[k]]) {
              out.valid  = false;
              out.vertex = v;
              out.left   = order[j];
              out.right  = order[k];
              return out;
            }
          }
  }
  return out;
}
} // namespace datamunge::algorithms
