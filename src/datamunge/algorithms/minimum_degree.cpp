/// @file minimum_degree.cpp
/// @brief Shared checked set-based graph elimination with actual fill capture.
#include <datamunge/algorithms/minimum_degree.hpp>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private canonical elimination producer.
namespace {
/// @brief Execute greedy or prescribed elimination with optional actual capture.
/// @param input Raw adjacency lists.
/// @param prescribed Optional validated-in-body complete order.
/// @param out Optional capture sink, null avoids history allocations.
/// @param cap Capture limit, zero disables.
/// @return Actual complete permutation.
/// @throws std::invalid_argument If IDs or permutation are invalid.
/// @throws std::length_error If vertex count or capture admission exceeds limits.
std::vector<int> run(
    const std::vector<std::vector<int>>& input,
    const std::vector<int>*              prescribed,
    MinimumDegreeTrace*                  out,
    std::size_t                          cap) {
  const auto count = input.size();
  if (count > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("elimination: too many vertices");
  const int n = static_cast<int>(count);
  for (auto& row : input)
    for (auto v : row)
      if (v < 0 || v >= n)
        throw std::invalid_argument("elimination: invalid graph ID");
  if (prescribed) {
    if (prescribed->size() != count)
      throw std::invalid_argument("elimination: incomplete permutation");
    std::vector<bool> seen(count);
    for (auto v : *prescribed) {
      if (v < 0 || v >= n || seen[v])
        throw std::invalid_argument("elimination: invalid permutation");
      seen[v] = true;
    }
  }
  /// @brief Admit actual captured units with checked addition before allocation.
  auto admit = [&](std::size_t amount) {
    if (!out)
      return;
    if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
        (cap && (out->captured_entries > cap || amount > cap - out->captured_entries)))
      throw std::length_error("elimination: capture quota exceeded");
    out->captured_entries += amount;
  };
  if (out) {
    admit(count);
    for (auto& row : input)
      admit(row.size());
    out->input = input;
  }
  std::vector<std::set<int>> adj(count);
  for (int i = 0; i < n; ++i)
    for (auto v : input[i])
      if (v != i) {
        adj[i].insert(v);
        adj[v].insert(i);
      }
  std::vector<bool> gone(count);
  std::vector<int>  order;
  order.reserve(count);
  for (int step = 0; step < n; ++step) {
    int v = -1;
    if (prescribed)
      v = (*prescribed)[step];
    else
      for (int i = 0; i < n; ++i)
        if (!gone[i] && (v < 0 || adj[i].size() < adj[v].size()))
          v = i;
    std::vector<int>  neighbors(adj[v].begin(), adj[v].end());
    MinimumDegreeStep record;
    if (out) {
      admit(2);
      admit(neighbors.size());
      record.vertex    = v;
      record.neighbors = neighbors;
    }
    order.push_back(v);
    gone[v] = true;
    for (std::size_t i = 0; i < neighbors.size(); ++i)
      for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
        auto a = neighbors[i], b = neighbors[j];
        if (out)
          ++out->pair_trials;
        if (!adj[a].contains(b)) {
          if (out) {
            admit(1);
            record.fill.emplace_back(a, b);
            ++out->fill_edges;
          }
          adj[a].insert(b);
          adj[b].insert(a);
        }
      }
    for (auto u : neighbors)
      adj[u].erase(v);
    adj[v].clear();
    if (out)
      out->steps.push_back(std::move(record));
  }
  return order;
}
} // namespace
std::vector<int> minimum_degree_ordering(const std::vector<std::vector<int>>& a) {
  return run(a, nullptr, nullptr, 0);
}
MinimumDegreeTrace minimum_degree_trace(const std::vector<std::vector<int>>& a, std::size_t cap) {
  MinimumDegreeTrace t;
  t.order = run(a, nullptr, &t, cap);
  return t;
}
MinimumDegreeTrace elimination_trace(
    const std::vector<std::vector<int>>& a,
    const std::vector<int>&              o,
    std::size_t                          cap) {
  MinimumDegreeTrace t;
  t.order = run(a, &o, &t, cap);
  return t;
}
std::vector<int> elimination_natural_order(int n) {
  if (n < 0)
    throw std::invalid_argument("elimination_natural_order: negative size");
  std::vector<int> o(n);
  std::iota(o.begin(), o.end(), 0);
  return o;
}
std::vector<std::vector<int>> minimum_degree_arrow(int n) {
  if (n < 0)
    throw std::invalid_argument("minimum_degree_arrow: negative size");
  std::vector<std::vector<int>> a(n);
  for (int i = 1; i < n; ++i) {
    a[0].push_back(i);
    a[i].push_back(0);
  }
  return a;
}
} // namespace datamunge::algorithms
