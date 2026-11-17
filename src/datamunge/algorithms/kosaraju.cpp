/// @file kosaraju.cpp
/// @brief Shared checked deterministic iterative two-pass SCC producer and actual operation capture.
#include <datamunge/algorithms/kosaraju.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Local complete graph validation and producing checked counters/events.
namespace scc_detail {
/// @brief Add logical entries/operations without wrapping.
/// @param count Mutable actual count.
/// @param amount Additional logical count.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& count, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("SCC count overflow");
  count += amount;
}
/// @brief Validate all row/endpoint geometry before access or copies.
/// @param n Original nonnegative vertex count.
/// @param adj Complete borrowed original adjacency.
/// @return Complete original n+E capture count.
/// @throws std::invalid_argument On negative n, row mismatch or invalid target.
/// @throws std::length_error On total record count overflow.
std::size_t validate(int n, const std::vector<std::vector<int>>& adj) {
  if (n < 0 || adj.size() != static_cast<std::size_t>(n))
    throw std::invalid_argument("SCC graph shape");
  std::size_t count = adj.size();
  for (const auto& row : adj) {
    add(count, row.size());
    for (int v : row)
      if (v < 0 || v >= n)
        throw std::invalid_argument("SCC target");
  }
  return count;
}
/// @brief Actual local optional event capture and selected work.
struct Observer {
  KosarajuTrace* trace; ///< Borrowed output for this execution, null without history.
  std::size_t    cap;   ///< Optional complete retained-record ceiling, zero off.
  KosarajuWork   work;  ///< Actual local selected operation counters.

  /// @brief Admit complete original records or next event before producing work.
  /// @param amount Additional original/event records.
  /// @throws std::length_error On logical overflow or enabled quota.
  void admit(std::size_t amount) {
    if (!trace)
      return;
    auto next = trace->captured_entries;
    add(next, amount);
    if (cap && next > cap)
      throw std::length_error("SCC trace quota");
    trace->captured_entries = next;
  }
  /// @brief Append an admitted actual operation in producing order.
  /// @param kind Actual operation phase.
  /// @param u Actual vertex/source.
  /// @param v Actual edge target, default-1 for vertex events.
  /// @param c Actual second-pass label, default-1 for first pass.
  /// @throws std::length_error On capture limits.
  /// @throws std::bad_alloc On event append failure.
  void event(SCCEventKind kind, int u, int v = -1, int c = -1) {
    if (!trace)
      return;
    admit(1);
    trace->events.push_back({kind, u, v, c});
  }
};
/// @brief Produce actual deterministic two-pass labels and optionally own original observations.
/// @param n Validated original vertex count.
/// @param adj Validated complete directed adjacency.
/// @param original Complete n+E initial count.
/// @param o Actual local work/capture state.
/// @return Owned actual component labels.
/// @throws std::length_error On capture/work/container limits.
/// @throws std::bad_alloc On native/capture allocation failure.
std::vector<int>
run(int n, const std::vector<std::vector<int>>& adj, std::size_t original, Observer& o) {
  o.admit(original);
  if (o.trace)
    o.trace->adjacency = adj;
  std::vector<char> vis(n, 0);
  std::vector<int>  order;
  order.reserve(n);
  for (int root = 0; root < n; ++root) {
    if (vis[root])
      continue;
    add(o.work.first_roots, 1);
    o.event(SCCEventKind::Discover, root);
    vis[root] = 1;
    std::vector<std::pair<int, std::size_t>> stack{{root, 0}};
    while (!stack.empty()) {
      auto& [u, i] = stack.back();
      if (i < adj[u].size()) {
        const int v = adj[u][i];
        o.event(SCCEventKind::FirstEdge, u, v);
        ++i;
        add(o.work.first_edge_tests, 1);
        if (!vis[v]) {
          o.event(SCCEventKind::Discover, v);
          vis[v] = 1;
          stack.push_back({v, 0});
        }
      } else {
        o.event(SCCEventKind::Finish, u);
        order.push_back(u);
        add(o.work.finishes, 1);
        stack.pop_back();
      }
    }
  }
  std::vector<std::vector<int>> radj(n);
  for (int u = 0; u < n; ++u)
    for (int v : adj[u])
      radj[v].push_back(u);
  std::vector<int> comp(n, -1);
  int              c = 0;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const int root = *it;
    if (comp[root] != -1)
      continue;
    add(o.work.second_roots, 1);
    o.event(SCCEventKind::Assign, root, -1, c);
    comp[root] = c;
    add(o.work.assignments, 1);
    std::vector<int> stack{root};
    while (!stack.empty()) {
      const int u = stack.back();
      stack.pop_back();
      for (int v : radj[u]) {
        o.event(SCCEventKind::TransposeEdge, u, v, c);
        add(o.work.second_edge_tests, 1);
        if (comp[v] == -1) {
          o.event(SCCEventKind::Assign, v, -1, c);
          comp[v] = c;
          add(o.work.assignments, 1);
          stack.push_back(v);
        }
      }
    }
    ++c;
  }
  if (o.trace) {
    o.trace->transpose    = std::move(radj);
    o.trace->finish_order = std::move(order);
  }
  return comp;
}
} // namespace scc_detail
std::vector<int> kosaraju_scc(int n, const std::vector<std::vector<int>>& adjacency) {
  const auto           initial = scc_detail::validate(n, adjacency);
  scc_detail::Observer o{nullptr, 0, {}};
  return scc_detail::run(n, adjacency, initial, o);
}
KosarajuAnalysis kosaraju_analyze(int n, const std::vector<std::vector<int>>& adjacency) {
  const auto           initial = scc_detail::validate(n, adjacency);
  scc_detail::Observer o{nullptr, 0, {}};
  KosarajuAnalysis     out;
  out.component = scc_detail::run(n, adjacency, initial, o);
  out.work      = o.work;
  return out;
}
KosarajuTrace
kosaraju_trace(int n, const std::vector<std::vector<int>>& adjacency, std::size_t max_entries) {
  const auto           initial = scc_detail::validate(n, adjacency);
  KosarajuTrace        out;
  scc_detail::Observer o{&out, max_entries, {}};
  out.component = scc_detail::run(n, adjacency, initial, o);
  out.work      = o.work;
  return out;
}
} // namespace datamunge::algorithms
