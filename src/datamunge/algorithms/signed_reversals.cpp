/// @file signed_reversals.cpp
/// @brief Single validated BFS storing actual selected predecessor states.
#include <algorithm>
#include <datamunge/algorithms/signed_reversals.hpp>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Validate complete signed genes without abs(INT_MIN).
/// @param p Borrowed permutation.
void validate(const std::vector<int>& p) {
  if (p.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() - 1))
    throw std::length_error("reversal: indices");
  std::vector<bool> seen(p.size() + 1);
  for (int x : p) {
    const auto a = x < 0 ? -static_cast<long long>(x) : static_cast<long long>(x);
    if (a < 1 || a > static_cast<long long>(p.size()) || seen[a])
      throw std::invalid_argument("reversal: permutation");
    seen[a] = true;
  }
}
/// @brief Apply one already validated reversal.
/// @param p Owned state.
/// @param i First index.
/// @param j Final index.
/// @return Transformed state.
std::vector<int> apply(std::vector<int> p, int i, int j) {
  while (i < j) {
    std::swap(p[i], p[j]);
    p[i] = -p[i];
    p[j] = -p[j];
    ++i;
    --j;
  }
  if (i == j)
    p[i] = -p[i];
  return p;
}
/// @brief Count actual framed breakpoint boundaries in a valid state.
/// @param p Valid signed state.
/// @return Boundary count.
int bp(const std::vector<int>& p) {
  int count = 0, previous = 0;
  for (int x : p) {
    if (static_cast<long long>(x) != static_cast<long long>(previous) + 1)
      ++count;
    previous = x;
  }
  if (static_cast<long long>(previous) + 1 != static_cast<long long>(p.size()) + 1)
    ++count;
  return count;
}
/// @brief One actual stored BFS predecessor.
struct Node {
  std::vector<int>    state;    ///< Owned actual state.
  std::size_t         parent{}; ///< Actual predecessor index, root points to itself.
  std::pair<int, int> edge;     ///< Actual producing reversal.
  int                 depth{};  ///< Actual BFS depth.
};
/// @brief Search and reconstruct actual first-discovery states.
/// @param start Validated inside producer.
/// @param o Policies.
/// @param capture Retain path states.
/// @return Exact result/work.
SignedReversalRun run(const std::vector<int>& start, const SignedReversalOptions& o, bool capture) {
  if (o.max_genes && start.size() > o.max_genes)
    throw std::length_error("reversal: genes");
  validate(start);
  const int        n = static_cast<int>(start.size());
  std::vector<int> identity(n);
  std::iota(identity.begin(), identity.end(), 1);
  std::vector<Node>                       nodes{{start, 0, {-1, -1}, 0}};
  std::map<std::vector<int>, std::size_t> seen{{start, 0}};
  SignedReversalRun                       r;
  r.states           = 1;
  std::size_t target = 0;
  bool        found  = start == identity;
  for (std::size_t head = 0; head < nodes.size() && !found; ++head) {
    const auto cur   = nodes[head].state;
    const auto depth = nodes[head].depth;
    if (r.layers.empty() || r.layers.back().depth != depth)
      r.layers.push_back({depth, 0, 0, 0});
    auto& layer = r.layers.back();
    ++layer.expanded;
    for (int i = 0; i < n && !found; ++i)
      for (int j = i; j < n; ++j) {
        if (r.generated == std::numeric_limits<std::size_t>::max() ||
            (o.max_edges && r.generated >= o.max_edges))
          throw std::length_error("reversal: edges");
        ++r.generated;
        ++layer.generated;
        auto next = apply(cur, i, j);
        if (seen.contains(next))
          continue;
        if (nodes.size() == nodes.max_size() || (o.max_states && nodes.size() >= o.max_states))
          throw std::length_error("reversal: states");
        const auto index = nodes.size();
        seen.emplace(next, index);
        nodes.push_back({std::move(next), head, {i, j}, depth + 1});
        ++r.states;
        ++layer.discovered;
        if (nodes.back().state == identity) {
          target = index;
          found  = true;
          break;
        }
      }
  }
  if (!found)
    throw std::logic_error("reversal: connected valid graph exhausted");
  std::vector<std::size_t> chain;
  for (auto k = target;; k = nodes[k].parent) {
    chain.push_back(k);
    if (k == 0)
      break;
  }
  std::reverse(chain.begin(), chain.end());
  for (auto k : chain) {
    if (k)
      r.result.reversals.push_back(nodes[k].edge);
    if (capture)
      r.path.push_back({nodes[k].state, bp(nodes[k].state), nodes[k].edge});
  }
  r.result.distance = nodes[target].depth;
  return r;
}
} // namespace
SignedReversalRun signed_reversal_run(const std::vector<int>& s, const SignedReversalOptions& o) {
  return run(s, o, false);
}
SignedReversalRun signed_reversal_trace(const std::vector<int>& s, const SignedReversalOptions& o) {
  return run(s, o, true);
}
SignedReversalResult sort_by_reversals(const std::vector<int>& s) {
  return signed_reversal_run(s).result;
}
int signed_breakpoints(const std::vector<int>& p) {
  validate(p);
  return bp(p);
}
std::vector<int> signed_apply_reversal(std::vector<int> p, int i, int j) {
  validate(p);
  if (i < 0 || j < i || static_cast<std::size_t>(j) >= p.size())
    throw std::invalid_argument("reversal: interval");
  return apply(std::move(p), i, j);
}
} // namespace datamunge::algorithms
