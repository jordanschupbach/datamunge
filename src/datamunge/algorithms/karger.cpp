/// @file karger.cpp
/// @brief Checked shared random-contraction multigraph kernel and actual component/trial capture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/karger.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Ranked union-find, checked work/schedule admission and actual contraction kernel details.
namespace karger_detail {
/// @brief Iterative compressed ranked union-find with owned per-trial state.
struct Dsu {
  std::vector<std::size_t> parent; ///< Actual representative pointers.
  std::vector<std::size_t> size;   ///< Actual root component sizes, used for bounded-height union.

  /// @brief Initialize n singleton groups.
  /// @param n Complete original vertex count.
  explicit Dsu(std::size_t n) : parent(n), size(n, 1) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  /// @brief Find and compress one actual representative without recursion.
  /// @param x Valid original vertex.
  /// @return Current internal root id, not canonical trace label.
  std::size_t find(std::size_t x) {
    auto root = x;
    while (parent[root] != root)
      root = parent[root];
    while (parent[x] != x) {
      const auto next = parent[x];
      parent[x]       = root;
      x               = next;
    }
    return root;
  }
  /// @brief Join distinct components, attaching the smaller to the larger.
  /// @param a Valid vertex in first component.
  /// @param b Valid vertex in another component.
  void join(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (size[a] > size[b])
      std::swap(a, b);
    parent[a] = b;
    size[b] += size[a];
  }
  /// @brief Observe actual minimum-original-id labels for every current component.
  /// @return Owned complete n labels; no union/search rerun.
  std::vector<std::size_t> labels() {
    std::vector<std::size_t> minimum(parent.size(), parent.size()), out(parent.size());
    for (std::size_t i = 0; i < parent.size(); ++i) {
      const auto root = find(i);
      minimum[root]   = std::min(minimum[root], i);
    }
    for (std::size_t i = 0; i < parent.size(); ++i)
      out[i] = minimum[find(i)];
    return out;
  }
};
/// @brief Increment one selected operation with representability checking.
/// @param n Counter mutated only on representable increment.
/// @throws std::length_error On overflow.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::length_error("Karger work overflow");
  ++n;
}
/// @brief Check the complete graph/schedule and execute one seeded trial sequence.
/// @param n Original vertex count.
/// @param edges Borrowed undirected original edges.
/// @param requested Positive trial count or0 automatic.
/// @param seed Local mt19937_64 seed.
/// @param o Complete input/trial and online capture ceilings.
/// @param capture Whether to own original edges and actual histories.
/// @return Actual best candidate and optional complete owned histories.
/// @throws std::invalid_argument On invalid graph input.
/// @throws std::length_error On ceilings or schedule/size/work/container overflow.
/// @throws std::bad_alloc On allocation failure.
KargerTrace run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             requested,
    std::uint64_t                                           seed,
    KargerOptions                                           o,
    bool                                                    capture) {
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("Karger endpoint");
  if (n < 2)
    throw std::invalid_argument("Karger needs two vertices");
  const auto max = std::numeric_limits<std::size_t>::max();
  if (edges.size() > max - n || n >= std::vector<std::size_t>().max_size())
    throw std::length_error("Karger complete shape");
  std::size_t count = requested;
  if (!count) {
    const auto factor = static_cast<std::size_t>(std::ceil(std::log(static_cast<double>(n))));
    if (n > max / n || n * n > max / factor)
      throw std::length_error("Karger automatic schedule");
    count = std::max<std::size_t>(1, n * n * factor);
  }
  if ((o.max_input_records && n + edges.size() > o.max_input_records) ||
      (o.max_trials && count > o.max_trials))
    throw std::length_error("Karger complete admission");
  KargerTrace out;
  out.vertex_count     = n;
  out.effective_trials = count;
  out.seed             = seed;
  if (capture)
    out.edges = edges;
  out.result.cut_size = max;
  std::mt19937_64 rng(seed);
  std::size_t     records = 0;
  /// @brief Admit one complete n-label/partition record before captured merge or trial completion.
  const auto admit = [&]() {
    if (!capture)
      return;
    if (n == max || records > max - (n + 1) ||
        (o.max_history_records &&
         (records > o.max_history_records || n + 1 > o.max_history_records - records)))
      throw std::length_error("Karger complete history record");
    records += n + 1;
  };
  for (std::size_t trial = 0; trial < count; ++trial) {
    Dsu         d(n);
    std::size_t groups = n, merges = 0;
    const auto  first = out.steps.size();
    while (groups > 2) {
      std::vector<std::size_t> crossing;
      for (std::size_t i = 0; i < edges.size(); ++i) {
        increment(out.work.candidate_edge_checks);
        const auto [u, v] = edges[i];
        if (u != v && d.find(u) != d.find(v))
          crossing.push_back(i);
      }
      if (crossing.empty())
        break;
      admit();
      const auto bound     = static_cast<std::uint64_t>(crossing.size()),
                 threshold = (std::uint64_t(0) - bound) % bound;
      std::uint64_t word;
      std::size_t   draws = 0;
      do {
        increment(out.work.random_draws);
        increment(draws);
        word = rng();
      } while (word < threshold);
      const auto rank = static_cast<std::size_t>(word % bound), edge = crossing[rank];
      const auto [u, v] = edges[edge];
      const auto before = groups;
      d.join(u, v);
      --groups;
      ++merges;
      increment(out.work.contractions);
      if (capture)
        out.steps.push_back(
            {trial, before, groups, crossing.size(), rank, edge, word, draws, d.labels()});
    }
    admit();
    std::size_t cut = 0;
    for (auto [u, v] : edges) {
      increment(out.work.cut_edge_checks);
      if (d.find(u) != d.find(v))
        increment(cut);
    }
    const bool               changed = cut < out.result.cut_size;
    std::vector<std::size_t> a, b;
    if (capture || changed) {
      const auto root = d.find(0);
      for (std::size_t v = 0; v < n; ++v)
        (d.find(v) == root ? a : b).push_back(v);
    }
    if (changed)
      out.result = {cut, a, b};
    if (capture)
      out.trials.push_back({cut, groups, first, merges, changed, std::move(a), std::move(b)});
    increment(out.work.trials);
  }
  return out;
}
} // namespace karger_detail
MinCutResult karger_min_cut(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    std::size_t                                             b,
    std::uint64_t                                           s) {
  return karger_detail::run(n, e, b, s, {0, 0, 0}, false).result;
}
KargerTrace karger_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    std::size_t                                             b,
    std::uint64_t                                           s,
    KargerOptions                                           o) {
  return karger_detail::run(n, e, b, s, o, true);
}
} // namespace datamunge::algorithms
