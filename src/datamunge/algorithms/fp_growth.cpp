/// @file fp_growth.cpp
/// @brief Checked original FP-tree construction and explicit depth-first conditional mining.
#include <algorithm>
#include <datamunge/algorithms/fp_growth.hpp>
#include <stdexcept>
#include <unordered_map>
namespace datamunge::algorithms {
/// @brief Private checked tree construction and online owned-capture helpers.
namespace fp_growth_detail {
/// @brief Add logical counts without unsigned wrap.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error Count overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("FP-growth count overflow");
  return a + b;
}
/// @brief Raw row container, preserving empty/repeated rows.
using Rows = std::vector<std::vector<std::size_t>>;
/// @brief Working weighted rows from normalized input or actual conditional paths.
using WeightedRows = std::vector<FPGrowthWeightedPath>;
/// @brief Sort/deduplicate each basket without changing the caller.
/// @param rows Raw input baskets.
/// @param work Receives actual normalized incidence count.
/// @return Owned canonical row sets in original row order.
/// @throws std::length_error Count/container overflow.
/// @throws std::bad_alloc Allocation fails.
Rows normalize(const Rows& rows, FPGrowthWork& work) {
  Rows out;
  out.reserve(rows.size());
  for (auto row : rows) {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    work.incidences = add(work.incidences, row.size());
    out.push_back(std::move(row));
  }
  return out;
}
/// @brief Canonicalize actual emitted patterns only after DFS completes.
/// @param sets Owned results modified to ascending IDs, cardinality/lex order.
void canonical_sort(std::vector<FrequentItemset>& sets) {
  for (auto& s : sets)
    std::sort(s.items.begin(), s.items.end());
  std::sort(sets.begin(), sets.end(), [](const FrequentItemset& a, const FrequentItemset& b) {
    if (a.items.size() != b.items.size())
      return a.items.size() < b.items.size();
    return a.items < b.items;
  });
}
/// @brief Admit retained entries before the associated history copy.
/// @param trace Owned capture in progress, null for ordinary/work calls.
/// @param entries Additional exact retained-history records.
/// @param limit Optional cap, zero disables only cap.
/// @throws std::length_error Count overflow or quota exceeded.
void admit(FPGrowthTrace* trace, std::size_t entries, std::size_t limit) {
  if (!trace)
    return;
  const auto total = add(trace->captured_entries, entries);
  if (limit && total > limit)
    throw std::length_error("FP-growth retained history cap");
  trace->captured_entries = total;
}
/// @brief Actual mutable tree with full-width indices, separate from owned history.
struct Tree {
  /// @brief One mutable native node with child lookups by item.
  struct Node {
    std::size_t item{};   ///< Opaque item ID; dummy root also uses SIZE_MAX.
    std::size_t count{};  ///< Actual weighted prefix count.
    std::size_t parent{}; ///< Parent index, SIZE_MAX for root.
    std::unordered_map<std::size_t, std::size_t> children; ///< Item-to-child indices.
  };
  std::vector<Node> nodes; ///< Creation-order nodes, root at0.
  std::unordered_map<std::size_t, std::vector<std::size_t>>
              header;        ///< Item-to-node indices in creation order.
  std::size_t captured_id{}; ///< Captured tree identity, unused without history.

  /// @brief Create only the historical dummy root.
  /// @throws std::bad_alloc Allocation fails.
  Tree() {
    nodes.push_back(
        {std::numeric_limits<std::size_t>::max(), 0, std::numeric_limits<std::size_t>::max(), {}});
  }
  /// @brief Test whether no nonroot nodes were inserted.
  /// @return True exactly when root is the only node.
  bool empty() const { return nodes.size() == 1; }
  /// @brief Insert one ordered unique item path with a positive conditional weight.
  /// @param ordered Items already sorted by this tree's retained frequency ranks.
  /// @param weight Positive multiplicity.
  /// @param work Actual insert/create/reuse counts updated.
  /// @throws std::length_error Count/container overflow.
  /// @throws std::bad_alloc Allocation fails.
  void insert(const std::vector<std::size_t>& ordered, std::size_t weight, FPGrowthWork& work) {
    std::size_t cur = 0;
    for (auto item : ordered) {
      work.insert_visits = add(work.insert_visits, 1);
      const auto  found  = nodes[cur].children.find(item);
      std::size_t next;
      if (found == nodes[cur].children.end()) {
        next = nodes.size();
        nodes.push_back({item, 0, cur, {}});
        nodes[cur].children[item] = next;
        header[item].push_back(next);
        work.nodes_created = add(work.nodes_created, 1);
      } else {
        next              = found->second;
        work.reused_nodes = add(work.reused_nodes, 1);
      }
      nodes[next].count = add(nodes[next].count, weight);
      cur               = next;
    }
  }
};
/// @brief Retain a fully built actual tree after admitting its complete copy.
/// @param trace Optional history.
/// @param tree Actual working tree, not changed.
/// @param prefix Actual DFS suffix sequence.
/// @param order Actual insertion rank order.
/// @param cap Optional online retained-history cap.
/// @return Captured tree index, zero placeholder without capture.
/// @throws std::length_error Count/quota/container failure.
/// @throws std::bad_alloc Allocation fails.
std::size_t capture_tree(
    FPGrowthTrace*                  trace,
    const Tree&                     tree,
    const std::vector<std::size_t>& prefix,
    const std::vector<std::size_t>& order,
    std::size_t                     cap) {
  if (!trace)
    return 0;
  auto entries = add(1, add(prefix.size(), add(tree.nodes.size(), order.size())));
  for (const auto& [item, nodes] : tree.header) {
    (void)item;
    entries = add(entries, add(1, nodes.size()));
  }
  admit(trace, entries, cap);
  FPGrowthTree saved;
  saved.prefix = prefix;
  saved.order  = order;
  saved.nodes.reserve(tree.nodes.size());
  for (const auto& n : tree.nodes)
    saved.nodes.push_back({n.item, n.count, n.parent});
  std::vector<std::size_t> ids;
  ids.reserve(tree.header.size());
  for (const auto& [item, nodes] : tree.header) {
    (void)nodes;
    ids.push_back(item);
  }
  std::sort(ids.begin(), ids.end());
  saved.header.reserve(ids.size());
  for (auto item : ids)
    saved.header.push_back({item, tree.header.at(item)});
  const auto id = trace->trees.size();
  trace->trees.push_back(std::move(saved));
  return id;
}
/// @brief Build one actual weighted tree with the original rank and insertion order.
/// @param weighted Unique-item weighted paths, positive weights.
/// @param threshold Absolute support threshold, possibly zero.
/// @param prefix Actual DFS suffix sequence.
/// @param work Actual build/frequency/filter/insertion counts.
/// @param trace Optional owned snapshots.
/// @param cap Optional online retained-history cap.
/// @return Actual working tree, with captured identity when requested.
/// @throws std::length_error Count/quota/container failure.
/// @throws std::bad_alloc Allocation fails.
Tree build_tree(
    const WeightedRows&             weighted,
    std::size_t                     threshold,
    const std::vector<std::size_t>& prefix,
    FPGrowthWork&                   work,
    FPGrowthTrace*                  trace,
    std::size_t                     cap) {
  work.trees_built = add(work.trees_built, 1);
  std::unordered_map<std::size_t, std::size_t> freq;
  for (const auto& row : weighted)
    for (auto item : row.items) {
      freq[item]                 = add(freq[item], row.weight);
      work.frequency_item_visits = add(work.frequency_item_visits, 1);
    }
  std::vector<std::pair<std::size_t, std::size_t>> ranked;
  for (const auto& [item, support] : freq)
    if (support >= threshold)
      ranked.push_back({item, support});
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second)
      return a.second > b.second;
    return a.first < b.first;
  });
  std::vector<std::size_t> order;
  order.reserve(ranked.size());
  for (const auto& [item, support] : ranked) {
    (void)support;
    order.push_back(item);
  }
  std::unordered_map<std::size_t, std::size_t> rank;
  for (std::size_t i = 0; i < order.size(); ++i)
    rank[order[i]] = i;
  Tree tree;
  for (const auto& row : weighted) {
    std::vector<std::size_t> kept;
    for (auto item : row.items) {
      work.filter_item_visits = add(work.filter_item_visits, 1);
      if (rank.count(item))
        kept.push_back(item);
    }
    std::sort(
        kept.begin(), kept.end(), [&](std::size_t a, std::size_t b) { return rank[a] < rank[b]; });
    if (!kept.empty())
      tree.insert(kept, row.weight, work);
  }
  if (!tree.empty())
    work.nonempty_trees = add(work.nonempty_trees, 1);
  tree.captured_id = capture_tree(trace, tree, prefix, order, cap);
  return tree;
}
/// @brief Compute actual header support totals, then the original least-frequent mining order.
/// @param tree Actual fully built tree.
/// @param threshold Absolute threshold.
/// @param work Actual support-sum node visits.
/// @return Item/support pairs ordered by support ascending, then item ID ascending.
/// @throws std::length_error Count/container overflow.
/// @throws std::bad_alloc Allocation fails.
std::vector<std::pair<std::size_t, std::size_t>>
mining_items(const Tree& tree, std::size_t threshold, FPGrowthWork& work) {
  std::vector<std::pair<std::size_t, std::size_t>> items;
  for (const auto& [item, nodes] : tree.header) {
    std::size_t total = 0;
    for (auto id : nodes) {
      total                    = add(total, tree.nodes[id].count);
      work.header_count_visits = add(work.header_count_visits, 1);
    }
    if (total >= threshold)
      items.push_back({item, total});
  }
  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second)
      return a.second < b.second;
    return a.first < b.first;
  });
  return items;
}
/// @brief Capture one actual emission before its conditional-base construction.
/// @param trace Optional owned history.
/// @param tree Captured source tree ID.
/// @param item Emitted extension.
/// @param support Actual header-count sum.
/// @param pattern Actual unsorted full emitted sequence.
/// @param cap Optional retained-history cap.
/// @return Step index, zero placeholder without capture.
/// @throws std::length_error Count/quota/container failure.
/// @throws std::bad_alloc Allocation fails.
std::size_t capture_step(
    FPGrowthTrace*                  trace,
    std::size_t                     tree,
    std::size_t                     item,
    std::size_t                     support,
    const std::vector<std::size_t>& pattern,
    std::size_t                     cap) {
  if (!trace)
    return 0;
  admit(trace, add(1, pattern.size()), cap);
  FPGrowthStep step;
  step.tree     = tree;
  step.item     = item;
  step.support  = support;
  step.pattern  = pattern;
  const auto id = trace->steps.size();
  trace->steps.push_back(std::move(step));
  return id;
}
/// @brief Append an actual formed conditional base after admitting its history copy.
/// @param trace Optional owned history.
/// @param step Actual emission index.
/// @param base Actual nearest-ancestor-first paths.
/// @param cap Optional retained-history cap.
/// @throws std::length_error Count/quota/container failure.
/// @throws std::bad_alloc Allocation fails.
void capture_base(
    FPGrowthTrace*      trace,
    std::size_t         step,
    const WeightedRows& base,
    std::size_t         cap) {
  if (!trace)
    return;
  std::size_t entries = 0;
  for (const auto& row : base)
    entries = add(entries, add(1, row.items.size()));
  admit(trace, entries, cap);
  trace->steps[step].base = base;
}
/// @brief One owned explicit DFS activation, replacing the historical recursive call.
struct Frame {
  Tree tree; ///< Actual working tree retained until all its items have been mined.
  std::vector<std::size_t>                         prefix; ///< Actual unsorted suffix sequence.
  std::vector<std::pair<std::size_t, std::size_t>> items;  ///< Actual sorted header supports.
  std::size_t next{}; ///< Next item to emit, advance before descent.
};
/// @brief Normalize, build and mine once with optional online snapshots.
/// @param rows Raw unchanged baskets.
/// @param threshold Absolute support threshold, including historical zero.
/// @param trace Optional owned history, already carrying admitted raw input.
/// @param cap Optional online retained-history cap.
/// @return Canonical output and actual work.
/// @throws std::length_error Count/quota/container failure.
/// @throws std::bad_alloc Allocation fails.
FPGrowthResult run(const Rows& rows, std::size_t threshold, FPGrowthTrace* trace, std::size_t cap) {
  FPGrowthResult result;
  auto&          work = result.work;
  work.transactions   = rows.size();
  for (const auto& row : rows)
    work.raw_items = add(work.raw_items, row.size());
  const auto   normalized = normalize(rows, work);
  WeightedRows weighted;
  weighted.reserve(normalized.size());
  for (const auto& row : normalized)
    weighted.push_back({row, 1});
  auto               tree  = build_tree(weighted, threshold, {}, work, trace, cap);
  auto               items = mining_items(tree, threshold, work);
  std::size_t        live  = tree.nodes.size();
  std::vector<Frame> frames;
  frames.push_back({std::move(tree), {}, std::move(items), 0});
  work.peak_frames     = 1;
  work.peak_live_nodes = live;
  while (!frames.empty()) {
    auto& frame = frames.back();
    if (frame.next == frame.items.size()) {
      live -= frame.tree.nodes.size();
      frames.pop_back();
      continue;
    }
    const auto [item, total] = frame.items[frame.next++];
    auto prefix              = frame.prefix;
    prefix.push_back(item);
    const auto step       = capture_step(trace, frame.tree.captured_id, item, total, prefix, cap);
    work.emitted_itemsets = add(work.emitted_itemsets, 1);
    work.emitted_items    = add(work.emitted_items, prefix.size());
    result.itemsets.push_back({prefix, total});
    WeightedRows conditional;
    for (auto idx : frame.tree.header.at(item)) {
      work.header_extract_visits     = add(work.header_extract_visits, 1);
      const auto               count = frame.tree.nodes[idx].count;
      std::vector<std::size_t> path;
      auto                     p = frame.tree.nodes[idx].parent;
      while (p != 0) {
        path.push_back(frame.tree.nodes[p].item);
        work.parent_hops = add(work.parent_hops, 1);
        p                = frame.tree.nodes[p].parent;
      }
      if (!path.empty()) {
        conditional.push_back({std::move(path), count});
        work.conditional_paths = add(work.conditional_paths, 1);
      }
    }
    capture_base(trace, step, conditional, cap);
    if (conditional.empty())
      continue;
    auto child = build_tree(conditional, threshold, prefix, work, trace, cap);
    if (trace)
      trace->steps[step].child = child.captured_id;
    if (child.empty())
      continue;
    auto child_items = mining_items(child, threshold, work);
    live             = add(live, child.nodes.size());
    frames.push_back({std::move(child), std::move(prefix), std::move(child_items), 0});
    work.peak_frames     = std::max(work.peak_frames, frames.size());
    work.peak_live_nodes = std::max(work.peak_live_nodes, live);
  }
  canonical_sort(result.itemsets);
  return result;
}
} // namespace fp_growth_detail
std::size_t fp_growth_trace_base_entries(const std::vector<std::vector<std::size_t>>& rows) {
  auto entries = rows.size();
  for (const auto& row : rows)
    entries = fp_growth_detail::add(entries, row.size());
  return entries;
}
FPGrowthResult fp_growth_frequent_itemsets_work(
    const std::vector<std::vector<std::size_t>>& rows,
    std::size_t                                  threshold) {
  return fp_growth_detail::run(rows, threshold, nullptr, 0);
}
std::vector<FrequentItemset> fp_growth_frequent_itemsets(
    const std::vector<std::vector<std::size_t>>& rows,
    std::size_t                                  threshold) {
  return fp_growth_detail::run(rows, threshold, nullptr, 0).itemsets;
}
FPGrowthTrace fp_growth_trace(
    const std::vector<std::vector<std::size_t>>& rows,
    std::size_t                                  threshold,
    std::size_t                                  cap) {
  FPGrowthTrace trace;
  fp_growth_detail::admit(&trace, fp_growth_trace_base_entries(rows), cap);
  trace.transactions      = rows;
  trace.min_support_count = threshold;
  auto result             = fp_growth_detail::run(rows, threshold, &trace, cap);
  trace.itemsets          = std::move(result.itemsets);
  trace.work              = result.work;
  return trace;
}
std::vector<std::vector<std::size_t>> fp_growth_example_transactions() {
  return {{0, 1, 2}, {0, 1}, {1, 2}, {0, 1, 2}, {3, 4}, {0, 3, 4}, {3, 4}, {0, 1, 2, 3}};
}
std::vector<std::string> fp_growth_example_labels() {
  return {"bread", "milk", "butter", "beer", "diapers"};
}
} // namespace datamunge::algorithms
