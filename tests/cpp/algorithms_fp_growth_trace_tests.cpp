#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/eclat.hpp>
#include <datamunge/algorithms/fp_growth.hpp>
#include <datamunge/plot/diagram.hpp>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <set>
using namespace datamunge::algorithms;
using Rows = std::vector<std::vector<std::size_t>>;
using Sets = std::map<std::vector<std::size_t>, std::size_t>;
namespace {
std::size_t support(const Rows& rows, const std::vector<std::size_t>& items) {
  std::size_t n = 0;
  for (const auto& r : rows)
    n += std::all_of(items.begin(), items.end(), [&](auto item) {
      return std::find(r.begin(), r.end(), item) != r.end();
    });
  return n;
}
Sets as_map(const std::vector<FrequentItemset>& sets) {
  Sets                                             out;
  std::pair<std::size_t, std::vector<std::size_t>> previous{};
  bool                                             first = true;
  for (const auto& s : sets) {
    EXPECT_FALSE(s.items.empty());
    EXPECT_TRUE(std::is_sorted(s.items.begin(), s.items.end()));
    EXPECT_EQ(std::set(s.items.begin(), s.items.end()).size(), s.items.size());
    const auto key = std::make_pair(s.items.size(), s.items);
    if (!first)
      EXPECT_LT(previous, key);
    first    = false;
    previous = key;
    EXPECT_TRUE(out.emplace(s.items, s.support).second);
  }
  return out;
}
// Independent subset membership oracle; no prefix tree, frequency ranks or conditional recursion.
Sets oracle(const Rows& rows, std::size_t threshold) {
  std::set<std::size_t> universe;
  for (const auto& r : rows)
    universe.insert(r.begin(), r.end());
  const std::vector<std::size_t> ids(universe.begin(), universe.end());
  Sets                           out;
  for (std::size_t mask = 1; mask < (std::size_t(1) << ids.size()); ++mask) {
    std::vector<std::size_t> items;
    for (std::size_t b = 0; b < ids.size(); ++b)
      if (mask & (std::size_t(1) << b))
        items.push_back(ids[b]);
    const auto count = support(rows, items);
    if (count && count >= threshold)
      out[items] = count;
  }
  return out;
}
void certificate(const FPGrowthTrace& t) {
  ASSERT_FALSE(t.trees.empty());
  EXPECT_TRUE(t.trees[0].prefix.empty());
  std::vector<std::vector<FPGrowthWeightedPath>> inputs(t.trees.size());
  for (const auto& r : t.transactions) {
    const std::set<std::size_t> items(r.begin(), r.end());
    inputs[0].push_back({std::vector<std::size_t>(items.begin(), items.end()), 1});
  }
  const auto               none = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> parent(t.trees.size(), none), seen(t.trees.size());
  std::vector<bool>        had_step(t.trees.size());
  std::vector<std::pair<std::size_t, std::size_t>> last(t.trees.size());
  std::set<std::pair<std::size_t, std::size_t>>    emitted;
  std::size_t entries = t.transactions.size(), raw = 0, incidences = 0, base_paths = 0, hops = 0,
              output_items = 0, header_extract = 0;
  for (const auto& r : t.transactions) {
    raw += r.size();
    incidences += std::set<std::size_t>(r.begin(), r.end()).size();
  }
  entries += raw;
  for (const auto& s : t.steps) {
    ASSERT_LT(s.tree, t.trees.size());
    const auto& tree    = t.trees[s.tree];
    auto        pattern = tree.prefix;
    pattern.push_back(s.item);
    EXPECT_EQ(s.pattern, pattern);
    EXPECT_EQ(std::set(pattern.begin(), pattern.end()).size(), pattern.size());
    EXPECT_EQ(s.support, support(t.transactions, s.pattern));
    EXPECT_GT(s.support, 0u);
    EXPECT_GE(s.support, t.min_support_count);
    EXPECT_TRUE(emitted.emplace(s.tree, s.item).second);
    if (had_step[s.tree])
      EXPECT_LT(last[s.tree], std::make_pair(s.support, s.item));
    had_step[s.tree] = true;
    last[s.tree]     = {s.support, s.item};
    const auto hi    = std::find_if(
        tree.header.begin(), tree.header.end(), [&](const auto& h) { return h.item == s.item; });
    ASSERT_NE(hi, tree.header.end());
    header_extract += hi->nodes.size();
    std::vector<FPGrowthWeightedPath> expected;
    for (auto id : hi->nodes) {
      std::vector<std::size_t> path;
      auto                     p = tree.nodes[id].parent;
      while (p) {
        path.push_back(tree.nodes[p].item);
        p = tree.nodes[p].parent;
      }
      if (!path.empty())
        expected.push_back({path, tree.nodes[id].count});
    }
    ASSERT_EQ(s.base.size(), expected.size());
    for (std::size_t i = 0; i < s.base.size(); ++i) {
      EXPECT_EQ(s.base[i].items, expected[i].items);
      EXPECT_EQ(s.base[i].weight, expected[i].weight);
      EXPECT_GT(s.base[i].weight, 0u);
      EXPECT_EQ(
          std::set(s.base[i].items.begin(), s.base[i].items.end()).size(), s.base[i].items.size());
      hops += s.base[i].items.size();
      entries += 1 + s.base[i].items.size();
    }
    base_paths += s.base.size();
    entries += 1 + s.pattern.size();
    output_items += s.pattern.size();
    if (s.base.empty())
      EXPECT_EQ(s.child, none);
    else {
      ASSERT_LT(s.child, t.trees.size());
      EXPECT_GT(s.child, s.tree);
      EXPECT_EQ(t.trees[s.child].prefix, s.pattern);
      EXPECT_EQ(seen[s.child]++, 0u);
      parent[s.child] = s.tree;
      inputs[s.child] = s.base;
    }
  }
  std::size_t created = 0, frequency = 0, insert = 0, nonempty = 0, peak_frames = 1, peak_nodes = 1;
  for (std::size_t id = 0; id < t.trees.size(); ++id) {
    const auto& tree = t.trees[id];
    ASSERT_FALSE(tree.nodes.empty());
    EXPECT_EQ(tree.nodes[0].item, none);
    EXPECT_EQ(tree.nodes[0].count, 0u);
    EXPECT_EQ(tree.nodes[0].parent, none);
    if (id)
      EXPECT_EQ(seen[id], 1u);
    std::map<std::size_t, std::size_t> freq;
    for (const auto& row : inputs[id])
      for (auto item : row.items) {
        freq[item] += row.weight;
        ++frequency;
      }
    std::set<std::size_t> want;
    for (auto [item, count] : freq)
      if (count >= t.min_support_count)
        want.insert(item);
    EXPECT_EQ(std::set(tree.order.begin(), tree.order.end()), want);
    EXPECT_EQ(tree.order.size(), want.size());
    for (std::size_t k = 1; k < tree.order.size(); ++k) {
      EXPECT_GE(freq.at(tree.order[k - 1]), freq.at(tree.order[k]));
      if (freq.at(tree.order[k - 1]) == freq.at(tree.order[k]))
        EXPECT_LT(tree.order[k - 1], tree.order[k]);
    }
    for (const auto& row : inputs[id])
      for (auto item : row.items)
        insert += want.count(item);
    std::set<std::pair<std::size_t, std::size_t>> edges;
    std::vector<std::size_t> header_seen(tree.nodes.size()), child_sum(tree.nodes.size());
    for (std::size_t k = 1; k < tree.nodes.size(); ++k) {
      const auto& node = tree.nodes[k];
      ASSERT_LT(node.parent, k);
      EXPECT_TRUE(edges.emplace(node.parent, node.item).second);
      EXPECT_TRUE(want.count(node.item));
      child_sum[node.parent] += node.count;
      std::vector<std::size_t> path;
      for (auto p = k; p; p = tree.nodes[p].parent)
        path.push_back(tree.nodes[p].item);
      std::reverse(path.begin(), path.end());
      std::size_t expected_count = 0;
      for (const auto& row : inputs[id]) {
        const std::set<std::size_t> items(row.items.begin(), row.items.end());
        std::vector<std::size_t>    projected;
        for (auto item : tree.order)
          if (items.count(item))
            projected.push_back(item);
        if (projected.size() >= path.size() &&
            std::equal(path.begin(), path.end(), projected.begin()))
          expected_count += row.weight;
      }
      EXPECT_EQ(node.count, expected_count);
      EXPECT_GT(node.count, 0u);
    }
    for (std::size_t k = 1; k < tree.nodes.size(); ++k)
      EXPECT_LE(child_sum[k], tree.nodes[k].count);
    for (std::size_t h = 0; h < tree.header.size(); ++h) {
      const auto& row = tree.header[h];
      if (h)
        EXPECT_LT(tree.header[h - 1].item, row.item);
      EXPECT_TRUE(std::is_sorted(row.nodes.begin(), row.nodes.end()));
      std::size_t total = 0;
      for (auto k : row.nodes) {
        ASSERT_GT(k, 0u);
        ASSERT_LT(k, tree.nodes.size());
        EXPECT_EQ(row.item, tree.nodes[k].item);
        EXPECT_EQ(header_seen[k]++, 0u);
        total += tree.nodes[k].count;
      }
      EXPECT_EQ(total, freq.at(row.item));
      entries += 1 + row.nodes.size();
    }
    for (std::size_t k = 1; k < header_seen.size(); ++k)
      EXPECT_EQ(header_seen[k], 1u);
    EXPECT_EQ(tree.header.size(), want.size());
    entries += 1 + tree.prefix.size() + tree.nodes.size() + tree.order.size();
    created += tree.nodes.size() - 1;
    nonempty += tree.nodes.size() > 1;
    if (id == 0 || tree.nodes.size() > 1) {
      std::size_t live = 0, frames = 0;
      for (auto p = id; p != none; p = parent[p]) {
        live += t.trees[p].nodes.size();
        ++frames;
      }
      peak_nodes  = std::max(peak_nodes, live);
      peak_frames = std::max(peak_frames, frames);
    }
  }
  const auto& w = t.work;
  EXPECT_EQ(w.transactions, t.transactions.size());
  EXPECT_EQ(w.raw_items, raw);
  EXPECT_EQ(w.incidences, incidences);
  EXPECT_EQ(w.trees_built, t.trees.size());
  EXPECT_EQ(w.nonempty_trees, nonempty);
  EXPECT_EQ(w.nodes_created, created);
  EXPECT_EQ(w.frequency_item_visits, frequency);
  EXPECT_EQ(w.filter_item_visits, frequency);
  EXPECT_EQ(w.insert_visits, insert);
  EXPECT_EQ(w.reused_nodes, insert - created);
  EXPECT_EQ(w.header_count_visits, created);
  EXPECT_EQ(w.header_extract_visits, header_extract);
  EXPECT_EQ(w.parent_hops, hops);
  EXPECT_EQ(w.conditional_paths, base_paths);
  EXPECT_EQ(w.emitted_itemsets, t.steps.size());
  EXPECT_EQ(w.emitted_items, output_items);
  EXPECT_EQ(w.peak_frames, peak_frames);
  EXPECT_EQ(w.peak_live_nodes, peak_nodes);
  EXPECT_EQ(t.captured_entries, entries);
  EXPECT_EQ(t.itemsets.size(), t.steps.size());
  EXPECT_EQ(as_map(t.itemsets), oracle(t.transactions, t.min_support_count));
}
} // namespace
TEST(FPGrowth, ExhaustiveThreeItemDatabasesAgainstSubsetMembership) {
  for (unsigned n = 0; n <= 3; ++n) {
    unsigned count = 1;
    for (unsigned i = 0; i < n; ++i)
      count *= 8;
    for (unsigned code = 0; code < count; ++code) {
      auto digits = code;
      Rows rows(n);
      for (auto& r : rows) {
        auto mask = digits % 8;
        digits /= 8;
        for (std::size_t i = 0; i < 3; ++i)
          if (mask & (1u << i))
            r.push_back(i);
      }
      for (std::size_t threshold = 0; threshold <= n + 1; ++threshold) {
        EXPECT_EQ(as_map(fp_growth_frequent_itemsets(rows, threshold)), oracle(rows, threshold));
        const auto trace = fp_growth_trace(rows, threshold, 0);
        certificate(trace);
      }
    }
  }
}
TEST(FPGrowth, SeededSparseDuplicateAndPermutationCertificates) {
  std::mt19937_64 rng(121);
  for (int fixture = 0; fixture < 250; ++fixture) {
    const auto u = 1 + rng() % 7;
    Rows       rows(rng() % 14);
    for (auto& r : rows)
      for (std::size_t i = 0; i < u; ++i)
        if (rng() % 3 == 0) {
          const auto item = i + 1 == u ? SIZE_MAX : 1000003 * i;
          r.push_back(item);
          if (rng() % 2)
            r.push_back(item);
        }
    const auto threshold = rng() % (rows.size() + 2);
    const auto trace     = fp_growth_trace(rows, threshold, 0);
    certificate(trace);
    const auto expected = as_map(trace.itemsets);
    for (auto& r : rows)
      std::shuffle(r.begin(), r.end(), rng);
    std::shuffle(rows.begin(), rows.end(), rng);
    EXPECT_EQ(as_map(fp_growth_frequent_itemsets(rows, threshold)), expected);
  }
}
TEST(FPGrowth, TinyActualTreeConditionalCountsAndExactOnlineQuota) {
  const Rows rows{{0, 1}, {0, 1}, {0}, {1}};
  const auto t = fp_growth_trace(rows, 2, 38);
  certificate(t);
  EXPECT_EQ(t.captured_entries, 38u);
  EXPECT_EQ(t.trees.size(), 2u);
  EXPECT_EQ(t.steps.size(), 3u);
  EXPECT_EQ(t.trees[0].nodes.size(), 4u);
  EXPECT_EQ(t.trees[0].nodes[1].count, 3u);
  EXPECT_EQ(t.trees[0].nodes[2].count, 2u);
  EXPECT_EQ(t.trees[0].header[1].nodes, (std::vector<std::size_t>{2, 3}));
  EXPECT_EQ(t.steps[1].base.size(), 1u);
  EXPECT_EQ(t.steps[1].base[0].items, (std::vector<std::size_t>{0}));
  EXPECT_EQ(t.steps[1].base[0].weight, 2u);
  EXPECT_EQ(t.steps[2].pattern, (std::vector<std::size_t>{1, 0}));
  EXPECT_THROW(fp_growth_trace(rows, 2, 37), std::length_error);
  EXPECT_NO_THROW(fp_growth_trace(rows, 2, 0));
}
TEST(FPGrowth, DenseAndDisjointExactWorkFamilies) {
  for (std::size_t u = 1; u <= 10; ++u) {
    Rows rows(5, std::vector<std::size_t>(u));
    for (auto& r : rows)
      std::iota(r.begin(), r.end(), 0);
    const auto  result = fp_growth_frequent_itemsets_work(rows, 5);
    const auto& w      = result.work;
    const auto  f      = (std::size_t(1) << u) - 1;
    EXPECT_EQ(result.itemsets.size(), f);
    for (auto& s : result.itemsets)
      EXPECT_EQ(s.support, 5u);
    EXPECT_EQ(w.nodes_created, f);
    EXPECT_EQ(w.trees_built, std::size_t(1) << (u - 1));
    EXPECT_EQ(w.reused_nodes, 4 * u);
    EXPECT_EQ(w.frequency_item_visits, 4 * u + f);
    EXPECT_EQ(w.filter_item_visits, 4 * u + f);
    EXPECT_EQ(w.insert_visits, 4 * u + f);
    EXPECT_EQ(w.header_count_visits, f);
    EXPECT_EQ(w.header_extract_visits, f);
    EXPECT_EQ(w.parent_hops, f - u);
    EXPECT_EQ(w.conditional_paths, (std::size_t(1) << (u - 1)) - 1);
    EXPECT_EQ(w.peak_frames, u);
    EXPECT_EQ(w.peak_live_nodes, u * (u + 3) / 2);
    EXPECT_EQ(w.emitted_items, u * (std::size_t(1) << (u - 1)));
    Rows disjoint(u);
    for (std::size_t i = 0; i < u; ++i)
      disjoint[i] = {i};
    const auto d = fp_growth_frequent_itemsets_work(disjoint, 1);
    EXPECT_EQ(d.itemsets.size(), u);
    EXPECT_EQ(d.work.trees_built, 1u);
    EXPECT_EQ(d.work.nodes_created, u);
    EXPECT_EQ(d.work.parent_hops, 0u);
    EXPECT_EQ(d.work.peak_frames, 1u);
    EXPECT_EQ(d.work.peak_live_nodes, u + 1);
  }
}
TEST(FPGrowth, HistoricalZeroEmptyAboveThresholdAndFullWidthIds) {
  const Rows rows{{SIZE_MAX, SIZE_MAX}, {}, {0, SIZE_MAX}, {0}};
  const auto t = fp_growth_trace(rows, 0);
  certificate(t);
  EXPECT_EQ(t.itemsets.size(), 3u);
  bool real_max = false;
  for (std::size_t i = 1; i < t.trees[0].nodes.size(); ++i)
    real_max |= t.trees[0].nodes[i].item == SIZE_MAX;
  EXPECT_TRUE(real_max);
  const auto none = fp_growth_trace(rows, SIZE_MAX);
  EXPECT_TRUE(none.itemsets.empty());
  EXPECT_EQ(none.trees.size(), 1u);
  EXPECT_EQ(none.trees[0].nodes.size(), 1u);
  EXPECT_EQ(none.work.peak_frames, 1u);
  EXPECT_EQ(none.work.peak_live_nodes, 1u);
  EXPECT_EQ(fp_growth_trace({}, 0, 2).captured_entries, 2u);
  EXPECT_THROW(fp_growth_trace({}, 0, 1), std::length_error);
  EXPECT_EQ(fp_growth_trace({{}, {}}, 0, 4).captured_entries, 4u);
  EXPECT_THROW(fp_growth_trace({{}, {}}, 0, 3), std::length_error);
  EXPECT_EQ(fp_growth_frequent_itemsets({{0}, {1}}, 0).size(), 2u);
  EXPECT_EQ(apriori_frequent_itemsets({{0}, {1}}, 0).size(), 3u);
  EXPECT_EQ(eclat_frequent_itemsets({{0}, {1}}, 0).size(), 3u);
}
TEST(FPGrowth, OwnershipRawDuplicatesAndWideBaseAdmission) {
  Rows       rows{{1, 0, 1}, {0, 1}, {0}, {1}};
  const auto before = rows;
  auto       t      = fp_growth_trace(rows, 2);
  EXPECT_EQ(rows, before);
  EXPECT_EQ(t.transactions, before);
  EXPECT_EQ(t.work.raw_items, 7u);
  EXPECT_EQ(t.work.incidences, 6u);
  const auto count = t.trees[0].nodes[1].count;
  rows[0].clear();
  EXPECT_EQ(t.transactions, before);
  const auto pattern = t.steps.back().pattern;
  t.itemsets.back().items.clear();
  EXPECT_EQ(t.steps.back().pattern, pattern);
  t.trees[1].nodes[1].count = 999;
  EXPECT_EQ(t.trees[0].nodes[1].count, count);
  EXPECT_EQ(t.steps[1].base[0].weight, 2u);
  Rows wide(1, std::vector<std::size_t>(10000));
  std::iota(wide[0].begin(), wide[0].end(), 0);
  EXPECT_EQ(fp_growth_trace_base_entries(wide), 10001u);
  EXPECT_THROW(fp_growth_trace(wide, 1, 10000), std::length_error);
  EXPECT_THROW(fp_growth_trace(wide, 1, 60000), std::length_error);
  EXPECT_EQ(wide[0].back(), 9999u);
}
TEST(FPGrowth, OriginalGroceriesAndPrunedChildTrees) {
  const auto rows = fp_growth_example_transactions();
  const auto t    = fp_growth_trace(rows, 2);
  certificate(t);
  EXPECT_EQ(t.itemsets.size(), 11u);
  EXPECT_EQ(as_map(t.itemsets), as_map(apriori_frequent_itemsets(rows, 2)));
  EXPECT_EQ(as_map(t.itemsets), as_map(eclat_frequent_itemsets(rows, 2)));
  const auto step = std::find_if(
      t.steps.begin(), t.steps.end(), [](const auto& s) { return s.tree == 0 && s.item == 4; });
  ASSERT_NE(step, t.steps.end());
  ASSERT_EQ(step->base.size(), 2u);
  EXPECT_EQ(step->base[0].items, (std::vector<std::size_t>{3}));
  EXPECT_EQ(step->base[0].weight, 2u);
  EXPECT_EQ(step->base[1].items, (std::vector<std::size_t>{3, 0}));
  EXPECT_EQ(step->base[1].weight, 1u);
  ASSERT_LT(step->child, t.trees.size());
  EXPECT_EQ(t.trees[step->child].order, (std::vector<std::size_t>{3}));
  EXPECT_EQ(t.trees[step->child].nodes[1].count, 3u);
  const auto pruned = fp_growth_trace({{0, 2}, {1, 2}, {0}, {1}}, 2);
  certificate(pruned);
  ASSERT_EQ(pruned.trees.size(), 2u);
  EXPECT_EQ(pruned.trees.back().nodes.size(), 1u);
  EXPECT_EQ(pruned.steps.back().child, 1u);
  EXPECT_EQ(pruned.work.peak_frames, 1u);
  EXPECT_EQ(itemset_label({0, 1}, fp_growth_example_labels()), "{bread, milk}");
  const auto w = fp_growth_frequent_itemsets_work(rows, 2);
  EXPECT_EQ(as_map(w.itemsets), as_map(t.itemsets));
  EXPECT_EQ(w.work.parent_hops, t.work.parent_hops);
  EXPECT_EQ(w.work.nodes_created, t.work.nodes_created);
}
namespace {
struct Box {
  double x, y, w, h;
};
std::vector<Box> boxes(const std::string& s) {
  std::vector<Box> out;
  const std::regex r(
      "<rect x=\"([-0-9.e+]+)\" y=\"([-0-9.e+]+)\" width=\"([-0-9.e+]+)\" height=\"([-0-9.e+]+)\"");
  for (std::sregex_iterator i(s.begin(), s.end(), r), end; i != end; ++i)
    out.push_back({std::stod((*i)[1]), std::stod((*i)[2]), std::stod((*i)[3]), std::stod((*i)[4])});
  return out;
}
void nonoverlap(const std::vector<Box>& b) {
  for (std::size_t i = 0; i < b.size(); ++i)
    for (std::size_t j = i + 1; j < b.size(); ++j) {
      const auto& a = b[i];
      const auto& c = b[j];
      EXPECT_TRUE(a.x + a.w <= c.x || c.x + c.w <= a.x || a.y + a.h <= c.y || c.y + c.h <= a.y);
    }
}
} // namespace
TEST(RootedTree, ActualMeasuredBoxesDoNotOverlapAndLabelsOwnStorage) {
  using datamunge::graphics::Diagram;
  std::vector<std::size_t> parents{SIZE_MAX, 0, 0, 1, 1, 2, 2};
  std::vector<std::string> labels{
      "root & <",
      "left",
      "right",
      "unbreakable012345678901234567890123456789012345678901234567890",
      "one\ntwo\nthree\nfour",
      "β and x²",
      ""};
  const auto original = labels;
  const auto d        = Diagram::rooted_tree(parents, labels, "Native tree");
  const auto svg      = d.to_svg();
  auto       b        = boxes(svg);
  ASSERT_EQ(b.size(), parents.size());
  nonoverlap(b);
  EXPECT_NE(svg.find("aria-label=\"root &amp; &lt;\""), std::string::npos);
  EXPECT_NE(svg.find("aria-label=\"β"), std::string::npos);
  labels[0]  = "mutated";
  parents[1] = 2;
  EXPECT_EQ(d.to_svg(), svg);
  EXPECT_NE(svg.find("aria-label=\"" + original[3] + "\""), std::string::npos);
  EXPECT_GT(b[3].w, 160);
}
TEST(RootedTree, ArbitraryParentIndexOrderAndDisconnectedCycles) {
  using datamunge::graphics::Diagram;
  const auto svg =
      Diagram::rooted_tree({SIZE_MAX, 3, 0, 2}, {"root", "leaf", "parent", "middle"}).to_svg();
  const auto b = boxes(svg);
  ASSERT_EQ(b.size(), 4u);
  nonoverlap(b);
  EXPECT_LT(b[0].y, b[2].y);
  EXPECT_LT(b[2].y, b[3].y);
  EXPECT_LT(b[3].y, b[1].y);
  EXPECT_THROW(Diagram::rooted_tree({SIZE_MAX, 2, 1}, {"r", "a", "b"}), std::invalid_argument);
  EXPECT_THROW(
      Diagram::rooted_tree({SIZE_MAX, 0, 3, 2}, {"r", "a", "b", "c"}), std::invalid_argument);
}
TEST(RootedTree, FullShapeRangeCycleValidationBeforeNodeCap) {
  using datamunge::graphics::Diagram;
  EXPECT_THROW(Diagram::rooted_tree({SIZE_MAX}, {"r", "extra"}), std::invalid_argument);
  EXPECT_THROW(Diagram::rooted_tree({0}, {"r"}), std::invalid_argument);
  EXPECT_THROW(Diagram::rooted_tree({SIZE_MAX, SIZE_MAX}, {"r", "a"}), std::invalid_argument);
  EXPECT_THROW(Diagram::rooted_tree({SIZE_MAX, 2}, {"r", "a"}), std::invalid_argument);
  EXPECT_THROW(Diagram::rooted_tree({SIZE_MAX, 1}, {"r", "a"}), std::invalid_argument);
  EXPECT_THROW(
      Diagram::rooted_tree({SIZE_MAX, 2, 1}, {"r", "a", "b"}, "", 1), std::invalid_argument);
  EXPECT_THROW(Diagram::rooted_tree({SIZE_MAX, 0}, {"r", "a"}, "", 1), std::length_error);
  EXPECT_NO_THROW(Diagram::rooted_tree({SIZE_MAX, 0}, {"r", "a"}, "", 2));
  EXPECT_NO_THROW(Diagram::rooted_tree({SIZE_MAX, 0}, {"r", "a"}, "", 0));
  EXPECT_NE(Diagram::rooted_tree({}, {}).to_svg().find("Empty tree"), std::string::npos);
}
TEST(RootedTree, HeapTraversalHandlesDeepChain) {
  using datamunge::graphics::Diagram;
  const std::size_t        n = 2000;
  std::vector<std::size_t> p(n), labels_ids(n);
  std::vector<std::string> labels(n, "x");
  p[0] = SIZE_MAX;
  for (std::size_t i = 1; i < n; ++i)
    p[i] = i - 1;
  EXPECT_THROW(Diagram::rooted_tree(p, labels), std::length_error);
  const auto d = Diagram::rooted_tree(p, labels, "", 0);
  const auto b = boxes(d.to_svg());
  ASSERT_EQ(b.size(), n);
  for (std::size_t i = 1; i < n; ++i) {
    EXPECT_LT(b[i - 1].y + b[i - 1].h, b[i].y);
    EXPECT_DOUBLE_EQ(b[i - 1].x, b[i].x);
  }
}
