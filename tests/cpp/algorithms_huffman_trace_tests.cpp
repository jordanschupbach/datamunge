#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/huffman_analysis.hpp>
#include <limits>
#include <map>
#include <set>
using namespace datamunge::algorithms;
namespace {
std::uint64_t optimal(
    std::vector<std::uint64_t>                           weights,
    std::map<std::vector<std::uint64_t>, std::uint64_t>& memo) {
  std::sort(weights.begin(), weights.end());
  if (weights.size() < 2)
    return 0;
  if (auto it = memo.find(weights); it != memo.end())
    return it->second;
  std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t i = 0; i < weights.size(); ++i)
    for (std::size_t j = i + 1; j < weights.size(); ++j) {
      std::vector<std::uint64_t> next;
      for (std::size_t k = 0; k < weights.size(); ++k)
        if (k != i && k != j)
          next.push_back(weights[k]);
      const auto merged = weights[i] + weights[j];
      next.push_back(merged);
      best = std::min(best, merged + optimal(next, memo));
    }
  memo[weights] = best;
  return best;
}
} // namespace
TEST(HuffmanTrace, ExhaustiveSmallCountsAndActualMinimumRemovals) {
  std::map<std::vector<std::uint64_t>, std::uint64_t> memo;
  for (unsigned code = 0; code < 4096; ++code) {
    std::vector<std::uint64_t> counts(6), positive;
    unsigned                   value = code;
    for (auto& c : counts) {
      c = value % 4;
      value /= 4;
      if (c)
        positive.push_back(c);
    }
    const auto  t = huffman_counts_trace(counts);
    const auto& r = t.run;
    const auto  m = positive.size();
    EXPECT_EQ(r.active_symbols, m);
    EXPECT_EQ(t.counts, counts);
    EXPECT_EQ(t.merges.size(), m ? m - 1 : 0);
    EXPECT_EQ(huffman_code_lengths(counts), r.lengths);
    const auto want = m == 1 ? positive[0] : optimal(positive, memo);
    EXPECT_EQ(r.weighted_bits, std::to_string(want));
    EXPECT_EQ(r.merge_cost, std::to_string(optimal(positive, memo)));
    std::multiset<std::pair<std::uint64_t, std::size_t>> heap;
    for (std::size_t i = 0; i < m; ++i) {
      ASSERT_TRUE(r.nodes[i].leaf);
      heap.insert({counts[r.nodes[i].symbol], i});
    }
    for (const auto& merge : t.merges) {
      ASSERT_GE(heap.size(), 2);
      const auto a = *heap.begin();
      heap.erase(heap.begin());
      const auto b = *heap.begin();
      heap.erase(heap.begin());
      EXPECT_EQ(merge.left, a.second);
      EXPECT_EQ(merge.right, b.second);
      EXPECT_EQ(merge.left_weight, std::to_string(a.first));
      EXPECT_EQ(merge.right_weight, std::to_string(b.first));
      EXPECT_EQ(merge.parent_weight, std::to_string(a.first + b.first));
      ASSERT_LT(merge.parent, r.nodes.size());
      EXPECT_EQ(r.nodes[merge.parent].left, a.second);
      EXPECT_EQ(r.nodes[merge.parent].right, b.second);
      heap.insert({a.first + b.first, merge.parent});
    }
    std::uint64_t weighted = 0;
    double        kraft    = 0;
    for (std::size_t s = 0; s < counts.size(); ++s) {
      if (!counts[s]) {
        EXPECT_EQ(r.lengths[s], 0);
        continue;
      }
      EXPECT_GT(r.lengths[s], 0);
      weighted += counts[s] * r.lengths[s];
      kraft += std::ldexp(1., -r.lengths[s]);
    }
    EXPECT_EQ(weighted, want);
    if (m >= 2)
      EXPECT_EQ(kraft, 1.);
    if (m == 1)
      EXPECT_EQ(kraft, .5);
    EXPECT_EQ(r.work.counts_visited, counts.size());
    EXPECT_EQ(r.work.heap_pushes, m + (m ? m - 1 : 0));
    EXPECT_EQ(r.work.heap_pops, 2 * (m ? m - 1 : 0));
    EXPECT_EQ(r.work.tree_visits, r.nodes.size());
  }
}
TEST(HuffmanTrace, BinaryRoundTripsAndStableByteOrder) {
  const std::string alphabet("\0a\x80\xff", 4);
  for (std::size_t length = 0; length <= 5; ++length) {
    const auto cases = std::size_t{1} << (2 * length);
    for (std::size_t code = 0; code < cases; ++code) {
      auto        value = code;
      std::string text;
      for (std::size_t i = 0; i < length; ++i) {
        text += alphabet[value % 4];
        value /= 4;
      }
      const auto t = huffman_build_trace(text);
      validate_huffman_code(t.run.code);
      const auto encoded = huffman_encode(t.run.code, text);
      EXPECT_EQ(huffman_decode(t.run.code, encoded), text);
      EXPECT_EQ(
          huffman_average_bits(t.run.code, text),
          text.empty() ? 0 : double(encoded.size()) / text.size());
      EXPECT_EQ(t.run.construction.weighted_bits, std::to_string(encoded.size()));
      auto reversed = text;
      std::reverse(reversed.begin(), reversed.end());
      EXPECT_EQ(huffman_build(reversed).codes, t.run.code.codes);
    }
  }
  std::string all;
  for (unsigned i = 0; i < 256; ++i)
    all.push_back(static_cast<char>(i));
  const auto c = huffman_build(all);
  for (const auto& [symbol, word] : c.codes)
    EXPECT_EQ(word.size(), 8);
  EXPECT_EQ(huffman_decode(c, huffman_encode(c, all)), all);
}
TEST(HuffmanTrace, MalformedModelsAndPayloadsNeverWalkUnchecked) {
  EXPECT_THROW(huffman_decode(huffman_build("aaaa"), "111x"), std::invalid_argument);
  EXPECT_THROW(huffman_decode(huffman_build("aaaa"), "1"), std::invalid_argument);
  EXPECT_THROW(huffman_decode(HuffmanCode{}, "0101"), std::invalid_argument);
  EXPECT_EQ(huffman_decode(HuffmanCode{}, ""), "");
  const auto code = huffman_build("abbccc");
  for (const auto& [symbol, word] : code.codes)
    if (word.size() > 1)
      EXPECT_THROW(huffman_decode(code, word.substr(0, word.size() - 1)), std::invalid_argument);
  EXPECT_THROW(huffman_encode(code, "z"), std::out_of_range);
  EXPECT_THROW(huffman_average_bits(code, "z"), std::out_of_range);
  auto bad = code;
  bad.root = 9999;
  EXPECT_THROW(huffman_decode(bad, ""), std::invalid_argument);
  bad      = code;
  bad.root = -2;
  EXPECT_THROW(huffman_encode(bad, ""), std::invalid_argument);
  bad                     = code;
  bad.tree[bad.root].left = bad.root;
  EXPECT_THROW(huffman_decode(bad, "0"), std::invalid_argument);
  bad                     = code;
  bad.tree[bad.root].left = 9999;
  EXPECT_THROW(huffman_decode(bad, "0"), std::invalid_argument);
  bad                     = code;
  bad.tree[bad.root].left = bad.tree[bad.root].right;
  EXPECT_THROW(validate_huffman_code(bad), std::invalid_argument);
  bad = code;
  bad.tree.push_back({'x', true, -1, -1});
  EXPECT_THROW(validate_huffman_code(bad), std::invalid_argument);
  bad = code;
  bad.codes.begin()->second += "0";
  EXPECT_THROW(huffman_average_bits(bad, ""), std::invalid_argument);
  bad = code;
  bad.codes.erase(bad.codes.begin());
  EXPECT_THROW(validate_huffman_code(bad), std::invalid_argument);
  bad              = code;
  bad.tree[0].left = 0;
  EXPECT_THROW(validate_huffman_code(bad), std::invalid_argument);
  auto reordered = huffman_build("aab");
  auto old       = reordered.tree;
  reordered.tree = {old[2], old[0], old[1]};
  reordered.root = 0;
  for (auto& node : reordered.tree)
    if (!node.leaf) {
      node.left  = node.left == 0 ? 1 : 2;
      node.right = node.right == 0 ? 1 : 2;
    }
  EXPECT_NO_THROW(validate_huffman_code(reordered));
  EXPECT_EQ(huffman_decode(reordered, huffman_encode(reordered, "aab")), "aab");
}
TEST(HuffmanTrace, WideWeightsCompleteLimitsAndOwnership) {
  const auto max = std::numeric_limits<std::uint64_t>::max();
  const auto t   = huffman_counts_trace({max, max, max, max});
  EXPECT_EQ(t.run.lengths, (std::vector<int>{2, 2, 2, 2}));
  EXPECT_EQ(t.run.total_weight, "73786976294838206460");
  EXPECT_EQ(t.run.weighted_bits, "147573952589676412920");
  EXPECT_EQ(t.run.merge_cost, t.run.weighted_bits);
  EXPECT_EQ(huffman_counts_trace({0, max, 0}).run.weighted_bits, "18446744073709551615");
  EXPECT_EQ(huffman_counts_trace({0, max, 0}).run.merge_cost, "0");
  EXPECT_EQ(huffman_counts_trace({1, 1, 2}, {3, 2}).merges.size(), 2);
  EXPECT_THROW(huffman_counts_trace({1, 1, 2}, HuffmanOptions{3, 1}), std::length_error);
  EXPECT_THROW(huffman_counts_run({1, 1, 2}, HuffmanOptions{2, 0}), std::length_error);
  EXPECT_EQ(huffman_counts_run({1, 1, 2}, {3, 1}).active_symbols, 3);
  EXPECT_EQ(huffman_build_trace("aab", {3, 1}).merges.size(), 1);
  EXPECT_THROW(huffman_build_run("aab", HuffmanOptions{2, 0}), std::length_error);
  auto copy                    = t;
  copy.merges[0].parent_weight = "0";
  EXPECT_NE(copy.merges[0].parent_weight, t.merges[0].parent_weight);
  std::vector<std::uint64_t> counts = {1, 1, 2};
  const auto                 owned  = huffman_counts_trace(counts);
  counts[0]                         = 99;
  EXPECT_EQ(owned.counts[0], 1);
}

TEST(HuffmanTrace, SummariesAndNativeActualTopology) {
  const auto t = huffman_counts_trace({5, 2, 1});
  const auto s = huffman_summary(t.counts, t.run.lengths);
  EXPECT_NEAR(s.mean_bits, 1.375, 1e-15);
  EXPECT_EQ(s.kraft_sum, 1);
  EXPECT_LT(s.entropy_bits, s.mean_bits);
  EXPECT_NO_THROW(huffman_tree_diagram(t, 3));
  EXPECT_THROW(huffman_tree_diagram(t, 2), std::length_error);
  auto broken                            = t;
  broken.run.nodes[broken.run.root].left = broken.run.root;
  EXPECT_THROW(huffman_tree_diagram(broken), std::invalid_argument);
  EXPECT_EQ(huffman_summary({0, 7}, {0, 1}).entropy_bits, 0);
  EXPECT_EQ(huffman_summary({0, 7}, {0, 1}).mean_bits, 1);
  EXPECT_THROW(huffman_summary({1}, {0}), std::invalid_argument);
  EXPECT_THROW(huffman_summary({0}, {1}), std::invalid_argument);
  EXPECT_EQ(huffman_example_text().size(), 89);
  EXPECT_NO_THROW(huffman_tree_diagram(huffman_counts_trace({0, 0})));
}
