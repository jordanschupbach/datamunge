#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/substrings.hpp>
#include <datamunge/algorithms/tree_sort.hpp>
#include <datamunge/algorithms/trial_division.hpp>
#include <datamunge/algorithms/tricubic_interpolation.hpp>
#include <datamunge/algorithms/trigram_search.hpp>
#include <datamunge/algorithms/truncated_binary.hpp>
#include <datamunge/algorithms/truncated_binary_exponential_backoff.hpp>
#include <datamunge/algorithms/trust_rank.hpp>
#include <datamunge/stats/embedding_diagnostics.hpp>
#include <datamunge/stats/tsne.hpp>
#include <datamunge/stats/umap.hpp>
#include <limits>
#include <numeric>
#include <set>
namespace a = datamunge::algorithms;
namespace s = datamunge::stats;
TEST(Batch404, TreeProducer) {
  auto r = a::tree_sort_work({5, 3, 8, 1, 9, 2, 7});
  EXPECT_EQ(r.sorted, (std::vector<int>{1, 2, 3, 5, 7, 8, 9}));
  std::size_t sum = std::accumulate(r.depth.begin(), r.depth.end(), std::size_t{});
  EXPECT_EQ(sum, r.comparisons);
  EXPECT_EQ(r.left[0], 1);
  EXPECT_EQ(r.right[0], 2);
}
TEST(Batch404, TreeLimits) {
  EXPECT_TRUE(a::tree_sort_work({}).sorted.empty());
  EXPECT_EQ(a::tree_sort_work({2, 2, 2}).depth, (std::vector<std::size_t>{0, 1, 2}));
  EXPECT_THROW(a::tree_sort_work({1, 2, 3}, 1), std::length_error);
  std::vector<int> v(1000);
  std::iota(v.begin(), v.end(), 0);
  EXPECT_EQ(a::tree_sort_work(v).comparisons, 499500);
}
TEST(Batch404, TrialFactors) {
  EXPECT_EQ(a::trial_division_factorize(360), (std::vector<std::uint64_t>{2, 2, 2, 3, 3, 5}));
  EXPECT_TRUE(a::trial_division_factorize(1).empty());
  EXPECT_EQ(a::trial_division_work(101 * 101).odd_candidates, 50);
  EXPECT_EQ(a::trial_division_factorize(UINT64_C(1) << 63), std::vector<std::uint64_t>(63, 2));
}
TEST(Batch404, TrialValidation) {
  EXPECT_THROW(a::trial_division_factorize(0), std::invalid_argument);
  EXPECT_THROW(a::trial_division_work(UINT64_MAX, 2), std::length_error);
  EXPECT_EQ(a::trial_division_factorize(2027), (std::vector<std::uint64_t>{2027}));
}
TEST(Batch404, TricubicNeighborhood) {
  std::vector<double> x{0, 1, 2, 3}, v(64);
  std::iota(v.begin(), v.end(), 0.);
  auto r = a::tricubic_work(x, x, x, v, 1.5, 1.5, 1.5);
  EXPECT_NEAR(r.value, 31.5, 1e-12);
  EXPECT_EQ(r.indices.size(), 64);
  EXPECT_NEAR(std::accumulate(r.weights.begin(), r.weights.end(), 0.), 1, 1e-12);
  for (int i = 0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(a::tricubic_interpolate(x, x, x, v, i, i, i), 21 * i);
}
TEST(Batch404, TricubicInvalid) {
  std::vector<double> x{0, 1}, v(8, 3);
  EXPECT_THROW(a::tricubic_work({}, x, x, v, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(a::tricubic_work(x, x, x, v, -1, 0, 0), std::invalid_argument);
  v.pop_back();
  EXPECT_THROW(a::tricubic_work(x, x, x, v, .5, .5, .5), std::invalid_argument);
  EXPECT_EQ(a::tricubic_example().x.size(), 101);
}
TEST(Batch404, TrigramSet) {
  EXPECT_DOUBLE_EQ(a::trigram_similarity("cat", "bat"), 1. / 7);
  EXPECT_DOUBLE_EQ(a::trigram_similarity("", ""), 1);
  EXPECT_EQ(a::trigrams("CAT"), a::trigrams("cat"));
  EXPECT_EQ(a::trigrams(std::string(1, char(0xc0))).size(), 2);
}
TEST(Batch404, TrigramRank) {
  auto r = a::trigram_search({"CAT", "cat", "bat"}, "cat");
  ASSERT_EQ(r.size(), 3);
  EXPECT_EQ(r[0].second, 0);
  EXPECT_EQ(r[1].second, 1);
  EXPECT_EQ(r[2].second, 2);
  EXPECT_TRUE(a::trigram_search({}, "cat").empty());
}
TEST(Batch404, BackoffSlots) {
  for (int c = 0; c < 30; ++c) {
    int w = a::contention_window(c);
    EXPECT_EQ(w, 1 << std::min(c, 10));
    for (int j = 0; j < w; ++j)
      EXPECT_EQ(a::backoff_slot(c, (j + .5) / w), j);
  }
  EXPECT_TRUE(a::backoff_give_up(16));
  EXPECT_FALSE(a::backoff_give_up(15));
}
TEST(Batch404, BackoffValidation) {
  EXPECT_THROW(a::contention_window(-1), std::invalid_argument);
  for (double u :
       {-1., 1., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    EXPECT_THROW(a::backoff_slot(2, u), std::invalid_argument);
  EXPECT_EQ(a::backoff_slot(10, std::nextafter(1., 0.)), 1023);
}
TEST(Batch404, BinaryFullWidth) {
  for (std::uint64_t n : std::vector<std::uint64_t>{1, 2, 3, 5, UINT64_C(1) << 63, UINT64_MAX}) {
    std::vector<std::uint64_t> values{0, n / 2, n - 1};
    auto                       b = a::truncated_binary_encode(values, n);
    EXPECT_EQ(a::truncated_binary_decode(b, n, 3), values);
  }
  EXPECT_EQ(a::truncated_binary_decode({}, 1, 3), (std::vector<std::uint64_t>{0, 0, 0}));
}
TEST(Batch404, BinaryValidation) {
  EXPECT_THROW(a::truncated_binary_encode({}, 0), std::invalid_argument);
  EXPECT_THROW(a::truncated_binary_encode({5}, 5), std::invalid_argument);
  EXPECT_THROW(a::truncated_binary_decode({true}, 5, 1), std::invalid_argument);
  EXPECT_THROW(a::truncated_binary_decode({false}, 1, 1), std::invalid_argument);
  EXPECT_THROW(a::truncated_binary_decode({}, 1, -1), std::invalid_argument);
  EXPECT_THROW(a::truncated_binary_decode({}, 1, 1000001), std::length_error);
}
TEST(Batch404, TrustFixedPoint) {
  auto r = a::trust_rank_work(3, {{0, 1}, {1, 2}, {2, 0}}, {0});
  ASSERT_TRUE(r.converged);
  double denominator = 1 + .85 + .85 * .85;
  EXPECT_NEAR(r.rank[0], 1 / denominator, 1e-12);
  EXPECT_NEAR(r.rank[1], .85 / denominator, 1e-12);
  EXPECT_NEAR(r.rank[2], .85 * .85 / denominator, 1e-12);
}
TEST(Batch404, TrustDangling) {
  auto r = a::trust_rank_work(2, {{0, 1}}, {0}, .5, 1e-12, 2, 3);
  ASSERT_EQ(r.iterates.size(), 3);
  EXPECT_EQ(r.iterates[1], (std::vector<double>{.5, .5}));
  EXPECT_EQ(r.iterates[2], (std::vector<double>{.75, .25}));
  EXPECT_FALSE(r.converged);
  EXPECT_THROW(a::trust_rank_work(2, {{2, 0}}, {0}), std::invalid_argument);
  EXPECT_THROW(
      a::trust_rank_work(2, {}, {0}, .85, std::numeric_limits<double>::quiet_NaN()),
      std::invalid_argument);
}
TEST(Batch404, SuffixProducer) {
  for (std::string text : {"", "banana", "aaaa", "abab", "$"}) {
    auto                  r = a::build_suffix_tree_work(text);
    std::set<std::string> all;
    for (std::size_t i = 0; i < text.size(); ++i)
      for (std::size_t n = 1; n <= text.size() - i; ++n) {
        auto q = text.substr(i, n);
        all.insert(q);
        EXPECT_TRUE(a::suffix_tree_contains(r.tree, q));
      }
    EXPECT_EQ(a::distinct_substring_count(r.tree), all.size());
    EXPECT_EQ(r.phases, text.size() + 1);
    EXPECT_EQ(r.nodes_after_phase.back(), r.tree.nodes.size());
  }
}
TEST(Batch404, SuffixBinary) {
  std::string text("a\0$a\xff", 5);
  auto        r = a::build_suffix_tree_work(text);
  EXPECT_TRUE(a::suffix_tree_contains(r.tree, std::string("\0$", 2)));
  EXPECT_FALSE(a::suffix_tree_contains(r.tree, "$$"));
  EXPECT_LE(r.tree.nodes.size(), 2 * (text.size() + 1));
  EXPECT_FALSE(a::suffix_tree_plot(r.tree).series().empty());
}
TEST(Batch404, TsneAffinities) {
  datamunge::dstruct::DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 3, 7});
  s::TSNEOptions o;
  o.perplexity     = 2;
  o.max_iterations = 0;
  s::TSNE m(d, {"x"}, o);
  auto&   c   = m.conditional_probabilities();
  auto&   p   = m.joint_probabilities();
  double  sum = 0;
  for (int i = 0; i < 4; ++i) {
    double row = 0;
    for (int j = 0; j < 4; ++j) {
      row += c(i, j);
      EXPECT_NEAR(p(i, j), (c(i, j) + c(j, i)) / 8, 1e-15);
      sum += p(i, j);
    }
    EXPECT_NEAR(row, 1, 1e-14);
    EXPECT_NEAR(m.achieved_perplexity()[i], 2, 2e-5);
  }
  EXPECT_NEAR(sum, 1, 1e-14);
}
TEST(Batch404, TsneInvalidAndReproducible) {
  datamunge::dstruct::DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 3, 7});
  s::TSNEOptions o;
  o.perplexity     = 2;
  o.max_iterations = 5;
  s::TSNE a(d, {"x"}, o), b(d, {"x"}, o);
  EXPECT_EQ(a.dimension(0), b.dimension(0));
  o.learning_rate = std::numeric_limits<double>::infinity();
  EXPECT_THROW(s::TSNE(d, {"x"}, o), std::invalid_argument);
  EXPECT_DOUBLE_EQ(s::embedding_abs_correlation({1, 2, 3}, {3, 2, 1}), 1);
}
TEST(Batch404, UmapAffinities) {
  datamunge::dstruct::DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 3, 7});
  s::UMAPOptions o;
  o.n_neighbors    = 2;
  o.max_iterations = 1;
  s::UMAP m(d, {"x"}, o);
  auto&   c = m.directed_memberships();
  auto&   p = m.fuzzy_memberships();
  for (int i = 0; i < 4; ++i) {
    double sum = 0;
    for (int j = 0; j < 4; ++j) {
      sum += c(i, j);
      EXPECT_NEAR(p(i, j), c(i, j) + c(j, i) - c(i, j) * c(j, i), 1e-15);
    }
    EXPECT_NEAR(sum, 1, 1e-5);
  }
}
TEST(Batch404, UmapLimitsAndFixture) {
  datamunge::dstruct::DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 3, 7});
  s::UMAPOptions o;
  o.n_neighbors          = 2;
  o.max_iterations       = 1;
  o.negative_sample_rate = std::numeric_limits<double>::infinity();
  EXPECT_THROW(s::UMAP(d, {"x"}, o), std::invalid_argument);
  auto r = s::embedding_swiss_roll(4);
  EXPECT_EQ(r.nrows(), 4);
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(r.double_at("y", i), r.double_at("height", i));
}

TEST(Batch404, PrettyTicksRepresentableProgress) {
  auto p = datamunge::graphics::RPlot::create();
  p.points({0, 1}, {1, std::nextafter(1., 0.)});
  p.pretty_ticks();
  auto svg = p.to_svg();
  EXPECT_NE(svg.find("<svg"), std::string::npos);
  EXPECT_LT(svg.size(), 100000);
  auto tiny = datamunge::graphics::RPlot::create();
  tiny.points({0, 1}, {0, std::numeric_limits<double>::denorm_min()});
  tiny.pretty_ticks();
  EXPECT_LT(tiny.to_svg().size(), 100000);
}
