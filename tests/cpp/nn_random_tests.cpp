// Tests for nn's seeded random tensors.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <set>
#include <vector>

using namespace datamunge;
using nn::DType;
using nn::Generator;
using nn::Shape;
using nn::Tensor;

namespace {

std::vector<double> vals(const Tensor& t) { return t.to_vector<double>(); }

double mean_of(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

double std_of(const std::vector<double>& v) {
  const double m = mean_of(v);
  double       s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / static_cast<double>(v.size()));
}

}  // namespace

TEST(NnRandom, SameSeedSameStream) {
  Generator a(42), b(42), c(43);
  std::vector<std::uint64_t> sa, sb, sc;
  for (int i = 0; i < 16; ++i) {
    sa.push_back(a.next_u64());
    sb.push_back(b.next_u64());
    sc.push_back(c.next_u64());
  }
  EXPECT_EQ(sa, sb);
  EXPECT_NE(sa, sc);
}

TEST(NnRandom, ReseedRestarts) {
  Generator g(7);
  const auto first = vals(nn::normal({8}, 0, 1, g));
  g.seed(7);
  EXPECT_EQ(vals(nn::normal({8}, 0, 1, g)), first);
  // The cached Box-Muller spare is dropped on reseed: an odd count still repeats.
  g.seed(9);
  const double n1 = g.normal();
  g.normal();
  g.seed(9);
  EXPECT_EQ(g.normal(), n1);
}

TEST(NnRandom, ManualSeedResetsDefault) {
  nn::manual_seed(123);
  const auto a = vals(nn::uniform({5}));
  nn::manual_seed(123);
  EXPECT_EQ(vals(nn::uniform({5})), a);
}

TEST(NnRandom, GeneratorDraws) {
  Generator g(1);
  for (int i = 0; i < 1000; ++i) {
    const double u = g.uniform01();
    EXPECT_GE(u, 0.0);
    EXPECT_LT(u, 1.0);
    EXPECT_LT(g.below(7), 7u);
  }
  EXPECT_THROW(g.below(0), std::invalid_argument);
}

TEST(NnRandom, Uniform) {
  Generator g(2);
  Tensor    u = nn::uniform({20000}, -2, 3, g);
  EXPECT_EQ(u.dtype(), DType::Float32);
  const auto v = vals(u);
  EXPECT_GE(*std::min_element(v.begin(), v.end()), -2.0);
  EXPECT_LT(*std::max_element(v.begin(), v.end()), 3.0);
  EXPECT_NEAR(mean_of(v), 0.5, 0.05);
  EXPECT_NEAR(std_of(v), 5 / std::sqrt(12.0), 0.05);
  EXPECT_EQ(nn::uniform({2, 3}, 0, 1, g, DType::Float64).dtype(), DType::Float64);
  EXPECT_EQ(nn::uniform({2, 3}, 0, 1, g).shape(), (Shape{2, 3}));
  EXPECT_THROW(nn::uniform({2}, 0, 1, g, DType::Int32), std::invalid_argument);
}

TEST(NnRandom, Normal) {
  Generator  g(3);
  const auto v = vals(nn::normal({20000}, 1.5, 2.0, g, DType::Float64));
  EXPECT_NEAR(mean_of(v), 1.5, 0.05);
  EXPECT_NEAR(std_of(v), 2.0, 0.05);
}

TEST(NnRandom, TruncatedNormalWithinTwoStd) {
  Generator  g(4);
  const auto v = vals(nn::truncated_normal({20000}, 1.0, 0.5, g));
  for (double x : v) {
    EXPECT_GE(x, 1.0 - 2 * 0.5 - 1e-6);
    EXPECT_LE(x, 1.0 + 2 * 0.5 + 1e-6);
  }
  EXPECT_NEAR(mean_of(v), 1.0, 0.02);
  // A normal truncated at +-2 sd has sd ~= 0.8796 of the parent.
  EXPECT_NEAR(std_of(v), 0.5 * 0.8796, 0.02);
}

TEST(NnRandom, Randint) {
  Generator g(5);
  Tensor    r = nn::randint({5000}, -3, 4, g);
  EXPECT_EQ(r.dtype(), DType::Int64);
  const auto       v = r.to_vector<std::int64_t>();
  std::set<std::int64_t> seen(v.begin(), v.end());
  EXPECT_EQ(seen, (std::set<std::int64_t>{-3, -2, -1, 0, 1, 2, 3}));
  EXPECT_THROW(nn::randint({1}, 2, 2, g), std::invalid_argument);
}

TEST(NnRandom, Bernoulli) {
  Generator  g(6);
  const auto v = vals(nn::bernoulli({20000}, 0.3, g));
  for (double x : v) EXPECT_TRUE(x == 0.0 || x == 1.0);
  EXPECT_NEAR(mean_of(v), 0.3, 0.02);
  // elementwise probabilities
  Tensor p = nn::tensor({0, 1, 0.5}, {}, DType::Float64);
  Tensor b = nn::bernoulli(nn::broadcast_to(p, {4000, 3}).contiguous(), g);
  EXPECT_EQ(b.dtype(), DType::Float64);
  EXPECT_EQ(b.shape(), (Shape{4000, 3}));
  const auto cols = vals(nn::mean(b, {0}));
  EXPECT_EQ(cols[0], 0.0);
  EXPECT_EQ(cols[1], 1.0);
  EXPECT_NEAR(cols[2], 0.5, 0.03);
  EXPECT_EQ(nn::bernoulli(nn::scalar(1.0), g).item(), 1.0);
}

TEST(NnRandom, CategoricalFrequencies) {
  Generator g(7);
  // logits for probabilities [0.2, 0.3, 0.5] in row 0, and a certain class in row 1
  Tensor logits = nn::tensor({{std::log(0.2), std::log(0.3), std::log(0.5)}, {-1e9, 0, -1e9}}, DType::Float64);
  Tensor s      = nn::categorical(logits, 20000, g);
  EXPECT_EQ(s.shape(), (Shape{2, 20000}));
  EXPECT_EQ(s.dtype(), DType::Int64);
  const auto        v = s.to_vector<std::int64_t>();
  std::vector<int>  counts(3, 0);
  for (int i = 0; i < 20000; ++i) ++counts[static_cast<std::size_t>(v[static_cast<std::size_t>(i)])];
  EXPECT_NEAR(counts[0] / 20000.0, 0.2, 0.02);
  EXPECT_NEAR(counts[1] / 20000.0, 0.3, 0.02);
  EXPECT_NEAR(counts[2] / 20000.0, 0.5, 0.02);
  for (int i = 20000; i < 40000; ++i) EXPECT_EQ(v[static_cast<std::size_t>(i)], 1);
  EXPECT_THROW(nn::categorical(nn::tensor({1, 2}), 1, g), std::invalid_argument);
}

TEST(NnRandom, MultinomialWithReplacement) {
  Generator  g(8);
  Tensor     s = nn::multinomial(nn::tensor({1, 0, 3}), 20000, true, g);  // unnormalised
  EXPECT_EQ(s.shape(), (Shape{20000}));
  const auto v = s.to_vector<std::int64_t>();
  std::vector<int> counts(3, 0);
  for (auto i : v) ++counts[static_cast<std::size_t>(i)];
  EXPECT_EQ(counts[1], 0);
  EXPECT_NEAR(counts[2] / 20000.0, 0.75, 0.02);
}

TEST(NnRandom, MultinomialWithoutReplacementNeverRepeats) {
  Generator g(9);
  for (int trial = 0; trial < 200; ++trial) {
    Tensor s = nn::multinomial(nn::tensor({{1, 2, 3, 4, 5}, {5, 0, 1, 1, 1}}), 4, false, g);
    EXPECT_EQ(s.shape(), (Shape{2, 4}));
    const auto v = s.to_vector<std::int64_t>();
    for (int r = 0; r < 2; ++r) {
      std::set<std::int64_t> row(v.begin() + r * 4, v.begin() + (r + 1) * 4);
      EXPECT_EQ(row.size(), 4u);
    }
    // row 1 has exactly 4 classes with positive weight, so class 1 is never drawn
    for (int k = 4; k < 8; ++k) EXPECT_NE(v[static_cast<std::size_t>(k)], 1);
  }
  EXPECT_THROW(nn::multinomial(nn::tensor({1, 1}), 3, false, g), std::invalid_argument);
  EXPECT_THROW(nn::multinomial(nn::tensor({1, -1}), 1, true, g), std::invalid_argument);
  EXPECT_THROW(nn::multinomial(nn::tensor({0, 0}), 1, true, g), std::invalid_argument);
}

TEST(NnRandom, Permutation) {
  Generator g(10);
  Tensor    p = nn::permutation(50, g);
  EXPECT_EQ(p.dtype(), DType::Int64);
  auto v = p.to_vector<std::int64_t>();
  std::vector<std::int64_t> sorted = v;
  std::sort(sorted.begin(), sorted.end());
  for (std::int64_t i = 0; i < 50; ++i) EXPECT_EQ(sorted[static_cast<std::size_t>(i)], i);
  EXPECT_NE(v, sorted);  // astronomically unlikely to be the identity
  EXPECT_EQ(nn::permutation(0, g).numel(), 0);
}

TEST(NnRandom, ShuffleKeepsRows) {
  Generator g(11);
  Tensor    x = nn::arange(0, 20).reshape({10, 2});
  Tensor    s = nn::shuffle(x, g);
  EXPECT_EQ(s.shape(), (Shape{10, 2}));
  const auto v = vals(s);
  std::vector<double> firsts;
  for (int r = 0; r < 10; ++r) {
    EXPECT_EQ(v[static_cast<std::size_t>(2 * r + 1)], v[static_cast<std::size_t>(2 * r)] + 1);  // rows intact
    firsts.push_back(v[static_cast<std::size_t>(2 * r)]);
  }
  std::sort(firsts.begin(), firsts.end());
  for (int r = 0; r < 10; ++r) EXPECT_EQ(firsts[static_cast<std::size_t>(r)], 2 * r);
  EXPECT_THROW(nn::shuffle(nn::scalar(1), g), std::invalid_argument);
}
