#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/lsh.hpp>
#include <numbers>
#include <random>
#include <set>
using namespace datamunge::algorithms;
TEST(LSHTrace, EveryRealProjectionBucketAndCandidate) {
  std::mt19937_64 rng(27);
  for (int run = 0; run < 50; ++run) {
    std::vector<std::vector<double>> data(20, std::vector<double>(3));
    for (auto& r : data)
      for (auto& v : r)
        v = int(rng() % 11) - 5;
    CosineLSH             index(data, 4, run % 8, 7);
    std::vector<double>   q{double(int(rng() % 11) - 5), 2, -1};
    auto                  t = index.query_trace(q);
    std::set<std::size_t> seen;
    double                best         = -2;
    std::size_t           winner       = SIZE_MAX;
    std::size_t           bucketvisits = 0;
    ASSERT_EQ(t.projections.size(), 4 * (run % 8));
    for (auto& p : t.projections) {
      double dot = 0;
      for (std::size_t j = 0; j < 3; ++j)
        dot += q[j] * index.hyperplanes()[p.table][p.bit][j];
      EXPECT_EQ(p.dot, dot);
      EXPECT_EQ(p.nonnegative, dot >= 0);
      EXPECT_FALSE(p.scaled);
    }
    for (auto& b : t.buckets) {
      EXPECT_EQ(b.signature, index.signature(q, b.table));
      std::vector<std::size_t> ids;
      for (std::size_t i = 0; i < data.size(); ++i)
        if (index.signature(data[i], b.table) == b.signature)
          ids.push_back(i);
      EXPECT_EQ(ids, b.members);
      bucketvisits += ids.size();
    }
    for (auto& c : t.candidates) {
      const bool duplicate = !seen.insert(c.id).second;
      EXPECT_EQ(c.duplicate, duplicate);
      if (!duplicate) {
        auto s = CosineLSH::cosine(q, data[c.id]);
        EXPECT_EQ(c.similarity, s);
        EXPECT_EQ(c.new_best, s > best);
        if (s > best) {
          best   = s;
          winner = c.id;
        }
      }
    }
    EXPECT_EQ(t.analysis.best, winner);
    EXPECT_EQ(t.analysis.similarity, best);
    EXPECT_EQ(t.analysis.examined, seen.size());
    EXPECT_EQ(t.analysis.bucket_visits, bucketvisits);
    EXPECT_EQ(t.analysis.seen_entries, data.size());
    EXPECT_EQ(t.analysis.best, index.query(q));
    EXPECT_EQ(t.captured_entries, 3 + t.projections.size() + t.buckets.size() + 2 * bucketvisits);
  }
}
TEST(LSHTrace, OwnedLifetimeDuplicateTiesZeroAndWidth) {
  std::vector<std::vector<double>> data{{1, 0}, {1, 0}, {0, 0}};
  CosineLSH                        index(data, 3, 0, 0);
  data[0]   = {-1, 0};
  auto copy = index;
  EXPECT_EQ(index.query({1, 0}), 0);
  EXPECT_EQ(copy.query({1, 0}), 0);
  EXPECT_EQ(index.data()[0][0], 1);
  auto t = index.query_trace({0, 0});
  EXPECT_EQ(t.analysis.best, 0);
  EXPECT_EQ(t.analysis.examined, 3);
  EXPECT_EQ(t.analysis.bucket_visits, 9);
  EXPECT_EQ(t.analysis.similarity, 0);
  EXPECT_EQ(t.captured_entries, 2 + 3 + 18);
  EXPECT_THROW(index.query_trace({1, 0}, 22), std::length_error);
  EXPECT_EQ(index.query_trace({1, 0}, 23).captured_entries, 23);
  CosineLSH full({{0, 0}}, 1, 64, 0);
  EXPECT_EQ(full.signature({0, 0}, 0), UINT64_MAX);
  EXPECT_EQ(CosineLSH({{1, 0}}, 0, 1).query({1, 0}), SIZE_MAX);
  EXPECT_EQ(CosineLSH({}, 2, 1).query({}), SIZE_MAX);
}
TEST(LSHTrace, CompleteFiniteGuardsAndScaledArithmetic) {
  EXPECT_THROW(CosineLSH({{}}, 1, 1), std::invalid_argument);
  EXPECT_THROW(CosineLSH({{1}, {2, 3}}, 1, 1), std::invalid_argument);
  EXPECT_THROW(CosineLSH({{NAN}}, 1, 1), std::invalid_argument);
  EXPECT_THROW(CosineLSH({{1}}, 1, 65), std::invalid_argument);
  CosineLSH   i({{1, 0}}, 1, 1);
  std::size_t examined = 123;
  EXPECT_THROW(i.query({NAN, 1}, &examined), std::invalid_argument);
  EXPECT_EQ(examined, 123);
  EXPECT_THROW(i.signature({1}, 0), std::invalid_argument);
  EXPECT_THROW(i.signature({1, 0}, 1), std::out_of_range);
  EXPECT_THROW(CosineLSH::cosine({1}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(CosineLSH::cosine({1}, {NAN}), std::invalid_argument);
  double m = std::numeric_limits<double>::max();
  EXPECT_NEAR(CosineLSH::cosine({m, m}, {m, m}), 1, 1e-15);
  EXPECT_NEAR(CosineLSH::cosine({m, -m}, {m, m}), 0, 1e-15);
  double tiny = std::numeric_limits<double>::denorm_min();
  EXPECT_DOUBLE_EQ(CosineLSH::cosine({tiny}, {tiny}), 1);
  for (double a : {-.1, std::numbers::pi + .1, std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(lsh_collision_probability(a), std::invalid_argument);
  EXPECT_EQ(lsh_collision_probability(0), 1);
  EXPECT_EQ(lsh_collision_probability(std::numbers::pi), 0);
}
