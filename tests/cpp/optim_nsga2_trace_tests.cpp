#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/optim/nsga2.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::optim;
namespace {
bool independent_dominates(const std::vector<double>& a, const std::vector<double>& b) {
  bool strict = false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] > b[i])
      return false;
    strict |= a[i] < b[i];
  }
  return strict;
}
std::size_t point_units(const std::vector<ParetoPoint>& p) {
  std::size_t n = 0;
  for (auto& v : p)
    n += 1 + v.coordinates.size() + v.objectives.size();
  return n;
}
std::size_t rank_units(const NSGA2RankingTrace& t) {
  std::size_t n = point_units(t.ranking.points) + 3 * t.ranking.points.size() + t.pairs.size();
  for (auto& f : t.ranking.fronts)
    n += 1 + f.size();
  for (auto& o : t.orders)
    n += 1 + o.order.size();
  return n;
}
struct Objective : MultiObjectiveFunction {
  std::vector<ParetoPoint> calls;
  std::vector<double>      evaluate(const std::vector<double>& x) override {
    std::vector<double> f{x[0], 1 - x[0] + x[1] * x[1]};
    calls.push_back({x, f});
    return f;
  }
};
} // namespace
TEST(NSGA2Trace, IndependentFrontPeelingAndEveryActualCrowdingOrder) {
  std::mt19937_64 rng(254263);
  for (int n = 0; n <= 20; ++n)
    for (int repeat = 0; repeat < 20; ++repeat) {
      std::vector<ParetoPoint> p;
      for (int i = 0; i < n; ++i)
        p.push_back({{double(i)}, {double(rng() % 5), double(rng() % 5), double(rng() % 3)}});
      auto             t = nsga2_rank_trace(p);
      std::vector<int> ranks(n, -1);
      int              remaining = n, rank = 0;
      while (remaining) {
        std::vector<int> front;
        for (int i = 0; i < n; ++i)
          if (ranks[i] < 0) {
            bool dominated = false;
            for (int j = 0; j < n; ++j)
              if (ranks[j] < 0 && j != i && independent_dominates(p[j].objectives, p[i].objectives))
                dominated = true;
            if (!dominated)
              front.push_back(i);
          }
        ASSERT_FALSE(front.empty());
        for (int i : front) {
          ranks[i] = rank;
          --remaining;
        }
        ++rank;
      }
      for (int i = 0; i < n; ++i)
        EXPECT_EQ(t.ranking.ranks[i], ranks[i]);
      EXPECT_EQ(t.pairs.size(), n * (n - 1) / 2);
      std::size_t calls = 0;
      for (auto& v : t.pairs) {
        int expected = independent_dominates(p[v.first].objectives, p[v.second].objectives)   ? -1
                       : independent_dominates(p[v.second].objectives, p[v.first].objectives) ? 1
                                                                                              : 0;
        EXPECT_EQ(v.relation, expected);
        EXPECT_EQ(v.calls, expected == -1 ? 1 : 2);
        calls += v.calls;
      }
      EXPECT_EQ(t.ranking.work.ranking_dominance_calls, calls);
      std::vector<double> crowding(n);
      std::size_t         gaps = 0;
      for (auto& o : t.orders) {
        auto ids = o.order, front = t.ranking.fronts[o.front];
        std::sort(ids.begin(), ids.end());
        std::sort(front.begin(), front.end());
        EXPECT_EQ(ids, front);
        for (std::size_t i = 1; i < o.order.size(); ++i)
          EXPECT_LE(
              p[o.order[i - 1]].objectives[o.objective], p[o.order[i]].objectives[o.objective]);
        crowding[o.order.front()] = INFINITY;
        crowding[o.order.back()]  = INFINITY;
        double range =
            p[o.order.back()].objectives[o.objective] - p[o.order.front()].objectives[o.objective];
        EXPECT_EQ(o.range, range);
        if (range > 0)
          for (std::size_t j = 1; j + 1 < o.order.size(); ++j)
            if (std::isfinite(crowding[o.order[j]])) {
              crowding[o.order[j]] += (p[o.order[j + 1]].objectives[o.objective] -
                                       p[o.order[j - 1]].objectives[o.objective]) /
                                      range;
              ++gaps;
            }
      }
      EXPECT_EQ(t.ranking.crowding, crowding);
      EXPECT_EQ(t.ranking.work.crowding_gaps, gaps);
      EXPECT_EQ(t.ranking.work.crowding_sorts, t.orders.size());
      EXPECT_EQ(t.captured_entries, rank_units(t));
      if (t.captured_entries > 1) {
        EXPECT_NO_THROW(nsga2_rank_trace(p, t.captured_entries));
        EXPECT_THROW(nsga2_rank_trace(p, t.captured_entries - 1), std::length_error);
      }
    }
}
TEST(NSGA2Trace, CapacityTruncatesNondominatedPointsAndMarksUnprocessedCrowding) {
  auto points = nsga2_truncation_points();
  auto t      = nsga2_survival_trace(points, 4);
  EXPECT_EQ(t.selected.size(), 4);
  for (auto r : t.combined.ranking.ranks)
    EXPECT_EQ(r, 0);
  std::vector<bool> chosen(6);
  for (auto i : t.selected) {
    EXPECT_FALSE(chosen[i]);
    chosen[i] = true;
  }
  for (std::size_t i = 0; i < 6; ++i)
    if (chosen[i])
      for (std::size_t j = 0; j < 6; ++j)
        if (!chosen[j])
          EXPECT_GE(t.combined.ranking.crowding[i], t.combined.ranking.crowding[j]);
  EXPECT_EQ(t.captured_entries, rank_units(t.combined) + 4);
  EXPECT_NO_THROW(nsga2_survival_trace(points, 4, t.captured_entries));
  EXPECT_THROW(nsga2_survival_trace(points, 4, t.captured_entries - 1), std::length_error);
  auto zero = nsga2_survival_trace(points, 0);
  EXPECT_TRUE(zero.selected.empty());
  EXPECT_TRUE(zero.combined.orders.empty());
  for (auto flag : zero.combined.ranking.crowding_assigned)
    EXPECT_FALSE(flag);
  auto constant = nsga2_rank_trace({{{}, {1, 1}}, {{}, {1, 1}}, {{}, {1, 1}}, {{}, {1, 1}}});
  EXPECT_EQ(constant.ranking.work.crowding_gaps, 0);
  for (auto& o : constant.orders)
    EXPECT_EQ(o.range, 0);
  auto original = nsga2_rank(nsga2_demo_points());
  EXPECT_EQ(original.fronts[0], (std::vector<std::size_t>{0, 1, 2, 3}));
  EXPECT_EQ(original.ranks, (std::vector<std::size_t>{0, 0, 0, 0, 1, 1}));
  EXPECT_NEAR(original.crowding[1], 1.5, 1e-15);
  EXPECT_NEAR(original.crowding[2], 1.125, 1e-15);
}
TEST(NSGA2Trace, ActualCallbacksTournamentAncestrySurvivorsAndCompleteQuota) {
  NSGA2Options options;
  options.population_size = 6;
  options.max_generations = 3;
  options.seed            = 7;
  Objective f;
  auto      t = NSGA2(options).trace(f, {0, 0}, {1, 1});
  EXPECT_EQ(f.calls.size(), 24);
  EXPECT_EQ(t.run.work.evaluations, 24);
  EXPECT_EQ(t.run.work.tournaments, 18);
  EXPECT_EQ(t.run.work.offspring_pairs, 9);
  EXPECT_EQ(t.run.work.ranking_passes, 4);
  EXPECT_EQ(t.run.work.ranking_pairs, 15 + 3 * 66);
  auto parent = t.initial.ranking;
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(parent.points[i].coordinates, f.calls[i].coordinates);
    EXPECT_EQ(parent.points[i].objectives, f.calls[i].objectives);
  }
  std::size_t units = 7 + 4 + rank_units(t.initial);
  for (std::size_t g = 0; g < t.generations.size(); ++g) {
    auto& gen = t.generations[g];
    EXPECT_EQ(gen.tournaments.size(), 6);
    EXPECT_EQ(gen.parents.size(), 3);
    for (auto& tr : gen.tournaments) {
      EXPECT_EQ(tr.rank_first, parent.ranks[tr.first]);
      EXPECT_EQ(tr.rank_second, parent.ranks[tr.second]);
      EXPECT_EQ(tr.crowding_first, parent.crowding[tr.first]);
      EXPECT_EQ(tr.crowding_second, parent.crowding[tr.second]);
      bool first = tr.rank_first != tr.rank_second ? tr.rank_first < tr.rank_second
                                                   : tr.crowding_first > tr.crowding_second;
      EXPECT_EQ(tr.winner, first ? tr.first : tr.second);
    }
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(gen.parents[j].first_parent, gen.tournaments[2 * j].winner);
      EXPECT_EQ(gen.parents[j].second_parent, gen.tournaments[2 * j + 1].winner);
      EXPECT_EQ(gen.parents[j].first_child, 6 + 2 * j);
      EXPECT_EQ(gen.parents[j].second_child, 6 + 2 * j + 1);
    }
    for (std::size_t j = 0; j < 6; ++j) {
      EXPECT_EQ(gen.combined.ranking.points[j].coordinates, parent.points[j].coordinates);
      EXPECT_EQ(gen.combined.ranking.points[6 + j].coordinates, f.calls[6 + g * 6 + j].coordinates);
      EXPECT_EQ(gen.combined.ranking.points[6 + j].objectives, f.calls[6 + g * 6 + j].objectives);
    }
    NSGA2Ranking next;
    for (auto i : gen.selected) {
      next.points.push_back(gen.combined.ranking.points[i]);
      next.ranks.push_back(gen.combined.ranking.ranks[i]);
      next.crowding.push_back(gen.combined.ranking.crowding[i]);
      EXPECT_TRUE(gen.combined.ranking.crowding_assigned[i]);
    }
    parent = std::move(next);
    units += 1 + gen.tournaments.size() + gen.parents.size() + rank_units(gen.combined) +
             gen.selected.size();
  }
  units += point_units(t.run.population) + point_units(t.run.front);
  EXPECT_EQ(t.captured_entries, units);
  Objective again;
  auto      r = NSGA2(options).run(again, {0, 0}, {1, 1});
  for (std::size_t i = 0; i < r.population.size(); ++i) {
    EXPECT_EQ(r.population[i].coordinates, t.run.population[i].coordinates);
    EXPECT_EQ(r.population[i].objectives, t.run.population[i].objectives);
  }
  Objective exact;
  EXPECT_NO_THROW(NSGA2(options).trace(exact, {0, 0}, {1, 1}, t.captured_entries));
  Objective shortage;
  EXPECT_THROW(
      NSGA2(options).trace(shortage, {0, 0}, {1, 1}, t.captured_entries - 1), std::length_error);
  EXPECT_GT(shortage.calls.size(), 0);
}
TEST(NSGA2Trace, CompletePreCallbackGuardsAndObjectiveFailureEffects) {
  Objective    f;
  NSGA2Options o;
  o.population_size = 6;
  o.max_generations = 0;
  EXPECT_THROW(NSGA2(o).run(f, {0}, {0}), std::invalid_argument);
  o.crossover_rate = NAN;
  EXPECT_THROW(NSGA2(o).run(f, {0, 0}, {1, 1}), std::invalid_argument);
  o.crossover_rate = .9;
  o.eta_mutation   = -1;
  EXPECT_THROW(NSGA2(o).run(f, {0, 0}, {1, 1}), std::invalid_argument);
  o.eta_mutation    = 20;
  o.population_size = 1000001;
  EXPECT_THROW(NSGA2(o).run(f, {0, 0}, {1, 1}), std::length_error);
  o.population_size = 6;
  o.max_generations = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(NSGA2(o).run(f, {0, 0}, {1, 1}), std::length_error);
  o.max_generations = 0;
  EXPECT_THROW(NSGA2(o).trace(f, {0, 0}, {1, 1}, 1), std::length_error);
  EXPECT_TRUE(f.calls.empty());
  struct Bad : MultiObjectiveFunction {
    int                 count = 0;
    std::vector<double> evaluate(const std::vector<double>&) override {
      ++count;
      return count == 1 ? std::vector<double>{0, 1} : std::vector<double>{0, 1, 2};
    }
  } bad;
  EXPECT_THROW(NSGA2(o).run(bad, {0, 0}, {1, 1}), std::domain_error);
  EXPECT_EQ(bad.count, 2);
  for (std::size_t requested : {0, 1, 5}) {
    o.population_size = requested;
    Objective small;
    auto      r = NSGA2(o).run(small, {0, 0}, {1, 1});
    EXPECT_EQ(r.effective_population, requested < 2 ? 2 : 6);
    EXPECT_EQ(r.work.evaluations, r.effective_population);
  }
  EXPECT_THROW(nsga2_rank({{{}, {0}}, {{}, {1}}}), std::invalid_argument);
  EXPECT_THROW(nsga2_rank({{{}, {0, 1}}, {{}, {1, NAN}}}), std::invalid_argument);
  EXPECT_THROW(nsga2_survival_trace(nsga2_demo_points(), 7), std::invalid_argument);
  double m = std::numeric_limits<double>::max();
  EXPECT_THROW(nsga2_rank({{{}, {m, -m}}, {{}, {-m, m}}}), std::overflow_error);
  EXPECT_TRUE(nsga2_rank({}).points.empty());
}
