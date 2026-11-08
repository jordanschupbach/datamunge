#include <gtest/gtest.h>

#include <array>
#include <datamunge/algorithms/general_problem_solver.hpp>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace {
using namespace datamunge::algorithms;
using A = std::vector<PlanningAction>;
using I = std::vector<std::size_t>;
using S = std::vector<std::uint64_t>;
using K = PlanningEventKind;

// Independent breadth-first exploration; no DFS best-depth pruning is used.
int distance(std::uint64_t initial, std::uint64_t goals, const A& ops) {
  std::queue<std::uint64_t> q;
  q.push(initial);
  std::unordered_map<std::uint64_t, int> d{{initial, 0}};
  while (!q.empty()) {
    const auto state = q.front();
    q.pop();
    if ((state & goals) == goals)
      return d.at(state);
    for (const auto& op : ops)
      if ((state & op.require) == op.require) {
        const auto next = (state & ~op.remove) | op.add;
        if (!d.contains(next)) {
          d.emplace(next, d.at(state) + 1);
          q.push(next);
        }
      }
  }
  return -1;
}
void verify(
    std::uint64_t      initial,
    std::uint64_t      goals,
    const A&           ops,
    std::size_t        budget,
    const PlanningRun& r) {
  ASSERT_EQ(r.result.actions.size(), r.action_indices.size());
  if (!r.result.found) {
    EXPECT_TRUE(r.action_indices.empty());
    EXPECT_EQ(r.result.final_state, 0);
    return;
  }
  EXPECT_LE(r.action_indices.size(), budget);
  auto state = initial;
  for (std::size_t k = 0; k < r.action_indices.size(); ++k) {
    const auto index = r.action_indices[k];
    ASSERT_LT(index, ops.size());
    const auto& op = ops[index];
    EXPECT_EQ(state & op.require, op.require);
    EXPECT_EQ(r.result.actions[k], op.name);
    state &= ~op.remove;
    state |= op.add;
  }
  EXPECT_EQ(state, r.result.final_state);
  EXPECT_EQ(state & goals, goals);
}

TEST(GeneralPlanner, OriginalMakeHasActualSixEventExecutionAndStableRecords) {
  const A    ops{{"make", 0, 1, 0}};
  const auto t = general_problem_solver_trace(0, 1, ops, 32, 6);
  ASSERT_TRUE(t.run.result.found);
  EXPECT_EQ(t.run.result.actions, (std::vector<std::string>{"make"}));
  EXPECT_EQ(t.run.action_indices, I{0});
  ASSERT_EQ(t.events.size(), 6);
  const std::array<K, 6> kinds{K::visit, K::test, K::apply, K::visit, K::goal, K::finish};
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_EQ(t.events[i].kind, kinds[i]);
  EXPECT_EQ(t.events[1].state, 0);
  EXPECT_TRUE(t.events[1].applicable);
  EXPECT_EQ(t.events[2].next_state, 1);
  EXPECT_EQ(t.events[3].depth, 1);
  EXPECT_EQ(t.events[5].action_index, 0);
  EXPECT_EQ(t.run.work.visits, 2);
  EXPECT_EQ(t.run.work.expanded, 1);
  EXPECT_EQ(t.run.work.operator_checks, 1);
  EXPECT_EQ(t.run.work.applicable, 1);
  EXPECT_EQ(t.run.work.peak_stack, 2);
  EXPECT_EQ(t.run.work.distinct_states, 1);
  const auto old = general_problem_solver(0, 1, ops);
  EXPECT_EQ(old.actions, t.run.result.actions);
  EXPECT_EQ(old.final_state, 1);
  verify(0, 1, ops, 32, t.run);
}

TEST(GeneralPlanner, ShallowerReopeningRepairsBoundedFalseNegativeWithoutShortestnessClaim) {
  const A ops{{"clear", 0, 0, 4}, {"set", 0, 6, 2}, {"consume", 2, 1, 7}};
  ASSERT_EQ(distance(4, 3, ops), 3);
  for (std::size_t budget = 0; budget < 3; ++budget)
    EXPECT_FALSE(general_problem_solver(4, 3, ops, budget).found);
  const auto t = general_problem_solver_trace(4, 3, ops, 3, 0);
  ASSERT_TRUE(t.run.result.found);
  EXPECT_EQ(t.run.action_indices, (I{1, 2, 1}));
  const auto c = check_planning_plan(4, 3, ops, t.run.action_indices, 4);
  EXPECT_TRUE(c.applicable);
  EXPECT_TRUE(c.reaches_goals);
  EXPECT_EQ(c.states, (S{4, 6, 1, 7}));
  EXPECT_GT(t.run.work.reopenings, 0);
  bool actual = false;
  for (const auto& e : t.events)
    if (e.kind == K::reopen && e.state == 6 && e.depth == 1 && e.previous_depth == 2) {
      actual = true;
      EXPECT_TRUE(e.has_previous_depth);
    }
  EXPECT_TRUE(actual);
  const auto four = general_problem_solver_run(4, 3, ops, 4);
  EXPECT_TRUE(four.result.found);
  EXPECT_EQ(four.action_indices, (I{0, 1, 2, 1}));
  EXPECT_GT(four.action_indices.size(), 3);
  verify(4, 3, ops, 4, four);
}

TEST(GeneralPlanner, ExhaustiveTwoFactOperatorsAndIndependentBoundedBfs) {
  for (unsigned x = 0; x < 64; ++x)
    for (unsigned y = 0; y < 64; ++y) {
      const A ops{{"a", x % 4, (x / 4) % 4, (x / 16) % 4}, {"b", y % 4, (y / 4) % 4, (y / 16) % 4}};
      for (std::uint64_t initial = 0; initial < 4; ++initial)
        for (std::uint64_t goals = 0; goals < 4; ++goals) {
          const auto d = distance(initial, goals, ops);
          for (std::size_t budget = 0; budget <= 4; ++budget) {
            const auto r = general_problem_solver_run(initial, goals, ops, budget);
            EXPECT_EQ(r.result.found, d >= 0 && static_cast<std::size_t>(d) <= budget);
            verify(initial, goals, ops, budget, r);
          }
        }
    }
}

TEST(GeneralPlanner, RandomThreeAndFourFactPlansAgreeWithIndependentBfs) {
  std::mt19937_64 rng(130);
  for (int trial = 0; trial < 3000; ++trial) {
    const std::uint64_t mask = trial % 2 ? 15 : 7;
    A                   ops;
    for (int i = 0; i < 3 + trial % 5; ++i)
      ops.push_back({"op" + std::to_string(i), rng() & mask, rng() & mask, rng() & mask});
    const auto initial = rng() & mask, goals = rng() & mask;
    const auto d = distance(initial, goals, ops);
    for (std::size_t budget = 0; budget <= 6; ++budget) {
      const auto r = general_problem_solver_run(initial, goals, ops, budget);
      EXPECT_EQ(r.result.found, d >= 0 && static_cast<std::size_t>(d) <= budget);
      verify(initial, goals, ops, budget, r);
    }
  }
}

TEST(GeneralPlanner, GoalBeforeDepthBeforeRepeatAndFailurePlaceholders) {
  const auto yes = general_problem_solver_trace(7, 3, {}, 0, 2);
  EXPECT_TRUE(yes.run.result.found);
  EXPECT_EQ(yes.run.result.final_state, 7);
  EXPECT_EQ(yes.run.work.expanded, 0);
  EXPECT_EQ(yes.events.back().kind, K::goal);
  const auto no = general_problem_solver_trace(7, 8, {}, 0, 2);
  EXPECT_FALSE(no.run.result.found);
  EXPECT_EQ(no.run.result.final_state, 0);
  EXPECT_EQ(no.events.back().kind, K::cutoff);
  EXPECT_EQ(no.run.work.depth_cutoffs, 1);
  const auto loop = general_problem_solver_trace(0, 1, {{"noop", 0, 0, 0}}, 1, 0);
  EXPECT_EQ(loop.run.work.depth_cutoffs, 1);
  EXPECT_EQ(loop.run.work.repeat_prunes, 0);
  const auto repeat = general_problem_solver_trace(0, 1, {{"noop", 0, 0, 0}}, 2, 0);
  EXPECT_EQ(repeat.run.work.depth_cutoffs, 0);
  EXPECT_EQ(repeat.run.work.repeat_prunes, 1);
  EXPECT_EQ(repeat.run.work.backtracks, 1);
  const auto blocked = general_problem_solver_trace(0, 1, {{"blocked", 2, 1, 0}}, 32, 0);
  EXPECT_FALSE(blocked.run.result.found);
  EXPECT_EQ(blocked.run.work.operator_checks, 1);
  EXPECT_EQ(blocked.run.work.applicable, 0);
  EXPECT_EQ(blocked.events.back().kind, K::exhausted);
  EXPECT_TRUE(general_problem_solver(99, 0, {}, std::numeric_limits<std::size_t>::max()).found);
}

TEST(GeneralPlanner, CertificatesValidateAllIndicesBeforeQuotaAndKeepActualFailurePrefix) {
  const A    ops{{"key", 0, 1, 0}, {"open", 1, 2, 1}, {"enter", 2, 4, 0}};
  const auto c = check_planning_plan(0, 4, ops, {0, 1, 2}, 4);
  EXPECT_TRUE(c.applicable);
  EXPECT_TRUE(c.reaches_goals);
  EXPECT_EQ(c.states, (S{0, 1, 2, 6}));
  EXPECT_EQ(c.applied, 3);
  const auto fail = check_planning_plan(0, 4, ops, {0, 2}, 3);
  EXPECT_FALSE(fail.applicable);
  EXPECT_FALSE(fail.reaches_goals);
  EXPECT_EQ(fail.applied, 1);
  EXPECT_EQ(fail.final_state, 1);
  EXPECT_EQ(fail.states, (S{0, 1}));
  EXPECT_THROW(check_planning_plan(0, 4, ops, {2, 999}, 1), std::out_of_range);
  EXPECT_THROW(check_planning_plan(0, 4, ops, {2, 0, 1}, 1), std::length_error);
  const auto empty = check_planning_plan(4, 4, ops, {}, 1);
  EXPECT_TRUE(empty.applicable);
  EXPECT_TRUE(empty.reaches_goals);
  EXPECT_EQ(empty.states, S{4});
  const auto missed = check_planning_plan(0, 4, ops, {0}, 2);
  EXPECT_TRUE(missed.applicable);
  EXPECT_FALSE(missed.reaches_goals);
  const auto goal_then_fail = check_planning_plan(4, 4, ops, {2}, 2);
  EXPECT_FALSE(goal_then_fail.applicable);
  EXPECT_TRUE(goal_then_fail.reaches_goals);
  EXPECT_EQ(goal_then_fail.final_state, 4);
}

TEST(GeneralPlanner, HighBitAddWinsArbitraryNamesAndIndicesDisambiguate) {
  const auto high = std::uint64_t{1} << 63;
  A          ops{{"", 0, high, high}, {"", high, 1, high}, {std::string("x\0y", 3), 1, 2, 0}};
  const auto r = general_problem_solver_run(0, 2, ops, 3);
  ASSERT_TRUE(r.result.found);
  EXPECT_EQ(r.action_indices, (I{0, 1, 2}));
  EXPECT_EQ(r.result.actions[0], r.result.actions[1]);
  EXPECT_EQ(r.result.actions[2].size(), 3);
  verify(0, 2, ops, 3, r);
  const auto c = check_planning_plan(0, 2, ops, r.action_indices, 4);
  EXPECT_EQ(c.states, (S{0, high, 1, 3}));
  const auto first = general_problem_solver_run(0, high, ops, 1);
  EXPECT_EQ(first.result.final_state, high);
  const auto all = planning_binary_counter(64);
  ASSERT_EQ(all.size(), 64);
  EXPECT_EQ(all.back().require, high - 1);
  EXPECT_EQ(all.back().add, high);
  EXPECT_EQ(all.back().remove, high - 1);
  EXPECT_THROW(planning_binary_counter(0), std::invalid_argument);
  EXPECT_THROW(planning_binary_counter(65), std::invalid_argument);
  EXPECT_THROW(
      planning_binary_counter(std::numeric_limits<std::size_t>::max()), std::invalid_argument);
}

TEST(GeneralPlanner, OnlineActualEventCapsAndIndependentOwnedSnapshots) {
  A          ops{{"make", 0, 1, 0}};
  const auto original = ops;
  EXPECT_THROW(general_problem_solver_trace(0, 1, ops, 32, 5), std::length_error);
  EXPECT_NO_THROW(general_problem_solver_trace(0, 1, ops, 32, 6));
  EXPECT_NO_THROW(general_problem_solver_trace(0, 1, ops, 32, 0));
  EXPECT_THROW(general_problem_solver_trace(1, 1, ops, 0, 1), std::length_error);
  EXPECT_NO_THROW(general_problem_solver_trace(1, 1, ops, 0, 2));
  auto t      = general_problem_solver_trace(0, 1, ops, 32, 6);
  ops[0].name = "changed";
  ops[0].add  = 999;
  EXPECT_EQ(t.operators[0].name, "make");
  EXPECT_EQ(t.operators[0].add, 1);
  t.operators[0].name    = "trace changed";
  t.events[2].next_state = 999;
  EXPECT_EQ(t.run.result.actions[0], "make");
  EXPECT_EQ(t.run.result.final_state, 1);
  EXPECT_EQ(t.events[3].state, 1);
  EXPECT_EQ(original[0].name, "make");
  EXPECT_EQ(original[0].add, 1);
  auto cert      = check_planning_plan(0, 1, original, {0}, 2);
  cert.states[0] = 999;
  EXPECT_EQ(cert.final_state, 1);
  EXPECT_EQ(cert.states[1], 1);
}

TEST(GeneralPlanner, ActualEventCertificatesReconstructWorkAndStackOnRandomSearches) {
  std::mt19937_64 rng(777);
  for (int trial = 0; trial < 400; ++trial) {
    A ops;
    for (int i = 0; i < 5; ++i)
      ops.push_back({"op" + std::to_string(i), rng() % 8, rng() % 8, rng() % 8});
    const auto  initial = rng() % 8, goals = rng() % 8;
    const auto  t      = general_problem_solver_trace(initial, goals, ops, 4, 0);
    std::size_t visits = 0, checks = 0, applies = 0, repeats = 0, reopens = 0, cuts = 0, backs = 0,
                peak = 0;
    std::unordered_map<std::uint64_t, std::size_t> best;
    for (const auto& e : t.events) {
      if (e.kind == K::visit) {
        ++visits;
        peak = std::max(peak, e.depth + 1);
      }
      if (e.kind == K::test) {
        ++checks;
        ASSERT_LT(e.action_index, ops.size());
        EXPECT_EQ(
            e.applicable, (e.state & ops[e.action_index].require) == ops[e.action_index].require);
      }
      if (e.kind == K::apply) {
        ++applies;
        ASSERT_LT(e.action_index, ops.size());
        const auto& o = ops[e.action_index];
        EXPECT_EQ(e.next_state, (e.state & ~o.remove) | o.add);
        EXPECT_EQ(e.state & o.require, o.require);
      }
      if (e.kind == K::repeat) {
        ++repeats;
        EXPECT_TRUE(e.has_previous_depth);
        EXPECT_LE(e.previous_depth, e.depth);
      }
      if (e.kind == K::reopen) {
        ++reopens;
        EXPECT_TRUE(e.has_previous_depth);
        EXPECT_GT(e.previous_depth, e.depth);
      }
      if (e.kind == K::cutoff) {
        ++cuts;
        EXPECT_EQ(e.depth, 4);
        EXPECT_NE(e.state & goals, goals);
      }
      if (e.kind == K::backtrack)
        ++backs;
      if (e.kind != K::apply)
        EXPECT_EQ(e.next_state, e.state);
    }
    const auto& w = t.run.work;
    EXPECT_EQ(w.visits, visits);
    EXPECT_EQ(w.operator_checks, checks);
    EXPECT_EQ(w.applicable, applies);
    EXPECT_EQ(w.repeat_prunes, repeats);
    EXPECT_EQ(w.reopenings, reopens);
    EXPECT_EQ(w.depth_cutoffs, cuts);
    EXPECT_EQ(w.backtracks, backs);
    EXPECT_EQ(w.peak_stack, peak);
    EXPECT_EQ(w.visits, w.applicable + 1);
    EXPECT_EQ(w.expanded, w.distinct_states + w.reopenings);
    const auto r = general_problem_solver_run(initial, goals, ops, 4);
    EXPECT_EQ(r.action_indices, t.run.action_indices);
    EXPECT_EQ(r.result.final_state, t.run.result.final_state);
    EXPECT_EQ(r.work.operator_checks, w.operator_checks);
    verify(initial, goals, ops, 4, t.run);
  }
  for (auto k :
       {K::visit,
        K::goal,
        K::cutoff,
        K::repeat,
        K::reopen,
        K::test,
        K::apply,
        K::backtrack,
        K::finish,
        K::exhausted})
    EXPECT_FALSE(planning_event_name(k).empty());
  EXPECT_THROW(planning_event_name(static_cast<K>(-1)), std::invalid_argument);
}

TEST(GeneralPlanner, DeepHeapCounterFamilyHasIndependentExactCostAndStateCertificate) {
  for (std::size_t bits = 1; bits <= 16; ++bits) {
    const auto goal = std::uint64_t{1} << (bits - 1);
    const auto ops  = planning_binary_counter(bits);
    const auto r    = general_problem_solver_run(0, goal, ops, goal);
    ASSERT_TRUE(r.result.found);
    EXPECT_EQ(r.action_indices.size(), goal);
    EXPECT_EQ(r.work.visits, 2 * goal);
    EXPECT_EQ(r.work.operator_checks, 2 * goal - 1);
    EXPECT_EQ(r.work.applicable, 2 * goal - 1);
    EXPECT_EQ(r.work.expanded, goal);
    EXPECT_EQ(r.work.distinct_states, goal);
    EXPECT_EQ(r.work.backtracks, goal - 1);
    EXPECT_EQ(r.work.reopenings, 0);
    EXPECT_EQ(r.work.peak_stack, goal + 1);
    EXPECT_EQ(r.work.depth_cutoffs, bits - 1);
    EXPECT_EQ(r.work.repeat_prunes, goal - bits);
    const auto c = check_planning_plan(0, goal, ops, r.action_indices, goal + 1);
    ASSERT_TRUE(c.applicable);
    EXPECT_TRUE(c.reaches_goals);
    ASSERT_EQ(c.states.size(), goal + 1);
    for (std::uint64_t i = 0; i <= goal; ++i)
      EXPECT_EQ(c.states[i], i);
    if (bits <= 8)
      EXPECT_FALSE(general_problem_solver(0, goal, ops, goal - 1).found);
  }
}
} // namespace
