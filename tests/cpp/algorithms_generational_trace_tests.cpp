#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/generational.hpp>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
using namespace datamunge::algorithms;
using V = std::vector<int>;
struct Oracle {
  std::vector<char> full, young;
  V                 freed, remembered;
};
Oracle closure(const ObjectGraph& g, const V& gen) {
  const auto                     n = static_cast<std::size_t>(g.num_objects);
  std::vector<std::vector<bool>> f(n, std::vector<bool>(n)), y = f;
  for (std::size_t i = 0; i < n; ++i) {
    f[i][i] = true;
    if (gen[i] == 0)
      y[i][i] = true;
    for (int c : g.refs[i]) {
      f[i][c] = true;
      if (gen[i] == 0 && gen[c] == 0)
        y[i][c] = true;
    }
  }
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        f[i][j] = f[i][j] || (f[i][k] && f[k][j]);
        y[i][j] = y[i][j] || (y[i][k] && y[k][j]);
      }
  Oracle o;
  o.full.assign(n, 0);
  o.young.assign(n, 0);
  for (int r : g.roots)
    for (std::size_t j = 0; j < n; ++j) {
      o.full[j] |= f[r][j];
      if (gen[r] == 0)
        o.young[j] |= y[r][j];
    }
  for (std::size_t i = 0; i < n; ++i)
    if (gen[i] == 1) {
      bool remembered = false;
      for (int c : g.refs[i])
        if (gen[c] == 0) {
          remembered = true;
          for (std::size_t j = 0; j < n; ++j)
            o.young[j] |= y[c][j];
        }
      if (remembered)
        o.remembered.push_back(i);
    }
  for (std::size_t i = 0; i < n; ++i)
    if (gen[i] == 0 && !o.young[i])
      o.freed.push_back(i);
  return o;
}
V retained(const std::vector<char>& marks) {
  V ids;
  for (std::size_t i = 0; i < marks.size(); ++i)
    if (marks[i])
      ids.push_back(static_cast<int>(i));
  return ids;
}
void check(const ObjectGraph& g, const V& gen) {
  const auto  o = closure(g, gen);
  const auto  r = minor_gc_run(g, gen);
  const auto& v = r.result;
  EXPECT_EQ(v.promoted, o.young);
  EXPECT_EQ(v.young_freed, o.freed);
  EXPECT_EQ(v.remembered_set, o.remembered);
  EXPECT_EQ(r.full_reachable, o.full);
  EXPECT_EQ(v.minor_scanned, std::count(o.young.begin(), o.young.end(), 1));
  EXPECT_EQ(v.full_scanned, std::count(o.full.begin(), o.full.end(), 1));
  for (std::size_t i = 0; i < gen.size(); ++i)
    if (gen[i] == 0 && o.full[i])
      EXPECT_TRUE(v.promoted[i]);
  GenerationalHeap h(g, gen);
  const auto       sparse = h.collect();
  EXPECT_EQ(sparse.survivors, retained(o.young));
  EXPECT_EQ(sparse.freed, o.freed);
  EXPECT_EQ(h.remembered_sources(), o.remembered);
  EXPECT_EQ(sparse.work.validation.rows, 0);
  EXPECT_EQ(sparse.work.discovery_objects, 0);
  EXPECT_EQ(sparse.work.old_fields, 0);
  EXPECT_EQ(sparse.work.full_objects, 0);
  EXPECT_EQ(sparse.work.dense_result_slots, 0);
}

TEST(GenerationalTrace, HandCheckedSeedsLifoReadsAndFullComparison) {
  const ObjectGraph g{5, {{1}, {2}, {}, {}, {}}, {0, 3}, {}};
  const V           gen{1, 0, 0, 0, 0};
  const auto        t = minor_gc_trace(g, gen, 0);
  EXPECT_EQ(t.run.result.promoted, (std::vector<char>{0, 1, 1, 1, 0}));
  EXPECT_EQ(t.run.result.young_freed, V{4});
  EXPECT_EQ(t.run.result.remembered_set, V{0});
  EXPECT_EQ(t.run.result.minor_scanned, 3);
  EXPECT_EQ(t.run.result.full_scanned, 4);
  V                        seeds, visits, full;
  std::vector<std::size_t> sizes;
  for (const auto& e : t.events) {
    if (e.kind == "seed")
      seeds.push_back(e.object);
    if (e.kind == "visit") {
      visits.push_back(e.object);
      sizes.push_back(e.stack_size);
    }
    if (e.kind == "full_visit")
      full.push_back(e.object);
  }
  EXPECT_EQ(seeds, (V{3, 1}));
  EXPECT_EQ(visits, (V{1, 2, 3}));
  EXPECT_EQ(sizes, (std::vector<std::size_t>{1, 1, 0}));
  EXPECT_EQ(full, (V{3, 0, 1, 2}));
  const auto& w = t.run.work;
  EXPECT_EQ(w.old_fields, 1);
  EXPECT_EQ(w.young_fields, 1);
  EXPECT_EQ(w.seed_candidates, 2);
  EXPECT_EQ(w.seed_marks, 2);
  EXPECT_EQ(w.peak_young_stack, 2);
  EXPECT_EQ(w.full_fields, 2);
  EXPECT_EQ(w.classified_young, 4);
  check(g, gen);
}
TEST(GenerationalTrace, ExhaustiveAllGraphsRootsAndBinaryAgesThroughThreeObjects) {
  std::size_t cases = 0;
  for (int n = 0; n <= 3; ++n)
    for (unsigned edges = 0; edges < (1u << (n * n)); ++edges)
      for (unsigned roots = 0; roots < (1u << n); ++roots)
        for (unsigned ages = 0; ages < (1u << n); ++ages) {
          ObjectGraph g;
          g.num_objects = n;
          g.refs.resize(n);
          V gen(n);
          for (int i = 0; i < n; ++i) {
            gen[i] = (ages >> i) & 1u;
            if ((roots >> i) & 1u)
              g.roots.push_back(i);
            for (int j = 0; j < n; ++j)
              if ((edges >> (i * n + j)) & 1u)
                g.refs[i].push_back(j);
          }
          check(g, gen);
          ++cases;
        }
  EXPECT_EQ(cases, 33033);
}
TEST(GenerationalTrace, FloatingGarbageIsConservativeAndCanExceedFullLiveness) {
  const ObjectGraph g{3, {{1}, {2}, {}}, {}, {}};
  const V           gen{1, 0, 0};
  const auto        r = minor_gc_run(g, gen);
  EXPECT_EQ(r.result.minor_scanned, 2);
  EXPECT_EQ(r.result.full_scanned, 0);
  EXPECT_EQ(r.result.remembered_set, V{0});
  EXPECT_TRUE(r.result.young_freed.empty());
  EXPECT_EQ(r.full_reachable, (std::vector<char>{0, 0, 0}));
  check(g, gen);
  const ObjectGraph bridge{3, {{1}, {2}, {}}, {0}, {}};
  const V           ages{0, 1, 0};
  const auto        b = minor_gc_run(bridge, ages);
  EXPECT_EQ(b.result.promoted, (std::vector<char>{1, 0, 1}));
  EXPECT_EQ(b.work.young_fields, 1);
  check(bridge, ages);
}
TEST(GenerationalTrace, MaintainedSlotBarrierMatchesIndependentClosuresAfterTenThousandWrites) {
  std::mt19937_64 rng(131);
  for (int trial = 0; trial < 100; ++trial) {
    const int   n = 2 + trial % 15;
    ObjectGraph g;
    g.num_objects = n;
    g.refs.resize(n);
    V gen(n);
    for (int i = 0; i < n; ++i) {
      gen[i] = rng() % 2;
      for (int k = 0; k < 3; ++k)
        g.refs[i].push_back(rng() % n);
      if (rng() % 4 == 0)
        g.roots.push_back(i);
    }
    GenerationalHeap h(g, gen);
    const auto       before = g.refs;
    for (int write = 0; write < 100; ++write) {
      const int         source = rng() % n, target = rng() % n;
      const std::size_t field = rng() % 3;
      EXPECT_EQ(h.set_reference(source, field, target), gen[source] == 1 && gen[target] == 0);
      g.refs[source][field] = target;
      const auto o          = closure(g, gen);
      const auto r          = h.collect();
      EXPECT_EQ(r.survivors, retained(o.young));
      EXPECT_EQ(r.freed, o.freed);
      EXPECT_EQ(h.remembered_sources(), o.remembered);
      EXPECT_EQ(h.snapshot().refs, g.refs);
    }
    EXPECT_EQ(h.preparation_work().old_fields, 3 * std::count(gen.begin(), gen.end(), 1));
    (void)before;
  }
}
TEST(GenerationalTrace, WholeValidationRejectsEveryUnsafeOriginalInputBeforeCapture) {
  const std::vector<ObjectGraph> bad{
      {-1, {}, {}, {}},
      {2, {{1}}, {}, {}},
      {1, {{}}, {1}, {}},
      {2, {{99}, {}}, {0}, {}},
      {1, {{-1}}, {}, {}},
      {1, {{}}, {-1}, {}}};
  for (const auto& g : bad) {
    const V ages(g.num_objects > 0 ? g.num_objects : 0, 0);
    EXPECT_THROW(minor_gc_trace(g, ages, 1), std::invalid_argument);
    EXPECT_THROW(minor_gc(g, ages), std::invalid_argument);
    EXPECT_THROW((GenerationalHeap(g, ages)), std::invalid_argument);
  }
  const ObjectGraph g{2, {{1}, {}}, {0}, {}};
  for (const V ages : {V{}, V{1}, V{1, 0, 0}, V{2, 0}, V{-1, 0}}) {
    EXPECT_THROW(minor_gc_trace(g, ages, 1), std::invalid_argument);
    EXPECT_THROW((GenerationalHeap(g, ages)), std::invalid_argument);
  }
  auto ignored  = g;
  ignored.sizes = {-1, 0, 999};
  EXPECT_NO_THROW(minor_gc(ignored, {1, 0}));
  EXPECT_NO_THROW((GenerationalHeap(ignored, {1, 0})));
  EXPECT_THROW(generational_chain_example(0, 0, 1), std::invalid_argument);
  EXPECT_THROW(
      generational_chain_example(std::numeric_limits<std::size_t>::max()), std::length_error);
  EXPECT_THROW(
      generational_chain_example(1, std::numeric_limits<std::size_t>::max()), std::length_error);
  EXPECT_THROW(generational_chain_example(4, 1, 6), std::length_error);
  EXPECT_NO_THROW(generational_chain_example(4, 1, 7));
}
TEST(GenerationalTrace, ActualOnlineCapsEmptyInputAndOwnedTraceSnapshots) {
  ObjectGraph g{1, {{}}, {0}, {}};
  V           gen{0};
  const auto  baseline = minor_gc_trace(g, gen, 0);
  ASSERT_EQ(baseline.events.size(), 8);
  EXPECT_THROW(minor_gc_trace(g, gen, 7), std::length_error);
  EXPECT_NO_THROW(minor_gc_trace(g, gen, 8));
  auto t = minor_gc_trace(g, gen, 8);
  g.roots.clear();
  gen[0] = 1;
  t.graph.refs[0].push_back(0);
  t.generations[0] = 1;
  t.events[0].kind = "changed";
  EXPECT_EQ(t.run.result.minor_scanned, 1);
  EXPECT_EQ(t.run.result.promoted, (std::vector<char>{1}));
  EXPECT_EQ(t.events[1].kind, "seed");
  const auto empty = minor_gc_trace({}, {}, 1);
  EXPECT_TRUE(empty.events.empty());
  EXPECT_TRUE(empty.run.result.promoted.empty());
  EXPECT_EQ(empty.run.result.minor_scanned, 0);
  EXPECT_EQ(empty.run.result.full_scanned, 0);
  EXPECT_EQ(empty.run.work.ages_checked, 0);
}
TEST(GenerationalTrace, OriginalFamilyAndOldSizeScalingExposeWholeCostAndSparseBoundary) {
  for (std::size_t old : {1, 2, 4, 16, 1024, 100000})
    for (std::size_t garbage : {0, 1, 5}) {
      const auto  input = generational_chain_example(old, garbage, 0);
      const auto  r     = minor_gc_run(input.graph, input.generations);
      const auto& w     = r.work;
      const auto  n     = old + 2 + garbage;
      EXPECT_EQ(r.result.minor_scanned, 2);
      EXPECT_EQ(r.result.full_scanned, old + 2);
      EXPECT_EQ(w.validation.rows, n);
      EXPECT_EQ(w.validation.references, old);
      EXPECT_EQ(w.old_fields, old);
      EXPECT_EQ(w.discovery_objects, n);
      EXPECT_EQ(w.classified_young, 2 + garbage);
      EXPECT_EQ(w.dense_result_slots, n);
      EXPECT_EQ(w.full_objects, old + 2);
      EXPECT_EQ(w.full_fields, old);
      EXPECT_EQ(w.full_result_slots, n);
      GenerationalHeap h(input.graph, input.generations);
      const auto       y = h.collect();
      EXPECT_EQ(y.survivors, (V{static_cast<int>(old), static_cast<int>(old + 1)}));
      EXPECT_EQ(y.freed.size(), garbage);
      EXPECT_EQ(y.work.root_checks, 2);
      EXPECT_EQ(y.work.remembered_fields, 1);
      EXPECT_EQ(y.work.seed_candidates, 2);
      EXPECT_EQ(y.work.young_objects, 2);
      EXPECT_EQ(y.work.classified_young, 2 + garbage);
      EXPECT_EQ(y.work.old_fields, 0);
      EXPECT_EQ(y.work.full_objects, 0);
      EXPECT_EQ(y.work.dense_result_slots, 0);
      EXPECT_EQ(h.preparation_work().old_fields, old);
    }
}
TEST(GenerationalTrace, LongYoungChainIsIterativeAndNeverMarksAnOldBoundaryTarget) {
  constexpr int n = 100000;
  ObjectGraph   g;
  g.num_objects = n;
  g.refs.resize(n);
  g.roots = {0, 0};
  for (int i = 0; i < n - 1; ++i)
    g.refs[i] = {i + 1};
  V          gen(n, 0);
  const auto r = minor_gc_run(g, gen);
  EXPECT_EQ(r.result.minor_scanned, n);
  EXPECT_EQ(r.result.full_scanned, n);
  EXPECT_EQ(r.work.young_fields, n - 1);
  EXPECT_EQ(r.work.seed_candidates, 2);
  EXPECT_EQ(r.work.seed_marks, 1);
  EXPECT_EQ(r.work.peak_young_stack, 1);
  EXPECT_TRUE(r.result.young_freed.empty());
  GenerationalHeap h(g, gen);
  const auto       y = h.collect();
  EXPECT_EQ(y.survivors.size(), n);
  EXPECT_EQ(y.survivors.front(), 0);
  EXPECT_EQ(y.survivors.back(), n - 1);
  EXPECT_EQ(y.work.remembered_fields, 0);
  EXPECT_EQ(y.work.young_fields, n - 1);
}
TEST(GenerationalTrace, ActualPrimitiveEventsIndependentlyCertifyWorkStackAndDecisions) {
  std::mt19937_64 rng(999);
  for (int trial = 0; trial < 300; ++trial) {
    const int   n = 1 + trial % 12;
    ObjectGraph g;
    g.num_objects = n;
    g.refs.resize(n);
    V gen(n);
    for (int i = 0; i < n; ++i) {
      gen[i] = rng() % 2;
      for (int k = 0; k < 3; ++k)
        g.refs[i].push_back(rng() % n);
      if (rng() % 3 == 0)
        g.roots.push_back(i);
    }
    const auto  t    = minor_gc_trace(g, gen, 0);
    std::size_t root = 0, old = 0, seed = 0, initial_marks = 0, visit = 0, field = 0, classify = 0,
                fr = 0, fv = 0, ff = 0, peak = 0;
    bool traversing = false;
    V    stack, fullstack;
    for (const auto& e : t.events) {
      if (e.kind == "root") {
        ++root;
        EXPECT_EQ(e.accepted, gen[e.object] == 0);
      }
      if (e.kind == "old_field") {
        ++old;
        EXPECT_EQ(gen[e.object], 1);
        EXPECT_EQ(e.accepted, gen[e.target] == 0);
      }
      if (e.kind == "seed")
        ++seed;
      if (e.kind == "mark") {
        stack.push_back(e.object);
        if (!traversing)
          ++initial_marks;
        EXPECT_EQ(gen[e.object], 0);
        EXPECT_EQ(e.stack_size, stack.size());
        peak = std::max(peak, stack.size());
      }
      if (e.kind == "visit") {
        ++visit;
        traversing = true;
        ASSERT_FALSE(stack.empty());
        EXPECT_EQ(stack.back(), e.object);
        stack.pop_back();
        EXPECT_EQ(stack.size(), e.stack_size);
      }
      if (e.kind == "young_field") {
        ++field;
        EXPECT_EQ(gen[e.object], 0);
        EXPECT_NE(
            std::find(g.refs[e.object].begin(), g.refs[e.object].end(), e.target),
            g.refs[e.object].end());
      }
      if (e.kind == "promote" || e.kind == "free") {
        ++classify;
        EXPECT_EQ(gen[e.object], 0);
        EXPECT_EQ(e.accepted, bool(t.run.result.promoted[e.object]));
      }
      if (e.kind == "full_root")
        ++fr;
      if (e.kind == "full_mark") {
        fullstack.push_back(e.object);
        EXPECT_EQ(fullstack.size(), e.stack_size);
      }
      if (e.kind == "full_visit") {
        ++fv;
        ASSERT_FALSE(fullstack.empty());
        EXPECT_EQ(fullstack.back(), e.object);
        fullstack.pop_back();
        EXPECT_EQ(fullstack.size(), e.stack_size);
      }
      if (e.kind == "full_field")
        ++ff;
    }
    const auto& w = t.run.work;
    EXPECT_EQ(w.root_checks, root);
    EXPECT_EQ(w.old_fields, old);
    EXPECT_EQ(w.seed_candidates, seed);
    EXPECT_EQ(w.seed_marks, initial_marks);
    EXPECT_EQ(w.young_objects, visit);
    EXPECT_EQ(w.young_fields, field);
    EXPECT_EQ(w.classified_young, classify);
    EXPECT_EQ(w.full_roots, fr);
    EXPECT_EQ(w.full_objects, fv);
    EXPECT_EQ(w.full_fields, ff);
    EXPECT_EQ(w.peak_young_stack, peak);
    EXPECT_TRUE(stack.empty());
    EXPECT_TRUE(fullstack.empty());
    check(g, gen);
  }
}
TEST(GenerationalTrace, BarrierHandlesDuplicateSlotsAndInvalidWritesAtomicallyWithCopyOwnership) {
  const ObjectGraph g{4, {{1, 1}, {2}, {}, {}}, {0}, {}};
  const V           gen{1, 0, 0, 1};
  GenerationalHeap  h(g, gen);
  auto              copy = h;
  EXPECT_EQ(h.remembered_sources(), V{0});
  EXPECT_TRUE(h.set_reference(0, 0, 2));
  EXPECT_EQ(h.collect().survivors, (V{1, 2}));
  EXPECT_FALSE(h.set_reference(0, 0, 3));
  EXPECT_EQ(h.remembered_sources(), V{0});
  EXPECT_FALSE(h.set_reference(0, 1, 3));
  EXPECT_TRUE(h.remembered_sources().empty());
  EXPECT_TRUE(h.collect().survivors.empty());
  EXPECT_EQ(copy.collect().survivors, (V{1, 2}));
  EXPECT_EQ(g.refs[0], (V{1, 1}));
  const auto before = h.snapshot().refs;
  EXPECT_THROW(h.set_reference(-1, 0, 1), std::out_of_range);
  EXPECT_THROW(h.set_reference(4, 0, 1), std::out_of_range);
  EXPECT_THROW(h.set_reference(0, 2, 1), std::out_of_range);
  EXPECT_THROW(h.set_reference(0, 0, -1), std::out_of_range);
  EXPECT_THROW(h.set_reference(0, 0, 4), std::out_of_range);
  EXPECT_EQ(h.snapshot().refs, before);
  EXPECT_FALSE(h.set_reference(1, 0, 2));
  EXPECT_TRUE(h.remembered_sources().empty());
  auto snapshot       = copy.snapshot();
  snapshot.refs[0][0] = 3;
  EXPECT_EQ(copy.snapshot().refs[0][0], 1);
  EXPECT_EQ(h.preparation_work().old_fields, 2);
}
} // namespace
