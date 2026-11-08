#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <datamunge/algorithms/stable_matching.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
using namespace datamunge::algorithms;
namespace {
std::size_t rank_of(const std::vector<std::size_t>& row, std::size_t x) {
  return std::size_t(std::find(row.begin(), row.end(), x) - row.begin());
}
bool feasible_oracle(const StableMatchingResult& m, std::size_t n) {
  if (m.proposer_match.size() != n || m.reviewer_match.size() != n)
    return false;
  std::vector<std::size_t> inverse(n, kUnmatched);
  std::set<std::size_t>    used;
  for (std::size_t i = 0; i < n; ++i) {
    auto j = m.proposer_match[i];
    if (j == kUnmatched)
      continue;
    if (j >= n || !used.insert(j).second)
      return false;
    inverse[j] = i;
  }
  return inverse == m.reviewer_match;
}
std::vector<std::pair<std::size_t, std::size_t>> blocks_oracle(
    const StableMatchingResult& m,
    const StablePreferences&    p,
    const StablePreferences&    r) {
  std::vector<std::pair<std::size_t, std::size_t>> out;
  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t j = 0; j < r.size(); ++j) {
      const auto pi = m.proposer_match[i], pj = m.reviewer_match[j];
      if (rank_of(p[i], j) < rank_of(p[i], pi) && rank_of(r[j], i) < rank_of(r[j], pj))
        out.push_back({i, j});
    }
  return out;
}
StableMatchingResult candidate(const std::vector<std::size_t>& permutation) {
  StableMatchingResult m;
  m.proposer_match = permutation;
  m.reviewer_match.resize(permutation.size());
  for (std::size_t i = 0; i < permutation.size(); ++i)
    m.reviewer_match[permutation[i]] = i;
  return m;
}
StablePreferences random_prefs(std::size_t n, std::mt19937_64& rng) {
  StablePreferences p(n, std::vector<std::size_t>(n));
  for (auto& row : p) {
    std::iota(row.begin(), row.end(), 0);
    std::shuffle(row.begin(), row.end(), rng);
  }
  return p;
}
void certificate(const GaleShapleyTrace& t) {
  const auto&          p = t.proposer_preferences;
  const auto&          r = t.reviewer_preferences;
  const auto           n = p.size();
  StableMatchingResult before;
  before.proposer_match.assign(n, kUnmatched);
  before.reviewer_match.assign(n, kUnmatched);
  std::vector<std::size_t> queue(n), counts(n);
  std::iota(queue.begin(), queue.end(), 0);
  std::vector<std::set<std::size_t>> arrivals(n), tried(n);
  std::size_t free_accepts = 0, replacements = 0, rejections = 0, entries = 3 * (n + n * n),
              number = 0;
  for (const auto& e : t.events) {
    SCOPED_TRACE(e.number);
    ASSERT_EQ(e.number, ++number);
    ASSERT_EQ(e.queue_before, queue);
    ASSERT_FALSE(queue.empty());
    ASSERT_EQ(queue.front(), e.proposer);
    const auto i = e.proposer, j = e.reviewer;
    ASSERT_LT(i, n);
    ASSERT_LT(j, n);
    ASSERT_EQ(before.proposer_match[i], kUnmatched);
    ASSERT_EQ(e.held, before.reviewer_match[j]);
    ASSERT_TRUE(tried[i].insert(j).second);
    ASSERT_EQ(j, p[i][counts[i]++]);
    EXPECT_EQ(e.next_choice, counts);
    arrivals[j].insert(i);
    const auto best =
        *std::min_element(arrivals[j].begin(), arrivals[j].end(), [&](auto a, auto b) {
          return rank_of(r[j], a) < rank_of(r[j], b);
        });
    ASSERT_EQ(e.reviewer_match[j], best);
    StableMatchingResult after{e.proposer_match, e.reviewer_match, e.number};
    ASSERT_TRUE(feasible_oracle(after, n));
    queue.erase(queue.begin());
    if (e.held == kUnmatched) {
      EXPECT_EQ(e.decision, "accept-free");
      EXPECT_EQ(e.newly_free, kUnmatched);
      ++free_accepts;
    } else if (best == i) {
      EXPECT_EQ(e.decision, "replace");
      EXPECT_EQ(e.newly_free, e.held);
      EXPECT_EQ(after.proposer_match[e.held], kUnmatched);
      EXPECT_LT(rank_of(r[j], i), rank_of(r[j], e.held));
      queue.push_back(e.held);
      ++replacements;
    } else {
      EXPECT_EQ(e.decision, "reject");
      EXPECT_EQ(e.newly_free, i);
      EXPECT_EQ(after.proposer_match[i], kUnmatched);
      queue.push_back(i);
      ++rejections;
    }
    EXPECT_EQ(e.queue_after, queue);
    std::set<std::size_t> free;
    for (std::size_t a = 0; a < n; ++a)
      if (after.proposer_match[a] == kUnmatched)
        free.insert(a);
    EXPECT_EQ(std::set<std::size_t>(queue.begin(), queue.end()), free);
    EXPECT_EQ(queue.size(), free.size());
    for (std::size_t a = 0; a < n; ++a) {
      if (a != i && a != e.held)
        EXPECT_EQ(after.proposer_match[a], before.proposer_match[a]);
      if (a != j)
        EXPECT_EQ(after.reviewer_match[a], before.reviewer_match[a]);
    }
    entries += 1 + 3 * n + e.queue_before.size() + e.queue_after.size();
    before = after;
  }
  EXPECT_EQ(t.result.proposer_match, before.proposer_match);
  EXPECT_EQ(t.result.reviewer_match, before.reviewer_match);
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(t.result.proposals, t.events.size());
  EXPECT_EQ(t.captured_entries, entries);
  EXPECT_LE(entries, gale_shapley_trace_entries(n));
  EXPECT_LE(t.result.proposals, gale_shapley_proposal_bound(n));
  EXPECT_EQ(t.work.rank_entries, n * n);
  EXPECT_EQ(t.work.proposals, number);
  EXPECT_EQ(t.work.next_choice_advances, number);
  EXPECT_EQ(t.work.accepted_free, free_accepts);
  EXPECT_EQ(free_accepts, n);
  EXPECT_EQ(t.work.replacements, replacements);
  EXPECT_EQ(t.work.rejections, rejections);
  EXPECT_EQ(t.work.queue_pushes, number);
  EXPECT_EQ(t.work.queue_pops, number);
  EXPECT_EQ(t.work.peak_free, n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(t.reviewer_rank[j][i], rank_of(r[j], i));
  EXPECT_TRUE(blocks_oracle(t.result, p, r).empty());
}
} // namespace
TEST(StableMatchingChecked, EveryCompleteProfileThroughThreeAgentsHasExactStableSetAndOptimality) {
  std::size_t cases = 0, max_three = 0;
  for (std::size_t n = 0; n <= 3; ++n) {
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    StablePreferences rows;
    do {
      rows.push_back(perm);
    } while (std::next_permutation(perm.begin(), perm.end()));
    std::size_t profiles = 1;
    for (std::size_t i = 0; i < n; ++i)
      profiles *= rows.size();
    for (std::size_t a = 0; a < profiles; ++a)
      for (std::size_t b = 0; b < profiles; ++b) {
        auto              x = a, y = b;
        StablePreferences p, r;
        for (std::size_t i = 0; i < n; ++i) {
          p.push_back(rows[x % rows.size()]);
          r.push_back(rows[y % rows.size()]);
          x /= rows.size();
          y /= rows.size();
        }
        const auto out = gale_shapley(p, r), opposite = reverse_stable_matching(gale_shapley(r, p));
        ASSERT_TRUE(feasible_oracle(out, n));
        ASSERT_TRUE(blocks_oracle(out, p, r).empty());
        ASSERT_TRUE(blocks_oracle(opposite, p, r).empty());
        auto                                  permutation = rows.front();
        std::vector<std::vector<std::size_t>> expected;
        do {
          const auto m      = candidate(permutation);
          const bool stable = blocks_oracle(m, p, r).empty();
          ASSERT_EQ(is_stable(m, p, r), stable);
          if (stable) {
            expected.push_back(permutation);
            for (std::size_t i = 0; i < n; ++i) {
              ASSERT_LE(rank_of(p[i], out.proposer_match[i]), rank_of(p[i], m.proposer_match[i]));
              ASSERT_GE(rank_of(r[i], out.reviewer_match[i]), rank_of(r[i], m.reviewer_match[i]));
              ASSERT_LE(
                  rank_of(r[i], opposite.reviewer_match[i]), rank_of(r[i], m.reviewer_match[i]));
            }
          }
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        const auto enumeration = enumerate_stable_matchings(p, r);
        ASSERT_EQ(enumeration.candidates_tested, stable_matching_permutations(n));
        ASSERT_EQ(enumeration.matchings.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
          ASSERT_EQ(enumeration.matchings[i].proposer_match, expected[i]);
          ASSERT_EQ(enumeration.matchings[i].proposals, 0u);
        }
        if (n <= 2)
          certificate(gale_shapley_trace(p, r));
        if (n == 3)
          max_three = std::max(max_three, out.proposals);
        ++cases;
      }
  }
  EXPECT_EQ(cases, 46674u);
  EXPECT_EQ(max_three, 7u);
}
TEST(StableMatchingChecked, ActualFifoCertificatesForRandomLargerProfiles) {
  std::mt19937_64 rng(125);
  for (int trial = 0; trial < 300; ++trial) {
    const auto n = 3 + rng() % 10;
    const auto p = random_prefs(n, rng);
    const auto r = random_prefs(n, rng);
    certificate(gale_shapley_trace(p, r));
  }
}
TEST(StableMatchingChecked, TinyReplacementAndWholeEntryCount) {
  const StablePreferences p{{0, 1}, {0, 1}}, r{{1, 0}, {0, 1}};
  const auto              t = gale_shapley_trace(p, r, 51);
  certificate(t);
  ASSERT_EQ(t.events.size(), 3u);
  EXPECT_EQ(t.result.proposer_match, (std::vector<std::size_t>{1, 0}));
  EXPECT_EQ(t.captured_entries, 45u);
  EXPECT_EQ(t.events[0].queue_before, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(t.events[1].decision, "replace");
  EXPECT_EQ(t.events[1].held, 0u);
  EXPECT_EQ(t.events[1].newly_free, 0u);
  EXPECT_EQ(t.events[1].queue_after, (std::vector<std::size_t>{0}));
  EXPECT_EQ(t.work.accepted_free, 2u);
  EXPECT_EQ(t.work.replacements, 1u);
  EXPECT_EQ(t.work.rejections, 0u);
}
TEST(StableMatchingChecked, AllSmallMalformedAndPartialMapsAreSafelyClassified) {
  const StablePreferences          p{{0, 1}, {0, 1}};
  const std::array<std::size_t, 4> ids{0, 1, kUnmatched, 2};
  for (auto a : ids)
    for (auto b : ids)
      for (auto c : ids)
        for (auto d : ids) {
          StableMatchingResult m{{a, b}, {c, d}, kUnmatched};
          const bool           feasible = feasible_oracle(m, 2);
          const auto           diag     = stable_matching_diagnostics(m, p, p);
          ASSERT_EQ(diag.feasible, feasible);
          const auto expected = feasible ? blocks_oracle(m, p, p)
                                         : std::vector<std::pair<std::size_t, std::size_t>>{};
          EXPECT_EQ(is_stable(m, p, p), feasible && expected.empty());
          EXPECT_EQ(diag.stable, feasible && expected.empty());
          EXPECT_EQ(diag.complete, feasible && a != kUnmatched && b != kUnmatched);
          ASSERT_EQ(diag.blocking_pairs.size(), expected.size());
          for (std::size_t k = 0; k < expected.size(); ++k) {
            const auto& w = diag.blocking_pairs[k];
            EXPECT_EQ(w.proposer, expected[k].first);
            EXPECT_EQ(w.reviewer, expected[k].second);
            EXPECT_LT(w.proposer_rank, w.proposer_partner_rank);
            EXPECT_LT(w.reviewer_rank, w.reviewer_partner_rank);
          }
          EXPECT_EQ(diag.pairs_examined, feasible ? 4u : 0u);
        }
  EXPECT_FALSE(is_stable({{0, 0}, {0, 0}, 0}, p, p));
  EXPECT_FALSE(is_stable({}, {{0}}, {{0}}));
  EXPECT_FALSE(is_stable({{1}, {0}, 0}, {{0}}, {{0}}));
  EXPECT_TRUE(is_stable({{0, 1}, {0, 1}, kUnmatched}, p, p));
  const auto partial = stable_matching_diagnostics({{kUnmatched, 1}, {kUnmatched, 1}, 0}, p, p);
  EXPECT_TRUE(partial.feasible);
  EXPECT_FALSE(partial.complete);
  EXPECT_FALSE(partial.stable);
  ASSERT_FALSE(partial.blocking_pairs.empty());
  EXPECT_EQ(partial.blocking_pairs[0].proposer_partner_rank, 3u);
}
TEST(StableMatchingChecked, OrientationUsesTheInverseMapAndPreservesCostMetadata) {
  const StablePreferences p{{2, 1, 0}, {2, 0, 1}, {0, 1, 2}}, r{{2, 0, 1}, {0, 1, 2}, {1, 0, 2}};
  const auto              a = gale_shapley(p, r), backward = gale_shapley(r, p);
  const auto              b = reverse_stable_matching(backward);
  EXPECT_EQ(a.proposer_match, (std::vector<std::size_t>{1, 2, 0}));
  EXPECT_EQ(backward.proposer_match, (std::vector<std::size_t>{2, 0, 1}));
  EXPECT_EQ(a.proposer_match, b.proposer_match);
  EXPECT_GT(rank_of(p[0], a.proposer_match[0]), rank_of(p[0], backward.proposer_match[0]));
  EXPECT_LE(rank_of(p[0], a.proposer_match[0]), rank_of(p[0], b.proposer_match[0]));
  EXPECT_EQ(reverse_stable_matching(b).proposer_match, backward.proposer_match);
  EXPECT_EQ(b.proposals, backward.proposals);
  EXPECT_THROW(reverse_stable_matching({{0, 0}, {0, 0}, 0}), std::invalid_argument);
  const StableMatchingResult partial{{kUnmatched, 0}, {1, kUnmatched}, 99};
  const auto                 inverted = reverse_stable_matching(partial);
  EXPECT_EQ(inverted.proposer_match, partial.reviewer_match);
  EXPECT_EQ(inverted.proposals, 99u);
}
TEST(StableMatchingChecked, PreferenceValidationPrecedesCandidatesAndAdmission) {
  const std::vector<std::pair<StablePreferences, StablePreferences>> bad{
      {{{0, 1}}, {{0}, {1}}},
      {{{0}}, {{0, 1}}},
      {{{0, 0}, {1, 0}}, {{0, 1}, {0, 1}}},
      {{{0, 2}, {1, 0}}, {{0, 1}, {0, 1}}},
      {{{0, 1}, {1, 0}}, {{0, 1}, {kUnmatched, 0}}}};
  for (const auto& [p, r] : bad) {
    EXPECT_THROW(gale_shapley(p, r), std::invalid_argument);
    EXPECT_THROW(gale_shapley_trace(p, r, 1), std::invalid_argument);
    EXPECT_THROW((void)is_stable({}, p, r), std::invalid_argument);
    EXPECT_THROW(stable_matching_diagnostics({}, p, r, 1), std::invalid_argument);
    EXPECT_THROW(enumerate_stable_matchings(p, r, 1), std::invalid_argument);
    EXPECT_THROW(stable_matching_ranks({}, p, r), std::invalid_argument);
  }
  EXPECT_THROW(
      stable_matching_ranks({{kUnmatched}, {kUnmatched}, 0}, {{0}}, {{0}}), std::invalid_argument);
}
TEST(StableMatchingChecked, WholeQuotasOwnershipAndVirtualCountOverflow) {
  StablePreferences p{{0, 1}, {0, 1}}, r{{1, 0}, {0, 1}};
  EXPECT_EQ(gale_shapley_trace_entries(2), 51u);
  EXPECT_EQ(gale_shapley_trace_entries(5), 636u);
  EXPECT_THROW(gale_shapley_trace(p, r, 50), std::length_error);
  EXPECT_THROW(gale_shapley_trace(p, r, 45), std::length_error);
  auto t = gale_shapley_trace(p, r, 51);
  p[0].clear();
  r.clear();
  t.events[0].proposer_match[0] = 99;
  EXPECT_EQ(t.proposer_preferences[0], (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(t.events[1].proposer_match[0], kUnmatched);
  EXPECT_EQ(t.result.proposer_match[0], 1u);
  const auto two = stable_matching_common_preferences(2);
  const auto m   = gale_shapley(two.proposer_preferences, two.reviewer_preferences);
  EXPECT_THROW(
      stable_matching_diagnostics(m, two.proposer_preferences, two.reviewer_preferences, 3),
      std::length_error);
  EXPECT_TRUE(
      stable_matching_diagnostics(m, two.proposer_preferences, two.reviewer_preferences, 4).stable);
  EXPECT_THROW(
      enumerate_stable_matchings(two.proposer_preferences, two.reviewer_preferences, 1),
      std::length_error);
  EXPECT_EQ(
      enumerate_stable_matchings(two.proposer_preferences, two.reviewer_preferences, 2)
          .candidates_tested,
      2u);
  EXPECT_THROW(stable_matching_common_preferences(2, 11), std::length_error);
  EXPECT_NO_THROW(stable_matching_common_preferences(2, 12));
  EXPECT_THROW(stable_matching_mutual_first_preferences(2, 11), std::length_error);
  EXPECT_NO_THROW(stable_matching_mutual_first_preferences(2, 12));
  const auto max = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(gale_shapley_proposal_bound(max), std::length_error);
  EXPECT_THROW(gale_shapley_trace_entries(max), std::length_error);
  EXPECT_THROW(stable_matching_permutations(max), std::length_error);
  EXPECT_THROW(stable_matching_common_preferences(max, 0), std::length_error);
  EXPECT_NO_THROW(gale_shapley_trace(two.proposer_preferences, two.reviewer_preferences, 0));
  EXPECT_NO_THROW(
      stable_matching_diagnostics(m, two.proposer_preferences, two.reviewer_preferences, 0));
  EXPECT_NO_THROW(
      enumerate_stable_matchings(two.proposer_preferences, two.reviewer_preferences, 0));
}
TEST(StableMatchingChecked, EmptyModelAndFactorialBoundsAreExplicit) {
  const auto m = gale_shapley({}, {});
  EXPECT_TRUE(m.proposer_match.empty());
  EXPECT_EQ(m.proposals, 0u);
  EXPECT_TRUE(is_stable(m, {}, {}));
  const auto d = stable_matching_diagnostics(m, {}, {});
  EXPECT_TRUE(d.feasible);
  EXPECT_TRUE(d.complete);
  EXPECT_TRUE(d.stable);
  EXPECT_EQ(d.pairs_examined, 0u);
  const auto ranks = stable_matching_ranks(m, {}, {});
  EXPECT_FALSE(ranks.has_agents);
  EXPECT_EQ(ranks.proposer_mean, 0);
  EXPECT_EQ(ranks.reviewer_mean, 0);
  const auto e = enumerate_stable_matchings({}, {});
  EXPECT_EQ(e.candidates_tested, 1u);
  ASSERT_EQ(e.matchings.size(), 1u);
  EXPECT_TRUE(e.matchings[0].proposer_match.empty());
  EXPECT_EQ(gale_shapley_proposal_bound(0), 0u);
  EXPECT_EQ(gale_shapley_trace_entries(0), 0u);
  EXPECT_EQ(gale_shapley_trace({}, {}).captured_entries, 0u);
  EXPECT_EQ(stable_matching_permutations(0), 1u);
  EXPECT_EQ(stable_matching_permutations(5), 120u);
  EXPECT_EQ(stable_matching_permutations(10), 3628800u);
}
TEST(StableMatchingChecked, DeterministicFamilyWorkCountsIncludeRankPreparation) {
  for (std::size_t n : std::array<std::size_t, 8>{0, 1, 2, 3, 8, 32, 128, 256}) {
    const auto first  = stable_matching_mutual_first_preferences(n),
               common = stable_matching_common_preferences(n);
    const auto a      = gale_shapley_work(first.proposer_preferences, first.reviewer_preferences),
               b      = gale_shapley_work(common.proposer_preferences, common.reviewer_preferences);
    EXPECT_EQ(a.result.proposals, n);
    EXPECT_EQ(b.result.proposals, n * (n + 1) / 2);
    EXPECT_EQ(a.work.rank_entries, n * n);
    EXPECT_EQ(b.work.rank_entries, n * n);
    EXPECT_EQ(a.work.replacements, 0u);
    EXPECT_EQ(a.work.rejections, 0u);
    EXPECT_EQ(b.work.replacements, 0u);
    EXPECT_EQ(b.work.rejections, b.result.proposals - n);
    EXPECT_EQ(a.work.accepted_free, n);
    EXPECT_EQ(b.work.accepted_free, n);
    EXPECT_EQ(b.work.queue_pushes, b.result.proposals);
    EXPECT_EQ(b.work.queue_pops, b.result.proposals);
    EXPECT_EQ(b.work.peak_free, n);
  }
}
TEST(StableMatchingChecked, OriginalFixtureAndPaperThreeStableExample) {
  const auto      profile = stable_matching_example_preferences();
  std::mt19937_64 rng(20240927);
  const auto      generated_p = random_prefs(5, rng);
  const auto      generated_r = random_prefs(5, rng);
  EXPECT_EQ(profile.proposer_preferences, generated_p);
  EXPECT_EQ(profile.reviewer_preferences, generated_r);
  // Explicit emitted rows make numerical assertions independent of std::shuffle mapping.
  const StablePreferences p{
      {3, 2, 1, 4, 0}, {0, 1, 3, 4, 2}, {3, 1, 4, 2, 0}, {0, 3, 2, 4, 1}, {1, 4, 0, 3, 2}};
  const StablePreferences r{
      {4, 0, 2, 3, 1}, {2, 3, 0, 4, 1}, {3, 2, 4, 0, 1}, {2, 3, 1, 4, 0}, {2, 1, 3, 4, 0}};
  const auto t = gale_shapley_trace(p, r);
  certificate(t);
  EXPECT_EQ(t.result.proposals, 9u);
  EXPECT_EQ(t.captured_entries, 275u);
  EXPECT_EQ(t.result.proposer_match, (std::vector<std::size_t>{2, 4, 3, 0, 1}));
  const auto hospital = reverse_stable_matching(gale_shapley(r, p));
  EXPECT_EQ(hospital.proposer_match, (std::vector<std::size_t>{1, 4, 3, 2, 0}));
  const auto a = stable_matching_ranks(t.result, p, r), b = stable_matching_ranks(hospital, p, r);
  EXPECT_DOUBLE_EQ(a.proposer_mean, 1.8);
  EXPECT_DOUBLE_EQ(b.proposer_mean, 2.8);
  EXPECT_EQ(a.proposer_first_choices, 3u);
  EXPECT_EQ(enumerate_stable_matchings(p, r).matchings.size(), 2u);
  const StablePreferences pp{{0, 1, 2}, {1, 2, 0}, {2, 0, 1}}, rr{{1, 2, 0}, {2, 0, 1}, {0, 1, 2}};
  const auto              all = enumerate_stable_matchings(pp, rr);
  ASSERT_EQ(all.matchings.size(), 3u);
  EXPECT_EQ(all.candidates_tested, 6u);
  for (std::size_t k = 0; k < 3; ++k) {
    const auto ranks = stable_matching_ranks(all.matchings[k], pp, rr);
    EXPECT_DOUBLE_EQ(ranks.proposer_mean, k + 1);
    EXPECT_DOUBLE_EQ(ranks.reviewer_mean, 3 - k);
    EXPECT_TRUE(blocks_oracle(all.matchings[k], pp, rr).empty());
  }
}
