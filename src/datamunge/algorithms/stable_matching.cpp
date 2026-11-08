/// @file stable_matching.cpp
/// @brief Shared FIFO deferred acceptance, safe blocking checks and bounded exact enumeration.
#include <algorithm>
#include <datamunge/algorithms/stable_matching.hpp>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private validation and actual matching/checking kernels.
namespace stable_matching_detail {
/// @brief Add checked counts.
/// @param a First count.
/// @param b Second count.
/// @return Sum.
/// @throws std::length_error Overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("stable matching count overflow");
  return a + b;
}
/// @brief Multiply checked counts.
/// @param a First count.
/// @param b Second count.
/// @return Product.
/// @throws std::length_error Overflow.
std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("stable matching count overflow");
  return a * b;
}
/// @brief Validate every shape/range and row permutation before rank construction.
/// @param p Proposer rows.
/// @param r Reviewer rows.
/// @return Common side size.
/// @throws std::invalid_argument Unequal, incomplete or nonpermutation preferences.
/// @throws std::length_error Count/container overflow.
/// @throws std::bad_alloc O(n) duplicate scratch fails.
std::size_t validate(const StablePreferences& p, const StablePreferences& r) {
  const auto n = p.size();
  if (r.size() != n)
    throw std::invalid_argument("gale_shapley: unequal side sizes");
  for (const auto* side : {&p, &r})
    for (const auto& row : *side)
      if (row.size() != n)
        throw std::invalid_argument("gale_shapley: complete rows required");
  for (const auto* side : {&p, &r})
    for (const auto& row : *side)
      for (auto id : row)
        if (id >= n)
          throw std::invalid_argument("gale_shapley: preference ID out of range");
  mul(n, n);
  if (n > std::vector<std::size_t>().max_size() || n > StablePreferences().max_size())
    throw std::length_error("stable matching row capacity");
  std::vector<bool> seen(n);
  for (const auto* side : {&p, &r})
    for (const auto& row : *side) {
      std::fill(seen.begin(), seen.end(), false);
      for (auto id : row) {
        if (seen[id])
          throw std::invalid_argument("gale_shapley: preference row is not a permutation");
        seen[id] = true;
      }
    }
  return n;
}
/// @brief Validate both map ranges before indirect indexing, then their mutual assignments.
/// @param m Candidate maps; cost metadata ignored.
/// @param n Expected side size.
/// @return Partial structural feasibility, including empty maps.
bool feasible(const StableMatchingResult& m, std::size_t n) {
  if (m.proposer_match.size() != n || m.reviewer_match.size() != n)
    return false;
  for (const auto* side : {&m.proposer_match, &m.reviewer_match})
    for (auto id : *side)
      if (id != kUnmatched && id >= n)
        return false;
  for (std::size_t i = 0; i < n; ++i)
    if (m.proposer_match[i] != kUnmatched && m.reviewer_match[m.proposer_match[i]] != i)
      return false;
  for (std::size_t j = 0; j < n; ++j)
    if (m.reviewer_match[j] != kUnmatched && m.proposer_match[m.reviewer_match[j]] != j)
      return false;
  return true;
}
/// @brief Determine completeness after feasibility validation.
/// @param m Feasible mutual maps.
/// @return Every agent matched; true when empty.
bool complete(const StableMatchingResult& m) {
  return std::find(m.proposer_match.begin(), m.proposer_match.end(), kUnmatched) ==
         m.proposer_match.end();
}
/// @brief Build an actual inverse table from validated permutations.
/// @param prefs Validated rows.
/// @param entries Optional actual write counter.
/// @return Owned zero-based ranks.
/// @throws std::bad_alloc Allocation fails.
StablePreferences rank_table(const StablePreferences& prefs, std::size_t* entries = nullptr) {
  const auto        n = prefs.size();
  StablePreferences rank(n, std::vector<std::size_t>(n));
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t pos = 0; pos < n; ++pos) {
      rank[a][prefs[a][pos]] = pos;
      if (entries)
        ++*entries;
    }
  return rank;
}
/// @brief Copy actual FIFO order without mutating the execution queue.
/// @param q Private queue copy.
/// @return Owned queued IDs.
/// @throws std::bad_alloc Allocation fails.
std::vector<std::size_t> queue_values(std::queue<std::size_t> q) {
  std::vector<std::size_t> v;
  v.reserve(q.size());
  while (!q.empty()) {
    v.push_back(q.front());
    q.pop();
  }
  return v;
}
/// @brief Execute one ordered kernel with optional admitted actual history.
/// @param p Proposer rows.
/// @param r Reviewer rows.
/// @param capture Whether snapshots are needed.
/// @param cap Optional whole entry quota.
/// @return Owned actual result/work and optional history.
/// @throws std::invalid_argument Invalid preferences.
/// @throws std::length_error Count/container/quota bound.
/// @throws std::bad_alloc Allocation fails.
GaleShapleyTrace
run(const StablePreferences& p, const StablePreferences& r, bool capture, std::size_t cap) {
  const auto n = validate(p, r);
  gale_shapley_proposal_bound(n);
  if (capture) {
    const auto count = gale_shapley_trace_entries(n);
    if (cap && count > cap)
      throw std::length_error("gale_shapley: trace admission quota");
  }
  GaleShapleyTrace out;
  if (capture) {
    out.proposer_preferences = p;
    out.reviewer_preferences = r;
    out.captured_entries     = mul(3, add(n, mul(n, n)));
  }
  out.reviewer_rank = rank_table(r, &out.work.rank_entries);
  out.result.proposer_match.assign(n, kUnmatched);
  out.result.reviewer_match.assign(n, kUnmatched);
  std::vector<std::size_t> next_choice(n, 0);
  std::queue<std::size_t>  free_proposers;
  for (std::size_t i = 0; i < n; ++i) {
    free_proposers.push(i);
    ++out.work.queue_pushes;
  }
  out.work.peak_free = free_proposers.size();
  while (!free_proposers.empty()) {
    GaleShapleyEvent e;
    if (capture)
      e.queue_before = queue_values(free_proposers);
    const auto i = free_proposers.front();
    free_proposers.pop();
    ++out.work.queue_pops;
    const auto j = p[i][next_choice[i]++];
    ++out.work.next_choice_advances;
    ++out.result.proposals;
    ++out.work.proposals;
    const auto  held       = out.result.reviewer_match[j];
    std::size_t newly_free = kUnmatched;
    const char* decision;
    if (held == kUnmatched) {
      decision = "accept-free";
      ++out.work.accepted_free;
      out.result.reviewer_match[j] = i;
      out.result.proposer_match[i] = j;
    } else if (out.reviewer_rank[j][i] < out.reviewer_rank[j][held]) {
      decision   = "replace";
      newly_free = held;
      ++out.work.replacements;
      out.result.proposer_match[held] = kUnmatched;
      free_proposers.push(held);
      ++out.work.queue_pushes;
      out.result.reviewer_match[j] = i;
      out.result.proposer_match[i] = j;
    } else {
      decision   = "reject";
      newly_free = i;
      ++out.work.rejections;
      free_proposers.push(i);
      ++out.work.queue_pushes;
    }
    out.work.peak_free = std::max(out.work.peak_free, free_proposers.size());
    if (capture) {
      e.number         = out.result.proposals;
      e.proposer       = i;
      e.reviewer       = j;
      e.held           = held;
      e.newly_free     = newly_free;
      e.decision       = decision;
      e.next_choice    = next_choice;
      e.proposer_match = out.result.proposer_match;
      e.reviewer_match = out.result.reviewer_match;
      e.queue_after    = queue_values(free_proposers);
      out.captured_entries =
          add(out.captured_entries,
              add(add(1, mul(3, n)), add(e.queue_before.size(), e.queue_after.size())));
      out.events.push_back(std::move(e));
    }
  }
  return out;
}
/// @brief Scan actual mutually preferred pairs on a feasible candidate.
/// @param m Feasible candidate.
/// @param p_rank Prepared proposer ranks.
/// @param r_rank Prepared reviewer ranks.
/// @param capture Whether to retain witnesses.
/// @param stop_first Whether first witness ends scanning.
/// @return Structural flags, actual pair count and optional witnesses.
/// @throws std::bad_alloc Witness allocation fails.
StabilityDiagnostics scan(
    const StableMatchingResult& m,
    const StablePreferences&    p_rank,
    const StablePreferences&    r_rank,
    bool                        capture,
    bool                        stop_first) {
  const auto           n = p_rank.size();
  StabilityDiagnostics d;
  d.feasible = true;
  d.complete = complete(m);
  d.stable   = true;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      ++d.pairs_examined;
      const auto pi = m.proposer_match[i];
      if (pi != kUnmatched && p_rank[i][j] >= p_rank[i][pi])
        continue;
      const auto pj = m.reviewer_match[j];
      if (pj != kUnmatched && r_rank[j][i] >= r_rank[j][pj])
        continue;
      d.stable = false;
      if (capture)
        d.blocking_pairs.push_back(
            {i,
             j,
             p_rank[i][j] + 1,
             r_rank[j][i] + 1,
             pi == kUnmatched ? n + 1 : p_rank[i][pi] + 1,
             pj == kUnmatched ? n + 1 : r_rank[j][pj] + 1});
      if (stop_first)
        return d;
    }
  return d;
}
/// @brief Admit generated row/cell storage before allocation.
/// @param n Side size.
/// @param cap Optional whole entry quota.
/// @return Nothing.
/// @throws std::length_error Count/container/quota exceeded.
void generator_admit(std::size_t n, std::size_t cap) {
  const auto count = mul(2, add(n, mul(n, n)));
  if (n > StablePreferences().max_size() || n > std::vector<std::size_t>().max_size() ||
      (cap && count > cap))
    throw std::length_error("stable preference generation quota");
}
} // namespace stable_matching_detail
StableMatchingResult gale_shapley(
    const StablePreferences& proposer_preferences,
    const StablePreferences& reviewer_preferences) {
  return stable_matching_detail::run(proposer_preferences, reviewer_preferences, false, 0).result;
}
GaleShapleyRun gale_shapley_work(
    const StablePreferences& proposer_preferences,
    const StablePreferences& reviewer_preferences) {
  auto out = stable_matching_detail::run(proposer_preferences, reviewer_preferences, false, 0);
  return std::move(static_cast<GaleShapleyRun&>(out));
}
std::size_t gale_shapley_proposal_bound(std::size_t n) {
  return n ? stable_matching_detail::add(stable_matching_detail::mul(n, n - 1), 1) : 0;
}
std::size_t gale_shapley_trace_entries(std::size_t n) {
  using stable_matching_detail::add;
  using stable_matching_detail::mul;
  const auto proposals = gale_shapley_proposal_bound(n);
  if (n > StablePreferences().max_size() || n > std::vector<std::size_t>().max_size() ||
      proposals > std::vector<GaleShapleyEvent>().max_size())
    throw std::length_error("gale_shapley trace vector capacity");
  return add(mul(3, add(n, mul(n, n))), mul(proposals, add(1, mul(5, n))));
}
GaleShapleyTrace gale_shapley_trace(
    const StablePreferences& proposer_preferences,
    const StablePreferences& reviewer_preferences,
    std::size_t              max_entries) {
  return stable_matching_detail::run(proposer_preferences, reviewer_preferences, true, max_entries);
}
bool is_stable(
    const StableMatchingResult& matching,
    const StablePreferences&    proposer_preferences,
    const StablePreferences&    reviewer_preferences) {
  const auto n = stable_matching_detail::validate(proposer_preferences, reviewer_preferences);
  if (!stable_matching_detail::feasible(matching, n))
    return false;
  const auto p_rank = stable_matching_detail::rank_table(proposer_preferences),
             r_rank = stable_matching_detail::rank_table(reviewer_preferences);
  return stable_matching_detail::scan(matching, p_rank, r_rank, false, true).stable;
}
StabilityDiagnostics stable_matching_diagnostics(
    const StableMatchingResult& matching,
    const StablePreferences&    proposer_preferences,
    const StablePreferences&    reviewer_preferences,
    std::size_t                 max_pairs) {
  const auto n = stable_matching_detail::validate(proposer_preferences, reviewer_preferences);
  if (!stable_matching_detail::feasible(matching, n))
    return {};
  const auto count = stable_matching_detail::mul(n, n);
  if (count > std::vector<StableBlockingPair>().max_size() || (max_pairs && count > max_pairs))
    throw std::length_error("stable blocking pair admission quota");
  const auto p_rank = stable_matching_detail::rank_table(proposer_preferences),
             r_rank = stable_matching_detail::rank_table(reviewer_preferences);
  return stable_matching_detail::scan(matching, p_rank, r_rank, true, false);
}
StableMatchingResult reverse_stable_matching(const StableMatchingResult& matching) {
  if (!stable_matching_detail::feasible(matching, matching.proposer_match.size()))
    throw std::invalid_argument("cannot reverse malformed matching");
  return {matching.reviewer_match, matching.proposer_match, matching.proposals};
}
StableMatchingRanks stable_matching_ranks(
    const StableMatchingResult& matching,
    const StablePreferences&    proposer_preferences,
    const StablePreferences&    reviewer_preferences) {
  const auto n = stable_matching_detail::validate(proposer_preferences, reviewer_preferences);
  if (!stable_matching_detail::feasible(matching, n) || !stable_matching_detail::complete(matching))
    throw std::invalid_argument("rank summary requires perfect mutual maps");
  const auto p_rank = stable_matching_detail::rank_table(proposer_preferences),
             r_rank = stable_matching_detail::rank_table(reviewer_preferences);
  StableMatchingRanks out;
  out.has_agents = n > 0;
  out.proposer_ranks.resize(n);
  out.reviewer_ranks.resize(n);
  std::size_t p_sum = 0, r_sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    out.proposer_ranks[i] = p_rank[i][matching.proposer_match[i]] + 1;
    out.reviewer_ranks[i] = r_rank[i][matching.reviewer_match[i]] + 1;
    p_sum                 = stable_matching_detail::add(p_sum, out.proposer_ranks[i]);
    r_sum                 = stable_matching_detail::add(r_sum, out.reviewer_ranks[i]);
    out.proposer_first_choices += out.proposer_ranks[i] == 1;
    out.reviewer_first_choices += out.reviewer_ranks[i] == 1;
  }
  if (n) {
    out.proposer_mean = static_cast<double>(p_sum) / static_cast<double>(n);
    out.reviewer_mean = static_cast<double>(r_sum) / static_cast<double>(n);
  }
  return out;
}
std::size_t stable_matching_permutations(std::size_t n) {
  std::size_t count = 1;
  for (std::size_t i = 2; i <= n; ++i)
    count = stable_matching_detail::mul(count, i);
  return count;
}
StableMatchingEnumeration enumerate_stable_matchings(
    const StablePreferences& proposer_preferences,
    const StablePreferences& reviewer_preferences,
    std::size_t              max_permutations) {
  const auto n     = stable_matching_detail::validate(proposer_preferences, reviewer_preferences),
             count = stable_matching_permutations(n);
  stable_matching_detail::mul(count, stable_matching_detail::mul(n, n));
  stable_matching_detail::mul(
      count, stable_matching_detail::add(1, stable_matching_detail::mul(2, n)));
  if (count > std::vector<StableMatchingResult>().max_size() ||
      (max_permutations && count > max_permutations))
    throw std::length_error("stable matching enumeration admission quota");
  const auto p_rank = stable_matching_detail::rank_table(proposer_preferences),
             r_rank = stable_matching_detail::rank_table(reviewer_preferences);
  StableMatchingEnumeration out;
  std::vector<std::size_t>  permutation(n);
  std::iota(permutation.begin(), permutation.end(), 0);
  do {
    StableMatchingResult candidate;
    candidate.proposer_match = permutation;
    candidate.reviewer_match.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      candidate.reviewer_match[permutation[i]] = i;
    const auto d = stable_matching_detail::scan(candidate, p_rank, r_rank, false, true);
    ++out.candidates_tested;
    out.pairs_examined = stable_matching_detail::add(out.pairs_examined, d.pairs_examined);
    if (d.stable)
      out.matchings.push_back(std::move(candidate));
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return out;
}
StableMatchingProfile stable_matching_common_preferences(std::size_t n, std::size_t max_entries) {
  stable_matching_detail::generator_admit(n, max_entries);
  std::vector<std::size_t> row(n);
  std::iota(row.begin(), row.end(), 0);
  StableMatchingProfile out;
  out.proposer_preferences.assign(n, row);
  out.reviewer_preferences = out.proposer_preferences;
  return out;
}
StableMatchingProfile stable_matching_mutual_first_preferences(
    std::size_t n,
    std::size_t max_entries) {
  stable_matching_detail::generator_admit(n, max_entries);
  StableMatchingProfile out;
  out.proposer_preferences.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto& row = out.proposer_preferences[i];
    row.reserve(n);
    row.push_back(i);
    for (std::size_t j = 0; j < n; ++j)
      if (j != i)
        row.push_back(j);
  }
  out.reviewer_preferences = out.proposer_preferences;
  return out;
}
StableMatchingProfile stable_matching_example_preferences() {
  auto            out = stable_matching_common_preferences(5);
  std::mt19937_64 rng(20240927);
  for (auto* side : {&out.proposer_preferences, &out.reviewer_preferences})
    for (auto& row : *side)
      std::shuffle(row.begin(), row.end(), rng);
  return out;
}
} // namespace datamunge::algorithms
