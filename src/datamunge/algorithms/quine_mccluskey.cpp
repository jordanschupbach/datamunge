/// @file quine_mccluskey.cpp
/// @brief Shared checked exact Boolean minimization and actual native round capture.
#include <algorithm>
#include <bit>
#include <datamunge/algorithms/quine_mccluskey.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical Boolean minimization implementation.
namespace datamunge::algorithms {
/// @brief Private domain and selected-search helpers.
namespace {
/// @brief Validate variable count and construct the exact representable domain mask.
/// @param n Variable count0..32.
/// @return Low n bits set, with no shift by32.
/// @throws std::invalid_argument Count outside0..32.
std::uint32_t domain(int n) {
  if (n < 0 || n > 32)
    throw std::invalid_argument("quine_mccluskey: variables outside0..32");
  return n == 32 ? 0xffffffffU : (std::uint32_t{1} << n) - 1;
}
/// @brief Admit a complete term vector length.
/// @param count Actual length, before optional deduplication.
/// @param o Term allocation policy.
/// @throws std::length_error Term policy or signed chart-id range exceeded.
void terms(std::size_t count, const QuineMcCluskeyOptions& o) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (o.max_terms && count > o.max_terms))
    throw std::length_error("quine_mccluskey: term policy");
}
/// @brief Normalize implicant representations by deterministic bit/mask order.
/// @param values Mutable implicants to sort and deduplicate.
void unique(std::vector<Implicant>& values) {
  std::sort(values.begin(), values.end(), [](const Implicant& a, const Implicant& b) {
    return a.bits < b.bits || (a.bits == b.bits && a.mask < b.mask);
  });
  values.erase(std::unique(values.begin(), values.end()), values.end());
}
/// @brief Execute one shared checked minimization with optional actual round capture.
/// @param n Variable count.
/// @param minterms Owned required values.
/// @param dontcares Owned optional values.
/// @param o Independent execution policies.
/// @param capture Retain actual merge rounds.
/// @return Exact minimized expression and actual chart/work.
QuineMcCluskeyRun solve(
    int                          n,
    std::vector<std::uint32_t>   minterms,
    std::vector<std::uint32_t>   dontcares,
    const QuineMcCluskeyOptions& o,
    bool                         capture) {
  const auto mask = domain(n);
  if (o.max_variables && static_cast<std::size_t>(n) > o.max_variables)
    throw std::length_error("quine_mccluskey: variable policy");
  if (dontcares.size() > std::numeric_limits<std::size_t>::max() - minterms.size())
    throw std::length_error("quine_mccluskey: input capacity");
  terms(minterms.size() + dontcares.size(), o);
  for (const auto m : minterms)
    if (m & ~mask)
      throw std::invalid_argument("quine_mccluskey: required minterm outside domain");
  for (const auto m : dontcares)
    if (m & ~mask)
      throw std::invalid_argument("quine_mccluskey: optional minterm outside domain");
  std::sort(minterms.begin(), minterms.end());
  minterms.erase(std::unique(minterms.begin(), minterms.end()), minterms.end());
  std::sort(dontcares.begin(), dontcares.end());
  dontcares.erase(std::unique(dontcares.begin(), dontcares.end()), dontcares.end());
  QuineMcCluskeyRun run;
  run.minterms                  = std::move(minterms);
  run.dontcares                 = std::move(dontcares);
  auto&                  result = run.result;
  std::vector<Implicant> current;
  for (auto m : run.minterms)
    current.push_back({m, 0});
  for (auto m : run.dontcares)
    current.push_back({m, 0});
  unique(current);
  std::size_t history = 0;
  while (!current.empty()) {
    std::vector<bool>      used(current.size(), false);
    std::vector<Implicant> next, added;
    for (std::size_t i = 0; i < current.size(); ++i)
      for (std::size_t j = i + 1; j < current.size(); ++j) {
        if (run.pair_tests == std::numeric_limits<std::size_t>::max() ||
            (o.max_pair_tests && run.pair_tests >= o.max_pair_tests))
          throw std::length_error("quine_mccluskey: pair policy");
        ++run.pair_tests;
        if (current[i].mask != current[j].mask)
          continue;
        const auto diff = current[i].bits ^ current[j].bits;
        if (diff && (diff & (diff - 1)) == 0) {
          terms(next.size() + 1, o);
          used[i] = used[j] = true;
          next.push_back({current[i].bits & ~diff, current[i].mask | diff});
          ++run.merges;
        }
      }
    for (std::size_t i = 0; i < current.size(); ++i)
      if (!used[i]) {
        terms(result.prime_implicants.size() + 1, o);
        result.prime_implicants.push_back(current[i]);
        if (capture)
          added.push_back(current[i]);
      }
    unique(next);
    if (capture) {
      const auto limit = std::numeric_limits<std::size_t>::max();
      if (current.size() > (limit - 1) / 3 || next.size() > (limit - 1 - 3 * current.size()) / 2 ||
          added.size() > (limit - 1 - 3 * current.size() - 2 * next.size()) / 2)
        throw std::length_error("quine_mccluskey: history capacity");
      const auto cost = 1 + 3 * current.size() + 2 * next.size() + 2 * added.size();
      if (cost > limit - history || (o.max_trace_entries && (history > o.max_trace_entries ||
                                                             cost > o.max_trace_entries - history)))
        throw std::length_error("quine_mccluskey: history policy");
      history += cost;
      run.rounds.push_back({current, used, next, added});
    }
    current = std::move(next);
  }
  unique(result.prime_implicants);
  const auto& primes = result.prime_implicants;
  run.chart.resize(run.minterms.size());
  std::size_t chart_entries = 0;
  for (std::size_t mi = 0; mi < run.minterms.size(); ++mi)
    for (std::size_t pi = 0; pi < primes.size(); ++pi)
      if (primes[pi].covers(run.minterms[mi])) {
        if (chart_entries == std::numeric_limits<std::size_t>::max() ||
            (o.max_chart_entries && chart_entries >= o.max_chart_entries))
          throw std::length_error("quine_mccluskey: chart policy");
        ++chart_entries;
        run.chart[mi].push_back(static_cast<int>(pi));
      }
  std::vector<bool> essential(primes.size(), false);
  for (const auto& row : run.chart)
    if (row.size() == 1)
      essential[row[0]] = true;
  std::vector<int> chosen;
  for (std::size_t i = 0; i < primes.size(); ++i)
    if (essential[i]) {
      result.essential.push_back(primes[i]);
      chosen.push_back(static_cast<int>(i));
    }
  std::vector<std::vector<int>> remaining;
  for (const auto& row : run.chart) {
    bool covered = false;
    for (auto pi : row)
      if (essential[pi]) {
        covered = true;
        break;
      }
    if (!covered)
      remaining.push_back(row);
  }
  if (!remaining.empty()) {
    run.covering = petrick_trace(remaining, o.covering);
    if (run.covering.result.minimal_covers.empty())
      throw std::runtime_error("quine_mccluskey: impossible native coverage chart");
    const std::vector<int>* best          = nullptr;
    std::size_t             best_literals = std::numeric_limits<std::size_t>::max();
    for (const auto& cover : run.covering.result.minimal_covers) {
      std::size_t literals = 0;
      for (int pi : cover)
        literals += static_cast<std::size_t>(n - std::popcount(primes[pi].mask));
      if (literals < best_literals) {
        best_literals = literals;
        best          = &cover;
      }
    }
    chosen.insert(chosen.end(), best->begin(), best->end());
  }
  std::sort(chosen.begin(), chosen.end());
  chosen.erase(std::unique(chosen.begin(), chosen.end()), chosen.end());
  for (auto i : chosen)
    result.minimal_cover.push_back(primes[i]);
  return run;
}
} // namespace
std::string implicant_to_string(const Implicant& imp, int n) {
  const auto mask = domain(n);
  if ((imp.bits | imp.mask) & ~mask)
    throw std::invalid_argument("implicant_to_string: bits outside domain");
  std::string s;
  for (int i = n - 1; i >= 0; --i)
    s += imp.mask & (std::uint32_t{1} << i) ? '-' : (imp.bits >> i) & 1 ? '1' : '0';
  return s;
}
QuineMcCluskeyRun quine_mccluskey_run(
    int                          n,
    std::vector<std::uint32_t>   m,
    std::vector<std::uint32_t>   d,
    const QuineMcCluskeyOptions& o) {
  return solve(n, std::move(m), std::move(d), o, false);
}
QuineMcCluskeyRun quine_mccluskey_trace(
    int                          n,
    std::vector<std::uint32_t>   m,
    std::vector<std::uint32_t>   d,
    const QuineMcCluskeyOptions& o) {
  return solve(n, std::move(m), std::move(d), o, true);
}
QuineMcCluskeyResult
quine_mccluskey(int n, std::vector<std::uint32_t> m, std::vector<std::uint32_t> d) {
  return quine_mccluskey_run(n, std::move(m), std::move(d)).result;
}
} // namespace datamunge::algorithms
