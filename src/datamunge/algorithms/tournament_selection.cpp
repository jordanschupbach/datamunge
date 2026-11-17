/// @file tournament_selection.cpp
/// @brief Actual checked historical modulo draws and first-on-tie winner updates.
#include <cmath>
#include <datamunge/algorithms/selection_rng.hpp>
#include <datamunge/algorithms/tournament_selection.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private per-call sampling producer.
namespace {
/// @brief Validate complete draw geometry, then execute and optionally retain actual contestants.
TournamentRun execute(
    const std::vector<double>& fitness,
    int                        n,
    int                        k,
    std::uint64_t              seed,
    const TournamentOptions&   o,
    bool                       capture) {
  if (n < 0 || k <= 0)
    throw std::invalid_argument("tournament counts");
  for (double f : fitness)
    if (!std::isfinite(f))
      throw std::invalid_argument("tournament finite fitness");
  if (n > 0 && fitness.empty())
    throw std::invalid_argument("tournament empty population");
  auto nn = std::size_t(n), kk = std::size_t(k);
  if ((o.max_population && fitness.size() > o.max_population) ||
      (o.max_picks && nn > o.max_picks) ||
      (nn && kk > std::numeric_limits<std::size_t>::max() / nn))
    throw std::length_error("tournament geometry");
  auto draws = nn * kk;
  if (o.max_contestants && draws > o.max_contestants)
    throw std::length_error("tournament complete draws");
  if (capture && o.max_records && draws > o.max_records)
    throw std::length_error("tournament capture geometry");
  TournamentRun r;
  r.counts.assign(fitness.size(), 0);
  r.contestant_counts.assign(fitness.size(), 0);
  r.indices.reserve(nn);
  detail::SelRng rng(seed);
  for (int s = 0; s < n; ++s) {
    std::size_t best = 0;
    for (int round = 0; round < k; ++round) {
      auto raw       = rng.next();
      auto candidate = static_cast<std::size_t>(raw % fitness.size());
      ++r.random_words;
      ++r.contestant_counts[candidate];
      auto before = round ? best : candidate;
      if (!round)
        best = candidate;
      else {
        ++r.comparisons;
        if (fitness[candidate] > fitness[best])
          best = candidate;
      }
      if (capture)
        r.steps.push_back({std::size_t(s), round, raw, candidate, before, best});
    }
    r.indices.push_back(best);
    ++r.counts[best];
  }
  return r;
}
} // namespace
TournamentRun tournament_run(
    const std::vector<double>& fitness,
    int                        n,
    int                        k,
    std::uint64_t              seed,
    const TournamentOptions&   options) {
  return execute(fitness, n, k, seed, options, false);
}
TournamentRun tournament_trace(
    const std::vector<double>& fitness,
    int                        n,
    int                        k,
    std::uint64_t              seed,
    const TournamentOptions&   options) {
  return execute(fitness, n, k, seed, options, true);
}
std::vector<std::size_t>
tournament_sample(const std::vector<double>& fitness, int n, int k, std::uint64_t seed) {
  return tournament_run(fitness, n, k, seed).indices;
}
} // namespace datamunge::algorithms
