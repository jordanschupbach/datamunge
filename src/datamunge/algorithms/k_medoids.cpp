/// @file k_medoids.cpp
/// @brief Shared checked random-start full best-swap producer and optional actual observations.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/k_medoids.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Local checked input, norm, objective and producing scan helpers.
namespace medoids_detail {
/// @brief Admit a logical count increment without wrap.
/// @param count Actual mutable count.
/// @param amount Additional logical count.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& count, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("medoids count overflow");
  count += amount;
}
/// @brief Admit complete finite rectangular data and optional full matrix quota before copies.
/// @param data Borrowed original points.
/// @param k Requested distinct identities.
/// @param cells Optional n*n cell ceiling, zero off.
/// @throws std::invalid_argument On k/shape/finite violations.
/// @throws std::length_error On matrix geometry or enabled cell ceiling.
void validate(const std::vector<std::vector<double>>& data, std::size_t k, std::size_t cells) {
  const auto n = data.size();
  if (!k || k > n)
    throw std::invalid_argument("medoids k");
  const auto dim = data.front().size();
  if (!dim)
    throw std::invalid_argument("medoids zero dimension");
  for (const auto& row : data) {
    if (row.size() != dim)
      throw std::invalid_argument("medoids ragged");
    for (double x : row)
      if (!std::isfinite(x))
        throw std::invalid_argument("medoids nonfinite");
  }
  if (n > std::numeric_limits<std::size_t>::max() / n)
    throw std::length_error("medoids matrix overflow");
  if (cells && n * n > cells)
    throw std::length_error("medoids matrix quota");
  if (n * n > std::vector<double>().max_size())
    throw std::length_error("medoids matrix size");
}
/// @brief Admit retained logical records before the operation they observe.
/// @param trace Optional producing capture.
/// @param amount Row/candidate/state records.
/// @param cap Optional total record ceiling, zero off.
/// @throws std::length_error On count overflow or enabled capture ceiling.
void admit(KMedoidsTrace* trace, std::size_t amount, std::size_t cap) {
  if (!trace)
    return;
  auto next = trace->captured_entries;
  add(next, amount);
  if (cap && next > cap)
    throw std::length_error("medoids trace quota");
  trace->captured_entries = next;
}
/// @brief Compute a finite representable Euclidean double norm with widened hypot accumulation.
/// @param a First matched finite coordinate row.
/// @param b Second matched finite coordinate row.
/// @return Rounded finite double Euclidean distance.
/// @throws std::overflow_error On unrepresentable double distance.
double distance(const std::vector<double>& a, const std::vector<double>& b) {
  long double value = 0;
  for (std::size_t d = 0; d < a.size(); ++d)
    value = std::hypot(value, static_cast<long double>(a[d]) - b[d]);
  if (!std::isfinite(value) || value > std::numeric_limits<double>::max())
    throw std::overflow_error("medoids distance");
  return static_cast<double>(value);
}
/// @brief Evaluate one actual full assignment with first-slot ties and sequential objective.
/// @param D Complete checked row-major matrix.
/// @param n Original row count.
/// @param medoids Actual proposed/retained medoid identities.
/// @param assignment Mutable full output slots.
/// @param work Actual producing counters.
/// @return Actual finite sequential double objective.
/// @throws std::length_error On work-count overflow.
/// @throws std::overflow_error On nonfinite objective sum.
double assign(
    const std::vector<double>&      D,
    std::size_t                     n,
    const std::vector<std::size_t>& medoids,
    std::vector<std::size_t>&       assignment,
    KMedoidsWork&                   work) {
  add(work.assignment_passes, 1);
  double cost = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t best = 0;
    double      bd   = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < medoids.size(); ++c) {
      add(work.distance_lookups, 1);
      if (D[i * n + medoids[c]] < bd) {
        bd   = D[i * n + medoids[c]];
        best = c;
      }
    }
    assignment[i] = best;
    cost += bd;
    if (!std::isfinite(cost))
      throw std::overflow_error("medoids objective");
  }
  return cost;
}
/// @brief Produce one shared fit and optionally append actual observations in execution order.
/// @param data Validated immutable rectangular points.
/// @param k Validated medoid count.
/// @param limit Caller complete scan ceiling.
/// @param seed Actual shuffle engine seed.
/// @param trace Optional actual capture, otherwise no input/state/history copies.
/// @param cap Optional retained-record ceiling.
/// @return Owned actual final result.
/// @throws std::length_error On capture/count limits.
/// @throws std::overflow_error On nonfinite double distances/objectives.
/// @throws std::bad_alloc On native/capture allocation failure.
KMedoidsResult run(
    const std::vector<std::vector<double>>& data,
    std::size_t                             k,
    std::size_t                             limit,
    std::uint64_t                           seed,
    KMedoidsTrace*                          trace,
    std::size_t                             cap) {
  const auto   n = data.size();
  KMedoidsWork work;
  admit(trace, n, cap);
  admit(trace, 1, cap);
  if (trace)
    trace->data = data;
  std::vector<double> D(n * n, 0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      D[i * n + j] = D[j * n + i] = distance(data[i], data[j]);
      add(work.distance_pairs, 1);
    }
  std::mt19937_64          rng(seed);
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), rng);
  std::vector<std::size_t> medoids(idx.begin(), idx.begin() + k), assignment(n);
  std::vector<char>        is_medoid(n);
  for (auto m : medoids)
    is_medoid[m] = 1;
  double cost = assign(D, n, medoids, assignment, work);
  if (trace)
    trace->states.push_back({medoids, assignment, cost});
  KMedoidsResult result;
  for (std::size_t iteration = 0; iteration < limit; ++iteration) {
    double      best_cost = cost;
    std::size_t best_m = 0, best_o = 0;
    bool        improved = false;
    for (std::size_t mi = 0; mi < k; ++mi)
      for (std::size_t o = 0; o < n; ++o) {
        if (is_medoid[o])
          continue;
        admit(trace, 1, cap);
        std::vector<std::size_t> trial = medoids, tmp(n);
        trial[mi]                      = o;
        const double tc                = assign(D, n, trial, tmp, work);
        add(work.candidates, 1);
        const bool changed = tc < best_cost - 1e-12;
        if (trace)
          trace->candidates.push_back({iteration, mi, o, tc, changed});
        if (changed) {
          best_cost = tc;
          best_m    = mi;
          best_o    = o;
          improved  = true;
        }
      }
    if (!improved) {
      if (trace)
        trace->stop = KMedoidsStop::NoImprovement;
      break;
    }
    admit(trace, 1, cap);
    is_medoid[medoids[best_m]] = 0;
    medoids[best_m]            = best_o;
    is_medoid[best_o]          = 1;
    cost                       = assign(D, n, medoids, assignment, work);
    result.iterations          = iteration + 1;
    if (trace)
      trace->states.push_back({medoids, assignment, cost});
  }
  result.medoids    = std::move(medoids);
  result.assignment = std::move(assignment);
  result.total_cost = cost;
  if (trace)
    trace->work = work;
  return result;
}
} // namespace medoids_detail
KMedoidsResult k_medoids(
    const std::vector<std::vector<double>>& data,
    std::size_t                             k,
    std::size_t                             max_iterations,
    std::uint64_t                           seed) {
  medoids_detail::validate(data, k, 0);
  return medoids_detail::run(data, k, max_iterations, seed, nullptr, 0);
}
KMedoidsTrace k_medoids_trace(
    const std::vector<std::vector<double>>& data,
    std::size_t                             k,
    std::size_t                             max_iterations,
    std::uint64_t                           seed,
    std::size_t                             max_entries,
    std::size_t                             max_distance_cells) {
  medoids_detail::validate(data, k, max_distance_cells);
  KMedoidsTrace out;
  out.result = medoids_detail::run(data, k, max_iterations, seed, &out, max_entries);
  return out;
}
} // namespace datamunge::algorithms
