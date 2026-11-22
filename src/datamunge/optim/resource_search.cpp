/// @file
/// @brief Synchronous and logical asynchronous scheduling with additive resource accounting.

#include <datamunge/optim/resource_search.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::optim {
/// @brief Private scheduler state and completion rules.
namespace {

/// @brief Ascending cumulative fidelity ladder ending at the configured maximum.
/// @param options Positive minimum and maximum resource and promotion factor.
/// @return Owned rung fidelities.
std::vector<std::size_t> ladder(const ResourceSearchOptions& options) {
  std::vector<std::size_t> levels{options.minimum_resource};
  while (levels.back() < options.maximum_resource) {
    const auto previous = levels.back();
    levels.push_back(previous > options.maximum_resource / options.reduction
        ? options.maximum_resource : previous * options.reduction);
  }
  return levels;
}

/// @brief Numerical loss ordering with identifier ties.
/// @param a First completed observation.
/// @param b Second completed observation.
/// @return Whether a has better loss, then lower identifier.
bool better(const ResourceObservation& a, const ResourceObservation& b) {
  return a.loss == b.loss ? a.trial < b.trial : a.loss < b.loss;
}

/// @brief Own trial counters and centralize advancement and best-fidelity selection.
struct Schedule {
  const StartTrial& start; ///< Borrowed initializer.
  const AdvanceTrial& advance; ///< Borrowed advancement.
  const ResourceSearchOptions& options; ///< Borrowed immutable controls.
  std::mt19937_64 rng; ///< Local seed generator.
  ResourceSearchResult result; ///< Owned completed observations and work.

  /// @brief Compute unspent resource before any asynchronous reservations.
  /// @return Resource budget less consumed amount.
  std::size_t remaining() const { return options.resource_budget - result.consumed; }

  /// @brief Initialize the next trial with a fresh seed and increment the start count.
  /// @return Fresh trial identifier.
  std::size_t begin() {
    const auto id = result.starts++;
    start(id, rng());
    return id;
  }

  /// @brief Complete a funded score callback and update the best observation.
  /// @param observation Tranche identity, resources and time; loss filled by the callback.
  /// @return Completed observation with finite loss.
  /// @throws std::domain_error Callback returns nonfinite loss.
  ResourceObservation complete(ResourceObservation observation) {
    observation.loss = advance(observation.trial, observation.previous, observation.resource);
    if (!std::isfinite(observation.loss)) throw std::domain_error("Nonfinite resource search loss");
    result.consumed += observation.resource - observation.previous;
    ++result.evaluations;
    if (!result.best || observation.resource > result.best->resource ||
        (observation.resource == result.best->resource && better(observation, *result.best))) {
      result.best = observation;
    }
    if (result.trace.size() < options.trace_limit) result.trace.push_back(observation);
    return observation;
  }
};

/// @brief Score a synchronous bracket, retaining the leading cohort at each rung.
/// @param schedule Shared trial and resource budget.
/// @param rungs Full fidelity ladder.
/// @param start_rung First rung of the bracket.
/// @param desired Maximum fresh starts in this bracket.
/// @param index Bracket identifier.
void run_bracket(Schedule& schedule, const std::vector<std::size_t>& rungs,
                 std::size_t start_rung, std::size_t desired, std::size_t index) {
  std::vector<ResourceObservation> active;
  const auto trials = std::min({desired, schedule.options.trials - schedule.result.starts,
                                schedule.remaining() / rungs[start_rung]});
  for (std::size_t i = 0; i < trials; ++i) {
    active.push_back(schedule.complete({schedule.begin(), 0, rungs[start_rung], index, 0, -1}));
  }
  for (std::size_t rung = start_rung + 1; rung < rungs.size() && !active.empty(); ++rung) {
    const auto extra = rungs[rung] - rungs[rung - 1];
    const auto promoted = std::min(std::max(std::size_t{1}, active.size() / schedule.options.reduction),
                                   schedule.remaining() / extra);
    std::stable_sort(active.begin(), active.end(), better);
    active.resize(promoted);
    for (auto& observation : active) {
      observation = schedule.complete({observation.trial, rungs[rung - 1], rungs[rung], index, 0, -1});
    }
  }
}

/// @brief Run an asynchronous schedule on logical workers.
/// @param schedule Shared callback and budget state.
/// @param rungs Cumulative fidelity ladder.
/// @param duration Optional virtual time callback.
/// @throws std::domain_error A duration cannot advance finite virtual time.
void run_asha(Schedule& schedule, const std::vector<std::size_t>& rungs, const TrialDuration& duration) {
  /// @brief Relative resource and completion of one running work tranche.
  struct Job {
    ResourceObservation observation; ///< Trial and target resource.
    std::size_t rung; ///< Destination rung index.
  };
  std::vector<std::optional<Job>> workers(std::min({schedule.options.workers, schedule.options.trials,
                                                    schedule.remaining() / rungs.front()}));
  std::vector<std::vector<ResourceObservation>> completed(rungs.size());
  std::vector<std::vector<std::size_t>> promoted(rungs.size() - 1);
  std::size_t reserved = 0;
  double now = 0;
  /// Reserve a funded job and assign its completion timestamp.
  const auto enqueue = [&](std::optional<Job>& worker, ResourceObservation observation, std::size_t rung) {
    const auto extra = observation.resource - observation.previous;
    const auto elapsed = duration ? duration(observation.trial, extra) : static_cast<double>(extra);
    observation.completion = now + elapsed;
    if (!(elapsed > 0 && observation.completion > now && std::isfinite(observation.completion))) {
      throw std::domain_error("Invalid virtual ASHA duration");
    }
    worker = Job{observation, rung};
    reserved += extra;
  };
  /// Promote an eligible completed trial or allocate a fresh initial tranche.
  const auto dispatch = [&](std::optional<Job>& worker) {
    const auto available = schedule.remaining() - reserved;
    for (std::size_t from = rungs.size() - 1; from != 0; --from) {
      const auto source = from - 1;
      if (rungs[from] - rungs[source] > available) continue;
      auto ranked = completed[source];
      std::stable_sort(ranked.begin(), ranked.end(), better);
      const auto count = ranked.size() / schedule.options.reduction;
      if (promoted[source].size() >= count) continue;
      for (std::size_t i = 0; i < count; ++i) {
        const auto id = ranked[i].trial;
        if (std::ranges::find(promoted[source], id) != promoted[source].end()) continue;
        promoted[source].push_back(id);
        enqueue(worker, {id, rungs[source], rungs[from], 0, 0, 0}, from);
        return;
      }
    }
    if (rungs.front() <= available && schedule.result.starts < schedule.options.trials) {
      enqueue(worker, {schedule.begin(), 0, rungs.front(), 0, 0, 0}, 0);
    }
  };

  for (;;) {
    for (auto& worker : workers) {
      if (!worker) dispatch(worker);
    }
    auto next = workers.end();
    for (auto it = workers.begin(); it != workers.end(); ++it) {
      if (*it && (next == workers.end() || (*it)->observation.completion < (*next)->observation.completion)) next = it;
    }
    if (next == workers.end()) break;
    const auto job = **next;
    now = job.observation.completion;
    reserved -= job.observation.resource - job.observation.previous;
    auto observed = schedule.complete(job.observation);
    if (job.rung + 1 < rungs.size()) completed[job.rung].push_back(observed);
    next->reset();
  }
}

} // namespace

ResourceSearchResult resource_search(const StartTrial& start, const AdvanceTrial& advance,
                                     const ResourceSearchOptions& options, const TrialDuration& duration) {
  const auto method = options.method;
  if (!start || !advance || options.resource_budget == 0 || options.trials == 0 ||
      options.minimum_resource == 0 || options.maximum_resource < options.minimum_resource ||
      options.reduction < 2 || (method == ResourceSearchMethod::asha && options.workers == 0) ||
      (method != ResourceSearchMethod::asha && method != ResourceSearchMethod::hyperband &&
       method != ResourceSearchMethod::successive_halving)) {
    throw std::invalid_argument("Invalid resource search callbacks or controls");
  }
  Schedule schedule{start, advance, options, std::mt19937_64(options.seed), {}};
  const auto rungs = ladder(options);
  if (method == ResourceSearchMethod::asha) {
    run_asha(schedule, rungs, duration);
  } else if (method == ResourceSearchMethod::successive_halving) {
    run_bracket(schedule, rungs, 0, options.trials, 0);
  } else {
    for (std::size_t index = 0; index < rungs.size() && schedule.result.starts < options.trials; ++index) {
      const auto bracket = rungs.size() - 1 - index;
      const auto available = options.trials - schedule.result.starts;
      const auto planned = std::ceil(static_cast<long double>(rungs.size()) /
                                     (bracket + 1) * std::pow(static_cast<long double>(options.reduction), bracket));
      const auto count = planned >= available ? available : static_cast<std::size_t>(planned);
      run_bracket(schedule, rungs, index, count, bracket);
    }
  }
  return std::move(schedule.result);
}

} // namespace datamunge::optim
