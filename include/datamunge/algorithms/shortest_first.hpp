#pragma once
/// @file shortest_first.hpp
/// @brief Checked event-driven shortest-job and shortest-remaining schedulers.
#include <cstddef>
#include <datamunge/algorithms/cpu_schedule_types.hpp>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief ShortestFirstOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShortestFirstOptions value{};
/// std::print("{}\n",value.max_processes); // 100000
/// @endcode
struct ShortestFirstOptions {
  std::size_t max_processes{100000};     ///< Complete input count; zero disables.
  std::size_t max_comparisons{20000000}; ///< Actual process eligibility scans; zero disables.
  std::size_t max_records{
      1000000}; ///< Optional decision records plus copied remaining-work entries; zero disables.
};
/// @brief ShortestFirstDecision stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShortestFirstDecision value{};
/// std::print("{}\n",value.idle); // false
/// @endcode
struct ShortestFirstDecision {
  int              time{};    ///< Actual decision time.
  int              end{};     ///< Actual next arrival or completion, or next arrival after idle.
  bool             idle{};    ///< No eligible process; id ignored when true.
  std::size_t      id{};      ///< Actual selected process ID, not vector position.
  std::vector<int> remaining; ///< Actual pre-dispatch work per original input row.
};
/// @brief ShortestFirstRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShortestFirstRun value{};
/// std::print("{}\n",value.preemptions); // 0
/// @endcode
struct ShortestFirstRun {
  ScheduleResult schedule; ///< Actual merged timeline and per-input-row completion metrics.
  std::size_t    scans{};  ///< Actual examined process rows, including ineligible rows.
  std::size_t
      dispatches{}; ///< Actual non-idle intervals, including continued execution across arrivals.
  std::size_t preemptions{}; ///< Actual changes away from an unfinished process.
  std::vector<ShortestFirstDecision>
      decisions; ///< Optional actual event decisions; includes idle jumps.
};
/// @brief Dispatch the shortest ready complete burst without preemption.
/// Single CPU, integer arrival>=0, burst>0, unique arbitrary IDs. Ties use earlier arrival then original input position, preserving historical behavior (not numeric ID). Event jumps skip idle time; SRT decisions occur at arrivals/completions, SJN only on completion/idle. SRT merged timeline exactly matches unit ticks. Zero context-switch cost and known bursts. SJN minimizes total wait when all jobs are initially available; with release times and optional idling that claim is false. SRT minimizes total completion/flow time under preemption and the stated ideal assumptions. Finite static inputs finish; future arrivals can starve long jobs. O(n) per decision, at most O(n) arrival/completion decisions, O(n^2) scans; optional history O(n^2). Complete horizon checked against int before simulation; outputs own storage and no shared state.
/// @param processes Complete process rows in deterministic tie order.
/// @param options Input, executed scan and optional captured-state policies; zero disables policies.
/// @return Owned actual schedule/work and optional event states.
/// @throws std::invalid_argument Negative arrival, nonpositive burst or duplicate ID.
/// @throws std::length_error Input/work/history policy, horizon or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shortest_job_next_run({{9,0,2},{4,0,1}});
/// std::print("{}\n",r.schedule.gantt.front().id); // 4
/// @endcode
/// @par References
/// - L. Schrage, A proof of the optimality of the shortest remaining processing time discipline, Operations Research 16(3), 1968, DOI 10.1287/opre.16.3.687.
/// @see shortest_job_next()
ShortestFirstRun shortest_job_next_run(
    const std::vector<SchedProcess>& processes,
    const ShortestFirstOptions&      options = {});
/// @brief Dispatch the shortest ready complete burst without preemption. Retain actual decisions.
/// Single CPU, integer arrival>=0, burst>0, unique arbitrary IDs. Ties use earlier arrival then original input position, preserving historical behavior (not numeric ID). Event jumps skip idle time; SRT decisions occur at arrivals/completions, SJN only on completion/idle. SRT merged timeline exactly matches unit ticks. Zero context-switch cost and known bursts. SJN minimizes total wait when all jobs are initially available; with release times and optional idling that claim is false. SRT minimizes total completion/flow time under preemption and the stated ideal assumptions. Finite static inputs finish; future arrivals can starve long jobs. O(n) per decision, at most O(n) arrival/completion decisions, O(n^2) scans; optional history O(n^2). Complete horizon checked against int before simulation; outputs own storage and no shared state.
/// @param processes Complete process rows in deterministic tie order.
/// @param options Input, executed scan and optional captured-state policies; zero disables policies.
/// @return Owned actual schedule/work and optional event states.
/// @throws std::invalid_argument Negative arrival, nonpositive burst or duplicate ID.
/// @throws std::length_error Input/work/history policy, horizon or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shortest_job_next_trace({{9,0,2},{4,0,1}});
/// std::print("{}\n",r.schedule.gantt.front().id); // 4
/// @endcode
/// @par References
/// - L. Schrage, A proof of the optimality of the shortest remaining processing time discipline, Operations Research 16(3), 1968, DOI 10.1287/opre.16.3.687.
/// @see shortest_job_next()
ShortestFirstRun shortest_job_next_trace(
    const std::vector<SchedProcess>& processes,
    const ShortestFirstOptions&      options = {});
/// @brief Return the preserved shortest_job_next schedule.
/// Uses the same checked native event producer and historical input-order tie rule.
/// @param procs Complete process rows; arrival>=0, burst>0, unique IDs.
/// @return Owned merged timeline and per-input-order metrics.
/// @throws std::invalid_argument Malformed process rows.
/// @throws std::length_error Default policy, time or counter capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shortest_job_next({{7,0,1}}).metrics.front().completion); // 1
/// @endcode
/// @see shortest_job_next_run()
ScheduleResult shortest_job_next(const std::vector<SchedProcess>& procs);
/// @brief Preempt at arrivals for shortest remaining work.
/// Single CPU, integer arrival>=0, burst>0, unique arbitrary IDs. Ties use earlier arrival then original input position, preserving historical behavior (not numeric ID). Event jumps skip idle time; SRT decisions occur at arrivals/completions, SJN only on completion/idle. SRT merged timeline exactly matches unit ticks. Zero context-switch cost and known bursts. SJN minimizes total wait when all jobs are initially available; with release times and optional idling that claim is false. SRT minimizes total completion/flow time under preemption and the stated ideal assumptions. Finite static inputs finish; future arrivals can starve long jobs. O(n) per decision, at most O(n) arrival/completion decisions, O(n^2) scans; optional history O(n^2). Complete horizon checked against int before simulation; outputs own storage and no shared state.
/// @param processes Complete process rows in deterministic tie order.
/// @param options Input, executed scan and optional captured-state policies; zero disables policies.
/// @return Owned actual schedule/work and optional event states.
/// @throws std::invalid_argument Negative arrival, nonpositive burst or duplicate ID.
/// @throws std::length_error Input/work/history policy, horizon or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shortest_remaining_time_run({{9,0,2},{4,0,1}});
/// std::print("{}\n",r.schedule.gantt.front().id); // 4
/// @endcode
/// @par References
/// - L. Schrage, A proof of the optimality of the shortest remaining processing time discipline, Operations Research 16(3), 1968, DOI 10.1287/opre.16.3.687.
/// @see shortest_remaining_time()
ShortestFirstRun shortest_remaining_time_run(
    const std::vector<SchedProcess>& processes,
    const ShortestFirstOptions&      options = {});
/// @brief Preempt at arrivals for shortest remaining work. Retain actual decisions.
/// Single CPU, integer arrival>=0, burst>0, unique arbitrary IDs. Ties use earlier arrival then original input position, preserving historical behavior (not numeric ID). Event jumps skip idle time; SRT decisions occur at arrivals/completions, SJN only on completion/idle. SRT merged timeline exactly matches unit ticks. Zero context-switch cost and known bursts. SJN minimizes total wait when all jobs are initially available; with release times and optional idling that claim is false. SRT minimizes total completion/flow time under preemption and the stated ideal assumptions. Finite static inputs finish; future arrivals can starve long jobs. O(n) per decision, at most O(n) arrival/completion decisions, O(n^2) scans; optional history O(n^2). Complete horizon checked against int before simulation; outputs own storage and no shared state.
/// @param processes Complete process rows in deterministic tie order.
/// @param options Input, executed scan and optional captured-state policies; zero disables policies.
/// @return Owned actual schedule/work and optional event states.
/// @throws std::invalid_argument Negative arrival, nonpositive burst or duplicate ID.
/// @throws std::length_error Input/work/history policy, horizon or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shortest_remaining_time_trace({{9,0,2},{4,0,1}});
/// std::print("{}\n",r.schedule.gantt.front().id); // 4
/// @endcode
/// @par References
/// - L. Schrage, A proof of the optimality of the shortest remaining processing time discipline, Operations Research 16(3), 1968, DOI 10.1287/opre.16.3.687.
/// @see shortest_remaining_time()
ShortestFirstRun shortest_remaining_time_trace(
    const std::vector<SchedProcess>& processes,
    const ShortestFirstOptions&      options = {});
/// @brief Return the preserved shortest_remaining_time schedule.
/// Uses the same checked native event producer and historical input-order tie rule.
/// @param procs Complete process rows; arrival>=0, burst>0, unique IDs.
/// @return Owned merged timeline and per-input-order metrics.
/// @throws std::invalid_argument Malformed process rows.
/// @throws std::length_error Default policy, time or counter capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shortest_remaining_time({{7,0,1}}).metrics.front().completion); // 1
/// @endcode
/// @see shortest_remaining_time_run()
ScheduleResult shortest_remaining_time(const std::vector<SchedProcess>& procs);
} // namespace datamunge::algorithms
