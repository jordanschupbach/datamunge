#pragma once
/// @file round_robin.hpp
/// @brief Checked round-robin FIFO execution with actual dispatch and queue histories.
#include <cstddef>
#include <datamunge/algorithms/cpu_schedule_types.hpp>
#include <datamunge/plot/plot.hpp>
#include <vector>
/// @brief Canonical integral CPU scheduling and actual event records.
namespace datamunge::algorithms {
/// @brief Complete input/busy-work and actual scan/history policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RoundRobinOptions value{};
/// std::print("{}\n",value.max_busy_ticks); // 1000000
/// @endcode
struct RoundRobinOptions {
  std::size_t max_processes{1000000};  ///< Complete input process count.
  std::size_t max_busy_ticks{1000000}; ///< Complete aggregate positive burst work.
  std::size_t max_arrival_checks{
      10000000}; ///< Actual per-process admission tests, including idle jumps.
  std::size_t max_trace_entries{
      1000000}; ///< Actual before/after queue ids plus one header per dispatch.
};
/// @brief One actual quantum dispatch with original internal input ids and queue states.
/// @par Example
/// @code
/// const dm::algorithms::RoundRobinStep value{};
/// std::print("{}\n",value.ready_after.size()); // 0
/// @endcode
struct RoundRobinStep {
  std::size_t process_index{}; ///< Selected actual input index; external id comes from the request.
  int         start{};         ///< Actual dispatch start boundary.
  int         end{};           ///< Actual end boundary after min(quantum,remaining) busy ticks.
  int         remaining_before{}; ///< Actual remaining burst before the dispatch.
  int         remaining_after{};  ///< Actual remaining burst after it.
  std::vector<std::size_t>
      ready_before; ///< Actual FIFO input ids before popping the selected front.
  std::vector<std::size_t>
      ready_after; ///< Actual FIFO after all intervening arrivals and optional requeue.
};
/// @brief Owned actual round-robin schedule and selected simulation work.
/// @par Example
/// @code
/// const dm::algorithms::RoundRobinRun value{};
/// std::print("{}\n",value.dispatches); // 0
/// @endcode
struct RoundRobinRun {
  ScheduleResult result;       ///< Actual final schedule in legacy field order.
  std::size_t    busy_ticks{}; ///< Actual executed unit intervals.
  std::size_t
      arrival_checks{}; ///< Actual input admission predicates across initial/tick/idle scans.
  std::size_t
      dispatches{}; ///< Actual FIFO pops, including contiguous repeated runs of a lone process.
  std::size_t preemptions{}; ///< Actual nonfinished dispatches requeued at the back.
  std::size_t idle_jumps{}; ///< Actual direct advances to the next unadmitted positive job arrival.
  std::vector<RoundRobinStep> steps; ///< Actual optional chronological dispatch histories.
};
/// @brief Schedule one checked FIFO round-robin workload.
/// Requires positive quantum, unique opaque ids and nonnegative integral arrivals/bursts. New arrivals are admitted after each actual busy tick, before requeuing a preempted process, with input-order ties. Zero-burst processes complete at their arrival and consume no queue/CPU time. Empty input returns zero averages. Idle periods jump directly to the next positive-work arrival, replacing the old idle-tick loop; busy Gantt intervals retain legacy merging. Uses checked int time to preserve public layouts; unrepresentable completion rejects. Selected arrival checks count each actual per-input admission test; busy ticks/dispatches/preemptions are observed separately. Time O(P*(busy ticks+idle jumps+1)+dispatches), O(P+timeline) working/result storage plus actual optional queues. Complete burst bound precedes simulation; actual history costs before+after FIFO lengths plus one per dispatch, admitted before retaining the completed record. Inputs unchanged, owned results, no shared state.
/// @param processes Complete borrowed integral workload.
/// @param quantum Strictly positive busy-time quantum.
/// @param options Complete workload and actual-work/history policies.
/// @return Actual legacy schedule, selected simulation work and optional same-pass FIFO histories.
/// @throws std::invalid_argument Nonpositive quantum, duplicate id or negative arrival/burst.
/// @throws std::length_error Input, aggregate busy, actual arrival-scan, history or capacity policy exceeded.
/// @throws std::overflow_error Completion boundary exceeds public int time range.
/// @par Example
/// @code
/// auto r=dm::algorithms::round_robin_run({{7,0,3},{91,0,1}},2);
/// std::print("{} {}\n",r.result.metrics[0].completion,r.result.metrics[1].completion); // 4 3
/// @endcode
/// @par References
/// - Silberschatz, Galvin and Gagne, Operating System Concepts,10th ed., CPU scheduling, https://www.os-book.com/OS10/index.html.
/// @see round_robin(), plot_round_robin_schedule()
RoundRobinRun round_robin_run(
    const std::vector<SchedProcess>& processes,
    int                              quantum,
    const RoundRobinOptions&         options = {});
/// @brief Schedule one checked FIFO round-robin workload with actual queue histories.
/// Requires positive quantum, unique opaque ids and nonnegative integral arrivals/bursts. New arrivals are admitted after each actual busy tick, before requeuing a preempted process, with input-order ties. Zero-burst processes complete at their arrival and consume no queue/CPU time. Empty input returns zero averages. Idle periods jump directly to the next positive-work arrival, replacing the old idle-tick loop; busy Gantt intervals retain legacy merging. Uses checked int time to preserve public layouts; unrepresentable completion rejects. Selected arrival checks count each actual per-input admission test; busy ticks/dispatches/preemptions are observed separately. Time O(P*(busy ticks+idle jumps+1)+dispatches), O(P+timeline) working/result storage plus actual optional queues. Complete burst bound precedes simulation; actual history costs before+after FIFO lengths plus one per dispatch, admitted before retaining the completed record. Inputs unchanged, owned results, no shared state.
/// @param processes Complete borrowed integral workload.
/// @param quantum Strictly positive busy-time quantum.
/// @param options Complete workload and actual-work/history policies.
/// @return Actual legacy schedule, selected simulation work and optional same-pass FIFO histories.
/// @throws std::invalid_argument Nonpositive quantum, duplicate id or negative arrival/burst.
/// @throws std::length_error Input, aggregate busy, actual arrival-scan, history or capacity policy exceeded.
/// @throws std::overflow_error Completion boundary exceeds public int time range.
/// @par Example
/// @code
/// auto r=dm::algorithms::round_robin_trace({{7,0,3},{91,0,1}},2);
/// std::print("{} {}\n",r.result.metrics[0].completion,r.result.metrics[1].completion); // 4 3
/// @endcode
/// @par References
/// - Silberschatz, Galvin and Gagne, Operating System Concepts,10th ed., CPU scheduling, https://www.os-book.com/OS10/index.html.
/// @see round_robin(), plot_round_robin_schedule()
RoundRobinRun round_robin_trace(
    const std::vector<SchedProcess>& processes,
    int                              quantum,
    const RoundRobinOptions&         options = {});
/// @brief Return the checked native round-robin schedule through the preserved signature.
/// Uses the same default bounded producer and preserves ordinary positive-burst workloads. Invalid controls now reject instead of hanging; zero-burst completion is explicitly its arrival time.
/// @param procs Complete nonnegative workload with unique external ids.
/// @param quantum Strictly positive integral quantum.
/// @return Owned actual process metrics and merged busy timeline.
/// @throws std::invalid_argument Invalid quantum/process.
/// @throws std::length_error Default policy exceeded.
/// @throws std::overflow_error Unrepresentable completion.
/// @par Example
/// @code
/// auto r=dm::algorithms::round_robin({{7,0,3}},2);
/// std::print("{}\n",r.metrics[0].completion); // 3
/// @endcode
/// @par References
/// - Silberschatz, Galvin and Gagne, Operating System Concepts,10th ed., CPU scheduling, https://www.os-book.com/OS10/index.html.
/// @see round_robin_run()
ScheduleResult round_robin(const std::vector<SchedProcess>& procs, int quantum);
/// @brief Plot an actual CPU busy timeline with sparse external process ids mapped to input-order rows.
/// Validates unique process ids, every positive chronological nonoverlapping segment, valid process references and the segment count policy. Displays actual half-open busy rectangles; idle gaps remain blank. This checks geometry, not a caller-edited scheduling proof. Inputs unchanged; returned editable native plot owns its geometry.
/// @param processes Original input-order process identities.
/// @param schedule Actual schedule or structurally valid edited timeline.
/// @param max_segments Complete plot segment policy, default1000000; zero disables.
/// @return Owned editable native Gantt plot.
/// @throws std::invalid_argument Duplicate/unknown process ids or invalid interval ordering.
/// @throws std::length_error Segment policy exceeded.
/// @par Example
/// @code
/// auto p=dm::algorithms::plot_round_robin_schedule({{7,0,3}},dm::algorithms::round_robin({{7,0,3}},2));
/// p.save_svg("round_robin_example.svg");
/// @endcode
/// @par References
/// - Silberschatz, Galvin and Gagne, Operating System Concepts,10th ed., CPU scheduling, https://www.os-book.com/OS10/index.html.
/// @see round_robin_trace()
graphics::RPlot plot_round_robin_schedule(
    const std::vector<SchedProcess>& processes,
    const ScheduleResult&            schedule,
    std::size_t                      max_segments = 1000000);
} // namespace datamunge::algorithms
