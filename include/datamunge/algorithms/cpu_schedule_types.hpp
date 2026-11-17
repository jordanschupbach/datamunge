#pragma once
/// @file cpu_schedule_types.hpp
/// @brief Shared CPU scheduling records preserving legacy layouts and units.
#include <cstddef>
#include <vector>
/// @brief Canonical integral single-CPU scheduling records.
namespace datamunge::algorithms {
/// @brief One integral arrival/burst request with an opaque external id.
/// Scheduling methods specify validation; checked round-robin requires unique ids and nonnegative arrival/burst.
/// @par Example
/// @code
/// const dm::algorithms::SchedProcess p{7,2,5};std::print("{}\n",p.burst); // 5
/// @endcode
struct SchedProcess {
  std::size_t id{};      ///< Opaque caller identity, not a dense array index.
  int         arrival{}; ///< Integral time at which the process becomes ready.
  int         burst{};   ///< Required nonnegative busy time.
};
/// @brief Owned per-process completion and derived timing quantities.
/// @par Example
/// @code
/// const dm::algorithms::SchedMetrics m{7,8,6,1};std::print("{}\n",m.waiting); // 1
/// @endcode
struct SchedMetrics {
  std::size_t id{};         ///< Original opaque process id.
  int         completion{}; ///< Integral completion boundary.
  int         turnaround{}; ///< Completion minus arrival.
  int         waiting{};    ///< Turnaround minus burst.
};
/// @brief One owned half-open contiguous busy interval.
/// @par Example
/// @code
/// const dm::algorithms::GanttSegment s{7,2,5};std::print("{}\n",s.end-s.start); // 3
/// @endcode
struct GanttSegment {
  std::size_t id{};    ///< Executing external process id.
  int         start{}; ///< Inclusive integral start time.
  int         end{};   ///< Exclusive integral end time.
};
/// @brief Owned per-input outcomes, merged busy intervals and sample average timing.
/// @par Example
/// @code
/// const dm::algorithms::ScheduleResult r{};std::print("{}\n",r.metrics.size()); // 0
/// @endcode
struct ScheduleResult {
  std::vector<SchedMetrics> metrics; ///< One outcome per caller input, in original order.
  std::vector<GanttSegment> gantt; ///< Chronological merged busy intervals; idle gaps are omitted.
  double avg_waiting{};            ///< Mean process waiting time, zero for an empty request.
  double avg_turnaround{};         ///< Mean process turnaround, zero for an empty request.
};
} // namespace datamunge::algorithms
