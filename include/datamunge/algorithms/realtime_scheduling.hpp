#pragma once
/// @file realtime_scheduling.hpp
/// @brief Checked synchronous periodic EDF/RMS scheduling, actual job evidence and native timelines.
#include <datamunge/plot/plot.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {

/// @brief An independent synchronous periodic task with implicit deadlines.
/// Jobs release at 0,T,2T,... and need C integer processor units by the next
/// release. One fully preemptive processor, zero overhead and no blocking are
/// assumed. APIs require positive period, nonnegative wcet and unique ids other
/// than SIZE_MAX (reserved for idle). C>T is a valid overloaded task.
/// @par Example
/// @code
/// const dm::algorithms::RTTask t{7,4,1};
/// std::print("{} {}\n",t.period,t.wcet); // 4 1
/// @endcode
/// @par References
/// - C. L. Liu and J. W. Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, JACM 20(1), 46–61. doi:10.1145/321738.321743.
/// @see edf_simulate(), rt_trace()
struct RTTask {
    std::size_t id; ///< Opaque task id; distinct from input row index.
    int period; ///< Positive integer release period and relative deadline.
    int wcet; ///< Nonnegative worst-case work per job, in processor units.
};

/// @brief An owned-label interval in a merged task execution timeline.
/// Half-open [start,end); returned slices have nonnegative start and end>start.
/// Adjacent units running the same task are merged, even across job boundaries.
/// @par Example
/// @code
/// const dm::algorithms::RTSlice s{7,0,2};
/// std::print("{}\n",s.end-s.start); // 2
/// @endcode
/// @see RTScheduleResult
struct RTSlice {
    std::size_t id; ///< Task id, or SIZE_MAX for idle.
    int start; ///< Inclusive integer start.
    int end; ///< Exclusive integer end.
};

/// @brief Merged task timeline and deadline verdict for a finite simulation.
/// Timeline covers [0,H) exactly. deadline_missed includes all job deadlines
/// through boundary H; unfinished jobs due after H are not yet misses. Copies
/// own their timeline. A finite clean prefix is not an infinite schedulability proof.
/// @par Example
/// @code
/// const auto r=dm::algorithms::edf_simulate({{7,2,1}},2);
/// std::print("{} {}\n",r.timeline.size(),r.deadline_missed); // 2 false
/// @endcode
/// @see edf_simulate(), rt_trace()
struct RTScheduleResult {
    std::vector<RTSlice> timeline; ///< Merged execution and idle intervals.
    bool deadline_missed; ///< Any unfinished job at its deadline <=H.
};

/// @brief Dynamic deadline or fixed period priority, with stable input-order ties.
/// Equal deadlines/periods prefer the earlier input task; jobs of one task run
/// in release order. Task-id numeric order has no scheduling significance.
/// @par Example
/// @code
/// const auto r=dm::algorithms::rt_simulate_work({{7,2,1}},2,dm::algorithms::RTPolicy::earliest_deadline);
/// std::print("{}\n",r.work.busy_ticks); // 1
/// @endcode
/// @par References
/// - Liu and Layland (1973), doi:10.1145/321738.321743.
/// @see rt_simulate_work()
enum class RTPolicy {
    earliest_deadline, ///< Pick the unfinished job with the earliest absolute deadline.
    rate_monotonic ///< Pick the unfinished job from the shortest-period task.
};

/// @brief Checked logical counters for the actual unit-granularity simulator.
/// Counts are not elapsed time or scheduler overhead. Preemption means the next
/// unit selects a different job while the previously selected job is unfinished;
/// stopping at the horizon is not a preemption. Zero-work releases complete at once.
/// @par Example
/// @code
/// const auto r=dm::algorithms::rt_simulate_work({{7,2,1}},4,dm::algorithms::RTPolicy::earliest_deadline);
/// std::print("{} {}\n",r.work.jobs_released,r.work.idle_ticks); // 2 2
/// @endcode
/// @see RTSimulation, RTTrace
struct RTWork {
    std::size_t ticks=0; ///< Number of simulated units, exactly H.
    std::size_t release_checks=0; ///< Task release tests, exactly N*H.
    std::size_t jobs_released=0; ///< All releases before H, including zero-work jobs.
    std::size_t jobs_completed=0; ///< Jobs completed by boundary H, including zero-work jobs.
    std::size_t busy_ticks=0; ///< Units executing a job.
    std::size_t idle_ticks=0; ///< Units with no ready job.
    std::size_t deadline_misses=0; ///< Distinct jobs unfinished at their deadline <=H.
    std::size_t peak_ready=0; ///< Maximum unfinished ready jobs after releases, before selection.
    std::size_t preemptions=0; ///< Actual changes away from an unfinished selected job.
};

/// @brief Finite scheduling outcome and work, with no retained job/tick history.
/// Independent calls leave input unchanged. Runtime retains unfinished jobs;
/// completed jobs are removed. There is no optional capture quota.
/// @par Example
/// @code
/// const auto r=dm::algorithms::rt_simulate_work({{7,2,3}},2,dm::algorithms::RTPolicy::earliest_deadline);
/// std::print("{} {}\n",r.schedule.deadline_missed,r.work.deadline_misses); // true 1
/// @endcode
/// @see rt_simulate_work(), rt_trace()
struct RTSimulation {
    RTScheduleResult schedule; ///< Merged timeline and verdict.
    RTWork work; ///< Checked work from the same run.
};

/// @brief Actual per-job release, execution, completion and deadline evidence.
/// Job ids are indices in RTTrace::jobs, ordered by release time then input task.
/// Deadlines use int64_t so release+period cannot overflow int. A zero-work job
/// completes at release; an unfinished job has completion=-1 even if already late.
/// @par Example
/// @code
/// const auto t=dm::algorithms::rt_trace({{7,2,3}},2);
/// std::print("{} {} {}\n",t.jobs[0].deadline,t.jobs[0].executed,t.jobs[0].missed); // 2 2 true
/// @endcode
/// @see RTTrace, rt_trace()
struct RTJob {
    std::size_t task_index=0; ///< Index in the owned task vector, not its external id.
    int release=0; ///< Inclusive release time before H.
    std::int64_t deadline=0; ///< Absolute implicit deadline, possibly greater than H.
    int executed=0; ///< Actual executed units by H, between zero and wcet.
    int completion=-1; ///< Completion boundary, or -1 if unfinished.
    bool missed=false; ///< Unfinished at this job's deadline <=H, even if completed later.
};

/// @brief Owned actual release records and per-unit job selections, without replay.
/// Capture has N task records, H tick selections and J released job records, so
/// captured_entries=N+H+J. Scalars, result timeline, working heap/counters, vector
/// capacity and allocator bytes are excluded: this is not a peak-RAM bound.
/// Retained history uses O(N+H+J) entries; result/runtime memory is additional.
/// @par Example
/// @code
/// const auto t=dm::algorithms::rt_trace({{7,2,1}},2);
/// std::print("{} {}\n",t.jobs.size(),t.captured_entries); // 1 4
/// @endcode
/// @see rt_trace(), rt_trace_entries()
struct RTTrace {
    std::vector<RTTask> tasks; ///< Exact owned input in tie-breaking order.
    int horizon=0; ///< Simulated [0,H), with deadline checks through H.
    RTPolicy policy=RTPolicy::earliest_deadline; ///< Selection policy.
    std::vector<RTJob> jobs; ///< Actual released jobs, including zero-work jobs.
    std::vector<std::size_t> tick_jobs; ///< Actual selected job index each unit; SIZE_MAX means idle.
    RTScheduleResult schedule; ///< Merged task result of this same run.
    RTWork work; ///< Checked logical work counters.
    std::size_t captured_entries=0; ///< Exact N+H+J retained entries.
};

/// @brief Sum periodic processor demand C/T as a floating-point display value.
///
/// Validates the full task domain. Empty input returns zero; zero-work tasks
/// contribute zero. Input is unchanged and calls share no mutable state. Floating
/// rounding applies; use edf_schedulable() for the exact implicit-deadline test.
/// @param tasks Independent synchronous tasks with implicit deadlines.
/// @return Approximate utilization, in processor units per time unit.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::bad_alloc If validation allocation fails.
/// @par Example
/// @code
/// std::print("{:.2f}\n",dm::algorithms::rt_utilization({{7,4,1}})); // 0.25
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see edf_schedulable()
double rt_utilization(const std::vector<RTTask>& tasks);

/// @brief Test exact rational utilization <=1 for the implicit-deadline EDF model.
///
/// Fully preemptive independent periodic tasks, one processor and zero overhead
/// are required. No floating tolerance is used. Nonnegative partial rational sums
/// use private integer limbs; worst-case O(N^2) word operations and O(N) words,
/// without denominator reduction. Empty/zero-work sets pass. Input is unchanged.
/// @param tasks Valid synchronous implicit-deadline tasks.
/// @return Whether exact sum(C/T)<=1; not a claim for blocking, jitter or multiprocessors.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::length_error On container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::edf_schedulable({{0,3,1},{1,3,1},{2,3,1}})); // true
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see rt_utilization(), edf_simulate()
bool edf_schedulable(const std::vector<RTTask>& tasks);

/// @brief Evaluate the Liu-Layland RMS sufficient utilization bound numerically.
///
/// Computes n*expm1(log(2)/n), avoiding cancellation for large n. The empty
/// set convention is one. This is a floating-point approximation to a sufficient
/// bound, not the maximum utilization RMS can ever schedule. No mutable state.
/// @param n Number of tasks; zero returns one.
/// @return Approximation to n*(2^(1/n)-1), tending to log(2).
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::rate_monotonic_bound(2)); // 0.828427
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see rate_monotonic_schedulable()
double rate_monotonic_bound(std::size_t n);

/// @brief Screen RMS using its sufficient utilization bound.
///
/// Compares floating utilization to the numeric bound without an added epsilon.
/// False is inconclusive, not proof of a missed deadline. Near-boundary rounding
/// applies to this screen; it is not an exact necessary RMS feasibility test.
/// Validates all tasks; input is unchanged and calls share no mutable state.
/// @param tasks Independent synchronous tasks under the RTTask model.
/// @return Whether the floating utilization is <= the numeric sufficient bound.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::bad_alloc If validation allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rate_monotonic_schedulable({{0,4,2},{1,6,2}})); // false
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see rate_monotonic_bound(), rate_monotonic_simulate()
bool rate_monotonic_schedulable(const std::vector<RTTask>& tasks);

/// @brief Order RMS task ids by increasing period, then input position.
///
/// Ids are opaque labels, not priority numbers. Returns owned ids and leaves
/// input unchanged. Stable sorting costs O(N log(N+1)) comparisons.
/// @param tasks Valid periodic tasks.
/// @return Task ids from highest to lowest fixed priority; empty for empty input.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::rate_monotonic_priorities({{99,5,1},{2,3,1}});
/// std::print("{} {}\n",p[0],p[1]); // 2 99
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see rate_monotonic_simulate()
std::vector<std::size_t> rate_monotonic_priorities(const std::vector<RTTask>& tasks);

/// @brief Validate and count the complete retained history before tracing.
///
/// J=sum_i ceil(H/T_i) counts all releases strictly before H, including zero
/// work. Returns N+H+J with checked arithmetic and tick/job container geometry.
/// Validation may allocate an id set; no input/history copy is made.
/// @param tasks Valid task records.
/// @param horizon Nonnegative integer H.
/// @return Exact retained logical-entry count; empty tasks still require H idle selections.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::invalid_argument On negative horizon.
/// @throws std::length_error On arithmetic or vector geometry overflow.
/// @throws std::bad_alloc If validation allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rt_trace_entries({{7,2,1}},2)); // 4
/// @endcode
/// @see rt_trace()
std::size_t rt_trace_entries(const std::vector<RTTask>& tasks,int horizon);

/// @brief Simulate periodic EDF or RMS without retaining optional history.
///
/// All releases are synchronous, deadlines implicit, execution unit-granular and
/// preemption free. Unfinished jobs remain ready after a miss; none are discarded.
/// Checks deadlines through H, including the final boundary; completion exactly
/// at a deadline is on time. Priority ties use input task then release order.
/// Runtime tests N releases per tick and uses a priority queue of up to B ready
/// jobs: O(H(N+1)+(H+J)log(B+1)) unit-word work, O(N+B+timeline) storage,
/// plus validation. No optional quota; input is unchanged, independent calls safe.
/// @param tasks Valid periodic tasks.
/// @param horizon Nonnegative H; zero returns no slices or jobs.
/// @param policy EDF or RMS; invalid enum values are rejected.
/// @return Merged [0,H) timeline, inclusive deadline verdict and checked counters.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::invalid_argument On negative horizon or invalid policy.
/// @throws std::length_error On geometry, counter or container overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::rt_simulate_work({{7,2,1}},2,dm::algorithms::RTPolicy::earliest_deadline);
/// std::print("{} {}\n",r.work.busy_ticks,r.work.idle_ticks); // 1 1
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see rt_trace(), edf_simulate()
RTSimulation rt_simulate_work(const std::vector<RTTask>& tasks,int horizon,RTPolicy policy);

/// @brief Capture actual released jobs and selected job ids for every simulated unit.
///
/// Same kernel and behavior as rt_simulate_work(). Validates full input/geometry
/// before the optional complete N+H+J quota; checks quota before owned copies or
/// history allocation. Default one million entries; zero disables only quota.
/// Working heap, result timeline and allocator bytes are excluded. No partial
/// trace escapes failure. Copies own their data and input is unchanged.
/// @param tasks Valid tasks, copied in original priority-tie order.
/// @param horizon Nonnegative H; deadlines through H are checked.
/// @param policy Default earliest_deadline; rate_monotonic is also supported.
/// @param max_entries Complete retained-entry quota, default 1000000; zero disables it.
/// @return Owned actual job/tick evidence, merged result and work; no replay.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::invalid_argument On negative horizon or invalid policy.
/// @throws std::length_error On quota, geometry, counter or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::rt_trace({{7,2,1}},2,dm::algorithms::RTPolicy::earliest_deadline,4);
/// std::print("{} {}\n",t.jobs[0].completion,t.tick_jobs[0]); // 1 0
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see RTTrace, rt_trace_entries(), rt_simulate_work()
RTTrace rt_trace(const std::vector<RTTask>& tasks,int horizon,RTPolicy policy=RTPolicy::earliest_deadline,
                 std::size_t max_entries=1000000);

/// @brief Return the ordinary EDF merged timeline and deadline verdict.
///
/// Forwards to rt_simulate_work(), retaining the original result layout.
/// No unfinished job is dropped. Checks deadlines at boundary H, unlike the
/// old implementation. Input is unchanged; no optional history or capture quota.
/// @param tasks Valid synchronous periodic tasks.
/// @param horizon Nonnegative integer horizon.
/// @return Merged [0,H) slices and any missed deadline through H.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::invalid_argument On negative horizon.
/// @throws std::length_error On geometry/counter/container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::edf_simulate({{7,2,1}},2);
/// std::print("{} {}\n",r.timeline.size(),r.deadline_missed); // 2 false
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see rt_simulate_work(), rt_trace()
inline RTScheduleResult edf_simulate(const std::vector<RTTask>& tasks,int horizon) {
    return rt_simulate_work(tasks,horizon,RTPolicy::earliest_deadline).schedule;
}

/// @brief Return the ordinary RMS merged timeline and deadline verdict.
///
/// Forwards to rt_simulate_work(), retaining the original result layout.
/// No unfinished job is dropped. Checks deadlines at boundary H, unlike the
/// old implementation. Input is unchanged; no optional history or capture quota.
/// @param tasks Valid synchronous periodic tasks.
/// @param horizon Nonnegative integer horizon.
/// @return Merged [0,H) slices and any missed deadline through H.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::invalid_argument On negative horizon.
/// @throws std::length_error On geometry/counter/container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::rate_monotonic_simulate({{7,2,1}},2);
/// std::print("{} {}\n",r.timeline.size(),r.deadline_missed); // 2 false
/// @endcode
/// @par References
/// - Liu and Layland (1973), Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment, doi:10.1145/321738.321743.
/// @see rt_simulate_work(), rt_trace()
inline RTScheduleResult rate_monotonic_simulate(const std::vector<RTTask>& tasks,int horizon) {
    return rt_simulate_work(tasks,horizon,RTPolicy::rate_monotonic).schedule;
}

/// @brief Draw a validated merged task timeline using native RPlot polygons.
///
/// Task ids map to input-order rows, so sparse labels do not create huge axes.
/// Idle intervals are blank. Validates task domain, nonnegative contiguous slices
/// starting at zero and known task/idle ids before plotting. Empty timeline is
/// a valid zero-duration view. Does not certify CPU demand or deadline verdicts
/// of caller-edited results. Returns owned editable plot; inputs are unchanged.
/// @param tasks Task records in desired display order.
/// @param schedule Merged timeline to render; need not come from the current policy.
/// @return Editable native plot, with time and task labels.
/// @throws std::invalid_argument On nonpositive period, negative wcet, duplicate ids or reserved idle id.
/// @throws std::invalid_argument On malformed intervals or unknown task id.
/// @throws std::length_error On container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto tasks=std::vector<dm::algorithms::RTTask>{{7,2,1}};
/// auto p=dm::algorithms::plot_rt_schedule(tasks,dm::algorithms::edf_simulate(tasks,2));
/// std::print("{}\n",p.has_x_limits()); // true
/// @endcode
/// @see edf_simulate(), rt_trace()
graphics::RPlot plot_rt_schedule(const std::vector<RTTask>& tasks,const RTScheduleResult& schedule);

} // namespace datamunge::algorithms
