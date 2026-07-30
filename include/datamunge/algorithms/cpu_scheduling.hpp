#pragma once

/// \file cpu_scheduling.hpp
/// \brief Classic CPU scheduling algorithms: round-robin, shortest-job-next,
///        shortest-remaining-time.
///
/// Each process is described by an arrival time and a CPU burst length. A scheduler decides
/// which ready process runs on the (single) CPU at each instant; the quality of a schedule
/// is measured by *turnaround time* (completion - arrival) and *waiting time* (turnaround -
/// burst), averaged over processes. This module simulates three schedulers integrally (time
/// quantum of one unit) and reports the per-process metrics and the execution timeline.

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A process to schedule.
struct SchedProcess {
    std::size_t id;       ///< Process id.
    int         arrival;  ///< Time the process becomes ready.
    int         burst;    ///< Total CPU time it needs.
};

/// Per-process outcome of a schedule.
struct SchedMetrics {
    std::size_t id;
    int         completion;  ///< Time the process finished.
    int         turnaround;  ///< completion - arrival.
    int         waiting;     ///< turnaround - burst.
};

/// A contiguous slice of the timeline where one process ran.
struct GanttSegment {
    std::size_t id;
    int         start;
    int         end;
};

/// The full result of a schedule.
struct ScheduleResult {
    std::vector<SchedMetrics> metrics;         ///< One entry per process (input order).
    std::vector<GanttSegment> gantt;           ///< Merged execution timeline.
    double                    avg_waiting;     ///< Mean waiting time.
    double                    avg_turnaround;  ///< Mean turnaround time.
};

namespace detail {

/// Append one unit of execution by `id` at `t` to a Gantt timeline, merging with the
/// previous segment if it is the same process running contiguously.
inline void gantt_tick(std::vector<GanttSegment>& g, std::size_t id, int t) {
    if (!g.empty() && g.back().id == id && g.back().end == t)
        g.back().end = t + 1;
    else
        g.push_back({id, t, t + 1});
}

/// Fill in metrics + averages from completion times.
inline ScheduleResult finalize(const std::vector<SchedProcess>& procs,
                               const std::vector<int>&          completion,
                               std::vector<GanttSegment>        gantt) {
    ScheduleResult r;
    r.gantt = std::move(gantt);
    double sum_wait = 0.0, sum_turn = 0.0;
    for (std::size_t i = 0; i < procs.size(); ++i) {
        int turnaround = completion[i] - procs[i].arrival;
        int waiting    = turnaround - procs[i].burst;
        r.metrics.push_back({procs[i].id, completion[i], turnaround, waiting});
        sum_wait += waiting;
        sum_turn += turnaround;
    }
    r.avg_waiting    = procs.empty() ? 0.0 : sum_wait / static_cast<double>(procs.size());
    r.avg_turnaround = procs.empty() ? 0.0 : sum_turn / static_cast<double>(procs.size());
    return r;
}

}  // namespace detail

/// \brief Round-robin scheduling: each ready process runs for at most `quantum` units before
///        being preempted to the back of the FIFO ready queue.
///
/// Newly arrived processes join the queue before a just-preempted process is re-added (the
/// standard convention), giving round-robin its characteristic fairness.
inline ScheduleResult round_robin(const std::vector<SchedProcess>& procs, int quantum) {
    const std::size_t n = procs.size();
    std::vector<int>  remaining(n), completion(n, 0);
    for (std::size_t i = 0; i < n; ++i) remaining[i] = procs[i].burst;

    std::vector<GanttSegment> gantt;
    std::queue<std::size_t>   ready;
    std::vector<bool>         enqueued(n, false);

    auto enqueue_arrivals = [&](int upto) {
        for (std::size_t i = 0; i < n; ++i)
            if (!enqueued[i] && procs[i].arrival <= upto && remaining[i] > 0) {
                ready.push(i);
                enqueued[i] = true;
            }
    };

    int         t         = 0;
    std::size_t completed = 0;
    enqueue_arrivals(t);
    while (completed < n) {
        if (ready.empty()) { ++t; enqueue_arrivals(t); continue; }  // CPU idle until next arrival
        std::size_t i     = ready.front();
        ready.pop();
        int slice = quantum < remaining[i] ? quantum : remaining[i];
        for (int s = 0; s < slice; ++s) {
            detail::gantt_tick(gantt, procs[i].id, t);
            ++t;
            --remaining[i];
            enqueue_arrivals(t);  // arrivals during this quantum queue ahead of the preempted job
        }
        if (remaining[i] == 0) {
            completion[i] = t;
            ++completed;
        } else {
            ready.push(i);  // preempted: back of the queue, after any new arrivals
        }
    }
    return detail::finalize(procs, completion, std::move(gantt));
}

/// \brief Shortest-job-next (non-preemptive): whenever the CPU is free, start the ready
///        process with the smallest total burst (ties by arrival, then id); run it to
///        completion.
inline ScheduleResult shortest_job_next(const std::vector<SchedProcess>& procs) {
    const std::size_t n = procs.size();
    std::vector<int>  completion(n, 0);
    std::vector<bool> done(n, false);
    std::vector<GanttSegment> gantt;

    int         t         = 0;
    std::size_t completed = 0;
    while (completed < n) {
        std::size_t best  = n;
        for (std::size_t i = 0; i < n; ++i) {
            if (done[i] || procs[i].arrival > t) continue;
            if (best == n || procs[i].burst < procs[best].burst ||
                (procs[i].burst == procs[best].burst && procs[i].arrival < procs[best].arrival))
                best = i;
        }
        if (best == n) { ++t; continue; }  // nothing ready: idle
        for (int s = 0; s < procs[best].burst; ++s) { detail::gantt_tick(gantt, procs[best].id, t); ++t; }
        completion[best] = t;
        done[best]       = true;
        ++completed;
    }
    return detail::finalize(procs, completion, std::move(gantt));
}

/// \brief Shortest-remaining-time (preemptive shortest-job-first): at each time unit run the
///        ready process with the least remaining work, preempting the running one whenever a
///        shorter job arrives.
inline ScheduleResult shortest_remaining_time(const std::vector<SchedProcess>& procs) {
    const std::size_t n = procs.size();
    std::vector<int>  remaining(n), completion(n, 0);
    for (std::size_t i = 0; i < n; ++i) remaining[i] = procs[i].burst;
    std::vector<GanttSegment> gantt;

    int         t         = 0;
    std::size_t completed = 0;
    while (completed < n) {
        std::size_t best = n;
        for (std::size_t i = 0; i < n; ++i) {
            if (remaining[i] == 0 || procs[i].arrival > t) continue;
            if (best == n || remaining[i] < remaining[best] ||
                (remaining[i] == remaining[best] && procs[i].arrival < procs[best].arrival))
                best = i;
        }
        if (best == n) { ++t; continue; }  // idle
        detail::gantt_tick(gantt, procs[best].id, t);
        ++t;
        if (--remaining[best] == 0) { completion[best] = t; ++completed; }
    }
    return detail::finalize(procs, completion, std::move(gantt));
}

}  // namespace datamunge::algorithms
