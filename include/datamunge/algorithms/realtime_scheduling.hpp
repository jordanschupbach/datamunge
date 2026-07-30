#pragma once

/// \file realtime_scheduling.hpp
/// \brief Real-time scheduling of periodic tasks: Earliest Deadline First (EDF) and
///        Rate-Monotonic Scheduling (RMS).
///
/// A periodic task \f$\tau_i\f$ releases a job every \f$T_i\f$ time units, each needing
/// \f$C_i\f$ units of CPU and due by the next release (implicit deadline). The processor
/// *utilization* is \f$U=\sum_i C_i/T_i\f$. Two classic single-processor schedulers:
///
/// - **EDF** (dynamic priority): always run the ready job with the earliest absolute
///   deadline. Optimal for implicit deadlines -- schedulable iff \f$U\le 1\f$.
/// - **RMS** (fixed priority): the shorter the period, the higher the (static) priority.
///   Schedulable if \f$U\le n(2^{1/n}-1)\f$ (Liu & Layland's sufficient bound; ~0.693 as
///   \f$n\to\infty\f$).
///
/// This module gives the utilization tests, the RMS priority order, and a preemptive
/// timeline simulation of each over a chosen horizon (reporting any deadline miss).

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// A periodic real-time task with implicit deadline (deadline == period).
struct RTTask {
    std::size_t id;      ///< Task id.
    int         period;  ///< Release period T.
    int         wcet;    ///< Worst-case execution time C per job.
};

/// A slice of the timeline where one task's job ran (id == SIZE_MAX means the CPU was idle).
struct RTSlice {
    std::size_t id;
    int         start;
    int         end;
};

/// Result of a real-time schedule simulation.
struct RTScheduleResult {
    std::vector<RTSlice> timeline;         ///< Merged execution timeline over the horizon.
    bool                 deadline_missed;  ///< True if any job missed its deadline.
};

/// \brief Total processor utilization \f$U=\sum_i C_i/T_i\f$.
inline double rt_utilization(const std::vector<RTTask>& tasks) {
    double u = 0.0;
    for (const auto& t : tasks) u += static_cast<double>(t.wcet) / static_cast<double>(t.period);
    return u;
}

/// \brief EDF schedulability test for implicit deadlines: schedulable iff \f$U\le 1\f$.
inline bool edf_schedulable(const std::vector<RTTask>& tasks) {
    return rt_utilization(tasks) <= 1.0 + 1e-9;
}

/// \brief Liu & Layland utilization bound \f$n(2^{1/n}-1)\f$ for \f$n\f$ tasks.
inline double rate_monotonic_bound(std::size_t n) {
    if (n == 0) return 1.0;
    return static_cast<double>(n) * (std::pow(2.0, 1.0 / static_cast<double>(n)) - 1.0);
}

/// \brief RMS sufficient schedulability test: \f$U\le n(2^{1/n}-1)\f$.
///
/// Sufficient but not necessary: a task set can fail this bound yet still be schedulable
/// (confirm with the timeline simulation or exact response-time analysis).
inline bool rate_monotonic_schedulable(const std::vector<RTTask>& tasks) {
    return rt_utilization(tasks) <= rate_monotonic_bound(tasks.size()) + 1e-9;
}

/// \brief RMS priority order: task ids sorted by increasing period (highest priority first).
inline std::vector<std::size_t> rate_monotonic_priorities(const std::vector<RTTask>& tasks) {
    std::vector<std::size_t> order(tasks.size());
    for (std::size_t i = 0; i < tasks.size(); ++i) order[i] = i;
    for (std::size_t a = 0; a < order.size(); ++a)
        for (std::size_t b = a + 1; b < order.size(); ++b)
            if (tasks[order[b]].period < tasks[order[a]].period ||
                (tasks[order[b]].period == tasks[order[a]].period && order[b] < order[a]))
                std::swap(order[a], order[b]);
    std::vector<std::size_t> ids;
    for (std::size_t i : order) ids.push_back(tasks[i].id);
    return ids;
}

namespace detail {

inline void rt_tick(std::vector<RTSlice>& tl, std::size_t id, int t) {
    if (!tl.empty() && tl.back().id == id && tl.back().end == t)
        tl.back().end = t + 1;
    else
        tl.push_back({id, t, t + 1});
}

/// Simulate `tasks` over [0, horizon) at unit granularity. `edf` picks earliest absolute
/// deadline; otherwise fixed priority by shortest period (RMS). Reports any deadline miss.
inline RTScheduleResult rt_simulate(const std::vector<RTTask>& tasks, int horizon, bool edf) {
    const std::size_t n = tasks.size();
    std::vector<int>  remaining(n, 0);  // remaining work of the current job of each task
    std::vector<int>  deadline(n, 0);   // absolute deadline of the current job
    RTScheduleResult  r;
    r.deadline_missed = false;

    for (int t = 0; t < horizon; ++t) {
        for (std::size_t i = 0; i < n; ++i) {
            if (t % tasks[i].period == 0) {                 // new job released
                if (remaining[i] > 0) r.deadline_missed = true;  // previous job unfinished at release
                remaining[i] = tasks[i].wcet;
                deadline[i]  = t + tasks[i].period;
            }
            if (remaining[i] > 0 && t >= deadline[i]) r.deadline_missed = true;
        }
        std::size_t best = n;
        for (std::size_t i = 0; i < n; ++i) {
            if (remaining[i] == 0) continue;
            if (best == n) { best = i; continue; }
            if (edf) {
                if (deadline[i] < deadline[best]) best = i;  // earliest deadline
            } else {
                if (tasks[i].period < tasks[best].period) best = i;  // shortest period = highest priority
            }
        }
        if (best == n) { rt_tick(r.timeline, static_cast<std::size_t>(-1), t); continue; }  // idle
        rt_tick(r.timeline, tasks[best].id, t);
        --remaining[best];
    }
    return r;
}

}  // namespace detail

/// \brief Preemptive EDF timeline over [0, horizon).
inline RTScheduleResult edf_simulate(const std::vector<RTTask>& tasks, int horizon) {
    return detail::rt_simulate(tasks, horizon, /*edf=*/true);
}

/// \brief Preemptive rate-monotonic timeline over [0, horizon).
inline RTScheduleResult rate_monotonic_simulate(const std::vector<RTTask>& tasks, int horizon) {
    return detail::rt_simulate(tasks, horizon, /*edf=*/false);
}

}  // namespace datamunge::algorithms
