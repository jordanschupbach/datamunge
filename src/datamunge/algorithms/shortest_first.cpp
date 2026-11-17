/// @file shortest_first.cpp
/// @brief Same-execution shortest scheduling with event jumps, validation and actual work.
#include <algorithm>
#include <datamunge/algorithms/shortest_first.hpp>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Execute checked shortest-first event decisions.
/// @param p Complete immutable rows.
/// @param o Policies.
/// @param preempt Reconsider at arrivals.
/// @param capture Retain decisions.
/// @return Owned actual schedule.
ShortestFirstRun
run(const std::vector<SchedProcess>& p, const ShortestFirstOptions& o, bool preempt, bool capture) {
  const auto n = p.size();
  if (o.max_processes && n > o.max_processes)
    throw std::length_error("shortest: processes");
  std::set<std::size_t> ids;
  long long             horizon = 0, latest = 0;
  std::vector<int>      remaining, completion(n);
  for (const auto& x : p) {
    if (x.arrival < 0 || x.burst <= 0 || !ids.insert(x.id).second)
      throw std::invalid_argument("shortest: process rows");
    latest = std::max(latest, static_cast<long long>(x.arrival));
    horizon += x.burst;
    if (horizon > std::numeric_limits<int>::max())
      throw std::length_error("shortest: time capacity");
    remaining.push_back(x.burst);
  }
  if (horizon + latest > std::numeric_limits<int>::max())
    throw std::length_error("shortest: horizon");
  ShortestFirstRun r;
  int              t    = 0;
  std::size_t      done = 0, previous = n, retained = 0;
  while (done < n) {
    std::size_t best = n;
    int         next = std::numeric_limits<int>::max();
    for (std::size_t i = 0; i < n; ++i) {
      if (r.scans == std::numeric_limits<std::size_t>::max() ||
          (o.max_comparisons && r.scans >= o.max_comparisons))
        throw std::length_error("shortest: scans");
      ++r.scans;
      if (!remaining[i])
        continue;
      if (p[i].arrival > t) {
        next = std::min(next, p[i].arrival);
        continue;
      }
      if (best == n || remaining[i] < remaining[best] ||
          (remaining[i] == remaining[best] && p[i].arrival < p[best].arrival))
        best = i;
    }
    const bool idle = best == n;
    int        end  = idle ? next : t + remaining[best];
    if (preempt && !idle)
      end = std::min(end, next);
    if (capture) {
      if (n == std::numeric_limits<std::size_t>::max() ||
          retained > std::numeric_limits<std::size_t>::max() - n - 1)
        throw std::length_error("shortest: history capacity");
      retained += n + 1;
      if (o.max_records && retained > o.max_records)
        throw std::length_error("shortest: history");
      r.decisions.push_back({t, end, idle, idle ? 0 : p[best].id, remaining});
    }
    if (idle) {
      t        = end;
      previous = n;
      continue;
    }
    if (previous != n && previous != best && remaining[previous])
      ++r.preemptions;
    ++r.dispatches;
    auto& g = r.schedule.gantt;
    if (!g.empty() && g.back().id == p[best].id && g.back().end == t)
      g.back().end = end;
    else
      g.push_back({p[best].id, t, end});
    remaining[best] -= end - t;
    t = end;
    if (!remaining[best]) {
      completion[best] = t;
      ++done;
    }
    previous = best;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const int turn = completion[i] - p[i].arrival, wait = turn - p[i].burst;
    r.schedule.metrics.push_back({p[i].id, completion[i], turn, wait});
    r.schedule.avg_waiting += static_cast<double>(wait);
    r.schedule.avg_turnaround += static_cast<double>(turn);
  }
  if (n) {
    r.schedule.avg_waiting /= static_cast<double>(n);
    r.schedule.avg_turnaround /= static_cast<double>(n);
  }
  return r;
}
} // namespace
ShortestFirstRun shortest_job_next_run(
    const std::vector<SchedProcess>& p,
    const ShortestFirstOptions&      o) {
  return run(p, o, false, false);
}
ShortestFirstRun shortest_job_next_trace(
    const std::vector<SchedProcess>& p,
    const ShortestFirstOptions&      o) {
  return run(p, o, false, true);
}
ShortestFirstRun shortest_remaining_time_run(
    const std::vector<SchedProcess>& p,
    const ShortestFirstOptions&      o) {
  return run(p, o, true, false);
}
ShortestFirstRun shortest_remaining_time_trace(
    const std::vector<SchedProcess>& p,
    const ShortestFirstOptions&      o) {
  return run(p, o, true, true);
}
ScheduleResult shortest_job_next(const std::vector<SchedProcess>& p) {
  return shortest_job_next_run(p).schedule;
}
ScheduleResult shortest_remaining_time(const std::vector<SchedProcess>& p) {
  return shortest_remaining_time_run(p).schedule;
}
} // namespace datamunge::algorithms
