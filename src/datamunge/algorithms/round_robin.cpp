/// @file round_robin.cpp
/// @brief One validated actual FIFO execution stream and native sparse-id Gantt renderer.
#include <algorithm>
#include <datamunge/algorithms/round_robin.hpp>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
/// @brief Canonical round-robin CPU implementation.
namespace datamunge::algorithms {
/// @brief Private synchronous FIFO execution.
namespace {
/// @brief Validate all requests then execute actual arrivals, dispatches and busy ticks once.
/// @param processes Borrowed workload.
/// @param quantum Positive slice bound.
/// @param o Policies.
/// @param capture Retain queue histories.
/// @return Owned actual execution result.
RoundRobinRun solve(
    const std::vector<SchedProcess>& processes,
    int                              quantum,
    const RoundRobinOptions&         o,
    bool                             capture) {
  if (quantum <= 0)
    throw std::invalid_argument("round robin: quantum");
  const auto n = processes.size(), max = std::numeric_limits<std::size_t>::max();
  if (o.max_processes && n > o.max_processes)
    throw std::length_error("round robin: population policy");
  std::unordered_set<std::size_t> ids;
  std::size_t                     total = 0, completed = 0;
  std::vector<int>                remaining(n), completion(n);
  std::vector<char>               enqueued(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& p = processes[i];
    if (p.arrival < 0 || p.burst < 0 || !ids.insert(p.id).second)
      throw std::invalid_argument("round robin: process");
    if (static_cast<std::size_t>(p.burst) > max - total)
      throw std::length_error("round robin: aggregate capacity");
    total += p.burst;
    if (o.max_busy_ticks && total > o.max_busy_ticks)
      throw std::length_error("round robin: complete busy policy");
    remaining[i] = p.burst;
    if (p.burst == 0) {
      completion[i] = p.arrival;
      enqueued[i]   = 1;
      ++completed;
    }
  }
  RoundRobinRun           out;
  std::deque<std::size_t> ready;
  int                     time     = 0;
  std::size_t             retained = 0;
  /// @brief Admit every newly ready positive-work process in caller order.
  const auto arrivals = [&]() {
    for (std::size_t i = 0; i < n; ++i) {
      if (out.arrival_checks == max ||
          (o.max_arrival_checks && out.arrival_checks >= o.max_arrival_checks))
        throw std::length_error("round robin: arrival work policy");
      ++out.arrival_checks;
      if (!enqueued[i] && processes[i].arrival <= time && remaining[i] > 0) {
        ready.push_back(i);
        enqueued[i] = 1;
      }
    }
  };
  arrivals();
  while (completed < n) {
    if (ready.empty()) {
      int next = std::numeric_limits<int>::max();
      for (std::size_t i = 0; i < n; ++i)
        if (!enqueued[i] && remaining[i] > 0)
          next = std::min(next, processes[i].arrival);
      if (next <= time)
        throw std::logic_error("round robin: admission invariant");
      time = next;
      ++out.idle_jumps;
      arrivals();
      continue;
    }
    const auto     id = ready.front();
    RoundRobinStep event;
    event.process_index    = id;
    event.start            = time;
    event.remaining_before = remaining[id];
    if (capture)
      event.ready_before.assign(ready.begin(), ready.end());
    ready.pop_front();
    ++out.dispatches;
    const int slice = std::min(quantum, remaining[id]);
    if (time > std::numeric_limits<int>::max() - slice)
      throw std::overflow_error("round robin: completion time");
    for (int tick = 0; tick < slice; ++tick) {
      auto& gantt = out.result.gantt;
      if (!gantt.empty() && gantt.back().id == processes[id].id && gantt.back().end == time)
        ++gantt.back().end;
      else
        gantt.push_back({processes[id].id, time, time + 1});
      ++time;
      --remaining[id];
      ++out.busy_ticks;
      arrivals();
    }
    if (remaining[id] == 0) {
      completion[id] = time;
      ++completed;
    } else {
      ready.push_back(id);
      ++out.preemptions;
    }
    if (capture) {
      event.end             = time;
      event.remaining_after = remaining[id];
      if (ready.size() == max || event.ready_before.size() > max - ready.size() - 1)
        throw std::length_error("round robin: queue history capacity");
      const auto cost = event.ready_before.size() + ready.size() + 1;
      if (cost > max - retained || (o.max_trace_entries && (retained > o.max_trace_entries ||
                                                            cost > o.max_trace_entries - retained)))
        throw std::length_error("round robin: history policy");
      retained += cost;
      event.ready_after.assign(ready.begin(), ready.end());
      out.steps.push_back(std::move(event));
    }
  }
  long double waiting = 0, turnaround = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& p    = processes[i];
    const int   turn = completion[i] - p.arrival, wait = turn - p.burst;
    out.result.metrics.push_back({p.id, completion[i], turn, wait});
    waiting += wait;
    turnaround += turn;
  }
  if (n) {
    out.result.avg_waiting    = static_cast<double>(waiting / n);
    out.result.avg_turnaround = static_cast<double>(turnaround / n);
  }
  return out;
}
} // namespace
RoundRobinRun
round_robin_run(const std::vector<SchedProcess>& p, int q, const RoundRobinOptions& o) {
  return solve(p, q, o, false);
}
RoundRobinRun
round_robin_trace(const std::vector<SchedProcess>& p, int q, const RoundRobinOptions& o) {
  return solve(p, q, o, true);
}
ScheduleResult round_robin(const std::vector<SchedProcess>& p, int q) {
  return solve(p, q, {}, false).result;
}
graphics::RPlot plot_round_robin_schedule(
    const std::vector<SchedProcess>& processes,
    const ScheduleResult&            schedule,
    std::size_t                      max_segments) {
  if (max_segments && schedule.gantt.size() > max_segments)
    throw std::length_error("round robin plot: segment policy");
  std::unordered_map<std::size_t, std::size_t> rows;
  std::vector<double>                          ticks;
  std::vector<std::string>                     labels;
  for (std::size_t i = 0; i < processes.size(); ++i) {
    if (!rows.emplace(processes[i].id, i).second)
      throw std::invalid_argument("round robin plot: duplicate id");
    ticks.push_back(static_cast<double>(i));
    labels.push_back(std::to_string(processes[i].id));
  }
  auto                             plot = graphics::RPlot::create();
  int                              end  = 0;
  const std::vector<graphics::RGB> palette{
      {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {202, 138, 4}, {147, 51, 234}};
  for (const auto& s : schedule.gantt) {
    const auto found = rows.find(s.id);
    if (found == rows.end() || s.start < end || s.start < 0 || s.end <= s.start)
      throw std::invalid_argument("round robin plot: interval/id");
    end              = s.end;
    const double row = found->second;
    plot.polygon(
        {double(s.start), double(s.end), double(s.end), double(s.start)},
        {row - .35, row - .35, row + .35, row + .35},
        palette[found->second % palette.size()],
        true);
  }
  plot.x_limits(0, std::max(1, end))
      .y_ticks(ticks, labels)
      .x_label("integral time boundary")
      .y_label("external process id (input order)")
      .title("Actual round-robin busy timeline");
  if (!processes.empty())
    plot.y_limits(-.6, processes.size() - .4);
  return plot;
}
} // namespace datamunge::algorithms
