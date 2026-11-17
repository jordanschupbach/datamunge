/// @file nonblocking_switch.cpp
/// @brief Canonical original-order bipartite two-color routing and actual BFS histories.
#include <array>
#include <datamunge/algorithms/nonblocking_switch.hpp>
#include <limits>
#include <queue>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete projected admission and shared incidence/coloring producer.
namespace {
/// @brief Add geometry/history counts without wrapping.
/// @param a First count.
/// @param b Second count.
/// @return Checked sum.
/// @throws std::length_error If sum overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("switch routes: count overflow");
  return a + b;
}
/// @brief Owned original-order input/output incidence lists after complete domain admission.
struct Incidences {
  std::vector<std::vector<std::size_t>> input;  ///< Original call IDs per input group.
  std::vector<std::vector<std::size_t>> output; ///< Original call IDs per output group.
};
/// @brief Check all endpoints, group/call geometry and optional caps, then every group degree.
/// @param ni Complete input group count.
/// @param no Complete output group count.
/// @param calls Complete original endpoint pairs.
/// @param options Projected group/call policies, capture ignored.
/// @return Owned original incidence lists.
/// @throws std::invalid_argument If any endpoint/degree is invalid.
/// @throws std::length_error If geometry or group/call policies are exceeded.
Incidences incidences(
    std::size_t                                             ni,
    std::size_t                                             no,
    const std::vector<std::pair<std::size_t, std::size_t>>& calls,
    SwitchRouteOptions                                      options) {
  for (auto [u, v] : calls)
    if (u >= ni || v >= no)
      throw std::invalid_argument("switch routes: endpoint out of range");
  auto groups = add(ni, no);
  if (ni > std::vector<std::vector<std::size_t>>{}.max_size() ||
      no > std::vector<std::vector<std::size_t>>{}.max_size() ||
      calls.size() > std::vector<std::size_t>{}.max_size())
    throw std::length_error("switch routes: container geometry");
  if ((options.max_groups && groups > options.max_groups) ||
      (options.max_calls && calls.size() > options.max_calls))
    throw std::length_error("switch routes: complete group/call cap");
  Incidences rows;
  rows.input.resize(ni);
  rows.output.resize(no);
  for (std::size_t i = 0; i < calls.size(); ++i) {
    auto [u, v] = calls[i];
    rows.input[u].push_back(i);
    rows.output[v].push_back(i);
    if (rows.input[u].size() > 2 || rows.output[v].size() > 2)
      throw std::invalid_argument("switch routes: endpoint degree exceeds two");
  }
  return rows;
}
/// @brief Execute the original ordered BFS producer and capture only actual assignments/scans.
/// @param ni Complete input groups.
/// @param no Complete output groups.
/// @param calls Complete original calls.
/// @param options Complete projected/history policies.
/// @param out Optional actual-history sink.
/// @return Actual route choices and producing work.
/// @throws std::invalid_argument If complete endpoint/degree domain fails.
/// @throws std::length_error If geometry/policies are exceeded.
SwitchRouteAnalysis run(
    std::size_t                                             ni,
    std::size_t                                             no,
    const std::vector<std::pair<std::size_t, std::size_t>>& calls,
    SwitchRouteOptions                                      options,
    SwitchRouteTrace*                                       out) {
  auto       rows = incidences(ni, no, calls, options);
  const auto n    = calls.size();
  if (out) {
    std::size_t reads = 0;
    for (auto [u, v] : calls)
      reads = add(reads, add(rows.input[u].size(), rows.output[v].size()));
    auto count = add(2, reads);
    for (int i = 0; i < 4; ++i)
      count = add(count, n);
    if (options.max_trace_entries && count > options.max_trace_entries)
      throw std::length_error("switch routes: complete capture cap");
    out->input_groups     = ni;
    out->output_groups    = no;
    out->connections      = calls;
    out->captured_entries = count;
    out->assignments.reserve(n);
    out->scans.reserve(reads);
  }
  SwitchRouteAnalysis a;
  a.routes.assign(n, 2);
  for (std::size_t start = 0; start < n; ++start)
    if (a.routes[start] == 2) {
      auto component  = a.components++;
      a.routes[start] = 0;
      ++a.assignments;
      if (out)
        out->assignments.push_back({start, 0, component, n, false, true});
      std::queue<std::size_t> q;
      q.push(start);
      while (!q.empty()) {
        auto e = q.front();
        q.pop();
        ++a.queue_pops;
        auto [u, v] = calls[e];
        /// @brief Visit actual input then output incidence candidates in their original order.
        auto scan = [&](const std::vector<std::size_t>& list, bool input) {
          for (auto f : list) {
            ++a.incidence_reads;
            auto before   = a.routes[f];
            bool assigned = before == 2;
            if (assigned) {
              a.routes[f] = 1 - a.routes[e];
              q.push(f);
              ++a.assignments;
              if (out)
                out->assignments.push_back({f, a.routes[f], component, e, input, false});
            } else if (f != e && a.routes[f] == a.routes[e])
              throw std::invalid_argument("switch routes: conflicting route");
            if (out)
              out->scans.push_back({e, f, input, before, assigned});
          }
        };
        scan(rows.input[u], true);
        scan(rows.output[v], false);
      }
    }
  return a;
}
} // namespace
std::vector<std::size_t> nonblocking_switch_routes(
    std::size_t                                             i,
    std::size_t                                             o,
    const std::vector<std::pair<std::size_t, std::size_t>>& c) {
  return run(i, o, c, {0, 0, 0}, nullptr).routes;
}
SwitchRouteAnalysis nonblocking_switch_analyze(
    std::size_t                                             i,
    std::size_t                                             o,
    const std::vector<std::pair<std::size_t, std::size_t>>& c,
    SwitchRouteOptions                                      options) {
  return run(i, o, c, options, nullptr);
}
SwitchRouteTrace nonblocking_switch_trace(
    std::size_t                                             i,
    std::size_t                                             o,
    const std::vector<std::pair<std::size_t, std::size_t>>& c,
    SwitchRouteOptions                                      options) {
  SwitchRouteTrace t;
  t.analysis = run(i, o, c, options, &t);
  return t;
}
bool nonblocking_switch_feasible(
    std::size_t                                             ni,
    std::size_t                                             no,
    const std::vector<std::pair<std::size_t, std::size_t>>& c,
    const std::vector<std::size_t>&                         r) {
  incidences(ni, no, c, {0, 0, 0});
  if (r.size() != c.size())
    return false;
  std::vector<std::array<bool, 2>> in(ni), out(no);
  for (std::size_t e = 0; e < c.size(); ++e) {
    auto [u, v] = c[e];
    auto color  = r[e];
    if (color > 1 || in[u][color] || out[v][color])
      return false;
    in[u][color]  = true;
    out[v][color] = true;
  }
  return true;
}
std::vector<std::pair<std::size_t, std::size_t>> nonblocking_switch_demo_calls() {
  return {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
}
std::vector<std::pair<std::size_t, std::size_t>> nonblocking_switch_cycle_calls(std::size_t n) {
  if (n > 1000000)
    throw std::length_error("switch cycle fixture: group cap");
  std::vector<std::pair<std::size_t, std::size_t>> c;
  for (std::size_t i = 0; i < n; ++i) {
    c.emplace_back(i, i);
    c.emplace_back(i, (i + 1) % n);
  }
  return c;
}
} // namespace datamunge::algorithms
