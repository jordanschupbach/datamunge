/// @file hungarian.cpp
/// @brief Shared checked square primal-dual assignment kernel and actual capture.
#include <cmath>
#include <datamunge/algorithms/hungarian.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked arithmetic/admission and shared execution implementation.
namespace {
/// @brief Reject a nonfinite computed scalar while preserving its ordinary value.
/// @param value Produced arithmetic value.
/// @return Unchanged finite value.
/// @throws std::overflow_error If value is not finite.
double finite(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Hungarian nonfinite producing arithmetic");
  return value;
}
/// @brief Increment one selected work count without unsigned wrap.
/// @param count Mutable counter.
/// @throws std::length_error On exhausted size_t.
void increment(std::size_t& count) {
  if (count == std::numeric_limits<std::size_t>::max())
    throw std::length_error("Hungarian work count");
  ++count;
}
/// @brief Validate complete square finite input and its optional entry cap.
/// @param cost Borrowed matrix.
/// @param options Admission policies.
/// @throws std::invalid_argument On shape/domain errors.
/// @throws std::length_error On dimension overflow or cell cap violation.
void validate(const std::vector<std::vector<double>>& cost, AssignmentOptions options) {
  const auto n   = cost.size();
  const auto max = std::numeric_limits<std::size_t>::max();
  if (!n)
    throw std::invalid_argument("hungarian: nonempty cost matrix required");
  if (n == max || n > max / n)
    throw std::length_error("Hungarian dimensions");
  if (options.max_input_cells && n * n > options.max_input_cells)
    throw std::length_error("Hungarian complete input cap");
  for (const auto& row : cost) {
    if (row.size() != n)
      throw std::invalid_argument("hungarian: square cost matrix required");
    for (double c : row)
      if (!std::isfinite(c))
        throw std::invalid_argument("hungarian: finite costs required");
  }
}
/// @brief Produce ordinary assignment and optional actual boundary histories.
/// Input has already passed complete validation/admission. Arithmetic order and
/// strict ascending-column tie choices follow the legacy kernel. No callbacks,
/// replay or observation-only recomputation of slacks/potentials are used.
/// @param cost Valid borrowed complete matrix.
/// @param options History policy; input was admitted by caller.
/// @param events Optional owned destination, initially empty; null disables capture.
/// @return Owned final assignment/duals/work.
/// @throws std::length_error On work/capture count overflow or complete event cap.
/// @throws std::overflow_error On nonfinite producing arithmetic/frontier.
/// @throws std::bad_alloc On allocation failure.
AssignmentRun execute(
    const std::vector<std::vector<double>>& cost,
    AssignmentOptions                       options,
    std::vector<AssignmentEvent>*           events) {
  const auto               n   = cost.size();
  constexpr double         inf = std::numeric_limits<double>::infinity();
  std::vector<double>      u(n + 1, 0), v(n + 1, 0);
  std::vector<std::size_t> p(n + 1, 0), way(n + 1, 0);
  AssignmentRun            out;
  std::size_t              units = 0;
  // Admission is checked before every producing transition; 0 disables only the optional cap.
  const auto admit = [&]() {
    if (!events)
      return;
    const auto max = std::numeric_limits<std::size_t>::max();
    if (n > (max - 1) / 4)
      throw std::length_error("Hungarian event dimensions");
    const auto next = 1 + 4 * n;
    if (units > max - next ||
        (options.max_history_entries && next > options.max_history_entries - units))
      throw std::length_error("Hungarian complete history cap");
    units += next;
  };
  // Copy actual arrays at a boundary, excluding private sentinel slots.
  const auto capture = [&](AssignmentEventKind               kind,
                           std::size_t                       introduced,
                           std::size_t                       expanded,
                           std::size_t                       column,
                           double                            delta,
                           const std::vector<unsigned char>& used) {
    if (!events)
      return;
    AssignmentEvent e;
    e.kind           = kind;
    e.introduced_row = introduced - 1;
    e.expanded_row   = expanded - 1;
    e.next_column    = column - 1;
    e.delta          = delta;
    e.row_potentials.assign(u.begin() + 1, u.end());
    e.column_potentials.assign(v.begin() + 1, v.end());
    e.reached_columns.assign(used.begin() + 1, used.end());
    e.column_to_row.reserve(n);
    for (std::size_t j = 1; j <= n; ++j)
      e.column_to_row.push_back(p[j] ? p[j] - 1 : kAssignmentUnmatched);
    events->push_back(std::move(e));
  };
  for (std::size_t i = 1; i <= n; ++i) {
    p[0]                          = i;
    std::size_t                j0 = 0, last_row = i;
    std::vector<double>        minv(n + 1, inf);
    std::vector<unsigned char> used(n + 1, 0);
    do {
      admit();
      used[j0]          = 1;
      const auto i0     = p[j0];
      last_row          = i0;
      double      delta = inf;
      std::size_t j1    = 0;
      for (std::size_t j = 1; j <= n; ++j) {
        if (used[j])
          continue;
        increment(out.work.slack_evaluations);
        const double cur = finite(finite(cost[i0 - 1][j - 1] - u[i0]) - v[j]);
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j]  = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1    = j;
        }
      }
      if (!j1 || !std::isfinite(delta))
        throw std::overflow_error("Hungarian no finite frontier");
      increment(out.work.reprices);
      for (std::size_t j = 0; j <= n; ++j) {
        if (used[j]) {
          increment(out.work.potential_pairs);
          u[p[j]] = finite(u[p[j]] + delta);
          v[j]    = finite(v[j] - delta);
        } else
          minv[j] = finite(minv[j] - delta);
      }
      capture(AssignmentEventKind::Reprice, i, i0, j1, delta, used);
      j0 = j1;
    } while (p[j0] != 0);
    admit();
    const auto endpoint = j0;
    do {
      const auto j1 = way[j0];
      increment(out.work.path_writes);
      p[j0] = p[j1];
      j0    = j1;
    } while (j0 != 0);
    capture(AssignmentEventKind::Augment, i, last_row, endpoint, 0, used);
  }
  out.result.assignment.assign(n, 0);
  for (std::size_t j = 1; j <= n; ++j)
    out.result.assignment[p[j] - 1] = j - 1;
  for (std::size_t i = 0; i < n; ++i)
    out.result.cost = finite(out.result.cost + cost[i][out.result.assignment[i]]);
  out.row_potentials.assign(u.begin() + 1, u.end());
  out.column_potentials.assign(v.begin() + 1, v.end());
  return out;
}
} // namespace
AssignmentResult hungarian(const std::vector<std::vector<double>>& cost) {
  validate(cost, {0, 0});
  return execute(cost, {0, 0}, nullptr).result;
}
AssignmentRun hungarian_run(
    const std::vector<std::vector<double>>& cost,
    AssignmentOptions                       options) {
  validate(cost, options);
  return execute(cost, options, nullptr);
}
AssignmentTrace hungarian_trace(
    const std::vector<std::vector<double>>& cost,
    AssignmentOptions                       options) {
  validate(cost, options);
  AssignmentTrace out;
  out.costs = cost;
  out.run   = execute(cost, options, &out.events);
  return out;
}
} // namespace datamunge::algorithms
