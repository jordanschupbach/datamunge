/// @file hits.cpp
/// @brief Shared checked HITS producing kernel and optional actual histories.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/hits.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Internal arithmetic, admission and producing execution helpers.
namespace hits_detail {
/// @brief Add logical work/capture counts with overflow checking.
/// @param a Existing count.
/// @param b Additional units.
/// @return Exact sum.
/// @throws std::length_error If the sum is unrepresentable.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("HITS count overflow");
  return a + b;
}
/// @brief Admit an exact logical count against an optional cap.
/// @param value Complete required units.
/// @param cap Maximum units, zero disabling policy.
/// @throws std::length_error If the cap is exceeded.
void admit(std::size_t value, std::size_t cap) {
  if (cap && value > cap)
    throw std::length_error("HITS policy limit");
}
/// @brief Normalize in the historical sequential arithmetic order.
/// @param values Owned scores updated in place.
/// @param work Actual selected visits accumulated in place.
/// @return Actual Euclidean norm, including zero.
/// @throws std::length_error On counter overflow.
/// @throws std::overflow_error On a nonfinite norm.
double normalize(std::vector<double>& values, HitsWork& work) {
  double norm = 0;
  for (double x : values)
    norm += x * x;
  work.norm_values = add(work.norm_values, values.size());
  norm             = std::sqrt(norm);
  if (!std::isfinite(norm))
    throw std::overflow_error("HITS nonfinite norm");
  if (norm > 0) {
    for (double& x : values)
      x /= norm;
    work.divisions = add(work.divisions, values.size());
  }
  return norm;
}
/// @brief Execute one shared producing kernel, optionally retaining actual states.
/// @param n Complete vertex count.
/// @param edges Borrowed ordered endorsements.
/// @param tolerance Finite positive stopping tolerance.
/// @param limit Complete iteration budget.
/// @param options Input and capture policies.
/// @param trace Optional owned capture destination.
/// @return Actual result, stop branch and work.
/// @throws std::invalid_argument On invalid input domains before policy admission.
/// @throws std::length_error On count, container or policy limits.
/// @throws std::overflow_error On nonfinite producing arithmetic.
/// @throws std::bad_alloc On allocation failure.
HitsRun execute(
    std::size_t                  n,
    const std::vector<HitsEdge>& edges,
    double                       tolerance,
    std::size_t                  limit,
    HitsOptions                  options,
    HitsTrace*                   trace) {
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("HITS endpoint");
  if (!std::isfinite(tolerance) || tolerance <= 0)
    throw std::invalid_argument("HITS tolerance");
  admit(n, options.max_vertices);
  admit(edges.size(), options.max_edges);
  if (n > std::vector<double>{}.max_size())
    throw std::length_error("HITS vector size");
  std::size_t step_units = 0;
  if (trace) {
    const auto initial = add(n, n);
    admit(initial, options.max_trace_entries);
    if (limit)
      step_units = add(1, add(initial, initial));
    trace->captured_entries = initial;
  }
  HitsRun run;
  run.result.hubs.assign(n, n ? 1 / std::sqrt(double(n)) : 0);
  run.result.authorities = run.result.hubs;
  if (trace) {
    trace->initial_hubs        = run.result.hubs;
    trace->initial_authorities = run.result.authorities;
  }
  for (std::size_t k = 0; k < limit; ++k) {
    HitsStep step;
    if (trace) {
      const auto required = add(trace->captured_entries, step_units);
      admit(required, options.max_trace_entries);
      if (trace->steps.size() == trace->steps.max_size())
        throw std::length_error("HITS history size");
      trace->captured_entries = required;
    }
    std::vector<double> a(n), h(n);
    for (auto [u, v] : edges)
      a[v] += run.result.hubs[u];
    run.work.authority_edges = add(run.work.authority_edges, edges.size());
    if (trace)
      step.raw_authorities = a;
    const double an = normalize(a, run.work);
    if (trace)
      step.authorities = a;
    for (auto [u, v] : edges)
      h[u] += a[v];
    run.work.hub_edges = add(run.work.hub_edges, edges.size());
    if (trace)
      step.raw_hubs = h;
    const double hn    = normalize(h, run.work);
    double       error = 0;
    for (std::size_t i = 0; i < n; ++i)
      error = std::max(
          error,
          std::max(
              std::abs(a[i] - run.result.authorities[i]), std::abs(h[i] - run.result.hubs[i])));
    run.work.change_vertices = add(run.work.change_vertices, n);
    run.result.authorities.swap(a);
    run.result.hubs.swap(h);
    run.result.iterations = k + 1;
    run.last_change       = error;
    run.has_step          = true;
    if (trace) {
      step.iteration      = k + 1;
      step.hubs           = run.result.hubs;
      step.authority_norm = an;
      step.hub_norm       = hn;
      step.change         = error;
      trace->steps.push_back(std::move(step));
    }
    if (error < tolerance) {
      run.stop = HitsStop::Tolerance;
      break;
    }
  }
  return run;
}
} // namespace hits_detail
HITSResult
hits(std::size_t n, const std::vector<HitsEdge>& edges, double tolerance, std::size_t limit) {
  return hits_detail::execute(n, edges, tolerance, limit, {0, 0, 0}, nullptr).result;
}
HitsRun hits_run(
    std::size_t                  n,
    const std::vector<HitsEdge>& edges,
    double                       tolerance,
    std::size_t                  limit,
    HitsOptions                  options) {
  return hits_detail::execute(n, edges, tolerance, limit, options, nullptr);
}
HitsTrace hits_trace(
    std::size_t                  n,
    const std::vector<HitsEdge>& edges,
    double                       tolerance,
    std::size_t                  limit,
    HitsOptions                  options) {
  HitsTrace trace;
  trace.run = hits_detail::execute(n, edges, tolerance, limit, options, &trace);
  return trace;
}
} // namespace datamunge::algorithms
