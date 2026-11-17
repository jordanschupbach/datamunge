/// @file reference_counting.cpp
/// @brief One checked actual-site count initialization and LIFO reclamation producer.
#include <datamunge/algorithms/reference_counting.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical heap simulation implementation.
namespace datamunge::algorithms {
/// @brief Private reference-count producer.
namespace {
/// @brief Execute one actual checked count cascade with optional mutation capture.
/// @param graph Complete borrowed graph.
/// @param o Input/work/history policies.
/// @param capture Retain real count events.
/// @return Owned initial/final counts and actual cascade result.
RefCountRun solve(const ObjectGraph& graph, const RefCountOptions& o, bool capture) {
  validate_object_graph(graph, false);
  const auto n = static_cast<std::size_t>(graph.num_objects);
  if (o.max_objects && n > o.max_objects)
    throw std::length_error("reference counting: input policy");
  RefCountRun      out;
  auto&            r = out.result;
  std::vector<int> counts(n, 0), pending;
  r.freed.assign(n, 0);
  /// @brief Admit one selected producer event before executing it.
  const auto tick = [&]() {
    if (out.operations == std::numeric_limits<std::size_t>::max() ||
        (o.max_operations && out.operations >= o.max_operations))
      throw std::length_error("reference counting: work policy");
    ++out.operations;
  };
  /// @brief Admit a real mutation record before its state change.
  const auto admit = [&]() {
    if (capture && o.max_trace_entries && out.events.size() >= o.max_trace_entries)
      throw std::length_error("reference counting: history policy");
  };
  /// @brief Apply one actual root/field incoming-reference increment.
  /// @param source Referring object or -1 for a root.
  /// @param target Referenced object id.
  /// @param kind Root/field mutation kind.
  const auto increment = [&](int source, int target, RefCountEventKind kind) {
    tick();
    admit();
    const int before = counts[target];
    if (before == std::numeric_limits<int>::max())
      throw std::length_error("reference counting: count capacity");
    ++counts[target];
    ++out.increments;
    if (capture)
      out.events.push_back({kind, source, target, before, counts[target], 0});
  };
  for (int root : graph.roots)
    increment(-1, root, RefCountEventKind::Root);
  for (int i = 0; i < graph.num_objects; ++i)
    for (int child : graph.refs[i])
      increment(i, child, RefCountEventKind::Field);
  r.refcount = counts;
  for (int i = 0; i < graph.num_objects; ++i) {
    tick();
    if (counts[i] == 0)
      pending.push_back(i);
  }
  while (!pending.empty()) {
    tick();
    const int object = pending.back();
    pending.pop_back();
    if (r.freed[object])
      continue;
    admit();
    r.freed[object] = 1;
    out.freed_order.push_back(object);
    if (capture)
      out.events.push_back(
          {RefCountEventKind::Free,
           object,
           object,
           counts[object],
           counts[object],
           pending.size()});
    for (int child : graph.refs[object]) {
      tick();
      admit();
      const int before = counts[child];
      if (before <= 0)
        throw std::logic_error("reference counting: inconsistent release");
      --counts[child];
      ++out.decrements;
      if (counts[child] == 0 && !r.freed[child])
        pending.push_back(child);
      if (capture)
        out.events.push_back(
            {RefCountEventKind::Release, object, child, before, counts[child], pending.size()});
    }
  }
  const auto live = reachable_from_roots(graph);
  for (int i = 0; i < graph.num_objects; ++i) {
    tick();
    if (!live[i] && !r.freed[i])
      r.leaked.push_back(i);
  }
  out.final_counts = std::move(counts);
  return out;
}
} // namespace
RefCountRun reference_counting_run(const ObjectGraph& g, const RefCountOptions& o) {
  return solve(g, o, false);
}
RefCountRun reference_counting_trace(const ObjectGraph& g, const RefCountOptions& o) {
  return solve(g, o, true);
}
RefCountResult reference_counting_collect(const ObjectGraph& g) {
  return reference_counting_run(g).result;
}
ObjectGraph reference_count_chain(std::size_t n, bool rooted, std::size_t max_objects) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (max_objects && n > max_objects))
    throw std::length_error("reference_count_chain: input policy");
  ObjectGraph graph;
  graph.num_objects = static_cast<int>(n);
  graph.refs.resize(n);
  for (std::size_t i = 1; i < n; ++i)
    graph.refs[i - 1].push_back(static_cast<int>(i));
  if (rooted && n)
    graph.roots.push_back(0);
  return graph;
}
} // namespace datamunge::algorithms
