/// @file object_liveness.cpp
/// @brief Shared canonical complete-guarded mark-on-push traversal and actual root/pop/reference records.
#include <algorithm>
#include <datamunge/algorithms/object_liveness.hpp>
namespace datamunge::algorithms {
/// @brief Private complete-capture arithmetic and shared iterative producer.
namespace {
/// @brief Add nonnegative logical counts without overflow.
/// @param a Current count.
/// @param b Additional count.
/// @return Exact representable sum.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Object liveness: count overflow");
  return a + b;
}
/// @brief Admit actual graph/event slots before their optional capture copy.
/// @param trace Optional actual owned output.
/// @param amount Actual new logical slots.
/// @param quota Optional logical capture cap.
void admit(ObjectLivenessTrace* trace, std::size_t amount, std::size_t quota) {
  if (!trace)
    return;
  trace->captured_entries = add(trace->captured_entries, amount);
  if (quota && trace->captured_entries > quota)
    throw std::length_error("Object liveness: capture quota");
}
/// @brief Mark actual roots/fields once, preserving canonical push and LIFO pop order.
/// @param graph Complete original graph, size metadata ignored semantically.
/// @param trace Optional actual owned graph/event output.
/// @param quota Optional selected capture cap.
/// @return Actual dense marks and selected validation/traversal work.
ObjectLivenessAnalysis
produce(const ObjectGraph& graph, ObjectLivenessTrace* trace, std::size_t quota) {
  ObjectLivenessAnalysis out;
  out.work.validation = validate_object_graph(graph);
  const auto n        = std::size_t(graph.num_objects);
  if (trace) {
    const auto base =
        add(add(add(n, n), out.work.validation.references),
            add(graph.roots.size(), graph.sizes.size()));
    admit(trace, base, quota);
    trace->graph = graph;
  }
  out.marked.assign(n, 0);
  out.work.initialized = n;
  std::vector<int> stack;
  auto             visit = [&](ObjectMarkEventKind kind, int object, int parent, std::size_t slot) {
    const auto before = stack.size();
    const bool marked = out.marked[object] != 0;
    if (!marked) {
      out.marked[object] = 1;
      ++out.work.marks;
      stack.push_back(object);
      ++out.work.pushes;
      out.work.maximum_stack = std::max(out.work.maximum_stack, stack.size());
    }
    if (trace) {
      admit(trace, 1, quota);
      trace->events.push_back({kind, object, parent, slot, marked, !marked, before, stack.size()});
    }
  };
  for (std::size_t slot = 0; slot < graph.roots.size(); ++slot) {
    ++out.work.roots;
    visit(ObjectMarkEventKind::Root, graph.roots[slot], -1, slot);
  }
  while (!stack.empty()) {
    const auto before = stack.size();
    const int  object = stack.back();
    stack.pop_back();
    ++out.work.pops;
    if (trace) {
      admit(trace, 1, quota);
      trace->events.push_back(
          {ObjectMarkEventKind::Pop,
           object,
           -1,
           std::numeric_limits<std::size_t>::max(),
           true,
           false,
           before,
           stack.size()});
    }
    for (std::size_t slot = 0; slot < graph.refs[object].size(); ++slot) {
      ++out.work.references;
      visit(ObjectMarkEventKind::Reference, graph.refs[object][slot], object, slot);
    }
  }
  return out;
}
} // namespace
ObjectLivenessAnalysis object_liveness_analyze(const ObjectGraph& graph) {
  return produce(graph, nullptr, 0);
}
ObjectLivenessTrace object_liveness_trace(const ObjectGraph& graph, std::size_t max_entries) {
  ObjectLivenessTrace t;
  t.analysis = produce(graph, &t, max_entries);
  return t;
}
std::vector<char> reachable_from_roots(const ObjectGraph& graph) {
  return produce(graph, nullptr, 0).marked;
}
} // namespace datamunge::algorithms
