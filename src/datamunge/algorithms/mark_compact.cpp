/// @file mark_compact.cpp
/// @brief Shared actual stable unit-slot forwarding and field/root rewrites after one canonical marking pass.
#include <datamunge/algorithms/mark_compact.hpp>
namespace datamunge::algorithms {
/// @brief Private actual capture admission and shared stable forwarding/rewrite producer.
namespace {
/// @brief Admit one actual phase record before copying it into collector history.
/// @param trace Optional actual owned collector output.
/// @param quota Optional logical capture cap.
void admit(MarkCompactTrace* trace, std::size_t quota) {
  if (!trace)
    return;
  if (trace->captured_entries == std::numeric_limits<std::size_t>::max() ||
      (quota && trace->captured_entries >= quota))
    throw std::length_error("Mark compact: capture quota");
  ++trace->captured_entries;
}
/// @brief Forward old slots in order, then translate every real live field and original root through that complete map.
/// @param graph Complete original graph, already structurally validated by its actual marking producer.
/// @param marks Actual root-reachable marks from that same execution.
/// @param marking Actual selected full-validation/live-traversal work.
/// @param trace Optional actual forwarding/field/root history.
/// @param quota Optional logical capture cap.
/// @return Actual stable unit-slot result, translated roots and selected phase work.
MarkCompactAnalysis compact(
    const ObjectGraph&        graph,
    const std::vector<char>&  marks,
    const ObjectLivenessWork& marking,
    MarkCompactTrace*         trace,
    std::size_t               quota) {
  MarkCompactAnalysis out;
  out.result.marked = marks;
  out.work.marking  = marking;
  out.result.new_address.assign(marks.size(), -1);
  for (std::size_t i = 0; i < marks.size(); ++i) {
    const auto before = out.result.order.size();
    const bool live   = marks[i] != 0;
    ++out.work.scanned;
    if (live) {
      out.result.new_address[i] = int(before);
      out.result.order.push_back(int(i));
      ++out.work.forwarded;
    }
    if (trace) {
      admit(trace, quota);
      trace->forwarding.push_back(
          {int(i), live, before, out.result.new_address[i], out.result.order.size()});
    }
  }
  for (int owner : out.result.order) {
    std::vector<int> fields;
    fields.reserve(graph.refs[owner].size());
    for (std::size_t field = 0; field < graph.refs[owner].size(); ++field) {
      const int target = graph.refs[owner][field];
      ++out.work.references;
      const int translated = out.result.new_address[target];
      if (translated < 0)
        throw std::logic_error("Mark compact: live owner referenced unreachable target");
      fields.push_back(translated);
      ++out.work.rewritten;
      if (trace) {
        admit(trace, quota);
        trace->references.push_back(
            {owner, out.result.new_address[owner], field, target, translated});
      }
    }
    out.result.updated_refs.push_back(std::move(fields));
  }
  out.relocated_roots.reserve(graph.roots.size());
  for (std::size_t slot = 0; slot < graph.roots.size(); ++slot) {
    const int old = graph.roots[slot], translated = out.result.new_address[old];
    if (translated < 0)
      throw std::logic_error("Mark compact: original root was not marked");
    out.relocated_roots.push_back(translated);
    ++out.work.roots;
    if (trace) {
      admit(trace, quota);
      trace->roots.push_back({slot, old, translated});
    }
  }
  return out;
}
} // namespace
MarkCompactAnalysis mark_compact_analyze(const ObjectGraph& graph) {
  auto marking = object_liveness_analyze(graph);
  return compact(graph, marking.marked, marking.work, nullptr, 0);
}
MarkCompactResult mark_compact(const ObjectGraph& graph) {
  return mark_compact_analyze(graph).result;
}
MarkCompactTrace mark_compact_trace(const ObjectGraph& graph, std::size_t max_entries) {
  MarkCompactTrace t;
  t.marking          = object_liveness_trace(graph, max_entries);
  t.captured_entries = t.marking.captured_entries;
  t.analysis = compact(graph, t.marking.analysis.marked, t.marking.analysis.work, &t, max_entries);
  return t;
}
ObjectGraph mark_compact_demo_graph() {
  return {10, {{2}, {}, {4}, {}, {6}, {}, {8}, {}, {}, {}}, {0}, {}};
}
} // namespace datamunge::algorithms
