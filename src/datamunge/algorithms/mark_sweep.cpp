/// @file mark_sweep.cpp
/// @brief Shared complete-heap sweep classification following one canonical actual marking pass.
#include <datamunge/algorithms/mark_sweep.hpp>
namespace datamunge::algorithms {
/// @brief Private selected capture admission and shared actual sweep loop.
namespace {
/// @brief Admit one real sweep record before its capture copy.
/// @param trace Optional actual owned collector history.
/// @param quota Optional logical capture cap.
void admit(MarkSweepTrace* trace, std::size_t quota) {
  if (!trace)
    return;
  if (trace->captured_entries == std::numeric_limits<std::size_t>::max() ||
      (quota && trace->captured_entries >= quota))
    throw std::length_error("Mark sweep: capture quota");
  ++trace->captured_entries;
}
/// @brief Classify each actual old heap slot once from the original producing mark state.
/// @param marks Actual final shared-producer marks.
/// @param marking Actual same-run mark/validation work.
/// @param trace Optional real sweep-step output.
/// @param quota Optional logical history cap.
/// @return Actual survivor/reclaimed partition and selected phase work.
MarkSweepAnalysis sweep(
    const std::vector<char>&  marks,
    const ObjectLivenessWork& marking,
    MarkSweepTrace*           trace,
    std::size_t               quota) {
  MarkSweepAnalysis out;
  out.result.marked = marks;
  out.work.marking  = marking;
  for (std::size_t i = 0; i < marks.size(); ++i) {
    const bool live        = marks[i] != 0;
    const auto before_live = out.result.live.size(), before_freed = out.result.freed.size();
    ++out.work.scanned;
    if (live) {
      out.result.live.push_back(int(i));
      ++out.work.live;
    } else {
      out.result.freed.push_back(int(i));
      ++out.work.reclaimed;
    }
    if (trace) {
      admit(trace, quota);
      trace->sweep.push_back(
          {int(i),
           live,
           before_live,
           before_freed,
           out.result.live.size(),
           out.result.freed.size()});
    }
  }
  return out;
}
} // namespace
MarkSweepAnalysis mark_sweep_analyze(const ObjectGraph& graph) {
  auto marking = object_liveness_analyze(graph);
  return sweep(marking.marked, marking.work, nullptr, 0);
}
MarkSweepResult mark_sweep(const ObjectGraph& graph) {
  return mark_sweep_analyze(graph).result;
}
MarkSweepTrace mark_sweep_trace(const ObjectGraph& graph, std::size_t max_entries) {
  MarkSweepTrace t;
  t.marking          = object_liveness_trace(graph, max_entries);
  t.captured_entries = t.marking.captured_entries;
  t.analysis         = sweep(t.marking.analysis.marked, t.marking.analysis.work, &t, max_entries);
  return t;
}
ObjectGraph mark_sweep_demo_graph() {
  return {8, {{1}, {2}, {}, {4}, {3}, {}, {7}, {}}, {0}, {}};
}
} // namespace datamunge::algorithms
