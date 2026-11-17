/// @file merge.cpp
/// @brief Checked stable two-run and shared tuple-heap ordinary/deduplicating merge producers.
#include <algorithm>
#include <datamunge/algorithms/merge.hpp>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
namespace datamunge::algorithms {
/// @brief Local complete-run validation, actual comparator counters and optional capture.
namespace merge_detail {
/// @brief Add a selected logical count without wrap.
/// @param count Actual mutable counter.
/// @param amount Additional logical entries/operations.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& count, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("merge count overflow");
  count += amount;
}
/// @brief Validate all sorted rows and complete combined output geometry.
/// @param lists Complete borrowed original runs.
/// @return Checked complete N values.
/// @throws std::invalid_argument On any unsorted row.
/// @throws std::length_error On total/output representability limits.
std::size_t validate(const std::vector<std::vector<int>>& lists) {
  std::size_t total = 0;
  for (const auto& row : lists) {
    if (!std::is_sorted(row.begin(), row.end()))
      throw std::invalid_argument("merge unsorted run");
    add(total, row.size());
  }
  if (total > std::vector<int>().max_size())
    throw std::length_error("merge output size");
  return total;
}
/// @brief Actual(value,list,index) ordering key, original identities retained.
using Node = std::tuple<int, std::size_t, std::size_t>;
/// @brief Actual whole-tuple heap comparator borrowing one local work object only.
struct Compare {
  MergeWork* work; ///< Borrowed actual counter during this kernel call, never returned.

  /// @brief Count one actual tuple ordering invocation and return its greater relation.
  /// @param a First actual heap node.
  /// @param b Second actual heap node.
  /// @return True if a sorts after b by(value,list,index).
  /// @throws std::length_error On comparator count overflow.
  bool operator()(const Node& a, const Node& b) const {
    add(work->node_comparisons, 1);
    return a > b;
  }
};
/// @brief Admit logical original/event entries before input copying or heap operation.
/// @param trace Optional local producing capture.
/// @param amount Additional original/event count.
/// @param cap Optional complete ceiling, zero off.
/// @throws std::length_error On count overflow or enabled quota.
void admit(MergeTrace* trace, std::size_t amount, std::size_t cap) {
  if (!trace)
    return;
  auto next = trace->captured_entries;
  add(next, amount);
  if (cap && next > cap)
    throw std::length_error("merge trace quota");
  trace->captured_entries = next;
}
/// @brief Produce one actual tuple-heap merge and optionally observe original operations.
/// @param lists Validated immutable original runs.
/// @param total Checked complete N.
/// @param unique Actual deduplication mode.
/// @param trace Optional local owned observation output.
/// @param cap Optional capture ceiling.
/// @return Owned actual values and selected work.
/// @throws std::length_error On count/capture/container limits.
/// @throws std::bad_alloc On heap/output/capture allocation failure.
MergeAnalysis run(
    const std::vector<std::vector<int>>& lists,
    std::size_t                          total,
    bool                                 unique,
    MergeTrace*                          trace,
    std::size_t                          cap) {
  admit(trace, lists.size(), cap);
  admit(trace, total, cap);
  if (trace)
    trace->lists = lists;
  MergeAnalysis out;
  out.work.input_values = total;
  std::priority_queue<Node, std::vector<Node>, Compare> heap(Compare{&out.work});
  if (!unique)
    out.values.reserve(total);
  /// @brief Insert one actual original run front and capture its producing heap boundary.
  /// @param l Original run identity.
  /// @param e Original element identity within that run.
  /// @throws std::length_error On capture/comparator/operation count limits.
  /// @throws std::bad_alloc On heap or actual-event append allocation failure.
  auto push = [&](std::size_t l, std::size_t e) {
    admit(trace, 1, cap);
    const auto before = heap.size();
    heap.emplace(lists[l][e], l, e);
    add(out.work.pushes, 1);
    out.work.peak_heap = std::max(out.work.peak_heap, heap.size());
    if (trace)
      trace->steps.push_back({MergeHeapKind::Push, lists[l][e], l, e, before, heap.size(), false});
  };
  for (std::size_t l = 0; l < lists.size(); ++l)
    if (!lists[l].empty())
      push(l, 0);
  while (!heap.empty()) {
    admit(trace, 1, cap);
    const auto [value, l, e] = heap.top();
    const auto before        = heap.size();
    heap.pop();
    add(out.work.pops, 1);
    const bool emit = !unique || out.values.empty() || value != out.values.back();
    if (emit) {
      out.values.push_back(value);
      add(out.work.emitted, 1);
    } else
      add(out.work.discarded_duplicates, 1);
    if (trace)
      trace->steps.push_back({MergeHeapKind::Pop, value, l, e, before, heap.size(), emit});
    if (e + 1 < lists[l].size())
      push(l, e + 1);
  }
  return out;
}
} // namespace merge_detail
std::vector<int> simple_merge(const std::vector<int>& a, const std::vector<int>& b) {
  if (!std::is_sorted(a.begin(), a.end()) || !std::is_sorted(b.begin(), b.end()))
    throw std::invalid_argument("merge unsorted run");
  std::size_t total = a.size();
  merge_detail::add(total, b.size());
  if (total > std::vector<int>().max_size())
    throw std::length_error("merge output size");
  std::vector<int> out;
  out.reserve(total);
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (b[j] < a[i])
      out.push_back(b[j++]);
    else
      out.push_back(a[i++]);
  }
  while (i < a.size())
    out.push_back(a[i++]);
  while (j < b.size())
    out.push_back(b[j++]);
  return out;
}
std::vector<int> k_way_merge(const std::vector<std::vector<int>>& lists) {
  const auto n = merge_detail::validate(lists);
  return merge_detail::run(lists, n, false, nullptr, 0).values;
}
std::vector<int> union_merge(const std::vector<std::vector<int>>& lists) {
  const auto n = merge_detail::validate(lists);
  return merge_detail::run(lists, n, true, nullptr, 0).values;
}
MergeAnalysis k_way_merge_analyze(const std::vector<std::vector<int>>& lists, bool unique) {
  const auto n = merge_detail::validate(lists);
  return merge_detail::run(lists, n, unique, nullptr, 0);
}
MergeTrace k_way_merge_trace(
    const std::vector<std::vector<int>>& lists,
    bool                                 unique,
    std::size_t                          max_entries) {
  const auto n = merge_detail::validate(lists);
  MergeTrace out;
  auto       a = merge_detail::run(lists, n, unique, &out, max_entries);
  out.values   = std::move(a.values);
  out.work     = a.work;
  return out;
}
} // namespace datamunge::algorithms
