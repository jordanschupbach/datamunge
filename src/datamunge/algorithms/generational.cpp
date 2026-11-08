/// @file generational.cpp
/// @brief Shared checked young traversal, explicit full comparison and maintained boundary-slot
/// model.
#include <algorithm>
#include <datamunge/algorithms/generational.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Implementation details for checked fixed-age graph collection.
namespace {
/// @brief Increment an actual selected-work counter without wrapping.
/// @param value Counter changed only in the current operation.
/// @throws std::length_error The next count cannot fit size_t.
void increment(std::size_t& value) {
  if (value == std::numeric_limits<std::size_t>::max())
    throw std::length_error("generational: counter overflow");
  ++value;
}
/// @brief Validate every graph id and exact binary ages before any collector indexing/copy.
/// @param g Graph to inspect, ignoring sizes.
/// @param gen Exact binary ages.
/// @return Actual successful validation and age-check work.
/// @throws std::invalid_argument Bad graph or generation length/value.
/// @throws std::length_error Validation counts overflow.
GenerationalWork validate(const ObjectGraph& g, const std::vector<int>& gen) {
  GenerationalWork work;
  work.validation = validate_object_graph(g, false);
  if (gen.size() != static_cast<std::size_t>(g.num_objects))
    throw std::invalid_argument("generational: exact generation length required");
  for (int age : gen) {
    if (age != 0 && age != 1)
      throw std::invalid_argument("generational: ages must be0 or1");
    increment(work.ages_checked);
  }
  return work;
}
/// @brief Build ascending sparse young ids and a dense inverse index after validation.
/// @param gen Validated binary ages.
/// @param ids Receives ascending young ids.
/// @param index Receives young position or -1 for each old id.
void tables(const std::vector<int>& gen, std::vector<int>& ids, std::vector<int>& index) {
  index.assign(gen.size(), -1);
  for (std::size_t i = 0; i < gen.size(); ++i)
    if (gen[i] == 0) {
      index[i] = static_cast<int>(ids.size());
      ids.push_back(static_cast<int>(i));
    }
}
/// @brief Capture the next actual primitive record when requested.
/// @param trace Optional owned trace; null skips capture entirely.
/// @param cap Positive actual-record quota; zero disables quota.
/// @param kind Bounded primitive label.
/// @param object Actual source/current id.
/// @param target Actual target/current id or -1 for remembered sources.
/// @param accepted Actual decision.
/// @param stack Actual stack size at this control point.
/// @throws std::length_error Next actual record exceeds quota/vector capacity.
void event(
    GenerationalTrace* trace,
    std::size_t        cap,
    const char*        kind,
    int                object,
    int                target,
    bool               accepted,
    std::size_t        stack) {
  if (!trace)
    return;
  if ((cap && trace->events.size() >= cap) || trace->events.size() == trace->events.max_size())
    throw std::length_error("generational: actual event cap");
  trace->events.push_back({kind, object, target, accepted, stack});
}
/// @brief Discover young direct roots and every old-to-young field in legacy order.
/// @param g Validated immutable graph.
/// @param index Validated dense young index.
/// @param work Actual work receiving discovery counts.
/// @param remembered Receives ascending distinct old sources.
/// @param trace Optional actual event recorder.
/// @param cap Positive online record quota; zero disables quota.
/// @return Owned ordered seed candidates, including duplicates.
/// @throws std::length_error Actual work or event quota overflows.
std::vector<int> discover(
    const ObjectGraph&      g,
    const std::vector<int>& index,
    GenerationalWork&       work,
    std::vector<int>&       remembered,
    GenerationalTrace*      trace,
    std::size_t             cap) {
  std::vector<int> seeds;
  for (int r : g.roots) {
    increment(work.root_checks);
    const bool young = index[r] >= 0;
    event(trace, cap, "root", r, r, young, 0);
    if (young) {
      seeds.push_back(r);
      increment(work.seed_candidates);
    }
  }
  for (int i = 0; i < g.num_objects; ++i) {
    increment(work.discovery_objects);
    if (index[i] < 0) {
      bool any = false;
      for (int c : g.refs[i]) {
        increment(work.old_fields);
        const bool young = index[c] >= 0;
        event(trace, cap, "old_field", i, c, young, 0);
        if (young) {
          seeds.push_back(c);
          increment(work.seed_candidates);
          any = true;
        }
      }
      if (any) {
        remembered.push_back(i);
        event(trace, cap, "remember", i, -1, true, 0);
      }
    }
  }
  return seeds;
}
/// @brief Sparse logical result of the one actual young-marking kernel.
struct YoungState {
  std::vector<int> survivors; ///< Ascending marked young ids.
  std::vector<int> freed;     ///< Ascending unmarked young ids.
};
/// @brief Mark all seeds first, execute ordered LIFO young traversal and classify sparse ids.
/// @param g Validated graph, unchanged during traversal.
/// @param ids Ascending young ids.
/// @param index Dense inverse young index, -1 for old objects.
/// @param seeds Ordered valid young candidates, possibly duplicated.
/// @param work Actual work receiving traversal/classification counts.
/// @param trace Optional primitive recorder.
/// @param cap Positive actual-record quota; zero disables quota.
/// @return Owned sparse survivor/garbage lists; no age or graph mutation.
/// @throws std::length_error Actual work or event quota overflows.
YoungState young(
    const ObjectGraph&      g,
    const std::vector<int>& ids,
    const std::vector<int>& index,
    const std::vector<int>& seeds,
    GenerationalWork&       work,
    GenerationalTrace*      trace,
    std::size_t             cap) {
  std::vector<char> marked(ids.size(), 0);
  std::vector<int>  stack;
  for (int r : seeds) {
    const auto position = static_cast<std::size_t>(index[r]);
    event(trace, cap, "seed", r, r, !marked[position], stack.size());
    if (!marked[position]) {
      marked[position] = 1;
      stack.push_back(r);
      increment(work.seed_marks);
      work.peak_young_stack = std::max(work.peak_young_stack, stack.size());
      event(trace, cap, "mark", r, r, true, stack.size());
    }
  }
  while (!stack.empty()) {
    const int u = stack.back();
    stack.pop_back();
    increment(work.young_objects);
    event(trace, cap, "visit", u, u, true, stack.size());
    for (int c : g.refs[u]) {
      increment(work.young_fields);
      const bool fresh = index[c] >= 0 && !marked[static_cast<std::size_t>(index[c])];
      event(trace, cap, "young_field", u, c, fresh, stack.size());
      if (fresh) {
        marked[static_cast<std::size_t>(index[c])] = 1;
        stack.push_back(c);
        work.peak_young_stack = std::max(work.peak_young_stack, stack.size());
        event(trace, cap, "mark", c, c, true, stack.size());
      }
    }
  }
  YoungState out;
  for (std::size_t k = 0; k < ids.size(); ++k) {
    increment(work.classified_young);
    const int id = ids[k];
    if (marked[k]) {
      out.survivors.push_back(id);
      event(trace, cap, "promote", id, id, true, 0);
    } else {
      out.freed.push_back(id);
      event(trace, cap, "free", id, id, false, 0);
    }
  }
  return out;
}
/// @brief Execute the actual full comparison after prior complete validation.
/// @param g Validated graph; same push-before-traverse order as reachable_from_roots.
/// @param work Actual work receiving full traversal counts.
/// @param trace Optional actual primitive recorder.
/// @param cap Positive record quota; zero disables quota.
/// @return Owned actual full-root mark vector.
/// @throws std::length_error Actual work or record quota overflows.
std::vector<char>
full(const ObjectGraph& g, GenerationalWork& work, GenerationalTrace* trace, std::size_t cap) {
  std::vector<char> marked(static_cast<std::size_t>(g.num_objects), 0);
  std::vector<int>  stack;
  for (int r : g.roots) {
    increment(work.full_roots);
    event(trace, cap, "full_root", r, r, !marked[r], stack.size());
    if (!marked[r]) {
      marked[r] = 1;
      stack.push_back(r);
      event(trace, cap, "full_mark", r, r, true, stack.size());
    }
  }
  while (!stack.empty()) {
    const int u = stack.back();
    stack.pop_back();
    increment(work.full_objects);
    event(trace, cap, "full_visit", u, u, true, stack.size());
    for (int c : g.refs[u]) {
      increment(work.full_fields);
      event(trace, cap, "full_field", u, c, !marked[c], stack.size());
      if (!marked[c]) {
        marked[c] = 1;
        stack.push_back(c);
        event(trace, cap, "full_mark", c, c, true, stack.size());
      }
    }
  }
  return marked;
}
/// @brief Execute the shared legacy-compatible discovery/young/full analysis after validation.
/// @param g Validated graph.
/// @param gen Validated binary ages.
/// @param work Completed validation work, extended by actual operations.
/// @param trace Optional owned event recorder.
/// @param cap Positive actual-record quota; zero disables quota.
/// @return Owned legacy result and full marks from this execution.
/// @throws std::length_error Actual work or event quota overflows.
GenerationalRun run(
    const ObjectGraph&      g,
    const std::vector<int>& gen,
    GenerationalWork        work,
    GenerationalTrace*      trace = nullptr,
    std::size_t             cap   = 0) {
  GenerationalRun  out;
  std::vector<int> ids, index;
  tables(gen, ids, index);
  const auto seeds = discover(g, index, work, out.result.remembered_set, trace, cap);
  auto       state = young(g, ids, index, seeds, work, trace, cap);
  out.result.promoted.assign(gen.size(), 0);
  work.dense_result_slots = gen.size();
  for (int id : state.survivors)
    out.result.promoted[id] = 1;
  out.result.young_freed   = std::move(state.freed);
  out.result.minor_scanned = static_cast<int>(work.young_objects);
  out.full_reachable       = full(g, work, trace, cap);
  out.result.full_scanned  = 0;
  for (char mark : out.full_reachable) {
    increment(work.full_result_slots);
    if (mark)
      ++out.result.full_scanned;
  }
  out.work = work;
  return out;
}
} // namespace

GenerationalResult minor_gc(const ObjectGraph& g, const std::vector<int>& gen) {
  return minor_gc_run(g, gen).result;
}
GenerationalRun minor_gc_run(const ObjectGraph& g, const std::vector<int>& gen) {
  return run(g, gen, validate(g, gen));
}
GenerationalTrace
minor_gc_trace(const ObjectGraph& g, const std::vector<int>& gen, std::size_t max_events) {
  const auto        work = validate(g, gen);
  GenerationalTrace out;
  out.graph       = g;
  out.generations = gen;
  out.run         = run(out.graph, out.generations, work, &out, max_events);
  return out;
}
GenerationalInput generational_chain_example(
    std::size_t old_count,
    std::size_t unreachable_young,
    std::size_t max_objects) {
  if (!old_count)
    throw std::invalid_argument("generational: old chain must be nonempty");
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (old_count > limit - 2 || unreachable_young > limit - 2 - old_count)
    throw std::length_error("generational: dense object id capacity");
  const auto count = old_count + 2 + unreachable_young;
  if ((max_objects && count > max_objects) || count > std::vector<int>{}.max_size() ||
      count > std::vector<std::vector<int>>{}.max_size())
    throw std::length_error("generational: complete object quota");
  GenerationalInput out;
  out.graph.num_objects = static_cast<int>(count);
  out.graph.refs.resize(count);
  out.generations.assign(count, 0);
  for (std::size_t i = 0; i < old_count; ++i) {
    out.generations[i] = 1;
    out.graph.refs[i].push_back(static_cast<int>(i + 1 < old_count ? i + 1 : old_count + 1));
  }
  out.graph.roots = {0, static_cast<int>(old_count)};
  return out;
}
GenerationalHeap::GenerationalHeap(const ObjectGraph& graph, const std::vector<int>& generations) {
  preparation_ = validate(graph, generations);
  graph_       = graph;
  generations_ = generations;
  tables(generations_, young_ids_, young_index_);
  for (int i = 0; i < graph_.num_objects; ++i) {
    increment(preparation_.discovery_objects);
    if (young_index_[i] < 0)
      for (std::size_t k = 0; k < graph_.refs[i].size(); ++k) {
        increment(preparation_.old_fields);
        if (young_index_[graph_.refs[i][k]] >= 0)
          slots_.emplace_hint(slots_.end(), i, k);
      }
  }
}
bool GenerationalHeap::set_reference(int source, std::size_t field, int target) {
  if (source < 0 || source >= graph_.num_objects || target < 0 || target >= graph_.num_objects ||
      field >= graph_.refs[source].size())
    throw std::out_of_range("generational: reference write outside modeled heap");
  const auto key      = std::pair{source, field};
  const bool boundary = young_index_[source] < 0 && young_index_[target] >= 0;
  if (boundary)
    slots_.insert(key);
  else
    slots_.erase(key);
  graph_.refs[source][field] = target;
  return boundary;
}
YoungCollection GenerationalHeap::collect() const {
  YoungCollection  out;
  std::vector<int> seeds;
  for (int r : graph_.roots) {
    increment(out.work.root_checks);
    if (young_index_[r] >= 0) {
      seeds.push_back(r);
      increment(out.work.seed_candidates);
    }
  }
  for (const auto& [source, field] : slots_) {
    increment(out.work.remembered_fields);
    seeds.push_back(graph_.refs[source][field]);
    increment(out.work.seed_candidates);
  }
  auto state    = young(graph_, young_ids_, young_index_, seeds, out.work, nullptr, 0);
  out.survivors = std::move(state.survivors);
  out.freed     = std::move(state.freed);
  return out;
}
std::vector<int> GenerationalHeap::remembered_sources() const {
  std::vector<int> out;
  for (const auto& [source, field] : slots_) {
    (void)field;
    if (out.empty() || out.back() != source)
      out.push_back(source);
  }
  return out;
}
ObjectGraph GenerationalHeap::snapshot() const {
  return graph_;
}
GenerationalWork GenerationalHeap::preparation_work() const {
  return preparation_;
}
} // namespace datamunge::algorithms
