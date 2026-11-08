#pragma once
/// @file object_graph.hpp
/// @brief Owned object-reference graphs, checked structure and iterative liveness traversal.
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned directed heap model with ordered roots, reference fields and optional sizes.
/// Objects have dense ids 0..num_objects-1 and exactly one refs row each.
/// Duplicate roots/fields, sharing, self references and cycles are permitted;
/// negative/null ids are not represented (omit an absent reference instead).
/// The optional sizes prefix supplies positive integral units when an algorithm
/// uses sizes; missing entries mean one. Size metadata may be ignored by methods
/// working only in object slots. Copies own all vectors; inputs are not borrowed.
/// This models references and layout, not C++ object lifetime or raw-memory arenas.
/// @par Example
/// @code
/// dm::algorithms::ObjectGraph g{3,{{1},{2},{}},{0},{2}};
/// std::print("{}\n",dm::algorithms::object_size(g,0)); // 2
/// @endcode
/// @see validate_object_graph(), reachable_from_roots(), cheney_copy()
struct ObjectGraph {
  int num_objects{0}; ///< Number of dense object ids; zero in a default empty graph.
  std::vector<std::vector<int>> refs; ///< Ordered old-id reference fields; exactly num_objects rows.
  std::vector<int> roots; ///< Ordered valid root ids; duplicates are retained.
  std::vector<int> sizes; ///< Optional positive size prefix in caller-chosen units; absent entries default to one.
};
/// @brief Actual successful checks performed by graph validation, without traversal replay.
/// rows counts every adjacency row, references counts every stored edge even in
/// garbage, roots counts every root slot, and sizes counts supplied metadata only
/// when requested. Empty graphs have zero counts. These are validation visits,
/// not collector payload copying or elapsed-time measurements.
/// @par Example
/// @code
/// const auto w=dm::algorithms::validate_object_graph({3,{{1},{},{2}},{0},{}});
/// std::print("{} {}\n",w.rows,w.references); // 3 2
/// @endcode
/// @see validate_object_graph()
struct ObjectGraphWork {
  std::size_t rows{0}; ///< Checked adjacency rows, including unreachable objects.
  std::size_t references{0}; ///< Checked stored reference ids, including duplicates and garbage edges.
  std::size_t roots{0}; ///< Checked root slots, including duplicates.
  std::size_t sizes{0}; ///< Checked supplied size entries, or zero when size checks were not requested.
};
/// @brief Validate every graph id/row and optionally the positive size prefix before traversal.
/// Rejects negative counts, mismatched adjacency length, and any out-of-range
/// root or field id, including unreachable fields. With check_sizes=true, sizes
/// must contain at most num_objects positive values. Otherwise sizes is entirely
/// ignored, preserving slot-only methods' metadata independence. No allocation or
/// mutation; O(N+E+R) visits plus checked size entries, constant auxiliary storage.
/// @param graph Owned graph to inspect without mutation.
/// @param check_sizes Whether to validate supplied size metadata, default false.
/// @return Counts of actual successful validation visits.
/// @throws std::invalid_argument If a checked shape, id or size is invalid.
/// @throws std::length_error If a validation counter cannot fit size_t.
/// @par Example
/// @code
/// const auto w=dm::algorithms::validate_object_graph({2,{{1},{}},{0},{3}},true);
/// std::print("{} {} {}\n",w.rows,w.references,w.sizes); // 2 1 1
/// @endcode
/// @see ObjectGraph, reachable_from_roots(), object_size()
inline ObjectGraphWork validate_object_graph(const ObjectGraph& graph,bool check_sizes=false) {
  if(graph.num_objects<0 || graph.refs.size()!=static_cast<std::size_t>(graph.num_objects))
    throw std::invalid_argument("ObjectGraph: expected nonnegative count and one reference row per object");
  ObjectGraphWork work;
  for(const auto& row:graph.refs) {
    ++work.rows;
    for(const int child:row) {
      if(child<0 || child>=graph.num_objects)throw std::invalid_argument("ObjectGraph: reference id outside graph");
      if(work.references==std::numeric_limits<std::size_t>::max())throw std::length_error("ObjectGraph: reference count overflow");
      ++work.references;
    }
  }
  for(const int root:graph.roots) {
    if(root<0 || root>=graph.num_objects)throw std::invalid_argument("ObjectGraph: root id outside graph");
    ++work.roots;
  }
  if(check_sizes) {
    if(graph.sizes.size()>static_cast<std::size_t>(graph.num_objects))throw std::invalid_argument("ObjectGraph: size prefix exceeds object count");
    for(const int size:graph.sizes) {
      if(size<=0)throw std::invalid_argument("ObjectGraph: object sizes must be positive");
      ++work.sizes;
    }
  }
  return work;
}
/// @brief Mark every root-reachable object using a checked iterative depth-first traversal.
/// Validates the whole graph before indexing; ignores sizes. Marks on pushing,
/// so duplicate edges, shared objects and cycles add no duplicate stack entries.
/// Empty roots yield all-zero marks, and a default empty graph yields an empty
/// vector. Owns the result and leaves inputs unchanged. Validation plus traversal
/// costs O(N+E+R); dense marks and the explicit stack require O(N) storage.
/// Separate calls share no state; concurrent input mutation is unsupported.
/// @param graph Object graph with valid dense ids and adjacency shape.
/// @return Owned byte marks, one iff the old object id is reachable.
/// @throws std::invalid_argument If graph structure or any id is invalid.
/// @throws std::length_error If a vector/counter capacity is exceeded.
/// @throws std::bad_alloc If marks or stack allocation fails.
/// @par Example
/// @code
/// const auto live=dm::algorithms::reachable_from_roots({3,{{1},{0},{}},{0},{}});
/// std::print("{} {} {}\n",int(live[0]),int(live[1]),int(live[2])); // 1 1 0
/// @endcode
/// @see validate_object_graph(), cheney_copy()
inline std::vector<char> reachable_from_roots(const ObjectGraph& graph) {
  (void)validate_object_graph(graph);
  std::vector<char> marked(static_cast<std::size_t>(graph.num_objects),0);
  std::vector<int> stack;
  for(const int root:graph.roots)if(!marked[root]){marked[root]=1;stack.push_back(root);}
  while(!stack.empty()) {
    const int old=stack.back();stack.pop_back();
    for(const int child:graph.refs[old])if(!marked[child]){marked[child]=1;stack.push_back(child);}
  }
  return marked;
}
/// @brief Read one object's positive size, defaulting to one after the supplied prefix.
/// Validates the nonnegative count, requested id and selected supplied size only;
/// it does not inspect unrelated rows, ids or size entries. No allocation or
/// mutation; constant work/storage. Use validate_object_graph(graph,true) for
/// complete structural and metadata checks.
/// @param graph Object graph providing optional size metadata.
/// @param id Dense object id to inspect, from zero through num_objects-1.
/// @return Selected positive size in caller-chosen units, or one if unspecified.
/// @throws std::out_of_range If the count is negative or id is outside its range.
/// @throws std::invalid_argument If the selected supplied size is nonpositive.
/// @par Example
/// @code
/// const dm::algorithms::ObjectGraph g{3,{{},{},{}},{},{7}};
/// std::print("{} {}\n",dm::algorithms::object_size(g,0),dm::algorithms::object_size(g,2)); // 7 1
/// @endcode
/// @see validate_object_graph()
inline int object_size(const ObjectGraph& graph,int id) {
  if(graph.num_objects<0 || id<0 || id>=graph.num_objects)throw std::out_of_range("ObjectGraph: size id outside graph");
  const auto index=static_cast<std::size_t>(id);
  if(index>=graph.sizes.size())return 1;
  if(graph.sizes[index]<=0)throw std::invalid_argument("ObjectGraph: selected size must be positive");
  return graph.sizes[index];
}
} // namespace datamunge::algorithms
