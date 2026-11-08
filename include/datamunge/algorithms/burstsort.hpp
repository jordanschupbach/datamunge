#pragma once
/// @file burstsort.hpp
/// @brief Byte-ordered burst-trie sorting with flat ownership, explicit work stacks and actual structure traces.

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Positive leaf capacity and optional whole-call trie-node limit.
/// A bucket bursts only when its size becomes greater than bucket_capacity.
/// max_nodes=0 means unlimited; a positive cap includes the root and throws before
/// an additional node would exceed it. Empty/singleton inputs build no trie.
/// @par Example
/// @code
/// dm::algorithms::BurstSortOptions o;o.bucket_capacity=1;
/// const auto t=dm::algorithms::burstsort_trace({"ab","aa","a",""},o);
/// std::print("{} {}\n",t.output.size(),t.output[3]); // 4 ab
/// @endcode
/// @see burstsort_with_options(), burstsort_trace()
struct BurstSortOptions {
  std::size_t bucket_capacity{8}; ///< Maximum nonburst leaf size; must be positive, default eight.
  std::size_t max_nodes{0}; ///< Complete-call node budget including root; zero for unlimited.
};

/// @brief One actual trie node, identified by its index in BurstSortTrace::nodes.
/// Root zero has parent=depth=0 and no incoming byte. Every other parent precedes
/// its child in creation order. Strings ending here precede all byte slots; their
/// original input indices retain encounter order. Compact parent links avoid
/// storing every full prefix, which could otherwise take quadratic space.
/// @par Example
/// @code
/// dm::algorithms::BurstSortOptions o;o.bucket_capacity=1;
/// const auto t=dm::algorithms::burstsort_trace({"ab","aa","a",""},o);
/// std::print("{} {}\n",t.nodes[1].depth,t.nodes[1].terminal_input_indices); // 1 [2]
/// @endcode
/// @see burstsort_prefix(), BurstSortSlot
struct BurstSortNode {
  std::size_t parent{0}; ///< Parent node id; root points to itself.
  unsigned char incoming_byte{0}; ///< Byte from parent to this node; unused at root.
  std::size_t depth{0}; ///< Number of prefix bytes consumed at this node.
  std::size_t strings{0}; ///< Final subtree string count, including terminals and all descendants.
  std::vector<std::size_t> terminal_input_indices; ///< Actual ending-string ids in encounter/output order.
};

/// @brief One occupied byte slot observed during actual ordered collection.
/// child=0 means a leaf bucket; a positive child names a real trie node (root is
/// never a child). Child slots own no bucket records. Leaf snapshots preserve
/// actual encounter order and actual sorted order; full string comparisons use
/// unsigned byte ordering, with original input id as the equality tie break.
/// Slots appear in collection preorder, with ascending byte slots within a node.
/// @par Example
/// @code
/// const auto t=dm::algorithms::burstsort_trace({"ab","aa"});
/// std::print("{} {}\n",t.slots[0].input_indices,t.slots[0].sorted_indices); // [0, 1] [1, 0]
/// @endcode
/// @see BurstSortTrace, BurstSortNode
struct BurstSortSlot {
  std::size_t node{0}; ///< Actual owning node id.
  unsigned char value{0}; ///< Routing byte, including zero as ordinary data.
  std::size_t child{0}; ///< Positive child id, or zero for a leaf bucket.
  std::size_t strings{0}; ///< Final leaf/subtree string count represented by this slot.
  std::size_t sort_comparisons{0}; ///< Actual full-string comparator calls for a leaf; zero for a child slot.
  std::vector<std::size_t> input_indices; ///< Leaf ids before sorting, empty for a child.
  std::vector<std::size_t> sorted_indices; ///< Leaf ids after sorting, empty for a child.
};

/// @brief One actual overflow and redistribution into a newly created child.
/// At creation, strings=bucket_capacity+1; later subtree sizes may be larger.
/// trigger_input_index is the item being processed, including during redistribution
/// from an earlier burst. No string values or full prefixes are copied into events.
/// @par Example
/// @code
/// dm::algorithms::BurstSortOptions o;o.bucket_capacity=1;
/// const auto t=dm::algorithms::burstsort_trace({"ab","aa"},o);
/// std::print("{} {}\n",t.bursts[0].strings,t.bursts[0].trigger_input_index); // 2 1
/// @endcode
/// @see BurstSortTrace
struct BurstSortBurst {
  std::size_t node{0}; ///< Parent owning the overflowing slot.
  unsigned char value{0}; ///< Slot byte that becomes a child edge.
  std::size_t child{0}; ///< Actual new child id in creation order.
  std::size_t child_depth{0}; ///< Newly consumed prefix length.
  std::size_t strings{0}; ///< Number of actual bucket ids redistributed at this event.
  std::size_t trigger_input_index{0}; ///< Original id being processed when overflow occurs.
};

/// @brief Owned sorted strings, stable original ids, actual bursts and final trie metadata.
/// Equal strings keep original-id order. Empty/singleton inputs take the historical
/// shortcut: nodes/slots/bursts are empty and all work counters are zero. Otherwise
/// snapshots come from the shared insertion/collection kernel, never a replay or a
/// first-byte reconstruction. Parent links represent deep prefixes without quadratic
/// prefix copies. Actual counters throw on size_t overflow. Calls share no mutable state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::burstsort_trace({"b","a","a",""});
/// std::print("{}\n",t.output_input_indices); // [3, 1, 2, 0]
/// @endcode
/// @see burstsort_trace(), burstsort_prefix()
struct BurstSortTrace {
  std::vector<std::string> input; ///< Owned original strings.
  std::vector<std::string> output; ///< Sorted string copies in unsigned byte order.
  std::vector<std::size_t> output_input_indices; ///< Actual stable original-id output order.
  bool shortcut{true}; ///< Whether fewer than two strings avoided building a trie.
  std::size_t bucket_capacity{8}; ///< Capacity used for this call.
  std::vector<BurstSortNode> nodes; ///< Actual nodes in append/creation order; root is zero when present.
  std::vector<BurstSortSlot> slots; ///< Actual occupied slots in ordered collection preorder.
  std::vector<BurstSortBurst> bursts; ///< Actual chronological overflow events.
  std::size_t insertion_visits{0}; ///< Processed (node,string) work items, including redistribution and terminals.
  std::size_t byte_routes{0}; ///< Actual string-byte reads for trie routing; excludes leaf string comparisons.
  std::size_t redistributed_strings{0}; ///< Sum of bucket sizes moved into children at bursts.
  std::size_t leaf_comparisons{0}; ///< Actual std::sort comparator invocations in leaf buckets.
  std::size_t collection_slot_scans{0}; ///< Actual 256-slot directory checks over every visited trie node.
  std::size_t max_depth{0}; ///< Deepest actual node prefix, zero if root-only or shortcut.
};

/// @brief Flat node/slot arenas and explicit insertion/collection stacks.
namespace burstsort_detail {
/// @brief One private node directory whose entries are slot ids plus one, or zero.
struct Node {
  BurstSortNode record; ///< Actual parent/depth/subtree/terminal metadata needed by the algorithm.
  std::array<std::size_t,256> handles{}; ///< Dense byte-to-slot directory; slots themselves are allocated only when used.
};
/// @brief One actual occupied slot, holding either ids or a child node index.
struct Slot {
  std::size_t node{0}; ///< Owning node id.
  unsigned char value{0}; ///< Routing byte.
  std::size_t child{0}; ///< Child id, or zero for a leaf bucket.
  std::vector<std::size_t> bucket; ///< Actual full-string input ids, not copied string values or suffixes.
};
/// @brief One explicit insertion work item, including items redistributed after a burst.
struct Work {
  std::size_t node{0}; ///< Node to visit.
  std::size_t input_index{0}; ///< Original string id.
};
/// @brief One explicit ordered-collection frame.
struct Frame {
  std::size_t node{0}; ///< Node whose directory is being scanned.
  std::size_t next_byte{0}; ///< Next byte slot, in [0,256].
};
/// @brief Add actual work without wrapping a size_t counter.
/// @param target Counter to update.
/// @param amount Increment, default one.
/// @throws std::overflow_error If the sum is not representable.
inline void add(std::size_t& target,std::size_t amount=1) {
  if (amount>std::numeric_limits<std::size_t>::max()-target) throw std::overflow_error("burstsort trace counter overflow");
  target+=amount;
}
/// @brief Reject a zero bucket capacity before building or copying work state.
/// @param options Requested algorithm settings.
/// @throws std::invalid_argument If bucket_capacity is zero.
inline void validate(BurstSortOptions options) {
  if (!options.bucket_capacity) throw std::invalid_argument("burstsort bucket capacity must be positive");
}

/// @brief Shared flat burst insertion and ordered leaf sorting with optional actual snapshots.
/// @tparam Record Whether to own input/topology/event snapshots and count actual work.
/// @param input Borrowed immutable strings; full values are copied only into final output and optional owned input.
/// @param options Positive capacity and optional whole-call node limit.
/// @return Owned sorted output/order and optional actual records; no partial result on failure.
/// @throws std::invalid_argument For zero capacity.
/// @throws std::length_error Before exceeding a positive node limit or container capacity.
/// @throws std::overflow_error If an actual trace counter cannot fit size_t.
/// @throws std::bad_alloc If work/output/snapshot storage cannot be allocated.
template<bool Record>
BurstSortTrace run(const std::vector<std::string>& input,BurstSortOptions options) {
  validate(options);BurstSortTrace trace;trace.bucket_capacity=options.bucket_capacity;
  if constexpr (Record) trace.input=input;
  if (input.size()<2) {
    trace.output=input;trace.output_input_indices.resize(input.size());
    return trace;
  }
  trace.shortcut=false;
  std::vector<Node> nodes(1);std::vector<Slot> slots;std::vector<Work> pending;
  for (std::size_t original=0;original<input.size();++original) {
    pending.push_back({0,original});
    while (!pending.empty()) {
      const auto work=pending.back();pending.pop_back();
      const auto depth=nodes[work.node].record.depth;
      ++nodes[work.node].record.strings;
      if constexpr (Record) add(trace.insertion_visits);
      const auto& text=input[work.input_index];
      if (depth==text.size()) {nodes[work.node].record.terminal_input_indices.push_back(work.input_index);continue;}
      const auto value=static_cast<unsigned char>(text[depth]);
      if constexpr (Record) add(trace.byte_routes);
      auto handle=nodes[work.node].handles[value];
      if (!handle) {
        slots.push_back({work.node,value,0,{}});handle=slots.size();nodes[work.node].handles[value]=handle;
      }
      const auto slot=handle-1;
      if (slots[slot].child) {pending.push_back({slots[slot].child,work.input_index});continue;}
      slots[slot].bucket.push_back(work.input_index);
      if (slots[slot].bucket.size()>options.bucket_capacity) {
        if (options.max_nodes && nodes.size()>=options.max_nodes)
          throw std::length_error("burstsort node budget exceeded");
        const auto child=nodes.size();nodes.emplace_back();
        nodes[child].record.parent=work.node;nodes[child].record.incoming_byte=value;nodes[child].record.depth=depth+1;
        slots[slot].child=child;
        auto redistributed=std::move(slots[slot].bucket);
        if constexpr (Record) {
          trace.bursts.push_back({work.node,value,child,depth+1,redistributed.size(),work.input_index});
          add(trace.redistributed_strings,redistributed.size());trace.max_depth=std::max(trace.max_depth,depth+1);
        }
        // Reverse push preserves the old recursive encounter order, without recursion.
        for (auto item=redistributed.rbegin();item!=redistributed.rend();++item) pending.push_back({child,*item});
      }
    }
  }
  trace.output_input_indices.reserve(input.size());
  std::vector<Frame> stack;
  /// @brief Emit actual terminal ids before beginning this node's ascending-byte scan.
  /// @param node Node to enter; pushing may grow the explicit collection stack.
  auto enter=[&](std::size_t node) {
    const auto& ended=nodes[node].record.terminal_input_indices;
    trace.output_input_indices.insert(trace.output_input_indices.end(),ended.begin(),ended.end());
    stack.push_back({node,0});
  };
  enter(0);
  while (!stack.empty()) {
    const auto node=stack.back().node;
    if (stack.back().next_byte==256) {stack.pop_back();continue;}
    const auto value=stack.back().next_byte++;
    if constexpr (Record) add(trace.collection_slot_scans);
    const auto handle=nodes[node].handles[value];if (!handle) continue;
    const auto slot=handle-1;const auto child=slots[slot].child;
    BurstSortSlot observed;
    if constexpr (Record) {
      observed.node=node;observed.value=static_cast<unsigned char>(value);observed.child=child;
      observed.strings=child?nodes[child].record.strings:slots[slot].bucket.size();
      if (!child) observed.input_indices=slots[slot].bucket;
    }
    if (child) {
      if constexpr (Record) trace.slots.push_back(std::move(observed));
      enter(child);
    } else {
      auto& bucket=slots[slot].bucket;
      /// @brief Compare full strings, breaking equal-value ties by original input id.
      /// @param a First original string id.
      /// @param b Second original string id.
      /// @return Whether a belongs before b in stable unsigned-byte lexical order.
      auto compare=[&](std::size_t a,std::size_t b) {
        if constexpr (Record) {add(trace.leaf_comparisons);add(observed.sort_comparisons);}
        const int order=input[a].compare(input[b]);return order<0 || (order==0 && a<b);
      };
      std::sort(bucket.begin(),bucket.end(),compare);
      trace.output_input_indices.insert(trace.output_input_indices.end(),bucket.begin(),bucket.end());
      if constexpr (Record) {observed.sorted_indices=bucket;trace.slots.push_back(std::move(observed));}
    }
  }
  trace.output.reserve(input.size());
  for (const auto id:trace.output_input_indices) trace.output.push_back(input[id]);
  if constexpr (Record) {
    trace.nodes.reserve(nodes.size());for (auto& node:nodes) trace.nodes.push_back(std::move(node.record));
  }
  return trace;
}
} // namespace burstsort_detail

/// @brief Sort strings with the original eight-item burst threshold and byte order.
/// Signature and default threshold are preserved. Empty/singleton vectors return
/// immediately. Flat node ownership and explicit insertion/collection stacks avoid
/// recursive depth in insertion, traversal and destruction. Buckets hold indices
/// into the unchanged input; full strings are copied once into output before a
/// final swap. On allocation failure the caller's vector remains unchanged. Equal
/// values retain input-id order; no collation, Unicode normalization or cache-speed
/// guarantee is implied. Ordinary calls allocate no topology/event snapshots.
/// @param a Strings to sort in unsigned byte lexicographic order, including NUL/high-bit bytes.
/// @throws std::bad_alloc If work/output storage cannot be allocated; a is unchanged.
/// @throws std::length_error If required container storage cannot fit; a is unchanged.
/// @par Example
/// @code
/// std::vector<std::string> a{"banana","apple","app",""};
/// dm::algorithms::burstsort(a);std::print("{} {}\n",a[1],a.back()); // app banana
/// @endcode
/// @par References
/// - R. Sinha and J. Zobel (2004), Cache-Conscious Sorting of Large Sets of Strings
///   with Dynamic Tries, https://people.eng.unimelb.edu.au/jzobel/fulltext/acmjea04.pdf.
///   This variant uses index buckets, dense byte directories and full-string leaf comparisons.
/// @see burstsort_with_options(), burstsort_trace()
inline void burstsort(std::vector<std::string>& a) {
  if (a.size()<2) return;
  auto result=burstsort_detail::run<false>(a,{});a.swap(result.output);
}

/// @brief Sort with a chosen positive bucket capacity and optional complete-call node cap.
/// Same kernel, byte order, stable identity ties and strong input preservation as
/// burstsort(). A node cap includes root; max_nodes=0 means unlimited. Exceeding a
/// positive cap throws without swapping any partially sorted data into a.
/// @param a Strings to sort, including empty strings and arbitrary byte content.
/// @param options Positive capacity and optional node limit; defaults to capacity eight, no cap.
/// @throws std::invalid_argument If capacity is zero, even for empty/singleton input.
/// @throws std::length_error If a node/container limit would be exceeded; a is unchanged.
/// @throws std::bad_alloc If work/output storage cannot be allocated; a is unchanged.
/// @par Example
/// @code
/// std::vector<std::string> a{"ab","aa","a",""};
/// dm::algorithms::BurstSortOptions o;o.bucket_capacity=1;
/// dm::algorithms::burstsort_with_options(a,o);std::print("{} {}\n",a.size(),a[3]); // 4 ab
/// @endcode
/// @par References
/// - R. Sinha and J. Zobel (2004), Cache-Conscious Sorting of Large Sets of Strings
///   with Dynamic Tries, https://people.eng.unimelb.edu.au/jzobel/fulltext/acmjea04.pdf.
/// @see burstsort(), BurstSortOptions
inline void burstsort_with_options(std::vector<std::string>& a,BurstSortOptions options={}) {
  burstsort_detail::validate(options);if (a.size()<2) return;
  auto result=burstsort_detail::run<false>(a,options);a.swap(result.output);
}

/// @brief Sort once and own actual burst events, final slot/node records and sorted input ids.
/// Uses the ordinary sort's kernel without replay or inferred first-byte counts.
/// No full prefix is stored per node; use burstsort_prefix() for selected nodes.
/// If D is input byte volume, Q node count, and n string count, algorithm storage
/// is O(D+n+256Q), with O(D+n+Q) additional owned trace storage. Routing/collection
/// work is O(D+n+256Q) plus actual full-string leaf comparisons. Long common prefixes
/// can produce deep one-child chains; the trie is not guaranteed shallow.
/// @param input Borrowed string vector, never mutated, including on failure.
/// @param options Positive capacity and optional node cap, default eight/unlimited.
/// @return Owned input/output, stable ids, actual work, events and final structure.
/// @throws std::invalid_argument If capacity is zero.
/// @throws std::length_error If a node/container limit would be exceeded.
/// @throws std::overflow_error Before an actual trace counter could wrap.
/// @throws std::bad_alloc If work/output/trace storage cannot be allocated.
/// @par Example
/// @code
/// dm::algorithms::BurstSortOptions o;o.bucket_capacity=1;
/// const auto t=dm::algorithms::burstsort_trace({"ab","aa","a",""},o);
/// std::print("{} {} {}\n",t.nodes.size(),t.bursts.size(),t.output_input_indices); // 2 1 [3, 2, 1, 0]
/// @endcode
/// @par References
/// - R. Sinha and J. Zobel (2004), Cache-Conscious Sorting of Large Sets of Strings
///   with Dynamic Tries, https://people.eng.unimelb.edu.au/jzobel/fulltext/acmjea04.pdf.
/// @see burstsort(), burstsort_prefix(), BurstSortTrace
inline BurstSortTrace burstsort_trace(const std::vector<std::string>& input,BurstSortOptions options={}) {
  return burstsort_detail::run<true>(input,options);
}

/// @brief Materialize one actual node prefix from compact parent links, in O(depth) work.
/// The generated trace stores no quadratic collection of full prefixes. This helper
/// checks root, strictly earlier parents and depth increments along the selected path.
/// A shortcut trace has no root. Result owns its bytes, including possible NULs.
/// @param trace Actual generated structure or a structurally compatible owned record.
/// @param node Node id in trace.nodes, with zero representing the root when present.
/// @return Owned byte prefix; empty at the root.
/// @throws std::out_of_range If node is not present, including a shortcut's missing root.
/// @throws std::invalid_argument If the selected parent/depth path or root is malformed.
/// @throws std::bad_alloc If prefix storage cannot be allocated.
/// @throws std::length_error If prefix storage cannot fit.
/// @par Example
/// @code
/// dm::algorithms::BurstSortOptions o;o.bucket_capacity=1;
/// const auto t=dm::algorithms::burstsort_trace({"ab","aa"},o);
/// std::print("{}\n",dm::algorithms::burstsort_prefix(t,1)); // a
/// @endcode
/// @see burstsort_trace(), BurstSortNode
inline std::string burstsort_prefix(const BurstSortTrace& trace,std::size_t node) {
  if (node>=trace.nodes.size()) throw std::out_of_range("burstsort node is absent");
  if (trace.nodes[0].parent || trace.nodes[0].depth) throw std::invalid_argument("burstsort root metadata is malformed");
  std::string prefix;
  while (node) {
    const auto& current=trace.nodes[node];
    if (current.parent>=node) throw std::invalid_argument("burstsort parent is not an earlier node");
    const auto& parent=trace.nodes[current.parent];
    if (parent.depth==std::numeric_limits<std::size_t>::max() || current.depth!=parent.depth+1)
      throw std::invalid_argument("burstsort prefix depths are inconsistent");
    prefix.push_back(static_cast<char>(current.incoming_byte));node=current.parent;
  }
  std::reverse(prefix.begin(),prefix.end());return prefix;
}

} // namespace datamunge::algorithms
