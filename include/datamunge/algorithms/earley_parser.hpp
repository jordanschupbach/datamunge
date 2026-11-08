#pragma once
/// @file earley_parser.hpp
/// @brief Indexed Earley recognition with nullable closure, actual item provenance and checked capture.
#include <datamunge/algorithms/context_free_grammar.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Sentinel used in both coordinates of an absent Earley provenance reference.
inline constexpr std::size_t earley_no_item=std::numeric_limits<std::size_t>::max();

/// @brief One dotted original production and its origin column.
/// In chart column k, prod names an original production, dot counts matched RHS
/// symbols and 0<=origin<=k. Equal triples in a column are deduplicated; duplicate
/// input productions retain distinct prod ids. Standalone default fields are a
/// placeholder, not a claim that a production exists. Copies own their scalars.
/// @par Example
/// @code
/// const dm::algorithms::EarleyItem a{0,1,0};
/// std::print("{}\n",a==dm::algorithms::EarleyItem{0,1,0}); // true
/// @endcode
/// @see EarleyTrace, earley_item_text()
struct EarleyItem {
 int prod=0; ///< Zero-based original production index.
 int dot=0; ///< Matched RHS symbol count, including zero and complete RHS length.
 int origin=0; ///< Column where this production's match began.

 /// @brief Compare the three coordinates without interpreting a grammar.
 /// @param other Borrowed item.
 /// @return Whether all coordinates agree.
 /// @par Example
 /// See EarleyItem's executed equality example.
 /// @see EarleyItem
 bool operator==(const EarleyItem& other) const noexcept {return prod==other.prod&&dot==other.dot&&origin==other.origin;}
};

/// @brief Numeric location of an item in an owned Earley chart.
/// Both default coordinates are earley_no_item, denoting no parent. Validity
/// depends on the containing trace; copying does not retain a pointer into it.
/// @par Example
/// @code
/// const dm::algorithms::EarleyLocation location{0,1};
/// std::print("{} {} {}\n",location.column,location.index,location==dm::algorithms::EarleyLocation{0,1}); // 0 1 true
/// @endcode
/// @see EarleyReason, earley_item_text()
struct EarleyLocation {
 std::size_t column=earley_no_item; ///< Input boundary, or sentinel for absent parent.
 std::size_t index=earley_no_item; ///< Insertion ordinal within the column, or sentinel.

 /// @brief Compare both coordinates, including absent references.
 /// @param other Borrowed reference.
 /// @return Whether both coordinates agree; does not validate either reference.
 /// @par Example
 /// See EarleyLocation's coordinate example; equality compares those coordinates.
 /// @see EarleyLocation
 bool operator==(const EarleyLocation& other) const noexcept {return column==other.column&&index==other.index;}
};

/// @brief First successful insertion cause of one unique Earley item.
/// Seed has no parents; prediction/scanning use first; completion uses first as
/// the waiter and second as the completed item, including a late nullable join.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g;g.start="S";g.productions={{"S",{"a"}}};
/// const auto t=dm::algorithms::earley_trace(g,{"a"});
/// std::print("{}\n",t.reasons[0][0].action==dm::algorithms::EarleyAction::seed); // true
/// @endcode
/// @see EarleyReason
enum class EarleyAction {seed, ///< Original start-production seed.
 predict, ///< Dot before a nonterminal predicts one of its original productions.
 scan, ///< Matching terminal token advances the dot to the next column.
 complete ///< A completed span advances an item waiting for its nonterminal.
};

/// @brief Owned first-insertion evidence attached to one actual chart item.
/// Parents are numeric chart locations, always lexicographically earlier than the
/// target. This gives an acyclic first-witness deduction graph even for nullable
/// cycles. Duplicate attempts do not replace the first reason. It is not a packed
/// parse forest and does not enumerate all derivations. Default is a placeholder.
/// @par Example
/// @code
/// const dm::algorithms::EarleyReason reason;
/// std::print("{}\n",reason.first.column==dm::algorithms::earley_no_item); // true
/// @endcode
/// @see EarleyTrace, EarleyAction
struct EarleyReason {
 EarleyAction action=EarleyAction::seed; ///< Actual successful insertion operation.
 EarleyLocation first; ///< Predictor/scanner predecessor, or completion waiter; absent for seed.
 EarleyLocation second; ///< Completed item for a completion; otherwise absent.
};

/// @brief Actual logical recognition work, excluding compilation, hashes, allocation and capture copies.
/// Every item is processed once. Predictions run once per requested nonterminal
/// per column; completion joins each waiter with each distinct completed span once.
/// Counts are not execution time and checked uint64 arithmetic cannot wrap.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g;g.start="S";g.productions={{"S",{"A"}},{"A",{"a"}}};
/// const auto r=dm::algorithms::earley_recognize_work(g,{"a"});
/// std::print("{} {}\n",r.work.unique_items,r.work.completion_joins); // 4 1
/// @endcode
/// @see EarleyResult, EarleyTrace
struct EarleyWork {
 std::uint64_t items_processed=0; ///< Unique agenda items popped, including completed items.
 std::uint64_t prediction_batches=0; ///< Distinct requested (column,nonterminal) pairs.
 std::uint64_t predicted_productions=0; ///< Productions attempted by those batches, including duplicates.
 std::uint64_t scan_checks=0; ///< Terminal-expecting items, even in the final column.
 std::uint64_t scan_matches=0; ///< Matching token checks that attempt an advance.
 std::uint64_t completed_spans=0; ///< Distinct completed (nonterminal,origin,end) spans published.
 std::uint64_t completion_joins=0; ///< Waiter/span pairs attempted, including late nullable joins.
 std::uint64_t nullable_joins=0; ///< Joins when a waiter arrives after its zero-width span was published.
 std::uint64_t insert_attempts=0; ///< Seed, prediction, matching scan and completion insertion attempts.
 std::uint64_t duplicate_items=0; ///< Attempts whose triple already existed in the target column.
 std::uint64_t unique_items=0; ///< Successful new triples across all columns.
};

/// @brief Owned recognition verdict, legacy-compatible column sizes and actual work.
/// The temporary chart/indexes are discarded; no input or provenance is retained.
/// Default is an uncomputed placeholder. A computed chart_sizes has N+1 entries,
/// including empty columns, even for an undefined start or rejected input.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g;g.start="S";g.productions={{"S",{"a"}}};
/// const auto r=dm::algorithms::earley_recognize_work(g,{"a"});
/// std::print("{} {}\n",r.accepted,r.chart_sizes.size()); // true 2
/// @endcode
/// @see earley_recognize_work()
struct EarleyResult {
 bool accepted=false; ///< Whether an original start production spans the whole input.
 std::vector<int> chart_sizes; ///< Unique original-production item counts, in boundary order.
 EarleyWork work; ///< Counts from that same completed run.
};

/// @brief Owned actual chart and first-insertion reasons from one Earley recognition run.
/// Copies own independent grammar/token/chart buffers. Items keep insertion order
/// and original production ids; reasons[k][i] explains chart[k][i]. Full chart
/// closure supports epsilon, unit cycles, left recursion and ambiguity, but returns
/// no parse forest or count of parses. Empty RHS means epsilon; symbol strings,
/// including empty strings and embedded NUL, are opaque atomic names. A symbol is
/// nonterminal exactly when it occurs on a LHS. A missing start production rejects.
/// Logical capture cost is base+2I, where base=2N+P+sum(rhs lengths)+2 and I is the
/// number of unique items. Base counts start/LHS/RHS/token strings and one column
/// metadata record per boundary; each item and its reason count once. String bytes,
/// runtime indexes, scalar fields and allocator overhead are excluded. Editing
/// returned records may invalidate numeric references; no automatic repair occurs.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g;g.start="S";g.productions={{"S",{}}};
/// const auto t=dm::algorithms::earley_trace(g,{});
/// std::print("{} {}\n",t.accepted,t.captured_entries); // true 5
/// @endcode
/// @par References
/// - J. Earley (1970), An Efficient Context-Free Parsing Algorithm, CACM13(2):94-102.
///   doi:10.1145/362007.362035. Uses indexed waiters and late nullable completion,
///   without lookahead, a synthetic start rule, Leo optimization or a parse forest.
/// @see earley_trace(), earley_item_text()
struct EarleyTrace {
 ContextFreeGrammar grammar; ///< Owned original rules/start, including duplicate productions.
 std::vector<std::string> tokens; ///< Owned atomic token names in input order.
 std::vector<std::vector<EarleyItem>> chart; ///< Actual unique items in column/insertion order.
 std::vector<std::vector<EarleyReason>> reasons; ///< First causes, parallel to chart.
 bool accepted=false; ///< Whole-input recognition verdict from this chart.
 EarleyWork work; ///< Actual same-run logical work.
 std::size_t captured_entries=0; ///< Complete base+2I logical capture count.
};

/// @brief Check input geometry and compute capture cost before any item is inserted.
/// No allocation. Symbols are opaque strings, not validated as source-code names.
/// Token/production/RHS lengths and each eventual column item count must fit int;
/// base arithmetic must fit size_t. This count does not promise enough memory.
/// @param grammar Borrowed raw grammar, unchanged; missing start is allowed.
/// @param token_count Number of atomic input tokens, zero allowed, at most INT_MAX.
/// @return 2*token_count+productions.size()+sum(rhs sizes)+2 logical entries.
/// @throws std::length_error For index/count arithmetic limits.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g;g.start="S";g.productions={{"S",{"a"}}};
/// std::print("{}\n",dm::algorithms::earley_trace_base_entries(g,1)); // 6
/// @endcode
/// @see earley_trace()
[[nodiscard]] std::size_t earley_trace_base_entries(const ContextFreeGrammar& grammar,std::size_t token_count);

/// @brief Recognize a general CFG with indexed prediction/completion and checked logical work.
/// Supports epsilon and late waiters without iterating a vector while extending it.
/// All raw index/count validation precedes allocation. Inputs are borrowed unchanged;
/// temporary charts/indexes are discarded after preparing an owned summary. Duplicate
/// productions retain original ids. Missing start rejects, including a default empty
/// grammar. For fixed grammar, expected O(N^3) time and O(N^2) chart/index storage
/// assume ordinary constant expected hash access; string preparation is additional.
/// This implementation promises no unconditional worst-case hash bound, all-parses
/// enumeration, lookahead, LR(0) precomputation or Leo optimization. Calls share no
/// mutable state; borrowed inputs must not be concurrently modified.
/// @param grammar CFG with opaque string symbols and empty RHS for epsilon.
/// @param tokens Atomic token names; an empty vector is the empty word.
/// @return Whole-input verdict, N+1 legacy int column counts and actual work.
/// @throws std::length_error For input/item/counter/container bounds.
/// @throws std::bad_alloc For compilation/chart/index/summary allocations.
/// @par Example
/// See EarleyWork and EarleyResult for executed recognizer examples.
/// @par References
/// - J. Earley (1970), CACM13(2):94-102, doi:10.1145/362007.362035.
/// - J. Aycock and R. N. Horspool (2002), Practical Earley Parsing, Computer Journal
///   45(6):620-630, doi:10.1093/comjnl/45.6.620. Its discussion of late nullable
///   waiters motivates the indexed completion handshake; its LR(0) machine is not used.
/// @see earley_recognize(), earley_chart_sizes(), earley_trace()
[[nodiscard]] EarleyResult earley_recognize_work(const ContextFreeGrammar& grammar,const std::vector<std::string>& tokens);

/// @brief Preserve the original boolean Earley recognition interface.
/// Same grammar semantics, ownership, closure, complexity and errors as
/// earley_recognize_work(); discards only its summary. No optional capture quota.
/// @param grammar Borrowed CFG; missing start rejects.
/// @param tokens Borrowed atomic tokens; empty is allowed.
/// @return True iff the complete token word is derivable from the start.
/// @throws std::length_error For index/item/counter/container limits.
/// @throws std::bad_alloc For temporary recognition storage.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g;g.start="S";g.productions={{"S",{"a"}}};
/// std::print("{}\n",dm::algorithms::earley_recognize(g,{"a"})); // true
/// @endcode
/// @par References
/// - J. Earley (1970), CACM13(2):94-102, doi:10.1145/362007.362035.
/// @see earley_recognize_work()
inline bool earley_recognize(const ContextFreeGrammar& grammar,const std::vector<std::string>& tokens) {
 return earley_recognize_work(grammar,tokens).accepted;
}

/// @brief Preserve original unique-item counts at boundaries zero through N.
/// Same semantics, validation, ownership and complexity as earley_recognize_work().
/// These count dotted items, not parse trees. One grammar's bounded per-column
/// counts do not establish a general complexity bound. No optional capture quota.
/// @param grammar Borrowed CFG, including duplicate productions.
/// @param tokens Borrowed atomic token word.
/// @return Owned N+1 int counts; empty columns are zero, no synthetic start item.
/// @throws std::length_error For index/item/counter/container limits.
/// @throws std::bad_alloc For temporary chart and output storage.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g;g.start="S";g.productions={{"S",{"a"}}};
/// const auto sizes=dm::algorithms::earley_chart_sizes(g,{"a"});
/// std::print("{} {}\n",sizes[0],sizes[1]); // 1 1
/// @endcode
/// @par References
/// - J. Earley (1970), CACM13(2):94-102, doi:10.1145/362007.362035.
/// @see EarleyTrace, earley_recognize_work()
inline std::vector<int> earley_chart_sizes(const ContextFreeGrammar& grammar,const std::vector<std::string>& tokens) {
 return earley_recognize_work(grammar,tokens).chart_sizes;
}

/// @brief Capture actual unique items and first-insertion parents from the shared recognizer.
/// Complete raw input/count validation and the base-budget check precede copies
/// and column allocation. Each unique item reserves two logical records before
/// insertion; duplicate attempts consume none. Same closure/order as ordinary
/// recognition. Reasons are captured on insertion, without replay. No partial
/// result escapes on failure. Returned copies own all buffers; concurrent calls
/// share no mutable state when inputs are not modified concurrently.
/// @param grammar Borrowed opaque-symbol grammar; duplicate productions retain ids.
/// @param tokens Borrowed atomic tokens, including an empty word.
/// @param max_entries Complete base+2I logical cap, default 1000000; zero disables only quota.
/// @return Independent grammar, tokens, chart, first reasons, verdict and work.
/// @throws std::length_error For raw/item/counter/container/capture bounds.
/// @throws std::bad_alloc For owned input/chart/reason or temporary index storage.
/// @par Example
/// See EarleyTrace's executed nullable-input capture example.
/// @par References
/// - J. Earley (1970), CACM13(2):94-102, doi:10.1145/362007.362035.
/// - J. Aycock and R. N. Horspool (2002), Practical Earley Parsing, doi:10.1093/comjnl/45.6.620.
/// @see earley_trace_base_entries(), earley_item_text(), earley_recognize_work()
[[nodiscard]] EarleyTrace earley_trace(const ContextFreeGrammar& grammar,const std::vector<std::string>& tokens,std::size_t max_entries=1000000);

/// @brief Format one selected actual item with escaped symbols, a dot and its origin.
/// Validates the selected location, production, dot and origin, not the whole
/// mutable trace or its proof. Debug-quoted strings distinguish an empty named
/// token from an empty RHS and escape embedded control bytes. No mutation or I/O.
/// @param trace Borrowed owned analysis, including its original grammar/tokens.
/// @param location Column and insertion ordinal of the selected item.
/// @return Owned text: pID "lhs" -> matched . remaining (origin).
/// @throws std::invalid_argument For invalid selected location/item coordinates.
/// @throws std::bad_alloc If formatted text cannot be allocated.
/// @par Example
/// @code
/// dm::algorithms::ContextFreeGrammar g;g.start="S";g.productions={{"S",{"a"}}};
/// const auto t=dm::algorithms::earley_trace(g,{"a"});
/// std::print("{}\n",dm::algorithms::earley_item_text(t,{0,0})); // p0 "S" -> . "a" (0)
/// @endcode
/// @see EarleyItem, EarleyLocation, earley_trace()
[[nodiscard]] std::string earley_item_text(const EarleyTrace& trace,EarleyLocation location);
} // namespace datamunge::algorithms
