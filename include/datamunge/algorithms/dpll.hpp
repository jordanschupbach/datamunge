#pragma once
/// @file dpll.hpp
/// @brief Validated deterministic DPLL, actual iterative search evidence and reproducible random3-CNF.
#include <cstddef>
#include <random>
#include <vector>

namespace datamunge::algorithms {
/// @brief Original SAT verdict and owned total model, preserving the historical two-field layout.
/// On SAT, assignment has num_vars+1 entries, index0 is zero and every variable
/// is +1 or -1; don't-cares are filled true. On UNSAT, assignment is empty.
/// Default construction is an uncomputed false/empty placeholder. No borrowed views.
/// @par Example
/// @code
/// const auto s=dm::algorithms::dpll_sat(1,{{1}});
/// std::print("{} {}\n",s.satisfiable,s.assignment); // true [0, 1]
/// @endcode
/// @see dpll_sat(), cnf_assignment_satisfies()
struct SatResult {
  bool satisfiable{false}; ///< Exact verdict on successful return; false by default.
  std::vector<int> assignment; ///< Owned total signed model on SAT; empty on UNSAT.
};

/// @brief Kind of an actual iterative DPLL transition or terminal node.
/// Enter records root/branch arrival; unit/pure record post-assignment states;
/// conflict identifies a falsified original clause; exhausted means both children
/// failed, not that its own partial state falsifies a clause. Backtrack restores
/// the parent state. Satisfied records a partial state already satisfying every
/// clause, before filling don't-cares in the separate final total model.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dpll_sat_trace(1,{{1}});
/// std::print("{}\n",t.events.front().kind==dm::algorithms::DpllEventKind::enter); // true
/// @endcode
/// @see DpllEvent
enum class DpllEventKind {
  enter, ///< Arrive at a newly numbered node; literal zero at root, signed branch literal otherwise.
  unit, ///< Assign the sole unassigned occurrence in an unsatisfied original clause.
  pure, ///< Assign the smallest variable with one polarity among unsatisfied clauses.
  conflict, ///< A first original clause has all its literals false.
  backtrack, ///< Return from a failed child; literal is that child's signed decision.
  exhausted, ///< Both true/false branches failed at this internal node.
  satisfied ///< Every original clause already contains a true literal.
};

/// @brief Checked logical work of the shared DPLL kernel, excluding validation/copies/allocation.
/// nodes counts root and branch entries. decisions counts split points, branches
/// counts attempted children; nodes=branches+1. Clause visits count outer
/// clause-loop iterations of status/unit/pure scans. Literal visits count each
/// inspected occurrence, including both passes over unsatisfied clauses in the
/// pure scan; a second literal pass does not add another outer clause visit.
/// Counters are operations for this implementation, not CPU timings or CDCL work.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dpll_sat_work(2,{{1},{-1,2}});
/// std::print("{} {} {}\n",r.work.nodes,r.work.unit_steps,r.work.decisions); // 1 2 0
/// @endcode
/// @see DpllMeasurement, dpll_sat_work()
struct DpllWork {
  std::size_t nodes=0; ///< Entered nodes, including root.
  std::size_t decisions=0; ///< Nodes splitting on the first unassigned variable.
  std::size_t branches=0; ///< True/false child attempts actually entered.
  std::size_t unit_steps=0; ///< Actual forced assignments.
  std::size_t pure_steps=0; ///< Actual satisfiability-preserving pure assignments.
  std::size_t conflicts=0; ///< Terminal nodes with a falsified original clause.
  std::size_t backtracks=0; ///< Failed-child returns to a restored parent.
  std::size_t max_depth=0; ///< Maximum decision depth; root has depth zero.
  std::size_t clause_visits=0; ///< Clauses actually inspected across all status/unit/pure scans.
  std::size_t literal_visits=0; ///< Literal occurrences inspected in those scans, including repeats.
};

/// @brief Owned actual model/verdict and work without search-history capture.
/// Carries no borrowed input references. Defaults are an uncomputed placeholder.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dpll_sat_work(1,{{1},{-1}});
/// std::print("{} {}\n",r.result.satisfiable,r.work.conflicts); // false 1
/// @endcode
/// @see dpll_sat_work(), DpllWork
struct DpllMeasurement {
  SatResult result; ///< Original total-model contract, with empty model on UNSAT.
  DpllWork work; ///< Actual checked logical counters.
};

/// @brief Owned actual partial assignment at a search event, with original clause identities.
/// assignment has num_vars+1 values: index0=0; other values0(unassigned),+1,-1.
/// node ids start at0 in entry order. Root parent uses SIZE_MAX. Unit/conflict
/// clause is the original zero-based index; all other kinds use clauses.size().
/// No formula normalization, assignment replay or borrowed state is represented.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dpll_sat_trace(2,{{1},{-1,2}});
/// std::print("{} {}\n",t.events[1].literal,t.events[1].clause); // 1 0
/// @endcode
/// @see DpllEventKind, DpllTrace
struct DpllEvent {
  DpllEventKind kind=DpllEventKind::enter; ///< Actual event kind; default is an uncomputed placeholder.
  std::size_t node=0; ///< Current node, including restored parent at backtrack.
  std::size_t parent=0; ///< Parent id, or SIZE_MAX at the root.
  std::size_t depth=0; ///< Current node's decision depth.
  int literal=0; ///< Assigned/entered/failed-child literal; zero for other events.
  std::size_t clause=0; ///< Unit/conflict clause index, otherwise input clause-count sentinel.
  std::vector<int> assignment; ///< Owned actual partial state at this event.
};

/// @brief Owned original CNF, actual search events, total result and work from one solve.
/// No replay: the same executing iterative kernel implements ordinary/work calls.
/// Trace entries count1 per formula plus1 per clause plus1 per input literal,
/// then1 per event plus1 per captured assignment element. The cap excludes scalar
/// metadata, working stack, allocator bytes and elapsed time. Root/event capacity
/// is checked before the affected capture; all input validation precedes limits.
/// Zero disables only the optional cap. A cap exception returns no partial verdict.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dpll_sat_trace(2,{{1},{-1,2}});
/// std::print("{} {}\n",t.events.size(),t.captured_entries); // 4 22
/// @endcode
/// @see dpll_sat_trace(), DpllEvent
struct DpllTrace {
  int num_vars=0; ///< Validated nonnegative variable bound.
  std::vector<std::vector<int>> clauses; ///< Owned original CNF, preserving order/repeats/tautologies.
  DpllMeasurement solve; ///< Total model/verdict and logical work.
  std::vector<DpllEvent> events; ///< Actual event order, including failure returns and exhaustion.
  std::size_t captured_entries=0; ///< Logical history entries as documented above.
};

/// @brief Decide validated propositional CNF satisfiability by deterministic DPLL.
/// Keeps the original signature, SatResult layout and search order: first raw
/// unit occurrence, then smallest-id pure variable, then first unassigned variable
/// true before false. Original clause/literal order, repetitions and tautologies
/// remain; duplicate occurrences can delay unit detection without changing SAT.
/// Empty CNF is true; an empty clause is false. Complete domain validation precedes
/// any verdict or dense allocation. Uses a heap-backed iterative frame stack,
/// avoiding C++ recursion by decision depth. Dense assignments use O(n) space each,
/// up to O(n^2+L) live storage for n variables and L input literals. Search can visit
/// O(2^n) nodes, with repeated polynomial scans per node; no learning/restarts,
/// watched literals, timeout or polynomial/resource guarantee is provided.
/// Input is unchanged; calls share no mutable state.
/// @param num_vars Nonnegative count, variable ids1..num_vars; dense models use n+1 entries.
/// @param clauses Borrowed CNF of nonzero signed ids, within the bound; INT_MIN invalid.
/// @return Exact SAT verdict and a total owned model, or false/empty on UNSAT.
/// @throws std::invalid_argument For negative bound, zero/INT_MIN/out-of-range literals.
/// @throws std::length_error If size/counter arithmetic or container capacity overflows.
/// @throws std::bad_alloc If dense/stack/result storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::dpll_sat(1,{{1},{-1}}).satisfiable); // false
/// @endcode
/// @par References
/// - M. Davis, G. Logemann and D. Loveland (1962), A machine program
///   for theorem-proving, CACM5(7),394-397. https://doi.org/10.1145/368273.368557
/// @see dpll_sat_work(), dpll_sat_trace(), cnf_assignment_satisfies()
SatResult dpll_sat(int num_vars,const std::vector<std::vector<int>>& clauses);

/// @brief Solve with actual logical work and no retained event/input history.
/// Same validation, search order, model, complexity and ownership as dpll_sat().
/// @param num_vars Nonnegative dense variable count.
/// @param clauses Borrowed original CNF, unchanged.
/// @return Owned total model/verdict and actual work counters.
/// @throws std::invalid_argument For invalid count/literal domain.
/// @throws std::length_error For count/size/container overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dpll_sat_work(2,{{1},{-1,2}});
/// std::print("{} {}\n",r.result.satisfiable,r.work.unit_steps); // true 2
/// @endcode
/// @see dpll_sat(), DpllWork
DpllMeasurement dpll_sat_work(int num_vars,const std::vector<std::vector<int>>& clauses);

/// @brief Capture actual DPLL nodes, propagation, conflicts and restored backtracks.
/// Same executing kernel as dpll_sat(); snapshots store raw original-clause ids
/// and partial states, not a replay or clause-learning proof. UNSAT is justified
/// by completed search, with explicit exhausted events for both-failed splits.
/// See DpllTrace for entry accounting and exclusions. Capture adds O(events*n)
/// copying/storage to the ordinary kernel; default cap is one million entries.
/// @param num_vars Nonnegative dense count.
/// @param clauses Borrowed CNF; complete validation precedes capture limits.
/// @param max_trace_entries Optional logical cap, default1000000; zero disables only cap.
/// @return Owned input, actual event sequence, total result and work.
/// @throws std::invalid_argument For invalid count/literal domain.
/// @throws std::length_error For cap, count/size/container overflow, before affected capture.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dpll_sat_trace(2,{{1},{-1,2}},22);
/// std::print("{}\n",t.solve.result.assignment); // [0, 1, 1]
/// @endcode
/// @par References
/// - M. Davis, G. Logemann and D. Loveland (1962), A machine program
///   for theorem-proving, CACM5(7),394-397. https://doi.org/10.1145/368273.368557
/// @see dpll_sat(), DpllTrace
DpllTrace dpll_sat_trace(int num_vars,const std::vector<std::vector<int>>& clauses,
                         std::size_t max_trace_entries=1000000);

/// @brief Check a supplied total signed model against every original CNF clause.
/// Fully validates clauses and model domain first. Index0 is ignored; positions
///1..n must be +/-1. Empty CNF is satisfied, a contained empty clause is not.
/// O(n+L) time, O(1) scratch, no allocation/mutation or shared state.
/// @param num_vars Nonnegative variable count.
/// @param clauses Borrowed original CNF, unchanged.
/// @param assignment Exactly n+1 entries; index0 unused.
/// @return True exactly when the valid supplied model satisfies all clauses.
/// @throws std::invalid_argument For invalid count/literal, model shape or non-total variable value.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cnf_assignment_satisfies(2,{{1,-2}},{0,-1,-1})); // true
/// @endcode
/// @see dpll_sat()
bool cnf_assignment_satisfies(int num_vars,const std::vector<std::vector<int>>& clauses,
                             const std::vector<int>& assignment);

/// @brief Generate the original report's reproducible three-distinct-variable random CNF stream.
/// Each clause samples rng()%n ids, rejects duplicate ids before drawing its sign,
/// then uses rng()&1 for polarity. Clause duplicates are allowed; no tautological
/// clause can occur. This preserves the original mt19937 draw order; modulo bias
/// means it is not an exactly uniform mathematical3-SAT sampler. O(m) expected
/// generation work for n>=3; rejection has no deterministic draw-count bound.
/// The full literal count/cap/domain are checked before allocation or engine mutation.
/// @param num_vars At least3, maximum id fits int.
/// @param num_clauses Number of output clauses, each of width3.
/// @param engine Borrowed mt19937 state, advanced only after successful validation.
/// @param max_literals Optional total literal cap, default1000000; zero disables only cap.
/// @return Owned original-order CNF with exactly3*num_clauses literal occurrences.
/// @throws std::invalid_argument If num_vars<3.
/// @throws std::length_error For count/cap/container overflow before generation.
/// @throws std::bad_alloc On allocation failure; engine may then be partly advanced.
/// @par Example
/// @code
/// std::mt19937 rng(4242);
/// const auto f=dm::algorithms::random_3sat_formula(25,50,rng);
/// std::print("{} {}\n",f.size(),f[0].size()); // 50 3
/// @endcode
/// @par References
/// - D. Mitchell, B. Selman and H. Levesque (1992), Hard and Easy Distributions of SAT Problems,
///   AAAI92,459-465. https://cdn.aaai.org/AAAI/1992/AAAI92-071.pdf (experiment model; the engine/modulo stream here preserves the original report).
/// @see dpll_sat_work()
std::vector<std::vector<int>> random_3sat_formula(int num_vars,std::size_t num_clauses,
                                                std::mt19937& engine,std::size_t max_literals=1000000);
} // namespace datamunge::algorithms
