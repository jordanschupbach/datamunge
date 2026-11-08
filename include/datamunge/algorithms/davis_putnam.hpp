#pragma once
/// @file
/// @brief Checked propositional CNF elimination, actual formula traces and resolution-growth controls.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief One satisfiability-preserving variable-removal rule.
/// Unit and pure record the signed true literal; resolution records a positive
/// variable id. Each native step removes that variable entirely. No branching
/// assignment or first-order quantifier procedure is represented.
/// @par Example
/// @code
/// const auto t=dm::algorithms::davis_putnam_trace(1,{{1}});
/// std::print("{}\n",t.steps[0].rule==dm::algorithms::DpRule::unit); // true
/// @endcode
/// @see DpStep, davis_putnam_trace()
enum class DpRule { unit, ///< First lexicographically ordered unit clause.
                    pure, ///< Smallest signed literal whose opposite is absent.
                    resolution ///< Smallest occurring variable, both polarities present.
};

/// @brief Actual rule applications and parent-pair work from one eliminating solve.
/// Pair counts include tautological and duplicate conclusions. Peak counts
/// distinct non-tautological clauses in normalized input or any completed state;
/// excludes old/new coexistence, temporary unions and retained history. Counters
/// exclude validation, sorting, normalization, scans, allocation and snapshot copies.
/// Owned scalar values have no shared state; they are not elapsed-time estimates.
/// @par Example
/// @code
/// const auto t=dm::algorithms::davis_putnam_trace(2,{{1,2},{1,-2},{-1,2},{-1,-2}});
/// std::print("{} {} {}\n",t.work.resolution_steps,t.work.resolution_pairs,t.work.peak_clauses); // 1 4 4
/// @endcode
/// @see DpTrace
struct DpWork {
  std::size_t unit_steps{0}; ///< Number of actual forced-literal removals.
  std::size_t pure_steps{0}; ///< Number of actual one-polarity variable removals.
  std::size_t resolution_steps{0}; ///< Number of actual two-polarity eliminations.
  std::size_t resolution_pairs{0}; ///< Cumulative positive-times-negative parent pairs visited.
  std::size_t tautological_resolvents{0}; ///< Candidate unions containing opposite literals, discarded.
  std::size_t duplicate_resolvents{0}; ///< Non-tautological conclusions already in the next set.
  std::size_t peak_clauses{0}; ///< Maximum completed-state distinct clause count, including initial input.
};

/// @brief Owned result of one actual unit, pure or resolution elimination.
/// after is the complete normalized formula: ascending distinct literals per
/// clause, lexicographically ordered distinct clauses, no tautologies. Empty
/// after means true; a contained empty clause means false. No subsumption is
/// performed. Adjacent snapshots are related by existential elimination of
/// abs(literal), not equivalence with that variable left free. All counts refer
/// to this transition; positive/negative/pairs/tautologies/duplicates are zero
/// for unit/pure steps. No views into the caller or previous states remain.
/// @par Example
/// @code
/// const auto t=dm::algorithms::davis_putnam_trace(2,{{1,2},{1,-2},{-1,2},{-1,-2}});
/// std::print("{} {} {}\n",t.steps[0].literal,t.steps[0].pairs,t.steps[0].after.size()); // 1 4 2
/// @endcode
/// @see DpTrace, DpRule
struct DpStep {
  DpRule rule{DpRule::unit}; ///< Rule actually used.
  int literal{0}; ///< Signed true unit/pure literal, or positive resolution variable.
  std::size_t before_clauses{0}; ///< Distinct clauses before this transition.
  std::size_t positive_clauses{0}; ///< Resolution parents containing the positive pivot.
  std::size_t negative_clauses{0}; ///< Resolution parents containing the negative pivot.
  std::size_t pairs{0}; ///< Positive times negative parent counts, including rejected unions.
  std::size_t tautologies{0}; ///< Tautological conclusions discarded at this step.
  std::size_t duplicates{0}; ///< Already-present non-tautological conclusions at this step.
  std::vector<std::vector<int>> after; ///< Complete owned formula after the actual step.
};

/// @brief Complete owned normalized input, eliminating history and SAT verdict.
/// Native ordering is deterministic: check empty clauses/sets first, then first
/// unit, then smallest signed pure literal, then smallest absolute variable.
/// All formulas are canonical as in DpStep. num_vars is a domain bound, not an
/// array allocation: unused and very large sparse ids cost no dense storage.
/// Every step eliminates at least one occurring variable; at most that many
/// steps exist. No satisfying assignment or independent proof certificate is
/// returned. Independent calls share no mutable state; public edits are unchecked.
/// @par Example
/// @code
/// const auto t=dm::algorithms::davis_putnam_trace(1,{{1},{-1}});
/// std::print("{} {} {}\n",t.satisfiable,t.steps.size(),t.steps.back().after[0].empty()); // false 1 true
/// @endcode
/// @see davis_putnam_trace(), davis_putnam()
struct DpTrace {
  int num_vars{0}; ///< Nonnegative caller-supplied variable bound.
  bool satisfiable{false}; ///< Exact propositional SAT verdict on successful return.
  std::vector<std::vector<int>> initial; ///< Owned normalized input, before any removal rule.
  std::vector<DpStep> steps; ///< Actual successive transitions, without replay.
  DpWork work; ///< Checked actual elimination and parent-pair counts.
};

/// @brief Decide propositional CNF satisfiability by Davis-Putnam elimination.
/// Literal +v/-v denotes variable v true/false, for 1<=v<=num_vars.
/// Empty CNF is true; any empty clause makes it false. Duplicate literals and
/// clauses are idempotent; tautological clauses are removed. Complete domain
/// validation precedes all verdicts, including trivially true/false inputs.
/// Uses the same executing kernel as davis_putnam_trace(), without retaining
/// history or optional limits. Each step removes a variable. With C active
/// clauses, width W and p/q pivot parents, a resolution step visits pq pairs,
/// with sorted-set unions and ordered-clause insertion costing conservatively
/// O(pq W log(W+1) log(C+pq+1)); scans/normalization and allocation are additional.
/// Time and memory can grow exponentially in occurring variables; no polynomial
/// or resource guarantee is implied. Caller input is unchanged; calls are independent.
/// @param num_vars Nonnegative maximum variable id, no dense allocation implied.
/// @param clauses_in Clauses of nonzero literals in [-num_vars,num_vars]; INT_MIN invalid.
/// @return True exactly when at least one assignment satisfies the full CNF.
/// @throws std::invalid_argument Negative bound, zero, INT_MIN or out-of-range literal.
/// @throws std::length_error Size/counter arithmetic or container capacity cannot be represented.
/// @throws std::bad_alloc Allocation fails, potentially after clause growth.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::davis_putnam(2,{{1,2},{-1,2}}),dm::algorithms::davis_putnam(1,{{1},{-1}})); // true false
/// @endcode
/// @par References
/// - M. Davis and H. Putnam (1960), A Computing Procedure for Quantification Theory, JACM 7(3), 201–215, doi:10.1145/321033.321034. This API implements its finite propositional elimination component, with canonical deduplication and deterministic ordering.
/// @see davis_putnam_trace()
bool davis_putnam(int num_vars, const std::vector<std::vector<int>>& clauses_in);

/// @brief Trace the actual Davis-Putnam kernel with complete owned formula snapshots.
/// Same normalization, rules and verdict as davis_putnam(), without replay.
/// The optional clause cap bounds each distinct normalized set during construction.
/// The cumulative pair cap is checked before an entire p*q resolution loop.
/// Snapshot entries count one per snapshot plus one per clause plus one per
/// literal, including initial and every after state; checked before copying that
/// snapshot. Complete domain validation precedes these limits. Zero disables
/// only an optional limit; overflow/container checks remain. On exceeding a
/// limit, throws with no partial result or SAT/UNSAT verdict. Caps do not bound
/// caller memory, input clause width, temporary unions, allocator bytes or time.
/// Live old/new formulas and all snapshots may coexist. Trace overhead is linear
/// in retained entries; kernel costs are documented on davis_putnam().
/// @param num_vars Nonnegative maximum variable id.
/// @param clauses_in Borrowed CNF, unchanged; same literal domain as davis_putnam().
/// @param max_clauses Maximum distinct clauses in each normalized set; default 100000, zero disables.
/// @param max_pairs Maximum cumulative visited resolution pairs; default 1000000, zero disables.
/// @param max_trace_entries Maximum cumulative snapshot entries; default 1000000, zero disables.
/// @return Owned initial formula, all actual transitions, counters and exact verdict.
/// @throws std::invalid_argument Invalid bound or literal, even inside a tautology.
/// @throws std::length_error Optional limit, arithmetic overflow or container capacity exceeded.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::davis_putnam_trace(2,{{1,2},{1,-2},{-1,2},{-1,-2}});
/// std::print("{} {} {}\n",t.satisfiable,t.work.resolution_pairs,t.steps.size()); // false 4 2
/// @endcode
/// @par References
/// - M. Davis and H. Putnam (1960), doi:10.1145/321033.321034; finite propositional rules with sorted deduplication.
/// @see davis_putnam(), DpTrace, davis_putnam_grid_formula()
DpTrace davis_putnam_trace(int num_vars, const std::vector<std::vector<int>>& clauses_in,
                          std::size_t max_clauses=100000, std::size_t max_pairs=1000000,
                          std::size_t max_trace_entries=1000000);

/// @brief Construct a square parent-grid CNF exposing one resolution growth step.
/// For k=order, x=1, a_i=2+i and b_j=2+k+j (zero-based), returns k
/// clauses (x OR a_i), k clauses (NOT x OR b_j), and the two clauses
/// (NOT a_0 OR ... OR NOT a_(k-1)), (NOT b_0 OR ... OR NOT b_(k-1)).
/// Uses variables 1..2k+1 and returns 2k+2 clauses, in that construction order,
/// without normalization. Always UNSAT: x=false forces every a true; x=true
/// forces every b true; each contradicts a negative tail. At k>=2 no initial
/// unit/pure exists; eliminating x yields k*k distinct two-literal resolvents
/// plus the two tails. At k=1 unit propagation takes priority. Owned output,
/// O(k) clauses/literals/time/storage; no randomness, shared state or solver call.
/// @param order Positive k with 2k+1 representable as int.
/// @param max_clauses Optional output clause-count cap, default 100000; zero disables.
/// @return Complete owned constructed CNF with exactly 2k+2 clauses and 6k literals.
/// @throws std::invalid_argument order is zero.
/// @throws std::length_error Variable ids, counts, capacity or optional cap exceeded, checked before allocation.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto f=dm::algorithms::davis_putnam_grid_formula(3);
/// const auto t=dm::algorithms::davis_putnam_trace(7,f);
/// std::print("{} {} {}\n",f.size(),t.steps[0].pairs,t.steps[0].after.size()); // 8 9 11
/// @endcode
/// @see davis_putnam_trace()
std::vector<std::vector<int>> davis_putnam_grid_formula(std::size_t order,
                                                       std::size_t max_clauses=100000);
} // namespace datamunge::algorithms
