#pragma once
/// @file
/// @brief Byte-oriented binary-normal-form CYK recognition, actual charts and first parse witnesses.
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {
/// @brief Epsilon-free binary-normal-form grammar with arbitrary integer nonterminal ids.
/// The original three-field layout is retained. unary maps one terminal byte
/// to the ids A with A->byte; it does not represent nonterminal unit rules.
/// binary entries {A,B,C} mean A->BC. Negative and sparse ids, including signed
/// extremes, are valid. Repeated productions are semantically idempotent for
/// recognition. No epsilon, unit-production or general-CFG conversion is supplied.
/// The start may appear on a right-hand side because no epsilon rule exists.
/// Public containers own their data; subsequent analyses own their results.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['x']={0};
/// std::print("{} {}\n",dm::algorithms::cyk_recognize(g,"x"),dm::algorithms::cyk_recognize(g,"")); // true false
/// @endcode
/// @par References
/// - D. H. Younger (1967), Recognition and parsing of context-free languages in time n^3, Information and Control 10(2), 189–208, doi:10.1016/S0019-9958(67)80007-X.
/// @see cyk_recognize(), cyk_analyze()
struct CNFGrammar {
  int start=0; ///< Start id, any int; default zero.
  std::unordered_map<char,std::vector<int>> unary; ///< Terminal byte to production left-hand ids; missing byte derives nothing.
  std::vector<std::array<int,3>> binary; ///< Ordered {lhs,left,right} productions; order selects first witnesses.
};

/// @brief Checked CYK geometry and worst-case dense-state/rule-loop sizes.
/// cells=n(n+1)/2, splits=n(n-1)(n+1)/6, state_slots=cells*m,
/// max_rule_tests=splits*r. These are potential capacities/work, not counts of
/// successful derivations. Empty input has all four counts zero. No ownership
/// beyond scalar values and no shared state.
/// @par Example
/// @code
/// const auto p=dm::algorithms::cyk_shape(4,4,3);
/// std::print("{} {} {} {}\n",p.cells,p.splits,p.state_slots,p.max_rule_tests); // 10 10 40 30
/// @endcode
/// @see cyk_shape(), CykAnalysis
struct CykShape {
  std::size_t input_size{0}; ///< Number n of terminal bytes.
  std::size_t nonterminal_count{0}; ///< Number m of distinct ids, including the start.
  std::size_t binary_rule_count{0}; ///< Number r of raw binary productions, including duplicates.
  std::size_t cells{0}; ///< Nonempty contiguous spans, n(n+1)/2.
  std::size_t splits{0}; ///< All positive two-child split points over all spans.
  std::size_t state_slots{0}; ///< Dense membership capacity cells*m.
  std::size_t max_rule_tests{0}; ///< Worst-case splits*r candidate rule tests.
};

/// @brief Checked actual operations from the same CYK kernel that recognizes the input.
/// Terminal candidates count raw matching unary-list entries, including duplicates.
/// split_points includes every split, even when an empty child skips the binary
/// rule loop. binary_rule_tests counts rules entered; matching_rules counts those
/// whose two children are present, including repeated/already-known conclusions.
/// new_states counts distinct (span,nonterminal) insertions. No replay or timing
/// estimates; empty input has zero work. Counters exclude grammar preparation,
/// dense initialization, chart extraction, allocation and witness reconstruction.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
/// const auto a=dm::algorithms::cyk_analyze(g,"aaaa");
/// std::print("{} {} {}\n",a.work.split_points,a.work.binary_rule_tests,a.work.new_states); // 10 10 10
/// @endcode
/// @see cyk_analyze(), CykAnalysis
struct CykWork {
  std::size_t terminal_lookups{0}; ///< One compiled byte-table lookup per input position.
  std::size_t terminal_rule_tests{0}; ///< Raw matching terminal candidates examined.
  std::size_t split_points{0}; ///< Actual positive splits visited across every span.
  std::size_t skipped_empty_splits{0}; ///< Splits whose left or right cell was empty.
  std::size_t binary_rule_tests{0}; ///< Raw binary rules entered after nonempty-child checks.
  std::size_t matching_rules{0}; ///< Rules whose two child memberships both hold.
  std::size_t new_states{0}; ///< Distinct cell memberships first inserted by any rule.
};

/// @brief First actual derivation establishing one nonterminal in a chart cell.
/// left_length=0 denotes a length-one terminal rule; terminal is then the input
/// byte, left/right are unused zero placeholders and binary_rule is SIZE_MAX.
/// Otherwise the children have original ids left/right, with lengths left_length
/// and cell.length-left_length. binary_rule indexes the original ordered binary
/// vector. Length, start and input live in the containing cell/analysis. A first
/// witness does not count or exclude alternative parses.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
/// const auto a=dm::algorithms::cyk_analyze(g,"aa");
/// const auto& e=dm::algorithms::cyk_cell(a,0,2).entries[0];
/// std::print("{} {} {}\n",e.nonterminal,e.left_length,e.binary_rule); // 0 1 0
/// @endcode
/// @see CykCell, cyk_first_parse()
struct CykEntry {
  int nonterminal{0}; ///< Original id deriving this cell's substring.
  std::size_t left_length{0}; ///< Positive split for binary witnesses; zero for a terminal witness.
  int left{0}; ///< Original left child id for binary rules; unused zero for a terminal.
  int right{0}; ///< Original right child id for binary rules; unused zero for a terminal.
  std::size_t binary_rule{std::numeric_limits<std::size_t>::max()}; ///< Original rule index, or SIZE_MAX for a terminal.
  char terminal{0}; ///< Actual byte for terminal witnesses; unused zero for binary witnesses.
};

/// @brief Owned membership and first witnesses for a nonempty half-open byte span.
/// entries is sorted by original nonterminal id and contains at most one entry
/// per id. Empty entries means no nonterminal derives the span. Storage contains
/// no references into grammar/input; begin is zero-based and length is positive.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['a']={-7,3};
/// const auto a=dm::algorithms::cyk_analyze(g,"a");
/// const auto& c=dm::algorithms::cyk_cell(a,0,1);
/// std::print("{} {} {}\n",c.begin,c.length,c.entries[0].nonterminal); // 0 1 -7
/// @endcode
/// @see cyk_cell(), CykEntry
struct CykCell {
  std::size_t begin{0}; ///< Starting byte offset in the owned input.
  std::size_t length{0}; ///< Positive byte length; native results never use zero-length cells.
  std::vector<CykEntry> entries; ///< Sorted distinct memberships and first actual witnesses.
};

/// @brief Complete owned CYK result, triangular chart and actual work.
/// cells is ordered by increasing length, then begin; all cells, including empty
/// ones, are retained. nonterminals lists all grammar ids, including unreachable
/// ids and start, in ascending order. First witnesses try increasing left length
/// then original binary-rule order. Duplicate terminal/binary rules do not create
/// duplicate memberships. No input/grammar views or shared mutable state remain.
/// Empty input rejects and has an empty chart; no epsilon convention is inferred.
/// Public edits are possible; helper access/reconstruction checks local structure
/// rather than certifying an edited result against the original grammar.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
/// const auto a=dm::algorithms::cyk_analyze(g,"aaa");
/// std::print("{} {} {}\n",a.accepted,a.cells.size(),a.nonterminals); // true 6 [0]
/// @endcode
/// @see cyk_analyze(), cyk_cell(), cyk_first_parse()
struct CykAnalysis {
  std::string input; ///< Owned original byte string, including embedded NULs.
  int start_symbol{0}; ///< Copied original start id.
  bool accepted{false}; ///< Start membership in the full span, false for empty input.
  std::vector<int> nonterminals; ///< Sorted distinct original grammar ids, including start.
  CykShape shape; ///< Checked potential geometry and work counts.
  CykWork work; ///< Actual terminal/split/rule/membership operations.
  std::vector<CykCell> cells; ///< Complete triangle in (length,begin) order, without zero-length spans.
};

/// @brief One node of an owned first-witness parse in preorder.
/// A leaf has length one, terminal equal to its input byte and both child indices
/// SIZE_MAX. A binary node's children partition its half-open span and follow
/// their parent in preorder; terminal is unused zero. Nodes contain original ids.
/// The returned vector's indices remain valid until the caller mutates it.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
/// const auto p=dm::algorithms::cyk_first_parse(dm::algorithms::cyk_analyze(g,"aa"));
/// std::print("{} {} {}\n",p.size(),p[0].left_child,p[0].right_child); // 3 1 2
/// @endcode
/// @see cyk_first_parse()
struct CykParseNode {
  int nonterminal{0}; ///< Original nonterminal id.
  std::size_t begin{0}; ///< Starting input byte offset.
  std::size_t length{0}; ///< Positive byte length of this node's yield.
  std::size_t left_child{std::numeric_limits<std::size_t>::max()}; ///< Preorder child index, or SIZE_MAX for a leaf.
  std::size_t right_child{std::numeric_limits<std::size_t>::max()}; ///< Preorder child index, or SIZE_MAX for a leaf.
  char terminal{0}; ///< Input byte at a leaf, unused zero at a binary node.
};

/// @brief Compute exact CYK span, split and dense-state capacities without allocating.
/// Cancels factors before multiplying, so representable results are not rejected
/// merely because an uncancelled product overflows. All returned counts must fit
/// size_t, even with zero binary rules. No mutation/shared state, O(1) time/storage.
/// @param input_size Number n of bytes, including zero.
/// @param nonterminal_count Distinct-id count m, at least one for the start.
/// @param binary_rule_count Raw binary-production count r, including zero.
/// @return Checked geometry; this function alone does not enforce analysis budgets.
/// @throws std::invalid_argument nonterminal_count is zero.
/// @throws std::length_error Any returned count cannot fit size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cyk_shape(6,4,3).splits); // 35
/// @endcode
/// @see CykShape, cyk_analyze()
CykShape cyk_shape(std::size_t input_size,std::size_t nonterminal_count,std::size_t binary_rule_count);

/// @brief Analyze every substring with the canonical byte-oriented CYK kernel.
/// Grammar semantics are those of CNFGrammar. Ids are compressed by sorting,
/// never by max id; compiled byte/rule memberships use dense indices. Complexity
/// is O(G log(G+1)+n*G+n^2*m+n^3*(r+1)) time and O(G+n^2*m) storage, including dense
/// initialization, first choices and chart extraction; G counts raw grammar ids.
/// For a fixed grammar the usual cubic bound applies. This computes membership
/// and one witness, not ambiguity counts, probabilities, all parses or CFG conversion.
/// Grammar preparation precedes the complete potential-capacity checks; those
/// checks precede input/chart/choice allocation and parsing. Budgets bound possible
/// state slots cells*m and binary-rule tests splits*r, rather than actual successes
/// or actual tests after empty-child skips. Zero disables only the corresponding
/// optional budget. No truncation. Empty input rejects with no chart/work.
/// @param g Read-only epsilon-free binary-normal-form grammar; all int ids valid.
/// @param s Read-only byte string, including unknown bytes and embedded NULs.
/// @param max_states Potential dense membership cap, default 1000000; zero disables.
/// @param max_rule_tests Worst-case binary-candidate cap, default 100000000; zero disables.
/// @return Complete owned chart, first witnesses, geometry and actual work.
/// @throws std::length_error Checked geometry/container capacity or optional cap exceeded.
/// @throws std::overflow_error An actual aggregate counter cannot fit size_t.
/// @throws std::bad_alloc Grammar/input/chart/history allocation failure.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
/// const auto a=dm::algorithms::cyk_analyze(g,"aaaa",10,10);
/// std::print("{} {}\n",a.accepted,a.work.matching_rules); // true 10
/// @endcode
/// @par References
/// - D. H. Younger (1967), doi:10.1016/S0019-9958(67)80007-X; binary-normal-form substring dynamic programming.
/// @see cyk_recognize(), cyk_shape(), cyk_first_parse()
CykAnalysis cyk_analyze(const CNFGrammar& g,const std::string& s,std::size_t max_states=1000000,std::size_t max_rule_tests=100000000);

/// @brief Borrow one checked cell of a native or locally edited CYK analysis.
/// O(1) checked triangular indexing. Validates requested bounds, full cell-vector
/// length and selected cell coordinates; it does not validate every membership or
/// certify grammar derivability. Borrow ends when the owner/cells are mutated or
/// destroyed. Empty input has no valid cell. No allocation or shared state.
/// @param analysis Owned result to inspect, unchanged.
/// @param begin Zero-based byte offset.
/// @param length Positive byte length entirely within input.
/// @return Const reference into analysis.cells.
/// @throws std::out_of_range Zero length or requested span outside the input.
/// @throws std::invalid_argument Cell-vector size or selected coordinates are malformed.
/// @throws std::length_error Geometry cannot be represented.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['x']={0};
/// const auto a=dm::algorithms::cyk_analyze(g,"x");
/// std::print("{}\n",dm::algorithms::cyk_cell(a,0,1).entries.size()); // 1
/// @endcode
/// @see CykCell, cyk_analyze()
const CykCell& cyk_cell(const CykAnalysis& analysis,std::size_t begin,std::size_t length);

/// @brief Recover one complete parse using the stored first witnesses, without rerunning CYK.
/// Returns empty for a native rejected/empty result; accepted n-byte inputs give
/// exactly 2n-1 nodes in preorder. Iterative traversal avoids recursive stack depth.
/// Validates full chart length, root acceptance consistency and visited local
/// witness uniqueness, terminal bytes, rule-index bounds and proper child spans.
/// Public edits are not certified against the original grammar (which is not
/// retained). Native reconstruction takes O(n*m) entry-search time and O(n) extra
/// storage. The complete node cap is checked before node/task allocation; zero
/// disables only this optional cap. No caller mutation or retained references.
/// @param analysis Read-only complete analysis with native or locally valid witnesses.
/// @param max_nodes Complete 2n-1 output cap, default 100000; zero disables.
/// @return Owned preorder nodes, or empty for rejection.
/// @throws std::invalid_argument Malformed chart/root or visited witness structure.
/// @throws std::length_error Node count/container capacity or optional cap exceeded.
/// @throws std::bad_alloc Output/task allocation failure.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
/// std::print("{}\n",dm::algorithms::cyk_first_parse(dm::algorithms::cyk_analyze(g,"aaa"),5).size()); // 5
/// @endcode
/// @see CykParseNode, cyk_analyze()
std::vector<CykParseNode> cyk_first_parse(const CykAnalysis& analysis,std::size_t max_nodes=100000);

/// @brief Internal canonical recognition entry used by the retained inline facade.
namespace cyk_detail {
/// @brief Run the same prepared CYK kernel without chart/witness capture or optional budgets.
/// @param g Read-only binary-normal-form grammar.
/// @param s Read-only byte string; empty rejects before grammar preparation.
/// @return Whether the start derives the entire nonempty byte string.
/// @throws std::length_error Checked geometry or container capacity exceeded.
/// @throws std::bad_alloc Grammar/membership allocation failure.
bool recognize(const CNFGrammar& g,const std::string& s);
}

/// @brief Recognize a byte string using the canonical CYK kernel and the original inline API.
/// Same grammar and integer-id semantics as cyk_analyze(), without chart/witness
/// output or optional budgets. Checked geometry/container limits still apply.
/// Empty input always rejects; an unknown byte rejects any complete derivation.
/// No caller mutation, retained references or shared state. Time is
/// O(G log(G+1)+n*G+n^2*m+n^3*(r+1)); storage O(G+n^2*m). No grammar conversion,
/// epsilon rules, Unicode tokenization, ambiguity counting or parse enumeration.
/// @param g Read-only epsilon-free binary-normal-form grammar.
/// @param s Read-only terminal bytes, including embedded NULs.
/// @return True iff the start derives the complete nonempty string.
/// @throws std::length_error Checked geometry or container capacity exceeded.
/// @throws std::bad_alloc Grammar/membership allocation failure.
/// @par Example
/// @code
/// dm::algorithms::CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
/// std::print("{} {}\n",dm::algorithms::cyk_recognize(g,"aaa"),dm::algorithms::cyk_recognize(g,"b")); // true false
/// @endcode
/// @par References
/// - D. H. Younger (1967), Recognition and parsing of context-free languages in time n^3, Information and Control 10(2), 189–208, doi:10.1016/S0019-9958(67)80007-X.
/// @see CNFGrammar, cyk_analyze(), cyk_first_parse()
inline bool cyk_recognize(const CNFGrammar& g,const std::string& s) {return cyk_detail::recognize(g,s);}
} // namespace datamunge::algorithms
