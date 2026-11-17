#pragma once
/// @file ll1_parser.hpp
/// @brief Checked FIRST/FOLLOW predictive tables, actual conflict witnesses and bounded real parse-stack traces.
#include <cstddef>
#include <datamunge/algorithms/context_free_grammar.hpp>
#include <map>
#include <set>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual bounded predictive-parser outcome, distinguishing rejection from exhausted action budget.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
/// std::print("{}\n",p.analyze({"a"}).status==dm::algorithms::LL1ParseStatus::Accepted); // true
/// @endcode
/// @see LL1ParseAnalysis
enum class LL1ParseStatus {
  Accepted,         ///< End marker reached after consuming every original token.
  NoEntry,          ///< Actual nonterminal/lookahead table entry absent.
  TerminalMismatch, ///< Actual terminal stack top differs from lookahead.
  ExtraInput,       ///< Stack end marker reached before original input exhausted.
  StepLimit ///< Explicit real action budget exhausted; not a language rejection certificate.
};
/// @brief Actual producing parse-stack action recorded for one complete step.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
/// std::print("{}\n",p.trace({"a"}).events[0].action==dm::algorithms::LL1ParseAction::Expand); // true
/// @endcode
/// @see LL1ParseEvent
enum class LL1ParseAction {
  Expand, ///< Actual table-selected production expansion.
  Match,  ///< Actual terminal consumed and stack symbol popped.
  Accept, ///< Actual complete-input end-marker acceptance.
  Reject  ///< Actual absent-entry/mismatch/extra-input stop.
};
/// @brief Actual selected monotone fixed-point and predictive-table construction counts.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
/// std::print("{}\n",p.construction_work().conflicts); // 0
/// @endcode
/// @par References
/// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
/// @see LL1Parser
struct LL1BuildWork {
  std::size_t first_passes{0}; ///< Actual FIRST fixed-point passes, including final no-change pass.
  std::size_t follow_passes{
      0}; ///< Actual FOLLOW fixed-point passes, including final no-change pass.
  std::size_t first_insertions{
      0}; ///< Actual new FIRST member insertions, including terminal self-members.
  std::size_t follow_insertions{0}; ///< Actual new FOLLOW members, including start end-marker seed.
  std::size_t table_attempts{
      0};                   ///< Actual FIRST/FOLLOW-driven production-to-cell insertion attempts.
  std::size_t conflicts{0}; ///< Actual attempts encountering another retained production index.
};
/// @brief Owned actual conflicting table-cell witness, with earliest retained and new production identities.
/// Conflicted models remain inspectable but parsing refuses them. Duplicate
/// indexed productions can also conflict; no silent arbitrary-choice recognizer.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p({{{"S",{"a"}},{"S",{"a","a"}}},"S"});
/// std::print("{}\n",p.conflicts().size()); // 1
/// @endcode
/// @par References
/// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
/// @see LL1Parser
struct LL1Conflict {
  std::string nonterminal;  ///< Actual conflicting left-hand nonterminal.
  std::string lookahead;    ///< Actual terminal/end-marker table column.
  int         retained{-1}; ///< Actual earliest production index retained in this cell.
  int         incoming{-1}; ///< Actual distinct later production index attempted for the same cell.
};
/// @brief Actual predictive-parser status and selected real action/input/stack counts.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
/// std::print("{} {}\n",p.analyze({"a"}).accepted,p.analyze({"a"}).consumed); // true 1
/// @endcode
/// @par References
/// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
/// @see LL1Parser
struct LL1ParseAnalysis {
  bool           accepted{false}; ///< True only for complete actual end-marker acceptance.
  LL1ParseStatus status{
      LL1ParseStatus::StepLimit}; ///< Actual accept/reject/explicit-budget outcome.
  std::size_t steps{
      0}; ///< Actual performed actions, including final accept/reject; budget check not an action.
  std::size_t consumed{0};        ///< Actual original tokens consumed by terminal matches.
  std::size_t maximum_stack{2};   ///< Actual maximum stack symbol count, including end marker.
  std::size_t remaining_stack{2}; ///< Actual stack symbol count on stop.
};
/// @brief Owned actual stack/input decision before and after one real parser action.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
/// std::print("{}\n",p.trace({"a"}).events[0].production); // 0
/// @endcode
/// @par References
/// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
/// @see LL1Parser
struct LL1ParseEvent {
  std::size_t    step{0};         ///< Actual zero-based producing action index.
  std::size_t    input_offset{0}; ///< Actual original input offset before action.
  std::string    top;             ///< Actual stack top before action.
  std::string    lookahead;       ///< Actual current input terminal or native end marker.
  LL1ParseAction action{LL1ParseAction::Reject}; ///< Actual expansion/match/accept/reject decision.
  int production{-1}; ///< Actual selected original production index on expansion, otherwise-1.
  std::vector<std::string> before; ///< Actual complete stack before action, bottom-to-top.
  std::vector<std::string> after;  ///< Actual complete committed stack after action, bottom-to-top.
};
/// @brief Owned complete original tokens, native initial stack and every actual bounded parse action.
/// Complete actual run until accept/reject/step budget, or capture exception;
/// budget exhaustion is explicitly represented and not a silent rejection.
/// Capture cap counts original token symbols,2 initial stack symbols and each
/// event record plus actual before/after stack symbols. String characters,
/// scalar fields/capacities/runtime/result copies and bytes/RAM are excluded.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
/// std::print("{}\n",p.trace({"a"}).analysis.steps); // 3
/// @endcode
/// @par References
/// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
/// @see LL1Parser
struct LL1ParseTrace {
  std::vector<std::string>
      tokens; ///< Owned complete original token sequence, without appended native end marker.
  std::vector<std::string> initial_stack; ///< Actual initial bottom-to-top end-marker/start stack.
  std::vector<LL1ParseEvent> events; ///< Actual chronological producing actions; no parser replay.
  LL1ParseAnalysis analysis; ///< Actual full accept/reject/budget status and selected counts.
  std::size_t      captured_entries{
      0}; ///< Actual original-token/initial-stack/event/before-stack/after-stack logical symbols/records.
};
/// @brief Own a checked grammar, its actual FIRST/FOLLOW predictive table and explicit cell-conflict witnesses.
/// A symbol is nonterminal exactly when used as a production lhs. Grammar start
/// must be a defined nonterminal; lhs/rhs symbols cannot be empty or "$"; epsilon
/// is an empty rhs vector, not a literal empty token. All production indices must
/// fit int. FIRST/FOLLOW use finite monotone insertion fixed points; build work
/// is grammar dependent, not a claimed constant-time preprocessing step.
/// Retain earliest production per table cell and record actual conflicting later
/// indices. is_ll1() means this indexed construction has no conflicts, including
/// duplicate-rule IDs; conflicted models are inspectable but parsing throws.
/// Parsing consumes complete finite string tokens, no lexing/backtracking/error
/// recovery. Explicit action budget separates incomplete computation from a
/// syntax rejection. Owned model has no borrowed grammar lifetime/shared state;
/// independent const inference is safe when no caller mutates the object.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
/// std::print("{} {}\n",p.is_ll1(),p.recognize({"a"})); // true true
/// @endcode
/// @par References
/// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
/// @see LL1ParseTrace, LL1Conflict
class LL1Parser {
 public:
  static constexpr const char* kEpsilon =
      ""; ///< Native epsilon FIRST marker, never a legal literal grammar/input symbol.
  static constexpr const char* kEndMarker =
      "$"; ///< Native parser end marker, never a legal literal grammar/input symbol.

  /// @brief Validate and own a grammar, then build real FIRST/FOLLOW sets, table and conflict witnesses.
  /// @param grammar Owned by-value original production list/start; argument copy can precede body validation.
  /// @throws std::invalid_argument On empty/undefined start or empty/reserved literal symbols.
  /// @throws std::length_error On full production-index/symbol/count geometry.
  /// @throws std::bad_alloc On owned grammar/set/table/witness allocations.
  /// @par Example
  /// @code
  /// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
  /// std::print("{}\n",p.is_ll1()); // true
  /// @endcode
  /// @par References
  /// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
  /// @see construction_work(), conflicts()
  explicit LL1Parser(ContextFreeGrammar grammar);
  /// @brief Inspect whether the actual indexed table construction encountered any distinct-rule conflicts.
  /// @return True exactly when conflicts() is empty.
  bool is_ll1() const noexcept { return conflicts_.empty(); }
  /// @brief Borrow actual FIRST members for a known terminal/nonterminal.
  /// @param symbol Exact grammar symbol.
  /// @return Model-owned set, including kEpsilon when nullable; lifetime is this object's.
  /// @throws std::out_of_range On unknown symbol.
  const std::set<std::string>& first(const std::string& symbol) const { return first_.at(symbol); }
  /// @brief Borrow actual FOLLOW members for a known nonterminal.
  /// @param nonterminal Exact original nonterminal.
  /// @return Model-owned set with native end marker when applicable.
  /// @throws std::out_of_range On terminal/unknown symbol.
  const std::set<std::string>& follow(const std::string& nonterminal) const {
    return follow_.at(nonterminal);
  }
  /// @brief Borrow all actual final FIRST sets without reconstruction.
  /// @return Const model-owned symbol-to-member map.
  const std::map<std::string, std::set<std::string>>& first_sets() const noexcept { return first_; }
  /// @brief Borrow all actual final FOLLOW sets without reconstruction.
  /// @return Const model-owned nonterminal-to-member map.
  const std::map<std::string, std::set<std::string>>& follow_sets() const noexcept {
    return follow_;
  }
  /// @brief Borrow the actual retained predictive table, with original production IDs.
  /// @return Const model-owned(nonterminal,lookahead)-to-rule map.
  const std::map<std::pair<std::string, std::string>, int>& table() const noexcept {
    return table_;
  }
  /// @brief Borrow actual construction conflict witnesses in producer order.
  /// @return Const model-owned conflicting cell/rule records.
  const std::vector<LL1Conflict>& conflicts() const noexcept { return conflicts_; }
  /// @brief Borrow selected actual FIRST/FOLLOW/table construction counts.
  /// @return Const model-owned producing work.
  const LL1BuildWork& construction_work() const noexcept { return build_work_; }
  /// @brief Borrow the owned validated original grammar and production ordering.
  /// @return Const grammar reference for this model's lifetime.
  const ContextFreeGrammar& grammar() const noexcept { return g_; }
  /// @brief Recognize a complete token sequence with the default100000 real-action budget.
  /// Throws on a conflicted table or exhausted default budget; neither is
  /// silently treated as language rejection or arbitrary first-rule parsing.
  /// @param tokens Borrowed complete tokens; empty input valid, literal empty/$ tokens forbidden.
  /// @return True for actual complete acceptance; false only for an actual syntax rejection.
  /// @throws std::invalid_argument On literal empty/$ input token.
  /// @throws std::logic_error On a conflicted predictive table.
  /// @throws std::length_error On default action-budget or stack/container exhaustion.
  /// @throws std::bad_alloc On input/stack/output/capture allocation failure.
  /// @par Example
  /// @code
  /// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
  /// std::print("{}\n",p.recognize({"a"})); // true
  /// @endcode
  /// @par References
  /// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
  /// @see LL1ParseAnalysis, LL1ParseTrace
  bool recognize(const std::vector<std::string>& tokens) const;
  /// @brief Return actual predictive-parser status/work with an explicit finite action budget.
  /// No optional history. StepLimit is explicit incomplete status, not rejection.
  /// @param tokens Borrowed complete original tokens; no empty/$ literals.
  /// @param max_steps Actual action budget, default100000;0 returns initial StepLimit status.
  /// @return Owned accept/reject/StepLimit status, consumed count and real stack/action counts.
  /// @throws std::invalid_argument On literal empty/$ input token.
  /// @throws std::logic_error On a conflicted predictive table.
  /// @throws std::length_error On stack/container or unrepresentable max_steps geometry.
  /// @throws std::bad_alloc On input/stack/output/capture allocation failure.
  /// @par Example
  /// @code
  /// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
  /// std::print("{}\n",p.analyze({"a"},0).status==dm::algorithms::LL1ParseStatus::StepLimit); // true
  /// @endcode
  /// @par References
  /// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
  /// @see LL1ParseAnalysis, LL1ParseTrace
  LL1ParseAnalysis analyze(const std::vector<std::string>& tokens, std::size_t max_steps = 100000)
      const;
  /// @brief Capture original tokens and every real predictive-stack action once.
  /// Initial input/stack and each known actual before/after event geometry are
  /// admitted before their copies; no replay or fabricated post-budget steps.
  /// @param tokens Borrowed complete original tokens, empty valid.
  /// @param max_steps Actual action budget, default100000;0 explicit initial StepLimit.
  /// @param max_entries Actual logical token/stack/event cap, default1000000;0 optional cap off.
  /// @return Owned complete actual bounded parse evidence with honest status.
  /// @throws std::invalid_argument On literal empty/$ input token.
  /// @throws std::logic_error On a conflicted predictive table.
  /// @throws std::length_error On actual capture quota/count or stack/action geometry.
  /// @throws std::bad_alloc On input/stack/output/capture allocation failure.
  /// @par Example
  /// @code
  /// const dm::algorithms::LL1Parser p({{{"S",{"a"}}},"S"});
  /// std::print("{}\n",p.trace({"a"}).events.size()); // 3
  /// @endcode
  /// @par References
  /// - Lewis and Stearns(1968), Syntax-Directed Transduction, Journal of the ACM15(3),465–488, https://doi.org/10.1145/321466.321477; explicit FIRST/FOLLOW table construction here.
  /// @see LL1ParseAnalysis, LL1ParseTrace
  LL1ParseTrace trace(
      const std::vector<std::string>& tokens,
      std::size_t                     max_steps   = 100000,
      std::size_t                     max_entries = 1000000) const;

 private:
  ContextFreeGrammar    g_;            ///< Owned validated original grammar.
  std::set<std::string> nonterminals_; ///< Cached defined lhs symbols for actual parser decisions.
  std::map<std::string, std::set<std::string>> first_;  ///< Actual finite fixed-point FIRST sets.
  std::map<std::string, std::set<std::string>> follow_; ///< Actual finite fixed-point FOLLOW sets.
  std::map<std::pair<std::string, std::string>, int>
                           table_;      ///< Actual earliest-rule predictive table.
  std::vector<LL1Conflict> conflicts_;  ///< Actual distinct-ID conflicting insertion attempts.
  LL1BuildWork             build_work_; ///< Actual selected constructor operations.

  /// @brief Test cached actual nonterminal membership.
  /// @param symbol Exact original grammar symbol.
  /// @return True exactly for a defined lhs symbol.
  bool is_nt(const std::string& symbol) const { return nonterminals_.contains(symbol); }
  /// @brief Read current FIRST of a complete rhs suffix, with native epsilon marker.
  /// @param sequence Borrowed original symbol sequence, possibly empty.
  /// @return Owned current FIRST union and nullable marker.
  std::set<std::string> first_of_seq(const std::vector<std::string>& sequence) const;
  /// @brief Insert one actual new FIRST member and count successful insertion.
  /// @param symbol Actual destination symbol.
  /// @param terminal Actual member or epsilon marker.
  /// @return True only for a new set member.
  bool insert_first(const std::string& symbol, const std::string& terminal);
  /// @brief Insert one actual new FOLLOW member and count successful insertion.
  /// @param symbol Actual destination nonterminal.
  /// @param terminal Actual terminal or end marker.
  /// @return True only for a new set member.
  bool insert_follow(const std::string& symbol, const std::string& terminal);
  /// @brief Compute monotone FIRST fixed point in original production/set order.
  void compute_first();
  /// @brief Compute monotone FOLLOW fixed point in original production/suffix order.
  void compute_follow();
  /// @brief Produce actual table attempts from final FIRST/FOLLOW in original production order.
  void build_table();
  /// @brief Retain earliest rule and record an actual conflicting later indexed rule.
  /// @param nonterminal Actual lhs cell row.
  /// @param lookahead Actual terminal/end-marker column.
  /// @param production Actual original rule index.
  void insert_entry(const std::string& nonterminal, const std::string& lookahead, int production);
  /// @brief Run one actual predictive stack with optional preadmitted action capture.
  /// @param tokens Completely validated borrowed original tokens.
  /// @param max_steps Actual finite action budget.
  /// @param capture Optional input/initial-stack-admitted owned trace.
  /// @param max_entries Actual logical cap,0 off.
  /// @return Actual accept/reject/budget outcome and selected action/stack counts.
  LL1ParseAnalysis parse(
      const std::vector<std::string>& tokens,
      std::size_t                     max_steps,
      LL1ParseTrace*                  capture,
      std::size_t                     max_entries) const;
};
/// @brief Return the original report's arithmetic expression grammar in predictive or left-recursive form.
/// Symbols are id,+,*,(,), with original E/T/F production ordering. The default
/// is right-recursive and nullable-tail LL(1); the left-recursive control has
/// actual table conflicts and is for construction inspection only.
/// @param left_recursive Select original conflicting left-recursive control if true, default false.
/// @return Owned complete original grammar with start E.
/// @throws std::bad_alloc On grammar string/vector allocations.
/// @par Example
/// @code
/// const dm::algorithms::LL1Parser p(dm::algorithms::ll1_expression_grammar());
/// std::print("{}\n",p.is_ll1()); // true
/// @endcode
/// @see LL1Parser
ContextFreeGrammar ll1_expression_grammar(bool left_recursive = false);
} // namespace datamunge::algorithms
