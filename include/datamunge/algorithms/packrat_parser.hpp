#pragma once
/// @file packrat_parser.hpp
/// @brief Checked named-rule-memoized byte PEG interpreter with actual rule/choice/expression histories.
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Original ordered PEG node kinds; choices commit to first successful child.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_lit("a").op==dm::algorithms::PegOp::Literal); // true
/// @endcode
/// @see PegExpr
enum class PegOp {
  Literal, ///< Match owned literal bytes, empty allowed.
  Ref,     ///< Call a defined named rule at the same position.
  Seq,     ///< Match all children in order, empty sequence succeeds.
  Choice,  ///< Commit to first successful ordered child, empty choice fails.
  Star,    ///< Greedy zero-or-more repetitions, stopping on no progress.
  Plus,    ///< Greedy one-or-more matches, nullable child allowed with progress guard.
  Opt,     ///< Optional child, failure consumes nothing.
  And,     ///< Positive lookahead, consumes no input.
  Not      ///< Negative lookahead, consumes no input.
};
/// @brief Owned byte PEG tree node, preserving original public field order.
/// Literal/Ref require no children; Ref names a defined nonempty rule. Seq/Choice
/// have arbitrary children and no text; unary operators need exactly one child.
/// Constructor validation checks all rules, including unreachable ones. Empty
/// literals/sequences are nullable; empty choice fails. No Unicode normalization.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_seq({dm::algorithms::peg_lit("a")}).children.size()); // 1
/// @endcode
/// @see PackratParser
struct PegExpr {
  PegOp       op{PegOp::Literal}; ///< Selected original operator, validated before parsing.
  std::string text; ///< Owned literal bytes or referenced rule name, empty for other kinds.
  std::vector<PegExpr> children; ///< Owned ordered child trees, no caller aliases.
};
/// @brief Construct one owned literal-byte expression.
/// @param s Original bytes, including empty/NUL/high bytes.
/// @return Owned structural node, without parsing or retaining references.
/// @throws std::bad_alloc If owned structural storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_lit("ab").text.size()); // 2
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_lit(std::string s);
/// @brief Construct one owned named-rule reference.
/// @param name Rule name, validated by parser construction.
/// @return Owned structural node, without parsing or retaining references.
/// @throws std::bad_alloc If owned structural storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_ref("S").text); // S
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_ref(std::string name);
/// @brief Construct one owned ordered child sequence.
/// @param es Owned original child order; empty sequence succeeds without consuming.
/// @return Owned structural node, without parsing or retaining references.
/// @throws std::bad_alloc If owned structural storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_seq({}).children.size()); // 0
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_seq(std::vector<PegExpr> es);
/// @brief Construct one owned first-success ordered choice.
/// @param es Owned alternative order; empty choice fails.
/// @return Owned structural node, without parsing or retaining references.
/// @throws std::bad_alloc If owned structural storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_choice({}).children.size()); // 0
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_choice(std::vector<PegExpr> es);
/// @brief Construct one owned zero-or-more greedy repetition node.
/// A nullable repeated child stops on no progress; it cannot create an infinite repetition.
/// @param e Owned child expression, validated by parser construction.
/// @return Owned unary structural node, without parsing or caller aliases.
/// @throws std::bad_alloc If child storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_star(dm::algorithms::peg_lit("a")).children.size()); // 1
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_star(PegExpr e);
/// @brief Construct one owned one-or-more greedy repetition node.
/// A nullable repeated child stops on no progress; it cannot create an infinite repetition.
/// @param e Owned child expression, validated by parser construction.
/// @return Owned unary structural node, without parsing or caller aliases.
/// @throws std::bad_alloc If child storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_plus(dm::algorithms::peg_lit("a")).children.size()); // 1
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_plus(PegExpr e);
/// @brief Construct one owned optional child node.
/// A nullable repeated child stops on no progress; it cannot create an infinite repetition.
/// @param e Owned child expression, validated by parser construction.
/// @return Owned unary structural node, without parsing or caller aliases.
/// @throws std::bad_alloc If child storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_opt(dm::algorithms::peg_lit("a")).children.size()); // 1
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_opt(PegExpr e);
/// @brief Construct one owned positive nonconsuming lookahead node.
/// A nullable repeated child stops on no progress; it cannot create an infinite repetition.
/// @param e Owned child expression, validated by parser construction.
/// @return Owned unary structural node, without parsing or caller aliases.
/// @throws std::bad_alloc If child storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_and(dm::algorithms::peg_lit("a")).children.size()); // 1
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_and(PegExpr e);
/// @brief Construct one owned negative nonconsuming lookahead node.
/// A nullable repeated child stops on no progress; it cannot create an infinite repetition.
/// @param e Owned child expression, validated by parser construction.
/// @return Owned unary structural node, without parsing or caller aliases.
/// @throws std::bad_alloc If child storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::peg_not(dm::algorithms::peg_lit("a")).children.size()); // 1
/// @endcode
/// @see PegExpr, PackratParser
PegExpr peg_not(PegExpr e);
/// @brief Optional complete input/grammar and actual work/depth/history controls.
/// @par Example
/// @code
/// dm::algorithms::PackratOptions o;std::print("{}\n",o.max_recursion_depth); // 512
/// @endcode
/// @see PackratParser::parse_trace()
struct PackratOptions {
  std::size_t max_input_bytes{1000000};   ///< Complete copied input byte cap,0 disables.
  std::size_t max_grammar_nodes{1000000}; ///< Complete owned expression-node count cap,0 disables.
  std::size_t max_rule_evaluations{
      1000000}; ///< Actual named-rule body evaluation cap,0 disables optional quota.
  std::size_t max_expression_evaluations{5000000}; ///< Actual expression visit cap,0 disables.
  std::size_t max_recursion_depth{
      512}; ///< Actual combined rule/expression frame depth cap,0 disables; native stack remains finite.
  std::size_t max_trace_entries{
      1000000}; ///< Actual logical input/event/memo capture cap,0 disables; ignored without capture.
};
/// @brief Actual last parse-attempt decision and selected counters.
/// @par Example
/// @code
/// dm::algorithms::PackratRun r;std::print("{}\n",r.completed); // false
/// @endcode
/// @see PackratParser::parse_trace()
struct PackratRun {
  bool accepted{false};       ///< True iff the named start matches the entire original byte string.
  bool matched_prefix{false}; ///< Actual start rule matched some prefix, possibly empty.
  std::size_t end{0};         ///< Actual matched prefix end,0 placeholder for failed start rule.
  bool completed{false}; ///< True only when the complete parse/capture operation returns normally.
  std::size_t rule_evaluations{0}; ///< Actual named-rule bodies evaluated, excludes memo hits.
  std::size_t memo_hits{
      0}; ///< Actual complete named-rule results served from memo, including failures.
  std::size_t expression_evaluations{
      0};                   ///< Actual expression visits; only named-rule results are memoized.
  std::size_t max_depth{0}; ///< Actual combined rule/expression frame high-water mark.
};
/// @brief Actual PEG observation emission site, not a report-side replay.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::PackratEventKind::memo_hit==dm::algorithms::PackratEventKind::memo_hit); // true
/// @endcode
/// @see PackratEvent
enum class PackratEventKind {
  rule_enter,        ///< Before evaluating a newly active named-rule body.
  rule_exit,         ///< After storing an actual complete named-rule result.
  memo_hit,          ///< Actual completed named-rule result reused from cache.
  expression_result, ///< Actual structural expression return, in postorder.
  choice_try,        ///< Before trying one actual ordered alternative.
  choice_accept      ///< Actual first successful alternative committed.
};
/// @brief Actual owned named-rule/expression/choice decision record.
/// @par Example
/// @code
/// dm::algorithms::PackratEvent e;std::print("{}\n",e.position); // 0
/// @endcode
/// @see PackratParser::parse_trace()
struct PackratEvent {
  PackratEventKind kind{
      PackratEventKind::rule_enter}; ///< Actual emission site defining remaining fields.
  std::string rule;                  ///< Owned actual containing/named rule identifier.
  PegOp       op{PegOp::Ref}; ///< Actual expression kind,Ref placeholder for rule-level events.
  std::size_t position{0};    ///< Actual original byte start position.
  bool        matched{false}; ///< Actual result flag; false placeholder for enter/choice_try.
  std::size_t end{0};         ///< Actual successful end,0 placeholder on failure/predecision.
  std::size_t alternative{
      0}; ///< Actual choice child index for try/accept/choice-result;0 otherwise.
  std::size_t expression{0}; ///< Actual one-based expression-visit ID;0 for rule-level events.
  std::size_t depth{0};      ///< Actual combined rule/expression frame depth at emission.
};
/// @brief Actual completed named-rule result retained in the final memo table.
/// @par Example
/// @code
/// dm::algorithms::PackratMemoRecord m;std::print("{}\n",m.matched); // false
/// @endcode
/// @see PackratParser::parse_trace()
struct PackratMemoRecord {
  std::string rule;           ///< Owned actual named rule.
  std::size_t position{0};    ///< Actual original byte start position.
  bool        matched{false}; ///< Actual memoized successful or failed recognition.
  std::size_t end{0};         ///< Actual successful prefix end,0 placeholder when failed.
};
/// @brief Owned original input and actual emitted events/final named-rule memo, no grammar replay.
/// @par Example
/// @code
/// dm::algorithms::PackratTrace t;std::print("{}\n",t.input.size()); // 0
/// @endcode
/// @see PackratParser::parse_trace()
struct PackratTrace {
  std::string input; ///< Owned complete original byte input, no caller alias.
  PackratRun  run;   ///< Actual complete same-producer decision and work.
  std::vector<PackratEvent>
      events; ///< Actual chronological rule/choice events and postorder expression results.
  std::vector<PackratMemoRecord>
              memo; ///< Actual final complete memo entries in map(rule,position) order.
  std::size_t captured_entries{
      0}; ///< Actual logical input bytesN+decision3+each event/memo record(1+rule-name bytes).
};
/// @brief Owned byte PEG interpreter memoizing completed named-rule/position results.
/// Complete grammar validation covers every node/operator/child arity/reference
/// and start name before parsing. Direct/indirect left recursion at an active
/// identical(rule,position) throws rather than recursing forever; right recursion
/// remains subject to actual depth/work limits. Predicates do not consume input;
/// ordered choice never revisits a successful child after a later sequence fails.
/// Empty repetition stops on no progress, preserving original nullable behavior.
/// Uses ordered std::map memo, not constant-time lookup. Only named rules are
/// memoized: repeated greedy expression scans can still be quadratic. Therefore
/// no general linear-time/linear-space packrat bound is promised by this API.
/// Each parse owns its copied input, clears old memo/counters and uses no caller
/// references/callbacks. A failed attempt clears incomplete memo and returns no
/// trace, retaining explicit partial last-attempt counters/completed=false.
/// Copies own grammar/input/cache; concurrent calls on one instance need caller
/// synchronization. Zero disables optional policies, not the finite native stack.
/// @par Example
/// @code
/// dm::algorithms::PackratParser p(dm::algorithms::packrat_anbn_rules(),"S");std::print("{} {}\n",p.parse("aabb"),p.rule_calls()); // true 3
/// @endcode
/// @par References
/// - B. Ford(2004), Parsing Expression Grammars: A Recognition-Based Syntactic Foundation, https://bford.info/pub/lang/peg/. This native named-rule-only map cache has no general linear-time guarantee or left-recursive extension.
/// @see PackratRun, PackratTrace
class PackratParser {
 public:
  /// @brief Own and completely validate a byte PEG grammar and original policies.
  /// @param rules Owned named expression trees; all names/references must be nonempty/defined.
  /// @param start Existing nonempty start-rule name.
  /// @param options Complete grammar/input and actual work/depth/history policies.
  /// @throws std::invalid_argument If any complete grammar/start/operator/reference/arity check fails.
  /// @throws std::length_error If complete grammar node geometry or selected quota fails.
  /// @throws std::bad_alloc If owned validation storage allocation fails.
  /// @par Example
  /// @code
  /// dm::algorithms::PackratParser p({{"S",dm::algorithms::peg_lit("a")}},"S");std::print("{}\n",p.parse("a")); // true
  /// @endcode
  /// @par References
  /// - B. Ford(2004), Parsing Expression Grammars: A Recognition-Based Syntactic Foundation, https://bford.info/pub/lang/peg/. This native named-rule-only map cache has no general linear-time guarantee or left-recursive extension.
  /// @see parse_trace()
  PackratParser(
      std::map<std::string, PegExpr> rules,
      std::string                    start,
      PackratOptions                 options = {});
  /// @brief Recognize the entire supplied byte input with the actual named-rule-memoized interpreter.
  /// Old input/memo/counters are reset per attempt. Input copied/owned, including
  /// NUL/high bytes; prefix success alone does not accept a longer input. Complete
  /// input-byte cap is checked before copying. Active same-key recursion throws.
  /// @param input Complete original byte string, empty allowed.
  /// @return True exactly when the start matches the whole input.
  /// @throws std::invalid_argument If active-key left recursion is encountered.
  /// @throws std::length_error If actual work/count/depth/input/history policy fails.
  /// @throws std::bad_alloc If owned input/memo/history allocation fails.
  /// @par Example
  /// @code
  /// dm::algorithms::PackratParser p({{"S",dm::algorithms::peg_lit("a")}},"S");std::print("{}\n",p.parse("ab")); // false
  /// @endcode
  /// @par References
  /// - B. Ford(2004), Parsing Expression Grammars: A Recognition-Based Syntactic Foundation, https://bford.info/pub/lang/peg/. This native named-rule-only map cache has no general linear-time guarantee or left-recursive extension.
  /// @see last_run(), rule_calls(), memo_hits()
  bool parse(const std::string& input);
  /// @brief Capture actual chronological parser observations and final completed memo entries.
  /// Old input/memo/counters are reset per attempt. Input copied/owned, including
  /// NUL/high bytes; prefix success alone does not accept a longer input. Complete
  /// input-byte cap is checked before copying. Active same-key recursion throws.
  /// History counts input bytesN+decisions3+each event/final memo(1+rule-name
  /// bytes), admission before every copy/append. Default1000000,0 disables.
  /// @param input Complete original byte string, empty allowed.
  /// @return Owned actual complete history/input/decision, no partial result on failure.
  /// @throws std::invalid_argument If active-key left recursion is encountered.
  /// @throws std::length_error If actual work/count/depth/input/history policy fails.
  /// @throws std::bad_alloc If owned input/memo/history allocation fails.
  /// @par Example
  /// @code
  /// dm::algorithms::PackratParser p({{"S",dm::algorithms::peg_lit("a")}},"S");auto t=p.parse_trace("a");std::print("{}\n",t.run.accepted); // true
  /// @endcode
  /// @par References
  /// - B. Ford(2004), Parsing Expression Grammars: A Recognition-Based Syntactic Foundation, https://bford.info/pub/lang/peg/. This native named-rule-only map cache has no general linear-time guarantee or left-recursive extension.
  /// @see last_run(), rule_calls(), memo_hits()
  PackratTrace parse_trace(const std::string& input);
  /// @brief Read actual named-rule body evaluations in the last attempt, even if it threw.
  /// @return Owned actual nonnegative count, with LONG_MAX guarded before increment.
  /// @par Example
  /// @code
  /// dm::algorithms::PackratParser p({{"S",dm::algorithms::peg_lit("a")}},"S");std::print("{}\n",p.rule_calls()); // 0
  /// @endcode
  /// @see parse_trace()
  long rule_calls() const noexcept;
  /// @brief Read actual completed named-rule memo hits in the last attempt, even if it threw.
  /// @return Owned actual nonnegative count, with LONG_MAX guarded before increment.
  /// @par Example
  /// @code
  /// dm::algorithms::PackratParser p({{"S",dm::algorithms::peg_lit("a")}},"S");std::print("{}\n",p.memo_hits()); // 0
  /// @endcode
  /// @see parse_trace()
  long memo_hits() const noexcept;
  /// @brief Copy complete/partial last-attempt status and selected work.
  /// @return Owned actual run snapshot.
  /// @par Example
  /// @code
  /// dm::algorithms::PackratParser p({{"S",dm::algorithms::peg_lit("a")}},"S");std::print("{}\n",p.last_run().completed); // false
  /// @endcode
  /// @see parse_trace()
  PackratRun last_run() const noexcept;

 private:
  /// @brief Internal recognition result, avoiding signed-index narrowing/failure sentinels.
  struct Match {
    bool        matched{false}; ///< Actual success flag.
    std::size_t end{0};         ///< Actual successful end,0 on failure.
  };
  /// @brief Actual rule memo state, active while its body is executing.
  struct Memo {
    bool  active{false}; ///< Detects unsupported same-key left recursion.
    Match result;        ///< Actual completed result when active=false.
  };
  std::map<std::string, PegExpr> rules_;   ///< Owned completely validated original grammar.
  std::string                    start_;   ///< Owned existing start-rule name.
  PackratOptions                 options_; ///< Original complete optional policies.
  std::string input_; ///< Owned complete last-attempt input, never a caller pointer.
  std::map<std::pair<std::string, std::size_t>, Memo>
              memo_;     ///< Actual current/last-attempt named-rule results.
  PackratRun  last_;     ///< Actual complete/partial last-attempt status/counters.
  std::size_t depth_{0}; ///< Current combined rule/expression frames, restored by scoped unwind.

  /// @brief Execute one owned-input attempt with optional actual history sink.
  /// @param input Complete original caller bytes.
  /// @param history Optional actual event/memo sink.
  /// @return Actual start match, with completed whole-input decision stored in last_.
  /// @throws std::invalid_argument If active-key left recursion is encountered.
  /// @throws std::length_error If complete/actual geometry or selected policies fail.
  Match execute(const std::string& input, PackratTrace* history);
  /// @brief Evaluate or reuse one actual named-rule result, guarding active keys.
  /// @param name Existing validated named rule.
  /// @param position Actual original byte position.
  /// @param history Optional owned actual observation sink.
  /// @return Actual completed match/end.
  /// @throws std::invalid_argument If an active same-key recurrence is encountered.
  /// @throws std::length_error If actual work/depth/history limits fail.
  Match parse_rule(const std::string& name, std::size_t position, PackratTrace* history);
  /// @brief Evaluate one actual structural PEG expression with checked byte positions.
  /// @param expression Validated owned grammar node.
  /// @param position Actual original byte position.
  /// @param owner Actual containing named rule, used only for observations.
  /// @param history Optional owned actual observation sink.
  /// @return Actual child/ordered/predicate recognition result.
  /// @throws std::invalid_argument If active-key left recursion is encountered.
  /// @throws std::length_error If actual work/depth/history limits fail.
  Match parse_expr(
      const PegExpr&     expression,
      std::size_t        position,
      const std::string& owner,
      PackratTrace*      history);
  /// @brief Admit/copy one actual emitted event to optional owned history.
  /// @param history Optional sink, null avoids capture.
  /// @param event Actual complete fixed record with owned rule name.
  /// @throws std::length_error If actual history quota/count geometry fails.
  void emit(PackratTrace* history, PackratEvent event);
};
/// @brief Return the original S <- 'a' S 'b' / 'a' 'b' grammar.
/// Fresh owned fixed-size grammar trees, without caller aliases or parsing.
/// @return Owned validated-shape rule map with start nameS.
/// @throws std::bad_alloc If owned structural storage allocation fails.
/// @par Example
/// @code
/// dm::algorithms::PackratParser p(dm::algorithms::packrat_anbn_rules(),"S");std::print("{}\n",p.rule_calls()); // 0
/// @endcode
/// @see PackratParser
std::map<std::string, PegExpr> packrat_anbn_rules();
/// @brief Return the original Start <- 'a'+ !'c' grammar.
/// Fresh owned fixed-size grammar trees, without caller aliases or parsing.
/// @return Owned validated-shape rule map with start nameStart.
/// @throws std::bad_alloc If owned structural storage allocation fails.
/// @par Example
/// @code
/// dm::algorithms::PackratParser p(dm::algorithms::packrat_negative_lookahead_rules(),"Start");std::print("{}\n",p.rule_calls()); // 0
/// @endcode
/// @see PackratParser
std::map<std::string, PegExpr> packrat_negative_lookahead_rules();
/// @brief Return S <- A 'b' / A 'c', A <- 'a'+, exposing one actual memo reuse.
/// Fresh owned fixed-size grammar trees, without caller aliases or parsing.
/// @return Owned validated-shape rule map with start nameS.
/// @throws std::bad_alloc If owned structural storage allocation fails.
/// @par Example
/// @code
/// dm::algorithms::PackratParser p(dm::algorithms::packrat_reuse_rules(),"S");std::print("{}\n",p.rule_calls()); // 0
/// @endcode
/// @see PackratParser
std::map<std::string, PegExpr> packrat_reuse_rules();
/// @brief Return S <- (&A 'a')* !'a', A <- 'a'*, exposing repeated suffix scans.
/// Fresh owned fixed-size grammar trees, without caller aliases or parsing.
/// @return Owned validated-shape rule map with start nameS.
/// @throws std::bad_alloc If owned structural storage allocation fails.
/// @par Example
/// @code
/// dm::algorithms::PackratParser p(dm::algorithms::packrat_quadratic_rules(),"S");std::print("{}\n",p.rule_calls()); // 0
/// @endcode
/// @see PackratParser
std::map<std::string, PegExpr> packrat_quadratic_rules();
/// @brief Construct n copies of'a' followed by n copies of'b', without parsing.
/// Complete2n byte geometry/quota is checked before allocation. O(n) work/output.
/// Empty n0 returns empty bytes; the original anbn grammar rejects that input.
/// @param n Original positive-block length,0 accepted.
/// @param max_bytes Complete output-byte cap, default1000000,0 disables.
/// @return Fresh owned a^n b^n byte string.
/// @throws std::length_error If complete2n geometry or selected quota fails.
/// @throws std::bad_alloc If owned output allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::packrat_balanced_input(2)); // aabb
/// @endcode
/// @see packrat_anbn_rules()
std::string packrat_balanced_input(std::size_t n, std::size_t max_bytes = 1000000);
} // namespace datamunge::algorithms
