#pragma once
/// @file pratt_parser.hpp
/// @brief Checked decimal arithmetic Pratt evaluation with actual binding decisions and evaluation events.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Native producing event kind for the Pratt evaluator.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pratt_trace("2").events.front().kind==dm::algorithms::PrattEventKind::Number); // true
/// @endcode
enum class PrattEventKind {
  Binding, ///< Actual infix binding/termination check.
  Number,  ///< Actual decimal token conversion.
  Prefix,  ///< Actual completed unary plus/minus.
  Infix,   ///< Actual completed binary operation.
  Group    ///< Actual completed parenthesized expression.
};
/// @brief Depth, selected parser-call/event and retained-record policies.
/// Zero independently disables the corresponding policy; default depth bounds recursive call frames rather than expression length alone.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::PrattOptions{}.max_depth); // 512
/// @endcode
struct PrattOptions {
  std::size_t max_depth{512}; ///< Maximum simultaneous parse_expr/atom frames, default512.
  std::size_t max_operations{
      1000000}; ///< Actual frame entries plus binding/conversion/completed-operator events.
  std::size_t max_trace_records{
      1000000}; ///< Actual retained event records, zero disables this cap.
};
/// @brief One actual binding decision or completed native evaluation event.
/// Offsets count bytes in the original borrowed string. End is the immediate parser position. Binding records expose the actual context/operator powers; other inapplicable scalar fields are zero.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pratt_trace("1+2*3").run.value); // 7
/// @endcode
struct PrattEvent {
  PrattEventKind kind;   ///< Actual producer event kind.
  std::size_t    offset; ///< Actual operator/token/open-parenthesis byte offset.
  std::size_t    end;    ///< Actual current parser byte position after this event.
  std::size_t    depth;  ///< Actual simultaneous recursive frames.
  int            minimum_binding_power{0}; ///< Actual parse_expr context for a Binding event.
  char           op{0}; ///< Actual operator, or zero for numbers and end-of-input checks.
  int            left_binding_power{0};  ///< Actual operator lookup result.
  int            right_binding_power{0}; ///< Actual operator right power (power uses29).
  bool           descend{false};         ///< Actual Binding decision to parse a right operand.
  double         lhs{0};                 ///< Actual left operand for an arithmetic event.
  double         rhs{0};   ///< Actual right operand, or operand for a prefix operation.
  double         value{0}; ///< Actual computed value for completed events.
};
/// @brief Owned scalar outcome and selected actual parser work.
/// No AST is allocated; arithmetic occurs during the same recursive parse.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pratt_run("1+2*3").infix_operations); // 2
/// @endcode
struct PrattRun {
  double      value{0};             ///< Actual finite scalar result.
  std::size_t numbers{0};           ///< Actual converted decimal tokens.
  std::size_t prefix_operations{0}; ///< Actual completed unary operations.
  std::size_t infix_operations{0};  ///< Actual completed binary operations.
  std::size_t groups{0};            ///< Actual completed parenthesis pairs.
  std::size_t binding_checks{
      0}; ///< Actual operator/context comparisons, including terminating checks.
  std::size_t maximum_depth{0};  ///< Actual maximum simultaneous recursive frames.
  std::size_t operations{0};     ///< Actual selected frame entries and emitted logical events.
  std::size_t bytes_consumed{0}; ///< Actual complete input length consumed, including whitespace.
};
/// @brief Ordinary native scalar/work and actual parser event order.
/// Captures the producing recursive evaluation, not a separately reconstructed parse tree.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pratt_trace("2^3^2").run.value); // 512
/// @endcode
struct PrattTrace {
  PrattRun                run;    ///< Actual scalar result and selected work.
  std::vector<PrattEvent> events; ///< Actual checks and completed operations in execution order.
};
/// @brief Evaluate by the shared checked Pratt producer.
/// Supports ASCII decimal numbers with at most one dot and no exponent suffix, whitespace, parentheses, unary plus/minus, binary+,-,*,/, and right-associative^. Powers bind above products above sums. Unary signs are parsed as atoms and therefore bind more tightly than power: -2^2 is4, preserving the historical grammar. Complete token conversion/input consumption is required; strings are borrowed unchanged. Local state, no AST; work is linear in input/token scans plus std::stod/std::pow costs, recursion follows grammar and is capped by default.
/// @param expression Borrowed complete arithmetic byte string.
/// @return Finite scalar.
/// @throws std::runtime_error If decimal grammar, parentheses or complete input consumption is malformed, including embedded NUL.
/// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
/// @throws std::overflow_error If arithmetic produces a nonfinite result, including division by zero or invalid real powers.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pratt_eval("2^3^2")); // 512
/// @endcode
/// @par References
/// - V. R. Pratt (1973), Top down operator precedence, Proceedings of the1st Annual ACM SIGACT-SIGPLAN Symposium on Principles of Programming Languages,41–51, doi:10.1145/512927.512931.
double pratt_eval(const std::string& expression);
/// @brief Evaluate by the shared checked Pratt producer.
/// Supports ASCII decimal numbers with at most one dot and no exponent suffix, whitespace, parentheses, unary plus/minus, binary+,-,*,/, and right-associative^. Powers bind above products above sums. Unary signs are parsed as atoms and therefore bind more tightly than power: -2^2 is4, preserving the historical grammar. Complete token conversion/input consumption is required; strings are borrowed unchanged. Local state, no AST; work is linear in input/token scans plus std::stod/std::pow costs, recursion follows grammar and is capped by default.
/// @param expression Borrowed complete arithmetic byte string.
/// @param options Depth/work/history policies; history cap unused for pratt_run.
/// @return Finite scalar and selected work.
/// @throws std::runtime_error If decimal grammar, parentheses or complete input consumption is malformed, including embedded NUL.
/// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
/// @throws std::overflow_error If arithmetic produces a nonfinite result, including division by zero or invalid real powers.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pratt_run("2^3^2").value); // 512
/// @endcode
/// @par References
/// - V. R. Pratt (1973), Top down operator precedence, Proceedings of the1st Annual ACM SIGACT-SIGPLAN Symposium on Principles of Programming Languages,41–51, doi:10.1145/512927.512931.
PrattRun pratt_run(const std::string& expression, PrattOptions options = {});
/// @brief Evaluate by the shared checked Pratt producer.
/// Supports ASCII decimal numbers with at most one dot and no exponent suffix, whitespace, parentheses, unary plus/minus, binary+,-,*,/, and right-associative^. Powers bind above products above sums. Unary signs are parsed as atoms and therefore bind more tightly than power: -2^2 is4, preserving the historical grammar. Complete token conversion/input consumption is required; strings are borrowed unchanged. Local state, no AST; work is linear in input/token scans plus std::stod/std::pow costs, recursion follows grammar and is capped by default.
/// @param expression Borrowed complete arithmetic byte string.
/// @param options Depth/work/history policies; history cap unused for pratt_run.
/// @return Finite scalar, selected work and actual event history.
/// @throws std::runtime_error If decimal grammar, parentheses or complete input consumption is malformed, including embedded NUL.
/// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
/// @throws std::overflow_error If arithmetic produces a nonfinite result, including division by zero or invalid real powers.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::pratt_trace("2^3^2").run.value); // 512
/// @endcode
/// @par References
/// - V. R. Pratt (1973), Top down operator precedence, Proceedings of the1st Annual ACM SIGACT-SIGPLAN Symposium on Principles of Programming Languages,41–51, doi:10.1145/512927.512931.
PrattTrace pratt_trace(const std::string& expression, PrattOptions options = {});
} // namespace datamunge::algorithms
