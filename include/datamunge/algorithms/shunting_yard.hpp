#pragma once
/// @file shunting_yard.hpp
/// @brief Strict binary arithmetic conversion and checked actual stack evaluation.
#include <cstddef>
#include <string>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief ShuntingYardOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShuntingYardOptions value{};
/// std::print("{}\n",value.max_bytes); // 1000000
/// @endcode
struct ShuntingYardOptions {
  std::size_t max_bytes{1000000}; ///< Complete expression/postfix input bytes; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Captured records plus copied string bytes/stack entries; zero disables.
};
/// @brief ShuntingYardStep stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShuntingYardStep value{};
/// std::print("{}\n",value.token.size()); // 0
/// @endcode
struct ShuntingYardStep {
  std::string token;     ///< Actual consumed token or final flush marker.
  std::string operators; ///< Actual pending operator stack, bottom to top.
  std::string output;    ///< Actual emitted postfix prefix.
};
/// @brief RpnStep stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::RpnStep value{};
/// std::print("{}\n",value.stack.size()); // 0
/// @endcode
struct RpnStep {
  std::string         token; ///< Actual consumed postfix token.
  std::vector<double> stack; ///< Actual evaluation stack after this token.
};
/// @brief ShuntingYardRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::ShuntingYardRun value{};
/// std::print("{}\n",value.tokens); // 0
/// @endcode
struct ShuntingYardRun {
  std::string postfix;                      ///< Actual emitted postfix expression.
  double      value{};                      ///< Actual checked evaluation result.
  std::size_t tokens{};                     ///< Actual lexical infix tokens.
  std::size_t operator_pops{};              ///< Actual emitted operators including final drain.
  std::vector<ShuntingYardStep> conversion; ///< Optional actual conversion states.
  std::vector<RpnStep>          evaluation; ///< Optional actual evaluation states.
};
/// @brief Convert and evaluate strict infix arithmetic.
/// Grammar: finite nonnegative decimal literals with at least one digit and at most one dot, binary + - * / ^, whitespace and matched parentheses. No unary signs, exponent literals, implicit multiplication or unknown characters. ^ is right associative, other operators left associative. Intermediate values may be negative. Empty input is invalid. Conversion is O(bytes) without optional copied snapshots; evaluation O(tokens) excluding pow implementation. Full stack and finite arithmetic validation, division by zero rejected. Capture storage is bounded by actual copied bytes/entries, not just records. No shared state; outputs own strings/stacks.
/// @param infix Complete borrowed expression.
/// @param options Complete byte and optional copied-history policies; zero disables policies.
/// @return Owned postfix, actual result, counters and optional real states.
/// @throws std::invalid_argument Malformed grammar or nonfinite literal.
/// @throws std::domain_error Zero division or nonfinite arithmetic result.
/// @throws std::length_error Byte/history policy or container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shunting_yard_run("3+4*2");
/// std::print("{} {}\n",r.postfix,r.value); // 3 4 2 * + 11
/// @endcode
/// @par References
/// - E. W. Dijkstra, ALGOL 60 Translation, EWD35, 1961.
/// @see evaluate_rpn()
ShuntingYardRun shunting_yard_run(
    const std::string&         infix,
    const ShuntingYardOptions& options = {});
/// @brief Convert and evaluate strict infix arithmetic with actual stacks.
/// Grammar: finite nonnegative decimal literals with at least one digit and at most one dot, binary + - * / ^, whitespace and matched parentheses. No unary signs, exponent literals, implicit multiplication or unknown characters. ^ is right associative, other operators left associative. Intermediate values may be negative. Empty input is invalid. Conversion is O(bytes) without optional copied snapshots; evaluation O(tokens) excluding pow implementation. Full stack and finite arithmetic validation, division by zero rejected. Capture storage is bounded by actual copied bytes/entries, not just records. No shared state; outputs own strings/stacks.
/// @param infix Complete borrowed expression.
/// @param options Complete byte and optional copied-history policies; zero disables policies.
/// @return Owned postfix, actual result, counters and optional real states.
/// @throws std::invalid_argument Malformed grammar or nonfinite literal.
/// @throws std::domain_error Zero division or nonfinite arithmetic result.
/// @throws std::length_error Byte/history policy or container capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shunting_yard_trace("3+4*2");
/// std::print("{} {}\n",r.postfix,r.value); // 3 4 2 * + 11
/// @endcode
/// @par References
/// - E. W. Dijkstra, ALGOL 60 Translation, EWD35, 1961.
/// @see evaluate_rpn()
ShuntingYardRun shunting_yard_trace(
    const std::string&         infix,
    const ShuntingYardOptions& options = {});
/// @brief Convert strict infix to postfix without evaluating arithmetic.
/// Uses the same grammar/conversion producer and default byte bound; a syntactically valid zero divisor is accepted here.
/// @param infix Complete borrowed binary expression.
/// @return Owned space-separated postfix.
/// @throws std::invalid_argument Malformed grammar/literal.
/// @throws std::length_error Default byte bound exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shunting_yard_to_postfix("2^3^2")); // 2 3 2 ^ ^
/// @endcode
/// @see shunting_yard_run()
std::string shunting_yard_to_postfix(const std::string& infix);
/// @brief Evaluate a complete whitespace-separated postfix expression.
/// Numbers use the infix decimal grammar, extended with a leading sign for standalone RPN. Requires two operands per binary operator and exactly one final stack entry. Rejects trailing numeric junk, zero division and nonfinite results. Default one-million byte bound; O(tokens) stack operations.
/// @param postfix Complete borrowed whitespace-delimited expression.
/// @return Finite computed result.
/// @throws std::invalid_argument Malformed token or stack arity.
/// @throws std::domain_error Zero division or nonfinite result.
/// @throws std::length_error Default input byte policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::evaluate_rpn("3 4 2 * +")); // 11
/// @endcode
/// @see shunting_yard_run()
double evaluate_rpn(const std::string& postfix);
/// @brief Parse and evaluate strict infix directly.
/// Uses the same checked producer and default policies.
/// @param infix Complete borrowed expression.
/// @return Finite actual result.
/// @throws std::invalid_argument Malformed grammar.
/// @throws std::domain_error Zero division/nonfinite arithmetic.
/// @throws std::length_error Default byte bound exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shunting_yard_evaluate("2^3^2")); // 512
/// @endcode
/// @see shunting_yard_run()
double shunting_yard_evaluate(const std::string& infix);
} // namespace datamunge::algorithms
