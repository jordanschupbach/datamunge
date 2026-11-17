#pragma once
/// @file recursive_descent_parser.hpp
/// @brief Checked decimal recursive-descent evaluation and actual owned syntax-tree construction.
#include <cstddef>
#include <datamunge/plot/diagram.hpp>
#include <limits>
#include <string>
#include <vector>
/// @brief Canonical parsers and owned producing syntax records.
namespace datamunge::algorithms {
/// @brief Actual arithmetic syntax-node kind.
/// @par Example
/// @code
/// const auto value=dm::algorithms::RecursiveNodeKind::Number;
/// std::print("{}\n",value==dm::algorithms::RecursiveNodeKind::Number); // true
/// @endcode
enum class RecursiveNodeKind {
  Number, ///< One parsed decimal numeric token.
  Unary,  ///< One actual unary plus/minus factor.
  Binary, ///< One actual left-associated additive or multiplicative operation.
  Group   ///< One actual parenthesized expression.
};
/// @brief Owned scalar syntax node appended when the native parser actually completes it.
/// @par Example
/// @code
/// const dm::algorithms::RecursiveNode value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RecursiveNode {
  RecursiveNodeKind kind{};      ///< Actual syntax kind.
  char              operation{}; ///< Actual operator; zero for Number, '(' for Group.
  std::size_t       begin{};     ///< First source byte in this syntax span.
  std::size_t       end{};       ///< Exclusive last source byte in the span.
  std::size_t       left{
      std::numeric_limits<std::size_t>::max()}; ///< First child id, or SIZE_MAX for a number.
  std::size_t right{
      std::numeric_limits<std::size_t>::max()}; ///< Second child id, or SIZE_MAX outside Binary.
  double value{};                               ///< Actual finite evaluated scalar at completion.
};
/// @brief Source, recursion, selected-call and actual-node policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RecursiveDescentOptions value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RecursiveDescentOptions {
  std::size_t max_bytes{1000000}; ///< Complete input byte-length policy.
  std::size_t max_depth{512};     ///< Maximum simultaneous grammar calls.
  std::size_t max_calls{1000000}; ///< Actual expr/term/factor/number entries.
  std::size_t max_nodes{1000000}; ///< Actual completed syntax nodes retained by the trace.
};
/// @brief Owned actual parser value, selected work and optional real syntax tree.
/// @par Example
/// @code
/// const dm::algorithms::RecursiveDescentResult value{};
/// std::print("{}\n",sizeof(value)>0); // true
/// @endcode
struct RecursiveDescentResult {
  double      value{};    ///< Actual finite expression result.
  std::string expression; ///< Owned complete original source bytes.
  std::vector<RecursiveNode>
              nodes; ///< Actual postorder completed nodes, only populated by the trace.
  std::size_t root{std::numeric_limits<
      std::size_t>::max()};        ///< Actual tree root id, SIZE_MAX for a noncapturing run.
  std::size_t calls{};             ///< Actual grammar nonterminal entries.
  std::size_t max_depth{};         ///< Actual simultaneous grammar-call depth.
  std::size_t numbers{};           ///< Actual decimal tokens converted.
  std::size_t unary_operations{};  ///< Actual unary plus/minus operations.
  std::size_t binary_operations{}; ///< Actual additive/multiplicative operations.
  std::size_t groups{};            ///< Actual completed parenthesized expressions.
};
/// @brief Evaluate decimal arithmetic using the shared checked recursive-descent producer.
/// Grammar: expr=term((+|-)term)*; term=factor((*|/)factor)*; factor=number|(+|-)factor|(expr). Unary signs bind at factor level; binary operators associate left. Decimal tokens allow one point and at least one digit, without exponents. Converts complete tokens using locale-independent from_chars; embedded NUL, malformed or trailing bytes reject. Every successful scalar operation must remain finite; division by zero throws. O(bytes+actual calls) work plus owned source and bounded call stack; input unchanged, no shared mutable state.
/// @param expression Borrowed complete source bytes.
/// @param options Input, depth and actual-call policies; node cap used only by capture.
/// @return Finite result, owned source and actual grammar/token/operation counts.
/// @throws std::runtime_error Malformed grammar, embedded NUL or trailing bytes.
/// @throws std::length_error Input, depth or call policy exceeded.
/// @throws std::overflow_error Numeric conversion or arithmetic is outside finite double range.
/// @par Example
/// @code
/// auto r=dm::algorithms::recursive_descent_run("1+2*3");
/// std::print("{}\n",r.value); // 7
/// @endcode
/// @see recursive_descent_trace(), pratt_eval()
RecursiveDescentResult recursive_descent_run(
    const std::string&             expression,
    const RecursiveDescentOptions& options = {});
/// @brief Capture the actual syntax tree constructed by the same evaluation producer.
/// Every completed number, unary operation, binary operation or group appends exactly one owned postorder node; child ids are real previously completed nodes. Source offsets refer to the owned unchanged expression. Nodes are admitted before insertion; zero cap disables retained-node policy. O(actual nodes) retained storage in addition to ordinary parsing.
/// @param expression Complete source bytes.
/// @param options Input, recursion, actual-call and actual-node policies.
/// @return Actual finite value, postorder syntax tree, root id and selected work.
/// @throws std::runtime_error Invalid complete grammar/source.
/// @throws std::length_error Any input/call/depth/node policy exceeded.
/// @throws std::overflow_error Nonfinite conversion/arithmetic.
/// @par Example
/// @code
/// auto t=dm::algorithms::recursive_descent_trace("1+2*3");
/// std::print("{} {}\n",t.nodes.size(),t.nodes[t.root].operation); // 5 +
/// @endcode
/// @see recursive_descent_run(), recursive_descent_diagram()
RecursiveDescentResult recursive_descent_trace(
    const std::string&             expression,
    const RecursiveDescentOptions& options = {});
/// @brief Return the checked finite scalar through the preserved recursive-descent signature.
/// Delegates to the same native producer. Previously partially converted repeated decimals, embedded NUL and nonfinite arithmetic now reject. Uses default complete-input, call-depth and selected-call policies; use the run API to configure them.
/// @param expression Complete borrowed source bytes.
/// @return Actual finite arithmetic value.
/// @throws std::runtime_error Malformed or trailing source.
/// @throws std::length_error Default execution policy exceeded.
/// @throws std::overflow_error Unrepresentable conversion or arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::recursive_descent_eval("10-2-3")); // 5
/// @endcode
/// @see recursive_descent_run()
double recursive_descent_eval(const std::string& expression);
/// @brief Draw a validated actual producing syntax tree with native measured label boxes.
/// Consumes retained nodes from recursive_descent_trace(), validates child ordering, ownership shape, finite values and source byte spans, then delegates layout to Diagram::rooted_tree(). Labels show real token/operator values. This validates display structure, not a caller-edited arithmetic proof. No evaluation or replay populates the tree. Layout follows the canonical rooted-tree complexity, O(N^2+text) time and O(N) working storage; inputs unchanged, returned geometry owned.
/// @param trace Actual owned syntax tree, or a structurally valid caller copy.
/// @param max_nodes Complete diagram node policy, default1000; zero disables.
/// @return Editable independent native tree diagram.
/// @throws std::invalid_argument Missing/invalid tree, child id, source span, node kind or value.
/// @throws std::length_error Diagram node policy exceeded.
/// @par Example
/// @code
/// auto d=dm::algorithms::recursive_descent_diagram(dm::algorithms::recursive_descent_trace("1+2"));
/// std::print("{}\n",!d.to_svg().empty()); // true
/// @endcode
/// @see recursive_descent_trace(), graphics::Diagram::rooted_tree()
graphics::Diagram recursive_descent_diagram(
    const RecursiveDescentResult& trace,
    std::size_t                   max_nodes = 1000);
} // namespace datamunge::algorithms
