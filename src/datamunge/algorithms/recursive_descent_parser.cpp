/// @file recursive_descent_parser.cpp
/// @brief Shared guarded grammar producer and actual completed syntax-tree records.
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <datamunge/algorithms/recursive_descent_parser.hpp>
#include <stdexcept>
#include <utility>
/// @brief Canonical arithmetic parsing implementation.
namespace datamunge::algorithms {
/// @brief Private grammar parser and actual-value cursor.
namespace {
/// @brief Actual completed value, node id and source byte span passed between nonterminals.
struct Value {
  double      value; ///< Actual finite scalar.
  std::size_t node;  ///< Actual captured node, or SIZE_MAX without capture.
  std::size_t begin; ///< Inclusive source byte beginning.
  std::size_t end;   ///< Exclusive source byte ending.
};
/// @brief Per-call guarded recursive-descent parser with no shared state.
class Parser {
 public:
  /// @brief Bind one complete source and its per-call policy.
  /// @param text Complete borrowed source, outliving this parser call.
  /// @param options Borrowed execution policies.
  /// @param capture Retain actual completed syntax nodes.
  Parser(const std::string& text, const RecursiveDescentOptions& options, bool capture)
      : text_(text), o_(options), capture_(capture) {}
  /// @brief Validate and consume the complete expression once.
  /// @return Owned actual value, source, work and optional tree.
  RecursiveDescentResult run() {
    if (o_.max_bytes && text_.size() > o_.max_bytes)
      throw std::length_error("recursive descent: byte policy");
    if (text_.find(char{0}) != std::string::npos)
      throw std::runtime_error("recursive descent: embedded NUL");
    const auto value = expr();
    skip();
    if (pos_ != text_.size())
      throw std::runtime_error("recursive descent: trailing source");
    result_.value      = value.value;
    result_.root       = value.node;
    result_.expression = text_;
    return std::move(result_);
  }

 private:
  const std::string&             text_;    ///< Borrowed complete source for this synchronous call.
  const RecursiveDescentOptions& o_;       ///< Borrowed per-call execution policies.
  bool                           capture_; ///< Retain actual syntax nodes.
  std::size_t                    pos_{};   ///< Actual next unread source byte.
  std::size_t                    depth_{}; ///< Actual simultaneous grammar frames.
  RecursiveDescentResult         result_;  ///< Owned actual accumulated values/work/tree.

  /// @brief Admit/count one actual grammar frame and restore depth on every exit.
  struct Frame {
    Parser& parser; ///< Borrowed enclosing parser, outliving this scoped guard.

    /// @brief Admit actual grammar depth/call count before entering its body.
    /// @param p Enclosing parser.
    explicit Frame(Parser& p) : parser(p) {
      if (p.result_.calls == std::numeric_limits<std::size_t>::max() ||
          (p.o_.max_calls && p.result_.calls >= p.o_.max_calls) ||
          (p.o_.max_depth && p.depth_ >= p.o_.max_depth))
        throw std::length_error("recursive descent: call/depth policy");
      ++p.result_.calls;
      ++p.depth_;
      p.result_.max_depth = std::max(p.result_.max_depth, p.depth_);
    }
    /// @brief Restore actual simultaneous depth during ordinary or exceptional exit.
    ~Frame() { --parser.depth_; }
  };
  /// @brief Consume actual whitespace bytes without interpreting embedded NUL as EOF.
  void skip() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])))
      ++pos_;
  }
  /// @brief Inspect the next nonspace byte without consuming it.
  /// @return Actual byte or zero only at the validated end of input.
  char peek() {
    skip();
    return pos_ == text_.size() ? char{0} : text_[pos_];
  }
  /// @brief Check/store one actual completed scalar syntax operation.
  /// @param kind Actual node kind.
  /// @param operation Actual operator byte, zero for Number.
  /// @param left Actual first child or SIZE_MAX.
  /// @param right Actual second child or SIZE_MAX.
  /// @param begin Actual inclusive source start.
  /// @param end Actual exclusive source end.
  /// @param value Actual arithmetic result to check before retaining.
  /// @return Completed finite value and actual node/source provenance.
  Value store(
      RecursiveNodeKind kind,
      char              operation,
      std::size_t       left,
      std::size_t       right,
      std::size_t       begin,
      std::size_t       end,
      double            value) {
    if (!std::isfinite(value))
      throw std::overflow_error("recursive descent: nonfinite arithmetic");
    auto node = std::numeric_limits<std::size_t>::max();
    if (capture_) {
      if (o_.max_nodes && result_.nodes.size() >= o_.max_nodes)
        throw std::length_error("recursive descent: node policy");
      node = result_.nodes.size();
      result_.nodes.push_back({kind, operation, begin, end, left, right, value});
    }
    return {value, node, begin, end};
  }
  /// @brief Parse actual additive left-associative operations.
  /// @return Actual completed expression value/node.
  Value expr() {
    Frame frame(*this);
    auto  left = term();
    for (;;) {
      const char op = peek();
      if (op != '+' && op != '-')
        return left;
      ++pos_;
      const auto right = term();
      ++result_.binary_operations;
      left = store(
          RecursiveNodeKind::Binary,
          op,
          left.node,
          right.node,
          left.begin,
          right.end,
          op == '+' ? left.value + right.value : left.value - right.value);
    }
  }
  /// @brief Parse actual multiplicative left-associative operations.
  /// @return Actual completed term value/node.
  Value term() {
    Frame frame(*this);
    auto  left = factor();
    for (;;) {
      const char op = peek();
      if (op != '*' && op != '/')
        return left;
      ++pos_;
      const auto right = factor();
      ++result_.binary_operations;
      left = store(
          RecursiveNodeKind::Binary,
          op,
          left.node,
          right.node,
          left.begin,
          right.end,
          op == '*' ? left.value * right.value : left.value / right.value);
    }
  }
  /// @brief Parse actual prefix signs, groups or decimal atoms.
  /// @return Actual completed factor value/node.
  Value factor() {
    Frame      frame(*this);
    const char op    = peek();
    const auto begin = pos_;
    if (op == '+' || op == '-') {
      ++pos_;
      const auto child = factor();
      ++result_.unary_operations;
      return store(
          RecursiveNodeKind::Unary,
          op,
          child.node,
          std::numeric_limits<std::size_t>::max(),
          begin,
          child.end,
          op == '-' ? -child.value : child.value);
    }
    if (op == '(') {
      ++pos_;
      const auto child = expr();
      if (peek() != ')')
        throw std::runtime_error("recursive descent: expected closing parenthesis");
      ++pos_;
      ++result_.groups;
      return store(
          RecursiveNodeKind::Group,
          '(',
          child.node,
          std::numeric_limits<std::size_t>::max(),
          begin,
          pos_,
          child.value);
    }
    return number();
  }
  /// @brief Scan and fully convert one actual decimal token using locale-independent from_chars.
  /// @return Actual finite number value/node.
  Value number() {
    Frame frame(*this);
    skip();
    const auto begin = pos_;
    bool       point = false, digit = false;
    while (pos_ < text_.size()) {
      const auto c = text_[pos_];
      if (c >= '0' && c <= '9') {
        digit = true;
        ++pos_;
      } else if (c == '.' && !point) {
        point = true;
        ++pos_;
      } else
        break;
    }
    if (!digit)
      throw std::runtime_error("recursive descent: expected decimal number");
    double     value = 0;
    const auto converted =
        std::from_chars(text_.data() + begin, text_.data() + pos_, value, std::chars_format::fixed);
    if (converted.ec == std::errc::result_out_of_range)
      throw std::overflow_error("recursive descent: decimal outside finite double range");
    if (converted.ec != std::errc{} || converted.ptr != text_.data() + pos_)
      throw std::runtime_error("recursive descent: incomplete decimal token");
    ++result_.numbers;
    return store(
        RecursiveNodeKind::Number,
        char{0},
        std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max(),
        begin,
        pos_,
        value);
  }
};
} // namespace
RecursiveDescentResult recursive_descent_run(
    const std::string&             text,
    const RecursiveDescentOptions& o) {
  return Parser(text, o, false).run();
}
RecursiveDescentResult recursive_descent_trace(
    const std::string&             text,
    const RecursiveDescentOptions& o) {
  return Parser(text, o, true).run();
}
double recursive_descent_eval(const std::string& text) {
  return recursive_descent_run(text).value;
}
graphics::Diagram recursive_descent_diagram(
    const RecursiveDescentResult& t,
    std::size_t                   max_nodes) {
  const auto n = t.nodes.size(), none = std::numeric_limits<std::size_t>::max();
  if (!n || t.root >= n)
    throw std::invalid_argument("recursive descent diagram: no captured tree");
  if (max_nodes && n > max_nodes)
    throw std::length_error("recursive descent diagram: node policy");
  std::vector<std::size_t> parents(n, none);
  std::vector<std::string> labels;
  labels.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& node = t.nodes[i];
    if (node.begin > node.end || node.end > t.expression.size() || !std::isfinite(node.value))
      throw std::invalid_argument("recursive descent diagram: span/value");
    std::string name;
    if (node.kind == RecursiveNodeKind::Number) {
      if (node.left != none || node.right != none)
        throw std::invalid_argument("recursive descent diagram: number children");
      name = t.expression.substr(node.begin, node.end - node.begin);
    } else if (
        node.kind == RecursiveNodeKind::Unary || node.kind == RecursiveNodeKind::Binary ||
        node.kind == RecursiveNodeKind::Group) {
      name = node.kind == RecursiveNodeKind::Group ? "group" : std::string(1, node.operation);
      if (node.left == none || (node.kind == RecursiveNodeKind::Binary && node.right == none) ||
          (node.kind != RecursiveNodeKind::Binary && node.right != none))
        throw std::invalid_argument("recursive descent diagram: child shape");
      for (auto child : {node.left, node.right})
        if (child != none) {
          if (child >= i || parents[child] != none)
            throw std::invalid_argument("recursive descent diagram: child ordering/ownership");
          parents[child] = i;
        }
    } else
      throw std::invalid_argument("recursive descent diagram: node kind");
    labels.push_back(name + "\nvalue " + std::to_string(node.value));
  }
  if (t.root != n - 1 || parents[t.root] != none)
    throw std::invalid_argument("recursive descent diagram: root parent");
  // Preserve operand order while adapting parser postorder to root-first display ids.
  std::vector<std::size_t> order, stack{t.root}, display_id(n);
  order.reserve(n);
  while (!stack.empty()) {
    const auto id = stack.back();
    stack.pop_back();
    display_id[id] = order.size();
    order.push_back(id);
    const auto& node = t.nodes[id];
    if (node.right != none)
      stack.push_back(node.right);
    if (node.left != none)
      stack.push_back(node.left);
  }
  if (order.size() != n)
    throw std::invalid_argument("recursive descent diagram: disconnected node");
  std::vector<std::size_t> display_parents(n, none);
  std::vector<std::string> display_labels(n);
  for (std::size_t display = 0; display < n; ++display) {
    const auto id            = order[display];
    display_parents[display] = parents[id] == none ? none : display_id[parents[id]];
    display_labels[display]  = std::move(labels[id]);
  }
  return graphics::Diagram::rooted_tree(
      display_parents, display_labels, "Actual producing arithmetic syntax tree", max_nodes);
}
} // namespace datamunge::algorithms
