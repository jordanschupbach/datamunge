/// @file pratt_parser.cpp
/// @brief One actual precedence-climbing evaluator with complete decimal tokens and guarded recursion.
#include <datamunge/algorithms/pratt_parser.hpp>
#include <cctype>
#include <utility>
#include <cmath>
#include <stdexcept>
#include <limits>
namespace datamunge::algorithms {
/// @brief Local recursive grammar and actual evaluation capture.
namespace pratt_detail {
/// @brief Borrowed complete source and one owning checked parse/evaluation execution.
class Evaluator {
  /// @brief Borrowed source, alive for this synchronous call.
  const std::string& text;
  /// @brief Local scalar policies.
  PrattOptions options;
  /// @brief Whether actual event records are retained.
  bool capture;
  /// @brief Owned actual result/work/history.
  PrattTrace out{};
  /// @brief Actual next source byte.
  std::size_t pos{0};
  /// @brief Actual active recursive frame count.
  std::size_t depth{0};
  /// @brief Admit a selected frame/event action before execution.
  /// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
  void tick() {
    if (out.run.operations == std::numeric_limits<std::size_t>::max() ||
        (options.max_operations && out.run.operations >= options.max_operations))
      throw std::length_error("pratt work cap");
    ++out.run.operations;
  }
  /// @brief Record an actual completed event under selected work/history policies.
  /// @param event Immediate actual parser values.
  /// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
  void record(PrattEvent event) {
    tick();
    event.end   = pos;
    event.depth = depth;
    if (capture) {
      if (options.max_trace_records && out.events.size() >= options.max_trace_records)
        throw std::length_error("pratt trace cap");
      out.events.push_back(event);
    }
  }
  /// @brief Enter a real recursive frame before parsing.
  /// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
  void enter() {
    tick();
    if (depth == std::numeric_limits<std::size_t>::max() ||
        (options.max_depth && depth >= options.max_depth))
      throw std::length_error("pratt depth cap");
    ++depth;
    if (depth > out.run.maximum_depth)
      out.run.maximum_depth = depth;
  }
  /// @brief Balance one actual recursive frame even during an exception.
  struct Frame {
    /// @brief Borrowed active evaluator.
    Evaluator& owner;
    /// @brief Enter the owner frame.
    /// @param owner Active local evaluator.
    /// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
    explicit Frame(Evaluator& owner) : owner(owner) { owner.enter(); }
    /// @brief Release the active frame without throwing.
    ~Frame() { --owner.depth; }
  };
  /// @brief Skip actual locale whitespace bytes using unsigned-char conversion.
  void whitespace() {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
      ++pos;
  }
  /// @brief Read the next actual byte after whitespace.
  /// @return Current byte or zero only at the real end.
  char peek() {
    whitespace();
    return pos < text.size() ? text[pos] : '\0';
  }
  /// @brief Native infix left binding lookup.
  /// @param op Actual candidate byte.
  /// @return 10 for sums,20 for products,30 for power,0 otherwise.
  static int lbp(char op) {
    switch (op) {
      case '+':
      case '-':
        return 10;
      case '*':
      case '/':
        return 20;
      case '^':
        return 30;
      default:
        return 0;
    }
  }
  /// @brief Native infix right binding lookup.
  /// @param op Actual candidate byte.
  /// @return 29 for right-associative power, otherwise the left binding power.
  static int rbp(char op) { return op == '^' ? 29 : lbp(op); }
  /// @brief Require a finite actual arithmetic result.
  /// @param v Actual computed result.
  /// @return Unchanged finite result.
  /// @throws std::overflow_error If arithmetic produces a nonfinite result, including division by zero or invalid real powers.
  static double finite(double v) {
    if (!std::isfinite(v))
      throw std::overflow_error("pratt nonfinite arithmetic");
    return v;
  }
  /// @brief Parse and evaluate one actual binding context.
  /// @param minimum Actual minimum left binding power.
  /// @return Actual finite evaluated scalar.
  /// @throws std::runtime_error If decimal grammar, parentheses or complete input consumption is malformed, including embedded NUL.
  /// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
  /// @throws std::overflow_error If arithmetic produces a nonfinite result, including division by zero or invalid real powers.
  double expression(int minimum) {
    Frame  frame(*this);
    double lhs = atom();
    for (;;) {
      char op     = peek();
      auto offset = pos;
      int  left = lbp(op), right = rbp(op);
      bool descend = left != 0 && left > minimum;
      ++out.run.binding_checks;
      record({PrattEventKind::Binding, offset, 0, 0, minimum, op, left, right, descend, 0, 0, 0});
      if (!descend)
        break;
      ++pos;
      double rhs = expression(right), before = lhs;
      switch (op) {
        case '+':
          lhs = finite(lhs + rhs);
          break;
        case '-':
          lhs = finite(lhs - rhs);
          break;
        case '*':
          lhs = finite(lhs * rhs);
          break;
        case '/':
          lhs = finite(lhs / rhs);
          break;
        case '^':
          lhs = finite(std::pow(lhs, rhs));
          break;
      }
      ++out.run.infix_operations;
      record(
          {PrattEventKind::Infix, offset, 0, 0, minimum, op, left, right, true, before, rhs, lhs});
    }
    return lhs;
  }
  /// @brief Parse one actual decimal, prefix atom or parenthesized expression.
  /// @return Actual finite atom value.
  /// @throws std::runtime_error If decimal grammar, parentheses or complete input consumption is malformed, including embedded NUL.
  /// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
  /// @throws std::overflow_error If arithmetic produces a nonfinite result, including division by zero or invalid real powers.
  double atom() {
    Frame frame(*this);
    char  c      = peek();
    auto  offset = pos;
    if (c == '(') {
      ++pos;
      double value = expression(0);
      if (peek() != ')')
        throw std::runtime_error("pratt expected closing parenthesis");
      ++pos;
      ++out.run.groups;
      record({PrattEventKind::Group, offset, 0, 0, 0, '(', 0, 0, false, 0, 0, value});
      return value;
    }
    if (c == '+' || c == '-') {
      ++pos;
      double operand = atom(), value = c == '-' ? -operand : operand;
      ++out.run.prefix_operations;
      record({PrattEventKind::Prefix, offset, 0, 0, 0, c, 0, 0, false, 0, operand, value});
      return value;
    }
    std::size_t begin = pos;
    bool        dot = false, digit = false;
    while (pos < text.size()) {
      char ch = text[pos];
      if (ch >= '0' && ch <= '9') {
        digit = true;
        ++pos;
      } else if (ch == '.' && !dot) {
        dot = true;
        ++pos;
      } else
        break;
    }
    if (!digit)
      throw std::runtime_error("pratt expected decimal number");
    std::size_t consumed = 0;
    double      value;
    try {
      value = std::stod(text.substr(begin, pos - begin), &consumed);
    } catch (const std::exception&) {
      throw std::runtime_error("pratt decimal conversion failed");
    }
    if (consumed != pos - begin || !std::isfinite(value))
      throw std::runtime_error("pratt incomplete decimal conversion");
    ++out.run.numbers;
    record({PrattEventKind::Number, begin, 0, 0, 0, 0, 0, 0, false, 0, 0, value});
    return value;
  }

 public:
  /// @brief Construct one synchronous local evaluator.
  /// @param source Borrowed complete expression.
  /// @param policy Scalar policies.
  /// @param history Whether to retain actual events.
  Evaluator(const std::string& source, PrattOptions policy, bool history)
      : text(source), options(policy), capture(history) {}
  /// @brief Evaluate and require real end-of-input after whitespace.
  /// @return Owned actual scalar/work and optional events.
  /// @throws std::runtime_error If decimal grammar, parentheses or complete input consumption is malformed, including embedded NUL.
  /// @throws std::length_error If selected work, depth or retained-event policy is exceeded.
  /// @throws std::overflow_error If arithmetic produces a nonfinite result, including division by zero or invalid real powers.
  PrattTrace run() {
    if (text.find(char{0}) != std::string::npos)
      throw std::runtime_error("pratt embedded NUL");
    out.run.value = expression(0);
    whitespace();
    if (pos != text.size())
      throw std::runtime_error("pratt trailing bytes");
    out.run.bytes_consumed = pos;
    return std::move(out);
  }
};
} // namespace pratt_detail
double pratt_eval(const std::string& expression) {
  return pratt_detail::Evaluator(expression, {}, false).run().run.value;
}
PrattRun pratt_run(const std::string& expression, PrattOptions options) {
  return pratt_detail::Evaluator(expression, options, false).run().run;
}
PrattTrace pratt_trace(const std::string& expression, PrattOptions options) {
  return pratt_detail::Evaluator(expression, options, true).run();
}
} // namespace datamunge::algorithms
