/// @file shunting_yard.cpp
/// @brief Strict producing parser and checked actual postfix stack execution.
#include <cctype>
#include <charconv>
#include <cmath>
#include <datamunge/algorithms/shunting_yard.hpp>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Recognize one binary operator.
/// @param c Input byte.
/// @return Whether supported.
bool op(char c) {
  return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
}
/// @brief Precedence rank.
/// @param c Operator.
/// @return Binding rank.
int precedence(char c) {
  return c == '^' ? 3 : (c == '*' || c == '/') ? 2 : 1;
}
/// @brief Validate and convert a complete finite decimal token.
/// @param token Complete token.
/// @param sign Permit standalone RPN sign.
/// @return Finite decimal value.
double number(std::string_view token, bool sign) {
  auto t = token;
  if (sign && !t.empty() && (t.front() == '+' || t.front() == '-'))
    t.remove_prefix(1);
  bool digit = false, dot = false;
  for (char c : t) {
    if (c >= '0' && c <= '9')
      digit = true;
    else if (c == '.' && !dot)
      dot = true;
    else
      throw std::invalid_argument("parser: decimal token");
  }
  if (!digit)
    throw std::invalid_argument("parser: missing digits");
  if (token.front() == '+')
    token.remove_prefix(1);
  double value{};
  const auto [p, e] = std::from_chars(token.data(), token.data() + token.size(), value);
  if (e != std::errc{} || p != token.data() + token.size() || !std::isfinite(value))
    throw std::invalid_argument("parser: finite decimal");
  return value;
}
/// @brief Bound actual copied history units with overflow checks.
/// @param retained Current retained amount.
/// @param extra New record/copy units.
/// @param limit Zero or policy.
void budget(std::size_t& retained, std::size_t extra, std::size_t limit) {
  if (extra > std::numeric_limits<std::size_t>::max() - retained)
    throw std::length_error("parser: history capacity");
  retained += extra;
  if (limit && retained > limit)
    throw std::length_error("parser: history policy");
}
/// @brief Convert once, recording real token states.
/// @param text Infix.
/// @param o Policies.
/// @param capture Capture states.
/// @return Actual postfix and states.
ShuntingYardRun convert(const std::string& text, const ShuntingYardOptions& o, bool capture) {
  if (o.max_bytes && text.size() > o.max_bytes)
    throw std::length_error("parser: bytes");
  ShuntingYardRun r;
  std::string     ops;
  bool            operand  = true;
  std::size_t     retained = 0;
  auto            emit     = [&](const std::string& t) {
    if (!r.postfix.empty())
      r.postfix += ' ';
    r.postfix += t;
  };
  auto pop = [&]() {
    emit(std::string(1, ops.back()));
    ops.pop_back();
    ++r.operator_pops;
  };
  auto record = [&](const std::string& t) {
    if (capture) {
      budget(retained, 1, o.max_history_values);
      budget(retained, t.size(), o.max_history_values);
      budget(retained, ops.size(), o.max_history_values);
      budget(retained, r.postfix.size(), o.max_history_values);
      r.conversion.push_back({t, ops, r.postfix});
    }
  };
  for (std::size_t i = 0; i < text.size();) {
    const unsigned char c = text[i];
    if (std::isspace(c)) {
      ++i;
      continue;
    }
    std::string tok;
    if (std::isdigit(c) || c == '.') {
      if (!operand)
        throw std::invalid_argument("parser: missing operator");
      const auto first = i;
      while (i < text.size() &&
             (std::isdigit(static_cast<unsigned char>(text[i])) || text[i] == '.'))
        ++i;
      tok = text.substr(first, i - first);
      number(tok, false);
      emit(tok);
      operand = false;
    } else if (c == '(') {
      if (!operand)
        throw std::invalid_argument("parser: implicit product");
      tok = "(";
      ops += '(';
      ++i;
    } else if (c == ')') {
      if (operand)
        throw std::invalid_argument("parser: missing operand");
      tok = ")";
      while (!ops.empty() && ops.back() != '(')
        pop();
      if (ops.empty())
        throw std::invalid_argument("parser: unmatched close");
      ops.pop_back();
      ++i;
    } else if (op(c)) {
      if (operand)
        throw std::invalid_argument("parser: binary operator");
      tok = std::string(1, c);
      while (!ops.empty() && op(ops.back()) &&
             (precedence(ops.back()) > precedence(c) ||
              (precedence(ops.back()) == precedence(c) && c != '^')))
        pop();
      ops += c;
      operand = true;
      ++i;
    } else
      throw std::invalid_argument("parser: unknown character");
    ++r.tokens;
    record(tok);
  }
  if (operand)
    throw std::invalid_argument("parser: incomplete expression");
  while (!ops.empty()) {
    if (ops.back() == '(')
      throw std::invalid_argument("parser: unmatched open");
    pop();
  }
  record("flush");
  return r;
}
/// @brief Evaluate postfix with actual stack records.
/// @param text Complete postfix.
/// @param o Policies.
/// @param records Optional capture destination.
/// @return Finite actual value.
double
evaluate(const std::string& text, const ShuntingYardOptions& o, std::vector<RpnStep>* records) {
  if (o.max_bytes && text.size() > o.max_bytes)
    throw std::length_error("rpn: bytes");
  std::istringstream  in(text);
  std::string         token;
  std::vector<double> stack;
  std::size_t         retained = 0;
  while (in >> token) {
    if (token.size() == 1 && op(token[0])) {
      if (stack.size() < 2)
        throw std::invalid_argument("rpn: operands");
      const double b = stack.back();
      stack.pop_back();
      const double a = stack.back();
      stack.pop_back();
      double value = 0;
      switch (token[0]) {
        case '+':
          value = a + b;
          break;
        case '-':
          value = a - b;
          break;
        case '*':
          value = a * b;
          break;
        case '/':
          if (b == 0)
            throw std::domain_error("rpn: zero divisor");
          value = a / b;
          break;
        case '^':
          value = std::pow(a, b);
          break;
      }
      if (!std::isfinite(value))
        throw std::domain_error("rpn: nonfinite result");
      stack.push_back(value);
    } else
      stack.push_back(number(token, true));
    if (records) {
      budget(retained, 1, o.max_history_values);
      budget(retained, token.size(), o.max_history_values);
      budget(retained, stack.size(), o.max_history_values);
      records->push_back({token, stack});
    }
  }
  if (stack.size() != 1)
    throw std::invalid_argument("rpn: final arity");
  return stack.back();
}
} // namespace
ShuntingYardRun shunting_yard_run(const std::string& t, const ShuntingYardOptions& o) {
  auto r         = convert(t, o, false);
  auto eval      = o;
  eval.max_bytes = 0;
  r.value        = evaluate(r.postfix, eval, nullptr);
  return r;
}
ShuntingYardRun shunting_yard_trace(const std::string& t, const ShuntingYardOptions& o) {
  auto r         = convert(t, o, true);
  auto eval      = o;
  eval.max_bytes = 0;
  r.value        = evaluate(r.postfix, eval, &r.evaluation);
  return r;
}
std::string shunting_yard_to_postfix(const std::string& t) {
  return convert(t, {}, false).postfix;
}
double evaluate_rpn(const std::string& t) {
  return evaluate(t, {}, nullptr);
}
double shunting_yard_evaluate(const std::string& t) {
  return shunting_yard_run(t).value;
}
} // namespace datamunge::algorithms
