#pragma once

/// @file tex.hpp
/// @brief Shared C++ mathematical serialization for symbolic algebra values.

#include <cmath>
#include <datamunge/tex.hpp>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace datamunge::algebra::detail {

/// @brief Validate a significant-digit count shared by algebra's TeX serializers.
/// @param precision Significant decimal digits, in [1, double's max_digits10].
/// @throws std::invalid_argument If precision is outside that range.
inline void validate_tex_precision(int precision) {
  if (precision < 1 || precision > std::numeric_limits<double>::max_digits10)
    throw std::invalid_argument("to_tex: precision must be between 1 and double's max_digits10");
}

/// @brief Serialize one double with locale-independent decimal/scientific TeX notation.
/// @param value Number; signed zero prints as 0, infinity and NaN use explicit math commands.
/// @param precision Significant decimal digits, in [1, double's max_digits10].
/// @return Raw math source without delimiters; finite exponents use an integer power of ten.
/// @throws std::invalid_argument If precision is outside the supported range.
inline std::string tex_number(double value, int precision) {
  validate_tex_precision(precision);
  if (std::isnan(value))
    return "\\mathrm{NaN}";
  if (std::isinf(value))
    return value < 0 ? "-\\infty" : "\\infty";
  if (value == 0)
    return "0";
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(precision) << value;
  const auto text     = stream.str();
  const auto exponent = text.find_first_of("eE");
  if (exponent == std::string::npos)
    return text;
  return text.substr(0, exponent) + " \\times 10^{" +
         std::to_string(std::stoi(text.substr(exponent + 1))) + "}";
}

/// @brief Render a logical variable name literally, with single ASCII letters in math italics.
/// @param name Literal variable name, including an empty name; longer names are text-escaped.
/// @return Raw math symbol or escaped amsmath text; whitespace follows Tex::text().
inline std::string tex_symbol(const std::string& name) {
  if (name.size() == 1 &&
      ((name[0] >= 'a' && name[0] <= 'z') || (name[0] >= 'A' && name[0] <= 'Z')))
    return name;
  return Tex::text(name).str();
}

/// @brief A rendered tree node together with its mathematical binding strength.
struct RenderedTex {
  std::string source;     ///< Raw math source owned by this rendered node.
  int         precedence; ///< Sum=1, product=2, unary minus=3, power=4, atomic/function/fraction=5.
};

/// @brief Add sized parentheses around a previously rendered node when grouping is needed.
/// @param node Rendered node to wrap.
/// @return Raw TeX source in a left/right parenthesis pair.
inline std::string tex_parentheses(const RenderedTex& node) {
  return "\\left(" + node.source + "\\right)";
}

/// @brief Render the immutable expression node tree without simplification or evaluation.
/// @tparam NodePtr Pointer to a node with op, value, name, left and right fields.
/// @param node Non-null tree node; children follow the expression operation's arity.
/// @param precision Significant decimal digits for constants; already validated by the caller.
/// @return Owned raw math source and binding strength. Fractions and function arguments group
///         their children; powers distinguish negative/nested-power bases and all exponents.
/// @throws std::invalid_argument For invalid numeric precision.
/// @throws std::logic_error If the node has an unknown operation tag.
template<class NodePtr>
RenderedTex expression_tex(const NodePtr& node, int precision) {
  using Op         = decltype(node->op);
  const auto left  = node->left ? expression_tex(node->left, precision) : RenderedTex{"", 5};
  const auto right = node->right ? expression_tex(node->right, precision) : RenderedTex{"", 5};
  switch (node->op) {
    case Op::Const: {
      const auto text       = tex_number(node->value, precision);
      const int  precedence = text.find("\\times") != std::string::npos ? 2
                              : node->value < 0                         ? 3
                                                                        : 5;
      return {text, precedence};
    }
    case Op::Var:
      return {tex_symbol(node->name), 5};
    case Op::Add:
      return {
          left.source + " + " +
              (!right.source.empty() && right.source[0] == '-' ? tex_parentheses(right)
                                                               : right.source),
          1};
    case Op::Sub:
      return {
          left.source + " - " +
              (right.precedence <= 1 || (!right.source.empty() && right.source[0] == '-')
                   ? tex_parentheses(right)
                   : right.source),
          1};
    case Op::Mul:
      return {
          (left.precedence < 2 ? tex_parentheses(left) : left.source) + " \\cdot " +
              (right.precedence < 2 || (!right.source.empty() && right.source[0] == '-')
                   ? tex_parentheses(right)
                   : right.source),
          2};
    case Op::Div:
      return {"\\frac{" + left.source + "}{" + right.source + "}", 5};
    case Op::Pow:
      return {
          (left.precedence <= 4 || node->left->op == Op::Div ? tex_parentheses(left)
                                                             : left.source) +
              "^{" + right.source + "}",
          4};
    case Op::Neg:
      return {"-" + (left.precedence <= 3 ? tex_parentheses(left) : left.source), 3};
    case Op::Sin:
      return {"\\sin" + tex_parentheses(left), 5};
    case Op::Cos:
      return {"\\cos" + tex_parentheses(left), 5};
    case Op::Exp:
      return {"\\exp" + tex_parentheses(left), 5};
    case Op::Log:
      return {"\\log" + tex_parentheses(left), 5};
  }
  throw std::logic_error("to_tex: unknown expression operation");
}

/// @brief Serialize a dense univariate polynomial with descending powers and normalized signs.
/// @param coefficients Coefficients in ascending degree order; empty/zero vectors print 0.
/// @param variable Raw TeX symbol for the formal variable, e.g. x, h or \\theta; not an arbitrary
/// sum.
/// @param precision Significant decimal digits, in [1, double's max_digits10].
/// @return Raw math source; zero terms and unit coefficients of nonconstant terms are omitted.
/// @throws std::invalid_argument For an empty variable or an unsupported precision.
inline std::string polynomial_tex(
    const std::vector<double>& coefficients,
    const std::string&         variable,
    int                        precision) {
  validate_tex_precision(precision);
  if (variable.empty())
    throw std::invalid_argument("to_tex: variable must not be empty");
  std::string result;
  for (std::size_t degree = coefficients.size(); degree-- > 0;) {
    const double coefficient = coefficients[degree];
    if (coefficient == 0)
      continue;
    const bool negative = coefficient < 0;
    if (!result.empty())
      result += negative ? " - " : " + ";
    else if (negative)
      result += '-';
    const double magnitude = std::fabs(coefficient);
    if (degree == 0 || magnitude != 1) {
      result += tex_number(magnitude, precision);
      if (degree != 0)
        result += "\\,";
    }
    if (degree != 0) {
      if (degree == 1)
        result += variable;
      else
        result += "{" + variable + "}^{" + std::to_string(degree) + "}";
    }
  }
  return result.empty() ? "0" : result;
}

} // namespace datamunge::algebra::detail
