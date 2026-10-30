#pragma once

/// @file algebra.hpp
/// @brief std::format / std::print support for datamunge::algebra.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Polynomials, expressions and rational functions print via their own to_string();
/// factorizations print as `TypeName(factor=(...), multiplicity=k)`.
///
/// @par Example
/// @code
/// const dm::algebra::Polynomial p({-1.0, 0.0, 1.0});
/// std::print("{}\n", p);   // x^2 - 1
/// @endcode

#include <datamunge/algebra/algebra.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <format>
#include <string>

/// @brief Prints a Polynomial via its to_string(), e.g. "x^2 - 1".
template <>
struct std::formatter<datamunge::algebra::Polynomial>
    : datamunge::format_detail::text_formatter<datamunge::algebra::Polynomial> {
  /// @brief The polynomial in descending powers of x.
  /// @param p The polynomial to print.
  /// @return p.to_string().
  static std::string text(const datamunge::algebra::Polynomial& p) { return p.to_string(); }
};

/// @brief Prints a MultivariatePolynomial via its to_string() (grevlex term order).
template <>
struct std::formatter<datamunge::algebra::MultivariatePolynomial>
    : datamunge::format_detail::text_formatter<datamunge::algebra::MultivariatePolynomial> {
  /// @brief The polynomial's terms in grevlex order.
  /// @param p The polynomial to print.
  /// @return p.to_string().
  static std::string text(const datamunge::algebra::MultivariatePolynomial& p) { return p.to_string(); }
};

/// @brief Prints a RationalFunction via its to_string(), e.g. "(x + 1) / (x - 1)".
template <>
struct std::formatter<datamunge::algebra::RationalFunction>
    : datamunge::format_detail::text_formatter<datamunge::algebra::RationalFunction> {
  /// @brief Numerator over denominator.
  /// @param f The rational function to print.
  /// @return f.to_string().
  static std::string text(const datamunge::algebra::RationalFunction& f) { return f.to_string(); }
};

/// @brief Prints a symbolic Expr via its to_string() (fully parenthesized infix).
template <>
struct std::formatter<datamunge::algebra::Expr> : datamunge::format_detail::text_formatter<datamunge::algebra::Expr> {
  /// @brief The expression tree as infix text.
  /// @param e The expression to print.
  /// @return e.to_string().
  static std::string text(const datamunge::algebra::Expr& e) { return e.to_string(); }
};

/// @brief Prints a Complex root as "Complex(re, im)".
template <>
struct std::formatter<datamunge::algebra::Complex>
    : datamunge::format_detail::text_formatter<datamunge::algebra::Complex> {
  /// @brief The real and imaginary parts.
  /// @param z The complex number to print.
  /// @return "Complex(re, im)".
  static std::string text(const datamunge::algebra::Complex& z) { return std::format("Complex({}, {})", z.re, z.im); }
};

/// @brief Prints a RealFactor as "RealFactor(factor=(..), multiplicity=k)".
template <>
struct std::formatter<datamunge::algebra::RealFactor>
    : datamunge::format_detail::text_formatter<datamunge::algebra::RealFactor> {
  /// @brief The irreducible factor and its multiplicity.
  /// @param f The factor to print.
  /// @return The one-line description.
  static std::string text(const datamunge::algebra::RealFactor& f) {
    return std::format("RealFactor(factor=({}), multiplicity={})", f.factor, f.multiplicity);
  }
};

/// @brief Prints a SquareFreeFactor as "SquareFreeFactor(factor=(..), multiplicity=k)".
template <>
struct std::formatter<datamunge::algebra::SquareFreeFactor>
    : datamunge::format_detail::text_formatter<datamunge::algebra::SquareFreeFactor> {
  /// @brief The square-free factor and its multiplicity.
  /// @param f The factor to print.
  /// @return The one-line description.
  static std::string text(const datamunge::algebra::SquareFreeFactor& f) {
    return std::format("SquareFreeFactor(factor=({}), multiplicity={})", f.factor, f.multiplicity);
  }
};

/// @brief Prints a RealFactorization as a product, e.g. "2 * (x - 1)^2 * (x^2 + 1)".
template <>
struct std::formatter<datamunge::algebra::RealFactorization>
    : datamunge::format_detail::text_formatter<datamunge::algebra::RealFactorization> {
  /// @brief The leading coefficient times each factor raised to its multiplicity (exponent 1 omitted).
  /// @param f The factorization to print.
  /// @return The product text.
  static std::string text(const datamunge::algebra::RealFactorization& f) {
    std::string out = std::format("{}", f.leading_coefficient);
    for (const auto& factor : f.factors) {
      out += std::format(" * ({})", factor.factor);
      if (factor.multiplicity != 1) out += std::format("^{}", factor.multiplicity);
    }
    return out;
  }
};

/// @brief Prints a PolyExtendedGcdResult as "PolyExtendedGcdResult(gcd=(..), s=(..), t=(..))".
template <>
struct std::formatter<datamunge::algebra::PolyExtendedGcdResult>
    : datamunge::format_detail::text_formatter<datamunge::algebra::PolyExtendedGcdResult> {
  /// @brief The gcd and its Bezout coefficients.
  /// @param r The result to print.
  /// @return The one-line description.
  static std::string text(const datamunge::algebra::PolyExtendedGcdResult& r) {
    return std::format("PolyExtendedGcdResult(gcd=({}), s=({}), t=({}))", r.gcd, r.s, r.t);
  }
};

/// @brief Prints RealRootIntervals as "RealRootIntervals([lo, hi], [lo, hi], ...)" (at most 6 shown).
template <>
struct std::formatter<datamunge::algebra::RealRootIntervals>
    : datamunge::format_detail::text_formatter<datamunge::algebra::RealRootIntervals> {
  /// @brief Each isolating interval; long lists are elided with "..." and a count.
  /// @param r The intervals to print.
  /// @return The one-line description.
  static std::string text(const datamunge::algebra::RealRootIntervals& r) {
    constexpr std::size_t max_shown = 6;
    const std::size_t n = r.lower.size() < r.upper.size() ? r.lower.size() : r.upper.size();
    std::string out = "RealRootIntervals(";
    for (std::size_t i = 0; i < n && i < max_shown; ++i) {
      if (i > 0) out += ", ";
      out += std::format("[{}, {}]", r.lower[i], r.upper[i]);
    }
    if (n > max_shown) out += std::format(", ... (n={})", n);
    return out + ")";
  }
};

/// @brief Prints a MonomialOrder as its enumerator name (Lex, Grlex, Grevlex).
template <>
struct std::formatter<datamunge::algebra::MonomialOrder>
    : datamunge::format_detail::text_formatter<datamunge::algebra::MonomialOrder> {
  /// @brief The enumerator name, or "MonomialOrder(<int>)" for an unknown value.
  /// @param order The order to print.
  /// @return The enumerator name.
  static std::string text(datamunge::algebra::MonomialOrder order) {
    using datamunge::algebra::MonomialOrder;
    switch (order) {
      case MonomialOrder::Lex: return "Lex";
      case MonomialOrder::Grlex: return "Grlex";
      case MonomialOrder::Grevlex: return "Grevlex";
    }
    return std::format("MonomialOrder({})", static_cast<int>(order));
  }
};
