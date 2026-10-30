#pragma once

/// @file autodiff.hpp
/// @brief std::format / std::print support for datamunge::autodiff.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::autodiff::Dual(2.0, 1.0));
/// // Dual(value=2, derivative=1)
/// @endcode

#include <datamunge/autodiff/dual.hpp>
#include <datamunge/autodiff/hyperdual.hpp>
#include <datamunge/autodiff/tape.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <format>
#include <string>

/// @brief Prints a forward-mode Dual as "Dual(value=v, derivative=d)".
template <>
struct std::formatter<datamunge::autodiff::Dual> : datamunge::format_detail::text_formatter<datamunge::autodiff::Dual> {
  /// @brief The value and its derivative.
  /// @param d The dual number.
  /// @return "Dual(value=v, derivative=d)".
  static std::string text(const datamunge::autodiff::Dual& d) {
    return std::format("Dual(value={}, derivative={})", d.value(), d.derivative());
  }
};

/// @brief Prints a HyperDual as "HyperDual(value=v, eps1=a, eps2=b, eps1eps2=c)".
template <>
struct std::formatter<datamunge::autodiff::HyperDual>
    : datamunge::format_detail::text_formatter<datamunge::autodiff::HyperDual> {
  /// @brief The value, both first-order parts and the mixed second-order part.
  /// @param d The hyper-dual number.
  /// @return "HyperDual(value=v, eps1=a, eps2=b, eps1eps2=c)".
  static std::string text(const datamunge::autodiff::HyperDual& d) {
    return std::format("HyperDual(value={}, eps1={}, eps2={}, eps1eps2={})", d.value(), d.eps1(), d.eps2(),
                       d.eps1eps2());
  }
};

/// @brief Prints a reverse-mode Tape as "Tape(nodes=n)".
template <>
struct std::formatter<datamunge::autodiff::Tape> : datamunge::format_detail::text_formatter<datamunge::autodiff::Tape> {
  /// @brief The number of recorded nodes.
  /// @param tape The tape.
  /// @return "Tape(nodes=n)".
  static std::string text(const datamunge::autodiff::Tape& tape) { return std::format("Tape(nodes={})", tape.size()); }
};

/// @brief Prints a tape Var as "Var(value=v, index=i)".
template <>
struct std::formatter<datamunge::autodiff::Var> : datamunge::format_detail::text_formatter<datamunge::autodiff::Var> {
  /// @brief The recorded value and its node index on the tape.
  /// @param v The tape variable.
  /// @return "Var(value=v, index=i)".
  static std::string text(const datamunge::autodiff::Var& v) {
    return std::format("Var(value={}, index={})", v.value(), v.index());
  }
};
