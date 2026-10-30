#pragma once

/// @file ode.hpp
/// @brief std::format / std::print support for datamunge::ode.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// @par Example
/// @code
/// const dm::ode::ODESolver solver(dm::ode::ODEOptions{.step_size = 0.25});
/// std::print("{}\n", solver.solve_builtin("exponential_decay", {1.0}, {1.0}, 0.0, 1.0));
/// // ODESolution(points=5, t=[0, 1], steps_taken=4, function_evaluations=16, y_end=[0.3678941994067486])
/// @endcode

#include <datamunge/format/text_formatter.hpp>
#include <datamunge/ode/ode_solver.hpp>

#include <format>
#include <string>

/// @brief Prints an ode::StepMethod as its enumerator name (e.g. "RK45").
template <>
struct std::formatter<datamunge::ode::StepMethod>
    : datamunge::format_detail::text_formatter<datamunge::ode::StepMethod> {
  /// @brief The enumerator name.
  /// @param method The step method.
  /// @return The enumerator name, or "StepMethod(<int>)" for an unknown value.
  static std::string text(datamunge::ode::StepMethod method) {
    using M = datamunge::ode::StepMethod;
    switch (method) {
      case M::Euler: return "Euler";
      case M::Midpoint: return "Midpoint";
      case M::RK4: return "RK4";
      case M::RK45: return "RK45";
      case M::AdamsBashforth: return "AdamsBashforth";
      case M::AdamsMoulton: return "AdamsMoulton";
    }
    return std::format("StepMethod({})", static_cast<int>(method));
  }
};

/// @brief Prints ode::ODEOptions as one line of its fields.
template <>
struct std::formatter<datamunge::ode::ODEOptions>
    : datamunge::format_detail::text_formatter<datamunge::ode::ODEOptions> {
  /// @brief The solver options.
  /// @param o The options.
  /// @return "ODEOptions(method=..., multistep_order=..., step_size=..., abs_tol=..., rel_tol=..., max_step=...)".
  static std::string text(const datamunge::ode::ODEOptions& o) {
    return std::format("ODEOptions(method={}, multistep_order={}, step_size={}, abs_tol={}, rel_tol={}, max_step={})",
                       o.method, o.multistep_order, o.step_size, o.abs_tol, o.rel_tol, o.max_step);
  }
};

/// @brief Prints an ode::ODESolution as a one-line summary (time span, step counts, final state).
template <>
struct std::formatter<datamunge::ode::ODESolution>
    : datamunge::format_detail::text_formatter<datamunge::ode::ODESolution> {
  /// @brief A compact trajectory summary; the full t/y arrays are not dumped.
  /// @param s The solution.
  /// @return "ODESolution(points=n, t=[t0, t1], steps_taken=k, function_evaluations=m, y_end=[...])".
  static std::string text(const datamunge::ode::ODESolution& s) {
    if (s.t.empty()) return "ODESolution(points=0)";
    std::string out = std::format("ODESolution(points={}, t=[{}, {}], steps_taken={}, function_evaluations={}", s.size(),
                                  s.t.front(), s.t.back(), s.steps_taken, s.function_evaluations);
    if (!s.y.empty()) out += ", y_end=" + datamunge::format_detail::preview(s.y.back());
    return out + ")";
  }
};

/// @brief Prints an ode::ODESolver as "ODESolver(...)" with its configured options.
template <>
struct std::formatter<datamunge::ode::ODESolver>
    : datamunge::format_detail::text_formatter<datamunge::ode::ODESolver> {
  /// @brief The type name followed by every option field (the ODEOptions text, renamed).
  /// @param s The solver.
  /// @return "ODESolver(method=..., ...)".
  static std::string text(const datamunge::ode::ODESolver& s) {
    const std::string options = std::format("{}", s.options());
    const auto open = options.find('(');
    return "ODESolver" + (open == std::string::npos ? std::string("()") : options.substr(open));
  }
};
