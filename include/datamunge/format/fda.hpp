#pragma once

/// @file fda.hpp
/// @brief std::format / std::print support for datamunge::fda.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::fda::BSplineBasis::open_uniform(3, 6));
/// // BSplineBasis(degree=3, basis_size=6, domain=[0, 1], knots=[0, 0, 0, 0, 0.3333333333333333, ..., 1] (n=10))
/// @endcode

#include <datamunge/fda/bspline.hpp>
#include <datamunge/fda/pspline.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <format>
#include <string>

/// @brief Prints an fda::LocalBSplineEvaluation as its non-zero basis indices and values.
template <>
struct std::formatter<datamunge::fda::LocalBSplineEvaluation>
    : datamunge::format_detail::text_formatter<datamunge::fda::LocalBSplineEvaluation> {
  /// @brief The local (non-zero) basis indices and values.
  /// @param e The local evaluation.
  /// @return "LocalBSplineEvaluation(indices=[...], values=[...])".
  static std::string text(const datamunge::fda::LocalBSplineEvaluation& e) {
    return std::format("LocalBSplineEvaluation(indices={}, values={})", datamunge::format_detail::preview(e.indices),
                       datamunge::format_detail::preview(e.values));
  }
};

/// @brief Prints an fda::BSplineBasis as its degree, size, domain and knot preview.
template <>
struct std::formatter<datamunge::fda::BSplineBasis>
    : datamunge::format_detail::text_formatter<datamunge::fda::BSplineBasis> {
  /// @brief The basis configuration.
  /// @param b The basis.
  /// @return "BSplineBasis(degree=d, basis_size=n, domain=[a, b], knots=[...])".
  static std::string text(const datamunge::fda::BSplineBasis& b) {
    const auto [lo, hi] = b.domain();
    return std::format("BSplineBasis(degree={}, basis_size={}, domain=[{}, {}], knots={})", b.degree(), b.basis_size(),
                       lo, hi, datamunge::format_detail::preview(b.knots()));
  }
};

/// @brief Prints an fda::BSpline (tensor-product basis) as its dimension, total size and per-axis degree/size.
template <>
struct std::formatter<datamunge::fda::BSpline> : datamunge::format_detail::text_formatter<datamunge::fda::BSpline> {
  /// @brief The tensor-product configuration.
  /// @param s The spline basis.
  /// @return "BSpline(dimensions=k, basis_size=n, degrees=[...], sizes=[...])".
  static std::string text(const datamunge::fda::BSpline& s) {
    std::vector<std::size_t> degrees, sizes;
    for (const auto& b : s.bases()) {
      degrees.push_back(b.degree());
      sizes.push_back(b.basis_size());
    }
    return std::format("BSpline(dimensions={}, basis_size={}, degrees={}, sizes={})", s.dimensions(), s.basis_size(),
                       datamunge::format_detail::preview(degrees), datamunge::format_detail::preview(sizes));
  }
};

/// @brief Prints fda::PSplineOptions as "PSplineOptions(lambda=..., penalty_order=...)".
template <>
struct std::formatter<datamunge::fda::PSplineOptions>
    : datamunge::format_detail::text_formatter<datamunge::fda::PSplineOptions> {
  /// @brief The smoothing options.
  /// @param o The options.
  /// @return "PSplineOptions(lambda=l, penalty_order=k)".
  static std::string text(const datamunge::fda::PSplineOptions& o) {
    return std::format("PSplineOptions(lambda={}, penalty_order={})", o.lambda, o.penalty_order);
  }
};

/// @brief Prints an fda::PSplineFitResult as a short multi-line fit summary.
template <>
struct std::formatter<datamunge::fda::PSplineFitResult>
    : datamunge::format_detail::text_formatter<datamunge::fda::PSplineFitResult> {
  /// @brief The fit statistics and a coefficient preview (fitted/residual arrays are not dumped).
  /// @param r The fit result.
  /// @return A multi-line "PSplineFitResult" summary.
  static std::string text(const datamunge::fda::PSplineFitResult& r) {
    return std::format(
        "PSplineFitResult(n={}, lambda={}, penalty_order={})\n"
        "  basis: {}\n"
        "  coefficients: {}\n"
        "  RSS={}, penalty={}, objective={}\n"
        "  R2={}, adj_R2={}, edf={}, residual_df={}",
        r.observations, r.lambda, r.penalty_order, r.basis, datamunge::format_detail::preview(r.coefficients),
        r.residual_sum_of_squares, r.penalty, r.objective, r.r_squared, r.adjusted_r_squared,
        r.effective_degrees_of_freedom, r.residual_degrees_of_freedom);
  }
};
