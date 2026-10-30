#pragma once

/// @file fem.hpp
/// @brief std::format / std::print support for datamunge::fem.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::fem::make_uniform_mesh1d(0.0, 1.0, 4));
/// // Mesh1D(nodes=5, elements=4, domain=[0, 1], x=[0, 0.25, 0.5, 0.75, 1])
/// @endcode

#include <datamunge/fem/fem.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <algorithm>
#include <format>
#include <string>

/// @brief Prints an fem::BCType as its enumerator name (e.g. "Robin").
template <>
struct std::formatter<datamunge::fem::BCType> : datamunge::format_detail::text_formatter<datamunge::fem::BCType> {
  /// @brief The enumerator name.
  /// @param type The boundary-condition type.
  /// @return "Dirichlet", "Neumann" or "Robin", or "BCType(<int>)" for an unknown value.
  static std::string text(datamunge::fem::BCType type) {
    switch (type) {
      case datamunge::fem::BCType::Dirichlet: return "Dirichlet";
      case datamunge::fem::BCType::Neumann: return "Neumann";
      case datamunge::fem::BCType::Robin: return "Robin";
    }
    return std::format("BCType({})", static_cast<int>(type));
  }
};

/// @brief Prints an fem::BoundaryCondition1D as one line of its fields.
template <>
struct std::formatter<datamunge::fem::BoundaryCondition1D>
    : datamunge::format_detail::text_formatter<datamunge::fem::BoundaryCondition1D> {
  /// @brief The condition type and its values (robin_coefficient only for Robin).
  /// @param bc The boundary condition.
  /// @return "BoundaryCondition1D(type=..., value=...[, robin_coefficient=...])".
  static std::string text(const datamunge::fem::BoundaryCondition1D& bc) {
    if (bc.type == datamunge::fem::BCType::Robin)
      return std::format("BoundaryCondition1D(type={}, value={}, robin_coefficient={})", bc.type, bc.value,
                         bc.robin_coefficient);
    return std::format("BoundaryCondition1D(type={}, value={})", bc.type, bc.value);
  }
};

/// @brief Prints an fem::Mesh1D as its size, domain and a node preview.
template <>
struct std::formatter<datamunge::fem::Mesh1D> : datamunge::format_detail::text_formatter<datamunge::fem::Mesh1D> {
  /// @brief The mesh size and node coordinates.
  /// @param m The mesh.
  /// @return "Mesh1D(nodes=n, elements=e, domain=[a, b], x=[...])".
  static std::string text(const datamunge::fem::Mesh1D& m) {
    if (m.nodes.empty()) return "Mesh1D(nodes=0)";
    return std::format("Mesh1D(nodes={}, elements={}, domain=[{}, {}], x={})", m.num_nodes(), m.num_elements(),
                       m.nodes.front(), m.nodes.back(), datamunge::format_detail::preview(m.nodes));
  }
};

/// @brief Prints an fem::Mesh2D as its node/triangle counts and bounding box.
template <>
struct std::formatter<datamunge::fem::Mesh2D> : datamunge::format_detail::text_formatter<datamunge::fem::Mesh2D> {
  /// @brief The mesh size and the bounding box of its nodes.
  /// @param m The mesh.
  /// @return "Mesh2D(nodes=n, triangles=t, bbox=[x0, x1] x [y0, y1])".
  static std::string text(const datamunge::fem::Mesh2D& m) {
    std::string out = std::format("Mesh2D(nodes={}, triangles={}", m.num_nodes(), m.num_triangles());
    if (!m.nodes.empty()) {
      double x0 = m.nodes.front().x, x1 = x0, y0 = m.nodes.front().y, y1 = y0;
      for (const auto& p : m.nodes) {
        x0 = std::min(x0, p.x);
        x1 = std::max(x1, p.x);
        y0 = std::min(y0, p.y);
        y1 = std::max(y1, p.y);
      }
      out += std::format(", bbox=[{}, {}] x [{}, {}]", x0, x1, y0, y1);
    }
    return out + ")";
  }
};

/// @brief Prints an fem::FEM1DResult as its size, domain and a solution preview.
template <>
struct std::formatter<datamunge::fem::FEM1DResult>
    : datamunge::format_detail::text_formatter<datamunge::fem::FEM1DResult> {
  /// @brief The nodal solution summary.
  /// @param r The result.
  /// @return "FEM1DResult(nodes=n, domain=[a, b], u=[...])".
  static std::string text(const datamunge::fem::FEM1DResult& r) {
    if (r.nodes.empty()) return std::format("FEM1DResult(nodes=0, u={})", datamunge::format_detail::preview(r.u));
    return std::format("FEM1DResult(nodes={}, domain=[{}, {}], u={})", r.nodes.size(), r.nodes.front(), r.nodes.back(),
                       datamunge::format_detail::preview(r.u));
  }
};

/// @brief Prints an fem::FEM1DTimeSeries as its size, time span and final-time solution preview.
template <>
struct std::formatter<datamunge::fem::FEM1DTimeSeries>
    : datamunge::format_detail::text_formatter<datamunge::fem::FEM1DTimeSeries> {
  /// @brief The transient solution summary (only the last time slice is previewed).
  /// @param s The time series.
  /// @return "FEM1DTimeSeries(nodes=n, steps=k, t=[t0, t1], u_end=[...])".
  static std::string text(const datamunge::fem::FEM1DTimeSeries& s) {
    std::string out = std::format("FEM1DTimeSeries(nodes={}, steps={}", s.num_nodes(), s.num_steps());
    if (!s.times.empty()) out += std::format(", t=[{}, {}]", s.times.front(), s.times.back());
    if (!s.u.empty()) out += ", u_end=" + datamunge::format_detail::preview(s.u.back());
    return out + ")";
  }
};

/// @brief Prints an fem::FEM2DResult as its size and the solution's range.
template <>
struct std::formatter<datamunge::fem::FEM2DResult>
    : datamunge::format_detail::text_formatter<datamunge::fem::FEM2DResult> {
  /// @brief The nodal solution summary (node coordinates are not dumped).
  /// @param r The result.
  /// @return "FEM2DResult(nodes=n, u_min=a, u_max=b)".
  static std::string text(const datamunge::fem::FEM2DResult& r) {
    if (r.u.empty()) return std::format("FEM2DResult(nodes={})", r.nodes.size());
    const auto [lo, hi] = std::minmax_element(r.u.begin(), r.u.end());
    return std::format("FEM2DResult(nodes={}, u_min={}, u_max={})", r.u.size(), *lo, *hi);
  }
};
