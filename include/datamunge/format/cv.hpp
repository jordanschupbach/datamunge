#pragma once

/// @file cv.hpp
/// @brief std::format / std::print support for datamunge::cv.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Detections print as `TypeName(field=value, ...)`; a LabelMap prints a header line plus a
/// capped grid of labels (at most 10 rows x 8 columns, elided with "...").
///
/// @par Example
/// @code
/// const dm::cv::HoughCircle c{10, 12, 5, 42};
/// std::print("{}\n", c);   // HoughCircle(cx=10, cy=12, radius=5, votes=42)
/// @endcode

#include <datamunge/cv/cv.hpp>
#include <datamunge/format/text_formatter.hpp>

#include <cstddef>
#include <format>
#include <string>

/// @brief Prints a KeyPoint as "KeyPoint(x=.., y=.., scale=.., response=..)".
template <>
struct std::formatter<datamunge::cv::KeyPoint> : datamunge::format_detail::text_formatter<datamunge::cv::KeyPoint> {
  /// @brief Location, scale and response.
  /// @param k The keypoint to print.
  /// @return The one-line description.
  static std::string text(const datamunge::cv::KeyPoint& k) {
    return std::format("KeyPoint(x={}, y={}, scale={}, response={})", k.x, k.y, k.scale, k.response);
  }
};

/// @brief Prints a Corner as "Corner(x=.., y=.., response=..)".
template <>
struct std::formatter<datamunge::cv::Corner> : datamunge::format_detail::text_formatter<datamunge::cv::Corner> {
  /// @brief Location and response.
  /// @param c The corner to print.
  /// @return The one-line description.
  static std::string text(const datamunge::cv::Corner& c) {
    return std::format("Corner(x={}, y={}, response={})", c.x, c.y, c.response);
  }
};

/// @brief Prints a HoughLine as "HoughLine(rho=.., theta=.., votes=..)".
template <>
struct std::formatter<datamunge::cv::HoughLine> : datamunge::format_detail::text_formatter<datamunge::cv::HoughLine> {
  /// @brief The normal-form parameters and vote count.
  /// @param l The line to print.
  /// @return The one-line description.
  static std::string text(const datamunge::cv::HoughLine& l) {
    return std::format("HoughLine(rho={}, theta={}, votes={})", l.rho, l.theta, l.votes);
  }
};

/// @brief Prints a HoughCircle as "HoughCircle(cx=.., cy=.., radius=.., votes=..)".
template <>
struct std::formatter<datamunge::cv::HoughCircle>
    : datamunge::format_detail::text_formatter<datamunge::cv::HoughCircle> {
  /// @brief Centre, radius and vote count.
  /// @param c The circle to print.
  /// @return The one-line description.
  static std::string text(const datamunge::cv::HoughCircle& c) {
    return std::format("HoughCircle(cx={}, cy={}, radius={}, votes={})", c.cx, c.cy, c.radius, c.votes);
  }
};

/// @brief Prints a FlowVector as "FlowVector(dx=.., dy=.., valid=..)".
template <>
struct std::formatter<datamunge::cv::FlowVector> : datamunge::format_detail::text_formatter<datamunge::cv::FlowVector> {
  /// @brief The displacement and validity flag.
  /// @param f The flow vector to print.
  /// @return The one-line description.
  static std::string text(const datamunge::cv::FlowVector& f) {
    return std::format("FlowVector(dx={}, dy={}, valid={})", f.dx, f.dy, f.valid);
  }
};

/// @brief Prints a LabelMap as "LabelMap WxH, count=N" followed by a capped grid of labels.
template <>
struct std::formatter<datamunge::cv::LabelMap> : datamunge::format_detail::text_formatter<datamunge::cv::LabelMap> {
  /// @brief The header line plus up to 10 rows x 8 columns of labels ("..." marks elided rows/columns).
  /// @param m The label map to print.
  /// @return The multi-line description.
  static std::string text(const datamunge::cv::LabelMap& m) {
    constexpr int max_rows = 10;
    constexpr int max_cols = 8;
    std::string out = std::format("LabelMap {}x{}, count={}", m.width, m.height, m.count);
    if (m.width <= 0 || m.height <= 0 ||
        m.labels.size() < static_cast<std::size_t>(m.width) * static_cast<std::size_t>(m.height)) {
      return out;
    }
    const int rows = m.height < max_rows ? m.height : max_rows;
    const int cols = m.width < max_cols ? m.width : max_cols;
    for (int y = 0; y < rows; ++y) {
      out += "\n ";
      for (int x = 0; x < cols; ++x) {
        out += std::format(" {}", m.labels[static_cast<std::size_t>(y) * static_cast<std::size_t>(m.width) +
                                           static_cast<std::size_t>(x)]);
      }
      if (cols < m.width) out += " ...";
    }
    if (rows < m.height) out += "\n  ...";
    return out;
  }
};

/// @brief Prints a StructuringElement as its enumerator name (Square, Cross).
template <>
struct std::formatter<datamunge::cv::StructuringElement>
    : datamunge::format_detail::text_formatter<datamunge::cv::StructuringElement> {
  /// @brief The enumerator name, or "StructuringElement(<int>)" for an unknown value.
  /// @param se The structuring element to print.
  /// @return The enumerator name.
  static std::string text(datamunge::cv::StructuringElement se) {
    using datamunge::cv::StructuringElement;
    switch (se) {
      case StructuringElement::Square: return "Square";
      case StructuringElement::Cross: return "Cross";
    }
    return std::format("StructuringElement({})", static_cast<int>(se));
  }
};

/// @brief Prints a PoolMode as its enumerator name (Max, Average).
template <>
struct std::formatter<datamunge::cv::PoolMode> : datamunge::format_detail::text_formatter<datamunge::cv::PoolMode> {
  /// @brief The enumerator name, or "PoolMode(<int>)" for an unknown value.
  /// @param mode The pooling mode to print.
  /// @return The enumerator name.
  static std::string text(datamunge::cv::PoolMode mode) {
    using datamunge::cv::PoolMode;
    switch (mode) {
      case PoolMode::Max: return "Max";
      case PoolMode::Average: return "Average";
    }
    return std::format("PoolMode({})", static_cast<int>(mode));
  }
};
