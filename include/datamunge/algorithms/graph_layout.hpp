#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief A planar point shared by graph layouts and polygon rasterization.
struct Point2D {
  double x{0.0}; ///< Horizontal coordinate.
  double y{0.0}; ///< Vertical coordinate.

  /// @brief Borrows coordinate 0 (x) or 1 (y); index must be less than two.
  double& operator[](std::size_t index) noexcept { return index == 0 ? x : y; }
  /// @brief Borrows a read-only coordinate; index must be less than two.
  const double& operator[](std::size_t index) const noexcept { return index == 0 ? x : y; }
};

struct Circle2D {
  Point2D center;
  double  radius{1.0};
};

std::vector<Circle2D> circle_packing_layout(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t iterations = 1000);

std::vector<Point2D> force_directed_layout(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t iterations = 500, std::uint64_t seed = 42);

std::vector<Point2D> spectral_layout(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges);

} // namespace datamunge::algorithms
