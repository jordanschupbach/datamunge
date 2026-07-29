#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

struct Point2D {
  double x{0.0};
  double y{0.0};
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
