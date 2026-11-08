#pragma once

#include <datamunge/algorithms/layout_geometry.hpp>
#include <datamunge/algorithms/force_directed_layout.hpp>
#include <datamunge/algorithms/circle_packing.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

std::vector<Point2D> spectral_layout(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges);

} // namespace datamunge::algorithms
