#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>

#include <datamunge/dstruct/matrix.hpp>

namespace datamunge::dstruct {

template <typename T, typename Hash = std::hash<T>>
class CountMinSketch {
 public:
  CountMinSketch(std::size_t width, std::size_t depth) : counters_(depth, width), width_(width), depth_(depth) { if (width == 0 || depth == 0) throw std::invalid_argument("CountMinSketch requires positive dimensions"); }
  void add(const T& value, std::size_t count = 1) { for (std::size_t row = 0; row < depth_; ++row) counters_.at(row, index(value, row)) += count; }
  std::size_t estimate(const T& value) const { std::size_t result = counters_.at(0, index(value, 0)); for (std::size_t row = 1; row < depth_; ++row) { const auto candidate = counters_.at(row, index(value, row)); if (candidate < result) result = candidate; } return result; }
  /// @brief Number of counters per hash row, fixed at construction.
  /// @return The width (always positive).
  [[nodiscard]] std::size_t width() const noexcept { return width_; }
  /// @brief Number of hash rows, fixed at construction.
  /// @return The depth (always positive).
  [[nodiscard]] std::size_t depth() const noexcept { return depth_; }
 private:
  std::size_t index(const T& value, std::size_t salt) const { const auto base = hash_(value); return (base ^ (salt * 0x9e3779b9U)) % width_; }
  Matrix<std::size_t> counters_; std::size_t width_, depth_; Hash hash_{};
};

}  // namespace datamunge::dstruct
