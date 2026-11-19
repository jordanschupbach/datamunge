#pragma once

/// @file soft_circle_raster.hpp
/// @brief Differentiable sampled rasterization of ordered soft circles.

#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Raster products for a scene of circles.
/// @par Example
/// See soft_circle_raster().
struct SoftCircleRaster {
  Tensor rgb;   ///< Composited colors [samples, 3] over a black background.
  Tensor alpha; ///< Individual coverage values [samples, circles] before occlusion.
};

/// @brief Samples sigmoid circle coverage and composites in front-to-back order.
/// Coverage is sigmoid(sharpness * (radius squared minus squared distance))
/// at each point. Circle order is fixed. The result is differentiable in the
/// coordinates, centers and colors.
/// @param coordinates Finite CPU Float32 or Float64 tensor [samples, 2].
/// @param centers Finite CPU matrix [circles, 2] in the coordinate dtype.
/// @param radii Positive radii, one per circle, with representable squares.
/// @param colors Finite CPU matrix [circles, 3] in the coordinate dtype.
/// @param sharpness Positive coverage slope representable in the coordinate dtype.
/// @return Composited RGB and the coverage of each circle.
/// @throws std::invalid_argument Incompatible sizes, dtypes, devices, radii or sharpness.
/// @par Example
/// @code
/// const auto xy = dm::nn::zeros({1, 2});
/// const auto result = dm::nn::soft_circle_raster(xy, xy, {1.}, dm::nn::ones({1, 3}), 1.);
/// std::print("{:.6f}\n", result.rgb.at({0, 0})); // 0.731059
/// @endcode
/// @see sigmoid()
inline SoftCircleRaster soft_circle_raster(const Tensor& coordinates, const Tensor& centers,
                                           const std::vector<double>& radii, const Tensor& colors,
                                           double sharpness) {
  if (!coordinates.defined() || !centers.defined() || !colors.defined() ||
      coordinates.dim() != 2 || coordinates.size(0) == 0 || coordinates.size(1) != 2 ||
      centers.dim() != 2 || centers.size(0) == 0 || centers.size(1) != 2 ||
      colors.shape() != Shape{centers.size(0), 3} ||
      radii.size() != static_cast<std::size_t>(centers.size(0)) ||
      (coordinates.dtype() != DType::Float32 && coordinates.dtype() != DType::Float64) ||
      coordinates.dtype() != centers.dtype() || coordinates.dtype() != colors.dtype() ||
      coordinates.device().backend != compute::Backend::CPU ||
      centers.device().backend != compute::Backend::CPU || colors.device().backend != compute::Backend::CPU ||
      !(sharpness > 0) || !std::isfinite(sharpness))
    throw std::invalid_argument("soft_circle_raster: incompatible sampled scene");
  const auto dtype = coordinates.dtype();
  const auto representable = [dtype](double value) {
    return value > 0 && std::isfinite(value) &&
           (dtype == DType::Float64 || (value >= std::numeric_limits<float>::denorm_min() &&
                                        value <= std::numeric_limits<float>::max()));
  };
  if (!representable(sharpness)) throw std::invalid_argument("soft_circle_raster: invalid coverage slope");
  std::vector<Tensor> coverage;
  coverage.reserve(radii.size());
  auto rgb = zeros({coordinates.size(0), 3}, dtype);
  auto visible = ones({coordinates.size(0), 1}, dtype);
  for (std::int64_t i = 0; i < centers.size(0); ++i) {
    const auto radius = radii[static_cast<std::size_t>(i)];
    const auto squared = radius * radius;
    if (!(radius > 0) || !representable(squared))
      throw std::invalid_argument("soft_circle_raster: invalid circle radius");
    const auto offset = coordinates - slice_axis(centers, 0, i, i + 1);
    const auto alpha = sigmoid(sharpness * (squared - sum(square(offset), {1}, true)));
    rgb = rgb + visible * (alpha * slice_axis(colors, 0, i, i + 1));
    visible = visible * (1. - alpha);
    coverage.push_back(alpha);
  }
  return {rgb, concat(coverage, 1)};
}

} // namespace datamunge::nn
