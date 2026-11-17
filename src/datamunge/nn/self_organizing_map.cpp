/// @file self_organizing_map.cpp
/// @brief Native differentiable tensor competitive objectives and explicit-plan conversions.
#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/self_organizing_map.hpp>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::nn {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Validate finite CPU Float64 tensor values without mutating inputs.
/// @param x Borrowed actual tensor.
void finite(const Tensor& x) {
  if (!x.defined() || x.dtype() != DType::Float64 || x.device().backend != compute::Backend::CPU ||
      !all(is_finite(x)).item<bool>())
    throw std::invalid_argument("tensor SOM: finite Float64 CPU tensors required");
}
/// @brief Check complete signed tensor shape product and independent element policy.
/// @param shape Actual required broadcast geometry.
/// @param cap Zero or enabled limit.
void geometry(const Shape& shape, std::size_t cap) {
  const auto n = shape_numel(shape);
  if (cap && static_cast<std::uint64_t>(n) > cap)
    throw std::length_error("tensor SOM: complete intermediate policy");
}
/// @brief Produce complete actual pairwise squared distances through native tensor primitives.
/// @param weights Borrowed finite codebook[K,D].
/// @param inputs Borrowed observations[N,D].
/// @param cap Complete broadcast policy.
/// @return Actual native distance tensor[N,K].
Tensor distances(const Tensor& weights, const Tensor& inputs, std::size_t cap) {
  finite(weights);
  finite(inputs);
  if (weights.dim() != 2 || inputs.dim() != 2 || weights.size(0) <= 0 || weights.size(1) <= 0 ||
      weights.size(1) != inputs.size(1))
    throw std::invalid_argument("tensor SOM: codebook/input geometry");
  geometry({inputs.size(0), weights.size(0), weights.size(1)}, cap);
  const auto value =
      sum(square(
              inputs.reshape({inputs.size(0), 1, inputs.size(1)}) -
              weights.reshape({1, weights.size(0), weights.size(1)})),
          {2});
  if (!all(is_finite(value)).item<bool>())
    throw std::overflow_error("tensor SOM: squared-distance range");
  return value;
}
/// @brief Copy a checked native rectangular value matrix into an owned Float64 tensor.
/// @param values Validated finite rows.
/// @param dimensions Complete positive column count.
/// @return Owned actual CPU Float64 matrix.
Tensor matrix(const std::vector<std::vector<double>>& values, std::size_t dimensions) {
  std::vector<double> flat;
  flat.reserve(values.size() * dimensions);
  for (const auto& row : values)
    flat.insert(flat.end(), row.begin(), row.end());
  return from_vector(
      flat, {static_cast<std::int64_t>(values.size()), static_cast<std::int64_t>(dimensions)});
}
} // namespace
SOMTensorPlan som_tensor_plan(
    const std::vector<std::vector<double>>& data,
    const algorithms::SOMParameters&        params,
    const algorithms::SOMOptions&           options) {
  auto          original = algorithms::som_training_plan(data, params, options);
  SOMTensorPlan out;
  out.inputs          = matrix(data, original.initial.dim);
  out.initial_weights = matrix(original.initial.weights, original.initial.dim);
  std::vector<double> coordinates;
  coordinates.reserve(original.initial.weights.size() * 2);
  for (std::size_t r = 0; r < params.rows; ++r)
    for (std::size_t c = 0; c < params.cols; ++c) {
      coordinates.push_back(static_cast<double>(r));
      coordinates.push_back(static_cast<double>(c));
    }
  out.grid =
      from_vector(coordinates, {static_cast<std::int64_t>(original.initial.weights.size()), 2});
  out.initialization_indices = std::move(original.initialization_indices);
  out.sample_indices         = std::move(original.sample_indices);
  out.learning_rates         = std::move(original.learning_rates);
  out.radii                  = std::move(original.radii);
  return out;
}
SOMTensorObjective som_neighborhood_objective(
    const Tensor& weights,
    const Tensor& sample,
    const Tensor& grid,
    double        radius,
    std::size_t   cap) {
  finite(weights);
  finite(sample);
  finite(grid);
  if (weights.dim() != 2 || weights.size(0) <= 0 || weights.size(1) <= 0 || sample.dim() != 1 ||
      sample.size(0) != weights.size(1) || grid.dim() != 2 || grid.size(0) != weights.size(0) ||
      grid.size(1) != 2 || grid.requires_grad() || !std::isfinite(radius) || radius <= 0)
    throw std::invalid_argument("tensor SOM: objective geometry/grid/radius");
  geometry({weights.size(0), weights.size(1)}, cap);
  geometry({weights.size(0), 2}, cap);
  SOMTensorObjective out;
  out.squared_distances = sum(square(weights - sample), {1});
  if (!all(is_finite(out.squared_distances)).item<bool>())
    throw std::overflow_error("tensor SOM: objective distance range");
  const auto winning  = argmin(out.squared_distances);
  out.winner          = static_cast<std::size_t>(winning.item<std::int64_t>());
  const auto position = gather(grid, winning, 0);
  const auto scaled   = (grid - position) / radius;
  const auto gaussian = exp(sum(square(scaled), {1}) * (-.5));
  out.influence       = where(greater_equal(gaussian, 1e-6), gaussian, 0.);
  out.objective       = sum(out.influence * out.squared_distances) * .5;
  if (!all(is_finite(out.influence)).item<bool>() || !all(is_finite(out.objective)).item<bool>())
    throw std::overflow_error("tensor SOM: neighborhood/objective range");
  return out;
}
Tensor som_assignments(const Tensor& weights, const Tensor& inputs, std::size_t cap) {
  auto squared = distances(weights, inputs, cap);
  return argmin(squared, 1);
}
Tensor som_tensor_quantization_error(const Tensor& weights, const Tensor& inputs, std::size_t cap) {
  auto squared = distances(weights, inputs, cap);
  if (!inputs.size(0))
    return zeros({}, DType::Float64);
  return mean(sqrt(min(squared, {1})));
}
graphics::RPlot som_lattice_plot(
    const Tensor&           weights,
    const Tensor&           inputs,
    std::size_t             rows,
    std::size_t             cols,
    const std::vector<int>& labels) {
  finite(weights);
  finite(inputs);
  if (!rows || !cols || rows > std::numeric_limits<std::size_t>::max() / cols ||
      weights.dim() != 2 || weights.size(1) != 2 ||
      static_cast<std::uint64_t>(weights.size(0)) != rows * cols || inputs.dim() != 2 ||
      inputs.size(1) != 2 ||
      (!labels.empty() && labels.size() != static_cast<std::size_t>(inputs.size(0))))
    throw std::invalid_argument("tensor SOM plot: geometry/labels");
  geometry({weights.size(0), 2}, 1000000);
  geometry({inputs.size(0), 2}, 1000000);
  for (int label : labels)
    if (label < 0 || label > 31)
      throw std::invalid_argument("tensor SOM plot: label ids0..31");
  auto w = weights.to_vector<double>(), data = inputs.to_vector<double>();
  auto plot = graphics::RPlot::create();
  const std::array<graphics::RGB, 4> colors{
      {{220, 38, 38}, {37, 99, 235}, {22, 163, 74}, {147, 51, 234}}};
  std::set<int> classes;
  if (labels.empty())
    classes.insert(0);
  else
    classes.insert(labels.begin(), labels.end());
  for (int label : classes) {
    std::vector<double> x, y;
    for (std::size_t i = 0; i < static_cast<std::size_t>(inputs.size(0)); ++i)
      if (labels.empty() || labels[i] == label) {
        x.push_back(data[2 * i]);
        y.push_back(data[2 * i + 1]);
      }
    if (!x.empty())
      plot.points(
          x,
          y,
          "generating class " + std::to_string(label),
          colors[static_cast<unsigned>(label) % colors.size()],
          3);
  }
  std::vector<double> wx, wy;
  for (std::size_t k = 0; k < rows * cols; ++k) {
    wx.push_back(w[2 * k]);
    wy.push_back(w[2 * k + 1]);
    const auto r = k / cols, c = k % cols;
    if (c + 1 < cols)
      plot.lines({w[2 * k], w[2 * (k + 1)]}, {w[2 * k + 1], w[2 * (k + 1) + 1]}, "", {51, 65, 85});
    if (r + 1 < rows)
      plot.lines(
          {w[2 * k], w[2 * (k + cols)]}, {w[2 * k + 1], w[2 * (k + cols) + 1]}, "", {51, 65, 85});
  }
  plot.points(wx, wy, "actual codebook units", {15, 23, 42}, 5);
  long double lowx = w[0], highx = w[0], lowy = w[1], highy = w[1];
  for (std::size_t i = 0; i < w.size(); i += 2) {
    lowx  = std::min(lowx, static_cast<long double>(w[i]));
    highx = std::max(highx, static_cast<long double>(w[i]));
    lowy  = std::min(lowy, static_cast<long double>(w[i + 1]));
    highy = std::max(highy, static_cast<long double>(w[i + 1]));
  }
  for (std::size_t i = 0; i < data.size(); i += 2) {
    lowx  = std::min(lowx, static_cast<long double>(data[i]));
    highx = std::max(highx, static_cast<long double>(data[i]));
    lowy  = std::min(lowy, static_cast<long double>(data[i + 1]));
    highy = std::max(highy, static_cast<long double>(data[i + 1]));
  }
  const auto half = .6L * std::max({highx - lowx, highy - lowy, 1.L}), cx = (lowx + highx) / 2,
             cy = (lowy + highy) / 2;
  for (auto value : {cx - half, cx + half, cy - half, cy + half})
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
      throw std::overflow_error("tensor SOM plot: equal-range bounds");
  plot.size(800, 840)
      .margins(80, 40, 72, 88)
      .x_limits(static_cast<double>(cx - half), static_cast<double>(cx + half))
      .y_limits(static_cast<double>(cy - half), static_cast<double>(cy + half))
      .x_label("input coordinate1")
      .y_label("input coordinate2");
  return plot;
}
} // namespace datamunge::nn
