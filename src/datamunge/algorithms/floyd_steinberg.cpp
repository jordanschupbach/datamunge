/// @file floyd_steinberg.cpp
/// @brief Shared validated raster quantization, actual updates and signed boundary accounting.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/floyd_steinberg.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private checked shapes, finite sums and shared raster kernel.
namespace {
/// @brief Validate a product before unsigned multiplication.
/// @param a First quantity.
/// @param b Second quantity.
/// @return Representable product.
/// @throws std::length_error Product exceeds size_t.
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("dither: product overflow");
  return a * b;
}
/// @brief Add two entry quantities without unsigned wrap.
/// @param a First quantity.
/// @param b Second quantity.
/// @return Representable sum.
/// @throws std::length_error Sum exceeds size_t.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("dither: entry sum overflow");
  return a + b;
}
/// @brief Accumulate one diagnostic term and reject nonfinite sums.
/// @param sum Mutable long-double accumulator.
/// @param value Term to add.
/// @throws std::overflow_error Result is nonfinite.
void accumulate(long double& sum, long double value) {
  sum += value;
  if (!std::isfinite(sum))
    throw std::overflow_error("dither: nonfinite diagnostic");
}
/// @brief Validate all row widths, then every source value, and prepare owned scalars.
/// @param image Borrowed rectangular finite source.
/// @return Dimensions, checked slot count and original intensity sum.
/// @throws std::invalid_argument Ragged rows or any nonfinite source.
/// @throws std::length_error Cell or slot product exceeds size_t.
/// @throws std::overflow_error Original intensity sum becomes nonfinite.
FloydSteinbergWork prepare(const std::vector<std::vector<double>>& image) {
  FloydSteinbergWork w;
  w.rows = image.size();
  w.cols = image.empty() ? 0 : image[0].size();
  for (const auto& row : image)
    if (row.size() != w.cols)
      throw std::invalid_argument("dither: ragged rows");
  w.pixels         = product(w.rows, w.cols);
  w.neighbor_slots = product(4, w.pixels);
  for (const auto& row : image)
    for (double value : row) {
      if (!std::isfinite(value))
        throw std::invalid_argument("dither: nonfinite input");
    }
  for (const auto& row : image)
    for (double value : row)
      accumulate(w.input_sum, static_cast<long double>(value));
  return w;
}
/// @brief Run the original arithmetic once on an owned scratch grid.
/// @param image Owned finite rectangular scratch, consumed without another input copy.
/// @param w Prepared scalars, updated with actual work.
/// @param trace Optional separate owned history, null for ordinary/work calls.
/// @return Owned integer binary output.
/// @throws std::overflow_error Actual transfer/update or diagnostic becomes nonfinite.
/// @throws std::length_error Output/history container capacity exhausted.
/// @throws std::bad_alloc Output/history allocation fails.
std::vector<std::vector<int>>
run(std::vector<std::vector<double>> image, FloydSteinbergWork& w, FloydSteinbergTrace* trace) {
  std::vector<std::vector<int>> output(w.rows, std::vector<int>(w.cols));
  for (std::size_t r = 0; r < w.rows; ++r)
    for (std::size_t c = 0; c < w.cols; ++c) {
      const double value = image[r][c];
      const int    q     = value < floyd_steinberg_threshold ? 0 : 255;
      const double error = value - q;
      if (!std::isfinite(error))
        throw std::overflow_error("dither: nonfinite quantization error");
      output[r][c] = q;
      if (q)
        ++w.white;
      else
        ++w.black;
      accumulate(w.output_sum, q);
      if (r == 0 && c == 0)
        w.working_min = w.working_max = value;
      else {
        w.working_min = std::min(w.working_min, value);
        w.working_max = std::max(w.working_max, value);
      }
      FloydSteinbergStep step;
      step.row           = r;
      step.col           = c;
      step.value_before  = value;
      step.quantized     = q;
      step.error         = error;
      step.update_begin  = trace ? trace->updates.size() : 0;
      const auto stencil = floyd_steinberg_stencil(w.rows, w.cols, r, c);
      for (const auto& n : stencil.candidates) {
        if (!n.in_bounds) {
          ++w.discarded_slots;
          step.discarded_numerator += n.numerator;
          continue;
        }
        const double before   = image[n.row][n.col],
                     transfer = error * static_cast<double>(n.numerator) / 16.0,
                     after    = before + transfer;
        if (!std::isfinite(transfer) || !std::isfinite(after))
          throw std::overflow_error("dither: nonfinite diffusion update");
        image[n.row][n.col] = after;
        ++w.updates;
        if (trace)
          trace->updates.push_back({n.direction, n.row, n.col, before, transfer, after});
      }
      step.boundary_loss = static_cast<long double>(error) * step.discarded_numerator / 16.L;
      accumulate(w.boundary_loss, step.boundary_loss);
      if (trace) {
        step.update_end = trace->updates.size();
        trace->steps.push_back(step);
      }
    }
  w.balance_residual = w.input_sum - w.output_sum - w.boundary_loss;
  if (!std::isfinite(w.balance_residual))
    throw std::overflow_error("dither: nonfinite balance residual");
  return output;
}
} // namespace
FloydSteinbergStencil floyd_steinberg_stencil(
    std::size_t rows,
    std::size_t cols,
    std::size_t row,
    std::size_t col) noexcept {
  FloydSteinbergStencil s;
  if (row >= rows || col >= cols)
    return s;
  s.count      = 4;
  s.candidates = {
      FloydSteinbergNeighbor{0, 7, false, 0, 0},
      {1, 3, false, 0, 0},
      {2, 5, false, 0, 0},
      {3, 1, false, 0, 0}};
  if (col < cols - 1)
    s.candidates[0] = {0, 7, true, row, col + 1};
  if (row < rows - 1) {
    if (col > 0)
      s.candidates[1] = {1, 3, true, row + 1, col - 1};
    s.candidates[2] = {2, 5, true, row + 1, col};
    if (col < cols - 1)
      s.candidates[3] = {3, 1, true, row + 1, col + 1};
  }
  return s;
}
std::vector<std::vector<int>> floyd_steinberg_dither(std::vector<std::vector<double>> image) {
  auto w = prepare(image);
  return run(std::move(image), w, nullptr);
}
FloydSteinbergResult floyd_steinberg_work(const std::vector<std::vector<double>>& image) {
  FloydSteinbergResult r;
  r.work   = prepare(image);
  r.output = run(image, r.work, nullptr);
  return r;
}
FloydSteinbergTrace floyd_steinberg_trace(
    const std::vector<std::vector<double>>& image,
    std::size_t                             cap) {
  auto       w     = prepare(image);
  const auto bound = add(product(2, w.rows), product(7, w.pixels));
  if ((cap && bound > cap) || w.pixels > std::vector<FloydSteinbergStep>().max_size() ||
      w.neighbor_slots > std::vector<FloydSteinbergUpdate>().max_size())
    throw std::length_error("dither: complete trace entry cap or container limit");
  FloydSteinbergTrace t;
  t.work   = w;
  t.input  = image;
  t.output = run(image, t.work, &t);
  return t;
}
std::vector<std::vector<int>> threshold_dither(const std::vector<std::vector<double>>& image) {
  const auto                    w = prepare(image);
  std::vector<std::vector<int>> out(w.rows, std::vector<int>(w.cols));
  for (std::size_t r = 0; r < w.rows; ++r)
    for (std::size_t c = 0; c < w.cols; ++c)
      out[r][c] = image[r][c] < floyd_steinberg_threshold ? 0 : 255;
  return out;
}
std::vector<std::vector<double>>
dither_gradient(std::size_t rows, std::size_t cols, std::size_t cap) {
  if (!rows || !cols)
    throw std::invalid_argument("dither gradient: dimensions must be positive");
  const auto n = product(rows, cols);
  if (cap && n > cap)
    throw std::length_error("dither gradient: complete pixel cap");
  std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
  if (cols > 1)
    for (auto& row : out)
      for (std::size_t c = 0; c < cols; ++c)
        row[c] = 255.0 * c / (cols - 1);
  return out;
}
DitherRegionStats dither_region_stats(
    const std::vector<std::vector<double>>& input,
    const std::vector<std::vector<int>>&    output,
    std::size_t                             first,
    std::size_t                             last) {
  const auto w = prepare(input);
  if (output.size() != w.rows)
    throw std::invalid_argument("dither stats: output shape mismatch");
  for (const auto& row : output)
    if (row.size() != w.cols)
      throw std::invalid_argument("dither stats: output shape mismatch");
  for (const auto& row : output)
    for (int value : row)
      if (value != 0 && value != 255)
        throw std::invalid_argument("dither stats: nonbinary output");
  if (first > last || last > w.cols)
    throw std::invalid_argument("dither stats: invalid column range");
  DitherRegionStats s;
  long double       original = 0, quantized = 0;
  for (std::size_t r = 0; r < w.rows; ++r)
    for (std::size_t c = first; c < last; ++c) {
      ++s.pixels;
      if (output[r][c])
        ++s.white;
      else
        ++s.black;
      accumulate(original, input[r][c]);
      accumulate(quantized, output[r][c]);
    }
  if (s.pixels) {
    s.input_mean     = original / s.pixels;
    s.output_mean    = quantized / s.pixels;
    s.white_fraction = static_cast<long double>(s.white) / s.pixels;
    s.mean_error     = s.output_mean - s.input_mean;
  }
  return s;
}
} // namespace datamunge::algorithms
