/// @file sobel.cpp
/// @brief One finite checked Sobel producing correlation and actual threshold summaries.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/sobel.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Check complete finite rectangular geometry before pixel reads.
/// @param image Borrowed image.
/// @param limit Zero or complete cell policy.
/// @return Common width.
std::size_t validate(const std::vector<std::vector<double>>& image, std::size_t limit) {
  const auto rows = image.size(), cols = rows ? image.front().size() : 0;
  if (cols && rows > std::numeric_limits<std::size_t>::max() / cols)
    throw std::length_error("sobel: cells capacity");
  if (limit && rows * cols > limit)
    throw std::length_error("sobel: cells policy");
  for (const auto& row : image) {
    if (row.size() != cols)
      throw std::invalid_argument("sobel: rectangular image");
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("sobel: finite intensity");
  }
  return cols;
}
/// @brief Execute actual pixel correlations once.
/// @param image Input.
/// @param o Policies.
/// @param capture Capture actual neighborhood terms.
/// @return Actual responses.
SobelRun run(const std::vector<std::vector<double>>& image, const SobelOptions& o, bool capture) {
  const auto rows = image.size(), cols = validate(image, o.max_pixels),
             interior = rows > 2 && cols > 2 ? (rows - 2) * (cols - 2) : 0;
  const auto limit    = std::numeric_limits<std::size_t>::max();
  if (interior > limit / 19 ||
      (capture && o.max_history_values && interior * 19 > o.max_history_values))
    throw std::length_error("sobel: capture/work geometry");
  SobelRun r;
  r.result.gx.assign(rows, std::vector<double>(cols));
  r.result.gy        = r.result.gx;
  r.result.magnitude = r.result.gx;
  constexpr std::array<int, 9> kx{-1, 0, 1, -2, 0, 2, -1, 0, 1}, ky{-1, -2, -1, 0, 0, 0, 1, 2, 1};
  for (std::size_t row = 1; row + 1 < rows; ++row)
    for (std::size_t col = 1; col + 1 < cols; ++col) {
      SobelPixel p;
      p.row    = row;
      p.column = col;
      for (std::size_t i = 0; i < 9; ++i) {
        const double v        = image[row - 1 + i / 3][col - 1 + i % 3];
        p.horizontal_terms[i] = v * kx[i];
        p.vertical_terms[i]   = v * ky[i];
        p.gx += p.horizontal_terms[i];
        p.gy += p.vertical_terms[i];
        ++r.sample_visits;
      }
      p.magnitude = std::hypot(p.gx, p.gy);
      if (!std::isfinite(p.gx) || !std::isfinite(p.gy) || !std::isfinite(p.magnitude))
        throw std::overflow_error("sobel: response range");
      r.result.gx[row][col]        = p.gx;
      r.result.gy[row][col]        = p.gy;
      r.result.magnitude[row][col] = p.magnitude;
      r.maximum_magnitude          = std::max(r.maximum_magnitude, p.magnitude);
      ++r.interior_pixels;
      if (capture)
        r.history.push_back(p);
    }
  return r;
}
} // namespace
SobelRun sobel_run(const std::vector<std::vector<double>>& image, const SobelOptions& o) {
  return run(image, o, false);
}
SobelRun sobel_trace(const std::vector<std::vector<double>>& image, const SobelOptions& o) {
  return run(image, o, true);
}
SobelResult sobel(const std::vector<std::vector<double>>& image) {
  return sobel_run(image).result;
}
std::vector<std::vector<double>> sobel_square_example() {
  std::vector<std::vector<double>> image(32, std::vector<double>(32));
  for (std::size_t i = 8; i < 24; ++i)
    for (std::size_t j = 8; j < 24; ++j)
      image[i][j] = 255;
  return image;
}
SobelEdges sobel_edges(const SobelResult& r, double fraction) {
  validate(r.magnitude, 1000000);
  if (!std::isfinite(fraction) || fraction < 0 || fraction > 1)
    throw std::invalid_argument("sobel: threshold fraction");
  SobelEdges out;
  for (const auto& row : r.magnitude)
    for (double v : row) {
      if (v < 0)
        throw std::invalid_argument("sobel: negative magnitude");
      out.maximum = std::max(out.maximum, v);
    }
  out.threshold = fraction * out.maximum;
  for (std::size_t i = 0; i < r.magnitude.size(); ++i)
    for (std::size_t j = 0; j < r.magnitude[i].size(); ++j)
      if (r.magnitude[i][j] > out.threshold)
        out.pixels.push_back({i, j});
  return out;
}
} // namespace datamunge::algorithms
