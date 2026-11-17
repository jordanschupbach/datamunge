/// @file ordered_dither.cpp
/// @brief Canonical independent Bayer pixel comparisons with checked actual histories and exact-cell plots.
#include <cmath>
#include <datamunge/algorithms/ordered_dither.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete shape admission and shared actual pixel producer.
namespace {
/// @brief Check complete grid multiplication before output/history allocation.
/// @param rows Complete row count.
/// @param cols Complete common width.
/// @return Exact representable pixel count.
/// @throws std::length_error If rows*cols overflows.
std::size_t cells(std::size_t rows, std::size_t cols) {
  if (cols && rows > std::numeric_limits<std::size_t>::max() / cols)
    throw std::length_error("ordered dither: cell geometry");
  return rows * cols;
}
/// @brief Validate the complete finite rectangular intensity image.
/// @param image Original caller grid.
/// @return Exact validated pixel count.
/// @throws std::invalid_argument If any row width or value is invalid.
/// @throws std::length_error If complete shape product overflows.
std::size_t validate(const std::vector<std::vector<double>>& image) {
  auto cols = image.empty() ? 0 : image[0].size();
  for (auto& row : image) {
    if (row.size() != cols)
      throw std::invalid_argument("ordered dither: rectangular grid required");
    for (double x : row)
      if (!std::isfinite(x))
        throw std::invalid_argument("ordered dither: finite intensities required");
  }
  return cells(image.size(), cols);
}
/// @brief Produce actual independent row-major Bayer comparisons and optional history.
/// @param image Completely validated before working/output allocation.
/// @param out Optional complete owned-history sink.
/// @param cap Complete logical quota,0 disables.
/// @return Actual binary output and selected tone/count work.
/// @throws std::invalid_argument If complete input validation fails.
/// @throws std::length_error If geometry or selected quota fails.
/// @throws std::overflow_error If actual wide tone sums become nonfinite.
OrderedDitherResult
run(const std::vector<std::vector<double>>& image, OrderedDitherTrace* out, std::size_t cap) {
  auto        n    = validate(image);
  std::size_t rows = image.size(), cols = rows ? image[0].size() : 0;
  if (out) {
    if (n > (std::numeric_limits<std::size_t>::max() - 2) / 3)
      throw std::length_error("ordered dither: trace geometry");
    auto count = 2 + 3 * n;
    if (cap && count > cap)
      throw std::length_error("ordered dither: complete capture quota");
    out->captured_entries = count;
    out->input            = image;
    out->steps.reserve(n);
  }
  OrderedDitherResult result;
  result.work.rows   = rows;
  result.work.cols   = cols;
  result.work.pixels = n;
  result.output.assign(rows, std::vector<int>(cols));
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) {
      OrderedDitherPixel s;
      s.row               = i;
      s.col               = j;
      s.rank              = bayer4_rank(i, j);
      s.input             = image[i][j];
      s.threshold         = bayer4_threshold(i, j);
      s.output            = s.input > s.threshold ? 255 : 0;
      result.output[i][j] = s.output;
      if (s.output)
        ++result.work.white;
      else
        ++result.work.black;
      result.work.input_sum += s.input;
      result.work.output_sum += s.output;
      if (!std::isfinite(result.work.input_sum) || !std::isfinite(result.work.output_sum))
        throw std::overflow_error("ordered dither: diagnostic sum");
      if (out)
        out->steps.push_back(s);
    }
  return result;
}
} // namespace
int bayer4_rank(std::size_t row, std::size_t col) noexcept {
  /// @brief Immutable native rank permutation, indexed only after modulo4.
  static constexpr int ranks[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
  return ranks[row % 4][col % 4];
}
double bayer4_threshold(std::size_t row, std::size_t col) noexcept {
  return (bayer4_rank(row, col) + .5) / 16. * 255.;
}
std::vector<std::vector<int>> ordered_dither(const std::vector<std::vector<double>>& image) {
  return run(image, nullptr, 0).output;
}
OrderedDitherResult ordered_dither_work(const std::vector<std::vector<double>>& image) {
  return run(image, nullptr, 0);
}
OrderedDitherTrace ordered_dither_trace(
    const std::vector<std::vector<double>>& image,
    std::size_t                             max_entries) {
  OrderedDitherTrace t;
  t.result = run(image, &t, max_entries);
  return t;
}
graphics::RPlot ordered_dither_image_plot(
    const std::vector<std::vector<int>>& image,
    std::size_t                          max_pixels) {
  auto rows = image.size(), cols = rows ? image[0].size() : 0;
  for (auto& row : image) {
    if (row.size() != cols)
      throw std::invalid_argument("ordered dither plot: rectangular grid required");
    for (int v : row)
      if (v != 0 && v != 255)
        throw std::invalid_argument("ordered dither plot: binary0/255 required");
  }
  auto n = cells(rows, cols);
  if ((max_pixels && n > max_pixels) || rows > 9007199254740991ULL || cols > 9007199254740991ULL)
    throw std::length_error("ordered dither plot: complete cell/coordinate quota");
  auto p = graphics::RPlot::create();
  p.x_label("original column").y_label("negative original row");
  if (!n)
    return p;
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) {
      double         x = double(j), y = -double(i);
      graphics::RGBA color =
          image[i][j] ? graphics::RGBA{255, 255, 255, 255} : graphics::RGBA{0, 0, 0, 255};
      p.polygon({x - .5, x + .5, x + .5, x - .5}, {y - .5, y - .5, y + .5, y + .5}, color, true);
    }
  p.x_limits(-.5, double(cols) - .5).y_limits(-double(rows) + .5, .5);
  return p;
}
} // namespace datamunge::algorithms
