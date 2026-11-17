/// @file median_filter.cpp
/// @brief Canonical median filter API, producing histories and checked native contracts.

#include <algorithm>
#include <datamunge/algorithms/median_filter.hpp>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical median filter production and capture.
namespace {
/// @brief Checked logical geometry addition.
/// @param a First count.
/// @param b Additional count.
/// @return Exact sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("median-filter geometry");
  return a + b;
}
/// @brief Checked logical geometry multiplication.
/// @param a First count.
/// @param b Second count.
/// @return Exact product.
/// @throws std::length_error On size_t overflow.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > SIZE_MAX / a)
    throw std::length_error("median-filter geometry");
  return a * b;
}
/// @brief Validate every row and recover complete scalar count.
/// @param image Input shape.
/// @return Exact rectangular scalar count.
/// @throws std::invalid_argument On ragged shape.
/// @throws std::length_error On nonrepresentable geometry.
std::size_t cells(const std::vector<std::vector<int>>& image) {
  auto cols = image.empty() ? 0 : image.front().size();
  for (auto& row : image)
    if (row.size() != cols)
      throw std::invalid_argument("median-filter ragged image");
  return multiply(image.size(), cols);
}
/// @brief Clamp an index plus a signed radius offset without overflowing an index addition.
/// @param position Valid original index.
/// @param delta Signed offset in [-INT_MAX,INT_MAX].
/// @param length Positive axis length.
/// @return Clamped source index.
std::size_t offset(std::size_t position, std::int64_t delta, std::size_t length) {
  if (delta < 0) {
    auto n = static_cast<std::size_t>(-delta);
    return n > position ? 0 : position - n;
  }
  auto n = static_cast<std::size_t>(delta);
  return n >= length - 1 - position ? length - 1 : position + n;
}
/// @brief Canonical sample collection, actual nth_element selection and optional producing capture.
/// @param image Original image.
/// @param radius Nonnegative radius.
/// @param trace Optional owned capture.
/// @param limit Optional logical cap, zero off.
/// @return Same-run output and actual work.
/// @throws std::invalid_argument On malformed radius/shape.
/// @throws std::length_error On geometry, capture admission or comparison-count overflow.
MedianFilterAnalysis produce(
    const std::vector<std::vector<int>>& image,
    int                                  radius,
    MedianFilterTrace*                   trace,
    std::size_t                          limit) {
  if (radius < 0)
    throw std::invalid_argument("median-filter negative radius");
  auto p    = cells(image);
  auto side = add(multiply(2, static_cast<std::size_t>(radius)), 1), w = multiply(side, side);
  if (w > std::vector<int>().max_size())
    throw std::length_error("median-filter window geometry");
  auto samples = multiply(p, w);
  auto capture = add(multiply(3, p), samples);
  if (trace && limit && capture > limit)
    throw std::length_error("median-filter complete capture quota");
  MedianFilterAnalysis out;
  out.image = image;
  if (trace) {
    trace->input            = image;
    trace->radius           = radius;
    trace->captured_entries = capture;
    trace->samples.reserve(samples);
    trace->pixels.reserve(p);
  }
  if (!p)
    return out;
  std::vector<int> window;
  window.reserve(w);
  const auto rows = image.size(), cols = image.front().size();
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) {
      window.clear();
      auto first = trace ? trace->samples.size() : 0;
      for (std::int64_t di = -std::int64_t(radius); di <= radius; ++di)
        for (std::int64_t dj = -std::int64_t(radius); dj <= radius; ++dj) {
          auto sr = offset(i, di, rows), sc = offset(j, dj, cols);
          int  value = image[sr][sc];
          window.push_back(value);
          ++out.work.samples;
          if (trace)
            trace->samples.push_back({i, j, sr, sc, value});
        }
      /// @brief Count actual standard-library selection comparisons without changing strict integer ordering.
      /// @param a Left compared value.
      /// @param b Right compared value.
      /// @return Whether a is strictly smaller.
      /// @throws std::length_error If the actual comparator count overflows.
      auto less = [&](int a, int b) {
        out.work.comparisons = add(out.work.comparisons, 1);
        return a < b;
      };
      std::nth_element(window.begin(), window.begin() + w / 2, window.end(), less);
      auto value      = window[w / 2];
      out.image[i][j] = value;
      ++out.work.pixels;
      if (trace)
        trace->pixels.push_back({i, j, first, w, value});
    }
  return out;
}
} // namespace
std::vector<std::vector<int>> median_filter(
    const std::vector<std::vector<int>>& image,
    int                                  radius) {
  return produce(image, radius, nullptr, 0).image;
}
MedianFilterAnalysis median_filter_analyze(const std::vector<std::vector<int>>& image, int radius) {
  return produce(image, radius, nullptr, 0);
}
MedianFilterTrace
median_filter_trace(const std::vector<std::vector<int>>& image, int radius, std::size_t limit) {
  MedianFilterTrace t;
  t.analysis = produce(image, radius, &t, limit);
  return t;
}
MedianFilterImageSummary median_filter_compare(
    const std::vector<std::vector<int>>& reference,
    const std::vector<std::vector<int>>& candidate) {
  auto p = cells(reference);
  cells(candidate);
  if (reference.size() != candidate.size() ||
      (!reference.empty() && reference.front().size() != candidate.front().size()))
    throw std::invalid_argument("median-filter comparison shape");
  MedianFilterImageSummary out;
  out.pixels        = p;
  long double total = 0;
  for (std::size_t i = 0; i < reference.size(); ++i)
    for (std::size_t j = 0; j < reference[i].size(); ++j) {
      auto value = candidate[i][j];
      out.changed += value != reference[i][j];
      out.extremes += value == 0 || value == 255;
      auto difference = std::int64_t(value) - reference[i][j];
      auto absolute   = static_cast<std::uint64_t>(difference < 0 ? -difference : difference);
      total += absolute;
      out.maximum_absolute_error = std::max(out.maximum_absolute_error, absolute);
    }
  out.mean_absolute_error = p ? double(total / p) : 0;
  return out;
}
MedianFilterDemo median_filter_demo() {
  MedianFilterDemo out;
  out.clean.assign(40, std::vector<int>(40));
  for (int i = 0; i < 40; ++i)
    for (int j = 0; j < 40; ++j)
      out.clean[i][j] = j < 20 ? 60 : 200;
  out.noisy = out.clean;
  std::mt19937_64                        rng(7);
  std::uniform_real_distribution<double> uniform(0, 1);
  for (int i = 0; i < 40; ++i)
    for (int j = 0; j < 40; ++j)
      if (uniform(rng) < .15) {
        out.noisy[i][j] = uniform(rng) < .5 ? 0 : 255;
        ++out.corruptions;
      }
  out.filtered         = median_filter_analyze(out.noisy, 1);
  out.noisy_summary    = median_filter_compare(out.clean, out.noisy);
  out.filtered_summary = median_filter_compare(out.clean, out.filtered.image);
  return out;
}
} // namespace datamunge::algorithms
