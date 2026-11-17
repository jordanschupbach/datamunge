#pragma once
/// @file histogram_equalization.hpp
/// @brief Checked rectangular clamped-byte equalization with actual CDF/mapping/output histograms.
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned row-major grayscale integer rows, with values clamped by equalization on read.
/// Rectangular shape is required, including all zero-width rows. Empty images are
/// allowed. Integers are not normalized floats or RGB channels; no storage/view borrowing.
/// @par Example
/// @code
/// const dm::algorithms::EqualizationImage x{{100,101},{102,103}};
/// std::print("{}\n",x.size()); // 2
/// @endcode
/// @see histogram_equalize()
using EqualizationImage = std::vector<std::vector<int>>;
/// @brief Complete rectangular row/pixel caps for the owned diagnostic entry point.
/// Each zero disables its policy; row objects and pixels are separate units.
/// Limits are not bytes/peak RAM and exclude fixed256-entry diagnostics/capacities.
/// Shapes and representability are checked before policy caps or output allocation.
/// @par Example
/// @code
/// const dm::algorithms::EqualizationOptions o;
/// std::print("{}\n",o.max_pixels); // 1000000
/// @endcode
/// @see histogram_equalize_trace()
struct EqualizationOptions {
  std::size_t max_rows{1000000};   ///< Complete row-object cap, default1m; zero disables.
  std::size_t max_pixels{1000000}; ///< Complete rows*columns cap, default1m; zero disables.
};
/// @brief Selected actual producing histogram/CDF/map/output visits, not elapsed time.
/// clamped_pixels counts distinct input locations outside0..255, once in the
/// histogram pass, not both clamp reads. Other fields exclude validation,
/// allocation and diagnostic storage/bookkeeping; counts are not total FLOPs.
/// @par Example
/// @code
/// const auto t=dm::algorithms::histogram_equalize_trace({{-1,256}});
/// std::print("{}\n",t.work.clamped_pixels); // 2
/// @endcode
/// @see EqualizationTrace
struct EqualizationWork {
  std::size_t histogram_pixels{0}; ///< Actual input positions counted into histogram.
  std::size_t clamped_pixels{0};   ///< Actual input locations outside0..255.
  std::size_t cdf_bins{0};         ///< Actual cumulative-count bins,256 even for empty input.
  std::size_t mapping_bins{0};     ///< Actual mapping entries computed,256.
  std::size_t output_pixels{0};    ///< Actual locations mapped and counted into output histogram.
};
/// @brief Owned output and actual fixed-size CDF/mapping tables from one producing execution.
/// No replay or per-pixel history. Integer counts are exact size_t values; the
/// map retains the legacy double formula and half-up conversion, not a rational
/// certificate. Constant/empty input uses the identity mapping; the output still
/// clamps any out-of-range input. All arrays and image rows copy independently.
/// @par Example
/// @code
/// const auto t=dm::algorithms::histogram_equalize_trace({{100,100,101,102}});
/// std::print("{} {}\n",t.cdf_min,t.mapping[101]); // 2 128
/// @endcode
/// @see histogram_equalize_trace(), EqualizationWork
struct EqualizationTrace {
  EqualizationImage image; ///< Actual owned mapped/clamped output, same rectangular shape.
  std::array<std::size_t, 256> histogram{}; ///< Actual input counts after clamping to0..255.
  std::array<std::size_t, 256> cdf{};       ///< Actual inclusive cumulative counts.
  std::array<int, 256> mapping{}; ///< Actual per-level lookup table, including unoccupied levels.
  std::array<std::size_t, 256>
                   output_histogram{};     ///< Counts read directly from actual mapped pixels.
  std::size_t      rows{0};                ///< Complete input/output row count.
  std::size_t      columns{0};             ///< Complete rectangular width,zero when no rows.
  std::size_t      pixels{0};              ///< Checked rows*columns.
  std::size_t      cdf_min{0};             ///< First nonzero CDF count,zero for no pixels.
  bool             identity_mapping{true}; ///< Empty or only one occupied clamped level.
  EqualizationWork work;                   ///< Selected actual producing visits.
};
/// @brief Private shape admission and shared actual histogram/mapping/output kernel.
namespace equalization_detail {
/// @brief Check complete dimensions and optional row/pixel policies without allocation.
/// @param rows Complete row count.
/// @param columns Complete rectangular column count.
/// @param options Optional full row/pixel limits.
/// @return Exact rows*columns.
/// @throws std::length_error On pixel overflow or complete row/pixel caps.
inline std::size_t dimensions(std::size_t rows, std::size_t columns, EqualizationOptions options) {
  if (columns && rows > std::numeric_limits<std::size_t>::max() / columns)
    throw std::length_error("equalization pixel count overflow");
  const auto pixels = rows * columns;
  if ((options.max_rows && rows > options.max_rows) ||
      (options.max_pixels && pixels > options.max_pixels))
    throw std::length_error("equalization complete input limit");
  return pixels;
}
/// @brief Validate all row widths and checked geometry before caps or allocation.
/// @param image Borrowed integer rows.
/// @param options Complete row/pixel policies.
/// @return Complete safe pixel count.
/// @throws std::invalid_argument On any ragged row, including a nonempty row after an empty first.
/// @throws std::length_error On pixel overflow or complete row/pixel caps.
inline std::size_t geometry(const EqualizationImage& image, EqualizationOptions options) {
  const auto rows = image.size(), columns = rows ? image.front().size() : 0;
  for (const auto& row : image)
    if (row.size() != columns)
      throw std::invalid_argument("equalization requires rectangular rows");
  return dimensions(rows, columns, options);
}
/// @brief Produce one shared equalization and actual fixed-size diagnostics without replay.
/// @param image Borrowed rectangular integer rows; validated here before caps/allocation.
/// @param options Complete row/pixel policies.
/// @return Owned mapped output, actual tables and selected work.
/// @throws std::invalid_argument On ragged input.
/// @throws std::length_error On geometry/policy/container size limits.
/// @throws std::bad_alloc On output allocation failure.
inline EqualizationTrace execute(const EqualizationImage& image, EqualizationOptions options) {
  EqualizationTrace t;
  t.pixels  = geometry(image, options);
  t.rows    = image.size();
  t.columns = t.rows ? image.front().size() : 0;
  t.image.assign(t.rows, std::vector<int>(t.columns));
  for (const auto& row : image)
    for (int v : row) {
      const int clamped = std::clamp(v, 0, 255);
      ++t.histogram[clamped];
      ++t.work.histogram_pixels;
      t.work.clamped_pixels += clamped != v;
    }
  std::size_t running = 0;
  bool        found   = false;
  for (std::size_t v = 0; v < 256; ++v) {
    running += t.histogram[v];
    t.cdf[v] = running;
    if (!found && running) {
      t.cdf_min = running;
      found     = true;
    }
    ++t.work.cdf_bins;
  }
  t.identity_mapping       = t.pixels == t.cdf_min;
  const double denominator = static_cast<double>(t.pixels - t.cdf_min);
  for (std::size_t v = 0; v < 256; ++v) {
    if (denominator <= 0)
      t.mapping[v] = static_cast<int>(v);
    else {
      double value =
          (static_cast<double>(t.cdf[v]) - static_cast<double>(t.cdf_min)) / denominator * 255.0;
      if (value < 0)
        value = 0;
      if (value > 255)
        value = 255;
      t.mapping[v] = static_cast<int>(value + .5);
    }
    ++t.work.mapping_bins;
  }
  for (std::size_t i = 0; i < t.rows; ++i)
    for (std::size_t j = 0; j < t.columns; ++j) {
      const int mapped = t.mapping[std::clamp(image[i][j], 0, 255)];
      t.image[i][j]    = mapped;
      ++t.output_histogram[mapped];
      ++t.work.output_pixels;
    }
  return t;
}
} // namespace equalization_detail
/// @brief Apply the preserved discrete CDF-min grayscale map after checking rectangular shape.
/// Values are clamped to0..255 before both histogram and mapping reads. For
/// nonconstant data, map(v)=clamp((cdf(v)-cdf_min)/(N-cdf_min)*255), converted
/// by int(value+.5). Arithmetic uses double as in the legacy implementation.
/// Constant images retain their clamped level, empty shapes retain their rows;
/// both use an identity lookup. In exact arithmetic nonconstant occupied endpoints map to0and255; double
/// conversions can lose count distinctions above2^53.
/// Discrete output need not be uniform and occupied levels can merge through
/// rounding; no claim to maximize contrast or reveal new information is made.
/// O(rows+N+256) work/O(rows+N) output plus fixed256-entry tables. No optional
/// policy caps; exact geometry checks precede allocation. No input mutation,
/// retained references or shared mutable state. Color/local/adaptive methods are
/// outside this fixed global grayscale transformation.
/// @param image Borrowed rectangular integer image; empty/zero-width rows allowed.
/// @return Owned mapped image, preserving shape.
/// @throws std::invalid_argument On ragged input before output allocation.
/// @throws std::length_error On geometry/container count limits.
/// @throws std::bad_alloc On output allocation failure.
/// @par Example
/// @code
/// const auto x=dm::algorithms::histogram_equalize({{100,100,101,102}});
/// std::print("{} {} {}\n",x[0][0],x[0][2],x[0][3]); // 0 128 255
/// @endcode
/// @par References
/// - OpenCV official tutorial, Histograms2: Histogram Equalization,
///   https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html .
///   This implementation preserves its own constant/clamping/rounding policy.
/// @see histogram_equalize_trace(), EqualizationTrace
inline EqualizationImage histogram_equalize(const EqualizationImage& image) {
  return equalization_detail::execute(image, {0, 0}).image;
}
/// @brief Capture actual counts, inclusive CDF, lookup mapping and output counts from one pass pair.
/// Same numerical/clamping/identity semantics as histogram_equalize(), with
/// complete row/pixel limits before output allocation. Diagnostics have fixed
///256-bin storage, so no unbounded history/capture cap is needed. Output remains
/// O(rows+N), with no replay or retained input references; copying is deep.
/// @param image Borrowed rectangular integer rows.
/// @param options Complete row/pixel caps, default1m each; zero disables either policy.
/// @return Owned actual diagnostics, mapped output and selected work.
/// @throws std::invalid_argument On ragged shape before caps/allocation.
/// @throws std::length_error On complete geometry/row/pixel/container limits.
/// @throws std::bad_alloc On output allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::histogram_equalize_trace({{7,7}}, {1,2});
/// std::print("{} {}\n",t.identity_mapping,t.image[0][0]); // true 7
/// @endcode
/// @see EqualizationTrace, histogram_equalize()
inline EqualizationTrace histogram_equalize_trace(
    const EqualizationImage& image,
    EqualizationOptions      options = {}) {
  return equalization_detail::execute(image, options);
}
} // namespace datamunge::algorithms
