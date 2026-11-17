#pragma once
/// @file ordered_dither.hpp
/// @brief Checked4x4 Bayer thresholds, actual pixel decisions and native binary-raster plotting.
#include <cstddef>
#include <datamunge/plot/plot.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual pixel counts and tone sums, excluding validation/plotting/allocation cost.
/// @par Example
/// @code
/// dm::algorithms::OrderedDitherWork w;std::print("{}\n",w.pixels); // 0
/// @endcode
/// @see ordered_dither_trace()
struct OrderedDitherWork {
  std::size_t rows{0};       ///< Validated original row count.
  std::size_t cols{0};       ///< Validated common row width,0 for no rows.
  std::size_t pixels{0};     ///< Complete processed pixel count, rows*cols.
  std::size_t white{0};      ///< Actual255 output count.
  std::size_t black{0};      ///< Actual0 output count.
  long double input_sum{0};  ///< Actual long-double original intensity sum, checked finite.
  long double output_sum{0}; ///< Actual long-double output sum, checked finite.
};
/// @brief Owned binary-valued int grid and actual selected work; not a packed bitmap.
/// @par Example
/// @code
/// auto r=dm::algorithms::ordered_dither_work({{0.,255.}});std::print("{}\n",r.work.white); // 1
/// @endcode
/// @see ordered_dither_trace()
struct OrderedDitherResult {
  std::vector<std::vector<int>> output; ///< Owned row-major0/255 values, including zero-width rows.
  OrderedDitherWork             work;   ///< Actual selected same-producer counters and tone sums.
};
/// @brief Actual row-major pixel, Bayer rank, comparison threshold and committed output.
/// @par Example
/// @code
/// auto t=dm::algorithms::ordered_dither_trace({{8.}});std::print("{}\n",t.steps[0].output); // 255
/// @endcode
/// @see ordered_dither_trace()
struct OrderedDitherPixel {
  std::size_t row{0};       ///< Actual original zero-based row.
  std::size_t col{0};       ///< Actual original zero-based column.
  int         rank{0};      ///< Actual tiled Bayer rank0..15.
  double      input{0};     ///< Actual original finite intensity, unclamped.
  double      threshold{0}; ///< Actual(rank+.5)/16*255 midpoint threshold.
  int         output{0};    ///< Actual strict input>threshold?255:0 decision; ties black.
};
/// @brief Owned complete input, actual pixel records and actual output/work.
/// @par Example
/// @code
/// auto t=dm::algorithms::ordered_dither_trace({{8.}});std::print("{}\n",t.captured_entries); // 5
/// @endcode
/// @see ordered_dither_trace()
struct OrderedDitherTrace {
  std::vector<std::vector<double>> input;  ///< Owned row-major original finite intensity grid.
  OrderedDitherResult              result; ///< Actual same-producer binary output and work.
  std::vector<OrderedDitherPixel>
              steps; ///< All actual pixel decisions in original row-major order.
  std::size_t captured_entries{
      0}; ///< Complete logical units2+3*pixels: dimensions2,inputN,outputN,recordsN.
};
/// @brief Read the fixed native4x4 Bayer rank at arbitrary nonnegative tiled coordinates.
/// Modulo4 indexing accepts SIZE_MAX coordinates without multiplication.
/// Fixed matrix rows are[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5].
/// Threshold formula(rank+.5)/16*255 preserves the original native convention.
/// O(1), no allocation, mutation, retained reference or shared mutable state.
/// @param row Original nonnegative image row.
/// @param col Original nonnegative image column.
/// @return Integer rank0..15.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bayer4_rank(1,0)); // 12
/// @endcode
/// @par References
/// - ImageMagick upstream ordered4x4 dispersed rank pattern, https://raw.githubusercontent.com/ImageMagick/ImageMagick/main/config/thresholds.xml. This API retains native midpoint(rank+.5)/16 thresholds; upstream uses(rank+1)/17, so outputs are not asserted identical.
/// @see ordered_dither_trace()
int bayer4_rank(std::size_t row, std::size_t col) noexcept;
/// @brief Read the exact dyadic midpoint threshold for the tiled native Bayer rank.
/// Modulo4 indexing accepts SIZE_MAX coordinates without multiplication.
/// Fixed matrix rows are[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5].
/// Threshold formula(rank+.5)/16*255 preserves the original native convention.
/// O(1), no allocation, mutation, retained reference or shared mutable state.
/// @param row Original nonnegative image row.
/// @param col Original nonnegative image column.
/// @return Finite midpoint intensity threshold in(0,255).
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bayer4_threshold(0,0)); // 7.96875
/// @endcode
/// @par References
/// - ImageMagick upstream ordered4x4 dispersed rank pattern, https://raw.githubusercontent.com/ImageMagick/ImageMagick/main/config/thresholds.xml. This API retains native midpoint(rank+.5)/16 thresholds; upstream uses(rank+1)/17, so outputs are not asserted identical.
/// @see ordered_dither_trace()
double bayer4_threshold(std::size_t row, std::size_t col) noexcept;
/// @brief Apply the canonical checked4x4 Bayer ordered threshold.
/// Complete rectangular shape, representable cell count and finite values are
/// checked before allocation/quantization. Empty and zero-width rows are valid.
/// Finite values outside nominal0..255 are accepted without clamping. Each cell
/// uses output255 iff input>(rank+.5)/16*255; exact threshold ties produce0.
/// No diffusion or neighbor dependency; tiled coordinates use original row/column.
/// Output is an int grid containing0/255, not a bit-packed image. No input mutation,
/// retained references, callbacks or shared mutable state. O(rows+pixels) work
/// including shape validation, O(rows+pixels) output and constant driver state.
/// Tone sums use checked long double; no general exact-tone or perceptual guarantee.
/// @param image Complete row-major finite rectangular grayscale image.
/// @return Owned actual std::vector<std::vector<int>>.
/// @throws std::invalid_argument If complete shape/finite input validation fails.
/// @throws std::length_error If complete geometry or selected trace quota fails.
/// @throws std::overflow_error If a wide diagnostic tone sum becomes nonfinite.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::ordered_dither({{0.,255.}})[0]); // [0, 255]
/// @endcode
/// @par References
/// - ImageMagick upstream ordered4x4 dispersed rank pattern, https://raw.githubusercontent.com/ImageMagick/ImageMagick/main/config/thresholds.xml. This API retains native midpoint(rank+.5)/16 thresholds; upstream uses(rank+1)/17, so outputs are not asserted identical.
/// @see bayer4_threshold(), ordered_dither_work(), ordered_dither_trace()
std::vector<std::vector<int>> ordered_dither(const std::vector<std::vector<double>>& image);
/// @brief Apply the canonical checked4x4 Bayer ordered threshold.
/// Complete rectangular shape, representable cell count and finite values are
/// checked before allocation/quantization. Empty and zero-width rows are valid.
/// Finite values outside nominal0..255 are accepted without clamping. Each cell
/// uses output255 iff input>(rank+.5)/16*255; exact threshold ties produce0.
/// No diffusion or neighbor dependency; tiled coordinates use original row/column.
/// Output is an int grid containing0/255, not a bit-packed image. No input mutation,
/// retained references, callbacks or shared mutable state. O(rows+pixels) work
/// including shape validation, O(rows+pixels) output and constant driver state.
/// Tone sums use checked long double; no general exact-tone or perceptual guarantee.
/// @param image Complete row-major finite rectangular grayscale image.
/// @return Owned actual OrderedDitherResult.
/// @throws std::invalid_argument If complete shape/finite input validation fails.
/// @throws std::length_error If complete geometry or selected trace quota fails.
/// @throws std::overflow_error If a wide diagnostic tone sum becomes nonfinite.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::ordered_dither_work({{0.,255.}}).work.pixels); // 2
/// @endcode
/// @par References
/// - ImageMagick upstream ordered4x4 dispersed rank pattern, https://raw.githubusercontent.com/ImageMagick/ImageMagick/main/config/thresholds.xml. This API retains native midpoint(rank+.5)/16 thresholds; upstream uses(rank+1)/17, so outputs are not asserted identical.
/// @see bayer4_threshold(), ordered_dither_work(), ordered_dither_trace()
OrderedDitherResult ordered_dither_work(const std::vector<std::vector<double>>& image);
/// @brief Capture every actual canonical ordered-threshold pixel decision.
/// Complete rectangular shape, representable cell count and finite values are
/// checked before allocation/quantization. Empty and zero-width rows are valid.
/// Finite values outside nominal0..255 are accepted without clamping. Each cell
/// uses output255 iff input>(rank+.5)/16*255; exact threshold ties produce0.
/// No diffusion or neighbor dependency; tiled coordinates use original row/column.
/// Output is an int grid containing0/255, not a bit-packed image. No input mutation,
/// retained references, callbacks or shared mutable state. O(rows+pixels) work
/// including shape validation, O(rows+pixels) output and constant driver state.
/// Tone sums use checked long double; no general exact-tone or perceptual guarantee.
/// Complete logical units2+3N are pre-admitted before input/output/history
/// allocation: dimensions2,original pixelsN,output pixelsN,fixed recordsN.
/// Default1000000, zero disables; no partial trace escapes on failure.
/// @param image Complete row-major finite rectangular grayscale image.
/// @param max_entries Complete logical history quota, default1000000,0 disables.
/// @return Owned actual OrderedDitherTrace.
/// @throws std::invalid_argument If complete shape/finite input validation fails.
/// @throws std::length_error If complete geometry or selected trace quota fails.
/// @throws std::overflow_error If a wide diagnostic tone sum becomes nonfinite.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::ordered_dither_trace({{7.96875}}).steps[0].output); // 0
/// @endcode
/// @par References
/// - ImageMagick upstream ordered4x4 dispersed rank pattern, https://raw.githubusercontent.com/ImageMagick/ImageMagick/main/config/thresholds.xml. This API retains native midpoint(rank+.5)/16 thresholds; upstream uses(rank+1)/17, so outputs are not asserted identical.
/// @see bayer4_threshold(), ordered_dither_work(), ordered_dither_trace()
OrderedDitherTrace ordered_dither_trace(
    const std::vector<std::vector<double>>& image,
    std::size_t                             max_entries = 1000000);
/// @brief Build a native exact-cell black/white plot of a checked binary-valued int grid.
/// Cells are centered at(column,-row), preserving original top-to-bottom row order.
/// Uses fixed black0/white255 palette limits, so a uniform raster retains its tone.
/// Complete rectangular binary shape and cell quota are checked first. Empty or
/// zero-width grids return an empty labeled plot. O(rows+pixels) work/storage;
/// plot owns copied layers, without input aliases, and does not write any file.
/// @param image Complete row-major grid containing only0or255.
/// @param max_pixels Complete output-cell cap, default1000000,0 disables.
/// @return Owned native RPlot, editable by caller and saveable as a local figure.
/// @throws std::invalid_argument If shape or binary value checks fail.
/// @throws std::length_error If complete count/coordinate geometry or quota fails.
/// @throws std::bad_alloc If owned plotting storage cannot be allocated.
/// @par Example
/// @code
/// auto p=dm::algorithms::ordered_dither_image_plot({{0,255}});std::print("{}\n",p.series().size()); // 2
/// @endcode
/// @see ordered_dither(), bayer4_rank()
graphics::RPlot ordered_dither_image_plot(
    const std::vector<std::vector<int>>& image,
    std::size_t                          max_pixels = 1000000);
} // namespace datamunge::algorithms
