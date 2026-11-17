#pragma once

/// @file median_filter.hpp
/// @brief Canonical median filter API, producing histories and checked native contracts.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Actual work of one square clamped-border median filtering execution.
/// @details Comparator calls count this standard library's nth_element execution;
/// they are not an exact platform-independent complexity formula.
/// @par Example
/// @code
/// auto a=dm::algorithms::median_filter_analyze({{7}},1);
/// std::print("{} {}\n",a.work.pixels,a.work.samples); // 1 9
/// @endcode
/// @see median_filter_analyze
/// @par Reference
/// Order-statistic filtering with an odd square neighbourhood.
struct MedianFilterWork {
  std::size_t pixels{0};      ///< Actual selected output medians.
  std::size_t samples{0};     ///< Actual window appends, including repeated border samples.
  std::size_t comparisons{0}; ///< Actual strict-int comparator calls inside nth_element.
};

/// @brief Same-run median image and actual selected work.
/// @par Example
/// @code
/// auto a=dm::algorithms::median_filter_analyze({{7}},0);
/// std::print("{}\n",a.image[0][0]); // 7
/// @endcode
/// @see median_filter_analyze
/// @par Reference
/// Clamped-border square-window rank filtering.
struct MedianFilterAnalysis {
  std::vector<std::vector<int>> image; ///< Owned result with the original rectangular shape.
  MedianFilterWork              work;  ///< Actual producer operations.
};

/// @brief One real clamped window sample read before selection.
/// @par Example
/// @code
/// auto t=dm::algorithms::median_filter_trace({{7}},1);
/// std::print("{}\n",t.samples[0].value); // 7
/// @endcode
/// @see MedianFilterTrace
/// @par Reference
/// Replicated-edge boundary extension for rank filtering.
struct MedianFilterSample {
  std::size_t output_row{0}; ///< Owner output row.
  std::size_t output_col{0}; ///< Owner output column.
  std::size_t source_row{0}; ///< Actual clamped input row read.
  std::size_t source_col{0}; ///< Actual clamped input column read.
  int         value{0};      ///< Actual value appended to the unsorted workspace.
};

/// @brief One actual completed pixel median and its captured sample span.
/// @par Example
/// @code
/// auto t=dm::algorithms::median_filter_trace({{7}},1);
/// std::print("{} {}\n",t.pixels[0].count,t.pixels[0].median); // 9 7
/// @endcode
/// @see MedianFilterTrace
/// @par Reference
/// Middle order statistic in an odd-cardinality neighbourhood.
struct MedianFilterPixel {
  std::size_t row{0};    ///< Actual output row.
  std::size_t col{0};    ///< Actual output column.
  std::size_t first{0};  ///< First actual sample record for this pixel.
  std::size_t count{0};  ///< Actual sample count, (2r+1)^2.
  int         median{0}; ///< Actual nth_element middle value assigned to output.
};

/// @brief Complete owned input, real sample reads and actual output assignments from one run.
/// @details Input/output scalar cells and sample/pixel records count toward the logical
/// quota; record fields, row headers, capacities, runtime workspace and bytes do not.
/// @par Example
/// @code
/// auto t=dm::algorithms::median_filter_trace({{7}},1);
/// std::print("{}\n",t.captured_entries); // 12
/// @endcode
/// @see median_filter_trace
/// @par Reference
/// Actual producing square-window rank-filter execution.
struct MedianFilterTrace {
  std::vector<std::vector<int>> input;     ///< Owned original rectangular image.
  int                           radius{0}; ///< Owned original nonnegative radius.
  MedianFilterAnalysis          analysis;  ///< Actual output and selected work from the same run.
  std::vector<MedianFilterSample>
      samples;                           ///< Actual row-major pixels, then row/column offset reads.
  std::vector<MedianFilterPixel> pixels; ///< Actual completed output writes in row-major order.
  std::size_t captured_entries{0};       ///< Complete logical entry count, 3P+P*(2r+1)^2.
};

/// @brief Replace every pixel by the median of its replicated-border odd square window.
/// @details Every input int is allowed. Read the original image for every output,
/// so filtering is simultaneous, not an in-place scan. The window has (2r+1)^2
/// samples; nth_element selects its middle value without fully sorting it.
/// Complete radius/shape/window/sample geometry checks precede indexing and allocation.
/// Empty images and rectangular zero-width images retain their shape. Radius zero
/// is the identity. For P pixels and W window cells, window collection is PW and
/// nth_element has average linear work per window; no worst-case PW guarantee is
/// made for every standard library's implementation. Auxiliary workspace is O(W),
/// plus O(P) returned image; this is not a sliding-histogram constant-radius method.
/// @param image Rectangular image of arbitrary int values.
/// @param radius Nonnegative square-window radius, default one.
/// @return Owned median-filtered image of the same shape.
/// @throws std::invalid_argument On negative radius or ragged rows.
/// @throws std::length_error Before work on nonrepresentable window/sample/output geometry.
/// @par Example
/// @code
/// auto out=dm::algorithms::median_filter({{100,100,100},{100,255,100},{100,100,100}});
/// std::print("{}\n",out[1][1]); // 100
/// @endcode
/// @see median_filter_analyze, median_filter_trace
/// @par Reference
/// Tukey, Exploratory Data Analysis (1977), nonlinear median smoothing.
std::vector<std::vector<int>> median_filter(
    const std::vector<std::vector<int>>& image,
    int                                  radius = 1);

/// @brief Return the canonical filter's same-run image and actual operation counts.
/// @param image Rectangular arbitrary-int image.
/// @param radius Nonnegative radius, default one.
/// @return Actual output, selected pixel/sample counts and comparator calls.
/// @throws std::invalid_argument On negative radius or ragged shape.
/// @throws std::length_error On nonrepresentable geometry or operation count.
/// @par Example
/// @code
/// auto a=dm::algorithms::median_filter_analyze({{2,3}},0);
/// std::print("{}\n",a.work.samples); // 2
/// @endcode
/// @see median_filter, median_filter_trace
/// @par Reference
/// Square-window median rank filtering with replicated boundaries.
MedianFilterAnalysis median_filter_analyze(
    const std::vector<std::vector<int>>& image,
    int                                  radius = 1);

/// @brief Capture every actual clamped sample and completed pixel from the canonical producer.
/// @details Admit complete count 3P+P*(2r+1)^2 before copying or allocation;
/// the default limit is one million logical entries. Zero disables that optional cap.
/// No sorting trace is fabricated: samples retain actual pre-selection append order,
/// while each pixel stores the actual selected middle value. Trace storage is O(PW).
/// @param image Rectangular arbitrary-int image.
/// @param radius Nonnegative radius, default one.
/// @param max_entries Complete logical capture cap, default one million; zero off.
/// @return Owned input, actual sample addresses/values, actual writes and same-run result.
/// @throws std::invalid_argument On negative radius or ragged shape before admission.
/// @throws std::length_error If geometry, complete admission or operation counting overflows.
/// @par Example
/// @code
/// auto t=dm::algorithms::median_filter_trace({{7}},1,12);
/// std::print("{}\n",t.samples.size()); // 9
/// @endcode
/// @see MedianFilterTrace, median_filter_analyze
/// @par Reference
/// Captured actual replicated-edge square-window rank filtering.
MedianFilterTrace median_filter_trace(
    const std::vector<std::vector<int>>& image,
    int                                  radius      = 1,
    std::size_t                          max_entries = 1000000);

/// @brief Measured differences of two equally shaped integer images.
/// @details Extremes count exact zero or255 values, not all errors. Absolute errors
/// use widened differences and long-double accumulation; means are measured doubles.
/// @par Example
/// @code
/// auto s=dm::algorithms::median_filter_compare({{60}},{{255}});
/// std::print("{} {}\n",s.changed,s.extremes); // 1 1
/// @endcode
/// @see median_filter_compare
/// @par Reference
/// Pixelwise absolute error and exact disagreement summaries.
struct MedianFilterImageSummary {
  std::size_t pixels{0};   ///< Actual total scalar cells.
  std::size_t changed{0};  ///< Candidate cells unequal to reference.
  std::size_t extremes{0}; ///< Candidate cells equal to zero or255.
  double      mean_absolute_error{
      0}; ///< Mean absolute widened difference; zero placeholder for no cells.
  std::uint64_t maximum_absolute_error{
      0}; ///< Largest widened absolute difference, zero for no cells.
};

/// @brief Measure a candidate image against an equally shaped reference.
/// @param reference Rectangular original integer image.
/// @param candidate Rectangular image with exactly the same row count and width.
/// @return Pixel count, disagreement count, candidate extremes and absolute-error summaries.
/// @throws std::invalid_argument On ragged or mismatched shapes.
/// @throws std::length_error On nonrepresentable complete cell count.
/// @par Example
/// @code
/// auto s=dm::algorithms::median_filter_compare({{60,200}},{{0,200}});
/// std::print("{}\n",s.mean_absolute_error); // 30
/// @endcode
/// @see MedianFilterImageSummary, median_filter_demo
/// @par Reference
/// Mean and maximum absolute pixel differences.
MedianFilterImageSummary median_filter_compare(
    const std::vector<std::vector<int>>& reference,
    const std::vector<std::vector<int>>& candidate);

/// @brief Original seeded edge/noise fixture and its actual native filter summaries.
/// @par Example
/// @code
/// auto d=dm::algorithms::median_filter_demo();
/// std::print("{}\n",d.clean.size()); // 40
/// @endcode
/// @see median_filter_demo
/// @par Reference
/// Original datamunge median_filter_analysis 40-by40, seed7 example.
struct MedianFilterDemo {
  std::vector<std::vector<int>> clean; ///< Native40-square left60/right200 vertical step.
  std::vector<std::vector<int>>
                           noisy; ///< Actual original seed7 salt/pepper draw sequence at rate0.15.
  MedianFilterAnalysis     filtered;         ///< Actual radius-one native output/work.
  MedianFilterImageSummary noisy_summary;    ///< Measured noisy-versus-clean differences.
  MedianFilterImageSummary filtered_summary; ///< Measured filtered-versus-clean differences.
  std::size_t corruptions{0}; ///< Actual accepted noise draws; not a nominal rate times pixels.
};

/// @brief Reproduce the original seeded40-square vertical-edge impulse-noise experiment.
/// @details mt19937_64 seed7; row-major uniform_real_distribution draws, threshold0.15,
/// then a second draw below0.5 chooses zero versus255. The exact standard-library
/// distribution stream is retained for this build, not claimed portable across all vendors.
/// @return Owned clean/noisy images, actual radius-one filter and measured summaries.
/// @par Example
/// @code
/// auto d=dm::algorithms::median_filter_demo();
/// std::print("{}\n",d.clean[20][5]); // 60
/// @endcode
/// @see MedianFilterDemo, median_filter_compare
/// @par Reference
/// Original datamunge median_filter_analysis numerical fixture.
MedianFilterDemo median_filter_demo();

} // namespace datamunge::algorithms
