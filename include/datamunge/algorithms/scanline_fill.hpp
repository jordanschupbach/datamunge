#pragma once
/// @file scanline_fill.hpp
/// @brief Checked even-odd pixel-center rasterization with native crossing histories.
#include <cstddef>
#include <datamunge/algorithms/layout_geometry.hpp>
#include <limits>
#include <stdexcept>
#include <vector>
/// @brief Canonical checked native algorithm producers and owned outputs.
namespace datamunge::algorithms {
/// @brief Integer pixel address whose center is (x+0.5,y+0.5).
/// @par Example
/// @code
/// const dm::algorithms::Pixel value{};
/// std::print("{}\n",value.x); // 0
/// @endcode
struct Pixel {
  int x{}; ///< Owned integer column and row.
  int y{}; ///< Owned integer column and row.

  /// @brief Compare both addresses.
  /// @param o Borrowed address.
  /// @return True when both coordinates agree.
  /// @par Example
  /// @code
  /// const dm::algorithms::Pixel p{1,2};
  /// std::print("{}\n",p==dm::algorithms::Pixel{1,2}); // true
  /// @endcode
  bool operator==(const Pixel& o) const { return x == o.x && y == o.y; }
};
/// @brief Inclusive integer horizontal span on one sampled row.
/// @par Example
/// @code
/// const dm::algorithms::ScanlineSpan value{};
/// std::print("{}\n",value.first); // 0
/// @endcode
struct ScanlineSpan {
  int first{}; ///< Inclusive column endpoints.
  int last{}; ///< Inclusive column endpoints.
};
/// @brief Clipping and independent intrinsic/work/capture limits; zero disables a size cap.
/// @par Example
/// @code
/// const dm::algorithms::ScanlineOptions value{};
/// std::print("{}\n",value.max_pixels); // 1000000
/// @endcode
struct ScanlineOptions {
  int         min_x{std::numeric_limits<int>::min()}; ///< Inclusive clip column lower bound.
  int         max_x{std::numeric_limits<int>::max()}; ///< Inclusive clip column upper bound.
  int         min_y{std::numeric_limits<int>::min()}; ///< Inclusive clip row lower bound.
  int         max_y{std::numeric_limits<int>::max()}; ///< Inclusive clip row upper bound.
  std::size_t max_vertices{100000};                   ///< Complete input vertex cap.
  std::size_t max_scanlines{1000000};                 ///< Complete clipped candidate-row cap.
  std::size_t max_edge_tests{10000000};               ///< Actual edge/row predicate cap.
  std::size_t max_pixels{1000000};                    ///< Intrinsic unique output-pixel cap.
  std::size_t max_trace_entries{1000000}; ///< Row headers, crossing scalars and span records.
};
/// @brief One actual sampled row, including clipped raw and unioned integer spans.
/// @par Example
/// @code
/// const dm::algorithms::ScanlineRow value{};
/// std::print("{}\n",value.crossings.size()); // 0
/// @endcode
struct ScanlineRow {
  int                       y{};       ///< Pixel row.
  long double               center{};  ///< Actual sampled y+0.5.
  std::vector<long double>  crossings; ///< Sorted actual floating intersections.
  std::vector<ScanlineSpan> paired;    ///< Actual nonempty clipped paired spans.
  std::vector<ScanlineSpan> merged;    ///< Actual union, preventing duplicate pixels.
};
/// @brief Owned unique pixels and actual raster work/history.
/// @par Example
/// @code
/// const dm::algorithms::ScanlineRun value{};
/// std::print("{}\n",value.scanlines); // 0
/// @endcode
struct ScanlineRun {
  std::vector<Pixel> pixels; ///< Unique pixels ordered by row then column.
  std::size_t scanlines{}; ///< Actual sampled rows, predicates and crossings.
  std::size_t edge_tests{}; ///< Actual sampled rows, predicates and crossings.
  std::size_t intersections{}; ///< Actual sampled rows, predicates and crossings.
  std::vector<ScanlineRow> rows; ///< Actual optional row history.
};
/// @brief Rasterize finite polygon vertices at pixel centers.
/// Uses even-odd consecutive sorted crossings. Edges include their lower y endpoint and exclude the upper; horizontal edges do not cross. Both paired x endpoints include centers on the boundary, preserving historical integer-polygon output. Candidate rows use ceil(ymin-0.5) through ceil(ymax-0.5) exclusive, correcting fractional-bottom omissions. Overlapping integer spans are unioned. Fewer than three finite vertices returns empty. Self intersections use this stated crossing rule. Floating interpolation is not an exact geometric predicate. Complete clipped row geometry is checked before work; pixel/edge/history budgets are admitted before their corresponding work. Clip before integer casts; all integer-address extremes are supported without increment overflow. O(rows*vertices + sum crossing-sort work + pixels) time. Input borrowed unchanged; outputs own their storage; calls share no state.
/// @param polygon Ordered borrowed finite vertices; last-to-first edge is implicit.
/// @param options Inclusive clip and independent bounds.
/// @return Owned pixels, actual work and optional row records.
/// @throws std::invalid_argument Nonfinite vertex or reversed clip.
/// @throws std::length_error Input, row, edge, pixel or retained-history capacity/policy exceeded.
/// @throws std::logic_error Floating crossings violate even pairing.
/// @par Example
/// @code
/// auto r=dm::algorithms::scanline_fill_run({{0,0},{2,0},{2,2},{0,2}});
/// std::print("{}\n",r.pixels.size()); // 4
/// @endcode
/// @par References
/// - Foley, van Dam, Feiner and Hughes (1990), Computer Graphics: Principles and Practice, second edition, scan conversion background.
/// @see scanline_fill()
ScanlineRun scanline_fill_run(
    const std::vector<Point2D>& polygon,
    const ScanlineOptions&      options = {});
/// @brief Rasterize finite polygon vertices at pixel centers with actual crossing histories.
/// Uses even-odd consecutive sorted crossings. Edges include their lower y endpoint and exclude the upper; horizontal edges do not cross. Both paired x endpoints include centers on the boundary, preserving historical integer-polygon output. Candidate rows use ceil(ymin-0.5) through ceil(ymax-0.5) exclusive, correcting fractional-bottom omissions. Overlapping integer spans are unioned. Fewer than three finite vertices returns empty. Self intersections use this stated crossing rule. Floating interpolation is not an exact geometric predicate. Complete clipped row geometry is checked before work; pixel/edge/history budgets are admitted before their corresponding work. Clip before integer casts; all integer-address extremes are supported without increment overflow. O(rows*vertices + sum crossing-sort work + pixels) time. Input borrowed unchanged; outputs own their storage; calls share no state.
/// @param polygon Ordered borrowed finite vertices; last-to-first edge is implicit.
/// @param options Inclusive clip and independent bounds.
/// @return Owned pixels, actual work and optional row records.
/// @throws std::invalid_argument Nonfinite vertex or reversed clip.
/// @throws std::length_error Input, row, edge, pixel or retained-history capacity/policy exceeded.
/// @throws std::logic_error Floating crossings violate even pairing.
/// @par Example
/// @code
/// auto r=dm::algorithms::scanline_fill_trace({{0,0},{2,0},{2,2},{0,2}});
/// std::print("{}\n",r.pixels.size()); // 4
/// @endcode
/// @par References
/// - Foley, van Dam, Feiner and Hughes (1990), Computer Graphics: Principles and Practice, second edition, scan conversion background.
/// @see scanline_fill()
ScanlineRun scanline_fill_trace(
    const std::vector<Point2D>& polygon,
    const ScanlineOptions&      options = {});
/// @brief Return unique raster pixels through the preserved Point2D overload.
/// Uses the checked native producer with default limits and no retained history.
/// @param poly Ordered finite vertices.
/// @return Owned unique pixels.
/// @throws std::invalid_argument Nonfinite coordinates.
/// @throws std::length_error Default geometry/work/output bounds exceeded.
/// @throws std::logic_error Invalid floating crossing parity.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::scanline_fill({{0,0},{2,0},{2,2},{0,2}}).size()); // 4
/// @endcode
/// @par References
/// - Foley, van Dam, Feiner and Hughes (1990), Computer Graphics: Principles and Practice, second edition, scan conversion background.
/// @see scanline_fill_run()
std::vector<Pixel> scanline_fill(const std::vector<Point2D>& poly);
/// @brief Adapt indexed coordinate pairs to the same native raster producer.
/// @tparam Point Copyable pair supporting finite numeric coordinates at indices zero and one.
/// @param poly Borrowed ordered vertices.
/// @return Owned unique integer pixels.
/// @throws std::invalid_argument Nonfinite converted coordinates.
/// @throws std::length_error Default input/work/output bound exceeded.
/// @par Example
/// @code
/// std::vector<std::array<double,2>> p{{0,0},{2,0},{2,2},{0,2}};
/// std::print("{}\n",dm::algorithms::scanline_fill(p).size()); // 4
/// @endcode
/// @par References
/// - Foley, van Dam, Feiner and Hughes (1990), Computer Graphics: Principles and Practice, second edition, scan conversion background.
/// @see scanline_fill_run()
template<class Point>
std::vector<Pixel> scanline_fill(const std::vector<Point>& poly) {
  if (poly.size() > 100000)
    throw std::length_error("scanline: vertices");
  std::vector<Point2D> p;
  p.reserve(poly.size());
  for (const auto& v : poly)
    p.push_back({static_cast<double>(v[0]), static_cast<double>(v[1])});
  return scanline_fill(p);
}
} // namespace datamunge::algorithms
