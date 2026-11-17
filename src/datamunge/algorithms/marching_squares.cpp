/// @file marching_squares.cpp
/// @brief Shared checked crossed-edge contour producer, explicit saddle pairing and actual radial observations.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/marching_squares.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete geometry, interpolation and actual-record helpers.
namespace {
/// @brief Check a logical product before allocation/counting.
/// @param a First count.
/// @param b Second count.
/// @return Exact representable product.
std::size_t product(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("Marching squares: count overflow");
  return a * b;
}
/// @brief Admit actual logical input/record slots before copying them.
/// @param trace Optional owned actual history.
/// @param count Actual new logical slots.
/// @param quota Optional capture cap.
void admit(MarchingSquaresTrace* trace, std::size_t count, std::size_t quota) {
  if (!trace)
    return;
  if (count > std::numeric_limits<std::size_t>::max() - trace->captured_entries)
    throw std::length_error("Marching squares: capture overflow");
  trace->captured_entries += count;
  if (quota && trace->captured_entries > quota)
    throw std::length_error("Marching squares: capture quota");
}
/// @brief Validate complete grid/level/rule geometry, including rows that would otherwise be skipped.
/// @param grid Complete original samples.
/// @param iso Requested finite level.
/// @param rule Explicit saddle pairing rule.
/// @return Checked row/column counts.
std::array<std::size_t, 2> validate(
    const std::vector<std::vector<double>>& grid,
    double                                  iso,
    MarchingSaddleConnectivity              rule) {
  if (!std::isfinite(iso) || (rule != MarchingSaddleConnectivity::AboveDisconnected &&
                              rule != MarchingSaddleConnectivity::AboveConnected))
    throw std::invalid_argument("Marching squares: invalid level or saddle rule");
  const std::size_t rows = grid.size(), cols = rows ? grid.front().size() : 0;
  for (const auto& row : grid) {
    if (row.size() != cols)
      throw std::invalid_argument("Marching squares: ragged grid");
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument("Marching squares: nonfinite sample");
  }
  if ((rows && rows - 1 > 9007199254740992ULL) || (cols && cols - 1 > 9007199254740992ULL))
    throw std::length_error(
        "Marching squares: grid index exceeds exact double integer coordinates");
  product(rows, cols);
  const auto cells = product(rows ? rows - 1 : 0, cols ? cols - 1 : 0);
  product(cells, 4);
  if (product(cells, 2) > std::vector<ContourSegment>().max_size())
    throw std::length_error("Marching squares: segment geometry");
  return {rows, cols};
}
/// @brief Evaluate the same finite ordinary edge fraction, with common positive scaling on overflow.
/// @param fa First finite crossing-edge sample.
/// @param fb Second finite crossing-edge sample with a different threshold classification.
/// @param iso Finite between-samples threshold.
/// @param scaled Output flag for nonfinite raw-difference fallback.
/// @return Finite interpolation fraction in[0,1].
double fraction(double fa, double fb, double iso, bool& scaled) {
  const double numerator = iso - fa, denominator = fb - fa;
  double       t = numerator / denominator;
  scaled         = false;
  if (!std::isfinite(numerator) || !std::isfinite(denominator) || !std::isfinite(t)) {
    scaled             = true;
    const double scale = std::max({std::abs(fa), std::abs(fb), std::abs(iso)});
    t                  = (iso / scale - fa / scale) / (fb / scale - fa / scale);
  }
  if (!std::isfinite(t))
    throw std::overflow_error("Marching squares: nonfinite crossing fraction");
  return std::clamp(t, 0., 1.);
}
/// @brief Classify and emit actual contour cells once, with optional exact producing observations.
/// @param grid Complete original samples.
/// @param iso Finite requested level.
/// @param rule Explicit saddle connectivity.
/// @param trace Optional actual owned history.
/// @param quota Optional actual logical capture cap.
/// @return Actual row-major segments and selected same-run work.
MarchingSquaresAnalysis produce(
    const std::vector<std::vector<double>>& grid,
    double                                  iso,
    MarchingSaddleConnectivity              rule,
    MarchingSquaresTrace*                   trace,
    std::size_t                             quota) {
  const auto shape = validate(grid, iso, rule);
  const auto rows = shape[0], cols = shape[1];
  if (trace) {
    admit(trace, product(rows, cols), quota);
    trace->grid         = grid;
    trace->iso          = iso;
    trace->connectivity = rule;
  }
  MarchingSquaresAnalysis out;
  for (std::size_t r = 0; r + 1 < rows; ++r)
    for (std::size_t c = 0; c + 1 < cols; ++c) {
      MarchingCell cell{
          r,
          c,
          {grid[r][c], grid[r][c + 1], grid[r + 1][c + 1], grid[r + 1][c]},
          0,
          {},
          {},
          out.segments.size(),
          out.segments.size()};
      std::array<bool, 4> above{};
      for (unsigned k = 0; k < 4; ++k) {
        above[k] = cell.values[k] >= iso;
        if (above[k])
          cell.case_index |= 1u << k;
        ++out.work.classifications;
      }
      ++out.work.cells;
      const std::array<std::array<unsigned, 2>, 4> ends{{{0, 1}, {1, 2}, {3, 2}, {0, 3}}};
      const double                                 x = double(c), y = double(r);
      const std::array<std::array<double, 2>, 4>   corners{
            {{x, y}, {x + 1, y}, {x + 1, y + 1}, {x, y + 1}}};
      for (unsigned edge = 0; edge < 4; ++edge) {
        auto a = ends[edge][0], b = ends[edge][1];
        cell.crossed[edge] = above[a] != above[b];
        if (!cell.crossed[edge])
          continue;
        bool         scaled;
        const double t    = fraction(cell.values[a], cell.values[b], iso, scaled);
        cell.points[edge] = {
            corners[a][0] + t * (corners[b][0] - corners[a][0]),
            corners[a][1] + t * (corners[b][1] - corners[a][1])};
        ++out.work.interpolations;
        if (trace) {
          admit(trace, 1, quota);
          trace->crossings.push_back(
              {r, c, edge, cell.values[a], cell.values[b], t, cell.points[edge], scaled});
        }
      }
      auto segment = [&](unsigned a, unsigned b) {
        if (!cell.crossed[a] || !cell.crossed[b])
          throw std::logic_error("Marching squares: case selected noncrossing edge");
        admit(trace, 1, quota);
        out.segments.push_back(
            {cell.points[a][0], cell.points[a][1], cell.points[b][0], cell.points[b][1]});
        ++out.work.segments;
      };
      switch (cell.case_index) {
        case 0:
        case 15:
          break;
        case 1:
        case 14:
          segment(3, 0);
          break;
        case 2:
        case 13:
          segment(0, 1);
          break;
        case 3:
        case 12:
          segment(3, 1);
          break;
        case 4:
        case 11:
          segment(1, 2);
          break;
        case 6:
        case 9:
          segment(0, 2);
          break;
        case 7:
        case 8:
          segment(3, 2);
          break;
        case 5:
        case 10: {
          ++out.work.saddle_cells;
          const bool first      = cell.case_index == 5;
          const bool disconnect = rule == MarchingSaddleConnectivity::AboveDisconnected;
          if (first == disconnect) {
            segment(3, 0);
            segment(1, 2);
          } else {
            segment(0, 1);
            segment(3, 2);
          }
          break;
        }
        default:
          throw std::logic_error("Marching squares: invalid case mask");
      }
      cell.segment_end = out.segments.size();
      if (trace) {
        admit(trace, 1, quota);
        trace->cells.push_back(cell);
      }
    }
  return out;
}
} // namespace
std::vector<ContourSegment> marching_squares(
    const std::vector<std::vector<double>>& grid,
    double                                  iso) {
  return produce(grid, iso, MarchingSaddleConnectivity::AboveDisconnected, nullptr, 0).segments;
}
MarchingSquaresAnalysis marching_squares_analyze(
    const std::vector<std::vector<double>>& grid,
    double                                  iso,
    MarchingSaddleConnectivity              connectivity) {
  return produce(grid, iso, connectivity, nullptr, 0);
}
MarchingSquaresTrace marching_squares_trace(
    const std::vector<std::vector<double>>& grid,
    double                                  iso,
    MarchingSaddleConnectivity              connectivity,
    std::size_t                             max_entries) {
  MarchingSquaresTrace t;
  t.analysis = produce(grid, iso, connectivity, &t, max_entries);
  return t;
}
MarchingRadialStudy marching_radial_study(std::size_t size, double radius) {
  if (!std::isfinite(radius) || radius < 0)
    throw std::invalid_argument("Marching squares: invalid radial level");
  product(size, size);
  if (size > std::vector<double>().max_size() ||
      size > std::vector<std::vector<double>>().max_size())
    throw std::length_error("Marching squares: radial allocation geometry");
  MarchingRadialStudy out;
  out.grid.assign(size, std::vector<double>(size));
  const double center = (double(size) - 1) * .5;
  for (std::size_t r = 0; r < size; ++r)
    for (std::size_t c = 0; c < size; ++c) {
      const double dx = double(c) - center, dy = double(r) - center;
      out.grid[r][c] = std::sqrt(dx * dx + dy * dy);
    }
  out.contour = marching_squares_analyze(out.grid, radius);
  for (const auto& s : out.contour.segments) {
    const double dx = .5 * (s[0] + s[2]) - center, dy = .5 * (s[1] + s[3]) - center;
    const double error = std::abs(std::sqrt(dx * dx + dy * dy) - radius);
    out.midpoint_errors.push_back(error);
    out.mean_midpoint_error += error;
    out.maximum_midpoint_error = std::max(out.maximum_midpoint_error, error);
  }
  if (!out.midpoint_errors.empty())
    out.mean_midpoint_error /= out.midpoint_errors.size();
  return out;
}
} // namespace datamunge::algorithms
