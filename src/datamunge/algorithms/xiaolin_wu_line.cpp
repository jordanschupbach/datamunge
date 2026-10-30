/// @file xiaolin_wu_line.cpp
/// @brief Validated two-pixel line filtering, viewport mapping and accumulated-ink plots.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/xiaolin_wu_line.hpp>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Local validation and viewport mapping helpers.
namespace {
/// @brief Reject unknown endpoint policies before processing even an empty frame.
/// @param policy Candidate policy.
/// @throws std::invalid_argument If policy is not a named enumerator.
void check_wu_policy(WuEndpointPolicy policy) {
  if (policy != WuEndpointPolicy::FullColumns && policy != WuEndpointPolicy::ClippedColumns)
    throw std::invalid_argument("Wu: unknown endpoint policy");
}
/// @brief Validate one floating coordinate before rounding it to an int pixel.
/// @param value Coordinate with space for a neighbouring pixel and loop increment.
/// @throws std::invalid_argument If not finite or outside [INT_MIN+2,INT_MAX-2].
void check_wu_coordinate(double value) {
  if (!std::isfinite(value) || value < double(std::numeric_limits<int>::min()) + 2 ||
      value > double(std::numeric_limits<int>::max()) - 2)
    throw std::invalid_argument("Wu: coordinate is non-finite or outside the safe int range");
}
/// @brief Copy finite numeric cells and linearly scale them into the viewport interior.
/// @param data Borrowed frame.
/// @param name Numeric coordinate column.
/// @param extent Pixel dimension, at least four.
/// @return Owned scaled coordinates; empty stays empty and constant maps to centre.
/// @throws std::invalid_argument If column/cells are not finite numeric values.
std::vector<double>
wu_scaled_column(const dstruct::DataFrame& data, const std::string& name, int extent) {
  if (!data.has_column(name))
    throw std::invalid_argument("Wu: missing coordinate column");
  const auto&         column = data.get_column(name);
  std::vector<double> values;
  values.reserve(data.nrows());
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    const double value = column.at(i);
    if (!std::isfinite(value))
      throw std::invalid_argument("Wu: coordinates must be finite numeric cells");
    values.push_back(value);
  }
  if (values.empty())
    return values;
  const auto   limits = std::minmax_element(values.begin(), values.end());
  const double lo = *limits.first, hi = *limits.second;
  const double span = hi - lo;
  // Use direct differences when safe; halve first only for an overflowing opposite-sign span.
  for (auto& value : values) {
    const double fraction = hi == lo ? 0.5
                            : std::isfinite(span)
                                ? (value - lo) / span
                                : (value * 0.5 - lo * 0.5) / (hi * 0.5 - lo * 0.5);
    value                 = 1 + fraction * (extent - 3);
  }
  return values;
}
} // namespace

std::vector<WuPixel> xiaolin_wu_line(double x0, double y0, double x1, double y1) {
  return xiaolin_wu_line(x0, y0, x1, y1, WuEndpointPolicy::FullColumns);
}
std::vector<WuPixel>
xiaolin_wu_line(double x0, double y0, double x1, double y1, WuEndpointPolicy endpoints) {
  check_wu_policy(endpoints);
  for (const double coordinate : {x0, y0, x1, y1})
    check_wu_coordinate(coordinate);
  const bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);
  if (steep) {
    std::swap(x0, y0);
    std::swap(x1, y1);
  }
  if (x0 > x1) {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }
  const double dx = x1 - x0, gradient = dx == 0 ? 0 : (y1 - y0) / dx;
  const int    start = static_cast<int>(std::lround(x0)), end = static_cast<int>(std::lround(x1));
  std::vector<WuPixel> out;
  for (int x = start; x <= end; ++x) {
    const double weight = endpoints == WuEndpointPolicy::FullColumns || dx == 0
                              ? 1.0
                              : std::max(0.0, std::min(x1, x + 0.5) - std::max(x0, x - 0.5));
    if (weight == 0)
      continue;
    const double minor    = y0 + gradient * (double(x) - x0);
    const int    lower    = static_cast<int>(std::floor(minor));
    const double fraction = minor - lower;
    if (steep) {
      out.push_back({lower, x, weight * (1 - fraction)});
      out.push_back({lower + 1, x, weight * fraction});
    } else {
      out.push_back({x, lower, weight * (1 - fraction)});
      out.push_back({x, lower + 1, weight * fraction});
    }
  }
  return out;
}
std::vector<WuPixel> wu_polyline(
    const dstruct::DataFrame& data,
    const std::string&        x_column,
    const std::string&        y_column,
    int                       width,
    int                       height,
    WuEndpointPolicy          endpoints) {
  check_wu_policy(endpoints);
  if (width < 4 || height < 4)
    throw std::invalid_argument("Wu: viewport dimensions must be at least four");
  const auto x = wu_scaled_column(data, x_column, width),
             y = wu_scaled_column(data, y_column, height);
  std::vector<WuPixel> out;
  if (x.size() == 1)
    return xiaolin_wu_line(x[0], y[0], x[0], y[0], endpoints);
  for (std::size_t i = 1; i < x.size(); ++i) {
    const auto segment = xiaolin_wu_line(x[i - 1], y[i - 1], x[i], y[i], endpoints);
    out.insert(out.end(), segment.begin(), segment.end());
  }
  return out;
}
dstruct::DataFrame wu_pixel_table(const std::vector<WuPixel>& pixels) {
  std::map<std::pair<int, int>, double> weights;
  for (const auto& p : pixels) {
    if (!std::isfinite(p.brightness) || p.brightness < 0 || p.brightness > 1)
      throw std::invalid_argument("Wu: contribution brightness must be finite and in [0,1]");
    weights[{p.x, p.y}] += p.brightness;
  }
  std::vector<double> x, y, weight;
  for (const auto& entry : weights) {
    x.push_back(entry.first.first);
    y.push_back(entry.first.second);
    weight.push_back(entry.second);
  }
  dstruct::DataFrame result;
  result.add_column("x", x);
  result.add_column("y", y);
  result.add_column("weight", weight);
  return result;
}
graphics::RPlot wu_pixel_plot(const std::vector<WuPixel>& pixels) {
  if (pixels.empty())
    throw std::invalid_argument("Wu: cannot plot an empty pixel set");
  const auto table = wu_pixel_table(pixels);
  const auto x = table.get_column("x").values(), y = table.get_column("y").values(),
             weight = table.get_column("weight").values();
  auto plot         = graphics::RPlot::create();
  plot.x_limits(
          *std::min_element(x.begin(), x.end()) - 0.5, *std::max_element(x.begin(), x.end()) + 0.5)
      .y_limits(
          *std::min_element(y.begin(), y.end()) - 0.5, *std::max_element(y.begin(), y.end()) + 0.5)
      .show_grid(false)
      .x_label("pixel x")
      .y_label("pixel y");
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto grey = static_cast<unsigned char>(std::lround(255 * (1 - std::min(1.0, weight[i]))));
    plot.polygon(
        {x[i] - 0.5, x[i] + 0.5, x[i] + 0.5, x[i] - 0.5},
        {y[i] - 0.5, y[i] - 0.5, y[i] + 0.5, y[i] + 0.5},
        graphics::RGB{grey, grey, grey},
        true);
  }
  return plot;
}
} // namespace datamunge::algorithms
