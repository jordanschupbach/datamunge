/// @file chart_layers.cpp
/// @brief Reusable statistical and geometric chart layers for RPlot.

#include <datamunge/plot/plot.hpp>
#include <datamunge/algorithms/marching_squares.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <stdexcept>

namespace datamunge::graphics {
namespace {

/// @brief Validates aligned, nonempty vectors of finite data coordinates.
/// @param rows Vectors with one element per observation.
/// @throws std::invalid_argument If a vector is empty, has a different size, or contains nonfinite data.
void require_coordinates(std::initializer_list<const std::vector<double>*> rows) {
  const auto size = (*rows.begin())->size();
  for (const auto* row : rows) {
    if (!size || row->size() != size ||
        !std::ranges::all_of(*row, [](double value) { return std::isfinite(value); })) {
      throw std::invalid_argument("plot layer: expected matching nonempty finite coordinate vectors");
    }
  }
}

/// @brief Validates ordered coordinates after the finite-coordinate check.
/// @param values Coordinates in drawing order.
/// @param minimum Minimum number of values.
/// @throws std::invalid_argument If too short or not strictly increasing.
void require_increasing(const std::vector<double>& values, std::size_t minimum = 1) {
  if (values.size() < minimum ||
      std::adjacent_find(values.begin(), values.end(), std::greater_equal<>{}) != values.end()) {
    throw std::invalid_argument("plot layer: expected strictly increasing coordinates");
  }
}

/// @brief Validates a finite, positive display size or data width.
/// @param size Width or size to check.
/// @throws std::invalid_argument If size is nonpositive or nonfinite.
void require_positive(double size) {
  if (!(size > 0) || !std::isfinite(size)) {
    throw std::invalid_argument("plot layer: width must be positive and finite");
  }
}

/// @brief Creates a disconnected segment layer with a validated stroke width.
/// @param color Color of every segment.
/// @param stroke_width Stroke width in pixels.
/// @return An empty segment layer.
/// @throws std::invalid_argument If the stroke width is nonpositive or nonfinite.
DataSeries segment_layer(RGBA color, double stroke_width) {
  require_positive(stroke_width);
  DataSeries series;
  series.kind = DataSeries::Kind::Segment;
  series.color = color;
  series.stroke_width = stroke_width;
  return series;
}

/// @brief Appends one segment endpoint pair to a segment layer.
/// @param series Segment layer being built.
/// @param x0 Start x value.
/// @param y0 Start y value.
/// @param x1 End x value.
/// @param y1 End y value.
/// @throws std::invalid_argument If an endpoint is nonfinite.
void append_segment(DataSeries& series, double x0, double y0, double x1, double y1) {
  if (!(std::isfinite(x0) && std::isfinite(y0) && std::isfinite(x1) && std::isfinite(y1))) {
    throw std::invalid_argument("plot layer: segment coordinates overflowed");
  }
  series.x.insert(series.x.end(), {x0, x1});
  series.y.insert(series.y.end(), {y0, y1});
}

/// @brief Verifies the category/series matrix before appending any layers.
/// @param x Category centers.
/// @param values One row of values per series.
/// @param labels Empty, or one label per series.
/// @param width Total width of a group or stack.
/// @throws std::invalid_argument If dimensions, values or width are invalid.
void require_bar_groups(const std::vector<double>& x, const std::vector<std::vector<double>>& values,
                        const std::vector<std::string>& labels, double width) {
  require_positive(width);
  if (values.empty() || (!labels.empty() && labels.size() != values.size())) {
    throw std::invalid_argument("bar groups: expected one label per series or no labels");
  }
  for (const auto& row : values) {
    require_coordinates({&x, &row});
  }
}

/// @brief Picks a series color from the supplied or built-in categorical palette.
/// @param colors Optional palette in series order, recycled if shorter than the series count.
/// @param index Zero-based series index.
/// @return Selected color.
RGBA bar_color(const std::vector<RGBA>& colors, std::size_t index) {
  const std::array<RGBA, 6> defaults{{{37, 99, 235}, {234, 88, 12}, {5, 150, 105},
                                     {124, 58, 237}, {190, 24, 93}, {71, 85, 105}}};
  return colors.empty() ? defaults[index % defaults.size()] : colors[index % colors.size()];
}

} // namespace

RPlot& RPlot::steps(const std::vector<double>& x, const std::vector<double>& y,
                    std::string label, RGBA color, double stroke_width) {
  require_coordinates({&x, &y});
  require_increasing(x);
  require_positive(stroke_width);
  std::vector<double> px{x.front()}, py{y.front()};
  for (std::size_t i = 1; i < x.size(); ++i) {
    px.insert(px.end(), {x[i], x[i]});
    py.insert(py.end(), {y[i - 1], y[i]});
  }
  return line(std::move(px), std::move(py), std::move(label), color, stroke_width);
}

RPlot& RPlot::stems(const std::vector<double>& x, const std::vector<double>& y,
                    double baseline, RGBA color, double stroke_width) {
  require_coordinates({&x, &y});
  auto series = segment_layer(color, stroke_width);
  for (std::size_t i = 0; i < x.size(); ++i) {
    append_segment(series, x[i], baseline, x[i], y[i]);
  }
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::lollipop(const std::vector<double>& x, const std::vector<double>& y,
                       double baseline, RGBA color, double marker_size) {
  require_positive(marker_size);
  return stems(x, y, baseline, color, 1.5).points(x, y, color, marker_size);
}

RPlot& RPlot::area(const std::vector<double>& x, const std::vector<double>& y,
                   double baseline, RGBA color) {
  require_coordinates({&x, &y});
  require_increasing(x, 2);
  if (!std::isfinite(baseline)) {
    throw std::invalid_argument("area: baseline must be finite");
  }
  auto px = x;
  auto py = y;
  px.insert(px.end(), {x.back(), x.front()});
  py.insert(py.end(), {baseline, baseline});
  return polygon(std::move(px), std::move(py), color);
}

RPlot& RPlot::ribbon(const std::vector<double>& x, const std::vector<double>& lower,
                     const std::vector<double>& upper, RGBA color) {
  require_coordinates({&x, &lower, &upper});
  require_increasing(x, 2);
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (lower[i] > upper[i]) throw std::invalid_argument("ribbon: lower exceeds upper");
  }
  auto px = x;
  auto py = upper;
  px.insert(px.end(), x.rbegin(), x.rend());
  py.insert(py.end(), lower.rbegin(), lower.rend());
  return polygon(std::move(px), std::move(py), color);
}

RPlot& RPlot::error_bars(const std::vector<double>& x, const std::vector<double>& lower,
                         const std::vector<double>& upper, double cap_width, RGBA color,
                         double stroke_width) {
  require_coordinates({&x, &lower, &upper});
  if (!(cap_width >= 0) || !std::isfinite(cap_width)) {
    throw std::invalid_argument("error bars: cap width must be finite and nonnegative");
  }
  auto series = segment_layer(color, stroke_width);
  const double half = cap_width / 2;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (lower[i] > upper[i]) throw std::invalid_argument("error bars: lower exceeds upper");
    append_segment(series, x[i], lower[i], x[i], upper[i]);
    if (cap_width > 0) {
      append_segment(series, x[i] - half, lower[i], x[i] + half, lower[i]);
      append_segment(series, x[i] - half, upper[i], x[i] + half, upper[i]);
    }
  }
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::rectangles(const std::vector<double>& left, const std::vector<double>& bottom,
                          const std::vector<double>& right, const std::vector<double>& top,
                          std::string label, RGBA color) {
  require_coordinates({&left, &bottom, &right, &top});
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (left[i] > right[i] || bottom[i] > top[i]) {
      throw std::invalid_argument("rectangles: lower bound exceeds upper bound");
    }
  }
  for (std::size_t i = 0; i < left.size(); ++i) {
    DataSeries series;
    series.kind = DataSeries::Kind::Polygon;
    series.x = {left[i], right[i], right[i], left[i]};
    series.y = {bottom[i], bottom[i], top[i], top[i]};
    if (i == 0) series.label = std::move(label);
    series.color = color;
    series.filled = true;
    add_series(std::move(series));
  }
  return *this;
}

RPlot& RPlot::horizontal_bars(const std::vector<double>& positions, const std::vector<double>& values,
                              std::string label, RGBA color, double bar_height) {
  require_coordinates({&positions, &values});
  require_positive(bar_height);
  std::vector<double> left, right, bottom, top;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    left.push_back(std::min(0.0, values[i]));
    right.push_back(std::max(0.0, values[i]));
    bottom.push_back(positions[i] - bar_height / 2);
    top.push_back(positions[i] + bar_height / 2);
  }
  return rectangles(left, bottom, right, top, std::move(label), color);
}

RPlot& RPlot::grouped_bars(const std::vector<double>& x, const std::vector<std::vector<double>>& values,
                           const std::vector<std::string>& labels, const std::vector<RGBA>& colors,
                           double group_width) {
  require_bar_groups(x, values, labels, group_width);
  const double width = group_width / static_cast<double>(values.size());
  require_positive(width);
  for (const auto center : x) {
    if (!std::isfinite(center - group_width / 2) || !std::isfinite(center + group_width / 2)) {
      throw std::invalid_argument("grouped bars: group bounds overflowed");
    }
  }
  for (std::size_t j = 0; j < values.size(); ++j) {
    auto positions = x;
    const double offset = (static_cast<double>(j) + .5) * width - group_width / 2;
    for (auto& position : positions) position += offset;
    bars(std::move(positions), values[j], labels.empty() ? "" : labels[j], bar_color(colors, j), width);
  }
  return *this;
}

RPlot& RPlot::stacked_bars(const std::vector<double>& x, const std::vector<std::vector<double>>& values,
                           const std::vector<std::string>& labels, const std::vector<RGBA>& colors,
                           double bar_width) {
  require_bar_groups(x, values, labels, bar_width);
  auto left = x;
  auto right = x;
  for (auto& value : left) value -= bar_width / 2;
  for (auto& value : right) value += bar_width / 2;
  std::vector<double> positive(x.size()), negative(x.size());
  std::vector<std::vector<double>> bottom(values.size(), positive), top(values.size(), positive);
  for (std::size_t j = 0; j < values.size(); ++j) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double value = values[j][i];
      double& total = value < 0 ? negative[i] : positive[i];
      const double previous = total;
      total += value;
      bottom[j][i] = std::min(previous, total);
      top[j][i] = std::max(previous, total);
    }
    require_coordinates({&left, &right, &bottom[j], &top[j]});
  }
  for (std::size_t j = 0; j < values.size(); ++j) {
    rectangles(left, bottom[j], right, top[j], labels.empty() ? "" : labels[j], bar_color(colors, j));
  }
  return *this;
}

RPlot& RPlot::waterfall(const std::vector<double>& x, const std::vector<double>& changes,
                         double bar_width, RGBA increase, RGBA decrease) {
  require_coordinates({&x, &changes});
  require_increasing(x);
  require_positive(bar_width);
  std::vector<double> left, right, bottom, top;
  double total = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double next = total + changes[i];
    left.push_back(x[i] - bar_width / 2);
    right.push_back(x[i] + bar_width / 2);
    bottom.push_back(std::min(total, next));
    top.push_back(std::max(total, next));
    total = next;
  }
  require_coordinates({&left, &right, &bottom, &top});
  for (std::size_t i = 0; i < x.size(); ++i) {
    rectangles({left[i]}, {bottom[i]}, {right[i]}, {top[i]}, "", changes[i] < 0 ? decrease : increase);
  }
  return *this;
}

RPlot& RPlot::ecdf(std::vector<double> sample, std::string label, RGBA color, double stroke_width) {
  require_coordinates({&sample});
  require_positive(stroke_width);
  std::sort(sample.begin(), sample.end());
  std::vector<double> x, y;
  double previous = 0;
  for (auto first = sample.cbegin(); first != sample.cend();) {
    const auto last = std::upper_bound(first, sample.cend(), *first);
    const auto cumulative = static_cast<double>(last - sample.cbegin()) / static_cast<double>(sample.size());
    x.insert(x.end(), {*first, *first});
    y.insert(y.end(), {previous, cumulative});
    previous = cumulative;
    first = last;
  }
  return line(std::move(x), std::move(y), std::move(label), color, stroke_width);
}

RPlot& RPlot::rug(const std::vector<double>& x, double bottom, double top, RGBA color) {
  if (bottom == top) throw std::invalid_argument("rug: tick endpoints must differ");
  return stems(x, std::vector<double>(x.size(), top), bottom, color, 1.0);
}

RPlot& RPlot::bubbles(const std::vector<double>& x, const std::vector<double>& y,
                      const std::vector<double>& radii, std::string label, RGBA color) {
  require_coordinates({&x, &y, &radii});
  if (std::ranges::any_of(radii, [](double radius) { return radius < 0; })) {
    throw std::invalid_argument("bubbles: radii must be nonnegative");
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (radii[i] > 0) {
      points({x[i]}, {y[i]}, std::move(label), color, radii[i]);
      label.clear();
    }
  }
  return *this;
}

RPlot& RPlot::arrows(const std::vector<double>& x, const std::vector<double>& y,
                     const std::vector<double>& dx, const std::vector<double>& dy,
                     double head_fraction, RGBA color, double stroke_width) {
  require_coordinates({&x, &y, &dx, &dy});
  if (!(head_fraction >= 0 && head_fraction <= 1)) {
    throw std::invalid_argument("arrows: head fraction must be in [0, 1]");
  }
  auto series = segment_layer(color, stroke_width);
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (dx[i] == 0 && dy[i] == 0) continue;
    const double end_x = x[i] + dx[i];
    const double end_y = y[i] + dy[i];
    append_segment(series, x[i], y[i], end_x, end_y);
    if (head_fraction == 0) continue;
    const double back_x = end_x - dx[i] * head_fraction;
    const double back_y = end_y - dy[i] * head_fraction;
    const double side_x = dy[i] * head_fraction / 2;
    const double side_y = dx[i] * head_fraction / 2;
    append_segment(series, end_x, end_y, back_x + side_x, back_y - side_y);
    append_segment(series, end_x, end_y, back_x - side_x, back_y + side_y);
  }
  if (!series.x.empty()) add_series(std::move(series));
  return *this;
}

RPlot& RPlot::contour(const std::vector<double>& x, const std::vector<double>& y,
                      const std::vector<std::vector<double>>& z, double level,
                      RGBA color, double stroke_width) {
  require_coordinates({&x});
  require_coordinates({&y});
  require_increasing(x, 2);
  require_increasing(y, 2);
  if (z.size() != y.size() || std::ranges::any_of(z, [&](const auto& row) { return row.size() != x.size(); })) {
    throw std::invalid_argument("contour: grid must have y rows of x samples");
  }
  auto series = segment_layer(color, stroke_width);
  const auto map = [](const auto& axis, double index) {
    const auto lo = std::min(static_cast<std::size_t>(index), axis.size() - 2);
    return std::lerp(axis[lo], axis[lo + 1], index - static_cast<double>(lo));
  };
  for (const auto& edge : algorithms::marching_squares(z, level)) {
    append_segment(series, map(x, edge[0]), map(y, edge[1]), map(x, edge[2]), map(y, edge[3]));
  }
  if (!series.x.empty()) add_series(std::move(series));
  return *this;
}

} // namespace datamunge::graphics
