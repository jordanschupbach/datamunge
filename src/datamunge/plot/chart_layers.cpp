/// @file chart_layers.cpp
/// @brief Reusable statistical and geometric chart layers for RPlot.

#include <datamunge/plot/plot.hpp>
#include <datamunge/algorithms/marching_squares.hpp>
#include <datamunge/stats/time_series.hpp>
#include <datamunge/vector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <numbers>
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

namespace {

/// @brief Checks aligned quotes and body/tick bounds before any financial marks are added.
/// @param x Strictly increasing horizontal positions.
/// @param open Opening values.
/// @param high High values.
/// @param low Low values.
/// @param close Closing values.
/// @param width Full width in x data units.
/// @throws std::invalid_argument If a quote or its geometry is invalid.
void validate_quotes(const std::vector<double>& x, const std::vector<double>& open,
                     const std::vector<double>& high, const std::vector<double>& low,
                     const std::vector<double>& close, double width) {
  require_coordinates({&x, &open, &high, &low, &close});
  require_increasing(x);
  require_positive(width);
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (low[i] > std::min(open[i], close[i]) || high[i] < std::max(open[i], close[i]) ||
        !std::isfinite(x[i] - width / 2) || !std::isfinite(x[i] + width / 2)) {
      throw std::invalid_argument("financial chart: quote lies outside its low/high or has invalid bounds");
    }
  }
}

} // namespace

RPlot& RPlot::candlesticks(const std::vector<double>& x, const std::vector<double>& open,
                           const std::vector<double>& high, const std::vector<double>& low,
                           const std::vector<double>& close, double width, RGBA rising, RGBA falling) {
  validate_quotes(x, open, high, low, close, width);
  auto up = segment_layer(rising, 1.5);
  auto down = segment_layer(falling, 1.5);
  for (std::size_t i = 0; i < x.size(); ++i) {
    auto& wicks = close[i] < open[i] ? down : up;
    append_segment(wicks, x[i], low[i], x[i], high[i]);
    if (close[i] == open[i]) {
      append_segment(wicks, x[i] - width / 2, open[i], x[i] + width / 2, open[i]);
    }
  }
  if (!up.x.empty()) add_series(std::move(up));
  if (!down.x.empty()) add_series(std::move(down));
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (close[i] == open[i]) continue;
    rectangles({x[i] - width / 2}, {std::min(open[i], close[i])},
               {x[i] + width / 2}, {std::max(open[i], close[i])},
               "", close[i] < open[i] ? falling : rising);
  }
  return *this;
}

RPlot& RPlot::ohlc(const std::vector<double>& x, const std::vector<double>& open,
                   const std::vector<double>& high, const std::vector<double>& low,
                   const std::vector<double>& close, double width, RGBA rising, RGBA falling) {
  validate_quotes(x, open, high, low, close, width);
  auto up = segment_layer(rising, 2.0);
  auto down = segment_layer(falling, 2.0);
  for (std::size_t i = 0; i < x.size(); ++i) {
    auto& strokes = close[i] < open[i] ? down : up;
    append_segment(strokes, x[i], low[i], x[i], high[i]);
    append_segment(strokes, x[i] - width / 2, open[i], x[i], open[i]);
    append_segment(strokes, x[i], close[i], x[i] + width / 2, close[i]);
  }
  if (!up.x.empty()) add_series(std::move(up));
  if (!down.x.empty()) add_series(std::move(down));
  return *this;
}

RPlot& RPlot::dumbbell(const std::vector<double>& positions, const std::vector<double>& start,
                       const std::vector<double>& end, RGBA first, RGBA second) {
  require_coordinates({&positions, &start, &end});
  return segments(start, positions, end, positions, {100, 116, 139}, 2.0)
      .points(start, positions, first).points(end, positions, second);
}

RPlot& RPlot::dot_stack(std::vector<double> sample, RGBA color, double marker_size) {
  require_coordinates({&sample});
  require_positive(marker_size);
  std::sort(sample.begin(), sample.end());
  std::vector<double> height(sample.size(), 1.0);
  for (std::size_t i = 1; i < sample.size(); ++i) {
    if (sample[i] == sample[i - 1]) height[i] = height[i - 1] + 1;
  }
  return points(std::move(sample), std::move(height), color, marker_size);
}

RPlot& RPlot::population_pyramid(const std::vector<double>& positions, const std::vector<double>& left,
                                 const std::vector<double>& right, std::string left_label,
                                 std::string right_label, double bar_height) {
  require_coordinates({&positions, &left, &right});
  if (std::ranges::any_of(left, [](double v) { return v < 0; }) ||
      std::ranges::any_of(right, [](double v) { return v < 0; })) {
    throw std::invalid_argument("population pyramid: counts must be nonnegative");
  }
  auto negative = left;
  for (double& value : negative) value = -value;
  return horizontal_bars(positions, negative, std::move(left_label), {37, 99, 235}, bar_height)
      .horizontal_bars(positions, right, std::move(right_label), {234, 88, 12}, bar_height);
}

RPlot& RPlot::frequency_polygon(const std::vector<double>& sample, std::size_t bins, RGBA color) {
  const auto histogram = hist(sample, bins);
  const auto& bars = histogram.series().front();
  auto x = bars.x;
  auto y = bars.y;
  x.insert(x.begin(), x.front() - bars.bar_width);
  x.push_back(x.back() + bars.bar_width);
  y.insert(y.begin(), 0);
  y.push_back(0);
  require_coordinates({&x, &y});
  return line(std::move(x), std::move(y), "", color);
}

RPlot& RPlot::qqplot(std::vector<double> x, std::vector<double> y, RGBA color) {
  require_coordinates({&x});
  require_coordinates({&y});
  const auto size = std::min(x.size(), y.size());
  if (size < 2) throw std::invalid_argument("qqplot: need at least two values per sample");
  std::sort(x.begin(), x.end());
  std::sort(y.begin(), y.end());
  const auto quantiles = [size](const std::vector<double>& sorted) {
    std::vector<double> result;
    result.reserve(size);
    for (std::size_t i = 0; i + 1 < size; ++i) {
      const double position = (static_cast<double>(i) / (size - 1)) * (sorted.size() - 1);
      const auto j = static_cast<std::size_t>(position);
      result.push_back(std::lerp(sorted[j], sorted[std::min(j + 1, sorted.size() - 1)], position - j));
    }
    result.push_back(sorted.back());
    return result;
  };
  return points(quantiles(x), quantiles(y), color);
}

RPlot& RPlot::bland_altman(const std::vector<double>& first, const std::vector<double>& second, RGBA color) {
  require_coordinates({&first, &second});
  if (first.size() < 2) throw std::invalid_argument("Bland–Altman: need at least two pairs");
  std::vector<double> means, differences;
  for (std::size_t i = 0; i < first.size(); ++i) {
    means.push_back(std::midpoint(first[i], second[i]));
    differences.push_back(first[i] - second[i]);
  }
  require_coordinates({&means, &differences});
  const Vector values(differences);
  const double bias = datamunge::mean(values);
  const double spread = 1.96 * datamunge::sd(values);
  const double lower = bias - spread;
  const double upper = bias + spread;
  if (!std::isfinite(lower) || !std::isfinite(upper)) {
    throw std::invalid_argument("Bland–Altman: agreement limits are not finite");
  }
  const auto [lo, hi] = std::ranges::minmax(differences);
  if (std::min(lo, lower) < std::max(hi, upper)) {
    y_limits(std::min(lo, lower), std::max(hi, upper));
  }
  return points(std::move(means), std::move(differences), color).abline_h(bias)
      .abline_h(lower, {100, 116, 139}).abline_h(upper, {100, 116, 139});
}

RPlot& RPlot::autocorrelation(const std::vector<double>& sample, std::size_t max_lag, bool partial, RGBA color) {
  require_coordinates({&sample});
  const auto values = partial ? stats::pacf(sample, max_lag) : stats::acf(sample, max_lag);
  std::vector<double> lags(values.size());
  std::iota(lags.begin(), lags.end(), partial ? 1.0 : 0.0);
  return stems(lags, values, 0, color);
}

RPlot& RPlot::forest(const std::vector<double>& positions, const std::vector<double>& estimate,
                     const std::vector<double>& lower, const std::vector<double>& upper, RGBA color) {
  require_coordinates({&positions, &estimate, &lower, &upper});
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (estimate[i] < lower[i] || estimate[i] > upper[i]) {
      throw std::invalid_argument("forest: estimate must lie within its interval");
    }
  }
  return segments(lower, positions, upper, positions, color, 2.0).points(estimate, positions, color);
}

RPlot& RPlot::pareto(const std::vector<double>& counts, const std::vector<std::string>& labels, RGBA color) {
  require_coordinates({&counts});
  const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
  if (!(total > 0.0) || !std::isfinite(total) || std::ranges::min(counts) < 0.0 ||
      (!labels.empty() && labels.size() != counts.size())) {
    throw std::invalid_argument("pareto: expected nonnegative weights, positive total and one label per category");
  }
  std::vector<std::size_t> order(counts.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return counts[a] > counts[b]; });
  std::vector<double> x(order.size()), share, cumulative;
  std::vector<std::string> names;
  std::iota(x.begin(), x.end(), 0.0);
  double running = 0.0;
  for (auto index : order) {
    share.push_back(100 * (counts[index] / total));
    running += counts[index];
    cumulative.push_back(100 * (running / total));
    names.push_back(labels.empty() ? std::to_string(index + 1) : labels[index]);
  }
  cumulative.back() = 100.0;
  return bars(x, std::move(share), "", color).line(x, std::move(cumulative), "", {234, 88, 12})
      .x_tick_labels(std::move(names)).y_limits(0, 100).y_label("Percent");
}

RPlot& RPlot::stacked_area(const std::vector<double>& x, const std::vector<std::vector<double>>& values,
                            const std::vector<RGBA>& colors) {
  require_bar_groups(x, values, {}, 1.0);
  require_increasing(x, 2);
  auto boundaries = values;
  std::vector<double> zero(x.size());
  for (std::size_t j = 0; j < values.size(); ++j) {
    const auto& lower = j == 0 ? zero : boundaries[j - 1];
    for (std::size_t i = 0; i < x.size(); ++i) {
      boundaries[j][i] += lower[i];
      if (values[j][i] < 0 || !std::isfinite(boundaries[j][i])) {
        throw std::invalid_argument("stacked area: heights must be nonnegative and cumulative bounds finite");
      }
    }
  }
  for (std::size_t j = 0; j < boundaries.size(); ++j) {
    ribbon(x, j == 0 ? zero : boundaries[j - 1], boundaries[j], bar_color(colors, j));
  }
  return *this;
}

RPlot& RPlot::parallel_coordinates(const std::vector<std::vector<double>>& rows,
                                    const std::vector<std::string>& labels, RGBA color) {
  if (rows.empty() || rows.front().size() < 2 || (!labels.empty() && labels.size() != rows.front().size())) {
    throw std::invalid_argument("parallel coordinates: need two or more axes with matching labels");
  }
  const auto size = rows.front().size();
  for (const auto& row : rows) {
    if (row.size() != size || !std::ranges::all_of(row, [](double value) { return value >= 0 && value <= 1; })) {
      throw std::invalid_argument("parallel coordinates: each row must have one normalized value per axis");
    }
  }
  std::vector<double> axes(size), zero(size), one(size, 1.0);
  std::iota(axes.begin(), axes.end(), 0.0);
  segments(axes, zero, axes, one, {148, 163, 184});
  for (const auto& row : rows) line(axes, row, "", color);
  if (!labels.empty()) x_tick_labels(labels);
  return y_limits(0, 1).show_grid(false);
}

RPlot& RPlot::ternary(const std::vector<std::vector<double>>& values,
                       const std::vector<std::string>& labels, RGBA color) {
  if (values.empty() || (!labels.empty() && labels.size() != 3)) {
    throw std::invalid_argument("ternary: need triples and either three labels or none");
  }
  const double height = std::sqrt(3.0) / 2.0;
  std::vector<double> x, y;
  for (const auto& row : values) {
    require_coordinates({&row});
    const double total = std::accumulate(row.begin(), row.end(), 0.0);
    if (row.size() != 3 || std::ranges::min(row) < 0 || !(total > 0) || !std::isfinite(total)) {
      throw std::invalid_argument("ternary: each row must have three nonnegative values and a positive finite sum");
    }
    x.push_back(row[1] / total + 0.5 * (row[2] / total));
    y.push_back(height * (row[2] / total));
  }
  polygon({0, 1, 0.5}, {0, 0, height}, {71, 85, 105}, false)
      .points(std::move(x), std::move(y), color);
  if (!labels.empty()) {
    text(-.04, -.05, labels[0]).text(1.04, -.05, labels[1]).text(.5, height + .05, labels[2]);
  }
  return *this;
}

RPlot& RPlot::wireframe(const std::vector<double>& x, const std::vector<double>& y,
                         const std::vector<std::vector<double>>& z, RGBA color) {
  require_coordinates({&x});
  require_coordinates({&y});
  require_increasing(x, 2);
  require_increasing(y, 2);
  if (z.size() != y.size()) throw std::invalid_argument("wireframe: expected one grid row per y coordinate");
  for (const auto& row : z) require_coordinates({&x, &row});
  auto edges = segment_layer(color, 1.0);
  for (std::size_t j = 0; j < y.size(); ++j) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double px = x[i] - .5 * y[j];
      const double py = z[j][i] + .5 * y[j];
      if (i > 0) append_segment(edges, x[i - 1] - .5 * y[j], z[j][i - 1] + .5 * y[j], px, py);
      if (j > 0) append_segment(edges, x[i] - .5 * y[j - 1], z[j - 1][i] + .5 * y[j - 1], px, py);
    }
  }
  add_series(std::move(edges));
  return *this;
}

RPlot& RPlot::treemap(const std::vector<double>& weights, const std::vector<std::string>& labels,
                      const std::vector<RGBA>& colors) {
  require_coordinates({&weights});
  if ((!labels.empty() && labels.size() != weights.size()) || std::ranges::min(weights) <= 0) {
    throw std::invalid_argument("treemap: expected positive weights and matching optional labels");
  }
  std::vector<double> prefix{0.0};
  for (const auto weight : weights) {
    const double sum = prefix.back() + weight;
    if (!std::isfinite(sum) || sum <= prefix.back()) {
      throw std::invalid_argument("treemap: weights cannot be resolved at the supplied scale");
    }
    prefix.push_back(sum);
  }
  /// @brief A range of weights and its assigned subrectangle.
  struct Tile {
    std::size_t first, last; ///< Half-open index interval.
    double left, bottom, right, top; ///< Tile bounds.
  };
  std::vector<Tile> pending{{0, weights.size(), 0.0, 0.0, 1.0, 1.0}};
  std::vector<Tile> leaves(weights.size());
  while (!pending.empty()) {
    const auto tile = pending.back();
    pending.pop_back();
    if (tile.last - tile.first == 1) {
      if (!(tile.right > tile.left && tile.top > tile.bottom)) {
        throw std::invalid_argument("treemap: tile is below coordinate resolution");
      }
      leaves[tile.first] = tile;
      continue;
    }
    const auto mid = std::midpoint(prefix[tile.first], prefix[tile.last]);
    const auto begin = prefix.begin() + static_cast<std::ptrdiff_t>(tile.first + 1);
    const auto end = prefix.begin() + static_cast<std::ptrdiff_t>(tile.last);
    auto cut = std::lower_bound(begin, end, mid);
    if (cut == end || (cut != begin && mid - *(cut - 1) <= *cut - mid)) --cut;
    const auto split = static_cast<std::size_t>(cut - prefix.begin());
    const auto ratio = (*cut - prefix[tile.first]) / (prefix[tile.last] - prefix[tile.first]);
    auto a = tile, b = tile;
    a.last = b.first = split;
    if (tile.right - tile.left >= tile.top - tile.bottom) {
      a.right = b.left = std::lerp(tile.left, tile.right, ratio);
    } else {
      a.top = b.bottom = std::lerp(tile.bottom, tile.top, ratio);
    }
    pending.push_back(b);
    pending.push_back(a);
  }
  for (const auto& tile : leaves) {
    rectangles({tile.left}, {tile.bottom}, {tile.right}, {tile.top}, "", bar_color(colors, tile.first));
  }
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const auto& tile = leaves[i];
    text(std::midpoint(tile.left, tile.right), std::midpoint(tile.bottom, tile.top), labels[i], {255, 255, 255});
  }
  return hide_axes().x_limits(0, 1).y_limits(0, 1);
}

RPlot& RPlot::arc_diagram(const std::vector<double>& positions, const std::vector<std::size_t>& from,
                           const std::vector<std::size_t>& to, RGBA color) {
  require_coordinates({&positions});
  require_increasing(positions);
  if (from.size() != to.size()) throw std::invalid_argument("arc diagram: mismatched edge vectors");
  auto arcs = segment_layer(color, 1.5);
  for (std::size_t i = 0; i < from.size(); ++i) {
    if (from[i] >= positions.size() || to[i] >= positions.size() || from[i] == to[i]) {
      throw std::invalid_argument("arc diagram: invalid edge endpoint");
    }
    const double left = positions[std::min(from[i], to[i])];
    const double right = positions[std::max(from[i], to[i])];
    const double center = std::midpoint(left, right);
    const double radius = right - center;
    double px = left, py = 0.0;
    for (int j = 1; j < 24; ++j) {
      const double angle = std::numbers::pi * (1.0 - j / 24.0);
      const double x = center + radius * std::cos(angle);
      const double y = radius * std::sin(angle);
      append_segment(arcs, px, py, x, y);
      px = x;
      py = y;
    }
    append_segment(arcs, px, py, right, 0.0);
  }
  if (!arcs.x.empty()) add_series(std::move(arcs));
  return points(positions, std::vector<double>(positions.size()), color);
}

RPlot& RPlot::choropleth(const std::vector<std::vector<double>>& x,
                          const std::vector<std::vector<double>>& y,
                          const std::vector<double>& values, std::vector<RGBA> palette) {
  require_coordinates({&values});
  if (x.size() != values.size() || y.size() != values.size()) {
    throw std::invalid_argument("choropleth: expected a coordinate ring for each value");
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    require_coordinates({&x[i], &y[i]});
    if (x[i].size() < 3) throw std::invalid_argument("choropleth: rings need at least three vertices");
  }
  const auto [lo, hi] = std::ranges::minmax(values);
  if (!std::isfinite(hi - lo)) throw std::invalid_argument("choropleth: color scale overflows");
  const auto colors = value_colors(values, std::move(palette));
  for (std::size_t i = 0; i < values.size(); ++i) polygon(x[i], y[i], colors[i]);
  return *this;
}

RPlot& RPlot::radar(const std::vector<double>& values, const std::vector<std::string>& labels,
                     RGBA color) {
  if (values.size() < 3 || (!labels.empty() && labels.size() != values.size()) ||
      !std::ranges::all_of(values, [](double value) { return value >= 0 && value <= 1; })) {
    throw std::invalid_argument("radar: need at least three normalized values and matching optional labels");
  }
  std::vector<double> unit_x, unit_y;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double angle = 2 * std::numbers::pi * (static_cast<double>(i) / values.size());
    unit_x.push_back(std::sin(angle));
    unit_y.push_back(std::cos(angle));
  }
  std::vector<double> zero(values.size());
  segments(zero, zero, unit_x, unit_y, {148, 163, 184});
  for (const double radius : {.25, .5, .75, 1.0}) {
    auto x = unit_x, y = unit_y;
    for (double& value : x) value *= radius;
    for (double& value : y) value *= radius;
    polygon(std::move(x), std::move(y), {148, 163, 184}, false);
  }
  auto x = unit_x, y = unit_y;
  for (std::size_t i = 0; i < values.size(); ++i) {
    x[i] *= values[i];
    y[i] *= values[i];
  }
  polygon(std::move(x), std::move(y), color, false);
  for (std::size_t i = 0; i < labels.size(); ++i) text(1.14 * unit_x[i], 1.14 * unit_y[i], labels[i]);
  return hide_axes().x_limits(-1.32, 1.32).y_limits(-1.32, 1.32);
}

} // namespace datamunge::graphics
