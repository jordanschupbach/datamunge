#include <datamunge/plot/plot.hpp>
#include <datamunge/plot/figure.hpp>

#include <datamunge/datamunge.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/algorithms/checksums.hpp>

#include "plot/cmu_serif_font_data.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#if defined(DATAMUNGE_HAVE_X11)
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <X11/Xutil.h>
#endif

namespace datamunge::graphics {
namespace {

struct Bounds {
  double x_min{0.0};
  double x_max{1.0};
  double y_min{0.0};
  double y_max{1.0};
};

struct Layout {
  double left{90.0};
  double right{24.0};
  double top{72.0};
  double bottom{84.0};
  double plot_w{0.0};
  double plot_h{0.0};
  Bounds bounds{};
};

/// @brief Baseline offset (in em) that visually centers a line of text on a y coordinate.
constexpr double kTextCenterEm = 0.35;
/// @brief Font size of the X11 viewer's toolbar labels.
constexpr double kButtonFontPx = 13.0;

/// @brief Baseline of x-axis tick labels: a 6px gap below the axis plus room for @p px-sized text.
double x_tick_baseline(const Layout& layout, double px) { return layout.top + layout.plot_h + 6.0 + 1.5 * px; }

struct Image {
  std::size_t               width{0};
  std::size_t               height{0};
  std::vector<std::uint8_t> pixels;

  Image(std::size_t w, std::size_t h, RGB background)
      : width(w), height(h), pixels(w * h * 3U, 0U) {
    fill(background);
  }

  void fill(RGB color) {
    const auto red = static_cast<std::uint8_t>(std::clamp(color.r, 0, 255));
    const auto green = static_cast<std::uint8_t>(std::clamp(color.g, 0, 255));
    const auto blue = static_cast<std::uint8_t>(std::clamp(color.b, 0, 255));
    for (std::size_t i = 0; i < width * height; ++i) {
      pixels[i * 3U + 0U] = red;
      pixels[i * 3U + 1U] = green;
      pixels[i * 3U + 2U] = blue;
    }
  }

  void set_pixel(int x, int y, RGB color) {
    if (color.alpha < 1.0) {
      blend_pixel(x, y, color, color.alpha);
      return;
    }
    if (x < 0 || y < 0) {
      return;
    }
    const auto ux = static_cast<std::size_t>(x);
    const auto uy = static_cast<std::size_t>(y);
    if (ux >= width || uy >= height) {
      return;
    }
    const std::size_t idx = (uy * width + ux) * 3U;
    pixels[idx + 0U] = static_cast<std::uint8_t>(std::clamp(color.r, 0, 255));
    pixels[idx + 1U] = static_cast<std::uint8_t>(std::clamp(color.g, 0, 255));
    pixels[idx + 2U] = static_cast<std::uint8_t>(std::clamp(color.b, 0, 255));
  }

  void blend_pixel(int x, int y, RGB color, double opacity) {
    if (x < 0 || y < 0 || opacity <= 0.0) {
      return;
    }
    const auto ux = static_cast<std::size_t>(x);
    const auto uy = static_cast<std::size_t>(y);
    if (ux >= width || uy >= height) {
      return;
    }
    const std::size_t idx = (uy * width + ux) * 3U;
    const double alpha = std::clamp(opacity, 0.0, 1.0);
    const auto blend = [alpha](std::uint8_t destination, int source) {
      return static_cast<std::uint8_t>(std::lround(
          static_cast<double>(destination) * (1.0 - alpha) + std::clamp(source, 0, 255) * alpha));
    };
    pixels[idx + 0U] = blend(pixels[idx + 0U], color.r);
    pixels[idx + 1U] = blend(pixels[idx + 1U], color.g);
    pixels[idx + 2U] = blend(pixels[idx + 2U], color.b);
  }

  void fill_rect(int x0, int y0, int x1, int y1, RGB color) {
    if (x0 > x1) {
      std::swap(x0, x1);
    }
    if (y0 > y1) {
      std::swap(y0, y1);
    }
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        set_pixel(x, y, color);
      }
    }
  }
};

std::string rgb_css(RGB c) {
  std::ostringstream out;
  // Same clamping as the raster path, so out-of-range channels still give a valid SVG color.
  out << "rgb(" << std::clamp(c.r, 0, 255) << "," << std::clamp(c.g, 0, 255) << "," << std::clamp(c.b, 0, 255)
      << ")";
  return out.str();
}

std::string xml_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char ch : s) {
    switch (ch) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&apos;"; break;
      default: out += ch; break;
    }
  }
  return out;
}

constexpr std::array<RGB, 8> kDefaultPalette = {{
    {37, 99, 235},
    {220, 38, 38},
    {22, 163, 74},
    {217, 119, 6},
    {124, 58, 237},
    {219, 39, 119},
    {8, 145, 178},
    {161, 98, 7},
}};

/// R's "type 7" sample quantile (the default used by both `quantile()` and `boxplot.stats()`).
double quantile_type7(const std::vector<double>& sorted, double p) {
  const auto n = sorted.size();
  if (n == 1) {
    return sorted.front();
  }
  const double h = (static_cast<double>(n) - 1.0) * p;
  const auto   lo = static_cast<std::size_t>(std::floor(h));
  const auto   hi = static_cast<std::size_t>(std::ceil(h));
  return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

void require_xy_same_size(const std::vector<double>& x,
                          const std::vector<double>& y,
                          const char*                fn) {
  if (x.size() != y.size()) {
    throw std::invalid_argument(std::string(fn) + ": x and y must have the same size");
  }
  if (x.empty()) {
    throw std::invalid_argument(std::string(fn) + ": series must not be empty");
  }
}

Bounds compute_bounds(const Plot& plot) {
  if (plot.series().empty()) {
    throw std::runtime_error("Plot::save: plot has no series");
  }

  Bounds b;
  bool   first = true;
  auto   include = [&](double x, double y) {
    if (first) {
      b.x_min = b.x_max = x;
      b.y_min = b.y_max = y;
      first = false;
    } else {
      b.x_min = std::min(b.x_min, x);
      b.x_max = std::max(b.x_max, x);
      b.y_min = std::min(b.y_min, y);
      b.y_max = std::max(b.y_max, y);
    }
  };

  for (const auto& series : plot.series()) {
    if (series.kind == DataSeries::Kind::Box) {
      // x = {position}; y = {whisker_lo, q1, median, q3, whisker_hi, outliers...}
      const double position = series.x.empty() ? 0.0 : series.x[0];
      const double half     = series.bar_width / 2.0;
      for (double value : series.y) {
        include(position - half, value);
        include(position + half, value);
      }
      continue;
    }
    for (std::size_t i = 0; i < series.x.size(); ++i) {
      include(series.x[i], series.y[i]);
      if (series.kind == DataSeries::Kind::Bar) {
        b.y_min = std::min(b.y_min, 0.0);
        const double half = series.bar_width / 2.0;
        include(series.x[i] - half, series.y[i]);
        include(series.x[i] + half, series.y[i]);
      }
    }
  }
  for (const auto& line : plot.reference_lines()) {
    if (line.vertical) {
      include(line.value, first ? 0.0 : b.y_min);
    }
  }

  if (b.x_min == b.x_max) {
    b.x_min -= 1.0;
    b.x_max += 1.0;
  }
  if (b.y_min == b.y_max) {
    b.y_min -= 1.0;
    b.y_max += 1.0;
  }

  const double x_pad = 0.05 * (b.x_max - b.x_min);
  const double y_pad = 0.10 * (b.y_max - b.y_min);
  b.x_min -= x_pad;
  b.x_max += x_pad;
  b.y_min -= y_pad;
  b.y_max += y_pad;
  if (plot.has_x_limits()) {
    b.x_min = plot.x_min();
    b.x_max = plot.x_max();
  }
  if (plot.has_y_limits()) {
    b.y_min = plot.y_min();
    b.y_max = plot.y_max();
  }
  return b;
}

Layout compute_layout(const Plot& plot, Bounds bounds) {
  Layout layout;
  if (plot.has_custom_margins()) {
    layout.left = plot.margin_left();
    layout.right = plot.margin_right();
    layout.top = plot.margin_top();
    layout.bottom = plot.margin_bottom();
  } else if (!plot.subtitle_text().empty()) {
    layout.top += 24.0;
  }
  layout.plot_w = static_cast<double>(plot.width()) - layout.left - layout.right;
  layout.plot_h = static_cast<double>(plot.height()) - layout.top - layout.bottom;
  if (layout.plot_w <= 0.0 || layout.plot_h <= 0.0) {
    throw std::runtime_error("Plot::save: invalid plot dimensions");
  }
  layout.bounds = bounds;
  return layout;
}

Layout compute_layout(const Plot& plot) { return compute_layout(plot, compute_bounds(plot)); }

double map_x(const Layout& layout, double x) {
  return layout.left
       + (x - layout.bounds.x_min) / (layout.bounds.x_max - layout.bounds.x_min) * layout.plot_w;
}

double map_y(const Layout& layout, double y) {
  return layout.top + layout.plot_h
       - (y - layout.bounds.y_min) / (layout.bounds.y_max - layout.bounds.y_min) * layout.plot_h;
}

std::string format_tick(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << value;
  std::string s = out.str();
  while (!s.empty() && s.back() == '0') {
    s.pop_back();
  }
  if (!s.empty() && s.back() == '.') {
    s.pop_back();
  }
  if (s == "-0") {
    s = "0";
  }
  return s;
}

/// @brief One axis tick: its pixel coordinate along the axis and its label.
struct AxisTick {
  double      pixel; ///< x pixel for the horizontal axis, y pixel for the vertical one
  std::string label; ///< text drawn beside the tick
};

/// @brief Ticks for one axis: the custom labels (at their positions, or 0, 1, 2, ...) when any
///        are set, otherwise six evenly spaced numeric ticks across the visible range.
std::vector<AxisTick> axis_ticks(const Layout&                   layout,
                                 bool                            horizontal,
                                 bool                            pretty,
                                 const std::vector<std::string>& labels,
                                 const std::vector<double>&      positions) {
  std::vector<AxisTick> ticks;
  if (!labels.empty()) {
    for (std::size_t i = 0; i < labels.size(); ++i) {
      const double value = positions.empty() ? static_cast<double>(i) : positions[i];
      ticks.push_back({horizontal ? map_x(layout, value) : map_y(layout, value), labels[i]});
    }
    return ticks;
  }
  const double lo = horizontal ? layout.bounds.x_min : layout.bounds.y_min;
  const double hi = horizontal ? layout.bounds.x_max : layout.bounds.y_max;
  if (pretty && hi > lo) {
    // Heckbert's nice number nearest a fifth of the range: 1, 2, 5 or 10 times 10^k.
    const double raw = (hi - lo) / 5.0;
    const double magnitude = std::pow(10.0, std::floor(std::log10(raw)));
    const double fraction = raw / magnitude;
    const double nice = fraction < 1.5 ? 1.0 : fraction < 3.0 ? 2.0 : fraction < 7.0 ? 5.0 : 10.0;
    const double step = nice * magnitude;
    for (double v = std::ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
      const double value = std::abs(v) < step * 1e-9 ? 0.0 : v;
      ticks.push_back({horizontal ? map_x(layout, value) : map_y(layout, value), format_tick(value)});
    }
    return ticks;
  }
  for (int i = 0; i <= 5; ++i) {
    const double fraction = static_cast<double>(i) / 5.0;
    const double pixel = horizontal ? layout.left + layout.plot_w * fraction
                                    : layout.top + layout.plot_h - layout.plot_h * fraction;
    ticks.push_back({pixel, format_tick(lo + (hi - lo) * fraction)});
  }
  return ticks;
}

std::vector<AxisTick> x_axis_ticks(const Plot& plot, const Layout& layout) {
  return axis_ticks(layout, true, plot.pretty_ticks_enabled(), plot.x_tick_label_list(), plot.x_tick_position_list());
}

std::vector<AxisTick> y_axis_ticks(const Plot& plot, const Layout& layout) {
  return axis_ticks(layout, false, plot.pretty_ticks_enabled(), plot.y_tick_label_list(), plot.y_tick_position_list());
}

/// @brief True when a tick's pixel lies within [lo, hi] (with half a pixel of slack).
bool tick_inside(double pixel, double lo, double hi) { return pixel >= lo - 0.5 && pixel <= hi + 0.5; }

/// @brief @p color mixed toward black by @p amount in [0, 1]; keeps alpha.
RGB darken(RGB color, double amount) {
  const auto mix = [amount](int channel) { return static_cast<int>(std::lround(channel * (1.0 - amount))); };
  return RGB{mix(color.r), mix(color.g), mix(color.b), color.alpha};
}

/// @brief CSS font stack for SVG text: the embedded Computer Modern subset, then installed copies.
constexpr const char* kSvgFontFamily = "'datamunge CMU Serif','CMU Serif','Latin Modern Roman','Computer Modern',serif";

/// @brief `<defs>` block embedding the CMU Serif subset (WOFF, base64) for SVG text; built once.
const std::string& svg_font_defs() {
  static const std::string defs = [] {
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto*    data = detail::cmu_serif::woff;
    const std::size_t size = std::size(detail::cmu_serif::woff);
    std::string    b64;
    b64.reserve((size + 2) / 3 * 4);
    for (std::size_t i = 0; i < size; i += 3) {
      const std::uint32_t n = (std::uint32_t{data[i]} << 16) | (i + 1 < size ? std::uint32_t{data[i + 1]} << 8 : 0U)
                            | (i + 2 < size ? std::uint32_t{data[i + 2]} : 0U);
      b64 += alphabet[(n >> 18) & 63U];
      b64 += alphabet[(n >> 12) & 63U];
      b64 += i + 1 < size ? alphabet[(n >> 6) & 63U] : '=';
      b64 += i + 2 < size ? alphabet[n & 63U] : '=';
    }
    return "  <defs><style>@font-face{font-family:'datamunge CMU Serif';src:url(data:font/woff;base64," + b64
         + ") format('woff');}</style></defs>\n";
  }();
  return defs;
}

void write_axes_and_grid(std::ostringstream& out, const Plot& plot, const Layout& layout) {
  out << "  <rect width=\"100%\" height=\"100%\" fill=\"" << rgb_css(plot.background_color())
      << "\"/>\n";
  out << "  <style>"
      // The default colour applies only to text without its own fill attribute: a CSS rule
      // would otherwise override the per-label colour of text().
      << "text{font-family:" << kSvgFontFamily << ";}text:not([fill]){fill:" << rgb_css(plot.axes_color()) << ";}"
      << "</style>\n";

  if (!plot.axes_hidden()) {
    if (plot.grid_visible()) {
      // Grid lines sit on the ticks, so they line up with the labels.
      for (const auto& tick : x_axis_ticks(plot, layout)) {
        if (!tick_inside(tick.pixel, layout.left, layout.left + layout.plot_w)) continue;
        out << "  <line x1=\"" << tick.pixel << "\" y1=\"" << layout.top << "\" x2=\"" << tick.pixel
            << "\" y2=\"" << layout.top + layout.plot_h << "\" stroke=\""
            << rgb_css(plot.major_grid_color()) << "\" stroke-width=\"1\"/>\n";
      }
      for (const auto& tick : y_axis_ticks(plot, layout)) {
        if (!tick_inside(tick.pixel, layout.top, layout.top + layout.plot_h)) continue;
        out << "  <line x1=\"" << layout.left << "\" y1=\"" << tick.pixel << "\" x2=\""
            << layout.left + layout.plot_w << "\" y2=\"" << tick.pixel << "\" stroke=\""
            << rgb_css(plot.major_grid_color()) << "\" stroke-width=\"1\"/>\n";
      }
    }

    out << "  <line x1=\"" << layout.left << "\" y1=\"" << layout.top + layout.plot_h
        << "\" x2=\"" << layout.left + layout.plot_w << "\" y2=\"" << layout.top + layout.plot_h
        << "\" stroke=\"" << rgb_css(plot.axes_color()) << "\" stroke-width=\"2\"/>\n";
    out << "  <line x1=\"" << layout.left << "\" y1=\"" << layout.top << "\" x2=\""
        << layout.left << "\" y2=\"" << layout.top + layout.plot_h << "\" stroke=\""
        << rgb_css(plot.axes_color()) << "\" stroke-width=\"2\"/>\n";

    for (const auto& tick : x_axis_ticks(plot, layout)) {
      out << "  <text x=\"" << tick.pixel << "\" y=\"" << x_tick_baseline(layout, plot.tick_font_px())
          << "\" font-size=\"" << plot.tick_font_px() << "\" text-anchor=\"middle\">" << xml_escape(tick.label)
          << "</text>\n";
    }
    for (const auto& tick : y_axis_ticks(plot, layout)) {
      out << "  <text x=\"" << layout.left - 10 << "\" y=\"" << tick.pixel + kTextCenterEm * plot.tick_font_px()
          << "\" font-size=\"" << plot.tick_font_px() << "\" text-anchor=\"end\">" << xml_escape(tick.label)
          << "</text>\n";
    }
  }

  out << "  <text x=\"" << plot.width() / 2.0
      << "\" y=\"36\" font-size=\"24\" text-anchor=\"middle\">"
      << xml_escape(plot.title_text()) << "</text>\n";
  if (!plot.subtitle_text().empty()) {
    out << "  <text x=\"" << plot.width() / 2.0
        << "\" y=\"60\" font-size=\"16\" text-anchor=\"middle\">"
        << xml_escape(plot.subtitle_text()) << "</text>\n";
  }
  if (!plot.axes_hidden()) {
    out << "  <text x=\"" << plot.width() / 2.0 << "\" y=\"" << plot.height() - 20
        << "\" font-size=\"16\" text-anchor=\"middle\">" << xml_escape(plot.x_label_text())
        << "</text>\n";
    out << "  <text x=\"24\" y=\"" << plot.height() / 2.0
        << "\" font-size=\"16\" text-anchor=\"middle\" transform=\"rotate(-90 24 "
        << plot.height() / 2.0 << ")\">" << xml_escape(plot.y_label_text()) << "</text>\n";
  }
}

/// @brief Number of legend rows: explicit legend() entries, else one per labelled non-text series.
std::size_t legend_row_count(const Plot& plot) {
  if (!plot.legend_entries().empty()) return plot.legend_entries().size();
  std::size_t rows = 0;
  for (const auto& series : plot.series())
    if (!series.label.empty() && series.kind != DataSeries::Kind::Text) ++rows;
  return rows;
}

/// @brief Pixel geometry of one box-and-whisker series (see DataSeries::Kind::Box).
struct BoxGeometry {
  double cx, left, right, cap_left, cap_right;  ///< Horizontal pixel positions.
  double lo, q1, median, q3, hi;               ///< Vertical pixel positions.
  std::vector<double> outliers;                ///< Vertical pixel positions of outliers.
};

BoxGeometry box_geometry(const DataSeries& series, const Layout& layout) {
  const double position = series.x[0];
  const double half = series.bar_width / 2.0;
  BoxGeometry g{};
  g.cx = map_x(layout, position);
  g.left = map_x(layout, position - half);
  g.right = map_x(layout, position + half);
  g.cap_left = map_x(layout, position - half / 2.0);
  g.cap_right = map_x(layout, position + half / 2.0);
  g.lo = map_y(layout, series.y[0]);
  g.q1 = map_y(layout, series.y[1]);
  g.median = map_y(layout, series.y[2]);
  g.q3 = map_y(layout, series.y[3]);
  g.hi = map_y(layout, series.y[4]);
  for (std::size_t i = 5; i < series.y.size(); ++i) g.outliers.push_back(map_y(layout, series.y[i]));
  return g;
}

/// @brief Returns a scatter point's recycled palette color, falling back to the series color.
RGB scatter_color(const DataSeries& series, std::size_t index) {
  return series.point_colors.empty() ? series.color : series.point_colors[index % series.point_colors.size()];
}

void write_series(std::ostringstream& out, const Plot& plot, const Layout& layout) {
  for (const auto& line : plot.reference_lines()) {
    const std::string color = rgb_css(line.color);
    double x0, y0, x1, y1;
    if (line.vertical) {
      x0 = x1 = map_x(layout, line.value);
      y0 = layout.top;
      y1 = layout.top + layout.plot_h;
    } else {
      x0 = layout.left;
      x1 = layout.left + layout.plot_w;
      y0 = map_y(layout, line.value + line.slope * layout.bounds.x_min);
      y1 = map_y(layout, line.value + line.slope * layout.bounds.x_max);
    }
    out << "  <line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x1 << "\" y2=\"" << y1
        << "\" stroke=\"" << color << "\" stroke-width=\"" << line.stroke_width << "\"/>\n";
  }

  for (const auto& series : plot.series()) {
    const std::string color = rgb_css(series.color);
    // Translucent series: scatter points each get their own opacity so overlaps darken; any
    // other kind is wrapped in one group so its own overlapping parts don't double up.
    const double alpha = std::clamp(series.color.alpha, 0.0, 1.0);
    const bool translucent = alpha < 1.0;
    const bool group_alpha = translucent && series.kind != DataSeries::Kind::Scatter;
    if (group_alpha) {
      out << "  <g opacity=\"" << alpha << "\">\n";
    }
    switch (series.kind) {
      case DataSeries::Kind::Line: {
        out << "  <polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\""
            << series.stroke_width << "\" points=\"";
        for (std::size_t i = 0; i < series.x.size(); ++i) {
          out << map_x(layout, series.x[i]) << "," << map_y(layout, series.y[i]) << " ";
        }
        out << "\"/>\n";
        break;
      }
      case DataSeries::Kind::Scatter: {
        for (std::size_t i = 0; i < series.x.size(); ++i) {
          const RGB point_color = scatter_color(series, i);
          const double point_alpha = std::clamp(point_color.alpha, 0.0, 1.0);
          out << "  <circle cx=\"" << map_x(layout, series.x[i]) << "\" cy=\""
              << map_y(layout, series.y[i]) << "\" r=\"" << series.marker_size << "\" fill=\""
              << rgb_css(point_color) << "\"";
          if (point_alpha < 1.0) {
            out << " fill-opacity=\"" << point_alpha << "\"";
          }
          out << "/>\n";
        }
        break;
      }
      case DataSeries::Kind::Bar: {
        for (std::size_t i = 0; i < series.x.size(); ++i) {
          const double x_left = map_x(layout, series.x[i] - series.bar_width / 2.0);
          const double x_right = map_x(layout, series.x[i] + series.bar_width / 2.0);
          const double y0 = map_y(layout, 0.0);
          const double y1 = map_y(layout, series.y[i]);
          const double rect_y = std::min(y0, y1);
          const double rect_h = std::abs(y1 - y0);
          out << "  <rect x=\"" << x_left << "\" y=\"" << rect_y << "\" width=\""
              << (x_right - x_left) << "\" height=\"" << rect_h << "\" fill=\"" << color
              << "\" fill-opacity=\"0.85\"/>\n";
        }
        break;
      }
      case DataSeries::Kind::Box: {
        if (series.x.empty() || series.y.size() < 5) {
          break;
        }
        // R's boxplot look: whisker caps half the box width (staplewex = 0.5), a box filled
        // with the series color, and a darker outline, median and open-circle outliers.
        const auto geo = box_geometry(series, layout);
        const std::string ink = rgb_css(darken(series.color, 0.35));
        const auto line = [&](double x0, double y0, double x1, double y1, double width) {
          out << "  <line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x1 << "\" y2=\"" << y1
              << "\" stroke=\"" << ink << "\" stroke-width=\"" << width << "\"/>\n";
        };
        line(geo.cx, geo.lo, geo.cx, geo.q1, 1.5);
        line(geo.cx, geo.q3, geo.cx, geo.hi, 1.5);
        line(geo.cap_left, geo.lo, geo.cap_right, geo.lo, 1.5);
        line(geo.cap_left, geo.hi, geo.cap_right, geo.hi, 1.5);
        out << "  <rect x=\"" << geo.left << "\" y=\"" << geo.q3 << "\" width=\"" << (geo.right - geo.left)
            << "\" height=\"" << (geo.q1 - geo.q3) << "\" fill=\"" << color << "\" fill-opacity=\"0.55\" stroke=\""
            << ink << "\" stroke-width=\"1.5\"/>\n";
        line(geo.left, geo.median, geo.right, geo.median, 2.5);
        for (const double y : geo.outliers) {
          out << "  <circle cx=\"" << geo.cx << "\" cy=\"" << y << "\" r=\"3\" fill=\"none\" stroke=\"" << ink
              << "\" stroke-width=\"1.2\"/>\n";
        }
        break;
      }
      case DataSeries::Kind::Polygon: {
        out << "  <polygon points=\"";
        for (std::size_t i = 0; i < series.x.size(); ++i) {
          out << map_x(layout, series.x[i]) << "," << map_y(layout, series.y[i]) << " ";
        }
        if (series.filled) {
          out << "\" fill=\"" << color << "\" stroke=\"" << color << "\" stroke-width=\"1\"/>\n";
        } else {
          out << "\" fill=\"none\" stroke=\"" << color << "\" stroke-width=\""
              << series.stroke_width << "\"/>\n";
        }
        break;
      }
      case DataSeries::Kind::Text: {
        if (series.x.empty()) {
          break;
        }
        out << "  <text x=\"" << map_x(layout, series.x[0]) << "\" y=\""
            << map_y(layout, series.y[0]) << "\" dy=\"0.35em\" font-size=\"" << series.marker_size
            << "\" fill=\"" << color << "\" text-anchor=\"middle\">" << xml_escape(series.label)
            << "</text>\n";
        break;
      }
      case DataSeries::Kind::Segment: {
        for (std::size_t i = 0; i + 1 < series.x.size(); i += 2) {
          out << "  <line x1=\"" << map_x(layout, series.x[i]) << "\" y1=\""
              << map_y(layout, series.y[i]) << "\" x2=\"" << map_x(layout, series.x[i + 1])
              << "\" y2=\"" << map_y(layout, series.y[i + 1]) << "\" stroke=\"" << color
              << "\" stroke-width=\"" << series.stroke_width << "\"/>\n";
        }
        break;
      }
    }
    if (group_alpha) {
      out << "  </g>\n";
    }
  }

  double legend_y = layout.top;
  if (const std::size_t rows = legend_row_count(plot); rows > 0) {
    // Translucent backing so the legend stays readable over data.
    const double x0 = layout.left + layout.plot_w - 140.0;
    out << "  <rect x=\"" << x0 - 8 << "\" y=\"" << legend_y - 18 << "\" width=\"" << 140.0 - 6 + 8
        << "\" height=\"" << 20.0 * static_cast<double>(rows) + 6 << "\" rx=\"3\" fill=\""
        << rgb_css(plot.background_color()) << "\" fill-opacity=\"0.85\"/>\n";
  }
  if (!plot.legend_entries().empty()) {
    for (const auto& entry : plot.legend_entries()) {
      const double x0 = layout.left + layout.plot_w - 140.0;
      out << "  <rect x=\"" << x0 << "\" y=\"" << legend_y - 12
          << "\" width=\"18\" height=\"8\" fill=\"" << rgb_css(entry.color) << "\"";
      if (entry.color.alpha < 1.0)
        out << " fill-opacity=\"" << std::clamp(entry.color.alpha, 0.0, 1.0) << "\"";
      out << "/>\n";
      out << "  <text x=\"" << x0 + 26 << "\" y=\"" << legend_y - 4 << "\" font-size=\"12\">"
          << xml_escape(entry.label) << "</text>\n";
      legend_y += 20.0;
    }
  } else {
    for (const auto& series : plot.series()) {
      if (series.label.empty() || series.kind == DataSeries::Kind::Text) {
        continue;
      }
      const double x0 = layout.left + layout.plot_w - 140.0;
      out << "  <rect x=\"" << x0 << "\" y=\"" << legend_y - 12
          << "\" width=\"18\" height=\"8\" fill=\"" << rgb_css(series.color) << "\"/>\n";
      out << "  <text x=\"" << x0 + 26 << "\" y=\"" << legend_y - 4 << "\" font-size=\"12\">"
          << xml_escape(series.label) << "</text>\n";
      legend_y += 20.0;
    }
  }
}

std::string svg_body(const Plot& plot) {
  const Layout layout = compute_layout(plot);
  std::ostringstream out;
  write_axes_and_grid(out, plot, layout);
  write_series(out, plot, layout);
  return out.str();
}

std::string svg_string(const Plot& plot) {
  std::ostringstream out;
  out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << plot.width()
      << "\" height=\"" << plot.height() << "\" viewBox=\"0 0 " << plot.width() << " "
      << plot.height() << "\">\n";
  out << svg_font_defs();
  out << svg_body(plot);
  out << "</svg>\n";
  return out.str();
}

void draw_circle(Image& image, double cx, double cy, double radius, RGB color) {
  constexpr int samples_per_axis = 4;
  radius = std::max(radius, 1.0);
  const double radius_squared = radius * radius;
  const int x_begin = static_cast<int>(std::floor(cx - radius - 0.5));
  const int x_end = static_cast<int>(std::ceil(cx + radius + 0.5));
  const int y_begin = static_cast<int>(std::floor(cy - radius - 0.5));
  const int y_end = static_cast<int>(std::ceil(cy + radius + 0.5));
  for (int y = y_begin; y <= y_end; ++y) {
    for (int x = x_begin; x <= x_end; ++x) {
      int covered_samples = 0;
      for (int sample_y = 0; sample_y < samples_per_axis; ++sample_y) {
        for (int sample_x = 0; sample_x < samples_per_axis; ++sample_x) {
          const double px = static_cast<double>(x)
                          + (static_cast<double>(sample_x) + 0.5) / samples_per_axis;
          const double py = static_cast<double>(y)
                          + (static_cast<double>(sample_y) + 0.5) / samples_per_axis;
          const double dx = px - cx;
          const double dy = py - cy;
          if (dx * dx + dy * dy <= radius_squared) {
            ++covered_samples;
          }
        }
      }
      if (covered_samples != 0) {
        image.blend_pixel(
            x, y, color, std::clamp(color.alpha, 0.0, 1.0) *
                static_cast<double>(covered_samples) / (samples_per_axis * samples_per_axis));
      }
    }
  }
}

void draw_line(Image& image, double x0, double y0, double x1, double y1, RGB color, double width) {
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)))));
  const double radius = std::max(1.0, std::round(width / 2.0));
  for (int i = 0; i <= steps; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(steps);
    const double px = x0 + dx * t;
    const double py = y0 + dy * t;
    draw_circle(image, px, py, radius, color);
  }
}

// ---- Text: embedded Computer Modern (CMU Serif) --------------------------------------------

namespace font = detail::cmu_serif;

/// @brief Where a text run sits relative to its anchor x (SVG `text-anchor`).
enum class TextAnchor { Start, Middle, End };

/// @brief Decodes UTF-8, mapping malformed sequences to U+FFFD.
std::vector<char32_t> decode_utf8(const std::string& text) {
  std::vector<char32_t> out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    const auto lead = static_cast<unsigned char>(text[i]);
    const int  extra = lead < 0x80 ? 0 : (lead >> 5) == 0x6 ? 1 : (lead >> 4) == 0xE ? 2 : (lead >> 3) == 0x1E ? 3 : -1;
    if (extra < 0 || i + static_cast<std::size_t>(extra) >= text.size()) {
      out.push_back(U'�');
      ++i;
      continue;
    }
    char32_t cp = extra == 0 ? lead : lead & (0x3F >> extra);
    bool     ok = true;
    for (int k = 1; k <= extra; ++k) {
      const auto next = static_cast<unsigned char>(text[i + static_cast<std::size_t>(k)]);
      if ((next & 0xC0) != 0x80) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (next & 0x3F);
    }
    out.push_back(ok ? cp : U'�');
    i += ok ? static_cast<std::size_t>(extra) + 1 : 1;
  }
  return out;
}

/// @brief Index of @p cp in the embedded glyph table, or of '?' when the font lacks it.
std::size_t glyph_index(char32_t cp) {
  const auto begin = std::begin(font::glyphs);
  const auto end = std::end(font::glyphs);
  const auto found = std::lower_bound(begin, end, cp, [](const font::GlyphEntry& g, char32_t c) { return g.codepoint < c; });
  if (found != end && found->codepoint == cp) return static_cast<std::size_t>(found - begin);
  return cp == U'?' ? 0 : glyph_index(U'?');
}

/// @brief Pair kerning between two glyph indices, in font units (0 when unkerned).
int kerning(std::size_t left, std::size_t right) {
  const auto key = std::pair{left, right};
  const auto found = std::lower_bound(std::begin(font::kerning), std::end(font::kerning), key,
                                      [](const font::KernPair& k, const std::pair<std::size_t, std::size_t>& v) {
                                        return std::pair<std::size_t, std::size_t>{k.left, k.right} < v;
                                      });
  if (found != std::end(font::kerning) && found->left == left && found->right == right) return found->value;
  return 0;
}

/// @brief Glyph indices of @p text plus each glyph's pen x offset in font units; returns the total advance.
double layout_glyphs(const std::string& text, std::vector<std::size_t>& indices, std::vector<double>& offsets) {
  indices.clear();
  offsets.clear();
  double pen = 0.0;
  for (const char32_t cp : decode_utf8(text)) {
    const std::size_t index = glyph_index(cp);
    if (!indices.empty()) pen += kerning(indices.back(), index);
    indices.push_back(index);
    offsets.push_back(pen);
    pen += font::glyphs[index].advance;
  }
  return pen;
}

/// @brief Width in pixels of @p text set in the embedded font at @p px pixels per em.
[[maybe_unused]] double text_width(const std::string& text, double px) {
  std::vector<std::size_t> indices;
  std::vector<double>      offsets;
  return layout_glyphs(text, indices, offsets) * px / font::units_per_em;
}

/// @brief Exact-area coverage accumulator: lines add signed area, a running row sum yields coverage.
class CoverageRaster {
 public:
  CoverageRaster(int width, int height)
      : width_(width), height_(height), acc_(static_cast<std::size_t>(width + 3) * static_cast<std::size_t>(height), 0.0f) {}

  /// @brief Adds the signed area contributed by the edge from (x0, y0) to (x1, y1).
  void line(double x0, double y0, double x1, double y1) {
    if (y0 == y1) return;
    const float dir = y0 < y1 ? 1.0f : -1.0f;
    if (y0 > y1) {
      std::swap(x0, x1);
      std::swap(y0, y1);
    }
    const double dxdy = (x1 - x0) / (y1 - y0);
    const int    row_begin = std::max(0, static_cast<int>(std::floor(y0)));
    const int    row_end = std::min(height_, static_cast<int>(std::ceil(y1)));
    const int    stride = width_ + 3;
    double       x = x0 + (std::max(y0, static_cast<double>(row_begin)) - y0) * dxdy;
    for (int row = row_begin; row < row_end; ++row) {
      const double dy = std::min(static_cast<double>(row + 1), y1) - std::max(static_cast<double>(row), y0);
      const double x_next = x + dxdy * dy;
      const float  d = static_cast<float>(dy) * dir;
      const double lo = std::clamp(std::min(x, x_next), 0.0, static_cast<double>(width_));
      const double hi = std::clamp(std::max(x, x_next), 0.0, static_cast<double>(width_));
      float*       cells = acc_.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(stride);
      const int    lo_i = static_cast<int>(std::floor(lo));
      const int    hi_i = static_cast<int>(std::ceil(hi));
      if (hi_i <= lo_i + 1) {
        const double mid = 0.5 * (lo + hi) - lo_i;
        cells[lo_i] += d * static_cast<float>(1.0 - mid);
        cells[lo_i + 1] += d * static_cast<float>(mid);
      } else {
        const double s = 1.0 / (hi - lo);
        const double lo_f = lo - lo_i;
        const double a0 = 0.5 * s * (1.0 - lo_f) * (1.0 - lo_f);
        const double hi_f = hi - hi_i + 1.0;
        const double am = 0.5 * s * hi_f * hi_f;
        cells[lo_i] += d * static_cast<float>(a0);
        if (hi_i == lo_i + 2) {
          cells[lo_i + 1] += d * static_cast<float>(1.0 - a0 - am);
        } else {
          const double a1 = s * (1.5 - lo_f);
          cells[lo_i + 1] += d * static_cast<float>(a1 - a0);
          for (int c = lo_i + 2; c < hi_i - 1; ++c) cells[c] += d * static_cast<float>(s);
          const double a2 = a1 + (hi_i - lo_i - 3) * s;
          cells[hi_i - 1] += d * static_cast<float>(1.0 - a2 - am);
        }
        cells[hi_i] += d * static_cast<float>(am);
      }
      x = x_next;
    }
  }

  /// @brief Blends the accumulated coverage into @p image at offset (@p ox, @p oy).
  void composite(Image& image, int ox, int oy, RGB color) const {
    const double opacity = std::clamp(color.alpha, 0.0, 1.0);
    const int    stride = width_ + 3;
    for (int row = 0; row < height_; ++row) {
      float        sum = 0.0f;
      const float* cells = acc_.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(stride);
      for (int col = 0; col < width_; ++col) {
        sum += cells[col];
        const double coverage = std::min(1.0, static_cast<double>(std::fabs(sum)));
        // A mild gamma lift keeps Computer Modern's hairline serifs legible at small sizes.
        if (coverage > 1.0 / 255.0) image.blend_pixel(ox + col, oy + row, color, opacity * std::pow(coverage, 0.8));
      }
    }
  }

 private:
  int                width_;  ///< raster width in pixels
  int                height_; ///< raster height in pixels
  std::vector<float> acc_;    ///< per-row signed area deltas (3 columns of slack)
};

/// @brief Draws @p text with its baseline at (@p x, @p y) in @p px-pixel Computer Modern.
///        @p vertical rotates it 90 degrees counter-clockwise about the anchor, like SVG rotate(-90).
void draw_text(Image&             image,
               double             x,
               double             y,
               const std::string& text,
               RGB                color,
               double             px,
               TextAnchor         anchor = TextAnchor::Start,
               bool               vertical = false) {
  if (text.empty() || px <= 0.0) return;
  std::vector<std::size_t> indices;
  std::vector<double>      offsets;
  const double             scale = px / font::units_per_em;
  const double             width = layout_glyphs(text, indices, offsets) * scale;
  const double             shift = anchor == TextAnchor::Middle ? -width / 2.0 : anchor == TextAnchor::End ? -width : 0.0;

  // Text space: u along the baseline, v downward. Map to the image, rotating when vertical.
  const auto to_image = [&](double u, double v) {
    u += shift;
    return vertical ? std::pair{x + v, y - u} : std::pair{x + u, y + v};
  };
  std::vector<std::array<double, 4>> edges;
  for (std::size_t g = 0; g < indices.size(); ++g) {
    const auto&   glyph = font::glyphs[indices[g]];
    const double  origin = offsets[g] * scale;
    const auto*   pts = font::points + 2U * glyph.point_begin;
    std::pair<double, double> start{}, pen{};
    const auto point = [&](int k) { return to_image(origin + pts[2 * k] * scale, -pts[2 * k + 1] * scale); };
    int next = 0;
    for (std::uint32_t o = 0; o < glyph.op_count; ++o) {
      switch (font::ops[glyph.op_begin + o]) {
        case font::MOVE:
          start = pen = point(next++);
          break;
        case font::LINE: {
          const auto p = point(next++);
          edges.push_back({pen.first, pen.second, p.first, p.second});
          pen = p;
          break;
        }
        case font::QUAD:
        case font::CUBIC: {
          const bool cubic = font::ops[glyph.op_begin + o] == font::CUBIC;
          const auto c1 = point(next++);
          const auto c2 = point(next++);
          const auto c3 = cubic ? point(next++) : c2;
          const auto end = cubic ? c3 : c2;
          const double hull = std::hypot(c1.first - pen.first, c1.second - pen.second)
                            + std::hypot(c2.first - c1.first, c2.second - c1.second)
                            + (cubic ? std::hypot(c3.first - c2.first, c3.second - c2.second) : 0.0);
          const int steps = std::clamp(static_cast<int>(std::ceil(std::sqrt(hull) * 1.5)), 1, 24);
          auto prev = pen;
          for (int s = 1; s <= steps; ++s) {
            const double t = static_cast<double>(s) / steps;
            const double m = 1.0 - t;
            std::pair<double, double> p;
            if (cubic) {
              const double a = m * m * m, b = 3 * m * m * t, c = 3 * m * t * t, d = t * t * t;
              p = {a * pen.first + b * c1.first + c * c2.first + d * c3.first,
                   a * pen.second + b * c1.second + c * c2.second + d * c3.second};
            } else {
              const double a = m * m, b = 2 * m * t, c = t * t;
              p = {a * pen.first + b * c1.first + c * c2.first, a * pen.second + b * c1.second + c * c2.second};
            }
            edges.push_back({prev.first, prev.second, p.first, p.second});
            prev = p;
          }
          pen = end;
          break;
        }
        case font::CLOSE:
          if (pen != start) edges.push_back({pen.first, pen.second, start.first, start.second});
          pen = start;
          break;
        default:
          break;
      }
    }
  }
  if (edges.empty()) return;

  double min_x = edges[0][0], max_x = min_x, min_y = edges[0][1], max_y = min_y;
  for (const auto& e : edges) {
    min_x = std::min({min_x, e[0], e[2]});
    max_x = std::max({max_x, e[0], e[2]});
    min_y = std::min({min_y, e[1], e[3]});
    max_y = std::max({max_y, e[1], e[3]});
  }
  // Clip the raster to the image so huge or off-canvas text stays cheap.
  const int ox = std::max(0, static_cast<int>(std::floor(min_x)));
  const int oy = std::max(0, static_cast<int>(std::floor(min_y)));
  const int x_end = std::min(static_cast<int>(image.width), static_cast<int>(std::ceil(max_x)) + 1);
  const int y_end = std::min(static_cast<int>(image.height), static_cast<int>(std::ceil(max_y)) + 1);
  if (x_end <= ox || y_end <= oy) return;
  CoverageRaster raster(x_end - ox, y_end - oy);
  for (const auto& e : edges) raster.line(e[0] - ox, e[1] - oy, e[2] - ox, e[3] - oy);
  raster.composite(image, ox, oy, color);
}

/// @brief Draws manual legend entries, or the automatic non-text series legend when none exist.
void draw_legend(Image& image, const Plot& plot, const Layout& layout) {
  int legend_y = static_cast<int>(layout.top);
  const int x0 = static_cast<int>(layout.left + layout.plot_w - 140.0);
  /// @brief Draws one legend swatch and label, then advances to the next row (same geometry as SVG).
  const auto draw_entry = [&](const std::string& label, RGB color) {
    image.fill_rect(x0, legend_y - 12, x0 + 18, legend_y - 4, color);
    draw_text(image, x0 + 26, legend_y - 4, label, plot.axes_color(), 12.0);
    legend_y += 20;
  };
  const std::size_t rows = legend_row_count(plot);
  if (rows > 0) {
    // Translucent backing so the legend stays readable over data.
    const RGB bg = plot.background_color();
    const int right = static_cast<int>(layout.left + layout.plot_w) - 6;
    for (int y = legend_y - 18; y < legend_y - 18 + 20 * static_cast<int>(rows) + 6; ++y)
      for (int x = x0 - 8; x < right; ++x) image.blend_pixel(x, y, bg, 0.85);
  }
  if (!plot.legend_entries().empty()) {
    for (const auto& entry : plot.legend_entries()) draw_entry(entry.label, entry.color);
  } else {
    for (const auto& series : plot.series()) {
      if (!series.label.empty() && series.kind != DataSeries::Kind::Text)
        draw_entry(series.label, series.color);
    }
  }
}

/// @brief Fills a simple or self-intersecting polygon (even-odd rule) given in pixel space,
///        sampling each pixel at its centre and blending with the color's alpha.
void fill_polygon(Image& image, const std::vector<double>& px, const std::vector<double>& py, RGB color) {
  if (px.size() < 3) return;
  const auto [y_lo, y_hi] = std::minmax_element(py.begin(), py.end());
  const int  row_begin = std::max(0, static_cast<int>(std::floor(*y_lo)));
  const int  row_end = std::min(static_cast<int>(image.height) - 1, static_cast<int>(std::ceil(*y_hi)));
  const double opacity = std::clamp(color.alpha, 0.0, 1.0);
  std::vector<double> crossings;
  for (int row = row_begin; row <= row_end; ++row) {
    const double y = static_cast<double>(row) + 0.5;
    crossings.clear();
    for (std::size_t i = 0, j = px.size() - 1; i < px.size(); j = i++) {
      const bool spans = (py[i] <= y && py[j] > y) || (py[j] <= y && py[i] > y);
      if (spans) crossings.push_back(px[i] + (y - py[i]) / (py[j] - py[i]) * (px[j] - px[i]));
    }
    std::sort(crossings.begin(), crossings.end());
    for (std::size_t k = 0; k + 1 < crossings.size(); k += 2) {
      const int col_begin = static_cast<int>(std::ceil(crossings[k] - 0.5));
      const int col_end = static_cast<int>(std::floor(crossings[k + 1] - 0.5));
      for (int col = col_begin; col <= col_end; ++col) image.blend_pixel(col, row, color, opacity);
    }
  }
}

Image rasterize(const Plot& plot, Bounds bounds) {
  const Layout layout = compute_layout(plot, bounds);
  const double tick_px = plot.tick_font_px();
  Image image(plot.width(), plot.height(), plot.background_color());
  const RGB axis_color = plot.axes_color();

  if (!plot.axes_hidden()) {
  if (plot.grid_visible()) {
    for (const auto& tick : x_axis_ticks(plot, layout)) {
      if (!tick_inside(tick.pixel, layout.left, layout.left + layout.plot_w)) continue;
      draw_line(image, tick.pixel, layout.top, tick.pixel, layout.top + layout.plot_h, plot.major_grid_color(), 1.0);
    }
    for (const auto& tick : y_axis_ticks(plot, layout)) {
      if (!tick_inside(tick.pixel, layout.top, layout.top + layout.plot_h)) continue;
      draw_line(image, layout.left, tick.pixel, layout.left + layout.plot_w, tick.pixel, plot.major_grid_color(), 1.0);
    }
  }

  draw_line(image,
            layout.left,
            layout.top + layout.plot_h,
            layout.left + layout.plot_w,
            layout.top + layout.plot_h,
            axis_color,
            2.0);
  draw_line(image,
            layout.left,
            layout.top,
            layout.left,
            layout.top + layout.plot_h,
            axis_color,
            2.0);

  for (const auto& tick : x_axis_ticks(plot, layout)) {
    draw_text(image, tick.pixel, x_tick_baseline(layout, tick_px), tick.label, axis_color, tick_px,
              TextAnchor::Middle);
  }
  for (const auto& tick : y_axis_ticks(plot, layout)) {
    draw_text(image, layout.left - 10.0, tick.pixel + kTextCenterEm * tick_px, tick.label, axis_color, tick_px,
              TextAnchor::End);
  }
  }

  const double mid_x = static_cast<double>(plot.width()) / 2.0;
  draw_text(image, mid_x, 36.0, plot.title_text(), axis_color, 24.0, TextAnchor::Middle);
  draw_text(image, mid_x, 60.0, plot.subtitle_text(), axis_color, 16.0, TextAnchor::Middle);

  if (!plot.axes_hidden()) {
    draw_text(image, mid_x, static_cast<double>(plot.height()) - 20.0, plot.x_label_text(), axis_color, 16.0,
              TextAnchor::Middle);
    draw_text(image, 24.0, static_cast<double>(plot.height()) / 2.0, plot.y_label_text(), axis_color, 16.0,
              TextAnchor::Middle, true);
  }

  for (const auto& series : plot.series()) {
    if (series.kind == DataSeries::Kind::Line) {
      for (std::size_t i = 1; i < series.x.size(); ++i) {
        draw_line(image,
                  map_x(layout, series.x[i - 1]),
                  map_y(layout, series.y[i - 1]),
                  map_x(layout, series.x[i]),
                  map_y(layout, series.y[i]),
                  series.color,
                  series.stroke_width);
      }
    } else if (series.kind == DataSeries::Kind::Scatter) {
      for (std::size_t i = 0; i < series.x.size(); ++i) {
        draw_circle(image,
                    map_x(layout, series.x[i]),
                    map_y(layout, series.y[i]),
                    std::max(1.0, series.marker_size),
                    scatter_color(series, i));
      }
    } else if (series.kind == DataSeries::Kind::Bar) {
      for (std::size_t i = 0; i < series.x.size(); ++i) {
        const int x_left = static_cast<int>(
            std::lround(map_x(layout, series.x[i] - series.bar_width / 2.0)));
        const int x_right = static_cast<int>(
            std::lround(map_x(layout, series.x[i] + series.bar_width / 2.0)));
        const int y0 = static_cast<int>(std::lround(map_y(layout, 0.0)));
        const int y1 = static_cast<int>(std::lround(map_y(layout, series.y[i])));
        image.fill_rect(x_left, std::min(y0, y1), x_right, std::max(y0, y1), series.color);
      }
    } else if (series.kind == DataSeries::Kind::Box) {
      if (series.x.empty() || series.y.size() < 5) continue;
      const auto geo = box_geometry(series, layout);
      const RGB ink = darken(series.color, 0.35);
      RGB fill = series.color;
      fill.alpha = 0.55 * std::clamp(series.color.alpha, 0.0, 1.0);
      fill_polygon(image, {geo.left, geo.right, geo.right, geo.left}, {geo.q3, geo.q3, geo.q1, geo.q1}, fill);
      draw_line(image, geo.cx, geo.lo, geo.cx, geo.q1, ink, 1.5);
      draw_line(image, geo.cx, geo.q3, geo.cx, geo.hi, ink, 1.5);
      draw_line(image, geo.cap_left, geo.lo, geo.cap_right, geo.lo, ink, 1.5);
      draw_line(image, geo.cap_left, geo.hi, geo.cap_right, geo.hi, ink, 1.5);
      for (const auto& [x0, y0, x1, y1] : {std::array{geo.left, geo.q3, geo.right, geo.q3},
                                           std::array{geo.right, geo.q3, geo.right, geo.q1},
                                           std::array{geo.right, geo.q1, geo.left, geo.q1},
                                           std::array{geo.left, geo.q1, geo.left, geo.q3}}) {
        draw_line(image, x0, y0, x1, y1, ink, 1.5);
      }
      draw_line(image, geo.left, geo.median, geo.right, geo.median, ink, 2.5);
      for (const double y : geo.outliers) {
        draw_circle(image, geo.cx, y, 3.6, ink);
        draw_circle(image, geo.cx, y, 2.4, plot.background_color());
      }
    } else if (series.kind == DataSeries::Kind::Text) {
      if (series.x.empty()) continue;
      // Centered on the point both ways, like R's text() (SVG uses dy="0.35em" to match).
      draw_text(image, map_x(layout, series.x[0]), map_y(layout, series.y[0]) + kTextCenterEm * series.marker_size,
                series.label, series.color, series.marker_size, TextAnchor::Middle);
    } else if (series.kind == DataSeries::Kind::Polygon) {
      std::vector<double> px(series.x.size());
      std::vector<double> py(series.y.size());
      for (std::size_t i = 0; i < series.x.size(); ++i) {
        px[i] = map_x(layout, series.x[i]);
        py[i] = map_y(layout, series.y[i]);
      }
      if (series.filled) {
        fill_polygon(image, px, py, series.color);
      } else {
        for (std::size_t i = 0; i < px.size(); ++i) {
          const std::size_t next = (i + 1) % px.size();
          draw_line(image, px[i], py[i], px[next], py[next], series.color, series.stroke_width);
        }
      }
    } else if (series.kind == DataSeries::Kind::Segment) {
      for (std::size_t i = 0; i + 1 < series.x.size(); i += 2) {
        draw_line(image,
                  map_x(layout, series.x[i]),
                  map_y(layout, series.y[i]),
                  map_x(layout, series.x[i + 1]),
                  map_y(layout, series.y[i + 1]),
                  series.color,
                  series.stroke_width);
      }
    }
  }

  draw_legend(image, plot, layout);

  return image;
}

Image rasterize(const Plot& plot) { return rasterize(plot, compute_bounds(plot)); }


void write_be32(std::ostream& out, std::uint32_t value) {
  out.put(static_cast<char>((value >> 24U) & 0xFFU));
  out.put(static_cast<char>((value >> 16U) & 0xFFU));
  out.put(static_cast<char>((value >> 8U) & 0xFFU));
  out.put(static_cast<char>(value & 0xFFU));
}

std::uint32_t crc32_bytes(const std::vector<std::uint8_t>& data) {
  std::uint32_t crc = 0xFFFFFFFFU;
  for (std::uint8_t byte : data) {
    crc ^= byte;
    for (int k = 0; k < 8; ++k) {
      const std::uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (0xEDB88320U & mask);
    }
  }
  return ~crc;
}

void write_png_chunk(std::ostream& out, const char type[4], const std::vector<std::uint8_t>& data) {
  write_be32(out, static_cast<std::uint32_t>(data.size()));
  std::vector<std::uint8_t> crc_input;
  crc_input.reserve(4U + data.size());
  for (int i = 0; i < 4; ++i) {
    out.put(type[i]);
    crc_input.push_back(static_cast<std::uint8_t>(type[i]));
  }
  for (std::uint8_t byte : data) {
    out.put(static_cast<char>(byte));
    crc_input.push_back(byte);
  }
  write_be32(out, crc32_bytes(crc_input));
}

std::uint32_t adler32_bytes(const std::vector<std::uint8_t>& data) {
  return algorithms::adler32({reinterpret_cast<const char*>(data.data()), data.size()});
}

std::vector<std::uint8_t> zlib_store_compress(const std::vector<std::uint8_t>& raw) {
  std::vector<std::uint8_t> out;
  out.reserve(raw.size() + raw.size() / 65535U + 16U);
  out.push_back(0x78U);
  out.push_back(0x01U);
  std::size_t offset = 0;
  while (offset < raw.size()) {
    const std::size_t remaining = raw.size() - offset;
    const std::uint16_t block_len =
        static_cast<std::uint16_t>(std::min<std::size_t>(remaining, 65535U));
    const bool final_block = offset + block_len == raw.size();
    out.push_back(final_block ? 0x01U : 0x00U);
    out.push_back(static_cast<std::uint8_t>(block_len & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((block_len >> 8U) & 0xFFU));
    const std::uint16_t nlen = static_cast<std::uint16_t>(~block_len);
    out.push_back(static_cast<std::uint8_t>(nlen & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((nlen >> 8U) & 0xFFU));
    out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
               raw.begin() + static_cast<std::ptrdiff_t>(offset + block_len));
    offset += block_len;
  }
  const std::uint32_t adler = adler32_bytes(raw);
  out.push_back(static_cast<std::uint8_t>((adler >> 24U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((adler >> 16U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((adler >> 8U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>(adler & 0xFFU));
  return out;
}

void write_png_image(const Image& image, std::ostream& out) {
  const std::array<unsigned char, 8> signature = {0x89U, 'P', 'N', 'G', '\r', '\n', 0x1AU, '\n'};
  out.write(reinterpret_cast<const char*>(signature.data()),
            static_cast<std::streamsize>(signature.size()));

  std::vector<std::uint8_t> ihdr(13U, 0U);
  const auto width = static_cast<std::uint32_t>(image.width);
  const auto height = static_cast<std::uint32_t>(image.height);
  ihdr[0] = static_cast<std::uint8_t>((width >> 24U) & 0xFFU);
  ihdr[1] = static_cast<std::uint8_t>((width >> 16U) & 0xFFU);
  ihdr[2] = static_cast<std::uint8_t>((width >> 8U) & 0xFFU);
  ihdr[3] = static_cast<std::uint8_t>(width & 0xFFU);
  ihdr[4] = static_cast<std::uint8_t>((height >> 24U) & 0xFFU);
  ihdr[5] = static_cast<std::uint8_t>((height >> 16U) & 0xFFU);
  ihdr[6] = static_cast<std::uint8_t>((height >> 8U) & 0xFFU);
  ihdr[7] = static_cast<std::uint8_t>(height & 0xFFU);
  ihdr[8] = 8U;
  ihdr[9] = 2U;
  write_png_chunk(out, "IHDR", ihdr);

  std::vector<std::uint8_t> raw;
  raw.reserve(image.height * (1U + image.width * 3U));
  for (std::size_t y = 0; y < image.height; ++y) {
    raw.push_back(0U);
    const auto start = static_cast<std::ptrdiff_t>(y * image.width * 3U);
    raw.insert(raw.end(),
               image.pixels.begin() + start,
               image.pixels.begin() + start + static_cast<std::ptrdiff_t>(image.width * 3U));
  }
  write_png_chunk(out, "IDAT", zlib_store_compress(raw));
  write_png_chunk(out, "IEND", {});
}

std::string hex_encode(const std::vector<std::uint8_t>& bytes) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string out;
  out.reserve(bytes.size() * 2U + 1U);
  for (std::uint8_t byte : bytes) {
    out.push_back(kHex[(byte >> 4U) & 0x0FU]);
    out.push_back(kHex[byte & 0x0FU]);
  }
  out.push_back('>');
  return out;
}

void write_pdf_image(const Image& image, std::ostream& out) {
  const std::string image_hex = hex_encode(image.pixels);
  std::ostringstream content_stream;
  content_stream << "q\n"
                 << image.width << " 0 0 " << image.height << " 0 0 cm\n"
                 << "/Im0 Do\n"
                 << "Q\n";
  const std::string content = content_stream.str();

  std::vector<long long> offsets;
  offsets.push_back(0);
  out << "%PDF-1.4\n";

  auto begin_object = [&](int number) {
    offsets.push_back(static_cast<long long>(out.tellp()));
    out << number << " 0 obj\n";
  };

  begin_object(1);
  out << "<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";

  begin_object(2);
  out << "<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n";

  begin_object(3);
  out << "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " << image.width << " " << image.height
      << "] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n";

  begin_object(4);
  out << "<< /Type /XObject /Subtype /Image /Width " << image.width << " /Height " << image.height
      << " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /ASCIIHexDecode /Length "
      << image_hex.size() << " >>\nstream\n"
      << image_hex << "\nendstream\nendobj\n";

  begin_object(5);
  out << "<< /Length " << content.size() << " >>\nstream\n"
      << content << "endstream\nendobj\n";

  const long long xref_offset = static_cast<long long>(out.tellp());
  out << "xref\n0 " << offsets.size() << "\n";
  out << "0000000000 65535 f \n";
  for (std::size_t i = 1; i < offsets.size(); ++i) {
    out << std::setw(10) << std::setfill('0') << offsets[i] << " 00000 n \n";
  }
  out << "trailer\n<< /Size " << offsets.size() << " /Root 1 0 R >>\n";
  out << "startxref\n" << xref_offset << "\n%%EOF\n";
}

/// @brief Encodes @p image as PNG bytes in memory.
std::vector<std::uint8_t> encode_png(const Image& image) {
  std::ostringstream out(std::ios::binary);
  write_png_image(image, out);
  const std::string bytes = out.str();
  return {bytes.begin(), bytes.end()};
}

/// @brief Encodes @p image as a one-page PDF in memory.
std::vector<std::uint8_t> encode_pdf(const Image& image) {
  std::ostringstream out(std::ios::binary);
  write_pdf_image(image, out);
  const std::string bytes = out.str();
  return {bytes.begin(), bytes.end()};
}

/// @brief Writes @p bytes to @p path, replacing any existing file.
void write_bytes(const std::vector<std::uint8_t>& bytes, const std::string& path, const char* who) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error(std::string(who) + ": could not open output file " + path);
  }
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

/// @brief Writes @p text to @p path, replacing any existing file.
void write_text(const std::string& text, const std::string& path, const char* who) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(std::string(who) + ": could not open output file " + path);
  }
  out << text;
}

void save_png_image(const Image& image, const std::string& path) {
  write_bytes(encode_png(image), path, "Plot::save");
}

void save_pdf_image(const Image& image, const std::string& path) {
  write_bytes(encode_pdf(image), path, "Plot::save");
}

void blit_scaled_image(const Image& src,
                       Image&       dst,
                       int          canvas_x,
                       int          canvas_y,
                       int          canvas_w,
                       int          canvas_h,
                       double       image_zoom,
                       double       pan_x,
                       double       pan_y) {
  if (src.width == 0 || src.height == 0 || canvas_w <= 0 || canvas_h <= 0) {
    return;
  }

  const double fit_scale =
      std::min(static_cast<double>(canvas_w) / static_cast<double>(src.width),
               static_cast<double>(canvas_h) / static_cast<double>(src.height));
  const double scale = std::max(fit_scale * image_zoom, 1e-9);
  const double draw_w = static_cast<double>(src.width) * scale;
  const double draw_h = static_cast<double>(src.height) * scale;
  const double draw_x = canvas_x + (static_cast<double>(canvas_w) - draw_w) * 0.5 + pan_x;
  const double draw_y = canvas_y + (static_cast<double>(canvas_h) - draw_h) * 0.5 + pan_y;

  const int x_begin = std::max(canvas_x, static_cast<int>(std::floor(draw_x)));
  const int y_begin = std::max(canvas_y, static_cast<int>(std::floor(draw_y)));
  const int x_end = std::min(canvas_x + canvas_w, static_cast<int>(std::ceil(draw_x + draw_w)));
  const int y_end = std::min(canvas_y + canvas_h, static_cast<int>(std::ceil(draw_y + draw_h)));

  for (int y = y_begin; y < y_end; ++y) {
    for (int x = x_begin; x < x_end; ++x) {
      // Sample at pixel centres and blend the four neighboring source pixels.  Nearest-neighbor
      // sampling made the plot visibly blocky whenever the viewer was resized or zoomed.
      const double source_x = (static_cast<double>(x) + 0.5 - draw_x) / scale - 0.5;
      const double source_y = (static_cast<double>(y) + 0.5 - draw_y) / scale - 0.5;
      if (source_x < -0.5 || source_y < -0.5
          || source_x > static_cast<double>(src.width) - 0.5
          || source_y > static_cast<double>(src.height) - 0.5) {
        continue;
      }
      const int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, static_cast<int>(src.width) - 1);
      const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, static_cast<int>(src.height) - 1);
      const int x1 = std::min(x0 + 1, static_cast<int>(src.width) - 1);
      const int y1 = std::min(y0 + 1, static_cast<int>(src.height) - 1);
      const double tx = std::clamp(source_x - std::floor(source_x), 0.0, 1.0);
      const double ty = std::clamp(source_y - std::floor(source_y), 0.0, 1.0);
      const auto sample = [&](int sx, int sy, std::size_t channel) {
        return static_cast<double>(src.pixels[
            (static_cast<std::size_t>(sy) * src.width + static_cast<std::size_t>(sx)) * 3U + channel]);
      };
      const auto interpolate = [&](std::size_t channel) {
        const double top = sample(x0, y0, channel) * (1.0 - tx) + sample(x1, y0, channel) * tx;
        const double bottom = sample(x0, y1, channel) * (1.0 - tx) + sample(x1, y1, channel) * tx;
        return static_cast<int>(std::lround(top * (1.0 - ty) + bottom * ty));
      };
      dst.set_pixel(x,
                    y,
                    RGB{
                        interpolate(0U),
                        interpolate(1U),
                        interpolate(2U),
                    });
    }
  }
}

/// @brief Tiles already-rasterized panels row-major into one image, each scaled to fit its cell.
Image tile_panels(const std::vector<Image>& panels,
                  std::size_t               rows,
                  std::size_t               cols,
                  std::size_t               width,
                  std::size_t               height,
                  const std::string&        title) {
  Image     out(width, height, {255, 255, 255});
  const int band = title.empty() ? 0 : static_cast<int>(RLayout::title_band_px);
  if (band > 0) {
    draw_text(out, static_cast<double>(width) / 2.0, band * 0.7, title, {55, 65, 81}, 24.0, TextAnchor::Middle);
  }
  const int cell_w = static_cast<int>(width / cols);
  const int cell_h = (static_cast<int>(height) - band) / static_cast<int>(rows);
  for (std::size_t i = 0; i < panels.size(); ++i) {
    const int col = static_cast<int>(i % cols);
    const int row = static_cast<int>(i / cols);
    blit_scaled_image(panels[i], out, col * cell_w, band + row * cell_h, cell_w, cell_h, 1.0, 0.0, 0.0);
  }
  return out;
}

/// @brief Rasterizes every panel of @p layout and tiles them into one image.
Image rasterize_layout(const RLayout& layout) {
  std::vector<Image> images;
  images.reserve(layout.panel_count());
  for (std::size_t i = 0; i < layout.panel_count(); ++i) images.push_back(rasterize(layout.panel(i)));
  return tile_panels(images, layout.rows(), layout.cols(), layout.width(), layout.height(),
                     layout.title_text());
}

#if defined(DATAMUNGE_HAVE_X11)
unsigned long expand_component(unsigned value, unsigned long mask) {
  if (mask == 0UL) {
    return 0UL;
  }
  unsigned shift = 0;
  while (((mask >> shift) & 1UL) == 0UL) {
    ++shift;
  }
  unsigned bits = 0;
  while (((mask >> (shift + bits)) & 1UL) != 0UL) {
    ++bits;
  }
  const unsigned long max_mask = (1UL << bits) - 1UL;
  const unsigned long scaled = (static_cast<unsigned long>(value) * max_mask + 127UL) / 255UL;
  return (scaled << shift) & mask;
}

unsigned long rgb_to_native_pixel(const Visual* visual, RGB color) {
  return expand_component(static_cast<unsigned>(std::clamp(color.r, 0, 255)), visual->red_mask)
       | expand_component(static_cast<unsigned>(std::clamp(color.g, 0, 255)), visual->green_mask)
       | expand_component(static_cast<unsigned>(std::clamp(color.b, 0, 255)), visual->blue_mask);
}

struct ViewerButton {
  enum class Action {
    PlotZoomIn,
    PlotZoomOut,
    PlotLeft,
    PlotRight,
    PlotUp,
    PlotDown,
    PlotReset,
  };

  int         x{0};
  int         y{0};
  int         w{0};
  int         h{0};
  std::string label;
  Action      action{Action::PlotZoomIn};
};

void draw_button(Image& image, const ViewerButton& button) {
  const RGB border{55, 65, 81};
  const RGB fill{243, 244, 246};
  const RGB text{31, 41, 55};
  image.fill_rect(button.x, button.y, button.x + button.w, button.y + button.h, border);
  image.fill_rect(
      button.x + 1, button.y + 1, button.x + button.w - 1, button.y + button.h - 1, fill);
  draw_text(image, button.x + button.w / 2.0, button.y + button.h / 2.0 + kTextCenterEm * kButtonFontPx,
            button.label, text, kButtonFontPx, TextAnchor::Middle);
}

std::vector<ViewerButton> make_viewer_buttons() {
  constexpr int x0 = 10;
  constexpr int y0 = 8;
  constexpr int h = 26;
  constexpr int gap = 8;
  const std::vector<std::pair<std::string, ViewerButton::Action>> specs = {
      {"PLOT+", ViewerButton::Action::PlotZoomIn},
      {"PLOT-", ViewerButton::Action::PlotZoomOut},
      {"LEFT", ViewerButton::Action::PlotLeft},
      {"RIGHT", ViewerButton::Action::PlotRight},
      {"UP", ViewerButton::Action::PlotUp},
      {"DOWN", ViewerButton::Action::PlotDown},
      {"RESET", ViewerButton::Action::PlotReset},
  };

  std::vector<ViewerButton> buttons;
  buttons.reserve(specs.size());
  int cursor_x = x0;
  for (const auto& spec : specs) {
    const int width = 18 + static_cast<int>(std::ceil(text_width(spec.first, kButtonFontPx)));
    buttons.push_back(ViewerButton{cursor_x, y0, width, h, spec.first, spec.second});
    cursor_x += width + gap;
  }
  return buttons;
}

const ViewerButton* find_button_at(const std::vector<ViewerButton>& buttons, int x, int y) {
  for (const auto& button : buttons) {
    if (x >= button.x && x <= button.x + button.w && y >= button.y && y <= button.y + button.h) {
      return &button;
    }
  }
  return nullptr;
}

void copy_image_to_ximage(const Image& image, const Visual* visual, XImage* ximage) {
  for (int y = 0; y < static_cast<int>(image.height); ++y) {
    for (int x = 0; x < static_cast<int>(image.width); ++x) {
      const std::size_t idx =
          (static_cast<std::size_t>(y) * image.width + static_cast<std::size_t>(x)) * 3U;
      const RGB color{
          static_cast<int>(image.pixels[idx + 0U]),
          static_cast<int>(image.pixels[idx + 1U]),
          static_cast<int>(image.pixels[idx + 2U]),
      };
      XPutPixel(ximage, x, y, rgb_to_native_pixel(visual, color));
    }
  }
}

void zoom_bounds(Bounds& bounds, double factor) {
  const double cx = (bounds.x_min + bounds.x_max) * 0.5;
  const double cy = (bounds.y_min + bounds.y_max) * 0.5;
  const double half_width = std::max((bounds.x_max - bounds.x_min) * factor * 0.5, 1e-9);
  const double half_height = std::max((bounds.y_max - bounds.y_min) * factor * 0.5, 1e-9);
  bounds.x_min = cx - half_width;
  bounds.x_max = cx + half_width;
  bounds.y_min = cy - half_height;
  bounds.y_max = cy + half_height;
}

void pan_bounds(Bounds& bounds, double dx_fraction, double dy_fraction) {
  const double dx = (bounds.x_max - bounds.x_min) * dx_fraction;
  const double dy = (bounds.y_max - bounds.y_min) * dy_fraction;
  bounds.x_min += dx;
  bounds.x_max += dx;
  bounds.y_min += dy;
  bounds.y_max += dy;
}

Image render_viewer_frame(const Image&                     plot_image,
                          int                              window_width,
                          int                              window_height,
                          const std::vector<ViewerButton>& buttons,
                          double                           image_zoom,
                          double                           pan_x,
                          double                           pan_y) {
  constexpr int menu_height = 44;
  const int safe_width = std::max(window_width, 240);
  const int safe_height = std::max(window_height, 160);
  Image frame(
      static_cast<std::size_t>(safe_width), static_cast<std::size_t>(safe_height), {255, 255, 255});

  frame.fill_rect(0, 0, safe_width - 1, menu_height - 1, {229, 231, 235});
  frame.fill_rect(0, menu_height - 1, safe_width - 1, menu_height - 1, {209, 213, 219});
  for (const auto& button : buttons) {
    draw_button(frame, button);
  }

  const int canvas_y = menu_height;
  const int canvas_h = safe_height - menu_height;
  frame.fill_rect(0, canvas_y, safe_width - 1, safe_height - 1, {255, 255, 255});
  blit_scaled_image(plot_image, frame, 0, canvas_y, safe_width, canvas_h, image_zoom, pan_x, pan_y);
  return frame;
}

/// @brief What the interactive viewer needs from a figure: a renderer plus data-space zoom/pan.
struct ViewerModel {
  std::function<Image()>             render;
  std::function<void(double)>        zoom;  ///< scales every data range by the factor
  std::function<void(double, double)> pan;   ///< shifts every data range by fractions of itself
  std::function<void()>              reset; ///< restores the original data ranges
};

void run_x11_viewer(const ViewerModel& model, const std::string& title_hint) {
  constexpr int menu_height = 44;
  Image plot_image = model.render();
  Display* display = XOpenDisplay(nullptr);
  if (display == nullptr) {
    throw std::runtime_error("Plot::show: could not connect to the X11 display");
  }
  const int screen = DefaultScreen(display);
  Visual* visual = DefaultVisual(display, screen);
  const int depth = DefaultDepth(display, screen);
  if (visual == nullptr || depth < 15) {
    XCloseDisplay(display);
    throw std::runtime_error("Plot::show: unsupported X11 visual");
  }

  Window window = XCreateSimpleWindow(display,
                                      RootWindow(display, screen),
                                      10,
                                      10,
                                      static_cast<unsigned int>(plot_image.width),
                                      static_cast<unsigned int>(plot_image.height + menu_height),
                                      1,
                                      BlackPixel(display, screen),
                                      WhitePixel(display, screen));
  const std::string title = title_hint.empty() ? "datamunge plot" : title_hint;
  XStoreName(display, window, title.c_str());
  XSelectInput(display, window, ExposureMask | KeyPressMask | StructureNotifyMask | ButtonPressMask);
  Atom wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(display, window, &wm_delete, 1);
  XMapWindow(display, window);

  GC gc = XCreateGC(display, window, 0, nullptr);
  XImage* ximage = nullptr;
  std::unique_ptr<char[]> buffer;
  double image_zoom = 1.0;
  double image_pan_x = 0.0;
  double image_pan_y = 0.0;
  int window_width = static_cast<int>(plot_image.width);
  int window_height = static_cast<int>(plot_image.height + menu_height);
  const std::vector<ViewerButton> buttons = make_viewer_buttons();

  auto recreate_ximage = [&](int width, int height) {
    if (ximage != nullptr) {
      ximage->data = nullptr;
      XDestroyImage(ximage);
      ximage = nullptr;
      buffer.reset();
    }
    ximage = XCreateImage(display,
                          visual,
                          static_cast<unsigned int>(depth),
                          ZPixmap,
                          0,
                          nullptr,
                          static_cast<unsigned int>(width),
                          static_cast<unsigned int>(height),
                          32,
                          0);
    if (ximage == nullptr) {
      throw std::runtime_error("Plot::show: could not allocate X11 image");
    }
    const std::size_t buffer_size = static_cast<std::size_t>(ximage->bytes_per_line)
                                  * static_cast<std::size_t>(ximage->height);
    buffer = std::make_unique<char[]>(buffer_size);
    std::fill(buffer.get(), buffer.get() + static_cast<std::ptrdiff_t>(buffer_size), 0);
    ximage->data = buffer.get();
  };

  auto rerasterize_plot = [&]() { plot_image = model.render(); };

  auto redraw = [&]() {
    if (ximage == nullptr || static_cast<int>(ximage->width) != window_width
        || static_cast<int>(ximage->height) != window_height) {
      recreate_ximage(window_width, window_height);
    }
    const Image frame = render_viewer_frame(
        plot_image, window_width, window_height, buttons, image_zoom, image_pan_x, image_pan_y);
    copy_image_to_ximage(frame, visual, ximage);
    XPutImage(display,
              window,
              gc,
              ximage,
              0,
              0,
              0,
              0,
              static_cast<unsigned int>(window_width),
              static_cast<unsigned int>(window_height));
    XFlush(display);
  };

  rerasterize_plot();
  redraw();

  bool running = true;
  while (running) {
    XEvent event;
    XNextEvent(display, &event);
    if (event.type == Expose) {
      redraw();
    } else if (event.type == ConfigureNotify) {
      if (event.xconfigure.width > 0 && event.xconfigure.height > 0
          && (event.xconfigure.width != window_width || event.xconfigure.height != window_height)) {
        window_width = event.xconfigure.width;
        window_height = event.xconfigure.height;
        redraw();
      }
    } else if (event.type == KeyPress) {
      bool changed = false;
      const KeySym keysym = XLookupKeysym(&event.xkey, 0);
      const bool shift_pressed = (event.xkey.state & ShiftMask) != 0U;
      if (shift_pressed) {
        if (keysym == XK_minus || keysym == XK_KP_Subtract) {
          image_zoom /= 1.25;
          changed = true;
        } else if (keysym == XK_equal || keysym == XK_plus || keysym == XK_KP_Add) {
          image_zoom *= 1.25;
          changed = true;
        }
      } else if (keysym == XK_h || keysym == XK_H || keysym == XK_Left) {
        image_pan_x -= 40.0;
        changed = true;
      } else if (keysym == XK_l || keysym == XK_L || keysym == XK_Right) {
        image_pan_x += 40.0;
        changed = true;
      } else if (keysym == XK_j || keysym == XK_J || keysym == XK_Down) {
        image_pan_y += 40.0;
        changed = true;
      } else if (keysym == XK_k || keysym == XK_K || keysym == XK_Up) {
        image_pan_y -= 40.0;
        changed = true;
      } else if (keysym == XK_0) {
        image_zoom = 1.0;
        image_pan_x = 0.0;
        image_pan_y = 0.0;
        changed = true;
      } else if (keysym == XK_q || keysym == XK_Q || keysym == XK_Escape) {
        running = false;
      }

      if (changed) {
        redraw();
      }
    } else if (event.type == ButtonPress && event.xbutton.button == Button1) {
      const ViewerButton* button = find_button_at(buttons, event.xbutton.x, event.xbutton.y);
      if (button != nullptr) {
        switch (button->action) {
          case ViewerButton::Action::PlotZoomIn:
            model.zoom(0.8);
            break;
          case ViewerButton::Action::PlotZoomOut:
            model.zoom(1.25);
            break;
          case ViewerButton::Action::PlotLeft:
            model.pan(-0.1, 0.0);
            break;
          case ViewerButton::Action::PlotRight:
            model.pan(0.1, 0.0);
            break;
          case ViewerButton::Action::PlotUp:
            model.pan(0.0, 0.1);
            break;
          case ViewerButton::Action::PlotDown:
            model.pan(0.0, -0.1);
            break;
          case ViewerButton::Action::PlotReset:
            model.reset();
            break;
        }
        rerasterize_plot();
        redraw();
      }
    } else if (event.type == ClientMessage
               && static_cast<Atom>(event.xclient.data.l[0]) == wm_delete) {
      running = false;
    } else if (event.type == DestroyNotify) {
      running = false;
    }
  }

  if (ximage != nullptr) {
    ximage->data = nullptr;
    XDestroyImage(ximage);
  }
  XFreeGC(display, gc);
  // XCloseDisplay destroys client-owned resources, including `window`.  Do not issue an
  // additional XDestroyWindow request here: a window manager may destroy the window while
  // processing its close request, leaving a race that produces an asynchronous BadWindow
  // error and causes Xlib's default error handler to terminate the process.
  XCloseDisplay(display);
}

void show_x11_plot(const Plot& plot, const std::string& title_hint) {
  Bounds       current = compute_bounds(plot);
  const Bounds original = current;
  const ViewerModel model{
      [&]() { return rasterize(plot, current); },
      [&](double factor) { zoom_bounds(current, factor); },
      [&](double dx, double dy) { pan_bounds(current, dx, dy); },
      [&]() { current = original; },
  };
  run_x11_viewer(model, title_hint);
}

#endif

} // namespace

Plot& Plot::size(std::size_t width, std::size_t height) {
  width_ = width;
  height_ = height;
  return *this;
}

Plot& Plot::title(std::string value) {
  title_ = std::move(value);
  return *this;
}

Plot& Plot::subtitle(std::string value) {
  subtitle_ = std::move(value);
  return *this;
}

Plot& Plot::x_label(std::string value) {
  x_label_ = std::move(value);
  return *this;
}

Plot& Plot::y_label(std::string value) {
  y_label_ = std::move(value);
  return *this;
}

Plot& Plot::background(RGB color) {
  background_ = color;
  return *this;
}

Plot& Plot::axis_color(RGB color) {
  axis_color_ = color;
  return *this;
}

Plot& Plot::grid_color(RGB color) {
  grid_color_ = color;
  return *this;
}

Plot& Plot::show_grid(bool enabled) {
  show_grid_ = enabled;
  return *this;
}

Plot& Plot::x_limits(double min_x, double max_x) {
  if (!(max_x > min_x)) {
    throw std::invalid_argument("Plot::x_limits: require max_x > min_x");
  }
  has_x_limits_ = true;
  x_min_ = min_x;
  x_max_ = max_x;
  return *this;
}

Plot& Plot::y_limits(double min_y, double max_y) {
  if (!(max_y > min_y)) {
    throw std::invalid_argument("Plot::y_limits: require max_y > min_y");
  }
  has_y_limits_ = true;
  y_min_ = min_y;
  y_max_ = max_y;
  return *this;
}

Plot& Plot::hide_axes(bool enabled) {
  hide_axes_ = enabled;
  return *this;
}

Plot& Plot::x_tick_labels(std::vector<std::string> labels) {
  x_tick_labels_ = std::move(labels);
  x_tick_positions_.clear();
  return *this;
}

Plot& Plot::x_ticks(std::vector<double> positions, std::vector<std::string> labels) {
  if (positions.size() != labels.size()) {
    throw std::invalid_argument("Plot::x_ticks: positions and labels must have the same length");
  }
  x_tick_positions_ = std::move(positions);
  x_tick_labels_ = std::move(labels);
  return *this;
}

Plot& Plot::y_tick_labels(std::vector<std::string> labels) {
  y_tick_labels_ = std::move(labels);
  y_tick_positions_.clear();
  return *this;
}

Plot& Plot::y_ticks(std::vector<double> positions, std::vector<std::string> labels) {
  if (positions.size() != labels.size()) {
    throw std::invalid_argument("Plot::y_ticks: positions and labels must have the same length");
  }
  y_tick_positions_ = std::move(positions);
  y_tick_labels_ = std::move(labels);
  return *this;
}

Plot& Plot::margins(double left, double right, double top, double bottom) {
  for (double margin : {left, right, top, bottom}) {
    if (!std::isfinite(margin) || margin < 0.0) {
      throw std::invalid_argument("Plot::margins: margins must be finite and non-negative");
    }
  }
  has_margins_ = true;
  margin_left_ = left;
  margin_right_ = right;
  margin_top_ = top;
  margin_bottom_ = bottom;
  return *this;
}

Plot& Plot::pretty_ticks(bool enabled) {
  pretty_ticks_ = enabled;
  return *this;
}

Plot& Plot::tick_font_size(double pixels) {
  if (!std::isfinite(pixels) || pixels <= 0.0) {
    throw std::invalid_argument("Plot::tick_font_size: size must be positive and finite");
  }
  tick_font_size_ = pixels;
  return *this;
}

std::string Plot::to_svg() const {
  return svg_string(*this);
}

void Plot::save_svg(const std::string& path) const {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Plot::save_svg: could not open output file");
  }
  out << to_svg();
}

void Plot::save(const std::string& path) const {
  const auto ext = std::filesystem::path(path).extension().string();
  if (ext == ".svg" || ext.empty()) {
    save_svg(path);
    return;
  }
  if (ext == ".png") {
    save_png(path);
    return;
  }
  if (ext == ".pdf") {
    save_pdf(path);
    return;
  }

  throw std::invalid_argument("Plot::save: unsupported extension");
}

std::vector<std::uint8_t> Plot::to_png() const { return encode_png(rasterize(*this)); }

std::vector<std::uint8_t> Plot::to_pdf() const { return encode_pdf(rasterize(*this)); }

void Plot::save_png(const std::string& path) const { write_bytes(to_png(), path, "Plot::save_png"); }

void Plot::save_pdf(const std::string& path) const { write_bytes(to_pdf(), path, "Plot::save_pdf"); }

void Plot::view(const std::string& title_hint) const {
  show(title_hint);
}

void Plot::show(const std::string& title_hint) const {
#if defined(DATAMUNGE_HAVE_X11)
  // Like RLayout::show and Splom::show: an empty hint names the window after the plot.
  show_x11_plot(*this, title_hint.empty() ? title_ : title_hint);
#else
  (void)title_hint;
  throw std::runtime_error("Plot::show: X11 support was not enabled at build time");
#endif
}

DataSeries& Plot::add_series(DataSeries series) {
  series_.push_back(std::move(series));
  return series_.back();
}

Plot& Plot::add_reference_line(ABLine line) {
  ablines_.push_back(line);
  return *this;
}

Plot& Plot::add_legend_entry(LegendEntry entry) {
  legend_entries_.push_back(std::move(entry));
  return *this;
}

RPlot::RPlot(std::vector<double> x, std::vector<double> y)
    : RPlot((require_xy_same_size(x, y, "Plot(x, y)"), RPlot::plot(std::move(x), std::move(y)))) {}

RPlot RPlot::create() {
  return RPlot{};
}

RPlot& RPlot::size(std::size_t width, std::size_t height) {
  Plot::size(width, height);
  return *this;
}

RPlot& RPlot::title(std::string value) {
  Plot::title(std::move(value));
  return *this;
}

RPlot& RPlot::subtitle(std::string value) {
  Plot::subtitle(std::move(value));
  return *this;
}

RPlot& RPlot::x_label(std::string value) {
  Plot::x_label(std::move(value));
  return *this;
}

RPlot& RPlot::y_label(std::string value) {
  Plot::y_label(std::move(value));
  return *this;
}

RPlot& RPlot::background(RGB color) {
  Plot::background(color);
  return *this;
}

RPlot& RPlot::axis_color(RGB color) {
  Plot::axis_color(color);
  return *this;
}

RPlot& RPlot::grid_color(RGB color) {
  Plot::grid_color(color);
  return *this;
}

RPlot& RPlot::show_grid(bool enabled) {
  Plot::show_grid(enabled);
  return *this;
}

RPlot& RPlot::x_limits(double min_x, double max_x) {
  Plot::x_limits(min_x, max_x);
  return *this;
}

RPlot& RPlot::y_limits(double min_y, double max_y) {
  Plot::y_limits(min_y, max_y);
  return *this;
}

RPlot& RPlot::hide_axes(bool enabled) {
  Plot::hide_axes(enabled);
  return *this;
}

RPlot& RPlot::x_tick_labels(std::vector<std::string> labels) {
  Plot::x_tick_labels(std::move(labels));
  return *this;
}

RPlot& RPlot::x_ticks(std::vector<double> positions, std::vector<std::string> labels) {
  Plot::x_ticks(std::move(positions), std::move(labels));
  return *this;
}

RPlot& RPlot::y_tick_labels(std::vector<std::string> labels) {
  Plot::y_tick_labels(std::move(labels));
  return *this;
}

RPlot& RPlot::y_ticks(std::vector<double> positions, std::vector<std::string> labels) {
  Plot::y_ticks(std::move(positions), std::move(labels));
  return *this;
}

RPlot& RPlot::margins(double left, double right, double top, double bottom) {
  Plot::margins(left, right, top, bottom);
  return *this;
}

RPlot& RPlot::tick_font_size(double pixels) {
  Plot::tick_font_size(pixels);
  return *this;
}

RPlot& RPlot::pretty_ticks(bool enabled) {
  Plot::pretty_ticks(enabled);
  return *this;
}

RPlot& RPlot::xlab(std::string value) { return x_label(std::move(value)); }

RPlot& RPlot::ylab(std::string value) { return y_label(std::move(value)); }

RPlot& RPlot::set_title(std::string value) { return title(std::move(value)); }

RPlot& RPlot::set_subtitle(std::string value) { return subtitle(std::move(value)); }

RPlot& RPlot::set_xlab(std::string value) { return xlab(std::move(value)); }

RPlot& RPlot::set_ylab(std::string value) { return ylab(std::move(value)); }

RPlot& RPlot::set_col(RGB color) {
  for (auto& series : series_) {
    series.color = color;
    series.point_colors.clear();
  }
  return *this;
}

RPlot& RPlot::set_col(std::string color) {
  std::transform(color.begin(), color.end(), color.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  const std::array<std::pair<std::string_view, RGB>, 11> named{{
      {"black", {0, 0, 0}}, {"white", {255, 255, 255}}, {"red", {255, 0, 0}},
      {"green", {0, 128, 0}}, {"blue", {0, 0, 255}}, {"yellow", {255, 255, 0}},
      {"cyan", {0, 255, 255}}, {"magenta", {255, 0, 255}}, {"gray", {128, 128, 128}},
      {"grey", {128, 128, 128}}, {"transparent", {0, 0, 0, 0.0}}}};
  for (const auto& entry : named) {
    if (color == entry.first) return set_col(entry.second);
  }
  if ((color.size() == 7 || color.size() == 9) && color.front() == '#') {
    std::uint32_t value = 0;
    const auto parsed = std::from_chars(color.data() + 1, color.data() + color.size(), value, 16);
    if (parsed.ec == std::errc{} && parsed.ptr == color.data() + color.size()) {
      const double alpha = color.size() == 9 ? (value & 0xff) / 255.0 : 1.0;
      if (color.size() == 9) value >>= 8;
      return set_col(RGB{static_cast<int>((value >> 16) & 0xff),
                         static_cast<int>((value >> 8) & 0xff),
                         static_cast<int>(value & 0xff), alpha});
    }
  }
  throw std::invalid_argument("RPlot::set_col: unknown color " + color);
}

RPlot RPlot::plot(std::vector<double> x, std::vector<double> y, std::string type, std::string label, RGB color) {
  require_xy_same_size(x, y, "RPlot::plot");
  RPlot p;
  if (type == "p") {
    p.points(x, y, std::move(label), color);
  } else if (type == "l") {
    p.line(x, y, std::move(label), color);
  } else if (type == "b") {
    p.line(x, y, std::move(label), color);
    p.points(std::move(x), std::move(y), "", color);
  } else {
    throw std::invalid_argument("RPlot::plot: type must be one of \"p\", \"l\", \"b\"");
  }
  return p;
}

RPlot RPlot::hist(std::vector<double> data, std::size_t bins, std::string label, RGB color) {
  if (data.empty()) {
    throw std::invalid_argument("RPlot::hist: data must not be empty");
  }
  if (bins == 0) {
    throw std::invalid_argument("RPlot::hist: bins must be positive");
  }
  if (!std::all_of(data.begin(), data.end(), [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument("RPlot::hist: data must be finite");
  }
  double lo = *std::min_element(data.begin(), data.end());
  double hi = *std::max_element(data.begin(), data.end());
  if (hi <= lo) {
    hi = lo + 1.0;
  }
  const double bin_width = (hi - lo) / static_cast<double>(bins);
  if (!std::isfinite(hi - lo) || !std::isfinite(bin_width) || bin_width <= 0.0) {
    throw std::invalid_argument("RPlot::hist: bin range must have finite positive width");
  }

  std::vector<double> counts(bins, 0.0);
  for (double value : data) {
    const double position = (value - lo) / bin_width;
    const auto idx = position >= static_cast<double>(bins - 1)
        ? bins - 1 : static_cast<std::size_t>(position);
    counts[idx] += 1.0;
  }

  std::vector<double> centers(bins);
  for (std::size_t i = 0; i < bins; ++i) {
    centers[i] = lo + (static_cast<double>(i) + 0.5) * bin_width;
  }

  RPlot p;
  DataSeries series;
  series.kind = DataSeries::Kind::Bar;
  series.x = std::move(centers);
  series.y = std::move(counts);
  series.label = std::move(label);
  series.color = color;
  series.bar_width = bin_width;
  p.add_series(std::move(series));
  p.title("Histogram").x_label("x").y_label("Frequency");
  return p;
}

RPlot RPlot::barplot(std::vector<double> heights, std::vector<std::string> names, std::string label, RGB color) {
  if (heights.empty()) {
    throw std::invalid_argument("RPlot::barplot: heights must not be empty");
  }
  if (!names.empty() && names.size() != heights.size()) {
    throw std::invalid_argument("RPlot::barplot: names must match heights in size");
  }
  std::vector<double> positions(heights.size());
  for (std::size_t i = 0; i < heights.size(); ++i) {
    positions[i] = static_cast<double>(i);
  }

  RPlot p;
  DataSeries series;
  series.kind = DataSeries::Kind::Bar;
  series.x = std::move(positions);
  series.y = std::move(heights);
  series.label = std::move(label);
  series.color = color;
  series.bar_width = 0.8;
  p.add_series(std::move(series));
  if (!names.empty()) {
    p.x_tick_labels(std::move(names));
  }
  return p;
}

RPlot RPlot::boxplot(std::vector<std::vector<double>> groups, std::vector<std::string> names, RGB color) {
  if (groups.empty()) {
    throw std::invalid_argument("RPlot::boxplot: groups must not be empty");
  }
  if (!names.empty() && names.size() != groups.size()) {
    throw std::invalid_argument("RPlot::boxplot: names must match groups in size");
  }

  RPlot p;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (groups[g].empty()) {
      throw std::invalid_argument("RPlot::boxplot: group must not be empty");
    }
    std::vector<double> sorted = groups[g];
    std::sort(sorted.begin(), sorted.end());
    const double q1 = quantile_type7(sorted, 0.25);
    const double median = quantile_type7(sorted, 0.5);
    const double q3 = quantile_type7(sorted, 0.75);
    const double iqr = q3 - q1;
    const double lower_fence = q1 - 1.5 * iqr;
    const double upper_fence = q3 + 1.5 * iqr;

    double whisker_lo = sorted.front();
    double whisker_hi = sorted.back();
    std::vector<double> outliers;
    for (double value : sorted) {
      if (value < lower_fence || value > upper_fence) {
        outliers.push_back(value);
      }
    }
    for (double value : sorted) {
      if (value >= lower_fence) {
        whisker_lo = value;
        break;
      }
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
      if (*it <= upper_fence) {
        whisker_hi = *it;
        break;
      }
    }

    DataSeries series;
    series.kind = DataSeries::Kind::Box;
    series.x = {static_cast<double>(g)};
    series.y = {whisker_lo, q1, median, q3, whisker_hi};
    series.y.insert(series.y.end(), outliers.begin(), outliers.end());
    series.color = color;
    series.bar_width = 0.6;
    p.add_series(std::move(series));
  }
  if (!names.empty()) {
    p.x_tick_labels(std::move(names));
  }
  return p;
}

RPlot RPlot::boxplot(std::vector<std::vector<double>> groups, std::vector<std::string> names, std::vector<RGB> colors) {
  if (colors.empty()) {
    throw std::invalid_argument("RPlot::boxplot: colors must not be empty");
  }
  const auto count = groups.size();
  RPlot p = boxplot(std::move(groups), std::move(names), colors.front());
  for (std::size_t g = 0; g < count; ++g) {
    p.series_[g].color = colors[g % colors.size()];
  }
  return p;
}

RPlot RPlot::pie(std::vector<double> values, std::vector<std::string> names, std::vector<RGB> colors) {
  if (values.empty()) {
    throw std::invalid_argument("RPlot::pie: values must not be empty");
  }
  double total = 0.0;
  for (double v : values) {
    if (v < 0.0) {
      throw std::invalid_argument("RPlot::pie: values must be non-negative");
    }
    total += v;
  }
  if (total <= 0.0) {
    throw std::invalid_argument("RPlot::pie: values must sum to a positive total");
  }
  if (!names.empty() && names.size() != values.size()) {
    throw std::invalid_argument("RPlot::pie: names must match values in size");
  }

  constexpr double kPi = 3.14159265358979323846;
  constexpr int    kArcSegments = 24;

  RPlot p;
  double angle = kPi / 2.0; // start at 12 o'clock
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double sweep = (values[i] / total) * 2.0 * kPi;
    const double end_angle = angle - sweep; // sweep clockwise
    const RGB    color = colors.empty() ? kDefaultPalette[i % kDefaultPalette.size()] : colors[i % colors.size()];

    DataSeries series;
    series.kind = DataSeries::Kind::Polygon;
    series.filled = true;
    series.color = color;
    series.x.push_back(0.0);
    series.y.push_back(0.0);
    for (int s = 0; s <= kArcSegments; ++s) {
      const double t = angle + (end_angle - angle) * static_cast<double>(s) / static_cast<double>(kArcSegments);
      series.x.push_back(std::cos(t));
      series.y.push_back(std::sin(t));
    }
    if (!names.empty()) {
      series.label = names[i];
    }
    p.add_series(std::move(series));

    const double mid = (angle + end_angle) / 2.0;
    if (!names.empty()) {
      p.text(1.15 * std::cos(mid), 1.15 * std::sin(mid), names[i], RGB{17, 24, 39}, 13.0);
    }
    angle = end_angle;
  }

  p.hide_axes(true).show_grid(false).x_limits(-1.4, 1.4).y_limits(-1.4, 1.4);
  return p;
}

RPlot RPlot::curve(datamunge::Callback& f, double from, double to, std::size_t n, std::string label, RGB color) {
  if (n < 2) {
    throw std::invalid_argument("RPlot::curve: n must be at least 2");
  }
  if (!(to > from)) {
    throw std::invalid_argument("RPlot::curve: require to > from");
  }
  std::vector<double> x(n);
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = from + (to - from) * static_cast<double>(i) / static_cast<double>(n - 1);
    y[i] = f.call(x[i]);
  }
  RPlot p;
  p.line(std::move(x), std::move(y), std::move(label), color);
  return p;
}

RPlot RPlot::qqnorm(std::vector<double> data, std::string label, RGB color) {
  if (data.size() < 2) {
    throw std::invalid_argument("RPlot::qqnorm: data must have at least two values");
  }
  std::vector<double> sorted = std::move(data);
  std::sort(sorted.begin(), sorted.end());
  const std::size_t   n = sorted.size();
  std::vector<double> theoretical(n);
  for (std::size_t i = 0; i < n; ++i) {
    theoretical[i] = random::normal_quantile((static_cast<double>(i) + 0.5) / static_cast<double>(n));
  }
  RPlot p;
  p.points(std::move(theoretical), std::move(sorted), std::move(label), color);
  p.title("Normal Q-Q").x_label("Theoretical Quantiles").y_label("Sample Quantiles");
  return p;
}

RPlot& RPlot::points(std::vector<double> x, std::vector<double> y, std::string label, RGB color, double marker_size) {
  require_xy_same_size(x, y, "RPlot::points");
  DataSeries series;
  series.kind = DataSeries::Kind::Scatter;
  series.x = std::move(x);
  series.y = std::move(y);
  series.label = std::move(label);
  series.color = color;
  series.marker_size = marker_size;
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::points_colored(std::vector<double> x, std::vector<double> y,
                             std::vector<RGB> colors, double marker_size) {
  require_xy_same_size(x, y, "RPlot::points_colored");
  if (colors.empty()) throw std::invalid_argument("RPlot::points_colored: empty palette");
  DataSeries series;
  series.kind = DataSeries::Kind::Scatter;
  series.x = std::move(x);
  series.y = std::move(y);
  series.color = colors.front();
  series.point_colors = std::move(colors);
  series.marker_size = marker_size;
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::line(std::vector<double> x, std::vector<double> y, std::string label, RGB color, double stroke_width) {
  require_xy_same_size(x, y, "RPlot::line");
  DataSeries series;
  series.kind = DataSeries::Kind::Line;
  series.x = std::move(x);
  series.y = std::move(y);
  series.label = std::move(label);
  series.color = color;
  series.stroke_width = stroke_width;
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::lines(std::vector<double> x, std::vector<double> y, std::string label, RGB color, double stroke_width) {
  return line(std::move(x), std::move(y), std::move(label), color, stroke_width);
}

RPlot& RPlot::points(std::vector<double> x, std::vector<double> y, RGB color, double marker_size) {
  return points(std::move(x), std::move(y), "", color, marker_size);
}

RPlot& RPlot::lines(std::vector<double> x, std::vector<double> y, RGB color, double stroke_width) {
  return line(std::move(x), std::move(y), "", color, stroke_width);
}

RPlot& RPlot::bars(std::vector<double> x, std::vector<double> y, std::string label, RGB color, double bar_width) {
  require_xy_same_size(x, y, "RPlot::bars");
  DataSeries series;
  series.kind = DataSeries::Kind::Bar;
  series.x = std::move(x);
  series.y = std::move(y);
  series.label = std::move(label);
  series.color = color;
  series.bar_width = bar_width;
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::box(double position, double whisker_lo, double q1, double median, double q3, double whisker_hi,
                   std::vector<double> outliers, RGB color, double width) {
  DataSeries series;
  series.kind = DataSeries::Kind::Box;
  series.x = {position};
  series.y = {whisker_lo, q1, median, q3, whisker_hi};
  series.y.insert(series.y.end(), outliers.begin(), outliers.end());
  series.color = color;
  series.bar_width = width;
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::abline(double intercept, double slope, RGB color, double stroke_width) {
  ABLine line;
  line.vertical = false;
  line.value = intercept;
  line.slope = slope;
  line.color = color;
  line.stroke_width = stroke_width;
  add_reference_line(line);
  return *this;
}

RPlot& RPlot::abline_h(double y_value, RGB color, double stroke_width) {
  return abline(y_value, 0.0, color, stroke_width);
}

RPlot& RPlot::abline_v(double x_value, RGB color, double stroke_width) {
  ABLine line;
  line.vertical = true;
  line.value = x_value;
  line.color = color;
  line.stroke_width = stroke_width;
  add_reference_line(line);
  return *this;
}

RPlot& RPlot::qqline(std::vector<double> data, RGB color, double stroke_width) {
  if (data.size() < 2) {
    throw std::invalid_argument("RPlot::qqline: data must have at least two values");
  }
  std::sort(data.begin(), data.end());
  const double q1_sample = quantile_type7(data, 0.25);
  const double q3_sample = quantile_type7(data, 0.75);
  const double q1_theoretical = random::normal_quantile(0.25);
  const double q3_theoretical = random::normal_quantile(0.75);
  const double slope = (q3_sample - q1_sample) / (q3_theoretical - q1_theoretical);
  const double intercept = q1_sample - slope * q1_theoretical;
  return abline(intercept, slope, color, stroke_width);
}

RPlot& RPlot::legend(std::vector<std::string> labels, std::vector<RGB> colors) {
  if (labels.size() != colors.size()) {
    throw std::invalid_argument("RPlot::legend: labels and colors must have the same size");
  }
  for (std::size_t i = 0; i < labels.size(); ++i) {
    add_legend_entry(LegendEntry{std::move(labels[i]), colors[i]});
  }
  return *this;
}

RPlot& RPlot::text(double x, double y, std::string label, RGB color, double font_size) {
  DataSeries series;
  series.kind = DataSeries::Kind::Text;
  series.x = {x};
  series.y = {y};
  series.label = std::move(label);
  series.color = color;
  series.marker_size = font_size;
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::polygon(std::vector<double> x, std::vector<double> y, RGB color, bool filled) {
  require_xy_same_size(x, y, "RPlot::polygon");
  DataSeries series;
  series.kind = DataSeries::Kind::Polygon;
  series.x = std::move(x);
  series.y = std::move(y);
  series.color = color;
  series.filled = filled;
  add_series(std::move(series));
  return *this;
}

RPlot& RPlot::image(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<std::vector<double>>& z, std::vector<RGB> palette, double z_min,
                    double z_max) {
  if (x.size() < 2 || y.size() < 2 || z.size() != x.size()) {
    throw std::invalid_argument("RPlot::image: need at least 2 x and y centres and one row of z per x");
  }
  for (const auto& row : z) {
    if (row.size() != y.size()) throw std::invalid_argument("RPlot::image: each row of z needs one value per y");
  }
  for (std::size_t i = 1; i < x.size(); ++i) {
    if (!(x[i] > x[i - 1])) throw std::invalid_argument("RPlot::image: x must be increasing");
  }
  for (std::size_t j = 1; j < y.size(); ++j) {
    if (!(y[j] > y[j - 1])) throw std::invalid_argument("RPlot::image: y must be increasing");
  }
  if (palette.empty()) palette = terrain_colors(12);
  double lo = std::numeric_limits<double>::infinity(), hi = -lo;
  for (const auto& row : z) {
    for (const double v : row) {
      if (std::isnan(v)) continue;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  }
  if (!std::isnan(z_min)) lo = z_min;
  if (!std::isnan(z_max)) hi = z_max;
  // Cell edges: midpoints between centres, a half step beyond the ends.
  auto edges = [](const std::vector<double>& c) {
    std::vector<double> e(c.size() + 1);
    e.front() = c.front() - (c[1] - c[0]) / 2.0;
    e.back() = c.back() + (c[c.size() - 1] - c[c.size() - 2]) / 2.0;
    for (std::size_t k = 1; k < c.size(); ++k) e[k] = (c[k - 1] + c[k]) / 2.0;
    return e;
  };
  const auto ex = edges(x), ey = edges(y);
  const double span = hi > lo ? hi - lo : 1.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    for (std::size_t j = 0; j < y.size(); ++j) {
      const double v = z[i][j];
      if (std::isnan(v)) continue;
      const double t = std::clamp((v - lo) / span, 0.0, 1.0);
      const auto k = std::min(palette.size() - 1, static_cast<std::size_t>(t * static_cast<double>(palette.size())));
      DataSeries series;
      series.kind = DataSeries::Kind::Polygon;
      series.filled = true;
      series.color = palette[k];
      series.x = {ex[i], ex[i + 1], ex[i + 1], ex[i]};
      series.y = {ey[j], ey[j], ey[j + 1], ey[j + 1]};
      add_series(std::move(series));
    }
  }
  return *this;
}

std::vector<RGB> terrain_colors(std::size_t n) {
  if (n == 0) throw std::invalid_argument("terrain_colors: n must be at least 1");
  // R: k = n %/% 2; hue 4/12 -> 2/12 -> 0, saturation 1 -> 1 -> 0, value 0.65 -> 0.9 -> 0.95.
  auto hsv = [](double h, double s, double v) {
    const double c = v * s, hp = h * 6.0, xx = c * (1.0 - std::abs(std::fmod(hp, 2.0) - 1.0)), m = v - c;
    double r = 0, g = 0, b = 0;
    if (hp < 1) r = c, g = xx;
    else if (hp < 2) r = xx, g = c;
    else if (hp < 3) g = c, b = xx;
    else if (hp < 4) g = xx, b = c;
    else if (hp < 5) r = xx, b = c;
    else r = c, b = xx;
    auto to8 = [](double u) { return static_cast<std::uint8_t>(std::lround(255.0 * u)); };
    return RGB{to8(r + m), to8(g + m), to8(b + m)};
  };
  auto seq = [](double a, double b, std::size_t len, std::size_t i) {
    return len <= 1 ? a : a + (b - a) * static_cast<double>(i) / static_cast<double>(len - 1);
  };
  std::vector<RGB> out;
  const std::size_t k = n / 2;
  for (std::size_t i = 0; i < k; ++i)
    out.push_back(hsv(seq(4.0 / 12, 2.0 / 12, k, i), 1.0, seq(0.65, 0.9, k, i)));
  const std::size_t rest = n - k;
  for (std::size_t i = 1; i <= rest; ++i)
    out.push_back(hsv(seq(2.0 / 12, 0.0, rest + 1, i), seq(1.0, 0.0, rest + 1, i), seq(0.9, 0.95, rest + 1, i)));
  return out;
}

std::vector<RGB> diverging_colors(std::size_t n) {
  if (n == 0) throw std::invalid_argument("diverging_colors: n must be at least 1");
  static constexpr std::array<std::array<double, 3>, 7> anchors{{{33, 102, 172},
                                                                 {103, 169, 207},
                                                                 {209, 229, 240},
                                                                 {247, 247, 247},
                                                                 {253, 219, 199},
                                                                 {239, 138, 98},
                                                                 {178, 24, 43}}};
  std::vector<RGB> out;
  for (std::size_t i = 0; i < n; ++i) {
    const double      t = n == 1 ? 0.5 : static_cast<double>(i) / static_cast<double>(n - 1);
    const double      u = t * 6.0;
    const std::size_t k = std::min<std::size_t>(5, static_cast<std::size_t>(u));
    const double      w = u - static_cast<double>(k);
    auto mix = [&](std::size_t c) {
      return static_cast<int>(std::lround((1.0 - w) * anchors[k][c] + w * anchors[k + 1][c]));
    };
    out.push_back(RGB{mix(0), mix(1), mix(2)});
  }
  return out;
}

std::vector<RGB> viridis_colors(std::size_t n) {
  if (n == 0) throw std::invalid_argument("viridis_colors: n must be at least 1");
  // viridisLite::viridis(10), interpolated linearly.
  static constexpr std::array<std::array<double, 3>, 10> anchors{{{68, 1, 84},
                                                                  {72, 40, 120},
                                                                  {62, 74, 137},
                                                                  {49, 104, 142},
                                                                  {38, 130, 142},
                                                                  {31, 158, 137},
                                                                  {53, 183, 121},
                                                                  {109, 205, 89},
                                                                  {180, 222, 44},
                                                                  {253, 231, 37}}};
  std::vector<RGB> out;
  for (std::size_t i = 0; i < n; ++i) {
    const double      t = n == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n - 1);
    const double      u = t * 9.0;
    const std::size_t k = std::min<std::size_t>(8, static_cast<std::size_t>(u));
    const double      w = u - static_cast<double>(k);
    auto mix = [&](std::size_t c) {
      return static_cast<int>(std::lround((1.0 - w) * anchors[k][c] + w * anchors[k + 1][c]));
    };
    out.push_back(RGB{mix(0), mix(1), mix(2)});
  }
  return out;
}

std::vector<RGB> value_colors(const std::vector<double>& values, std::vector<RGB> palette, double z_min,
                              double z_max) {
  if (palette.empty()) palette = viridis_colors(256);
  double lo = z_min, hi = z_max;
  if (std::isnan(lo) || std::isnan(hi)) {
    double data_lo = std::numeric_limits<double>::infinity(), data_hi = -data_lo;
    for (const double v : values) {
      if (std::isnan(v)) continue;
      data_lo = std::min(data_lo, v);
      data_hi = std::max(data_hi, v);
    }
    if (std::isnan(lo)) lo = data_lo;
    if (std::isnan(hi)) hi = data_hi;
  }
  const double     span = hi - lo;
  const auto       last = static_cast<double>(palette.size() - 1);
  std::vector<RGB> out;
  out.reserve(values.size());
  for (const double v : values) {
    if (std::isnan(v)) {
      out.push_back(RGB{190, 190, 190});
      continue;
    }
    const double t = (span > 0.0 && std::isfinite(span)) ? std::clamp((v - lo) / span, 0.0, 1.0) : 0.5;
    out.push_back(palette[static_cast<std::size_t>(std::lround(t * last))]);
  }
  return out;
}

RPlot& RPlot::segments(std::vector<double> x0, std::vector<double> y0, std::vector<double> x1, std::vector<double> y1, RGB color, double stroke_width) {
  if (x0.size() != y0.size() || x0.size() != x1.size() || x0.size() != y1.size()) {
    throw std::invalid_argument("RPlot::segments: x0, y0, x1, y1 must have the same size");
  }
  if (x0.empty()) {
    throw std::invalid_argument("RPlot::segments: must not be empty");
  }
  DataSeries series;
  series.kind = DataSeries::Kind::Segment;
  series.color = color;
  series.stroke_width = stroke_width;
  series.x.reserve(x0.size() * 2U);
  series.y.reserve(x0.size() * 2U);
  for (std::size_t i = 0; i < x0.size(); ++i) {
    series.x.push_back(x0[i]);
    series.x.push_back(x1[i]);
    series.y.push_back(y0[i]);
    series.y.push_back(y1[i]);
  }
  add_series(std::move(series));
  return *this;
}

RLayout RLayout::create(std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("RLayout::create: rows and cols must be positive");
  }
  RLayout layout;
  layout.rows_ = rows;
  layout.cols_ = cols;
  return layout;
}

RLayout& RLayout::add(const Plot& panel) {
  if (panels_.size() >= rows_ * cols_) {
    throw std::invalid_argument("RLayout::add: layout is already full");
  }
  panels_.push_back(Panel{panel, svg_body(panel)});
  return *this;
}

RLayout& RLayout::size(std::size_t width, std::size_t height) {
  width_ = width;
  height_ = height;
  return *this;
}

RLayout& RLayout::title(std::string value) {
  title_ = std::move(value);
  return *this;
}

const Plot& RLayout::panel(std::size_t index) const {
  if (index >= panels_.size()) {
    throw std::out_of_range("RLayout::panel: index out of range");
  }
  return panels_[index].plot;
}

std::string RLayout::to_svg() const {
  std::ostringstream out;
  const double band = title_.empty() ? 0.0 : static_cast<double>(title_band_px);
  const double cell_w = static_cast<double>(width_) / static_cast<double>(cols_);
  const double cell_h = (static_cast<double>(height_) - band) / static_cast<double>(rows_);
  out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width_ << "\" height=\""
      << height_ << "\" viewBox=\"0 0 " << width_ << " " << height_ << "\">\n";
  out << svg_font_defs();
  out << "  <rect width=\"100%\" height=\"100%\" fill=\"rgb(255,255,255)\"/>\n";
  if (!title_.empty()) {
    out << "  <text x=\"" << static_cast<double>(width_) / 2.0 << "\" y=\"" << band * 0.7
        << "\" font-size=\"24\" text-anchor=\"middle\" "
        << "font-family=\"" << kSvgFontFamily << "\" fill=\"rgb(55,65,81)\">"
        << xml_escape(title_) << "</text>\n";
  }
  for (std::size_t i = 0; i < panels_.size(); ++i) {
    const std::size_t row = i / cols_;
    const std::size_t col = i % cols_;
    const double       x = static_cast<double>(col) * cell_w;
    const double       y = band + static_cast<double>(row) * cell_h;
    out << "  <svg x=\"" << x << "\" y=\"" << y << "\" width=\"" << cell_w << "\" height=\""
        << cell_h << "\" viewBox=\"0 0 " << panels_[i].plot.width() << " "
        << panels_[i].plot.height() << "\">\n"
        << panels_[i].svg_fragment << "  </svg>\n";
  }
  out << "</svg>\n";
  return out.str();
}

void RLayout::save_svg(const std::string& path) const {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("RLayout::save_svg: could not open output file");
  }
  out << to_svg();
}

void RLayout::save(const std::string& path) const {
  const auto ext = std::filesystem::path(path).extension().string();
  if (ext == ".svg" || ext.empty()) {
    save_svg(path);
  } else if (ext == ".png") {
    save_png(path);
  } else if (ext == ".pdf") {
    save_pdf(path);
  } else {
    throw std::invalid_argument("RLayout::save: unsupported extension");
  }
}

std::vector<std::uint8_t> RLayout::to_png() const { return encode_png(rasterize_layout(*this)); }

std::vector<std::uint8_t> RLayout::to_pdf() const { return encode_pdf(rasterize_layout(*this)); }

void RLayout::save_png(const std::string& path) const { write_bytes(to_png(), path, "RLayout::save_png"); }

void RLayout::save_pdf(const std::string& path) const { write_bytes(to_pdf(), path, "RLayout::save_pdf"); }

RLayout& RLayout::fit_panels() {
  const std::size_t band = title_.empty() ? 0 : title_band_px;
  const std::size_t cell_w = std::max<std::size_t>(1, width_ / cols_);
  const std::size_t cell_h = std::max<std::size_t>(1, (height_ > band ? height_ - band : 1) / rows_);
  // Below min_panel_px the margins would not fit, so draw on a larger canvas with the cell's
  // aspect ratio and let rendering scale it into the cell.
  const double shortest = static_cast<double>(std::min(cell_w, cell_h));
  const double scale = shortest < static_cast<double>(min_panel_px) ? min_panel_px / shortest : 1.0;
  const auto canvas_w = static_cast<std::size_t>(std::lround(static_cast<double>(cell_w) * scale));
  const auto canvas_h = static_cast<std::size_t>(std::lround(static_cast<double>(cell_h) * scale));
  for (auto& panel : panels_) {
    panel.plot.size(canvas_w, canvas_h);
    panel.svg_fragment = svg_body(panel.plot);
  }
  return *this;
}

void RLayout::view(const std::string& title_hint) const {
  show(title_hint);
}

void RLayout::show(const std::string& title_hint) const {
  if (panels_.empty()) {
    throw std::invalid_argument("RLayout::show: layout has no panels");
  }
#if defined(DATAMUNGE_HAVE_X11)
  struct PanelView {
    Bounds current;
    Bounds original;
  };
  std::vector<PanelView> views;
  views.reserve(panels_.size());
  for (const auto& panel : panels_) {
    const Bounds bounds = compute_bounds(panel.plot);
    views.push_back(PanelView{bounds, bounds});
  }
  const ViewerModel model{
      [&]() {
        std::vector<Image> images;
        images.reserve(panels_.size());
        for (std::size_t i = 0; i < panels_.size(); ++i) {
          images.push_back(rasterize(panels_[i].plot, views[i].current));
        }
        return tile_panels(images, rows_, cols_, width_, height_, title_);
      },
      [&](double factor) {
        for (auto& view : views) zoom_bounds(view.current, factor);
      },
      [&](double dx, double dy) {
        for (auto& view : views) pan_bounds(view.current, dx, dy);
      },
      [&]() {
        for (auto& view : views) view.current = view.original;
      },
  };
  run_x11_viewer(model, title_hint.empty() ? title_ : title_hint);
#else
  (void)title_hint;
  throw std::runtime_error("RLayout::show: X11 support was not enabled at build time");
#endif
}

Figure::Figure(Plot plot) : content_(std::move(plot)) {}

Figure::Figure(RLayout layout) : content_(std::move(layout)) {}

Figure& Figure::title(std::string value) {
  if (auto* plot = std::get_if<Plot>(&content_)) {
    plot->title(std::move(value));
  } else if (single_) {
    // Keep a single plot's title on the plot itself, not in a figure band.
    auto& layout = std::get<RLayout>(content_);
    Plot inner = layout.panel(0);
    inner.title(std::move(value));
    content_ = RLayout::create(1, 1).add(inner).size(layout.width(), layout.height()).fit_panels();
  } else {
    std::get<RLayout>(content_).title(std::move(value)).fit_panels();
  }
  return *this;
}

Figure& Figure::size(std::size_t width, std::size_t height) {
  if (auto* plot = std::get_if<Plot>(&content_)) {
    // A small single plot is drawn as a 1 x 1 grid so its canvas can stay min_panel_px or more.
    if (std::min(width, height) < RLayout::min_panel_px) {
      content_ = RLayout::create(1, 1).add(*plot).size(width, height).fit_panels();
      single_ = true;
      return *this;
    }
    plot->size(width, height);
  } else {
    std::get<RLayout>(content_).size(width, height).fit_panels();
  }
  return *this;
}

std::size_t Figure::width() const noexcept {
  return std::visit([](const auto& c) { return c.width(); }, content_);
}

std::size_t Figure::height() const noexcept {
  return std::visit([](const auto& c) { return c.height(); }, content_);
}

const std::string& Figure::title_text() const noexcept {
  if (single_) return std::get<RLayout>(content_).panel(0).title_text();
  return std::visit([](const auto& c) -> const std::string& { return c.title_text(); }, content_);
}

bool Figure::is_grid() const noexcept { return !single_ && std::holds_alternative<RLayout>(content_); }

std::size_t Figure::panel_count() const noexcept {
  if (const auto* layout = std::get_if<RLayout>(&content_)) return single_ ? 1 : layout->panel_count();
  return 1;
}

const Plot& Figure::panel(std::size_t index) const {
  if (const auto* layout = std::get_if<RLayout>(&content_)) return layout->panel(index);
  if (index != 0) throw std::out_of_range("Figure::panel: index out of range");
  return std::get<Plot>(content_);
}

std::string Figure::to_svg() const {
  return std::visit([](const auto& c) { return c.to_svg(); }, content_);
}

std::vector<std::uint8_t> Figure::to_png() const {
  return std::visit([](const auto& c) { return c.to_png(); }, content_);
}

std::vector<std::uint8_t> Figure::to_pdf() const {
  return std::visit([](const auto& c) { return c.to_pdf(); }, content_);
}

void Figure::save(const std::string& path) const {
  std::visit([&](const auto& c) { c.save(path); }, content_);
}

void Figure::save_svg(const std::string& path) const { write_text(to_svg(), path, "Figure::save_svg"); }

void Figure::save_png(const std::string& path) const { write_bytes(to_png(), path, "Figure::save_png"); }

void Figure::save_pdf(const std::string& path) const { write_bytes(to_pdf(), path, "Figure::save_pdf"); }

void Figure::show(const std::string& title_hint) const {
  std::visit([&](const auto& c) { c.show(title_hint); }, content_);
}

} // namespace datamunge::graphics
