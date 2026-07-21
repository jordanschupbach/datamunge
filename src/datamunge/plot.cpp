#include <datamunge/plot/plot.hpp>

#include <datamunge/datamunge.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(DATAMUNGE_HAVE_X11)
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <X11/Xutil.h>
#endif

namespace datamunge::plot {
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
  out << "rgb(" << c.r << "," << c.g << "," << c.b << ")";
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

Layout compute_layout(const Plot& plot) {
  Layout layout;
  layout.plot_w = static_cast<double>(plot.width()) - layout.left - layout.right;
  layout.plot_h = static_cast<double>(plot.height()) - layout.top - layout.bottom;
  if (layout.plot_w <= 0.0 || layout.plot_h <= 0.0) {
    throw std::runtime_error("Plot::save: invalid plot dimensions");
  }
  layout.bounds = compute_bounds(plot);
  return layout;
}

Layout compute_layout(const Plot& plot, Bounds bounds) {
  Layout layout;
  layout.plot_w = static_cast<double>(plot.width()) - layout.left - layout.right;
  layout.plot_h = static_cast<double>(plot.height()) - layout.top - layout.bottom;
  if (layout.plot_w <= 0.0 || layout.plot_h <= 0.0) {
    throw std::runtime_error("Plot::save: invalid plot dimensions");
  }
  layout.bounds = bounds;
  return layout;
}

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

void write_axes_and_grid(std::ostringstream& out, const Plot& plot, const Layout& layout) {
  out << "  <rect width=\"100%\" height=\"100%\" fill=\"" << rgb_css(plot.background_color())
      << "\"/>\n";
  out << "  <style>"
      << "text{font-family:Helvetica,Arial,sans-serif;fill:" << rgb_css(plot.axes_color()) << ";}"
      << "</style>\n";

  const bool categorical = !plot.x_tick_label_list().empty();

  if (!plot.axes_hidden()) {
    if (plot.grid_visible()) {
      for (int i = 0; i <= 5; ++i) {
        const double tx = layout.left + layout.plot_w * static_cast<double>(i) / 5.0;
        const double ty = layout.top + layout.plot_h * static_cast<double>(i) / 5.0;
        out << "  <line x1=\"" << tx << "\" y1=\"" << layout.top << "\" x2=\"" << tx
            << "\" y2=\"" << layout.top + layout.plot_h << "\" stroke=\""
            << rgb_css(plot.major_grid_color()) << "\" stroke-width=\"1\"/>\n";
        out << "  <line x1=\"" << layout.left << "\" y1=\"" << ty << "\" x2=\""
            << layout.left + layout.plot_w << "\" y2=\"" << ty << "\" stroke=\""
            << rgb_css(plot.major_grid_color()) << "\" stroke-width=\"1\"/>\n";
      }
    }

    out << "  <line x1=\"" << layout.left << "\" y1=\"" << layout.top + layout.plot_h
        << "\" x2=\"" << layout.left + layout.plot_w << "\" y2=\"" << layout.top + layout.plot_h
        << "\" stroke=\"" << rgb_css(plot.axes_color()) << "\" stroke-width=\"2\"/>\n";
    out << "  <line x1=\"" << layout.left << "\" y1=\"" << layout.top << "\" x2=\""
        << layout.left << "\" y2=\"" << layout.top + layout.plot_h << "\" stroke=\""
        << rgb_css(plot.axes_color()) << "\" stroke-width=\"2\"/>\n";

    if (categorical) {
      const auto& labels = plot.x_tick_label_list();
      for (std::size_t i = 0; i < labels.size(); ++i) {
        const double tx = map_x(layout, static_cast<double>(i));
        out << "  <text x=\"" << tx << "\" y=\"" << layout.top + layout.plot_h + 24
            << "\" font-size=\"12\" text-anchor=\"middle\">" << xml_escape(labels[i])
            << "</text>\n";
      }
      for (int i = 0; i <= 5; ++i) {
        const double yv = layout.bounds.y_min + (layout.bounds.y_max - layout.bounds.y_min)
                                                     * static_cast<double>(i) / 5.0;
        const double ty = layout.top + layout.plot_h - layout.plot_h * static_cast<double>(i) / 5.0;
        out << "  <text x=\"" << layout.left - 10 << "\" y=\"" << ty + 4
            << "\" font-size=\"12\" text-anchor=\"end\">" << format_tick(yv) << "</text>\n";
      }
    } else {
      for (int i = 0; i <= 5; ++i) {
        const double xv = layout.bounds.x_min + (layout.bounds.x_max - layout.bounds.x_min)
                                                     * static_cast<double>(i) / 5.0;
        const double yv = layout.bounds.y_min + (layout.bounds.y_max - layout.bounds.y_min)
                                                     * static_cast<double>(i) / 5.0;
        const double tx = layout.left + layout.plot_w * static_cast<double>(i) / 5.0;
        const double ty = layout.top + layout.plot_h - layout.plot_h * static_cast<double>(i) / 5.0;
        out << "  <text x=\"" << tx << "\" y=\"" << layout.top + layout.plot_h + 24
            << "\" font-size=\"12\" text-anchor=\"middle\">" << format_tick(xv) << "</text>\n";
        out << "  <text x=\"" << layout.left - 10 << "\" y=\"" << ty + 4
            << "\" font-size=\"12\" text-anchor=\"end\">" << format_tick(yv) << "</text>\n";
      }
    }
  }

  out << "  <text x=\"" << plot.width() / 2.0
      << "\" y=\"36\" font-size=\"24\" text-anchor=\"middle\">"
      << xml_escape(plot.title_text()) << "</text>\n";
  if (!plot.axes_hidden()) {
    out << "  <text x=\"" << plot.width() / 2.0 << "\" y=\"" << plot.height() - 20
        << "\" font-size=\"16\" text-anchor=\"middle\">" << xml_escape(plot.x_label_text())
        << "</text>\n";
    out << "  <text x=\"24\" y=\"" << plot.height() / 2.0
        << "\" font-size=\"16\" text-anchor=\"middle\" transform=\"rotate(-90 24 "
        << plot.height() / 2.0 << ")\">" << xml_escape(plot.y_label_text()) << "</text>\n";
  }
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
          out << "  <circle cx=\"" << map_x(layout, series.x[i]) << "\" cy=\""
              << map_y(layout, series.y[i]) << "\" r=\"" << series.marker_size << "\" fill=\""
              << color << "\"/>\n";
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
        const double position = series.x[0];
        const double lo = series.y[0];
        const double q1 = series.y[1];
        const double median = series.y[2];
        const double q3 = series.y[3];
        const double hi = series.y[4];
        const double half = series.bar_width / 2.0;
        const double cx = map_x(layout, position);
        const double x_left = map_x(layout, position - half);
        const double x_right = map_x(layout, position + half);
        out << "  <line x1=\"" << cx << "\" y1=\"" << map_y(layout, lo) << "\" x2=\"" << cx
            << "\" y2=\"" << map_y(layout, q1) << "\" stroke=\"" << color
            << "\" stroke-width=\"1.5\"/>\n";
        out << "  <line x1=\"" << cx << "\" y1=\"" << map_y(layout, q3) << "\" x2=\"" << cx
            << "\" y2=\"" << map_y(layout, hi) << "\" stroke=\"" << color
            << "\" stroke-width=\"1.5\"/>\n";
        out << "  <line x1=\"" << x_left << "\" y1=\"" << map_y(layout, lo) << "\" x2=\""
            << x_right << "\" y2=\"" << map_y(layout, lo) << "\" stroke=\"" << color
            << "\" stroke-width=\"1.5\"/>\n";
        out << "  <line x1=\"" << x_left << "\" y1=\"" << map_y(layout, hi) << "\" x2=\""
            << x_right << "\" y2=\"" << map_y(layout, hi) << "\" stroke=\"" << color
            << "\" stroke-width=\"1.5\"/>\n";
        const double box_top = map_y(layout, q3);
        const double box_bottom = map_y(layout, q1);
        out << "  <rect x=\"" << x_left << "\" y=\"" << box_top << "\" width=\""
            << (x_right - x_left) << "\" height=\"" << (box_bottom - box_top) << "\" fill=\""
            << color << "\" fill-opacity=\"0.35\" stroke=\"" << color
            << "\" stroke-width=\"1.5\"/>\n";
        out << "  <line x1=\"" << x_left << "\" y1=\"" << map_y(layout, median) << "\" x2=\""
            << x_right << "\" y2=\"" << map_y(layout, median) << "\" stroke=\"" << color
            << "\" stroke-width=\"2\"/>\n";
        for (std::size_t i = 5; i < series.y.size(); ++i) {
          out << "  <circle cx=\"" << cx << "\" cy=\"" << map_y(layout, series.y[i])
              << "\" r=\"3\" fill=\"" << color << "\"/>\n";
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
            << map_y(layout, series.y[0]) << "\" font-size=\"" << series.marker_size
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
  }

  double legend_y = layout.top;
  if (!plot.legend_entries().empty()) {
    for (const auto& entry : plot.legend_entries()) {
      const double x0 = layout.left + layout.plot_w - 140.0;
      out << "  <rect x=\"" << x0 << "\" y=\"" << legend_y - 12
          << "\" width=\"18\" height=\"8\" fill=\"" << rgb_css(entry.color) << "\"/>\n";
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
  out << svg_body(plot);
  out << "</svg>\n";
  return out.str();
}

using Glyph = std::array<std::uint8_t, 7>;

Glyph glyph_for(char ch) {
  if (ch >= 'a' && ch <= 'z') {
    ch = static_cast<char>(ch - 'a' + 'A');
  }
  switch (ch) {
    case 'A': return {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'B': return {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    case 'C': return {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    case 'D': return {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
    case 'E': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    case 'F': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    case 'G': return {0x0E, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0E};
    case 'H': return {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'I': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F};
    case 'J': return {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C};
    case 'K': return {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    case 'L': return {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    case 'M': return {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11};
    case 'N': return {0x11, 0x19, 0x19, 0x15, 0x13, 0x13, 0x11};
    case 'O': return {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'P': return {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    case 'Q': return {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    case 'R': return {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    case 'S': return {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    case 'T': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'U': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'V': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    case 'W': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11};
    case 'X': return {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
    case 'Y': return {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
    case 'Z': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};
    case '0': return {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
    case '1': return {0x04, 0x0C, 0x14, 0x04, 0x04, 0x04, 0x1F};
    case '2': return {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    case '3': return {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E};
    case '4': return {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    case '5': return {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E};
    case '6': return {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    case '7': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    case '8': return {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    case '9': return {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E};
    case '.': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C};
    case ',': return {0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x08};
    case '-': return {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00};
    case '+': return {0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00};
    case ':': return {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00};
    case '/': return {0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10};
    case '(': return {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02};
    case ')': return {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08};
    case '_': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F};
    case ' ': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    default: return {0x00, 0x0E, 0x01, 0x06, 0x04, 0x00, 0x04};
  }
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
            x, y, color, static_cast<double>(covered_samples) / (samples_per_axis * samples_per_axis));
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

void draw_char(Image& image, int x, int y, char ch, RGB color, int scale) {
  const Glyph glyph = glyph_for(ch);
  for (int row = 0; row < 7; ++row) {
    for (int col = 0; col < 5; ++col) {
      if (((glyph[row] >> (4 - col)) & 0x1U) != 0U) {
        image.fill_rect(x + col * scale,
                        y + row * scale,
                        x + (col + 1) * scale - 1,
                        y + (row + 1) * scale - 1,
                        color);
      }
    }
  }
}

int draw_text(Image& image, int x, int y, const std::string& text, RGB color, int scale) {
  int cursor = x;
  for (char ch : text) {
    draw_char(image, cursor, y, ch, color, scale);
    cursor += 6 * scale;
  }
  return cursor;
}

void draw_vertical_text(Image& image, int x, int y, const std::string& text, RGB color, int scale) {
  int cursor_y = y;
  for (char ch : text) {
    draw_char(image, x, cursor_y, ch, color, scale);
    cursor_y += 8 * scale;
  }
}

Image rasterize(const Plot& plot) {
  const Layout layout = compute_layout(plot);
  Image image(plot.width(), plot.height(), plot.background_color());
  const RGB axis_color = plot.axes_color();

  if (plot.grid_visible()) {
    for (int i = 0; i <= 5; ++i) {
      const double tx = layout.left + layout.plot_w * static_cast<double>(i) / 5.0;
      const double ty = layout.top + layout.plot_h * static_cast<double>(i) / 5.0;
      draw_line(image, tx, layout.top, tx, layout.top + layout.plot_h, plot.major_grid_color(), 1.0);
      draw_line(image, layout.left, ty, layout.left + layout.plot_w, ty, plot.major_grid_color(), 1.0);
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

  for (int i = 0; i <= 5; ++i) {
    const double xv = layout.bounds.x_min
                    + (layout.bounds.x_max - layout.bounds.x_min) * static_cast<double>(i) / 5.0;
    const double yv = layout.bounds.y_min
                    + (layout.bounds.y_max - layout.bounds.y_min) * static_cast<double>(i) / 5.0;
    const int tx = static_cast<int>(
        std::lround(layout.left + layout.plot_w * static_cast<double>(i) / 5.0));
    const int ty = static_cast<int>(
        std::lround(layout.top + layout.plot_h - layout.plot_h * static_cast<double>(i) / 5.0));
    const std::string x_label = format_tick(xv);
    const std::string y_label = format_tick(yv);
    draw_text(image,
              tx - static_cast<int>(x_label.size()) * 6,
              static_cast<int>(layout.top + layout.plot_h + 18.0),
              x_label,
              axis_color,
              2);
    draw_text(image, 8, ty - 6, y_label, axis_color, 2);
  }

  const std::string title = plot.title_text();
  const int title_width = static_cast<int>(title.size()) * 12;
  draw_text(image, static_cast<int>(plot.width() / 2) - title_width / 2, 18, title, axis_color, 2);

  const std::string x_text = plot.x_label_text();
  const int x_width = static_cast<int>(x_text.size()) * 12;
  draw_text(image,
            static_cast<int>(plot.width() / 2) - x_width / 2,
            static_cast<int>(plot.height()) - 32,
            x_text,
            axis_color,
            2);
  draw_vertical_text(image, 24, 72, plot.y_label_text(), axis_color, 2);

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
                    series.color);
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
    }
  }

  int legend_y = static_cast<int>(layout.top);
  for (const auto& series : plot.series()) {
    if (series.label.empty()) {
      continue;
    }
    const int x0 = static_cast<int>(layout.left + layout.plot_w - 160.0);
    image.fill_rect(x0, legend_y - 10, x0 + 18, legend_y - 2, series.color);
    draw_text(image, x0 + 28, legend_y - 16, series.label, axis_color, 2);
    legend_y += 24;
  }

  return image;
}

Image rasterize(const Plot& plot, Bounds bounds) {
  const Layout layout = compute_layout(plot, bounds);
  Image image(plot.width(), plot.height(), plot.background_color());
  const RGB axis_color = plot.axes_color();

  if (plot.grid_visible()) {
    for (int i = 0; i <= 5; ++i) {
      const double tx = layout.left + layout.plot_w * static_cast<double>(i) / 5.0;
      const double ty = layout.top + layout.plot_h * static_cast<double>(i) / 5.0;
      draw_line(image, tx, layout.top, tx, layout.top + layout.plot_h, plot.major_grid_color(), 1.0);
      draw_line(image, layout.left, ty, layout.left + layout.plot_w, ty, plot.major_grid_color(), 1.0);
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

  for (int i = 0; i <= 5; ++i) {
    const double xv = layout.bounds.x_min
                    + (layout.bounds.x_max - layout.bounds.x_min) * static_cast<double>(i) / 5.0;
    const double yv = layout.bounds.y_min
                    + (layout.bounds.y_max - layout.bounds.y_min) * static_cast<double>(i) / 5.0;
    const int tx = static_cast<int>(
        std::lround(layout.left + layout.plot_w * static_cast<double>(i) / 5.0));
    const int ty = static_cast<int>(
        std::lround(layout.top + layout.plot_h - layout.plot_h * static_cast<double>(i) / 5.0));
    const std::string x_label = format_tick(xv);
    const std::string y_label = format_tick(yv);
    draw_text(image,
              tx - static_cast<int>(x_label.size()) * 6,
              static_cast<int>(layout.top + layout.plot_h + 18.0),
              x_label,
              axis_color,
              2);
    draw_text(image, 8, ty - 6, y_label, axis_color, 2);
  }

  const std::string title = plot.title_text();
  const int title_width = static_cast<int>(title.size()) * 12;
  draw_text(image, static_cast<int>(plot.width() / 2) - title_width / 2, 18, title, axis_color, 2);

  const std::string x_text = plot.x_label_text();
  const int x_width = static_cast<int>(x_text.size()) * 12;
  draw_text(image,
            static_cast<int>(plot.width() / 2) - x_width / 2,
            static_cast<int>(plot.height()) - 32,
            x_text,
            axis_color,
            2);
  draw_vertical_text(image, 24, 72, plot.y_label_text(), axis_color, 2);

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
                    series.color);
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
    }
  }

  int legend_y = static_cast<int>(layout.top);
  for (const auto& series : plot.series()) {
    if (series.label.empty()) {
      continue;
    }
    const int x0 = static_cast<int>(layout.left + layout.plot_w - 160.0);
    image.fill_rect(x0, legend_y - 10, x0 + 18, legend_y - 2, series.color);
    draw_text(image, x0 + 28, legend_y - 16, series.label, axis_color, 2);
    legend_y += 24;
  }

  return image;
}

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
  constexpr std::uint32_t mod = 65521U;
  std::uint32_t a = 1U;
  std::uint32_t b = 0U;
  for (std::uint8_t byte : data) {
    a = (a + byte) % mod;
    b = (b + a) % mod;
  }
  return (b << 16U) | a;
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

void save_png_image(const Image& image, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Plot::save: could not open PNG output file");
  }

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

void save_pdf_image(const Image& image, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Plot::save: could not open PDF output file");
  }

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
  const int text_width = static_cast<int>(button.label.size()) * 6;
  draw_text(image,
            button.x + std::max(4, (button.w - text_width) / 2),
            button.y + std::max(3, (button.h - 7) / 2),
            button.label,
            text,
            1);
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
    const int width = 18 + static_cast<int>(spec.first.size()) * 6;
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

void show_x11_plot(const Plot& plot, const std::string& title_hint) {
  constexpr int menu_height = 44;
  Image plot_image = rasterize(plot);
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
  Bounds current_bounds = compute_bounds(plot);
  const Bounds original_bounds = current_bounds;
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

  auto rerasterize_plot = [&]() { plot_image = rasterize(plot, current_bounds); };

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
            zoom_bounds(current_bounds, 0.8);
            break;
          case ViewerButton::Action::PlotZoomOut:
            zoom_bounds(current_bounds, 1.25);
            break;
          case ViewerButton::Action::PlotLeft:
            pan_bounds(current_bounds, -0.1, 0.0);
            break;
          case ViewerButton::Action::PlotRight:
            pan_bounds(current_bounds, 0.1, 0.0);
            break;
          case ViewerButton::Action::PlotUp:
            pan_bounds(current_bounds, 0.0, 0.1);
            break;
          case ViewerButton::Action::PlotDown:
            pan_bounds(current_bounds, 0.0, -0.1);
            break;
          case ViewerButton::Action::PlotReset:
            current_bounds = original_bounds;
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
  return *this;
}

void Plot::save_svg(const std::string& path) const {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Plot::save_svg: could not open output file");
  }
  out << svg_string(*this);
}

void Plot::save(const std::string& path) const {
  const auto ext = std::filesystem::path(path).extension().string();
  if (ext == ".svg" || ext.empty()) {
    save_svg(path);
    return;
  }

  const Image image = rasterize(*this);
  if (ext == ".png") {
    save_png_image(image, path);
    return;
  }
  if (ext == ".pdf") {
    save_pdf_image(image, path);
    return;
  }

  throw std::invalid_argument("Plot::save: unsupported extension");
}

void Plot::view(const std::string& title_hint) const {
  show(title_hint);
}

void Plot::show(const std::string& title_hint) const {
#if defined(DATAMUNGE_HAVE_X11)
  show_x11_plot(*this, title_hint);
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

RPlot RPlot::create() {
  return RPlot{};
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
  double lo = *std::min_element(data.begin(), data.end());
  double hi = *std::max_element(data.begin(), data.end());
  if (hi <= lo) {
    hi = lo + 1.0;
  }
  const double bin_width = (hi - lo) / static_cast<double>(bins);

  std::vector<double> counts(bins, 0.0);
  for (double value : data) {
    auto idx = static_cast<std::ptrdiff_t>((value - lo) / bin_width);
    idx = std::clamp<std::ptrdiff_t>(idx, 0, static_cast<std::ptrdiff_t>(bins) - 1);
    counts[static_cast<std::size_t>(idx)] += 1.0;
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
  panels_.push_back(Panel{panel.width(), panel.height(), svg_body(panel)});
  return *this;
}

RLayout& RLayout::size(std::size_t width, std::size_t height) {
  width_ = width;
  height_ = height;
  return *this;
}

void RLayout::save_svg(const std::string& path) const {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("RLayout::save_svg: could not open output file");
  }
  const double cell_w = static_cast<double>(width_) / static_cast<double>(cols_);
  const double cell_h = static_cast<double>(height_) / static_cast<double>(rows_);
  out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width_ << "\" height=\""
      << height_ << "\" viewBox=\"0 0 " << width_ << " " << height_ << "\">\n";
  out << "  <rect width=\"100%\" height=\"100%\" fill=\"rgb(255,255,255)\"/>\n";
  for (std::size_t i = 0; i < panels_.size(); ++i) {
    const std::size_t row = i / cols_;
    const std::size_t col = i % cols_;
    const double       x = static_cast<double>(col) * cell_w;
    const double       y = static_cast<double>(row) * cell_h;
    out << "  <svg x=\"" << x << "\" y=\"" << y << "\" width=\"" << cell_w << "\" height=\""
        << cell_h << "\" viewBox=\"0 0 " << panels_[i].width << " " << panels_[i].height
        << "\">\n"
        << panels_[i].svg_fragment << "  </svg>\n";
  }
  out << "</svg>\n";
}

void RLayout::save(const std::string& path) const {
  save_svg(path);
}

} // namespace datamunge::plot
