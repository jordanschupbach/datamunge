#include <datamunge/plot/plot.hpp>

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
  bool first = true;
  for (const auto& series : plot.series()) {
    for (std::size_t i = 0; i < series.x.size(); ++i) {
      const double x = series.x[i];
      const double y = series.y[i];
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
      if (series.kind == DataSeries::Kind::Bar) {
        b.y_min = std::min(b.y_min, 0.0);
      }
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

std::string svg_string(const Plot& plot) {
  const Layout layout = compute_layout(plot);

  std::ostringstream out;
  out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << plot.width()
      << "\" height=\"" << plot.height() << "\" viewBox=\"0 0 " << plot.width() << " "
      << plot.height() << "\">\n";
  out << "  <rect width=\"100%\" height=\"100%\" fill=\"" << rgb_css(plot.background_color())
      << "\"/>\n";
  out << "  <style>"
      << "text{font-family:Helvetica,Arial,sans-serif;fill:" << rgb_css(plot.axes_color()) << ";}"
      << "</style>\n";

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

  for (int i = 0; i <= 5; ++i) {
    const double xv = layout.bounds.x_min
                    + (layout.bounds.x_max - layout.bounds.x_min) * static_cast<double>(i) / 5.0;
    const double yv = layout.bounds.y_min
                    + (layout.bounds.y_max - layout.bounds.y_min) * static_cast<double>(i) / 5.0;
    const double tx = layout.left + layout.plot_w * static_cast<double>(i) / 5.0;
    const double ty = layout.top + layout.plot_h - layout.plot_h * static_cast<double>(i) / 5.0;
    out << "  <text x=\"" << tx << "\" y=\"" << layout.top + layout.plot_h + 24
        << "\" font-size=\"12\" text-anchor=\"middle\">" << format_tick(xv) << "</text>\n";
    out << "  <text x=\"" << layout.left - 10 << "\" y=\"" << ty + 4
        << "\" font-size=\"12\" text-anchor=\"end\">" << format_tick(yv) << "</text>\n";
  }

  out << "  <text x=\"" << plot.width() / 2.0
      << "\" y=\"36\" font-size=\"24\" text-anchor=\"middle\">"
      << xml_escape(plot.title_text()) << "</text>\n";
  out << "  <text x=\"" << plot.width() / 2.0 << "\" y=\"" << plot.height() - 20
      << "\" font-size=\"16\" text-anchor=\"middle\">" << xml_escape(plot.x_label_text())
      << "</text>\n";
  out << "  <text x=\"24\" y=\"" << plot.height() / 2.0
      << "\" font-size=\"16\" text-anchor=\"middle\" transform=\"rotate(-90 24 "
      << plot.height() / 2.0 << ")\">" << xml_escape(plot.y_label_text()) << "</text>\n";

  for (const auto& series : plot.series()) {
    const std::string color = rgb_css(series.color);
    if (series.kind == DataSeries::Kind::Line) {
      out << "  <polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\""
          << series.stroke_width << "\" points=\"";
      for (std::size_t i = 0; i < series.x.size(); ++i) {
        out << map_x(layout, series.x[i]) << "," << map_y(layout, series.y[i]) << " ";
      }
      out << "\"/>\n";
    } else if (series.kind == DataSeries::Kind::Scatter) {
      for (std::size_t i = 0; i < series.x.size(); ++i) {
        out << "  <circle cx=\"" << map_x(layout, series.x[i]) << "\" cy=\""
            << map_y(layout, series.y[i]) << "\" r=\"" << series.marker_size << "\" fill=\""
            << color << "\"/>\n";
      }
    } else if (series.kind == DataSeries::Kind::Bar) {
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
    }
  }

  double legend_y = layout.top;
  for (const auto& series : plot.series()) {
    if (series.label.empty()) {
      continue;
    }
    const double x0 = layout.left + layout.plot_w - 140.0;
    out << "  <rect x=\"" << x0 << "\" y=\"" << legend_y - 12
        << "\" width=\"18\" height=\"8\" fill=\"" << rgb_css(series.color) << "\"/>\n";
    out << "  <text x=\"" << x0 + 26 << "\" y=\"" << legend_y - 4 << "\" font-size=\"12\">"
        << xml_escape(series.label) << "</text>\n";
    legend_y += 20.0;
  }

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

void draw_circle(Image& image, int cx, int cy, int radius, RGB color) {
  radius = std::max(radius, 1);
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx * dx + dy * dy <= radius * radius) {
        image.set_pixel(cx + dx, cy + dy, color);
      }
    }
  }
}

void draw_line(Image& image, double x0, double y0, double x1, double y1, RGB color, double width) {
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)))));
  const int radius = std::max(1, static_cast<int>(std::round(width / 2.0)));
  for (int i = 0; i <= steps; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(steps);
    const int px = static_cast<int>(std::lround(x0 + dx * t));
    const int py = static_cast<int>(std::lround(y0 + dy * t));
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
                    static_cast<int>(std::lround(map_x(layout, series.x[i]))),
                    static_cast<int>(std::lround(map_y(layout, series.y[i]))),
                    std::max(1, static_cast<int>(std::lround(series.marker_size))),
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
                    static_cast<int>(std::lround(map_x(layout, series.x[i]))),
                    static_cast<int>(std::lround(map_y(layout, series.y[i]))),
                    std::max(1, static_cast<int>(std::lround(series.marker_size))),
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
      const int src_x =
          static_cast<int>(std::floor((static_cast<double>(x) - draw_x) / scale));
      const int src_y =
          static_cast<int>(std::floor((static_cast<double>(y) - draw_y) / scale));
      if (src_x < 0 || src_y < 0 || src_x >= static_cast<int>(src.width)
          || src_y >= static_cast<int>(src.height)) {
        continue;
      }
      const std::size_t src_idx =
          (static_cast<std::size_t>(src_y) * src.width + static_cast<std::size_t>(src_x)) * 3U;
      dst.set_pixel(x,
                    y,
                    RGB{
                        static_cast<int>(src.pixels[src_idx + 0U]),
                        static_cast<int>(src.pixels[src_idx + 1U]),
                        static_cast<int>(src.pixels[src_idx + 2U]),
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
  XDestroyWindow(display, window);
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

ScatterPlot ScatterPlot::create() {
  return ScatterPlot{};
}

ScatterPlot& ScatterPlot::points(std::vector<double> x,
                                 std::vector<double> y,
                                 std::string         label,
                                 RGB                 color,
                                 double              marker_size) {
  require_xy_same_size(x, y, "ScatterPlot::points");
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

ScatterPlot& ScatterPlot::line(std::vector<double> x,
                               std::vector<double> y,
                               std::string         label,
                               RGB                 color,
                               double              stroke_width) {
  require_xy_same_size(x, y, "ScatterPlot::line");
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

LinePlot LinePlot::create() {
  return LinePlot{};
}

LinePlot& LinePlot::line(std::vector<double> x,
                         std::vector<double> y,
                         std::string         label,
                         RGB                 color,
                         double              stroke_width) {
  require_xy_same_size(x, y, "LinePlot::line");
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

BarChart BarChart::create() {
  return BarChart{};
}

BarChart& BarChart::bars(std::vector<double> x,
                         std::vector<double> y,
                         std::string         label,
                         RGB                 color,
                         double              bar_width) {
  require_xy_same_size(x, y, "BarChart::bars");
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

} // namespace datamunge::plot
