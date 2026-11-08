/// @file diagram.cpp
/// @brief Native diagram layout, boundary anchoring and fitted SVG rendering.

#include <algorithm>
#include <cmath>
#include <datamunge/plot/diagram.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include "cmu_serif_font_data.hpp"

namespace datamunge::graphics {
/// @brief Private native diagram validation, text layout and SVG glyph helpers.
namespace {

/// @brief Reject nonfinite coordinates before they can contaminate fitted bounds.
/// @param p Coordinate to check.
/// @throws std::invalid_argument If either coordinate is nonfinite.
void check_point(DiagramPoint p) {
  if (!std::isfinite(p.x) || !std::isfinite(p.y))
    throw std::invalid_argument("Diagram: coordinates must be finite");
}

/// @brief Validate a positive, finite size.
/// @param value Size to validate.
/// @throws std::invalid_argument If value is not positive and finite.
void positive(double value) {
  if (!std::isfinite(value) || value <= 0)
    throw std::invalid_argument("Diagram: sizes must be positive and finite");
}

/// @brief Validate opacity before rendering; finite channel values are clamped on output.
/// @param color Colour to check.
/// @throws std::invalid_argument For nonfinite opacity.
void check_color(RGB color) {
  if (!std::isfinite(color.alpha))
    throw std::invalid_argument("Diagram: opacity must be finite");
}

/// @brief Validate a line style.
/// @param stroke Width and colour to check.
/// @throws std::invalid_argument For invalid size or opacity.
void check_stroke(const DiagramStroke& stroke) {
  positive(stroke.width);
  check_color(stroke.color);
}

/// @brief XML-escape owned user labels without interpreting markup.
/// @param text UTF-8 text.
/// @return Escaped text suitable for an XML text node.
std::string escape(const std::string& text) {
  std::string result;
  for (const char c : text) {
    if (c == '&')
      result += "&amp;";
    else if (c == '<')
      result += "&lt;";
    else if (c == '>')
      result += "&gt;";
    else if (c == '"')
      result += "&quot;";
    else if (c == '\'')
      result += "&apos;";
    else
      result += c;
  }
  return result;
}

/// @brief CSS colour with clamped channels; opacity is emitted separately.
/// @param color Plot colour.
/// @return CSS rgb() expression.
std::string css(RGB color) {
  return "rgb(" + std::to_string(std::clamp(color.r, 0, 255)) + "," +
         std::to_string(std::clamp(color.g, 0, 255)) + "," +
         std::to_string(std::clamp(color.b, 0, 255)) + ")";
}

/// @brief One embedded-font glyph, including Unicode super/subscript transforms.
struct LabelGlyph {
  std::size_t index; ///< Index into the plotting font's glyph table.
  double      scale; ///< Relative glyph size; .7 for a script, one otherwise.
  double      shift; ///< Baseline shift in ems, positive downwards.
};

/// @brief Decode UTF-8 into native glyphs, constructing scripts from base glyphs.
/// @param text One UTF-8 line.
/// @return Glyphs in reading order; unsupported characters use a question mark.
std::vector<LabelGlyph> label_glyphs(const std::string& text) {
  std::vector<LabelGlyph> result;
  for (std::size_t i = 0; i < text.size();) {
    const auto lead      = static_cast<unsigned char>(text[i++]);
    char32_t   cp        = lead;
    int        remaining = 0;
    if ((lead & 0xe0) == 0xc0) {
      cp        = lead & 0x1f;
      remaining = 1;
    } else if ((lead & 0xf0) == 0xe0) {
      cp        = lead & 0x0f;
      remaining = 2;
    } else if ((lead & 0xf8) == 0xf0) {
      cp        = lead & 0x07;
      remaining = 3;
    }
    while (remaining-- > 0 && i < text.size())
      cp = (cp << 6) | (static_cast<unsigned char>(text[i++]) & 0x3f);
    double scale = 1, shift = 0;
    if (cp >= U'₀' && cp <= U'₉') {
      cp    = U'0' + cp - U'₀';
      scale = .7;
      shift = .25;
    } else if (cp == U'⁰' || (cp >= U'⁴' && cp <= U'⁹')) {
      cp    = U'0' + cp - U'⁰';
      scale = .7;
      shift = -.4;
    } else if (cp == U'¹' || cp == U'²' || cp == U'³') {
      cp    = cp == U'¹' ? U'1' : cp == U'²' ? U'2' : U'3';
      scale = .7;
      shift = -.4;
    } else if (cp == U'⁻' || cp == U'⁺' || cp == U'ᵀ') {
      cp    = cp == U'⁻' ? U'−' : cp == U'⁺' ? U'+' : U'T';
      scale = .7;
      shift = -.4;
    } else if (cp == U'ℓ')
      cp = U'l';
    const auto first = std::begin(detail::cmu_serif::glyphs),
               last  = std::end(detail::cmu_serif::glyphs);
    auto glyph       = std::lower_bound(
        first, last, cp, [](const auto& g, char32_t c) { return g.codepoint < c; });
    if (glyph == last || glyph->codepoint != cp)
      glyph = std::lower_bound(
          first, last, U'?', [](const auto& g, char32_t c) { return g.codepoint < c; });
    result.push_back({static_cast<std::size_t>(glyph - first), scale, shift});
  }
  return result;
}

/// @brief Measure advances using exactly the glyphs and transforms used by SVG export.
/// @param text UTF-8 label line.
/// @param size Font size in pixels.
/// @return Width in pixels.
double text_width(const std::string& text, double size) {
  double width = 0;
  for (const auto glyph : label_glyphs(text))
    width += detail::cmu_serif::glyphs[glyph.index].advance * size * glyph.scale / 1000;
  return width;
}

/// @brief Preserve explicit newlines and optionally wrap words to a preferred width.
/// @param label UTF-8 multiline text.
/// @param size Font size in pixels.
/// @param width Preferred width in pixels; infinity disables wrapping.
/// @return Lines, including explicitly empty ones.
std::vector<std::string> lines(const std::string& label, double size, double width) {
  std::vector<std::string> result;
  std::istringstream       input(label);
  std::string              line;
  while (std::getline(input, line)) {
    if (!std::isfinite(width) || text_width(line, size) <= width) {
      result.push_back(line);
      continue;
    }
    std::istringstream words(line);
    std::string        word, current;
    while (words >> word) {
      const auto candidate = current.empty() ? word : current + " " + word;
      if (!current.empty() && text_width(candidate, size) > width) {
        result.push_back(current);
        current = word;
      } else
        current = candidate;
    }
    result.push_back(current);
  }
  if (result.empty() || (!label.empty() && label.back() == '\n'))
    result.emplace_back();
  return result;
}

/// @brief Write a reusable SVG path from the plotting font's native quadratic/cubic outlines.
/// @param out SVG output stream.
/// @param index Glyph index in the embedded font.
void glyph_path(std::ostream& out, std::size_t index) {
  namespace font     = detail::cmu_serif;
  const auto& glyph  = font::glyphs[index];
  const auto* points = font::points + 2U * glyph.point_begin;
  std::size_t next   = 0;
  out << "<path id=\"dm-g-" << index << "\" d=\"";
  for (std::uint32_t i = 0; i < glyph.op_count; ++i) {
    const auto op      = font::ops[glyph.op_begin + i];
    const char command = op == font::MOVE    ? 'M'
                         : op == font::LINE  ? 'L'
                         : op == font::QUAD  ? 'Q'
                         : op == font::CUBIC ? 'C'
                                             : 'Z';
    out << command;
    const int count = op == font::CLOSE ? 0 : op == font::CUBIC ? 3 : op == font::QUAD ? 2 : 1;
    for (int j = 0; j < count; ++j, ++next)
      out << points[2 * next] << ' ' << points[2 * next + 1] << ' ';
  }
  out << "\"/>\n";
}

} // namespace

Diagram::Diagram(double unit_pixels, double max_width) : unit_(unit_pixels), max_width_(max_width) {
  positive(unit_);
  positive(max_width_);
}

Diagram&
Diagram::node(std::string id, DiagramPoint centre, std::string label, DiagramNodeStyle style) {
  check_point(centre);
  positive(style.width);
  positive(style.height);
  positive(style.font_size);
  if (!std::isfinite(style.padding) || style.padding < 0 || !std::isfinite(style.corner_radius) ||
      style.corner_radius < 0)
    throw std::invalid_argument(
        "Diagram: padding and corner radius must be finite and nonnegative");
  check_color(style.fill);
  check_color(style.border);
  check_color(style.ink);
  if (id.empty() || std::any_of(items_.begin(), items_.end(), [&](const auto& item) {
        return item.kind == "node" && item.id == id;
      }))
    throw std::invalid_argument("Diagram: node ids must be nonempty and unique");
  Item item;
  item.kind         = "node";
  item.id           = std::move(id);
  item.centre       = {centre.x * unit_, centre.y * unit_};
  item.style        = style;
  item.fill         = style.fill;
  item.stroke.color = style.border;
  item.lines =
      lines(label, style.font_size, std::max(1.0, style.width * unit_ - 2 * style.padding));
  double width = 0;
  for (const auto& line : item.lines)
    width = std::max(width, text_width(line, style.font_size));
  item.width = std::max(style.width * unit_, width + 2 * style.padding);
  item.height =
      std::max(style.height * unit_, item.lines.size() * style.font_size * 1.3 + 2 * style.padding);
  if (style.circle) {
    const double diameter = std::max(
        {item.width,
         item.height,
         std::hypot(width, item.lines.size() * style.font_size * 1.3) + 2 * style.padding});
    item.width = item.height = diameter;
  }
  items_.push_back(std::move(item));
  return *this;
}

Diagram& Diagram::text(DiagramPoint centre, std::string label, double font_size, RGB ink) {
  check_point(centre);
  positive(font_size);
  check_color(ink);
  Item item;
  item.kind            = "text";
  item.centre          = {centre.x * unit_, centre.y * unit_};
  item.style.font_size = font_size;
  item.style.ink       = ink;
  item.lines           = lines(label, font_size, std::numeric_limits<double>::infinity());
  for (const auto& line : item.lines)
    item.width = std::max(item.width, text_width(line, font_size));
  item.height = item.lines.size() * font_size * 1.3;
  items_.push_back(std::move(item));
  return *this;
}

Diagram& Diagram::label(DiagramPoint centre, std::string label, double font_size, RGB ink) {
  text(centre, std::move(label), font_size, ink);
  items_.back().fill = {255, 255, 255};
  return *this;
}

Diagram& Diagram::path(std::vector<DiagramPoint> points, DiagramStroke stroke) {
  check_stroke(stroke);
  for (auto& point : points) {
    check_point(point);
    point.x *= unit_;
    point.y *= unit_;
  }
  points.erase(
      std::unique(
          points.begin(), points.end(), [](auto a, auto b) { return a.x == b.x && a.y == b.y; }),
      points.end());
  if (points.size() < 2)
    throw std::invalid_argument("Diagram: a path needs two distinct points");
  Item item;
  item.kind   = "path";
  item.points = std::move(points);
  item.stroke = stroke;
  items_.push_back(std::move(item));
  return *this;
}

Diagram& Diagram::polygon(std::vector<DiagramPoint> points, RGB fill, DiagramStroke stroke) {
  if (points.size() < 3)
    throw std::invalid_argument("Diagram: a polygon needs three vertices");
  check_color(fill);
  check_stroke(stroke);
  for (auto& point : points) {
    check_point(point);
    point.x *= unit_;
    point.y *= unit_;
  }
  Item item;
  item.kind   = "polygon";
  item.points = std::move(points);
  item.stroke = stroke;
  item.fill   = fill;
  items_.push_back(std::move(item));
  return *this;
}

Diagram& Diagram::ellipse(
    DiagramPoint  centre,
    double        radius_x,
    double        radius_y,
    RGB           fill,
    DiagramStroke stroke) {
  check_point(centre);
  positive(radius_x);
  positive(radius_y);
  check_color(fill);
  check_stroke(stroke);
  Item item;
  item.kind   = "ellipse";
  item.centre = {centre.x * unit_, centre.y * unit_};
  item.width  = 2 * radius_x * unit_;
  item.height = 2 * radius_y * unit_;
  item.stroke = stroke;
  item.fill   = fill;
  items_.push_back(std::move(item));
  return *this;
}

Diagram& Diagram::edge(
    const std::string& from,
    const std::string& to,
    std::string        label,
    DiagramStroke      stroke) {
  return connect(from, to, std::move(label), stroke, true);
}

Diagram& Diagram::link(
    const std::string& from,
    const std::string& to,
    std::string        label,
    DiagramStroke      stroke) {
  return connect(from, to, std::move(label), stroke, false);
}

Diagram& Diagram::connect(
    const std::string& from,
    const std::string& to,
    std::string        label,
    DiagramStroke      stroke,
    bool               arrow) {
  check_stroke(stroke);
  const auto find = [&](const std::string& id) -> const Item& {
    const auto found = std::find_if(items_.begin(), items_.end(), [&](const auto& item) {
      return item.kind == "node" && item.id == id;
    });
    if (found == items_.end())
      throw std::invalid_argument("Diagram: unknown node " + id);
    return *found;
  };
  const auto   a = find(from), b = find(to);
  const double dx = b.centre.x - a.centre.x, dy = b.centre.y - a.centre.y;
  if (dx == 0 && dy == 0)
    throw std::invalid_argument("Diagram: edge nodes must have different centres");
  const auto boundary = [&](const Item& item, double sign) -> DiagramPoint {
    const double t =
        item.style.circle
            ? item.width / (2 * std::hypot(dx, dy))
            : 1 / std::max(2 * std::abs(dx) / item.width, 2 * std::abs(dy) / item.height);
    return {(item.centre.x + sign * dx * t) / unit_, (item.centre.y + sign * dy * t) / unit_};
  };
  const auto start = boundary(a, 1), end = boundary(b, -1);
  if ((end.x - start.x) * dx + (end.y - start.y) * dy <= 0)
    throw std::invalid_argument("Diagram: connected node boxes overlap");
  stroke.arrow = arrow;
  path({start, end}, stroke);
  if (!label.empty()) {
    const double length = std::hypot(dx, dy);
    text(
        {(start.x + end.x) / 2 - dy / length * .34, (start.y + end.y) / 2 + dx / length * .34},
        std::move(label),
        16);
    items_.back().fill = {255, 255, 255};
  }
  return *this;
}

std::string Diagram::to_svg() const {
  if (items_.empty())
    throw std::runtime_error("Diagram: cannot render an empty diagram");
  double left = std::numeric_limits<double>::infinity(), right = -left, bottom = left, top = -left;
  const auto bound = [&](DiagramPoint p, double margin) {
    left   = std::min(left, p.x - margin);
    right  = std::max(right, p.x + margin);
    bottom = std::min(bottom, p.y - margin);
    top    = std::max(top, p.y + margin);
  };
  for (const auto& item : items_) {
    const double margin = item.stroke.width + (item.stroke.arrow ? 12 : 4);
    if (!item.points.empty())
      for (const auto p : item.points)
        bound(p, margin);
    else {
      bound({item.centre.x - item.width / 2, item.centre.y - item.height / 2}, margin);
      bound({item.centre.x + item.width / 2, item.centre.y + item.height / 2}, margin);
    }
  }
  left -= 24;
  bottom -= 24;
  right += 24;
  top += 24;
  const double width = right - left, height = top - bottom,
               scale = std::min(1.0, max_width_ / width);
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::fixed << std::setprecision(3);
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width * scale << "\" height=\""
      << height * scale << "\" viewBox=\"0 0 " << width << ' ' << height
      << "\" role=\"img\"><title>Method diagram</title>\n";
  out << "<defs>\n";
  std::set<std::size_t> used_glyphs;
  for (const auto& item : items_)
    for (const auto& line : item.lines)
      for (const auto glyph : label_glyphs(line))
        used_glyphs.insert(glyph.index);
  for (const auto index : used_glyphs)
    glyph_path(out, index);
  out << "</defs>\n";
  out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
  const auto px    = [&](double x) { return x - left; };
  const auto py    = [&](double y) { return top - y; };
  const auto paint = [&](const Item& item) {
    out << " fill=\"" << css(item.fill) << "\" fill-opacity=\""
        << std::clamp(item.fill.alpha, 0.0, 1.0) << "\" stroke=\"" << css(item.stroke.color)
        << "\" stroke-opacity=\"" << std::clamp(item.stroke.color.alpha, 0.0, 1.0)
        << "\" stroke-width=\"" << item.stroke.width << "\" stroke-linejoin=\"round\"";
    if (item.stroke.dashed)
      out << " stroke-dasharray=\"6 4\"";
  };
  for (const auto& item : items_) {
    if (item.kind == "path" || item.kind == "polygon") {
      out << (item.kind == "path" ? "<polyline points=\"" : "<polygon points=\"");
      for (const auto p : item.points)
        out << px(p.x) << ',' << py(p.y) << ' ';
      out << '"';
      paint(item);
      out << "/>\n";
      if (item.kind == "path" && item.stroke.arrow) {
        const auto   tip = item.points.back(), previous = item.points[item.points.size() - 2];
        const double length = std::hypot(tip.x - previous.x, tip.y - previous.y);
        const double dx = (tip.x - previous.x) / length, dy = (tip.y - previous.y) / length;
        const double size = std::min(10.0 + item.stroke.width, length * .7);
        out << "<polygon points=\"" << px(tip.x) << ',' << py(tip.y) << ' '
            << px(tip.x - size * dx - size * .42 * dy) << ','
            << py(tip.y - size * dy + size * .42 * dx) << ' '
            << px(tip.x - size * dx + size * .42 * dy) << ','
            << py(tip.y - size * dy - size * .42 * dx) << "\" fill=\"" << css(item.stroke.color)
            << "\" fill-opacity=\"" << std::clamp(item.stroke.color.alpha, 0.0, 1.0) << "\"/>\n";
      }
    } else if (item.kind == "ellipse" || (item.kind == "node" && item.style.circle)) {
      out << "<ellipse cx=\"" << px(item.centre.x) << "\" cy=\"" << py(item.centre.y) << "\" rx=\""
          << item.width / 2 << "\" ry=\"" << item.height / 2 << '"';
      paint(item);
      out << "/>\n";
    } else if (item.kind == "node") {
      out << "<rect x=\"" << px(item.centre.x - item.width / 2) << "\" y=\""
          << py(item.centre.y + item.height / 2) << "\" width=\"" << item.width << "\" height=\""
          << item.height << "\" rx=\"" << item.style.corner_radius << '"';
      paint(item);
      out << "/>\n";
    } else if (item.fill.alpha != 0) {
      out << "<rect x=\"" << px(item.centre.x - item.width / 2 - 4) << "\" y=\""
          << py(item.centre.y + item.height / 2 + 3) << "\" width=\"" << item.width + 8
          << "\" height=\"" << item.height + 6 << "\" fill=\"white\"/>\n";
    }
    for (std::size_t i = 0; i < item.lines.size(); ++i) {
      const double baseline =
          py(item.centre.y) +
          (static_cast<double>(i) - (item.lines.size() - 1) / 2.0) * item.style.font_size * 1.3 +
          item.style.font_size * .32;
      // Keep selectable source text; visible outlines use the exact measured font.
      out << "<text aria-hidden=\"true\" opacity=\"0\" x=\"" << px(item.centre.x) << "\" y=\""
          << baseline << "\" text-anchor=\"middle\" font-size=\"" << item.style.font_size
          << "\" fill=\"" << css(item.style.ink) << "\" fill-opacity=\""
          << std::clamp(item.style.ink.alpha, 0.0, 1.0) << "\">" << escape(item.lines[i])
          << "</text>\n";
      out << "<g role=\"img\" aria-label=\"" << escape(item.lines[i]) << "\" fill=\""
          << css(item.style.ink) << "\" fill-opacity=\""
          << std::clamp(item.style.ink.alpha, 0.0, 1.0) << "\"><title>" << escape(item.lines[i])
          << "</title>\n";
      double pen = px(item.centre.x) - text_width(item.lines[i], item.style.font_size) / 2;
      for (const auto glyph : label_glyphs(item.lines[i])) {
        const double scale = item.style.font_size * glyph.scale / 1000;
        out << "<use href=\"#dm-g-" << glyph.index << "\" transform=\"translate(" << pen << ' '
            << baseline + glyph.shift * item.style.font_size << ") scale(" << std::setprecision(6)
            << scale << ' ' << -scale << std::setprecision(3) << ")\"/>\n";
        pen += detail::cmu_serif::glyphs[glyph.index].advance * scale;
      }
      out << "</g>\n";
    }
  }
  out << "</svg>\n";
  return out.str();
}

void Diagram::save_svg(const std::string& path) const {
  const auto    svg = to_svg();
  std::ofstream file(path, std::ios::binary);
  file << svg;
  file.flush();
  if (!file)
    throw std::runtime_error("Diagram: cannot write " + path);
}

Diagram Diagram::rooted_tree(
    const std::vector<std::size_t>& parents,
    const std::vector<std::string>& labels,
    const std::string&              title,
    std::size_t                     max_nodes) {
  const auto n = parents.size();
  if (labels.size() != n)
    throw std::invalid_argument("Diagram tree: parent/label sizes differ");
  const auto none = std::numeric_limits<std::size_t>::max();
  if (n && parents[0] != none)
    throw std::invalid_argument("Diagram tree: index0 must be the root");
  for (std::size_t i = 1; i < n; ++i)
    if (parents[i] >= n || parents[i] == i)
      throw std::invalid_argument("Diagram tree: invalid parent");
  if (n > (none - 2) / 2 || 2 * n + 2 > std::vector<Item>().max_size() ||
      n > std::vector<std::vector<std::size_t>>().max_size())
    throw std::length_error("Diagram tree: primitive capacity");
  std::vector<std::vector<std::size_t>> children(n);
  for (std::size_t i = 1; i < n; ++i)
    children[parents[i]].push_back(i);
  std::vector<std::size_t> order, stack, depth(n);
  order.reserve(n);
  stack.reserve(n);
  if (n)
    stack.push_back(0);
  while (!stack.empty()) {
    const auto id = stack.back();
    stack.pop_back();
    order.push_back(id);
    for (auto i = children[id].rbegin(); i != children[id].rend(); ++i) {
      depth[*i] = depth[id] + 1;
      stack.push_back(*i);
    }
  }
  if (order.size() != n)
    throw std::invalid_argument("Diagram tree: disconnected cycle");
  if (max_nodes && n > max_nodes)
    throw std::length_error("Diagram tree: node cap");
  Diagram d(1, 1100);
  if (!n) {
    d.text({0, 0}, "Empty tree", 24);
    if (!title.empty())
      d.text({0, 60}, title, 28);
    return d;
  }
  DiagramNodeStyle style;
  style.width     = 160;
  style.height    = 65;
  style.font_size = 20;
  style.padding   = 8;
  double width = 0, height = 0;
  for (std::size_t i = 0; i < n; ++i) {
    d.node(std::to_string(i), {0, 0}, labels[i], style);
    width  = std::max(width, d.items_.back().width);
    height = std::max(height, d.items_.back().height);
  }
  std::vector<double> left(n), right(n);
  std::size_t         leaves = 0;
  for (auto id : order)
    if (children[id].empty())
      left[id] = right[id] = double(leaves++) * (width + 40);
  for (auto i = order.rbegin(); i != order.rend(); ++i) {
    const auto id = *i;
    if (!children[id].empty()) {
      left[id]  = left[children[id].front()];
      right[id] = right[children[id].back()];
    }
    d.items_[id].centre = {(left[id] + right[id]) / 2, -double(depth[id]) * (height + 50)};
  }
  for (std::size_t i = 1; i < n; ++i)
    d.edge(std::to_string(parents[i]), std::to_string(i));
  if (!title.empty())
    d.text({d.items_[0].centre.x, height / 2 + 60}, title, 28);
  return d;
}

} // namespace datamunge::graphics
