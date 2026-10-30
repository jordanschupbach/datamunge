/// @file confusion_plot.cpp
/// @brief Compact, annotated native SVG confusion matrices with explicit normalization.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

#include "datamunge/nn/metrics.hpp"

namespace datamunge::nn {
/// @brief Private SVG escaping, color interpolation and text measurement.
namespace {
/// @brief Escape XML text and attributes, replacing unsupported control bytes.
/// @param value User text to escape.
/// @return XML-safe text retaining ordinary UTF-8 bytes.
std::string escape(const std::string& value) {
  std::string out;
  for (unsigned char c : value) {
    switch (c) {
      case '&':
        out += "&amp;";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '"':
        out += "&quot;";
        break;
      case '\'':
        out += "&apos;";
        break;
      default:
        out += (c < 32 && c != 9 && c != 10 && c != 13) ? '?' : char(c);
    }
  }
  return out;
}
/// @brief Bound monospace text width, counting non-ASCII code points as two cells.
/// @param text UTF-8 label.
/// @return Conservative displayed cell count.
std::size_t text_cells(const std::string& text) {
  std::size_t count = 0;
  for (unsigned char c : text)
    if ((c & 0xc0) != 0x80)
      count += c < 128 ? 1 : 2;
  return count;
}
/// @brief Parse a safe six-digit RGB color.
/// @param color SVG color including '#'.
/// @return Red, green and blue bytes.
/// @throws std::invalid_argument For any other color syntax.
std::array<int, 3> rgb(const std::string& color) {
  if (color.size() != 7 || color[0] != '#' ||
      color.find_first_not_of("0123456789abcdefABCDEF", 1) != std::string::npos)
    throw std::invalid_argument("ConfusionMatrix SVG: colors must be #RRGGBB");
  return {
      std::stoi(color.substr(1, 2), nullptr, 16),
      std::stoi(color.substr(3, 2), nullptr, 16),
      std::stoi(color.substr(5, 2), nullptr, 16)};
}
/// @brief Format a cell's selected count or percentage.
/// @param count Exact observation count.
/// @param value Normalized value or count used for coloring.
/// @param normalized Whether percentage text is requested.
/// @return Exact integer or percentage rounded to one decimal place.
std::string annotation(std::uint64_t count, long double value, bool normalized) {
  if (!normalized)
    return std::to_string(count);
  std::ostringstream s;
  s.imbue(std::locale::classic());
  s << std::fixed << std::setprecision(1) << 100 * value << '%';
  return s.str();
}
} // namespace

ConfusionMatrix ConfusionMatrix::from_counts(const Tensor& counts) {
  if (!counts.defined() || counts.dim() != 2 || counts.size(0) < 1 ||
      counts.size(0) != counts.size(1) || !is_integral(counts.dtype()))
    throw std::invalid_argument(
        "ConfusionMatrix::from_counts: need a nonempty square integer tensor");
  const auto values = counts.to_vector<std::int64_t>();
  if (std::any_of(values.begin(), values.end(), [](auto n) { return n < 0; }))
    throw std::invalid_argument("ConfusionMatrix::from_counts: counts must be nonnegative");
  const auto      n = static_cast<std::size_t>(counts.size(0));
  ConfusionMatrix result(n);
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t c = 0; c < n; ++c)
      result.cells_[r][c] = values[r * n + c];
  return result;
}

std::string ConfusionMatrix::to_svg(const ConfusionMatrixPlotOptions& o) const {
  if (!std::isfinite(o.cell_size) || o.cell_size < 24 || o.cell_size > 160 ||
      !std::isfinite(o.font_size) || o.font_size < 6 || o.font_size > 24 ||
      (!o.labels.empty() && o.labels.size() != classes()))
    throw std::invalid_argument("ConfusionMatrix SVG: invalid size or class labels");
  const auto  low = rgb(o.low_color), high = rgb(o.high_color);
  const char* normalization = nullptr;
  switch (o.normalization) {
    case ConfusionMatrixNormalization::None:
      normalization = "counts";
      break;
    case ConfusionMatrixNormalization::True:
      normalization = "true-row proportions";
      break;
    case ConfusionMatrixNormalization::Predicted:
      normalization = "predicted-column proportions";
      break;
    case ConfusionMatrixNormalization::All:
      normalization = "total proportions";
      break;
    default:
      throw std::invalid_argument("ConfusionMatrix SVG: unknown normalization");
  }
  std::vector<std::string> labels = o.labels;
  if (labels.empty())
    for (std::size_t i = 0; i < classes(); ++i)
      labels.push_back(std::to_string(i));
  std::size_t longest = 1;
  for (const auto& label : labels)
    longest = std::max(longest, text_cells(label));
  const double      label_width       = double(longest) * o.font_size * .65;
  const double      side              = classes() * o.cell_size;
  const double      axis_padding      = std::max(0., (15. * o.font_size * .65 - side) / 2.);
  const double      left              = std::max(label_width + 42., axis_padding + 18.);
  const double      width             = std::max(180., left + side + axis_padding + 18.);
  const double      title_font        = width < 280. ? 12. : 16.;
  const double      title_line_height = title_font + 6.;
  const std::size_t title_wrap =
      std::max<std::size_t>(4, std::size_t((width - 28) / (title_font * .65)));
  std::vector<std::string> title_lines;
  std::string              line;
  std::size_t              columns = 0;
  for (unsigned char c : o.title) {
    if ((c & 0xc0) != 0x80) {
      const auto cells = c < 128 ? 1u : 2u;
      if (columns + cells > title_wrap || c == '\n') {
        const auto split = (c == '\n' || c == ' ') ? std::string::npos : line.find_last_of(" -");
        if (split != std::string::npos && split > 0) {
          title_lines.push_back(line.substr(0, split + (line[split] == '-' ? 1 : 0)));
          line    = line.substr(split + 1);
          columns = text_cells(line);
        } else {
          title_lines.push_back(line);
          line.clear();
          columns = 0;
        }
      }
      if (c == '\n')
        continue;
      if (c == ' ' && line.empty())
        continue;
      columns += cells;
    }
    line += char(c);
  }
  if (!line.empty())
    title_lines.push_back(line);
  const double top = title_line_height * title_lines.size() + 48 +
                     std::max(0., (10. * o.font_size * .65 - side) / 2.);
  const double             bottom = label_width * .87 + 36;
  const double             height = top + side + bottom + 48;
  std::vector<long double> row(classes()), column(classes());
  long double              total = 0, maximum = 1;
  for (std::size_t r = 0; r < classes(); ++r)
    for (std::size_t c = 0; c < classes(); ++c) {
      row[r] += cells_[r][c];
      column[c] += cells_[r][c];
      total += cells_[r][c];
      maximum = std::max(maximum, static_cast<long double>(cells_[r][c]));
    }
  const bool normalized = o.normalization != ConfusionMatrixNormalization::None;
  if (normalized)
    maximum = 1;
  std::ostringstream s;
  s.imbue(std::locale::classic());
  s << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
    << "\" viewBox=\"0 0 " << width << ' ' << height
    << "\" role=\"img\" aria-labelledby=\"cm-title cm-description\">\n"
    << "<title id=\"cm-title\">" << escape(o.title)
    << "</title><desc id=\"cm-description\">True classes are rows from top to bottom; predicted "
       "classes are columns from left to right. "
    << normalization << "; zero support produces zero. Cell tooltips retain exact counts.</desc>\n"
    << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/><g id=\"cm-content\" "
       "font-family=\"monospace\" "
       "fill=\"#172033\" font-size=\""
    << o.font_size << "\">\n";
  double title_y = 20;
  for (const auto& title : title_lines) {
    s << "<text x=\"14\" y=\"" << title_y << "\" font-size=\"" << title_font
      << "\" font-weight=\"bold\">" << escape(title) << "</text>\n";
    title_y += title_line_height;
  }
  s << "<text x=\"" << left + side / 2 << "\" y=\"" << top - 14
    << "\" text-anchor=\"middle\">Predicted class</text>\n"
    << "<text transform=\"translate(" << std::max(14., o.font_size) << ' ' << top + side / 2
    << ") rotate(-90)\" text-anchor=\"middle\">True class</text>\n";
  for (std::size_t r = 0; r < classes(); ++r) {
    const double cy = top + (r + .5) * o.cell_size;
    s << "<text x=\"" << left - 8 << "\" y=\"" << cy + o.font_size * .3 << "\" text-anchor=\"end\">"
      << escape(labels[r]) << "</text>\n";
    const double cx = left + (r + .5) * o.cell_size;
    s << "<text transform=\"translate(" << cx << ' ' << top + side + 12
      << ") rotate(-60)\" text-anchor=\"end\">" << escape(labels[r]) << "</text>\n";
    for (std::size_t c = 0; c < classes(); ++c) {
      const auto  count       = cells_[r][c];
      long double denominator = 1;
      if (o.normalization == ConfusionMatrixNormalization::True)
        denominator = row[r];
      if (o.normalization == ConfusionMatrixNormalization::Predicted)
        denominator = column[c];
      if (o.normalization == ConfusionMatrixNormalization::All)
        denominator = total;
      const long double  value    = denominator ? count / denominator : 0;
      const double       fraction = double(value / maximum);
      std::array<int, 3> color;
      for (int k = 0; k < 3; ++k)
        color[k] = int(std::lround(low[k] + fraction * (high[k] - low[k])));
      const double x = left + c * o.cell_size, y = top + r * o.cell_size;
      s << "<g data-true=\"" << r << "\" data-predicted=\"" << c << "\" data-count=\"" << count
        << "\" data-value=\"" << value << "\"><title>" << escape(labels[r]) << " → "
        << escape(labels[c]) << ": " << count << "</title>\n"
        << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << o.cell_size << "\" height=\""
        << o.cell_size << "\" fill=\"rgb(" << color[0] << ',' << color[1] << ',' << color[2]
        << ")\" stroke=\"white\"/>\n";
      if (o.show_values) {
        const auto text = annotation(count, value, normalized);
        const auto font = std::min(o.font_size, (o.cell_size - 8) / (text.size() * .65));
        s << "<text x=\"" << x + o.cell_size / 2 << "\" y=\"" << y + o.cell_size / 2 + font * .3
          << "\" font-size=\"" << font << "\" text-anchor=\"middle\" fill=\""
          << (color[0] * .299 + color[1] * .587 + color[2] * .114 < 145 ? "white" : "#172033")
          << "\">" << text << "</text>\n";
      }
      s << "</g>\n";
    }
  }
  const double legend_y = top + side + bottom;
  s << "<defs><linearGradient id=\"cm-scale\"><stop stop-color=\"" << o.low_color
    << "\"/><stop offset=\"1\" stop-color=\"" << o.high_color << "\"/></linearGradient></defs>\n"
    << "<rect x=\"" << left << "\" y=\"" << legend_y << "\" width=\"" << side
    << "\" height=\"10\" fill=\"url(#cm-scale)\"/>\n"
    << "<text x=\"" << left << "\" y=\"" << legend_y + 25 << "\">0</text><text x=\"" << left + side
    << "\" y=\"" << legend_y + 25 << "\" text-anchor=\"end\">"
    << (normalized ? "100%" : annotation(static_cast<std::uint64_t>(maximum), maximum, false))
    << "</text>\n</g></svg>\n";
  return s.str();
}

void ConfusionMatrix::save_svg(const std::string& path, const ConfusionMatrixPlotOptions& options)
    const {
  const auto    svg = to_svg(options);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file || !(file << svg))
    throw std::runtime_error("ConfusionMatrix::save_svg: cannot write " + path);
  file.close();
  if (!file)
    throw std::runtime_error("ConfusionMatrix::save_svg: cannot finish " + path);
}
} // namespace datamunge::nn
