/// @file model_plot_render.cpp
/// @brief Native left-to-right flow, graph and LeNet-style SVG architecture renderers.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "datamunge/nn/model_plot.hpp"

namespace datamunge::nn {
/// @brief Private SVG labels, category colors and tensor-proportional glyph geometry.
namespace {
/// @brief Escape user text for XML 1.0 element content and attributes.
std::string xml(const std::string& text) {
  std::string out;
  for (const unsigned char c : text) {
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
/// @brief Wrap complete UTF-8 code points, preserving every character except whitespace breaks.
std::vector<std::string> wrap(const std::string& text, std::size_t width = 30) {
  std::vector<std::string> lines;
  std::string              line;
  std::size_t              columns = 0;
  for (std::size_t i = 0; i < text.size();) {
    const auto        first = static_cast<unsigned char>(text[i]);
    const std::size_t count = first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
    if (columns == width || text[i] == '\n') {
      const auto space = line.find_last_of(' ');
      if (text[i] != '\n' && space != std::string::npos && space > width / 3) {
        lines.push_back(line.substr(0, space));
        line.erase(0, space + 1);
        columns = std::count_if(
            line.begin(), line.end(), [](unsigned char c) { return (c & 0xc0) != 0x80; });
      } else {
        lines.push_back(line);
        line.clear();
        columns = 0;
      }
      if (text[i] == '\n') {
        ++i;
        continue;
      }
      if (line.empty() && text[i] == ' ') {
        ++i;
        continue;
      }
    }
    line += text.substr(i, count);
    i += count;
    ++columns;
  }
  if (!line.empty() || lines.empty())
    lines.push_back(line);
  return lines;
}
/// @brief Format actual dimensions in an ASCII label.
std::string dimensions(const Shape& shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i)
    s += (i ? " x " : "") + std::to_string(shape[i]);
  return s + "]";
}

/// @brief Semantic categories shared by the legend and every drawing style.
enum class Category {
  Input,      ///< External example input.
  Conv,       ///< Convolution or depthwise convolution.
  Dense,      ///< Dense or classifier transform.
  Pool,       ///< Pooling or spatial averaging.
  Activation, ///< Elementwise nonlinear activation.
  Norm,       ///< Normalization.
  Block,      ///< Named composite block.
  Merge,      ///< Additive or concatenation join.
  Output,     ///< Chosen forward output.
  Operation,  ///< Other tensor primitive.
  Parameter   ///< Explicit gradient leaf.
};
/// @brief Assign a category from recorded layer metadata, falling back to its display label.
Category category(const ModelPlotNode& node) {
  if (node.kind == "Input")
    return Category::Input;
  if (node.kind == "Output")
    return Category::Output;
  if (node.kind == "Parameter")
    return Category::Parameter;
  auto s = node.label + " " + node.kind;
  std::transform(
      s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (s.find("conv") != s.npos)
    return Category::Conv;
  if (s.find("dense") != s.npos || s.find("fc6") != s.npos || s.find("fc7") != s.npos ||
      s.find("classifier") != s.npos)
    return Category::Dense;
  if (s.find("pool") != s.npos || s.find("average") != s.npos)
    return Category::Pool;
  if (s.find("norm") != s.npos)
    return Category::Norm;
  if (s.find("relu") != s.npos || s.find("tanh") != s.npos || s.find("sigmoid") != s.npos ||
      s.find("gelu") != s.npos || s.find("silu") != s.npos)
    return Category::Activation;
  if (node.kind == "Composite stage")
    return Category::Block;
  if (node.label == "Add" || s.find("concat") != s.npos)
    return Category::Merge;
  return Category::Operation;
}
/// @brief Borrow all category colors in enum order.
std::array<std::string, 11> palette(const ModelPlotPalette& p) {
  return {
      p.input,
      p.convolution,
      p.dense,
      p.pooling,
      p.activation,
      p.normalization,
      p.block,
      p.merge,
      p.output,
      p.operation,
      p.parameter};
}
/// @brief Human-readable category names in palette order.
const std::array<const char*, 11> category_names{
    "Input",
    "Convolution",
    "Dense",
    "Pooling",
    "Activation",
    "Normalization",
    "Block",
    "Merge",
    "Output",
    "Operation",
    "Parameter"};
/// @brief Render a compact style name without relying on integer enum values.
const char* style_name(ModelPlotStyle style) {
  switch (style) {
    case ModelPlotStyle::Flow:
      return "Flow";
    case ModelPlotStyle::Graph:
      return "Graph";
    case ModelPlotStyle::LeNet:
      return "LeNet";
  }
  throw std::invalid_argument("model plot: unknown drawing style");
}
/// @brief Render the effective detail level, including the compatibility override.
const char* detail_name(const ModelPlotOptions& options) {
  if (options.expand_layers || options.detail == ModelPlotDetail::Operations)
    return "operations";
  return options.detail == ModelPlotDetail::Layers ? "layers" : "blocks";
}
/// @brief Bounded feature glyph with optional representative dense-unit indices.
struct Glyph {
  double                    face_width{50};   ///< Front-face horizontal span in pixels.
  double                    face_height{100}; ///< Front-face vertical span in pixels.
  double                    depth{0};         ///< Isometric depth or stacked-plane horizontal span.
  double                    width{50};   ///< Complete graphical bounding width, excluding labels.
  double                    height{100}; ///< Complete graphical bounding height, excluding labels.
  int                       planes{1};   ///< Number of representative feature-map planes.
  std::vector<std::int64_t> units;       ///< Sampled real unit indices; -1 denotes an ellipsis.
};
/// @brief Spatial face axes, channels and vector length before diagram-wide scaling.
struct Dimensions {
  double width{1};       ///< Last spatial dimension; one for vectors and scalars.
  double height{1};      ///< Product of preceding spatial axes, excluding batch.
  double channels{1};    ///< Channels-last count for spatial tensors.
  double vector{1};      ///< Feature count for rank-one/two tensors.
  bool   spatial{false}; ///< Whether the tensor has batch, space and channel axes.
};
/// @brief Interpret channels-last tensor shapes, excluding their batch dimension.
/// @param shape Actual output dimensions; empty shapes represent scalars.
/// @return Spatial axes/channel count, or a vector length, with scalar dimensions of one.
Dimensions tensor_dimensions(const Shape& shape) {
  Dimensions d;
  d.spatial = shape.size() >= 3;
  if (d.spatial) {
    d.channels = double(shape.back());
    d.width    = double(shape[shape.size() - 2]);
    for (std::size_t axis = 1; axis + 2 < shape.size(); ++axis)
      d.height *= double(shape[axis]);
  } else if (!shape.empty()) {
    d.vector = double(shape.back());
  }
  return d;
}
/// @brief Convert a nonnegative dimension to pixels with a shared reference and explicit floor.
/// @param value Actual nonnegative dimension length.
/// @param reference Diagram-wide maximum for this dimension family, at least one.
/// @param pixels Pixel extent assigned to the reference dimension.
/// @param scale Validated linear, square-root or logarithmic transform.
/// @param minimum Positive pixel visibility floor, applied after scaling.
/// @return Finite pixel length at or above minimum for valid tensor dimensions.
double scaled(double value, double reference, double pixels, ModelPlotScale scale, double minimum) {
  double ratio = value / reference;
  if (scale == ModelPlotScale::SquareRoot)
    ratio = std::sqrt(ratio);
  else if (scale == ModelPlotScale::Logarithmic)
    ratio = std::log1p(value) / std::log1p(reference);
  return std::max(minimum, pixels * ratio);
}
/// @brief Determine glyph envelopes from shared spatial, channel and vector references.
/// @param node Recorded node whose output shape determines geometry.
/// @param options Validated style, dimension scale and sampling controls.
/// @param maximum Shared maxima for spatial axes, channel depth and vector length.
/// @return Face/envelope dimensions and representative sheet or neuron samples.
Glyph glyph(const ModelPlotNode& node, const ModelPlotOptions& options, const Dimensions& maximum) {
  Glyph       g;
  const auto  d = tensor_dimensions(node.shape);
  const auto& s = options.sizing;
  if (d.spatial) {
    g.face_width =
        scaled(d.width, maximum.width, s.spatial_size, s.spatial_scale, s.minimum_extent);
    g.face_height =
        scaled(d.height, maximum.width, s.spatial_size, s.spatial_scale, s.minimum_extent);
    g.depth =
        scaled(d.channels, maximum.channels, s.channel_size, s.channel_scale, s.minimum_extent);
  } else {
    g.face_width = 18.;
    g.face_height =
        scaled(d.vector, maximum.vector, s.spatial_size, s.channel_scale, s.minimum_extent);
    g.depth = s.minimum_extent;
  }
  g.width  = g.face_width + g.depth;
  g.height = g.face_height + g.depth;
  if (options.style == ModelPlotStyle::Graph) {
    if (options.show_neurons && node.shape.size() == 2 &&
        (node.kind == "Input" || node.kind.starts_with("Dense"))) {
      const auto count = node.shape.back();
      if (count <= options.max_neurons) {
        for (std::int64_t i = 0; i < count; ++i)
          g.units.push_back(i);
      } else {
        const int first = options.max_neurons / 2, last = options.max_neurons - first;
        for (int i = 0; i < first; ++i)
          g.units.push_back(i);
        g.units.push_back(-1);
        for (auto i = count - last; i < count; ++i)
          g.units.push_back(i);
      }
    }
  } else if (options.style == ModelPlotStyle::LeNet) {
    g.planes = d.spatial ? std::max(1, int(std::min(d.channels, double(s.max_planes)))) : 1;
  }
  return g;
}
/// @brief Pixel ordinate of a sampled unit distributed inside its shape-scaled envelope.
/// @param g Glyph with a nonempty unit sample.
/// @param index Index into that sample.
/// @param center Vertical center of the glyph in pixels.
/// @return Vertical coordinate in SVG pixels.
double unit_y(const Glyph& g, std::size_t index, double center) {
  return center + g.height * ((double(index) + .5) / double(g.units.size()) - .5);
}
/// @brief Human-readable transform name for scale disclosures and SVG metadata.
/// @param scale Selected dimension transform.
/// @return Borrowed static transform name.
/// @throws std::invalid_argument If scale is not a declared enumeration value.
const char* scale_name(ModelPlotScale scale) {
  switch (scale) {
    case ModelPlotScale::Linear:
      return "linear";
    case ModelPlotScale::SquareRoot:
      return "square-root";
    case ModelPlotScale::Logarithmic:
      return "logarithmic";
  }
  throw std::invalid_argument("model plot: unknown dimension scale");
}
} // namespace

namespace model_plot_detail {
void validate_options(const ModelPlotOptions& o) {
  if ((o.width != 0 && (o.width < 400 || o.width > 10000)) || o.max_neurons < 2 ||
      o.max_neurons > 32)
    throw std::invalid_argument("model plot: width must be 0 or 400..10000 and max_neurons 2..32");
  (void)style_name(o.style);
  if (o.detail != ModelPlotDetail::Blocks && o.detail != ModelPlotDetail::Layers &&
      o.detail != ModelPlotDetail::Operations)
    throw std::invalid_argument("model plot: unknown detail level");
  if (o.labels.type != ModelPlotLabel::Full && o.labels.type != ModelPlotLabel::Name &&
      o.labels.type != ModelPlotLabel::Shape && o.labels.type != ModelPlotLabel::NameAndShape &&
      o.labels.type != ModelPlotLabel::None)
    throw std::invalid_argument("model plot: unknown label content");
  (void)scale_name(o.sizing.spatial_scale);
  (void)scale_name(o.sizing.channel_scale);
  /// @brief Reject nonfinite numeric settings and values outside their documented ranges.
  const auto range = [](double value, double low, double high) {
    if (!std::isfinite(value) || value < low || value > high)
      throw std::invalid_argument("model plot: numeric setting outside its documented range");
  };
  range(o.opacity, 0., 1.);
  range(o.edge_opacity, 0., 1.);
  range(o.funnel_opacity, 0., 1.);
  range(o.line_width, std::numeric_limits<double>::min(), 10.);
  range(o.labels.size, 6., 72.);
  range(o.labels.wrap, 8., 120.);
  range(o.sizing.spatial_size, 16., 2000.);
  range(o.sizing.channel_size, 4., 1000.);
  range(o.sizing.minimum_extent, std::numeric_limits<double>::min(), 16.);
  range(o.sizing.max_planes, 1., 32.);
  range(o.spacing.layer, 8., 1000.);
  range(o.spacing.lane, 8., 1000.);
  range(o.spacing.label, 0., 1000.);
  range(o.spacing.margin, 16., 1000.);
  auto                     colors = palette(o.colors);
  std::vector<std::string> all_colors(colors.begin(), colors.end());
  all_colors.insert(
      all_colors.end(), {o.colors.background, o.colors.text, o.colors.edge, o.colors.outline});
  for (const auto& color : all_colors)
    if (color.size() != 7 || color[0] != '#' ||
        !std::all_of(
            color.begin() + 1, color.end(), [](unsigned char c) { return std::isxdigit(c); }))
      throw std::invalid_argument("model plot: palette colors must be #RRGGBB");
}
} // namespace model_plot_detail

std::string ModelDiagram::to_svg() const {
  /// @brief Forward ranks become horizontal columns; parallel branches occupy vertical lanes.
  std::vector<std::vector<std::size_t>> columns;
  std::vector<std::size_t>              rank(nodes_.size());
  std::vector<bool>                     from_input(nodes_.size());
  std::vector<std::vector<std::size_t>> consumers(nodes_.size());
  std::vector<double>                   x(nodes_.size()), y(nodes_.size());
  std::vector<std::vector<std::string>> labels(nodes_.size());
  std::vector<Glyph>                    glyphs;
  std::vector<Category>                 categories;
  const auto                            colors = palette(options_.colors);
  std::set<Category>                    used;
  Dimensions                            maximum;
  for (const auto& n : nodes_) {
    const auto d = tensor_dimensions(n.shape);
    if (d.spatial) {
      maximum.width    = std::max({maximum.width, d.width, d.height});
      maximum.channels = std::max(maximum.channels, d.channels);
    } else {
      maximum.vector = std::max(maximum.vector, d.vector);
    }
  }
  const auto&         label       = options_.labels;
  const auto&         spacing     = options_.spacing;
  const double        line_height = label.size * 1.3;
  const auto          label_type  = label.type;
  std::vector<double> label_heights(nodes_.size());
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    const auto& n = nodes_[i];
    from_input[i] = n.kind == "Input";
    for (auto parent : n.inputs) {
      rank[i]       = std::max(rank[i], rank[parent] + 1);
      from_input[i] = from_input[i] || from_input[parent];
      consumers[parent].push_back(i);
    }
    categories.push_back(category(n));
    used.insert(categories.back());
    if (label_type == ModelPlotLabel::Full || label_type == ModelPlotLabel::Name ||
        label_type == ModelPlotLabel::NameAndShape)
      labels[i] = wrap(n.label, label.wrap);
    if (label_type == ModelPlotLabel::Full && n.kind != n.label) {
      const auto kind = wrap(n.kind, label.wrap);
      labels[i].insert(labels[i].end(), kind.begin(), kind.end());
    }
    if (label_type == ModelPlotLabel::Full || label_type == ModelPlotLabel::Shape ||
        label_type == ModelPlotLabel::NameAndShape) {
      const auto shape = wrap(dimensions(n.shape), label.wrap);
      labels[i].insert(labels[i].end(), shape.begin(), shape.end());
    }
    if (label_type == ModelPlotLabel::Full && n.parameters) {
      const auto count = wrap(std::to_string(n.parameters) + " learned values", label.wrap);
      labels[i].insert(labels[i].end(), count.begin(), count.end());
    }
    for (const auto& line : labels[i]) {
      std::size_t cells = 0;
      for (unsigned char c : line)
        if ((c & 0xc0) != 0x80)
          cells += c < 128 ? 1 : 2;
      label_heights[i] = std::max(label_heights[i], (cells * .65 + 1.) * label.size);
    }
    glyphs.push_back(glyph(n, options_, maximum));
  }
  // Pack parameter-only transforms just before their earliest consumer. Moving
  // the complete chain also handles reshaped/depthwise weights without a tall
  // stack of independent preprocessing branches beside the input.
  for (std::size_t i = nodes_.size(); i-- > 0;) {
    if (from_input[i] || consumers[i].empty())
      continue;
    std::size_t earliest = nodes_.size();
    for (auto child : consumers[i])
      earliest = std::min(earliest, rank[child]);
    if (earliest > 0)
      rank[i] = earliest - 1;
  }
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (columns.size() <= rank[i])
      columns.resize(rank[i] + 1);
    columns[rank[i]].push_back(i);
  }
  std::vector<double> column_width(columns.size());
  for (std::size_t c = 0; c < columns.size(); ++c) {
    for (auto i : columns[c])
      column_width[c] = std::max(
          column_width[c],
          std::max(glyphs[i].width, label.size + line_height * double(labels[i].size())));
  }
  const double margin = spacing.margin, gap = spacing.layer;
  double       width = 2. * margin + gap * double(columns.empty() ? 0 : columns.size() - 1);
  for (auto w : column_width)
    width += w;
  const double extra = std::max(0., double(options_.width) - width);
  width += extra;
  const double header_width =
      std::clamp((options_.width ? double(options_.width) : width) - 2. * margin, 240., 1000.);
  width              = std::max(width, header_width + 2. * margin);
  const auto title   = wrap(options_.title, std::size_t(header_width / 12));
  const auto summary = wrap(
      std::to_string(nodes_.size()) + " nodes | " + std::to_string(parameters_) +
          " unique learned values | " + style_name(options_.style) + " / " + detail_name(options_),
      std::size_t(header_width / 7));
  const int  legend_columns = std::max(1, int(header_width / 130));
  const auto legend_rows =
      options_.legend ? (used.size() + legend_columns - 1) / legend_columns : 0;
  std::ostringstream note_text;
  note_text.imbue(std::locale::classic());
  note_text << "Shape scale: spatial " << scale_name(options_.sizing.spatial_scale)
            << "; channels/vectors " << scale_name(options_.sizing.channel_scale) << "; floor "
            << options_.sizing.minimum_extent
            << " px. Sheets/neurons sampled; exact shapes in labels/tooltips.";
  const auto   note = wrap(note_text.str(), std::size_t(header_width / 6));
  const double header =
      margin + 22. * double(title.size()) + 18. * double(summary.size()) +
      (options_.legend ? 22. * double(legend_rows) + 20. + 16. * double(note.size()) : 16.);
  double height = header + margin;
  double left   = margin;
  for (std::size_t c = 0; c < columns.size(); ++c) {
    auto& column = columns[c];
    std::stable_sort(column.begin(), column.end(), [&](auto a, auto b) {
      /// @brief Order parallel nodes by their parents' mean vertical position.
      const auto center = [&](auto i) {
        double value = 0.;
        for (auto j : nodes_[i].inputs)
          value += y[j];
        return nodes_[i].inputs.empty() ? 0. : value / double(nodes_[i].inputs.size());
      };
      return center(a) < center(b);
    });
    double cursor = header + 32.;
    for (std::size_t r = 0; r < column.size(); ++r) {
      const auto i = column[r];
      x[i]         = left + column_width[c] / 2.;
      y[i]         = cursor + glyphs[i].height / 2.;
      cursor += glyphs[i].height + (labels[i].empty() ? 0. : spacing.label + label_heights[i]);
      if (r + 1 < column.size())
        cursor += spacing.lane;
    }
    height = std::max(height, cursor + margin);
    left += column_width[c] + gap + (columns.size() > 1 ? extra / double(columns.size() - 1) : 0.);
  }
  std::ostringstream svg;
  svg.imbue(std::locale::classic());
  svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
      << "\" viewBox=\"0 0 " << width << ' ' << height
      << "\" role=\"img\" data-model-layout=\"left-to-right\""
      << " data-model-style=\"" << style_name(options_.style) << "\" data-model-detail=\""
      << detail_name(options_) << "\" aria-labelledby=\"model-title model-description\">\n"
      << "<title id=\"model-title\">" << xml(options_.title) << "</title>\n"
      << "<desc id=\"model-description\">Left-to-right executed differentiable forward graph; "
      << "node labels rotate 90 degrees counterclockwise. Shapes and dependencies are exact. "
      << "Glyphs use shared shape scales and an explicit visibility floor. Feature-map sheets "
         "and neurons are sampled. "
      << "Constants and detached paths are not expanded.</desc>\n"
      << "<rect width=\"100%\" height=\"100%\" fill=\"" << options_.colors.background << "\"/>\n"
      << "<defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" "
         "markerWidth=\"6\" "
         "markerHeight=\"6\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" "
         "fill=\""
      << options_.colors.edge << "\"/></marker></defs>\n"
      << "<g font-family=\"sans-serif\" fill=\"" << options_.colors.text << "\">\n";
  double header_y = margin;
  for (const auto& line : title) {
    svg << "<text x=\"" << margin << "\" y=\"" << header_y
        << "\" font-size=\"20\" font-weight=\"600\">" << xml(line) << "</text>\n";
    header_y += 22.;
  }
  header_y += 5.;
  for (const auto& line : summary) {
    svg << "<text x=\"" << margin << "\" y=\"" << header_y << "\" font-size=\"12\">" << xml(line)
        << "</text>\n";
    header_y += 18.;
  }
  if (options_.legend) {
    std::size_t index = 0;
    for (const auto c : used) {
      const double lx = margin + 130. * double(index % legend_columns),
                   ly = header_y + 22. * double(index / legend_columns);
      svg << "<rect x=\"" << lx << "\" y=\"" << ly - 10.
          << "\" width=\"12\" height=\"12\" rx=\"2\" fill=\"" << colors[std::size_t(c)]
          << "\"/><text x=\"" << lx + 18. << "\" y=\"" << ly << "\" font-size=\"11\">"
          << category_names[std::size_t(c)] << "</text>\n";
      ++index;
    }
    for (std::size_t line = 0; line < note.size(); ++line)
      svg << "<text x=\"" << margin << "\" y=\""
          << header_y + 22. * double(legend_rows) + 8. + 16. * double(line)
          << "\" font-size=\"11\">" << xml(note[line]) << "</text>\n";
  }
  // Every dependency leaves the source's right port and enters the destination's left port.
  for (std::size_t i = 0; i < nodes_.size(); ++i)
    for (const auto j : nodes_[i].inputs) {
      const auto & a = glyphs[j], &b = glyphs[i];
      const double from = x[j] + a.width / 2., to = x[i] - b.width / 2.;
      const auto&  color    = colors[std::size_t(categories[i])];
      const bool   adjacent = rank[i] == rank[j] + 1;
      if (options_.style == ModelPlotStyle::Flow && adjacent)
        svg << "<polygon class=\"flow-funnel\" points=\"" << from << ',' << y[j] - a.height / 2.
            << ' ' << to << ',' << y[i] - b.height / 2. << ' ' << to << ',' << y[i] + b.height / 2.
            << ' ' << from << ',' << y[j] + a.height / 2. << "\" fill=\"" << color
            << "\" fill-opacity=\"" << options_.funnel_opacity << "\"/>\n";
      if (options_.style == ModelPlotStyle::LeNet && adjacent)
        svg << "<path class=\"feature-connector\" d=\"M " << from << ' ' << y[j] - a.height / 2.
            << " L " << to << ' ' << y[i] - b.height / 2. << " M " << from << ' '
            << y[j] + a.height / 2. << " L " << to << ' ' << y[i] + b.height / 2. << "\" stroke=\""
            << color << "\" stroke-opacity=\"" << options_.edge_opacity << "\" stroke-width=\""
            << options_.line_width << "\" fill=\"none\"/>\n";
      if (options_.style == ModelPlotStyle::Graph && adjacent &&
          nodes_[i].kind.starts_with("Dense") && !a.units.empty() && !b.units.empty())
        for (std::size_t u = 0; u < a.units.size(); ++u)
          if (a.units[u] >= 0)
            for (std::size_t v = 0; v < b.units.size(); ++v)
              if (b.units[v] >= 0)
                svg << "<path class=\"neuron-connection\" d=\"M " << from << ' '
                    << unit_y(a, u, y[j]) << " L " << to << ' ' << unit_y(b, v, y[i])
                    << "\" fill=\"none\" stroke=\"" << options_.colors.edge << "\" stroke-width=\""
                    << options_.line_width * .56 << "\" stroke-opacity=\"" << options_.edge_opacity
                    << "\"/>\n";
      svg << "<path data-from=\"" << j << "\" data-to=\"" << i << "\" d=\"M " << from << ' '
          << y[j];
      if (adjacent) {
        const double middle = (from + to) / 2.;
        svg << " C " << middle << ' ' << y[j] << ' ' << middle << ' ' << y[i] << ' ' << to << ' '
            << y[i];
      } else {
        const double lane = header + 12. + 8. * double(j % 3),
                     bend = std::min(16., (to - from) / 5.);
        svg << " C " << from + bend << ' ' << y[j] << ' ' << from + bend << ' ' << lane << ' '
            << from + 2. * bend << ' ' << lane << " L " << to - 2. * bend << ' ' << lane << " C "
            << to - bend << ' ' << lane << ' ' << to - bend << ' ' << y[i] << ' ' << to << ' '
            << y[i];
      }
      svg << "\" fill=\"none\" stroke=\"" << options_.colors.edge << "\" stroke-opacity=\""
          << options_.edge_opacity << "\" stroke-width=\"" << options_.line_width
          << "\" marker-end=\"url(#arrow)\"/>\n";
    }
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    const auto& g     = glyphs[i];
    const auto& color = colors[std::size_t(categories[i])];
    svg << "<g data-node=\"" << i << "\" data-category=\""
        << category_names[std::size_t(categories[i])] << "\"><title>"
        << xml(nodes_[i].label + " " + dimensions(nodes_[i].shape)) << "</title>\n"
        << "<g id=\"model-glyph-" << i << "\" data-left=\"" << x[i] - g.width / 2.
        << "\" data-right=\"" << x[i] + g.width / 2. << "\" data-face-width=\"" << g.face_width
        << "\" data-face-height=\"" << g.face_height << "\" data-depth=\"" << g.depth
        << "\" stroke=\"" << options_.colors.outline << "\" stroke-width=\"" << options_.line_width
        << "\" fill-opacity=\"" << options_.opacity << "\">\n";
    if (options_.style == ModelPlotStyle::Graph) {
      if (g.units.empty()) {
        svg << "<rect x=\"" << x[i] - g.width / 2. << "\" y=\"" << y[i] - g.height / 2.
            << "\" width=\"" << g.width << "\" height=\"" << g.height << "\" rx=\""
            << std::min({14., g.width / 4., g.height / 4.}) << "\" fill=\"" << color << "\"/>\n";
      } else
        for (std::size_t u = 0; u < g.units.size(); ++u) {
          const double uy     = unit_y(g, u, y[i]);
          const double radius = std::min(7., g.height / (2. * double(g.units.size())));
          if (g.units[u] < 0) {
            for (int dot = -1; dot <= 1; ++dot)
              svg << "<circle cx=\"" << x[i] << "\" cy=\"" << uy + radius * .55 * dot << "\" r=\""
                  << radius * .17 << "\" stroke=\"none\" fill=\"" << options_.colors.text
                  << "\"/>\n";
          } else {
            svg << "<circle data-unit=\"" << g.units[u] << "\" cx=\"" << x[i] << "\" cy=\"" << uy
                << "\" r=\"" << radius << "\" fill=\"" << color << "\"/>\n";
          }
        }
    } else if (options_.style == ModelPlotStyle::Flow) {
      const double fx = x[i] - g.width / 2., fy = y[i] - g.face_height / 2. + g.depth / 2.;
      const double w = g.face_width, h = g.face_height, d = g.depth;
      svg << "<polygon class=\"volume-top\" points=\"" << fx << ',' << fy << ' ' << fx + d << ','
          << fy - d << ' ' << fx + w + d << ',' << fy - d << ' ' << fx + w << ',' << fy
          << "\" fill=\"" << color << "\"/>\n"
          << "<polygon class=\"volume-side\" points=\"" << fx + w << ',' << fy << ' ' << fx + w + d
          << ',' << fy - d << ' ' << fx + w + d << ',' << fy + h - d << ' ' << fx + w << ','
          << fy + h << "\" fill=\"" << color << "\"/>\n"
          << "<rect class=\"volume-front\" x=\"" << fx << "\" y=\"" << fy << "\" width=\"" << w
          << "\" height=\"" << h << "\" fill=\"" << color << "\"/>\n";
    } else {
      // The depth guide preserves the channel envelope even with one sampled sheet.
      const double front_x = x[i] - g.width / 2.;
      const double front_y = y[i] - g.face_height / 2. + g.depth / 2.;
      svg << "<path class=\"feature-depth\" d=\"M " << front_x << ' ' << front_y << " L "
          << front_x + g.depth << ' ' << front_y - g.depth << "\" fill=\"none\"/>\n";
      for (int plane = g.planes - 1; plane >= 0; --plane) {
        const double offset = g.planes > 1 ? g.depth * plane / (g.planes - 1) : 0.;
        const double fx     = x[i] - g.width / 2. + offset;
        const double fy     = y[i] - g.face_height / 2. + (g.height - g.face_height) / 2. - offset;
        svg << "<rect class=\"feature-plane\" data-plane=\"" << plane << "\" x=\"" << fx
            << "\" y=\"" << fy << "\" width=\"" << g.face_width << "\" height=\"" << g.face_height
            << "\" fill=\"" << color << "\"/>\n";
      }
    }
    svg << "</g>\n";
    if (labels[i].empty()) {
      svg << "</g>\n";
      continue;
    }
    const double label_y = y[i] + g.height / 2. + spacing.label + label_heights[i] / 2.;
    svg << "<g id=\"model-label-" << i << "\" data-node-label=\"" << i
        << "\" transform=\"translate(" << x[i] << ' ' << label_y
        << ") rotate(-90)\" text-anchor=\"middle\" font-family=\"monospace\">\n";
    for (std::size_t line = 0; line < labels[i].size(); ++line)
      svg << "<text x=\"0\" y=\""
          << line_height * (double(line) - double(labels[i].size() - 1) / 2.) + label.size * .3
          << "\" font-size=\"" << label.size << "\"" << (line == 0 ? " font-weight=\"600\"" : "")
          << '>' << xml(labels[i][line]) << "</text>\n";
    svg << "</g></g>\n";
  }
  svg << "</g></svg>\n";
  return svg.str();
}
} // namespace datamunge::nn
