#pragma once

/// @file plot.hpp
/// @brief std::format / std::print support for datamunge::graphics (RPlot/dm::Plot, RLayout, GGPlot,
///        Splom, dm::Histogram, colors, and the pipe-composition helpers).
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Plots print a short description (kind, canvas size, title, axis labels, layers); they are
/// never rendered. Colors print as `RGB(r, g, b)`, or `RGBA(r, g, b, alpha)` when translucent.
///
/// @par Example
/// @code
/// auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 9.0)).set_title("Squares");
/// std::print("{}\n", plt);
/// // RPlot 960x640 title="Squares"
/// //   axes: x="x", y="y"
/// //   series: 1 [Scatter n=3]
/// std::print("{}\n", dm::RGB{255, 0, 0});   // RGB(255, 0, 0)
/// @endcode

#include <datamunge/format/text_formatter.hpp>
#include <datamunge/plot/compose.hpp>
#include <datamunge/plot/ggplot.hpp>
#include <datamunge/plot/histogram.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/plot/splom.hpp>

#include <cstddef>
#include <format>
#include <string>
#include <string_view>
#include <vector>

/// @brief Prints an RGB color as "RGB(r, g, b)", or "RGBA(r, g, b, alpha)" when alpha != 1 (RGBA is the same type).
template <>
struct std::formatter<datamunge::graphics::RGB> : datamunge::format_detail::text_formatter<datamunge::graphics::RGB> {
  /// @brief The channel values (and alpha when not opaque).
  /// @param c The color to print.
  /// @return "RGB(r, g, b)" or "RGBA(r, g, b, alpha)".
  static std::string text(const datamunge::graphics::RGB& c) {
    if (c.alpha == 1.0) return std::format("RGB({}, {}, {})", c.r, c.g, c.b);
    return std::format("RGBA({}, {}, {}, {})", c.r, c.g, c.b, c.alpha);
  }
};

/// @brief Prints a DataSeries::Kind as its enumerator name (Scatter, Line, Bar, Box, Polygon, Text, Segment).
template <>
struct std::formatter<datamunge::graphics::DataSeries::Kind>
    : datamunge::format_detail::text_formatter<datamunge::graphics::DataSeries::Kind> {
  /// @brief The enumerator name, or "Kind(<int>)" for an unknown value.
  /// @param kind The layer kind to print.
  /// @return The enumerator name.
  static std::string text(datamunge::graphics::DataSeries::Kind kind) {
    using Kind = datamunge::graphics::DataSeries::Kind;
    switch (kind) {
      case Kind::Scatter: return "Scatter";
      case Kind::Line: return "Line";
      case Kind::Bar: return "Bar";
      case Kind::Box: return "Box";
      case Kind::Polygon: return "Polygon";
      case Kind::Text: return "Text";
      case Kind::Segment: return "Segment";
    }
    return std::format("Kind({})", static_cast<int>(kind));
  }
};

/// @brief Prints a DataSeries as "DataSeries(kind=.., n=.., label=.., color=..)".
template <>
struct std::formatter<datamunge::graphics::DataSeries>
    : datamunge::format_detail::text_formatter<datamunge::graphics::DataSeries> {
  /// @brief The layer kind, point count, label (when set) and color; coordinates are not dumped.
  /// @param s The layer to print.
  /// @return The one-line description.
  static std::string text(const datamunge::graphics::DataSeries& s) {
    std::string out = std::format("DataSeries(kind={}, n={}", s.kind, s.x.size());
    if (!s.label.empty()) out += std::format(", label={:?}", s.label);
    return out + std::format(", color={})", s.color);
  }
};

/// @brief Prints an ABLine as "ABLine(intercept=a, slope=b, ...)" or "ABLine(vertical, x=v, ...)".
template <>
struct std::formatter<datamunge::graphics::ABLine> : datamunge::format_detail::text_formatter<datamunge::graphics::ABLine> {
  /// @brief The line's position, color and width.
  /// @param l The reference line to print.
  /// @return The one-line description.
  static std::string text(const datamunge::graphics::ABLine& l) {
    if (l.vertical) {
      return std::format("ABLine(vertical, x={}, color={}, stroke_width={})", l.value, l.color, l.stroke_width);
    }
    return std::format("ABLine(intercept={}, slope={}, color={}, stroke_width={})", l.value, l.slope, l.color,
                       l.stroke_width);
  }
};

/// @brief Prints a LegendEntry as `LegendEntry("label", RGB(..))`.
template <>
struct std::formatter<datamunge::graphics::LegendEntry>
    : datamunge::format_detail::text_formatter<datamunge::graphics::LegendEntry> {
  /// @brief The row text and swatch color.
  /// @param e The legend row to print.
  /// @return The one-line description.
  static std::string text(const datamunge::graphics::LegendEntry& e) {
    return std::format("LegendEntry({:?}, {})", e.label, e.color);
  }
};

namespace datamunge::format_detail {

/// @brief Shared description of any Plot (base, RPlot, Histogram): header, axes, layers and extras.
/// @param plot The plot to describe.
/// @param name The type name to print first ("Plot", "RPlot", "Histogram").
/// @return Several lines: "<name> WxH title=..", "  axes: ..", "  series: N [..]", and optional extras.
inline std::string plot_description(const datamunge::graphics::Plot& plot, std::string_view name) {
  constexpr std::size_t max_series = 6;
  std::string out = std::format("{} {}x{}", name, plot.width(), plot.height());
  if (!plot.title_text().empty()) out += std::format(" title={:?}", plot.title_text());
  if (!plot.subtitle_text().empty()) out += std::format(" subtitle={:?}", plot.subtitle_text());
  out += std::format("\n  axes: x={:?}, y={:?}", plot.x_label_text(), plot.y_label_text());
  if (plot.has_x_limits()) out += std::format(", xlim=[{}, {}]", plot.x_min(), plot.x_max());
  if (plot.has_y_limits()) out += std::format(", ylim=[{}, {}]", plot.y_min(), plot.y_max());
  if (plot.axes_hidden()) out += " (hidden)";
  const auto& series = plot.series();
  out += std::format("\n  series: {} [", series.size());
  for (std::size_t i = 0; i < series.size() && i < max_series; ++i) {
    if (i > 0) out += ", ";
    out += std::format("{}", series[i].kind);
    if (!series[i].label.empty()) out += std::format(" {:?}", series[i].label);
    out += std::format(" n={}", series[i].x.size());
  }
  if (series.size() > max_series) out += ", ...";
  out += "]";
  if (!plot.reference_lines().empty()) out += std::format("\n  reference lines: {}", plot.reference_lines().size());
  if (!plot.legend_entries().empty()) {
    out += "\n  legend: [";
    for (std::size_t i = 0; i < plot.legend_entries().size(); ++i) {
      if (i > 0) out += ", ";
      out += std::format("{:?}", plot.legend_entries()[i].label);
    }
    out += "]";
  }
  return out;
}

}  // namespace datamunge::format_detail

/// @brief Prints a graphics::Plot base as a short description (size, title, axis labels, layers); never renders.
template <>
struct std::formatter<datamunge::graphics::Plot> : datamunge::format_detail::text_formatter<datamunge::graphics::Plot> {
  /// @brief The multi-line plot description.
  /// @param plot The plot to print.
  /// @return The description, starting "Plot WxH".
  static std::string text(const datamunge::graphics::Plot& plot) {
    return datamunge::format_detail::plot_description(plot, "Plot");
  }
};

/// @brief Prints an RPlot (dm::Plot) as a short description (size, title, axis labels, layers); never renders.
template <>
struct std::formatter<datamunge::graphics::RPlot> : datamunge::format_detail::text_formatter<datamunge::graphics::RPlot> {
  /// @brief The multi-line plot description.
  /// @param plot The plot to print.
  /// @return The description, starting "RPlot WxH".
  static std::string text(const datamunge::graphics::RPlot& plot) {
    return datamunge::format_detail::plot_description(plot, "RPlot");
  }
};

/// @brief Prints a dm::Histogram as a short plot description (size, title, axis labels, bars); never renders.
template <>
struct std::formatter<datamunge::Histogram> : datamunge::format_detail::text_formatter<datamunge::Histogram> {
  /// @brief The multi-line plot description.
  /// @param plot The histogram to print.
  /// @return The description, starting "Histogram WxH".
  static std::string text(const datamunge::Histogram& plot) {
    return datamunge::format_detail::plot_description(plot, "Histogram");
  }
};

/// @brief Prints an RLayout as its grid, size, title and one line per panel; never renders.
template <>
struct std::formatter<datamunge::graphics::RLayout> : datamunge::format_detail::text_formatter<datamunge::graphics::RLayout> {
  /// @brief "RLayout RxC grid WxH title=.. panels=N" then "  [i] title=.. series=k" per panel.
  /// @param layout The layout to print.
  /// @return The multi-line description.
  static std::string text(const datamunge::graphics::RLayout& layout) {
    std::string out = std::format("RLayout {}x{} grid {}x{}", layout.rows(), layout.cols(), layout.width(),
                                  layout.height());
    if (!layout.title_text().empty()) out += std::format(" title={:?}", layout.title_text());
    out += std::format(" panels={}", layout.panel_count());
    for (std::size_t i = 0; i < layout.panel_count(); ++i) {
      const auto& panel = layout.panel(i);
      out += std::format("\n  [{}] title={:?} series={}", i, panel.title_text(), panel.series().size());
    }
    return out;
  }
};

/// @brief Prints an Aes mapping as `Aes(x="..", y="..", ...)`, omitting unmapped aesthetics.
template <>
struct std::formatter<datamunge::graphics::Aes> : datamunge::format_detail::text_formatter<datamunge::graphics::Aes> {
  /// @brief The mapped column names.
  /// @param a The mapping to print.
  /// @return The one-line description.
  static std::string text(const datamunge::graphics::Aes& a) {
    std::string out = "Aes(";
    bool first = true;
    const auto field = [&](std::string_view name, const std::string& value) {
      if (value.empty()) return;
      if (!first) out += ", ";
      out += std::format("{}={:?}", name, value);
      first = false;
    };
    field("x", a.x);
    field("y", a.y);
    field("color", a.color);
    field("fill", a.fill);
    field("group", a.group);
    return out + ")";
  }
};

/// @brief Prints a GGTheme as its enumerator name (Gray, Minimal, BW, Classic).
template <>
struct std::formatter<datamunge::graphics::GGTheme> : datamunge::format_detail::text_formatter<datamunge::graphics::GGTheme> {
  /// @brief The enumerator name, or "GGTheme(<int>)" for an unknown value.
  /// @param theme The theme to print.
  /// @return The enumerator name.
  static std::string text(datamunge::graphics::GGTheme theme) {
    using datamunge::graphics::GGTheme;
    switch (theme) {
      case GGTheme::Gray: return "Gray";
      case GGTheme::Minimal: return "Minimal";
      case GGTheme::BW: return "BW";
      case GGTheme::Classic: return "Classic";
    }
    return std::format("GGTheme({})", static_cast<int>(theme));
  }
};

/// @brief Prints a GGPlot as "GGPlot rows=n aes=Aes(..) layers=[..] theme=T" plus any facet and labs() text.
template <>
struct std::formatter<datamunge::graphics::GGPlot> : datamunge::format_detail::text_formatter<datamunge::graphics::GGPlot> {
  /// @brief The data size, mapping, layer geoms, theme, and the facet/title/labels when set; never renders.
  /// @param g The plot to print.
  /// @return The one-line description.
  static std::string text(const datamunge::graphics::GGPlot& g) {
    std::string out = std::format("GGPlot rows={} aes={} layers={} theme={}", g.data().nrows(), g.mapping(),
                                  datamunge::format_detail::preview(g.layer_kinds(), 8), g.theme());
    if (!g.facet_column().empty()) out += std::format(" facet={:?}", g.facet_column());
    if (!g.title_text().empty()) out += std::format(" title={:?}", g.title_text());
    if (!g.x_label().empty()) out += std::format(" xlab={:?}", g.x_label());
    if (!g.y_label().empty()) out += std::format(" ylab={:?}", g.y_label());
    return out;
  }
};

/// @brief Prints a Splom::Diagonal as its enumerator name (Density, Histogram, Names).
template <>
struct std::formatter<datamunge::graphics::Splom::Diagonal>
    : datamunge::format_detail::text_formatter<datamunge::graphics::Splom::Diagonal> {
  /// @brief The enumerator name, or "Diagonal(<int>)" for an unknown value.
  /// @param kind The diagonal style to print.
  /// @return The enumerator name.
  static std::string text(datamunge::graphics::Splom::Diagonal kind) {
    using Diagonal = datamunge::graphics::Splom::Diagonal;
    switch (kind) {
      case Diagonal::Density: return "Density";
      case Diagonal::Histogram: return "Histogram";
      case Diagonal::Names: return "Names";
    }
    return std::format("Diagonal({})", static_cast<int>(kind));
  }
};

/// @brief Prints a Splom as its size, title, variables and color groups; never renders.
template <>
struct std::formatter<datamunge::graphics::Splom> : datamunge::format_detail::text_formatter<datamunge::graphics::Splom> {
  /// @brief "Splom WxH title=.. variables=[..] (n=k)" plus the group_by() groups when set.
  /// @param s The scatterplot matrix to print.
  /// @return The one-line description.
  static std::string text(const datamunge::graphics::Splom& s) {
    std::string out = std::format("Splom {}x{}", s.width(), s.height());
    if (!s.title_text().empty()) out += std::format(" title={:?}", s.title_text());
    out += std::format(" variables={}", datamunge::format_detail::preview(s.variables(), 8));
    if (!s.groups().empty()) out += std::format(" groups={}", datamunge::format_detail::preview(s.groups(), 8));
    return out;
  }
};

/// @brief Prints PlotArgs as `PlotArgs(title="..", ...)`, listing the options it would set.
template <>
struct std::formatter<datamunge::PlotArgs> : datamunge::format_detail::text_formatter<datamunge::PlotArgs> {
  /// @brief The configured text options (read back by applying them to a scratch plot; the argument is untouched).
  /// @param args The options to print.
  /// @return The one-line description; options the PlotArgs does not set are omitted.
  static std::string text(const datamunge::PlotArgs& args) {
    // Pre-fill with a sentinel so options explicitly set (even to "") are distinguishable from untouched ones.
    const std::string unset = "\x01";
    datamunge::Plot scratch;
    scratch.title(unset).subtitle(unset).xlab(unset).ylab(unset);
    args.apply(scratch);
    std::string out = "PlotArgs(";
    bool first = true;
    const auto field = [&](std::string_view name, const std::string& value) {
      if (value == unset) return;
      if (!first) out += ", ";
      out += std::format("{}={:?}", name, value);
      first = false;
    };
    field("title", scratch.title_text());
    field("subtitle", scratch.subtitle_text());
    field("xlab", scratch.x_label_text());
    field("ylab", scratch.y_label_text());
    return out + ")";
  }
};

/// @brief Prints PointsArgs as "PointsArgs(col=[..])".
template <>
struct std::formatter<datamunge::PointsArgs> : datamunge::format_detail::text_formatter<datamunge::PointsArgs> {
  /// @brief The configured point colors.
  /// @param args The options to print.
  /// @return "PointsArgs()" or "PointsArgs(col=[...])".
  static std::string text(const datamunge::PointsArgs& args) {
    if (args.colors().empty()) return "PointsArgs()";
    return std::format("PointsArgs(col={})", datamunge::format_detail::preview(args.colors()));
  }
};

/// @brief Prints a Points layer as "Points(n=.., col=[..])".
template <>
struct std::formatter<datamunge::Points> : datamunge::format_detail::text_formatter<datamunge::Points> {
  /// @brief The point count and colors (read back by applying the layer to a scratch plot; the argument is untouched).
  /// @param layer The layer to print.
  /// @return The one-line description.
  static std::string text(const datamunge::Points& layer) {
    datamunge::Plot scratch;
    layer.apply(scratch);
    if (scratch.series().empty()) return "Points()";
    const auto& s = scratch.series().front();
    if (s.point_colors.empty()) return std::format("Points(n={})", s.x.size());
    return std::format("Points(n={}, col={})", s.x.size(), datamunge::format_detail::preview(s.point_colors));
  }
};

namespace datamunge::format_detail {

/// @brief Formats the legend rows a LegendArgs/Legend would add, by applying it to a scratch plot.
/// @tparam Layer LegendArgs or Legend (anything with `apply(dm::Plot&) const`).
/// @param name The type name to print.
/// @param layer The legend specification (untouched).
/// @return `<name>()` or `<name>(labels=[..], col=[..])`.
template <typename Layer>
std::string legend_description(std::string_view name, const Layer& layer) {
  datamunge::Plot scratch;
  layer.apply(scratch);
  const auto& entries = scratch.legend_entries();
  if (entries.empty()) return std::format("{}()", name);
  std::vector<std::string> labels;
  std::vector<datamunge::graphics::RGB> colors;
  for (const auto& e : entries) {
    labels.push_back(e.label);
    colors.push_back(e.color);
  }
  return std::format("{}(labels={::?}, col={})", name, labels, preview(colors));
}

}  // namespace datamunge::format_detail

/// @brief Prints LegendArgs as `LegendArgs(labels=["a", "b"], col=[..])`.
template <>
struct std::formatter<datamunge::LegendArgs> : datamunge::format_detail::text_formatter<datamunge::LegendArgs> {
  /// @brief The legend rows the options would draw (colors recycled to the label count).
  /// @param args The options to print.
  /// @return "LegendArgs()" or "LegendArgs(labels=[...], col=[...])".
  static std::string text(const datamunge::LegendArgs& args) {
    return datamunge::format_detail::legend_description("LegendArgs", args);
  }
};

/// @brief Prints a Legend layer as `Legend(labels=["a", "b"], col=[..])`.
template <>
struct std::formatter<datamunge::Legend> : datamunge::format_detail::text_formatter<datamunge::Legend> {
  /// @brief The legend rows the layer would draw.
  /// @param legend The layer to print.
  /// @return "Legend()" or "Legend(labels=[...], col=[...])".
  static std::string text(const datamunge::Legend& legend) {
    return datamunge::format_detail::legend_description("Legend", legend);
  }
};

/// @brief Prints a PlotLabel::Target as its enumerator name (Title, Subtitle, XLabel, YLabel).
template <>
struct std::formatter<datamunge::PlotLabel::Target>
    : datamunge::format_detail::text_formatter<datamunge::PlotLabel::Target> {
  /// @brief The enumerator name, or "Target(<int>)" for an unknown value.
  /// @param target The label target to print.
  /// @return The enumerator name.
  static std::string text(datamunge::PlotLabel::Target target) {
    using Target = datamunge::PlotLabel::Target;
    switch (target) {
      case Target::Title: return "Title";
      case Target::Subtitle: return "Subtitle";
      case Target::XLabel: return "XLabel";
      case Target::YLabel: return "YLabel";
    }
    return std::format("Target({})", static_cast<int>(target));
  }
};

/// @brief Prints a PlotLabel as `PlotLabel(Title, "text")`.
template <>
struct std::formatter<datamunge::PlotLabel> : datamunge::format_detail::text_formatter<datamunge::PlotLabel> {
  /// @brief The label's target and text.
  /// @param label The label to print.
  /// @return The one-line description.
  static std::string text(const datamunge::PlotLabel& label) {
    return std::format("PlotLabel({}, {:?})", label.target, label.text);
  }
};
