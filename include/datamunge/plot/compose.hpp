#pragma once

/// @file compose.hpp
/// @brief Value-based plot layers for C++ pipe expressions.
///
/// Builds plots the way ggplot2 adds layers with `+`: `dm::Plot() | dm::points(x, y) |
/// dm::legend()...`. Each layer (Points, Legend, PlotLabel) is an owned, reusable value, and
/// each `|` returns a new plot, so the same layer can be piped into several plots.
///
/// @par Example
/// @code
/// const auto iris = dm::datasets::iris();
/// const auto species = dm::factor(dm::c(iris.get_column<std::string>("Species").values()));
/// const auto palette = dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255},
///                            dm::RGB{220, 100, 255});
/// dm::Plot plt = dm::Plot().set_title("Iris Dataset") |
///                dm::points(iris.get_column("Sepal.Width").values(),
///                           iris.get_column("Sepal.Length").values())
///                    .set_col(palette.select(species)) |
///                dm::legend()
///                    .set_labels(dm::c("setosa", "versicolor", "virginica"))
///                    .set_col(palette);
/// plt.save("iris.svg");
/// @endcode

#include <datamunge/vector.hpp>

#include <initializer_list>
#include <map>
#include <variant>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge {

/// @brief Owned plot-level text options for `Plot().with(PlotArgs{...})`, the named-argument
///        spelling of R's `plot(..., main =, sub =, xlab =, ylab =)`.
///
/// Options are brace-initialized `{"name", "value"}` pairs. Supported names are title,
/// subtitle, xlab, and ylab. Omitted fields keep their current values; an empty string clears
/// a field. Unknown or duplicate names throw std::invalid_argument when the PlotArgs is
/// constructed, before anything is applied, so a typo cannot half-update a plot.
///
/// @par Example
/// @code
/// auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 9.0))
///                .with(dm::PlotArgs{{"title", "Squares"}, {"xlab", "n"}, {"ylab", "n^2"}});
/// std::print("{} | {}\n", plt.title_text(), plt.y_label_text());  // Squares | n^2
/// dm::PlotArgs{{"main", "Squares"}};  // throws: PlotArgs: unknown option main
/// @endcode
/// @see dm::Plot::with(), dm::title(), dm::Plot::set_title()
class PlotArgs {
 public:
  /// @brief Constructs empty options, leaving all plot text unchanged.
  PlotArgs() = default;
  /// @brief Copies named text values and validates their names and uniqueness.
  /// @param options  `{name, text}` pairs; name is title, subtitle, xlab, or ylab, each at most once.
  /// @throws std::invalid_argument  For an unknown or repeated option name.
  PlotArgs(std::initializer_list<std::pair<std::string, std::string>> options) {
    for (const auto& option : options) {
      if (option.first != "title" && option.first != "subtitle" &&
          option.first != "xlab" && option.first != "ylab")
        throw std::invalid_argument("PlotArgs: unknown option " + option.first);
      if (!text_.emplace(option).second)
        throw std::invalid_argument("PlotArgs: duplicate option " + option.first);
    }
  }
  /// @brief Updates the selected text fields of a plot; no option values are borrowed afterward.
  /// Called by dm::Plot::with(); call it directly only when writing a custom option group.
  /// @param plot  The plot whose title/subtitle/axis labels are overwritten.
  void apply(Plot& plot) const {
    for (const auto& option : text_) {
      if (option.first == "title") plot.title(option.second);
      else if (option.first == "subtitle") plot.subtitle(option.second);
      else if (option.first == "xlab") plot.xlab(option.second);
      else plot.ylab(option.second);
    }
  }

 private:
  /// @brief Validated, unique names and owned text values.
  std::map<std::string, std::string> text_;
};

/// @brief Named options for a points() layer, used as `points(x, y).with(PointsArgs{...})`;
///        currently "col" supplies recycled per-point RGB(A) colors (R's `col =`).
///
/// Options are brace-initialized `{"name", value}` pairs and own their values. Point i takes
/// `col[i % col.size()]`, so a one-color palette paints every point and a per-row palette
/// (e.g. `palette.select(factor)`) colors by group. Unknown or duplicate names and empty
/// palettes throw std::invalid_argument, so a misspelled option cannot silently change a plot.
///
/// @par Example
/// @code
/// const auto palette = dm::c(dm::RGB{255, 0, 0}, dm::RGB{0, 0, 255});
/// auto plt = dm::Plot() |
///            dm::points(dm::c(1.0, 2.0, 3.0, 4.0), dm::c(4.0, 3.0, 2.0, 1.0))
///                .with(dm::PointsArgs{{"col", palette}});  // red, blue, red, blue
/// plt.save("alternating.svg");
/// @endcode
/// @see Points::with(), Points::set_col()
class PointsArgs {
 public:
  /// @brief Constructs options using the default point style.
  PointsArgs() = default;
  /// @brief Copies named palettes, validating option names and palette lengths.
  /// @param options  At most one `{"col", palette}` pair with a nonempty palette.
  /// @throws std::invalid_argument  For an unknown name, a repeated "col", or an empty palette.
  PointsArgs(std::initializer_list<std::pair<std::string, Vector<RGB>>> options) {
    for (const auto& option : options) {
      if (option.first != "col") throw std::invalid_argument("PointsArgs: unknown option " + option.first);
      if (!colors_.empty()) throw std::invalid_argument("PointsArgs: duplicate col option");
      if (option.second.empty()) throw std::invalid_argument("PointsArgs: col must not be empty");
      colors_ = option.second.values();
    }
  }
  /// @brief Borrows the palette; empty means the default point color.
  /// @return The "col" palette, valid while these options live.
  const std::vector<RGB>& colors() const noexcept { return colors_; }

 private:
  /// @brief Owned palette, recycled by a point layer when it is applied.
  std::vector<RGB> colors_;
};

/// @brief An owned, reusable scatter layer created by points(), the pipe-friendly form of
///        R's `points(x, y, col = ...)` / ggplot2's `geom_point()`.
///
/// The layer copies its coordinates and styling, so it can outlive the data it came from and
/// be piped into any number of plots. Nothing is drawn until it is combined with a plot via
/// `plot | layer` (or apply()). Without set_col() the points use the default blue
/// `{37, 99, 235}` with 4-pixel markers. Point layers are unlabeled, so they never add rows
/// to the automatic legend; use dm::legend() for a key.
///
/// @par Example
/// @code
/// const auto iris = dm::datasets::iris();
/// const auto layer = dm::points(iris.get_column("Petal.Length").values(),
///                               iris.get_column("Petal.Width").values());
/// auto plt = dm::Plot().set_title("Petals") | layer;
/// std::print("{}\n", plt.series().size());  // 1
/// @endcode
/// @see dm::points(), operator|(Plot, const Points&), PointsArgs
class Points {
 public:
  /// @brief Takes ownership of equally sized, nonempty coordinates.
  /// @param x  Horizontal coordinates, one per point.
  /// @param y  Vertical coordinates, paired with @p x by position.
  /// @throws std::invalid_argument If coordinate lengths differ or are empty.
  Points(std::vector<double> x, std::vector<double> y) : x_(std::move(x)), y_(std::move(y)) {
    if (x_.empty() || x_.size() != y_.size())
      throw std::invalid_argument("points: coordinates must have equal nonzero lengths");
  }
  /// @brief Returns a copy with the supplied options; leaves this reusable layer unchanged.
  ///
  /// Replaces all options of the copy (an empty PointsArgs restores the default color).
  ///
  /// @param options  Named options such as `{{"col", palette}}`.
  /// @return A new layer with the same coordinates and @p options.
  ///
  /// @par Example
  /// @code
  /// const auto base = dm::points(dm::c(1.0, 2.0, 3.0), dm::c(3.0, 1.0, 2.0));
  /// const auto red = base.with(dm::PointsArgs{{"col", dm::c(dm::RGB{255, 0, 0})}});
  /// auto plt = dm::Plot() | base | red;  // base still uses the default color
  /// @endcode
  /// @see set_col(), PointsArgs
  Points with(PointsArgs options) const {
    Points result = *this;
    result.options_ = std::move(options);
    return result;
  }
  /// @brief Sets per-point colors, like R's `points(x, y, col = palette[f])`.
  ///
  /// Replaces this layer's palette with an owned copy. Point i takes
  /// `colors[i % colors.size()]`, so pass one color for a uniform layer or one color per row
  /// (typically `palette.select(factor)`) to color by group. Partial cycles are allowed
  /// silently, and each color keeps its own alpha. Mutates this layer, unlike with().
  ///
  /// @param colors  Nonempty palette, recycled to the number of points.
  /// @return This layer, for chaining.
  /// @throws std::invalid_argument If colors is empty; the layer is left unchanged.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// const auto species = dm::factor(dm::c(iris.get_column<std::string>("Species").values()));
  /// const auto palette = dm::c(dm::RGB{230, 159, 0}, dm::RGB{86, 180, 233},
  ///                            dm::RGB{0, 158, 115});
  /// auto plt = dm::Plot() |
  ///            dm::points(iris.get_column("Sepal.Width").values(),
  ///                       iris.get_column("Sepal.Length").values())
  ///                .set_col(palette.select(species));  // one color per row
  /// std::print("{}\n", plt.series()[0].point_colors.size());  // 150
  /// @endcode
  /// @see with(), dm::Legend::set_col(), dm::Plot::set_col()
  Points& set_col(const Vector<RGB>& colors) {
    options_ = PointsArgs{{"col", colors}};
    return *this;
  }
  /// @brief Appends this layer to a plot, recycling its colors to the number of points.
  ///
  /// Mutates @p plot in place; `plot | layer` is the copying equivalent.
  ///
  /// @param plot  Plot that receives a new scatter series.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot();
  /// dm::points(dm::c(1.0, 2.0), dm::c(2.0, 1.0)).apply(plt);
  /// std::print("{}\n", plt.series().size());  // 1
  /// @endcode
  void apply(Plot& plot) const {
    if (options_.colors().empty()) plot.points(x_, y_);
    else plot.points_colored(x_, y_, options_.colors());
  }

 private:
  /// @brief Owned horizontal coordinates in input order.
  std::vector<double> x_;
  /// @brief Owned vertical coordinates paired with x_.
  std::vector<double> y_;
  /// @brief Owned styling to apply to this layer alone.
  PointsArgs options_;
};

/// @brief Creates a scatter layer for `plot | layer`; style it with set_col() or
///        with(PointsArgs{...}) (R's `points()`, ggplot2's `geom_point()`).
///
/// The coordinates are copied (or moved from rvalues); `dm::Vector<double>` and the
/// `std::vector<double>` returned by `get_column(...).values()` both work.
///
/// @param x  Horizontal coordinates, one per point.
/// @param y  Vertical coordinates, paired with @p x by position.
/// @return An owned Points layer with the default blue color.
/// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
///
/// @par Example
/// @code
/// const auto iris = dm::datasets::iris();
/// auto plt = dm::Plot().set_title("Iris") |
///            dm::points(iris.get_column("Sepal.Width").values(),
///                       iris.get_column("Sepal.Length").values())
///                .set_col(dm::c(dm::RGB{220, 100, 255}));
/// plt.save("iris.svg");
/// @endcode
/// @see Points, dm::legend(), dm::Plot::points()
inline Points points(std::vector<double> x, std::vector<double> y) {
  return Points(std::move(x), std::move(y));
}

/// @brief Named options for a manually labeled legend, used as
///        `legend().with(LegendArgs{{"labels", ...}, {"col", ...}})` (R's `legend()` arguments).
///
/// Nonempty constructor options must supply both a nonempty string vector under "labels"
/// and a nonempty RGB(A) vector under "col", in either order. Colors recycle to the label
/// count, including partial cycles. Empty options retain the plot's automatic legend.
/// Unknown, duplicate, missing, empty, or wrongly typed values throw std::invalid_argument
/// at construction.
///
/// @par Example
/// @code
/// const auto palette = dm::c(dm::RGB{37, 99, 235}, dm::RGB{220, 38, 38});
/// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)) |
///            dm::legend().with(dm::LegendArgs{{"labels", dm::c("treated", "control")},
///                                             {"col", palette}});
/// std::print("{}\n", plt.legend_entries()[0].label);  // treated
/// @endcode
/// @see Legend::with(), Legend::set_labels(), Legend::set_col()
class LegendArgs {
 public:
  /// @brief Either owned text labels or an owned RGB(A) palette, depending on the option name.
  using Value = std::variant<Vector<std::string>, Vector<RGB>>;
  /// @brief Constructs options that retain the existing automatic legend.
  LegendArgs() = default;
  /// @brief Copies and validates named labels and colors, independent of their input order.
  /// @param options  `{"labels", Vector<std::string>}` and `{"col", Vector<RGB>}`, both or neither.
  /// @throws std::invalid_argument  For an unknown name, a repeated or empty option, a value of
  ///         the wrong type, or only one of the two options.
  LegendArgs(std::initializer_list<std::pair<std::string, Value>> options) {
    for (const auto& option : options) {
      if (option.first == "labels") {
        const auto* labels = std::get_if<Vector<std::string>>(&option.second);
        if (!labels || labels->empty() || !labels_.empty())
          throw std::invalid_argument("LegendArgs: labels must be a unique nonempty string vector");
        labels_ = *labels;
      } else if (option.first == "col") {
        const auto* colors = std::get_if<Vector<RGB>>(&option.second);
        if (!colors || colors->empty() || !colors_.empty())
          throw std::invalid_argument("LegendArgs: col must be a unique nonempty color vector");
        colors_ = *colors;
      } else {
        throw std::invalid_argument("LegendArgs: unknown option " + option.first);
      }
    }
    if (labels_.empty() != colors_.empty())
      throw std::invalid_argument("LegendArgs: supply both labels and col");
  }
  /// @brief Appends manual entries with recycled colors, or leaves an automatic legend unchanged.
  /// @param plot  Plot that receives one legend row per label.
  /// @throws std::invalid_argument If a fluent legend supplied only labels or only colors.
  void apply(Plot& plot) const {
    if (labels_.empty() != colors_.empty())
      throw std::invalid_argument("LegendArgs: supply both labels and col");
    if (!labels_.empty()) plot.legend(labels_.values(), colors_.recycled(labels_.size()).values());
  }

 private:
  /// @brief Allows the fluent layer to build labels and colors independently before apply().
  friend class Legend;

  /// @brief Owned legend labels in display order; empty selects automatic mode.
  Vector<std::string> labels_;
  /// @brief Owned nonempty palette when labels are supplied.
  Vector<RGB> colors_;
};

/// @brief An owned, reusable legend layer created by legend(), the pipe-friendly form of
///        R's `legend(legend = labels, col = colors)`.
///
/// A bare `dm::legend()` keeps the plot's automatic legend (one row per labeled series).
/// Once labels and colors are set, piping the layer adds one manual row per label, which
/// replaces the automatic legend. Labels and colors may be set in either order, but both must
/// be present by the time the layer is applied. The layer owns copies of both.
///
/// @par Example
/// @code
/// const auto palette = dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255},
///                            dm::RGB{220, 100, 255});
/// auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 2.0, 3.0)) |
///            dm::legend()
///                .set_labels(dm::c("setosa", "versicolor", "virginica"))
///                .set_col(palette);
/// std::print("{}\n", plt.legend_entries().size());  // 3
/// @endcode
/// @see dm::legend(), LegendArgs, dm::Plot::legend()
class Legend {
 public:
  /// @brief Constructs a layer that retains the plot's automatic legend.
  Legend() = default;
  /// @brief Returns a copy with new options, leaving this layer unchanged.
  ///
  /// Replaces both labels and colors of the copy; validation already happened when the
  /// LegendArgs was constructed.
  ///
  /// @param options  Named `{"labels", ...}` / `{"col", ...}` options, or empty for automatic.
  /// @return A new legend layer holding @p options.
  ///
  /// @par Example
  /// @code
  /// const auto key = dm::legend().with(dm::LegendArgs{
  ///     {"labels", dm::c("a", "b")}, {"col", dm::c(dm::RGB{0, 0, 0}, dm::RGB{255, 0, 0})}});
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)) | key;
  /// @endcode
  /// @see LegendArgs
  Legend with(LegendArgs options) const {
    Legend result = *this;
    result.options_ = std::move(options);
    return result;
  }
  /// @brief Sets the legend row texts (R's `legend =`); preserves the palette.
  ///
  /// Replaces the owned labels, in display order. Both labels and colors must be supplied
  /// before applying the layer, in either order.
  ///
  /// @param labels  Nonempty row texts, e.g. `dm::c("setosa", "versicolor", "virginica")`.
  /// @return This layer, for chaining.
  /// @throws std::invalid_argument If labels is empty; the layer is left unchanged.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)) |
  ///            dm::legend().set_labels(dm::c("data")).set_col(dm::c(dm::RGB{37, 99, 235}));
  /// std::print("{}\n", plt.legend_entries()[0].label);  // data
  /// @endcode
  /// @see set_col()
  Legend& set_labels(const Vector<std::string>& labels) {
    if (labels.empty()) throw std::invalid_argument("legend: labels must not be empty");
    options_.labels_ = labels;
    return *this;
  }
  /// @brief Sets the swatch colors (R's `col =`); preserves the labels.
  ///
  /// Replaces the owned palette. When applied, row i takes `colors[i % colors.size()]`,
  /// retaining alpha; pass the same palette used for the points so the key matches them.
  ///
  /// @param colors  Nonempty palette, recycled to the label count.
  /// @return This layer, for chaining.
  /// @throws std::invalid_argument If colors is empty; the layer is left unchanged.
  ///
  /// @par Example
  /// @code
  /// const auto two = dm::c(dm::RGB{0, 0, 0}, dm::RGB{255, 0, 0});
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)) |
  ///            dm::legend().set_labels(dm::c("a", "b", "c")).set_col(two);
  /// for (const auto& row : plt.legend_entries()) std::print("{}:{} ", row.label, row.color.r);
  /// // a:0 b:255 c:0
  /// @endcode
  /// @see set_labels(), dm::Points::set_col()
  Legend& set_col(const Vector<RGB>& colors) {
    if (colors.empty()) throw std::invalid_argument("legend: col must not be empty");
    options_.colors_ = colors;
    return *this;
  }
  /// @brief Applies this layer's legend entries to a plot.
  ///
  /// Mutates @p plot in place; `plot | layer` is the copying equivalent.
  ///
  /// @param plot  Plot that receives the manual legend rows.
  /// @throws std::invalid_argument If only labels or only colors have been supplied.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0));
  /// dm::legend().set_labels(dm::c("data")).set_col(dm::c(dm::RGB{37, 99, 235})).apply(plt);
  /// @endcode
  void apply(Plot& plot) const { options_.apply(plot); }

 private:
  /// @brief Owned legend labels and palette, or automatic-legend options.
  LegendArgs options_;
};

/// @brief Creates a legend layer for `plot | layer`; fill it with set_labels()/set_col() or
///        with(LegendArgs{...}) (R's `legend()`).
///
/// An unconfigured legend() is valid and leaves the automatic legend unchanged; a layer with
/// only labels or only colors throws when piped.
///
/// @return An empty Legend layer.
///
/// @par Example
/// @code
/// const auto palette = dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255},
///                            dm::RGB{220, 100, 255});
/// auto plt = dm::Plot().set_title("Iris") |
///            dm::legend()
///                .set_labels(dm::c("setosa", "versicolor", "virginica"))
///                .set_col(palette);
/// dm::Plot() | dm::legend().set_labels(dm::c("x"));  // throws: supply both labels and col
/// @endcode
/// @see Legend, dm::points()
inline Legend legend() { return {}; }

/// @brief An owned heading or axis-label update for a plot pipe.
struct PlotLabel {
  /// @brief The plot text field updated when this label is applied.
  enum class Target {
    Title, ///< Main heading above the chart.
    Subtitle, ///< Secondary heading beneath the title.
    XLabel, ///< Horizontal-axis label.
    YLabel, ///< Vertical-axis label.
  };

  /// @brief Text to display, escaped by the renderer.
  std::string text;
  /// @brief Destination of the text; defaults to the main title.
  Target target{Target::Title};
};

/// @brief Creates a title update for a plot pipe, like ggplot2's `+ ggtitle(text)`.
/// @param text  Heading text; empty clears the title.
/// @return A PlotLabel targeting the title.
///
/// @par Example
/// @code
/// const auto x = dm::seql(0.0, 1.0, 11);
/// auto plt = dm::Plot() | dm::title("Identity") | dm::subtitle("y = x") |
///            dm::xlab("input") | dm::ylab("output") | dm::points(x, x);
/// std::print("{} | {}\n", plt.title_text(), plt.x_label_text());  // Identity | input
/// @endcode
/// @see dm::Plot::set_title(), subtitle(), xlab(), ylab()
inline PlotLabel title(std::string text) { return {std::move(text), PlotLabel::Target::Title}; }
/// @brief Creates a subtitle update for a plot pipe, like ggplot2's `+ labs(subtitle = text)`.
/// @param text  Subtitle text; empty removes the subtitle.
/// @return A PlotLabel targeting the subtitle.
/// @par Example
/// @code
/// auto plt = dm::Plot() | dm::title("Iris") | dm::subtitle("Sepal Width vs Length");
/// @endcode
/// @see title(), dm::Plot::set_subtitle()
inline PlotLabel subtitle(std::string text) { return {std::move(text), PlotLabel::Target::Subtitle}; }
/// @brief Creates a horizontal-axis label update for a plot pipe, like ggplot2's `+ xlab(text)`.
/// @param text  Axis label text.
/// @return A PlotLabel targeting the x-axis label.
/// @par Example
/// @code
/// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(3.0, 4.0)) | dm::xlab("Sepal Width");
/// @endcode
/// @see ylab(), dm::Plot::set_xlab()
inline PlotLabel xlab(std::string text) { return {std::move(text), PlotLabel::Target::XLabel}; }
/// @brief Creates a vertical-axis label update for a plot pipe, like ggplot2's `+ ylab(text)`.
/// @param text  Axis label text.
/// @return A PlotLabel targeting the y-axis label.
/// @par Example
/// @code
/// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(3.0, 4.0)) | dm::ylab("Sepal Length");
/// @endcode
/// @see xlab(), dm::Plot::set_ylab()
inline PlotLabel ylab(std::string text) { return {std::move(text), PlotLabel::Target::YLabel}; }

/// @brief Pipes a point layer into a plot: `plot | dm::points(x, y)` (ggplot2's `+ geom_point()`).
///
/// Takes the plot by value, appends the layer's scatter series to that copy, and returns it.
/// An lvalue plot on the left is therefore unchanged, and the layer can be reused. Because
/// `|` binds more loosely than `.`, `dm::Plot().set_title("t") | dm::points(...).set_col(...)`
/// finishes each side's setter chain before composing. The result is a dm::Plot even when the
/// left side is a dm::Histogram.
///
/// @param plot  Plot to extend (copied, or moved from a temporary).
/// @param layer  Scatter layer to append.
/// @return The extended plot.
///
/// @par Example
/// @code
/// const auto base = dm::Plot().set_title("Base");
/// const auto with_pts = base | dm::points(dm::c(1.0, 2.0), dm::c(1.0, 2.0));
/// std::print("{} {}\n", base.series().size(), with_pts.series().size());  // 0 1
/// @endcode
/// @see dm::points(), Points::apply()
inline Plot operator|(Plot plot, const Points& layer) {
  layer.apply(plot);
  return plot;
}
/// @brief Pipes a legend layer into a plot: `plot | dm::legend().set_labels(...).set_col(...)`.
///
/// Same copy semantics as the Points overload: the plot is taken by value, the legend rows are
/// added to the copy, and the copy is returned. Pipe the legend after the point layers it
/// describes, as in R where `legend()` follows `plot()`.
///
/// @param plot  Plot to extend (copied, or moved from a temporary).
/// @param layer  Legend layer to append; an empty dm::legend() changes nothing.
/// @return The plot with the manual legend rows.
/// @throws std::invalid_argument  If the layer has only labels or only colors.
///
/// @par Example
/// @code
/// auto plt = dm::Plot() | dm::points(dm::c(1.0, 2.0), dm::c(1.0, 2.0)) |
///            dm::legend().set_labels(dm::c("obs")).set_col(dm::c(dm::RGB{37, 99, 235}));
/// @endcode
/// @see dm::legend(), Legend::apply()
inline Plot operator|(Plot plot, const Legend& layer) {
  layer.apply(plot);
  return plot;
}
/// @brief Pipes a heading or axis label into a plot: `plot | dm::title("...")`.
///
/// Updates one text field on a copy of the plot and returns it; an empty label clears the
/// field. Equivalent to the matching set_title()/set_subtitle()/set_xlab()/set_ylab() call.
///
/// @param plot  Plot to update (copied, or moved from a temporary).
/// @param label  Text and target produced by title(), subtitle(), xlab(), or ylab().
/// @return The updated plot.
///
/// @par Example
/// @code
/// auto plt = dm::Plot() | dm::points(dm::c(1.0, 2.0), dm::c(1.0, 2.0)) |
///            dm::title("Two points") | dm::xlab("a") | dm::ylab("b");
/// @endcode
/// @see title(), subtitle(), xlab(), ylab()
inline Plot operator|(Plot plot, const PlotLabel& label) {
  switch (label.target) {
    case PlotLabel::Target::Title: plot.title(label.text); break;
    case PlotLabel::Target::Subtitle: plot.subtitle(label.text); break;
    case PlotLabel::Target::XLabel: plot.xlab(label.text); break;
    case PlotLabel::Target::YLabel: plot.ylab(label.text); break;
  }
  return plot;
}

} // namespace datamunge
