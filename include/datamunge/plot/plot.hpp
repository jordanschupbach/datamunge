#pragma once

/// @file plot.hpp
/// @brief R-base-graphics-style plotting: the RPlot value (spelled `dm::Plot`), its colors and
///        layer records, and the RLayout multi-panel figure.
///
/// A plot is built in memory from layers (DataSeries, ABLine, LegendEntry) and rendered only by
/// to_svg(), save() (SVG, PNG or PDF), or show() (an interactive X11 window). Nothing here needs
/// a display except show()/view().
///
/// @par Example
/// @code
/// const auto x = dm::seql(0.0, 6.0, 25);
/// auto plt = dm::Plot(x, dm::sin(x)).set_title("Sine");
/// plt.lines(x, dm::sin(x)).abline_h(0.0);
/// auto fig = dm::graphics::RLayout::create(1, 2).add(plt).add(dm::Plot::hist(dm::rnorm(0, 1, 200)));
/// fig.save("two_panels.svg");
/// @endcode
///
/// @par References
/// - R Core Team. The R graphics package: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/00Index.html
///
/// @see RPlot, RLayout, compose.hpp

#include <cstddef>
#include <cstdint>
#include <string>
#include <limits>
#include <vector>

namespace datamunge {
class Callback;
} // namespace datamunge

/// @brief Plotting: RPlot (dm::Plot) and RLayout (R base graphics), GGPlot (ggplot2-style),
///        Splom (pairs plots), and Figure, the common output type the free dm::plot()
///        functions return. Named `graphics` (after R's package) so that `dm::plot` can be a
///        function.
namespace datamunge::graphics {

/// @brief A plotting color: 8-bit red, green, and blue channels plus an alpha opacity,
///        the C++ counterpart of R's `rgb(r, g, b, alpha, maxColorValue = 255)`.
///
/// An aggregate, so brace-initialize it: `dm::RGB{255, 0, 0}` is opaque red and
/// `dm::RGB{255, 0, 0, 0.5}` is half-transparent red. A default-constructed RGB is the
/// library's dark-slate ink `{31, 41, 55}`, fully opaque. Channels are not validated: keep
/// them in [0, 255] (both the raster and SVG renderers clamp out-of-range channels), and
/// keep alpha in [0, 1] (both renderers clamp it). Every plotting API takes RGB by value,
/// and `dm::c(...)` of several RGBs builds a palette usable with `select()` and set_col().
///
/// @par Example
/// @code
/// const auto gold = dm::RGB{255, 215, 0};
/// const auto glass = dm::RGBA{37, 99, 235, 0.3};  // translucent blue
/// const dm::RGB ink;                              // {31, 41, 55}, opaque
/// std::print("{} {} {}\n", gold.r, glass.alpha, ink.r);  // 255 0.3 31
/// const auto palette = dm::c(gold, glass);       // a dm::Vector<dm::RGB>
/// @endcode
/// @see dm::RGBA, dm::Plot::set_col(), dm::points()
struct RGB {
  int r{31};         ///< Red channel, 0-255 (default 31).
  int g{41};         ///< Green channel, 0-255 (default 41).
  int b{55};         ///< Blue channel, 0-255 (default 55).
  double alpha{1.0}; ///< Opacity in [0, 1]; 1 (the default) is opaque, 0 is invisible.
};

/// @brief Alias of RGB for colors with a meaningful alpha channel.
///
/// `RGBA{r, g, b, alpha}` reads better than `RGB{r, g, b, alpha}` for translucent colors; the
/// two names denote the same type and mix freely in palettes.
///
/// @par Example
/// @code
/// auto plt = dm::Plot(dm::rnorm(0, 1, 500), dm::rnorm(0, 1, 500))
///                .set_col(dm::RGBA{37, 99, 235, 0.3});  // overplotting shows as density
/// plt.save("cloud.svg");
/// @endcode
/// @see dm::RGB
using RGBA = RGB;

/// @brief One drawn layer of a plot (points, a line, bars, a box, a polygon, a label, or
///        segments), as stored by RPlot's layer verbs and read back through Plot::series().
///
/// A plain aggregate holding data coordinates and styling. Which fields matter depends on
/// kind; each Kind value documents how it reads x, y, and the other fields. Layers are drawn
/// in insertion order, after the reference lines, and every layer's coordinates widen the
/// automatic axis ranges (bars also pull the y range down or up to 0). A nonempty label puts
/// the layer in the automatic legend unless it is a Text layer. User code normally creates
/// layers through RPlot verbs (points(), lines(), bars(), ...) and only reads this struct.
///
/// @par Example
/// @code
/// auto plt = dm::Plot().points(dm::c(1.0, 2.0), dm::c(3.0, 4.0), "pts");
/// const dm::graphics::DataSeries& s = plt.series()[0];
/// std::print("{} {} {} {}\n", s.kind == dm::graphics::DataSeries::Kind::Scatter, s.x.size(),
///            s.label, s.marker_size);  // true 2 pts 4
/// @endcode
/// @see Plot::series(), RPlot::points(), RPlot::lines()
struct DataSeries {
  /// @brief How a DataSeries is drawn and how its coordinate vectors are read.
  enum class Kind {
    Scatter, ///< One marker (radius marker_size px) per (x[i], y[i]); colors from point_colors when set.

    Line,    ///< A polyline joining (x[i], y[i]) in order, stroke_width px wide (R's `lines()`).

    Bar,     ///< Vertical bars from y = 0 to y[i], centered on x[i], bar_width data units wide.

    /// @brief One box-and-whisker at x[0]: y = {whisker_lo, q1, median, q3, whisker_hi,
    ///        outliers...}; bar_width is the box width in data units.
    Box,
    Polygon, ///< A closed polygon through (x[i], y[i]); filled selects fill vs. outline only.

    /// @brief A centered text label at (x[0], y[0]); label holds the text and marker_size the
    ///        font size in pixels. Never listed in the automatic legend.
    Text,
    /// @brief Disconnected segments joining (x[2i], y[2i]) to (x[2i+1], y[2i+1]), stroke_width px
    ///        wide (R's `segments()`).
    Segment,
  };

  Kind                kind{Kind::Line};  ///< How the layer is drawn (default Line).
  std::vector<double> x;                 ///< Horizontal data coordinates; meaning depends on kind.
  std::vector<double> y;                 ///< Vertical data coordinates; meaning depends on kind.
  std::string         label;             ///< Legend text (or the drawn text for Kind::Text); empty omits it.
  RGB                 color{};           ///< Layer color, including alpha (default RGB{}, dark slate).
  double              stroke_width{2.0}; ///< Line, Segment, and outline-Polygon width in pixels (default 2).
  double              marker_size{4.0};  ///< Scatter marker radius, or Text font size, in pixels (default 4).
  double              bar_width{0.8};    ///< Bar or Box width in data units (default 0.8).
  bool                filled{false};     ///< Polygon only: true fills it, false draws its outline (default false).

  /// @brief Optional scatter palette, recycled in input order; empty uses color for every point.
  std::vector<RGB>    point_colors;
};

/// @brief A straight reference line across the whole plotting area, as drawn by R's
///        `abline(a, b)`, `abline(h =)` and `abline(v =)`.
///
/// A sloped or horizontal line is y = value + slope * x evaluated at both ends of the visible
/// x range; a vertical line sits at x = value and ignores slope. Reference lines are drawn
/// beneath every DataSeries. Vertical lines widen the automatic x range to include them;
/// sloped and horizontal lines never change the axis ranges. Created by RPlot::abline(),
/// abline_h(), abline_v(), and qqline(); read back through Plot::reference_lines().
///
/// @par Example
/// @code
/// auto plt = dm::Plot(dm::c(0.0, 1.0), dm::c(0.0, 2.0)).abline(0.5, 2.0).abline_v(0.25);
/// const auto& lines = plt.reference_lines();
/// std::print("{} {} {}\n", lines[0].slope, lines[1].vertical, lines[1].value);  // 2 true 0.25
/// @endcode
///
/// @par References
/// - R `abline()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/abline.html
///
/// @see RPlot::abline(), Plot::reference_lines()
struct ABLine {
  bool   vertical{false}; ///< True for a vertical line at x = value (default false).

  double value{0.0}; ///< y-intercept for a sloped/horizontal line; x-position for a vertical line

  double slope{0.0};         ///< Change in y per unit x; ignored when vertical (default 0).
  RGB    color{0, 0, 0};     ///< Line color (default opaque black).
  double stroke_width{1.0};  ///< Line width in pixels (default 1).
};

/// @brief One manually specified legend row, as drawn by R's `legend()`.
///
/// When a plot has any manual entries, they replace the default auto-generated
/// (one-row-per-labeled-series) legend. Each row is a color swatch followed by the label,
/// stacked from the top-right corner of the plotting area in insertion order.
///
/// @par Example
/// @code
/// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).legend({"data"}, {dm::RGB{0, 0, 255}});
/// std::print("{} {}\n", plt.legend_entries()[0].label, plt.legend_entries()[0].color.b);  // data 255
/// @endcode
///
/// @par References
/// - R `legend()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/legend.html
///
/// @see RPlot::legend(), dm::legend(), Plot::legend_entries()
struct LegendEntry {
  std::string label;          ///< Row text.
  RGB         color{0, 0, 0}; ///< Swatch color, including alpha (default opaque black).
};

/// @brief Renderer-facing state shared by every 2-D plot: canvas, labels, axes, and series.
///
/// User code works with the derived RPlot (spelled `dm::Plot`), whose setters return `RPlot&`;
/// this base holds the data, the read-only accessors, and the output functions to_svg(),
/// save(), and show(). The base-class setters below return `Plot&` and are what RPlot's
/// same-named setters forward to; call the RPlot forms in user code so chains keep the RPlot
/// type. A `const Plot&` is also what RLayout::add() accepts and RLayout::panel() returns.
///
/// Every output function renders from the stored state, so it throws std::runtime_error for a
/// plot with no data series (reference lines and legend rows alone are not enough) or a canvas
/// too small for its margins. Plots are copyable values with no shared state; distinct plots may
/// be used from different threads, but one plot must not be mutated while it is rendered.
///
/// @par Example
/// @code
/// const dm::graphics::Plot& base = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 4.0)).set_title("Base");
/// std::print("{} {}x{}\n", base.title_text(), base.width(), base.height());  // Base 960x640
/// @endcode
/// @see RPlot, RLayout
class Plot {
 public:
  /// @brief Virtual destructor, so a derived RPlot can be destroyed through a Plot pointer.
  virtual ~Plot() = default;

  /// @brief Base-class form of RPlot::size(): sets the canvas size in pixels (default 960x640).
  /// @param width  Canvas width in pixels.
  /// @param height  Canvas height in pixels.
  /// @return This plot, for chaining.
  /// @see RPlot::size()
  Plot& size(std::size_t width, std::size_t height);
  /// @brief Base-class form of RPlot::title(): sets the main heading; empty removes it.
  /// @param value  Heading text (moved into the plot).
  /// @return This plot, for chaining.
  /// @see RPlot::set_title()
  Plot& title(std::string value);
  /// @brief Sets an optional subtitle displayed beneath the title; owns the supplied text.
  /// @param value  Subtitle text; empty removes it.
  /// @return This plot, for chaining.
  /// @see RPlot::set_subtitle()
  Plot& subtitle(std::string value);
  /// @brief Base-class form of RPlot::x_label(): sets the horizontal-axis label.
  /// @param value  Label text (moved into the plot).
  /// @return This plot, for chaining.
  /// @see RPlot::set_xlab()
  Plot& x_label(std::string value);
  /// @brief Base-class form of RPlot::y_label(): sets the vertical-axis label.
  /// @param value  Label text (moved into the plot).
  /// @return This plot, for chaining.
  /// @see RPlot::set_ylab()
  Plot& y_label(std::string value);
  /// @brief Base-class form of RPlot::background(): sets the canvas fill color.
  /// @param color  Fill for the whole canvas (default white).
  /// @return This plot, for chaining.
  /// @see RPlot::background()
  Plot& background(RGB color);
  /// @brief Base-class form of RPlot::axis_color(): sets the axis-line and text color.
  /// @param color  Color for axis lines, ticks, axis labels, title, and subtitle.
  /// @return This plot, for chaining.
  /// @see RPlot::axis_color()
  Plot& axis_color(RGB color);
  /// @brief Base-class form of RPlot::grid_color(): sets the grid-line color.
  /// @param color  Grid-line color (default light gray `{229, 231, 235}`).
  /// @return This plot, for chaining.
  /// @see RPlot::grid_color()
  Plot& grid_color(RGB color);
  /// @brief Base-class form of RPlot::show_grid(): turns the background grid on or off.
  /// @param enabled  True (the default) draws the grid.
  /// @return This plot, for chaining.
  /// @see RPlot::show_grid()
  Plot& show_grid(bool enabled = true);
  /// @brief Base-class form of RPlot::x_limits(): fixes the visible x range.
  /// @param min_x  Left edge of the plotting area, in data units.
  /// @param max_x  Right edge; must be greater than @p min_x.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  Unless `max_x > min_x` (NaN fails); the plot is unchanged.
  /// @see RPlot::x_limits()
  Plot& x_limits(double min_x, double max_x);
  /// @brief Base-class form of RPlot::y_limits(): fixes the visible y range.
  /// @param min_y  Bottom edge of the plotting area, in data units.
  /// @param max_y  Top edge; must be greater than @p min_y.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  Unless `max_y > min_y` (NaN fails); the plot is unchanged.
  /// @see RPlot::y_limits()
  Plot& y_limits(double min_y, double max_y);
  /// @brief Base-class form of RPlot::hide_axes(): hides or restores axes, ticks, grid, and axis labels.
  /// @param enabled  True (the default) hides them.
  /// @return This plot, for chaining.
  /// @see RPlot::hide_axes()
  Plot& hide_axes(bool enabled = true);
  /// @brief Base-class form of RPlot::x_tick_labels(): labels the x axis at 0, 1, 2, ...
  /// @param labels  Tick texts; empty restores the numeric ticks.
  /// @return This plot, for chaining.
  /// @see RPlot::x_tick_labels()
  Plot& x_tick_labels(std::vector<std::string> labels);
  /// @brief Replaces the numeric x ticks with @p labels drawn at data positions @p positions.
  /// @param positions  x data coordinates of the ticks.
  /// @param labels  Tick texts, one per position.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument If the two vectors differ in length.
  /// @see RPlot::x_ticks()
  Plot& x_ticks(std::vector<double> positions, std::vector<std::string> labels);
  /// @brief Replaces the numeric y ticks with @p labels drawn at y = 0, 1, 2, ...
  /// @param labels  Tick texts; empty restores the numeric ticks.
  /// @return This plot, for chaining.
  /// @see RPlot::y_tick_labels()
  Plot& y_tick_labels(std::vector<std::string> labels);
  /// @brief Replaces the numeric y ticks with @p labels drawn at data positions @p positions.
  /// @param positions  y data coordinates of the ticks.
  /// @param labels  Tick texts, one per position.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument If the two vectors differ in length.
  /// @see RPlot::y_ticks()
  Plot& y_ticks(std::vector<double> positions, std::vector<std::string> labels);
  /// @brief Overrides the automatic outer margins (in pixels) around the plotting area, e.g. for
  ///        small multi-panel tiles. The defaults leave room for a title, axis labels, and ticks;
  ///        tight margins suit panels whose title and axis labels are empty.
  /// @param left  Left margin in pixels (default 90).
  /// @param right  Right margin in pixels (default 24).
  /// @param top  Top margin in pixels (default 72, plus 24 when a subtitle is set).
  /// @param bottom  Bottom margin in pixels (default 84).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument If any margin is negative or non-finite.
  /// @see RPlot::margins()
  Plot& margins(double left, double right, double top, double bottom);
  /// @brief Sets the tick-label font size in pixels (default 12); the raster viewer draws ticks
  ///        at half-size glyphs when this is below 12.
  /// @param pixels  Font size in pixels.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument If @p pixels is not positive and finite.
  /// @see RPlot::tick_font_size()
  Plot& tick_font_size(double pixels);
  /// @brief Places numeric ticks at round values (steps of 1, 2 or 5 times a power of ten, like
  ///        R's `pretty()`) instead of six evenly spaced ticks across the range (default on).
  /// @param enabled  True turns round ticks on; false restores six even ticks.
  /// @return This plot, for chaining.
  /// @see RPlot::pretty_ticks()
  Plot& pretty_ticks(bool enabled = true);

  /// @brief Returns the canvas width in pixels (default 960).
  /// @return Width set by size().
  std::size_t width() const noexcept { return width_; }
  /// @brief Returns the canvas height in pixels (default 640).
  /// @return Height set by size().
  std::size_t height() const noexcept { return height_; }
  /// @brief Borrows the main heading ("datamunge plot" until set) for the lifetime of this plot.
  /// @return The current title; empty when the heading was cleared.
  const std::string& title_text() const noexcept { return title_; }
  /// @brief Borrows the subtitle for the lifetime of this plot.
  /// @return The current subtitle; empty (the default) when none is drawn.
  const std::string& subtitle_text() const noexcept { return subtitle_; }
  /// @brief Borrows the horizontal-axis label ("x" until set) for the lifetime of this plot.
  /// @return The current x-axis label.
  const std::string& x_label_text() const noexcept { return x_label_; }
  /// @brief Borrows the vertical-axis label ("y" until set) for the lifetime of this plot.
  /// @return The current y-axis label.
  const std::string& y_label_text() const noexcept { return y_label_; }
  /// @brief Borrows the drawn layers (points, lines, bars, ...) in the order they were added.
  /// @return The series list; references are invalidated by adding another layer.
  const std::vector<DataSeries>& series() const noexcept { return series_; }
  /// @brief Returns the canvas fill color set by background().
  /// @return The background color (default white `{255, 255, 255}`).
  RGB background_color() const noexcept { return background_; }
  /// @brief Returns the axis and text color set by axis_color().
  /// @return The axis color (default gray `{55, 65, 81}`).
  RGB axes_color() const noexcept { return axis_color_; }
  /// @brief Returns the grid-line color set by grid_color().
  /// @return The grid color (default light gray `{229, 231, 235}`).
  RGB major_grid_color() const noexcept { return grid_color_; }
  /// @brief Reports whether the background grid is drawn (see show_grid()).
  /// @return True (the default) when the grid is drawn.
  bool grid_visible() const noexcept { return show_grid_; }
  /// @brief Reports whether x_limits() fixed the x range.
  /// @return False (the default) while the x range is fitted to the data.
  bool has_x_limits() const noexcept { return has_x_limits_; }
  /// @brief Reports whether y_limits() fixed the y range.
  /// @return False (the default) while the y range is fitted to the data.
  bool has_y_limits() const noexcept { return has_y_limits_; }
  /// @brief Returns the fixed left x limit; meaningful only when has_x_limits() is true.
  /// @return The stored lower x limit (0 until x_limits() is called).
  double x_min() const noexcept { return x_min_; }
  /// @brief Returns the fixed right x limit; meaningful only when has_x_limits() is true.
  /// @return The stored upper x limit (1 until x_limits() is called).
  double x_max() const noexcept { return x_max_; }
  /// @brief Returns the fixed bottom y limit; meaningful only when has_y_limits() is true.
  /// @return The stored lower y limit (0 until y_limits() is called).
  double y_min() const noexcept { return y_min_; }
  /// @brief Returns the fixed top y limit; meaningful only when has_y_limits() is true.
  /// @return The stored upper y limit (1 until y_limits() is called).
  double y_max() const noexcept { return y_max_; }
  /// @brief Reports whether hide_axes() suppressed axes, ticks, grid, and axis labels.
  /// @return False (the default) when axes are drawn.
  bool axes_hidden() const noexcept { return hide_axes_; }
  /// @brief Borrows the custom x tick texts; empty means numeric ticks.
  /// @return Labels set by x_tick_labels() or x_ticks().
  const std::vector<std::string>& x_tick_label_list() const noexcept { return x_tick_labels_; }
  /// @brief Data positions of x_tick_label_list(); empty means 0, 1, 2, ...
  /// @return Positions set by x_ticks(); cleared by x_tick_labels().
  const std::vector<double>& x_tick_position_list() const noexcept { return x_tick_positions_; }
  /// @brief Borrows the custom y tick texts; empty means numeric ticks.
  /// @return Labels set by y_tick_labels() or y_ticks().
  const std::vector<std::string>& y_tick_label_list() const noexcept { return y_tick_labels_; }
  /// @brief Data positions of y_tick_label_list(); empty means 0, 1, 2, ...
  /// @return Positions set by y_ticks(); cleared by y_tick_labels().
  const std::vector<double>& y_tick_position_list() const noexcept { return y_tick_positions_; }
  /// @brief Borrows the reference lines added by abline(), abline_h(), abline_v(), and qqline().
  /// @return Lines in insertion order; references are invalidated by adding another line.
  const std::vector<ABLine>& reference_lines() const noexcept { return ablines_; }
  /// @brief Borrows the manual legend rows added by legend(); empty means an automatic legend.
  /// @return Manual entries in insertion order.
  const std::vector<LegendEntry>& legend_entries() const noexcept { return legend_entries_; }
  /// @brief True once margins() has been called; the renderer then uses the explicit margins.
  /// @return False (the default) while the automatic margins apply.
  bool has_custom_margins() const noexcept { return has_margins_; }
  /// @brief Returns the left margin in pixels; used only when has_custom_margins() is true.
  /// @return The stored left margin (default 90).
  double margin_left() const noexcept { return margin_left_; }
  /// @brief Returns the right margin in pixels; used only when has_custom_margins() is true.
  /// @return The stored right margin (default 24).
  double margin_right() const noexcept { return margin_right_; }
  /// @brief Returns the top margin in pixels; used only when has_custom_margins() is true.
  /// @return The stored top margin (default 72).
  double margin_top() const noexcept { return margin_top_; }
  /// @brief Returns the bottom margin in pixels; used only when has_custom_margins() is true.
  /// @return The stored bottom margin (default 84).
  double margin_bottom() const noexcept { return margin_bottom_; }
  /// @brief Returns the tick-label font size set by tick_font_size().
  /// @return Font size in pixels (default 12).
  double tick_font_px() const noexcept { return tick_font_size_; }
  /// @brief Reports whether pretty_ticks() is on.
  /// @return False (the default) for six evenly spaced numeric ticks.
  bool pretty_ticks_enabled() const noexcept { return pretty_ticks_; }

  /// @brief Renders the plot as a standalone SVG document and returns it as a string.
  ///
  /// The document starts with an XML declaration and embeds the plot font, so it displays
  /// identically in any browser. The plot is not modified, and no file or display is used.
  ///
  /// @return The complete SVG text.
  /// @throws std::runtime_error  If the plot has no data series, or the canvas is smaller than
  ///         its margins.
  ///
  /// @par Example
  /// @code
  /// const auto svg = dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)).set_title("Tiny").to_svg();
  /// std::print("{}\n", svg.starts_with("<?xml"));  // true
  /// std::print("{}\n", svg.contains("Tiny"));      // true
  /// @endcode
  /// @see save(), show()
  std::string to_svg() const;
  /// @brief Writes the plot to @p path, choosing the format from the file extension
  ///        (R's `svg()`/`png()`/`pdf()` devices followed by `dev.off()`, in one call).
  ///
  /// `.svg` or no extension writes the same text as to_svg(); `.png` writes the raster image
  /// the X11 viewer displays; `.pdf` wraps that raster in a single-page PDF. The extension is
  /// case-sensitive. An existing file is overwritten. Needs no display.
  ///
  /// @param path  Destination file; relative paths resolve against the working directory.
  /// @throws std::invalid_argument  For any other extension (e.g. `.jpg`).
  /// @throws std::runtime_error  If the file cannot be opened for writing, the plot has no data
  ///         series, or the canvas is smaller than its margins.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 9.0)).set_title("Squares");
  /// plt.save("squares.svg");  // vector
  /// plt.save("squares.png");  // raster
  /// plt.save("squares.pdf");  // raster in a one-page PDF
  /// @endcode
  /// @see save_svg(), to_svg(), show()
  void save(const std::string& path) const;
  /// @brief Writes to_svg() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be opened for writing, or rendering fails
  ///         as described for to_svg().
  ///
  /// @par Example
  /// @code
  /// dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).save_svg("diagonal.txt");  // SVG text
  /// @endcode
  /// @see save()
  void save_svg(const std::string& path) const;
  /// @brief Renders the plot as PNG image bytes (the same raster save() writes for `.png`).
  /// @return The complete PNG file contents.
  /// @throws std::runtime_error  If rendering fails as described for to_svg().
  ///
  /// @par Example
  /// @code
  /// const auto png = dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)).to_png();
  /// std::print("{}\n", png[1] == 'P');  // true: PNG signature
  /// @endcode
  /// @see save_png(), to_pdf(), to_svg()
  std::vector<std::uint8_t> to_png() const;
  /// @brief Renders the plot as a one-page PDF holding the raster image.
  /// @return The complete PDF file contents.
  /// @throws std::runtime_error  If rendering fails as described for to_svg().
  /// @see save_pdf(), to_png()
  std::vector<std::uint8_t> to_pdf() const;
  /// @brief Writes to_png() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be written or rendering fails.
  /// @see save(), to_png()
  void save_png(const std::string& path) const;
  /// @brief Writes to_pdf() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be written or rendering fails.
  /// @see save(), to_pdf()
  void save_pdf(const std::string& path) const;
  /// @brief Same as show(); kept as an alias for older code.
  /// @param title_hint  Window title; empty uses the plot's title.
  /// @throws std::runtime_error  As for show().
  /// @see show()
  void view(const std::string& title_hint = "") const;
  /// @brief Opens the plot in an interactive X11 window and blocks until it is closed
  ///        (like printing a plot in an interactive R session).
  ///
  /// The window has toolbar and keyboard zoom/pan (it re-renders from data bounds) and closes
  /// on q or Escape; the function returns only then. The window title is @p title_hint, or
  /// the plot's title (title_text()) when the hint is empty. Requires a build with X11 support and a reachable `DISPLAY`; use save()
  /// in headless code such as tests and CI.
  ///
  /// @param title_hint  Window title; empty (the default) uses the plot's title.
  /// @throws std::runtime_error  If X11 support was not compiled in, the display cannot be
  ///         opened, its visual is unsupported, or the plot has no data series.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::runif(0, 1, 1000);
  /// const auto y = dm::sin(20 / (x + 0.25)) + dm::rnorm(0, 0.2, 1000);
  /// dm::Plot(x, y).set_title("Doppler").set_col("blue").show("Doppler signal");
  /// @endcode
  /// @see save(), to_svg()
  void show(const std::string& title_hint = "") const;

 protected:
  /// @brief Appends a layer as-is (no validation) and returns a reference to the stored copy.
  /// @param series  The layer to store.
  /// @return The stored layer; invalidated by the next add_series() call.
  DataSeries& add_series(DataSeries series);
  /// @brief Appends a reference line as-is (no validation).
  /// @param line  The line to store.
  /// @return This plot, for chaining.
  Plot&       add_reference_line(ABLine line);
  /// @brief Appends a manual legend row as-is (no validation).
  /// @param entry  The row to store.
  /// @return This plot, for chaining.
  Plot&       add_legend_entry(LegendEntry entry);

 private:
  /// @brief RPlot edits series_ directly, e.g. set_col() recolors every stored layer.
  friend class RPlot;

  std::size_t width_{960};                ///< Canvas width in pixels.
  std::size_t height_{640};               ///< Canvas height in pixels.
  std::string title_{"datamunge plot"};   ///< Main heading; empty draws none.

  /// @brief Optional second heading; empty suppresses the subtitle and its extra margin.
  std::string subtitle_;
  std::string x_label_{"x"};              ///< Horizontal-axis label.
  std::string y_label_{"y"};              ///< Vertical-axis label.
  RGB         background_{255, 255, 255}; ///< Canvas fill color.
  RGB         axis_color_{55, 65, 81};    ///< Axis lines and default text color.
  RGB         grid_color_{229, 231, 235}; ///< Grid-line color.
  bool        show_grid_{true};           ///< Whether the background grid is drawn.
  bool        has_x_limits_{false};       ///< True once x_limits() fixed the x range.
  bool        has_y_limits_{false};       ///< True once y_limits() fixed the y range.
  bool        hide_axes_{false};          ///< Whether axes, ticks, grid, and axis labels are hidden.
  double      x_min_{0.0};                ///< Fixed left x limit (used when has_x_limits_).
  double      x_max_{1.0};                ///< Fixed right x limit (used when has_x_limits_).
  double      y_min_{0.0};                ///< Fixed bottom y limit (used when has_y_limits_).
  double      y_max_{1.0};                ///< Fixed top y limit (used when has_y_limits_).
  bool        has_margins_{false};        ///< True once margins() replaced the automatic margins.
  double      margin_left_{90.0};         ///< Left margin in pixels (used when has_margins_).
  double      margin_right_{24.0};        ///< Right margin in pixels (used when has_margins_).
  double      margin_top_{72.0};          ///< Top margin in pixels (used when has_margins_).
  double      margin_bottom_{84.0};       ///< Bottom margin in pixels (used when has_margins_).
  double      tick_font_size_{12.0};      ///< Tick-label font size in pixels.
  bool        pretty_ticks_{true};  ///< round-valued numeric ticks instead of six even ones
  std::vector<DataSeries>  series_;           ///< Drawn layers, in insertion order.
  std::vector<ABLine>      ablines_;          ///< Reference lines, drawn beneath the layers.
  std::vector<LegendEntry> legend_entries_;   ///< Manual legend rows; empty means automatic.
  std::vector<std::string> x_tick_labels_;    ///< non-empty replaces the numeric x ticks
  std::vector<double>      x_tick_positions_; ///< empty: x labels sit at 0, 1, 2, ...
  std::vector<std::string> y_tick_labels_;    ///< non-empty replaces the numeric y ticks
  std::vector<double>      y_tick_positions_; ///< empty: y labels sit at 0, 1, 2, ...
};

/// @brief A single R-base-graphics-style plot, spelled `dm::Plot` in user code.
///
/// Covers R's high-level plotting calls (`plot()`, `hist()`, `barplot()`, `boxplot()`, `pie()`,
/// `curve()`, `qqnorm()`, ...) as constructors and static factories, plus the "add to the
/// current plot" verbs R exposes as separate functions (`points()`, `lines()`, `abline()`,
/// `legend()`, `text()`, ...) as chainable member functions. A plot is an ordinary value: it
/// owns copies of its data, copies cheaply enough to pass around, and draws nothing until it
/// is rendered with to_svg(), save(), or show().
///
/// Two composition styles mix freely. Setters such as set_title() and verbs such as points()
/// mutate the plot and return `RPlot&` for chaining; the pipe `plot | dm::points(...)` (see
/// compose.hpp) appends a reusable layer to a copy. A fresh plot has the title
/// "datamunge plot", axis labels "x" and "y", a 960x640 canvas, and a light grid.
///
/// @par Example
/// @code
/// const auto x = dm::seql(-2.0, 2.0, 9);
/// auto plt = dm::Plot(x, x * x)
///                .set_title("Parabola")
///                .set_xlab("x")
///                .set_ylab("x squared");
/// plt.lines(x, x * x, dm::RGB{220, 38, 38}).abline_h(0.0);
/// plt.save("parabola.svg");
/// @endcode
/// @see dm::points(), dm::legend(), dm::Histogram, dm::Splom, RLayout
class RPlot : public Plot {
 public:
  /// @brief Constructs an empty plot with default labels and styling, ready for layers.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot();
  /// std::print("{} | {} | {}\n", plt.title_text(), plt.x_label_text(), plt.y_label_text());
  /// // datamunge plot | x | y
  /// @endcode
  RPlot() = default;
  /// @brief Constructs a scatter plot of @p y against @p x, like R's `plot(x, y)`.
  ///
  /// Equivalent to `RPlot::plot(x, y)`: one unlabeled scatter series drawn in blue
  /// `{37, 99, 235}` with 4-pixel markers; axis ranges fit the data. The coordinates are
  /// copied (or moved from rvalues); `dm::Vector<double>` converts implicitly. Restyle the
  /// points afterwards with set_col().
  ///
  /// @param x  Horizontal coordinates, one per point.
  /// @param y  Vertical coordinates, paired with @p x by position.
  /// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// auto plt = dm::Plot(iris.get_column("Petal.Length").values(),
  ///                     iris.get_column("Petal.Width").values())
  ///                .set_col("blue");
  /// plt.save("petals.svg");
  /// @endcode
  /// @see points(), set_col(), plot()
  RPlot(std::vector<double> x, std::vector<double> y);

  /// @brief Returns a fresh empty plot; same as the default constructor `dm::Plot()`.
  ///
  /// Kept for language bindings, where a named factory reads better than a constructor.
  ///
  /// @return An empty plot with default labels and styling.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot::create();
  /// std::print("{}\n", plt.series().size());  // 0
  /// @endcode
  /// @see RPlot()
  static RPlot create();

  /// @brief Applies C++ option objects such as PlotArgs and returns this plot for chaining.
  ///
  /// Calls `options.apply(*this)` once; the option group decides what it changes.
  ///
  /// Exceptions thrown by `Options::apply()` propagate; PlotArgs::apply() throws none.
  ///
  /// @tparam Options An owned option group exposing apply(RPlot&); the plot retains no reference to it.
  /// @param options  The option group to apply.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(3.0, 4.0))
  ///                .with(dm::PlotArgs{{"title", "Line"}, {"ylab", "height"}});
  /// std::print("{} | {}\n", plt.title_text(), plt.y_label_text());  // Line | height
  /// @endcode
  /// @see dm::PlotArgs
  template <typename Options>
  RPlot& with(const Options& options) {
    options.apply(*this);
    return *this;
  }

  /// @brief Sets the canvas size in pixels (default 960 by 640), like the `width`/`height`
  ///        of R's `png()` or `svg()` device.
  ///
  /// Not validated here: rendering throws std::runtime_error when the canvas leaves no room
  /// inside the margins (by default the canvas must exceed 114 by 156 pixels).
  ///
  /// @param width  Canvas width in pixels.
  /// @param height  Canvas height in pixels.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).size(400, 300);
  /// std::print("{}x{}\n", plt.width(), plt.height());  // 400x300
  /// @endcode
  /// @see margins()
  RPlot& size(std::size_t width, std::size_t height);
  /// @brief Sets the main heading, like R's `main =`; an empty string removes it.
  /// @param value  Heading text (copied).
  /// @return This plot, for chaining.
  /// @see set_title()
  RPlot& title(std::string value);
  /// @brief Sets the subtitle and returns this plot for chaining RPlot verbs; same as set_subtitle().
  /// @param value  Subtitle text drawn under the title; empty removes it.
  /// @return This plot, for chaining.
  /// @see set_subtitle()
  RPlot& subtitle(std::string value);
  /// @brief Sets the horizontal-axis label; same as xlab() and set_xlab().
  /// @param value  Label text (copied).
  /// @return This plot, for chaining.
  /// @see set_xlab()
  RPlot& x_label(std::string value);
  /// @brief Sets the vertical-axis label; same as ylab() and set_ylab().
  /// @param value  Label text (copied).
  /// @return This plot, for chaining.
  /// @see set_ylab()
  RPlot& y_label(std::string value);
  /// @brief R-style alias for x_label(), like R's `xlab =`.
  /// @param value  Label text (copied).
  /// @return This plot, for chaining.
  /// @see set_xlab()
  RPlot& xlab(std::string value);
  /// @brief R-style alias for y_label(), like R's `ylab =`.
  /// @param value  Label text (copied).
  /// @return This plot, for chaining.
  /// @see set_ylab()
  RPlot& ylab(std::string value);
  /// @brief Sets the main heading drawn above the plotting area (R's `main =`).
  ///
  /// The plot owns a copy of the text. An empty string clears the heading; a fresh plot
  /// shows "datamunge plot" until this is called. Same as title().
  ///
  /// @param value  Heading text.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(2.0, 4.0, 8.0))
  ///                .set_title("Growth")
  ///                .set_subtitle("Doubling every step")
  ///                .set_xlab("Step")
  ///                .set_ylab("Count");
  /// std::print("{} / {}\n", plt.title_text(), plt.subtitle_text());
  /// // Growth / Doubling every step
  /// @endcode
  /// @see set_subtitle(), set_xlab(), set_ylab(), dm::title()
  RPlot& set_title(std::string value);
  /// @brief Sets the smaller second heading drawn beneath the title (unlike R's `sub =`,
  ///        which sits below the x axis).
  ///
  /// The plot owns a copy of the text. Empty (the default) draws no subtitle and reserves no
  /// extra top margin. Same as subtitle().
  ///
  /// @param value  Subtitle text.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot().set_title("Iris Dataset").set_subtitle("Sepal Width vs Length");
  /// plt.set_subtitle("");  // remove it again
  /// @endcode
  /// @see set_title(), dm::subtitle()
  RPlot& set_subtitle(std::string value);
  /// @brief Sets the horizontal-axis label (R's `xlab =`).
  ///
  /// The plot owns a copy of the text; the default is "x" and an empty string hides the
  /// label. Same as xlab() and x_label().
  ///
  /// @param value  Axis label text.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(3.0, 4.0)).set_xlab("Sepal Width");
  /// std::print("{}\n", plt.x_label_text());  // Sepal Width
  /// @endcode
  /// @see set_ylab(), dm::xlab()
  RPlot& set_xlab(std::string value);
  /// @brief Sets the vertical-axis label (R's `ylab =`).
  ///
  /// The plot owns a copy of the text; the default is "y" and an empty string hides the
  /// label. Same as ylab() and y_label().
  ///
  /// @param value  Axis label text.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(3.0, 4.0)).set_ylab("Sepal Length");
  /// std::print("{}\n", plt.y_label_text());  // Sepal Length
  /// @endcode
  /// @see set_xlab(), dm::ylab()
  RPlot& set_ylab(std::string value);
  /// @brief Recolors every data series already on the plot (R's `col =` applied after the fact).
  ///
  /// Every existing series (points, lines, bars, ...) takes @p color, and per-point palettes
  /// from dm::points(...).set_col() or points_colored() are cleared, so the whole plot becomes
  /// one color. Only layers present now are affected: points added later keep their own
  /// colors. Reference lines (abline) and manual legend entries keep their explicit colors.
  ///
  /// @param color  New color, including its alpha.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(3.0, 1.0, 2.0))
  ///                .set_col(dm::RGB{220, 100, 255});
  /// std::print("{}\n", plt.series()[0].color.r);  // 220
  /// @endcode
  /// @see set_col(std::string), dm::points()
  RPlot& set_col(RGB color);
  /// @brief Same as set_col(RGB) but takes a color name or hex string, like R's `col = "blue"`.
  ///
  /// Matching is case-insensitive. Names are black, white, red, green, blue, yellow, cyan,
  /// magenta, gray/grey, and transparent (a much smaller set than R's `colors()`). Green is
  /// RGB(0,128,0) as in CSS, and transparent is black with zero alpha. Hex strings are
  /// `#RRGGBB` or `#RRGGBBAA`, where AA is an alpha byte (`80` is about 0.5).
  ///
  /// @param color  Color name or hex string.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  For unknown names or malformed hex; the plot is unchanged.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::runif(0, 1, 100);
  /// auto plt = dm::Plot(x, dm::rnorm(0, 1, 100)).set_col("blue");
  /// plt.set_col("#FF000080");  // half-transparent red
  /// const auto& col = plt.series()[0].color;
  /// std::print("{} {:.2f}\n", col.r, col.alpha);  // 255 0.50
  /// @endcode
  /// @see set_col(RGB)
  RPlot& set_col(std::string color);
  /// @brief Sets the fill color of the whole canvas, like R's `par(bg =)`.
  ///
  /// The default is white `{255, 255, 255}`. The color fills the margins as well as the
  /// plotting area. Its alpha is ignored: the background is always drawn opaque.
  ///
  /// @param color  Canvas fill color.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0))
  ///                .background(dm::RGB{17, 24, 39})
  ///                .axis_color(dm::RGB{229, 231, 235})
  ///                .grid_color(dm::RGB{55, 65, 81});  // a dark theme
  /// std::print("{} {}\n", plt.background_color().r, plt.axes_color().r);  // 17 229
  /// @endcode
  ///
  /// @par References
  /// - R `par()` (`bg`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
  ///
  /// @see axis_color(), grid_color()
  RPlot& background(RGB color);
  /// @brief Sets the color of the axis lines and of all plot-level text: tick labels, axis
  ///        labels, title, and subtitle (R's `col.axis`, `col.lab`, and `col.main` in one).
  ///
  /// The default is slate gray `{55, 65, 81}`. Legend text follows it too; text() labels and
  /// data layers keep their own colors.
  ///
  /// @param color  Axis and text color.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// See background(), which builds a dark theme with all three color setters.
  ///
  /// @par References
  /// - R `par()` (`col.axis`, `col.lab`, `col.main`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
  ///
  /// @see background(), grid_color()
  RPlot& axis_color(RGB color);
  /// @brief Sets the color of the background grid lines (default light gray `{229, 231, 235}`).
  ///
  /// Has no visible effect while the grid is off (show_grid(false)) or the axes are hidden.
  ///
  /// @param color  Grid-line color.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// See background(), which builds a dark theme with all three color setters.
  /// @see show_grid(), background()
  RPlot& grid_color(RGB color);
  /// @brief Turns the light background grid on or off (on by default), like adding or
  ///        omitting R's `grid()`.
  ///
  /// One vertical line per x tick and one horizontal line per y tick, so the grid always lines
  /// up with the tick labels (round values, evenly spaced ticks, or custom tick positions).
  /// Hidden axes hide the grid too.
  ///
  /// @param enabled  True (the default) draws the grid; false omits it.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).show_grid(false);
  /// std::print("{}\n", plt.grid_visible());  // false
  /// @endcode
  ///
  /// @par References
  /// - R `grid()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/grid.html
  ///
  /// @see grid_color(), hide_axes()
  RPlot& show_grid(bool enabled = true);
  /// @brief Fixes the visible x range, like R's `xlim = c(min_x, max_x)`.
  ///
  /// Without limits the x range is fitted to every layer (and vertical reference line), plus
  /// 5% padding on each side. With limits, the plotting area spans exactly [@p min_x, @p max_x]
  /// with no padding (R pads `xlim` by 4% unless `xaxs = "i"`). Data outside the range is not
  /// clipped and may be drawn over the margins.
  ///
  /// @param min_x  Left edge of the plotting area, in data units.
  /// @param max_x  Right edge; must be greater than @p min_x.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  Unless `max_x > min_x` (NaN fails); the plot is unchanged.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 9.0))
  ///                .x_limits(0.0, 4.0)
  ///                .y_limits(0.0, 10.0);
  /// std::print("{} {} {}\n", plt.has_x_limits(), plt.x_max(), plt.y_max());  // true 4 10
  /// @endcode
  ///
  /// @par References
  /// - R `plot.default()` (`xlim`, `ylim`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/plot.default.html
  ///
  /// @see y_limits()
  RPlot& x_limits(double min_x, double max_x);
  /// @brief Fixes the visible y range, like R's `ylim = c(min_y, max_y)`.
  ///
  /// Without limits the y range is fitted to every layer, plus 10% padding at each end (bar
  /// layers also include y = 0). With limits, the plotting area spans exactly
  /// [@p min_y, @p max_y]. Data outside the range is not clipped.
  ///
  /// @param min_y  Bottom edge of the plotting area, in data units.
  /// @param max_y  Top edge; must be greater than @p min_y.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  Unless `max_y > min_y` (NaN fails); the plot is unchanged.
  ///
  /// @par Example
  /// See x_limits(), which sets both ranges.
  ///
  /// @par References
  /// - R `plot.default()` (`xlim`, `ylim`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/plot.default.html
  ///
  /// @see x_limits()
  RPlot& y_limits(double min_y, double max_y);
  /// @brief Hides the axis lines, tick labels, grid, and axis labels, like R's `axes = FALSE,
  ///        ann = FALSE` (the title and subtitle are still drawn).
  ///
  /// pie() turns this on. Margins are unchanged, so call margins() as well to let the data use
  /// the freed space.
  ///
  /// @param enabled  True (the default) hides them; false shows them again.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)).hide_axes();
  /// std::print("{}\n", plt.axes_hidden());  // true
  /// @endcode
  ///
  /// @par References
  /// - R `plot.default()` (`axes`, `ann`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/plot.default.html
  ///
  /// @see show_grid(), margins()
  RPlot& hide_axes(bool enabled = true);
  /// @brief Replaces the numeric x ticks with @p labels at x = 0, 1, 2, ..., like R's
  ///        `axis(1, at = 0:(n - 1), labels = labels)`.
  ///
  /// Matches the bar positions of barplot() and boxplot(). Clears any positions set by
  /// x_ticks(); an empty vector restores the numeric ticks.
  ///
  /// @param labels  Tick texts, in order.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(0.0, 1.0, 2.0), dm::c(5.0, 3.0, 4.0))
  ///                .x_tick_labels({"low", "mid", "high"});
  /// std::print("{}\n", plt.x_tick_label_list()[2]);  // high
  /// @endcode
  ///
  /// @par References
  /// - R `axis()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/axis.html
  ///
  /// @see x_ticks(), y_tick_labels()
  RPlot& x_tick_labels(std::vector<std::string> labels);
  /// @brief Replaces the numeric x ticks with @p labels drawn at data positions @p positions,
  ///        like R's `axis(1, at = positions, labels = labels)`.
  ///
  /// Positions need not be sorted or inside the visible range. Two empty vectors restore the
  /// numeric ticks.
  ///
  /// @param positions  x data coordinates of the ticks.
  /// @param labels  Tick texts, one per position.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the two vectors differ in length; the plot is unchanged.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::seql(0.0, 6.283, 50);
  /// auto plt = dm::Plot(x, dm::sin(x))
  ///                .x_ticks({0.0, 3.1416, 6.2832}, {"0", "pi", "2 pi"})
  ///                .y_ticks({-1.0, 0.0, 1.0}, {"min", "0", "max"});
  /// std::print("{}\n", plt.x_tick_position_list()[1]);  // 3.1416
  /// @endcode
  ///
  /// @par References
  /// - R `axis()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/axis.html
  ///
  /// @see x_tick_labels(), y_ticks()
  RPlot& x_ticks(std::vector<double> positions, std::vector<std::string> labels);
  /// @brief Replaces the numeric y ticks with @p labels at y = 0, 1, 2, ..., like R's
  ///        `axis(2, at = 0:(n - 1), labels = labels)`.
  ///
  /// Clears any positions set by y_ticks(); an empty vector restores the numeric ticks.
  ///
  /// @param labels  Tick texts, in order.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(3.0, 5.0), dm::c(0.0, 1.0)).y_tick_labels({"no", "yes"});
  /// std::print("{}\n", plt.y_tick_label_list().size());  // 2
  /// @endcode
  ///
  /// @par References
  /// - R `axis()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/axis.html
  ///
  /// @see y_ticks(), x_tick_labels()
  RPlot& y_tick_labels(std::vector<std::string> labels);
  /// @brief Replaces the numeric y ticks with @p labels drawn at data positions @p positions,
  ///        like R's `axis(2, at = positions, labels = labels)`.
  ///
  /// Two empty vectors restore the numeric ticks.
  ///
  /// @param positions  y data coordinates of the ticks.
  /// @param labels  Tick texts, one per position.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the two vectors differ in length; the plot is unchanged.
  ///
  /// @par Example
  /// See x_ticks(), which sets both axes.
  ///
  /// @par References
  /// - R `axis()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/axis.html
  ///
  /// @see y_tick_labels(), x_ticks()
  RPlot& y_ticks(std::vector<double> positions, std::vector<std::string> labels);
  /// @brief Overrides the automatic outer margins around the plotting area, in pixels, like
  ///        R's `par(mai =)` (which uses inches and orders bottom, left, top, right).
  ///
  /// The automatic margins are left 90, right 24, top 72 (96 when a subtitle is set), and
  /// bottom 84; they leave room for the title, axis labels, and ticks. Explicit margins are used
  /// as given, with no extra room for a subtitle. Tight margins suit small multi-panel tiles
  /// whose title and axis labels are empty. Rendering throws std::runtime_error if the margins
  /// leave no plotting area.
  ///
  /// @param left  Left margin in pixels.
  /// @param right  Right margin in pixels.
  /// @param top  Top margin in pixels.
  /// @param bottom  Bottom margin in pixels.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If any margin is negative or non-finite; the plot is unchanged.
  ///
  /// @par Example
  /// @code
  /// auto tile = dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0))
  ///                 .set_title("").set_xlab("").set_ylab("")
  ///                 .margins(30, 10, 10, 20)
  ///                 .tick_font_size(9);
  /// std::print("{} {}\n", tile.has_custom_margins(), tile.margin_left());  // true 30
  /// @endcode
  ///
  /// @par References
  /// - R `par()` (`mai`, `mar`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
  ///
  /// @see size(), tick_font_size(), RLayout
  RPlot& margins(double left, double right, double top, double bottom);
  /// @brief Sets the tick-label font size in pixels (default 12), like R's `cex.axis`.
  ///
  /// In SVG output the size is used as given. The raster outputs (PNG, PDF, show()) draw tick
  /// labels with half-size glyphs when the size is below 12.
  ///
  /// @param pixels  Font size in pixels.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If @p pixels is not positive and finite; the plot is unchanged.
  ///
  /// @par Example
  /// See margins(), which shrinks the ticks for a small tile.
  ///
  /// @par References
  /// - R `par()` (`cex.axis`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
  ///
  /// @see margins()
  RPlot& tick_font_size(double pixels);
  /// @brief Places numeric ticks at round values (default on) instead of six evenly spaced
  ///        ticks across the range, in the spirit of R's `pretty()`.
  ///
  /// The step is 1, 2, or 5 times a power of ten, chosen as the "nice number" nearest a fifth
  /// of the visible range (Heckbert's rule), and ticks are its multiples inside the range.
  /// That gives about four to seven ticks, close to R's `pretty()`: a padded range of -0.365
  /// to 7.665 gets ticks at 0, 2, 4 and 6. Custom tick labels take precedence. The background
  /// grid follows the ticks.
  ///
  /// @param enabled  True turns round ticks on; false restores six even ticks.
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(0.0, 100.0), dm::c(0.0, 1.0)).pretty_ticks();
  /// std::print("{}\n", plt.to_svg().contains(">40</text>"));  // true: ticks 0, 20, ..., 100
  /// @endcode
  ///
  /// @par References
  /// - R `pretty()`: https://stat.ethz.ch/R-manual/R-devel/library/base/html/pretty.html
  /// - P. S. Heckbert (1990). "Nice Numbers for Graph Labels". In Graphics Gems, 61-63.
  ///   Academic Press. doi:10.1016/B978-0-08-050753-8.50020-8
  ///
  /// @see x_ticks(), show_grid()
  RPlot& pretty_ticks(bool enabled = true);

  /// @brief Builds a scatter, line, or line-plus-points plot of @p y against @p x, like R's
  ///        `plot(x, y, type = "p" | "l" | "b")`.
  ///
  /// `"p"` (the default) adds one scatter series with 4-pixel markers; `"l"` adds one 2-pixel
  /// line joining the points in input order; `"b"` adds the line and then unlabeled points on
  /// top (R's `"b"` leaves gaps in the line around each point; this draws it through them). The
  /// title and axis labels keep their defaults. `RPlot(x, y)` is `plot(x, y)`.
  ///
  /// @param x  Horizontal coordinates, one per point.
  /// @param y  Vertical coordinates, paired with @p x by position.
  /// @param type  `"p"` (default), `"l"`, or `"b"`; case-sensitive.
  /// @param label  Legend text; empty (the default) keeps the plot out of the automatic legend.
  ///        With `"b"` only the line carries it.
  /// @param color  Color of the points and line (default blue `{37, 99, 235}`).
  /// @return A new plot holding the series.
  /// @throws std::invalid_argument  If the coordinates are empty or differ in length, or
  ///         @p type is not one of the three codes.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::seql(0.0, 4.0, 5);
  /// auto plt = dm::Plot::plot(x, x * x, "b", "x squared");
  /// std::print("{} series\n", plt.series().size());  // 2 series
  /// @endcode
  ///
  /// @par References
  /// - R `plot.default()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/plot.default.html
  ///
  /// @see RPlot(std::vector<double>, std::vector<double>), points(), lines()
  static RPlot plot(std::vector<double> x,
                     std::vector<double> y,
                     std::string         type = "p",
                     std::string         label = "",
                     RGB                 color = {37, 99, 235});

  /// @brief Builds a frequency histogram of @p data with @p bins equal-width bins, like R's
  ///        `hist(x, breaks = bins)`.
  ///
  /// The bins split [min, max] of the data into equal widths (R instead treats `breaks` as a
  /// suggestion and uses `pretty()` boundaries). Bins include their left edge and exclude their
  /// right edge, except that the maximum goes in the final bin; R's default (`right = TRUE`) is
  /// the opposite, so a value on an inner boundary is counted one bin higher here. Constant
  /// data uses the range [value, value + 1]. Each bin is one bar of height equal to its count,
  /// centered on the bin with width equal to the bin width. The title is "Histogram", the x
  /// label "x", and the y label "Frequency". The data is consumed; the plot keeps only counts.
  ///
  /// @param data  Observations; all must be finite.
  /// @param bins  Number of bins (default 10).
  /// @param label  Legend text; empty (the default) shows no legend.
  /// @param color  Bar color (default light blue `{96, 165, 250}`).
  /// @return A new plot holding one Bar series.
  /// @throws std::invalid_argument For empty/non-finite data, zero bins, or a range/bin
  ///         width that cannot be represented as a finite positive double.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot::hist({1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0}, 3);
  /// const auto& bars = plt.series()[0];  // bins [1,2) [2,3) [3,4]
  /// std::print("{} {} {}\n", bars.y[0], bars.y[1], bars.y[2]);  // 1 2 4
  /// @endcode
  ///
  /// @par References
  /// - R `hist()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/hist.html
  ///
  /// @see dm::Histogram, bars()
  static RPlot hist(std::vector<double> data,
                     std::size_t         bins = 10,
                     std::string         label = "",
                     RGB                 color = {96, 165, 250});

  /// @brief Builds a bar chart with one bar per height, like R's
  ///        `barplot(heights, names.arg = names)`.
  ///
  /// Bar i is centered at x = i (0, 1, 2, ...) and is 0.8 units wide (R puts default bars at
  /// 0.7, 1.9, 3.1, ...). Bars rise from 0, so negative heights point down, and the y range
  /// always includes 0. Nonempty @p names become x tick labels under the bars. The title and
  /// axis labels keep their defaults.
  ///
  /// @param heights  Bar heights, in display order.
  /// @param names  Tick labels, one per bar; empty (the default) keeps numeric ticks.
  /// @param label  Legend text; empty (the default) shows no legend.
  /// @param color  Bar color (default green `{22, 163, 74}`).
  /// @return A new plot holding one Bar series.
  /// @throws std::invalid_argument  If @p heights is empty or @p names is nonempty with a
  ///         different length.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot::barplot({3.0, 5.0, 2.0}, {"a", "b", "c"}).set_title("Counts");
  /// std::print("{} {}\n", plt.series()[0].x[2], plt.x_tick_label_list()[2]);  // 2 c
  /// @endcode
  ///
  /// @par References
  /// - R `barplot()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/barplot.html
  ///
  /// @see bars(), x_tick_labels()
  static RPlot barplot(std::vector<double>      heights,
                        std::vector<std::string> names = {},
                        std::string              label = "",
                        RGB                       color = {22, 163, 74});

  /// @brief Builds side-by-side box-and-whisker plots, one per group, like R's
  ///        `boxplot(list(...), names = names)`.
  ///
  /// Group g is drawn at x = g (0, 1, 2, ...) as a 0.6-unit-wide box from the first to the
  /// third quartile with a line at the median. Quartiles are R's default "type 7" sample
  /// quantiles. Whiskers reach the most extreme values within 1.5 IQR of the box (Tukey's
  /// rule), and values beyond are drawn as outlier points. R's `boxplot()` instead uses
  /// Tukey's hinges (`fivenum()`), so for some sample sizes the box edges and the fences differ
  /// slightly from R: for 1, ..., 9, 40 this gives quartiles 3.25 and 7.75 where R draws
  /// hinges at 3 and 8. All boxes share @p color; the title and axis labels keep their defaults.
  ///
  /// @param groups  One vector of observations per box; each must be nonempty.
  /// @param names  Tick labels under the boxes; empty (the default) keeps numeric ticks.
  /// @param color  Box, whisker, and outlier color (default light blue `{96, 165, 250}`).
  /// @return A new plot holding one Box series per group (see DataSeries::Kind::Box for its layout).
  /// @throws std::invalid_argument  If @p groups or any group is empty, or @p names is nonempty
  ///         with a different length.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot::boxplot({{1, 2, 3, 4, 5, 6, 7, 8, 9, 40}, {2, 4, 6}}, {"a", "b"});
  /// const auto& y = plt.series()[0].y;  // whisker_lo, q1, median, q3, whisker_hi, outliers...
  /// std::print("{} {} {} {} {} | {}\n", y[0], y[1], y[2], y[3], y[4], y[5]);
  /// // 1 3.25 5.5 7.75 9 | 40
  /// plt.save("boxes.svg");
  /// @endcode
  ///
  /// @par References
  /// - J. W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley. (Box plot and the 1.5 IQR fences.)
  /// - R. McGill, J. W. Tukey, W. A. Larsen (1978). "Variations of Box Plots". The American
  ///   Statistician 32(1), 12-16. doi:10.1080/00031305.1978.10479236
  /// - R. J. Hyndman, Y. Fan (1996). "Sample Quantiles in Statistical Packages". The American
  ///   Statistician 50(4), 361-365. doi:10.1080/00031305.1996.10473566 (quantile type 7.)
  /// - R `boxplot()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/boxplot.html
  /// - R `boxplot.stats()`: https://stat.ethz.ch/R-manual/R-devel/library/grDevices/html/boxplot.stats.html
  ///
  /// @see box(), x_tick_labels()
  static RPlot boxplot(std::vector<std::vector<double>> groups,
                        std::vector<std::string>         names = {},
                        RGB                               color = {96, 165, 250});

  /// @brief boxplot() with one color per box, like R's `boxplot(..., col = colors)`.
  ///
  /// @param groups  One vector of observations per box; each must be nonempty.
  /// @param names  Tick labels under the boxes; empty keeps numeric ticks.
  /// @param colors  Box colors, recycled when shorter than @p groups; must be nonempty.
  /// @return A new plot holding one Box series per group.
  /// @throws std::invalid_argument  As boxplot(), or if @p colors is empty.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::graphics::RPlot::boxplot({{1, 2, 3}, {2, 4, 6}}, {"a", "b"},
  ///                                     std::vector<dm::RGB>{{78, 121, 167}, {242, 142, 43}});
  /// @endcode
  /// @see boxplot(std::vector<std::vector<double>>, std::vector<std::string>, RGB)
  static RPlot boxplot(std::vector<std::vector<double>> groups,
                        std::vector<std::string>         names,
                        std::vector<RGB>                  colors);

  /// @brief Builds a pie chart with one wedge per value, like R's
  ///        `pie(x, labels = names, col = colors, clockwise = TRUE)`.
  ///
  /// Wedge i spans the fraction values[i] / sum(values) of the circle. Wedges start at 12
  /// o'clock and run clockwise (R's default starts at 3 o'clock and runs counter-clockwise).
  /// Each wedge is a filled unit-radius polygon with 24 arc segments. Names, when given, are
  /// drawn at radius 1.15 beside the middle of each wedge's arc and also label the wedges, so the
  /// automatic legend lists them. Colors are recycled; empty uses an 8-color built-in palette
  /// starting blue, red, green. The axes and grid are hidden and both ranges are fixed at
  /// [-1.4, 1.4]. The pie is drawn in data units, so it is round only when the plotting area is
  /// square, e.g. `.size(598, 640)` with the default margins; the default 960x640 canvas draws an
  /// ellipse.
  ///
  /// @param values  Wedge sizes; nonnegative, with a positive sum.
  /// @param names  Wedge labels; empty (the default) draws none.
  /// @param colors  Wedge colors, recycled; empty (the default) uses the built-in palette.
  /// @return A new plot holding one Polygon series per wedge plus one Text series per name.
  /// @throws std::invalid_argument  If @p values is empty, has a negative entry, or sums to 0,
  ///         or @p names is nonempty with a different length.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot::pie({3.0, 2.0, 1.0}, {"rent", "food", "fun"}).size(598, 640);
  /// std::print("{}\n", plt.series().size());  // 6
  /// plt.save("budget.svg");
  /// @endcode
  ///
  /// @par References
  /// - R `pie()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/pie.html
  ///
  /// @see polygon(), text()
  static RPlot pie(std::vector<double>      values,
                    std::vector<std::string> names = {},
                    std::vector<RGB>         colors = {});

  /// @brief Plots a function over [@p from, @p to] as a line, like R's `curve(expr, from, to, n)`.
  ///
  /// Evaluates `f.call(x)` at @p n evenly spaced points including both ends, in increasing
  /// order, and joins them with a 2-pixel line. @p f is used only during this call and is not
  /// retained. Non-finite results are stored as-is. Exceptions thrown by `f.call()` propagate.
  ///
  /// @param f  The function to draw: a dm::Callback subclass overriding `call(double)`.
  /// @param from  Left end of the range.
  /// @param to  Right end; must be greater than @p from.
  /// @param n  Number of sample points, at least 2 (default 101, as in R).
  /// @param label  Legend text; empty (the default) shows no legend.
  /// @param color  Line color (default red `{220, 38, 38}`).
  /// @return A new plot holding one Line series.
  /// @throws std::invalid_argument  If @p n is below 2 or @p to is not greater than @p from.
  ///
  /// @par Example
  /// @code
  /// struct Square : dm::Callback {
  ///   double call(double x) override { return x * x; }
  /// };
  /// Square square;
  /// auto plt = dm::Plot::curve(square, -1.0, 1.0, 5);
  /// std::print("{}\n", plt.series()[0].y);  // [1, 0.25, 0, 0.25, 1]
  /// @endcode
  ///
  /// @par References
  /// - R `curve()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/curve.html
  ///
  /// @see lines()
  static RPlot curve(datamunge::Callback& f,
                      double               from,
                      double               to,
                      std::size_t          n = 101,
                      std::string          label = "",
                      RGB                  color = {220, 38, 38});

  /// @brief Builds a normal Q-Q plot: the sorted @p data against standard-normal quantiles,
  ///        like R's `qqnorm(y)`.
  ///
  /// The i-th smallest value (i = 1..n) is plotted against the normal quantile of the plotting
  /// position (i - 0.5) / n. R's `ppoints()` uses that rule only for n > 10 and
  /// (i - 3/8) / (n + 1/4) for n <= 10, so the x coordinates differ from R for small samples.
  /// Points that fall on a straight line indicate normally distributed data; add the reference
  /// line with qqline(). The title is "Normal Q-Q", the axis labels "Theoretical Quantiles"
  /// and "Sample Quantiles".
  ///
  /// @param data  The sample; at least two values. Consumed (sorted in place).
  /// @param label  Legend text (default "sample", so the automatic legend shows one row; pass
  ///        "" to hide it).
  /// @param color  Point color (default blue `{37, 99, 235}`).
  /// @return A new plot holding one Scatter series.
  /// @throws std::invalid_argument  If @p data has fewer than two values.
  ///
  /// @par Example
  /// @code
  /// const std::vector<double> sample{3.0, 1.0, 2.0, 5.0, 4.0};
  /// auto plt = dm::Plot::qqnorm(sample).qqline(sample);
  /// std::print("{:.4f}\n", plt.series()[0].x[0]);  // -1.2816
  /// @endcode
  ///
  /// @par References
  /// - M. B. Wilk, R. Gnanadesikan (1968). "Probability plotting methods for the analysis of
  ///   data". Biometrika 55(1), 1-17. doi:10.1093/biomet/55.1.1
  /// - R `qqnorm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/qqnorm.html
  /// - R `ppoints()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ppoints.html
  ///
  /// @see qqline()
  static RPlot qqnorm(std::vector<double> data, std::string label = "sample", RGB color = {37, 99, 235});

  /// @brief Adds a scatter series to this plot, like R's `points(x, y, col = color)`.
  ///
  /// The coordinates are copied (or moved from rvalues) into a new series drawn above the
  /// existing ones; axis ranges grow to include it. A nonempty @p label adds a row to the
  /// automatic legend (shown only while no manual legend() entries exist).
  ///
  /// @param x  Horizontal coordinates, one per point.
  /// @param y  Vertical coordinates, paired with @p x by position.
  /// @param label  Legend text for this series; empty (the default) keeps it out of the legend.
  /// @param color  Marker color for every point (default blue `{37, 99, 235}`).
  /// @param marker_size  Marker radius in pixels (default 4).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::seql(0.0, 6.0, 13);
  /// auto plt = dm::Plot().set_title("sin and cos");
  /// plt.points(x, dm::sin(x), "sin", dm::RGB{37, 99, 235})
  ///    .points(x, dm::cos(x), "cos", dm::RGB{220, 38, 38}, 6.0);
  /// plt.save("trig.svg");  // automatic legend lists "sin" and "cos"
  /// @endcode
  ///
  /// @par References
  /// - R `points()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/points.html
  ///
  /// @see points_colored(), dm::points(), set_col()
  RPlot& points(std::vector<double> x,
                std::vector<double> y,
                std::string         label = "",
                RGB                 color = {37, 99, 235},
                double              marker_size = 4.0);
  /// @brief Appends scatter points with a recycled per-point RGB(A) palette.
  /// Coordinates must have equal nonzero lengths; colors must not be empty.
  /// A partial palette cycle is allowed without a warning. Colors retain their own alpha.
  /// The series is unlabeled, so it never appears in the automatic legend.
  /// @param x  Horizontal coordinates.
  /// @param y  Vertical coordinates, paired with @p x.
  /// @param colors  Palette; point i uses `colors[i % colors.size()]`.
  /// @param marker_size  Marker radius in pixels (default 4).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument If the coordinate lengths or palette are invalid.
  ///
  /// @par Example
  /// @code
  /// const auto palette = dm::c(dm::RGB{255, 0, 0}, dm::RGB{0, 0, 255});
  /// auto plt = dm::Plot().points_colored(dm::c(1.0, 2.0, 3.0), dm::c(3.0, 2.0, 1.0), palette);
  /// std::print("{}\n", plt.series()[0].point_colors.size());  // 2 (red, blue, red)
  /// @endcode
  ///
  /// @par References
  /// - R `points()` (vector `col`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/points.html
  ///
  /// @see dm::points() and Points::set_col(), the pipe-friendly form of this call.
  RPlot& points_colored(std::vector<double> x, std::vector<double> y,
                       std::vector<RGB> colors, double marker_size = 4.0);
  /// @brief Adds a line joining the points in input order, like R's `lines(x, y, col = color,
  ///        lwd = stroke_width)`; same as lines().
  ///
  /// The points are not sorted, so unsorted x draws a zigzag. The coordinates are copied (or
  /// moved from rvalues) into a new Line series drawn above existing layers; axis ranges grow
  /// to include it. A nonempty @p label adds a row to the automatic legend.
  ///
  /// @param x  Horizontal coordinates, one per vertex.
  /// @param y  Vertical coordinates, paired with @p x by position.
  /// @param label  Legend text; empty (the default) keeps it out of the legend.
  /// @param color  Line color (default red `{220, 38, 38}`).
  /// @param stroke_width  Line width in pixels (default 2).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
  ///
  /// @par Example
  /// See lines().
  ///
  /// @par References
  /// - R `lines()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/lines.html
  ///
  /// @see lines(), plot()
  RPlot& line(std::vector<double> x,
              std::vector<double> y,
              std::string         label = "",
              RGB                 color = {220, 38, 38},
              double              stroke_width = 2.0);
  /// @brief Adds a line joining the points in input order, like R's `lines(x, y, col = color,
  ///        lwd = stroke_width)`.
  ///
  /// Same as line(). The points are not sorted, so unsorted x draws a zigzag. The coordinates
  /// are copied (or moved from rvalues) into a new Line series drawn above existing layers;
  /// axis ranges grow to include it. A nonempty @p label adds a row to the automatic legend.
  ///
  /// @param x  Horizontal coordinates, one per vertex.
  /// @param y  Vertical coordinates, paired with @p x by position.
  /// @param label  Legend text; empty (the default) keeps it out of the legend.
  /// @param color  Line color (default red `{220, 38, 38}`).
  /// @param stroke_width  Line width in pixels (default 2).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::seql(0.0, 6.0, 25);
  /// auto plt = dm::Plot(x, dm::sin(x) + dm::rnorm(0, 0.1, 25));
  /// plt.lines(x, dm::sin(x), "sin(x)", dm::RGB{220, 38, 38}, 3.0);
  /// std::print("{}\n", plt.series()[1].stroke_width);  // 3
  /// @endcode
  ///
  /// @par References
  /// - R `lines()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/lines.html
  ///
  /// @see line(), lines(std::vector<double>, std::vector<double>, RGB, double), plot()
  RPlot& lines(std::vector<double> x,
               std::vector<double> y,
               std::string         label = "",
               RGB                 color = {220, 38, 38},
               double              stroke_width = 2.0);
  /// @brief Same as points(x, y, label, color, marker_size) with an empty label, so the
  ///        series stays out of the automatic legend.
  /// @param x  Horizontal coordinates.
  /// @param y  Vertical coordinates, paired with @p x.
  /// @param color  Marker color for every point.
  /// @param marker_size  Marker radius in pixels (default 4).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot();
  /// plt.points(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 9.0), dm::RGB{22, 163, 74}, 3.0);
  /// @endcode
  /// @see points()
  RPlot& points(std::vector<double> x, std::vector<double> y, RGB color, double marker_size = 4.0);
  /// @brief Unlabeled line shorthand: `lines(x, y, {r, g, b}, width)`.
  /// @param x  Horizontal coordinates.
  /// @param y  Vertical coordinates, paired with @p x.
  /// @param color  Line color.
  /// @param stroke_width  Line width in pixels (default 2).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot().lines(dm::c(0.0, 1.0, 2.0), dm::c(0.0, 1.0, 0.0), dm::RGB{0, 0, 0}, 1.0);
  /// std::print("{}\n", plt.series()[0].label.empty());  // true
  /// @endcode
  /// @see lines()
  RPlot& lines(std::vector<double> x, std::vector<double> y, RGB color, double stroke_width = 2.0);
  /// @brief Adds vertical bars centered at arbitrary x positions, the low-level form behind
  ///        barplot() and hist() (R's `rect()` from 0 to each height).
  ///
  /// Bar i spans x[i] +/- bar_width / 2 and rises from 0 to y[i] (negative heights point
  /// down); the y range always includes 0. Bars are drawn 85% opaque in SVG output. A nonempty
  /// @p label adds a row to the automatic legend.
  ///
  /// @param x  Bar centers, in data units.
  /// @param y  Bar heights, paired with @p x by position.
  /// @param label  Legend text; empty (the default) keeps it out of the legend.
  /// @param color  Bar color (default green `{22, 163, 74}`).
  /// @param bar_width  Bar width in data units (default 0.8).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot().bars(dm::c(1.0, 2.5, 4.0), dm::c(3.0, -1.0, 2.0), "net",
  ///                            dm::RGB{22, 163, 74}, 1.0);
  /// std::print("{}\n", plt.series()[0].bar_width);  // 1
  /// @endcode
  ///
  /// @par References
  /// - R `rect()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/rect.html
  ///
  /// @see barplot(), hist()
  RPlot& bars(std::vector<double> x,
              std::vector<double> y,
              std::string         label = "",
              RGB                 color = {22, 163, 74},
              double              bar_width = 0.8);
  /// @brief Adds one box-and-whisker from precomputed statistics at an arbitrary x position,
  ///        the low-level form behind boxplot() (R's `bxp()`).
  ///
  /// Draws a box from @p q1 to @p q3 with a thicker line at @p median, whiskers out to
  /// @p whisker_lo and @p whisker_hi with end caps, and one dot per outlier. The values are
  /// not checked for order or finiteness. In SVG the box is filled at 35% opacity.
  ///
  /// @param position  Box center, in x data units.
  /// @param whisker_lo  Lower whisker end.
  /// @param q1  Lower box edge (first quartile).
  /// @param median  Median line.
  /// @param q3  Upper box edge (third quartile).
  /// @param whisker_hi  Upper whisker end.
  /// @param outliers  Values drawn as individual points (default none).
  /// @param color  Box, whisker, and outlier color (default light blue `{96, 165, 250}`).
  /// @param width  Box width in data units (default 0.6).
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot().box(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, {8.0}).box(1.0, 2.0, 3.0, 3.5, 4.0, 6.0);
  /// std::print("{} {}\n", plt.series().size(), plt.series()[0].y.size());  // 2 6
  /// @endcode
  ///
  /// @par References
  /// - J. W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley.
  /// - R `bxp()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/bxp.html
  ///
  /// @see boxplot()
  RPlot& box(double              position,
             double              whisker_lo,
             double              q1,
             double              median,
             double              q3,
             double              whisker_hi,
             std::vector<double> outliers = {},
             RGB                 color = {96, 165, 250},
             double              width = 0.6);

  /// @brief Adds the reference line y = intercept + slope * x across the plotting area, like
  ///        R's `abline(a = intercept, b = slope)`.
  ///
  /// The line runs between the left and right edges of the visible x range and is drawn
  /// beneath every data layer. It does not change the axis ranges, and it is not clipped to the
  /// plotting area, so a steep line can extend into the margins. Stored as an ABLine.
  ///
  /// @param intercept  y value at x = 0.
  /// @param slope  Change in y per unit x.
  /// @param color  Line color (default opaque black).
  /// @param stroke_width  Line width in pixels (default 1).
  /// @return This plot, for chaining.
  ///
  /// @warning Only to_svg() and `.svg` saves draw reference lines. The raster outputs (`.png`,
  ///          `.pdf`, and show()) currently omit them.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::c(1.0, 2.0, 3.0, 4.0);
  /// auto plt = dm::Plot(x, dm::c(2.1, 3.9, 6.2, 7.8))
  ///                .abline(0.0, 2.0, dm::RGB{220, 38, 38})  // y = 2x
  ///                .abline_h(5.0)
  ///                .abline_v(2.5, dm::RGB{37, 99, 235}, 2.0);
  /// std::print("{}\n", plt.reference_lines().size());  // 3
  /// plt.save("ablines.svg");
  /// @endcode
  ///
  /// @par References
  /// - R `abline()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/abline.html
  ///
  /// @see abline_h(), abline_v(), qqline(), ABLine
  RPlot& abline(double intercept, double slope, RGB color = {0, 0, 0}, double stroke_width = 1.0);
  /// @brief Adds a horizontal reference line at y = @p y_value, like R's `abline(h = y_value)`.
  ///
  /// Same as `abline(y_value, 0.0, color, stroke_width)`: drawn beneath the data and not
  /// included in the axis ranges, so a value outside the visible y range is off the plot.
  ///
  /// @param y_value  Height of the line, in data units.
  /// @param color  Line color (default opaque black).
  /// @param stroke_width  Line width in pixels (default 1).
  /// @return This plot, for chaining.
  ///
  /// @warning Only to_svg() and `.svg` saves draw reference lines. The raster outputs (`.png`,
  ///          `.pdf`, and show()) currently omit them.
  ///
  /// @par Example
  /// See abline().
  ///
  /// @par References
  /// - R `abline()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/abline.html
  ///
  /// @see abline(), abline_v()
  RPlot& abline_h(double y_value, RGB color = {0, 0, 0}, double stroke_width = 1.0);
  /// @brief Adds a vertical reference line at x = @p x_value, like R's `abline(v = x_value)`.
  ///
  /// Spans the full height of the plotting area beneath the data. Unlike horizontal and
  /// sloped lines, it widens the automatic x range to include @p x_value.
  ///
  /// @param x_value  Horizontal position of the line, in data units.
  /// @param color  Line color (default opaque black).
  /// @param stroke_width  Line width in pixels (default 1).
  /// @return This plot, for chaining.
  ///
  /// @warning Only to_svg() and `.svg` saves draw reference lines. The raster outputs (`.png`,
  ///          `.pdf`, and show()) currently omit them.
  ///
  /// @par Example
  /// See abline().
  ///
  /// @par References
  /// - R `abline()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/abline.html
  ///
  /// @see abline(), abline_h()
  RPlot& abline_v(double x_value, RGB color = {0, 0, 0}, double stroke_width = 1.0);

  /// @brief Adds the reference line of a normal Q-Q plot, like R's `qqline(y)`: the line
  ///        through the first and third quartiles of @p data and of the standard normal.
  ///
  /// Pass the same data given to qqnorm(). The sample quartiles are R's default "type 7"
  /// quantiles, as in R's `qqline()` defaults (`probs = c(0.25, 0.75)`, `qtype = 7`). The
  /// slope is the sample IQR divided by the normal IQR (about 1.349), so for normal data it
  /// estimates the standard deviation and the intercept estimates the mean. Added through
  /// abline().
  ///
  /// @param data  The sample; at least two values. Taken by value, so the caller's data is unchanged.
  /// @param color  Line color (default red `{220, 38, 38}`).
  /// @param stroke_width  Line width in pixels (default 1.5).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If @p data has fewer than two values.
  ///
  /// @warning Only to_svg() and `.svg` saves draw reference lines. The raster outputs (`.png`,
  ///          `.pdf`, and show()) currently omit them.
  ///
  /// @par Example
  /// @code
  /// const std::vector<double> sample{3.0, 1.0, 2.0, 5.0, 4.0};
  /// auto plt = dm::Plot::qqnorm(sample).qqline(sample);
  /// const auto& line = plt.reference_lines()[0];
  /// std::print("{:.4f} {:.4f}\n", line.value, line.slope);  // 3.0000 1.4826
  /// @endcode
  ///
  /// @par References
  /// - R `qqline()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/qqnorm.html
  /// - R. J. Hyndman, Y. Fan (1996). "Sample Quantiles in Statistical Packages". The American
  ///   Statistician 50(4), 361-365. doi:10.1080/00031305.1996.10473566
  ///
  /// @see qqnorm(), abline()
  RPlot& qqline(std::vector<double> data, RGB color = {220, 38, 38}, double stroke_width = 1.5);

  /// @brief Adds manual legend rows, like R's `legend(legend = labels, col = colors)`.
  ///
  /// Each label is paired with the color at the same position and drawn as a swatch row in
  /// the legend box. Once any manual row exists, it replaces the automatic legend built from
  /// labeled series. Calling again appends more rows. Unlike dm::legend(), colors are not
  /// recycled here: the vectors must match in length. Empty vectors add nothing.
  ///
  /// @param labels  Row texts, in display order.
  /// @param colors  Swatch colors, one per label.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If @p labels and @p colors differ in length.
  ///
  /// @par Example
  /// @code
  /// const auto up = dm::RGB{37, 99, 235};
  /// const auto down = dm::RGB{220, 38, 38};
  /// auto plt = dm::Plot();
  /// plt.points(dm::c(1.0, 2.0), dm::c(1.0, 2.0), up)
  ///    .points(dm::c(1.0, 2.0), dm::c(2.0, 1.0), down)
  ///    .legend({"rising", "falling"}, {up, down});
  /// std::print("{}\n", plt.legend_entries().size());  // 2
  /// @endcode
  ///
  /// @par References
  /// - R `legend()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/legend.html
  ///
  /// @see dm::legend() for the recycling, pipe-friendly layer.
  RPlot& legend(std::vector<std::string> labels, std::vector<RGB> colors);
  /// @brief Draws @p label centered on the data point (@p x, @p y), like R's
  ///        `text(x, y, labels = label)`.
  ///
  /// The text is centered both horizontally and vertically on the point, like R's default
  /// `adj`. The point counts toward the automatic axis ranges, but the text's extent does not,
  /// so long labels near the edge can overflow. Text layers never appear in the legend, and
  /// set_col() recolors them along with the data.
  ///
  /// @param x  Horizontal position, in data units.
  /// @param y  Vertical position, in data units.
  /// @param label  The text to draw (XML-escaped in SVG output).
  /// @param color  Text color (default near-black `{17, 24, 39}`).
  /// @param font_size  Font size in pixels (default 14).
  /// @return This plot, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 2.0))
  ///                .text(2.0, 4.4, "peak", dm::RGB{220, 38, 38}, 16.0);
  /// std::print("{} {}\n", plt.series()[1].label, plt.series()[1].marker_size);  // peak 16
  /// @endcode
  ///
  /// @par References
  /// - R `text()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/text.html
  ///
  /// @see pie(), legend()
  RPlot& text(double x, double y, std::string label, RGB color = {17, 24, 39}, double font_size = 14.0);
  /// @brief Draws the closed polygon through the vertices (x[i], y[i]), like R's
  ///        `polygon(x, y, col = color)`.
  ///
  /// The last vertex joins the first automatically. Filled polygons (the default) are painted
  /// with @p color (plus a 1-pixel outline in SVG; the raster fill uses the even-odd rule);
  /// unfilled ones draw a 2-pixel outline only (R's default is unfilled). Translucent colors work well
  /// for shaded regions such as confidence bands. Vertices count toward the axis ranges.
  ///
  /// @param x  Vertex x coordinates, in order around the polygon.
  /// @param y  Vertex y coordinates, paired with @p x.
  /// @param color  Fill (and outline) color (default blue `{37, 99, 235}`).
  /// @param filled  True (the default) fills; false draws the outline only.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the coordinates are empty or have different lengths.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::Plot()
  ///                .polygon({0.0, 1.0, 0.5}, {0.0, 0.0, 1.0}, dm::RGBA{37, 99, 235, 0.4})
  ///                .polygon({0.2, 0.8, 0.8, 0.2}, {0.1, 0.1, 0.4, 0.4}, dm::RGB{0, 0, 0}, false);
  /// std::print("{}\n", plt.series()[1].filled);  // false
  /// @endcode
  ///
  /// @par References
  /// - R `polygon()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/polygon.html
  ///
  /// @see pie(), segments()
  RPlot& polygon(std::vector<double> x,
                 std::vector<double> y,
                 RGB                 color = {37, 99, 235},
                 bool                filled = true);
  /// @brief Draws a grid of coloured rectangles, one per value of @p z, like R's `image(x, y, z)`.
  ///
  /// Cell (i, j) is centred on (x[i], y[j]) and reaches halfway to its neighbours (a half step
  /// beyond the outermost centres). Its colour is the palette entry for z[i][j] on a linear
  /// scale from @p z_min to @p z_max; NaN cells are left empty. Each cell is added as a filled
  /// polygon, so the grid should have at most tens of thousands of cells.
  ///
  /// @param x  Cell centres along x, increasing (at least 2).
  /// @param y  Cell centres along y, increasing (at least 2).
  /// @param z  Values, z[i][j] at (x[i], y[j]); z.size() == x.size() and each row has y.size() entries.
  /// @param palette  Colours from low to high (default terrain_colors(12)).
  /// @param z_min  Value mapped to the first colour; NaN (the default) uses the smallest z.
  /// @param z_max  Value mapped to the last colour; NaN (the default) uses the largest z.
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the sizes do not match, x or y is not increasing, or the
  ///         palette is empty.
  ///
  /// @par Example
  /// @code
  /// auto plt = dm::graphics::RPlot::create().image({0.0, 1.0}, {0.0, 1.0, 2.0}, {{1, 2, 3}, {4, 5, 6}});
  /// std::print("{}\n", plt.series().size());  // 6
  /// @endcode
  ///
  /// @par References
  /// - R `image()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/image.html
  ///
  /// @see terrain_colors(), polygon()
  RPlot& image(const std::vector<double>&              x,
               const std::vector<double>&              y,
               const std::vector<std::vector<double>>& z,
               std::vector<RGB>                        palette = {},
               double                                  z_min = std::numeric_limits<double>::quiet_NaN(),
               double                                  z_max = std::numeric_limits<double>::quiet_NaN());
  /// @brief Draws one straight segment from (x0[i], y0[i]) to (x1[i], y1[i]) for each i, like
  ///        R's `segments(x0, y0, x1, y1)`.
  ///
  /// All segments share @p color and @p stroke_width and form one Segment layer; their end
  /// points count toward the axis ranges. Useful for error bars, residual sticks, and
  /// dendrograms.
  ///
  /// @param x0  Start x coordinates.
  /// @param y0  Start y coordinates.
  /// @param x1  End x coordinates.
  /// @param y1  End y coordinates.
  /// @param color  Segment color (default opaque black).
  /// @param stroke_width  Segment width in pixels (default 1).
  /// @return This plot, for chaining.
  /// @throws std::invalid_argument  If the four vectors differ in length or are empty.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::c(1.0, 2.0, 3.0);
  /// const auto y = dm::c(2.0, 3.5, 3.0);
  /// auto plt = dm::Plot(x, y).segments(x, y - 0.5, x, y + 0.5);  // +/- 0.5 error bars
  /// std::print("{}\n", plt.series()[1].x.size());  // 6 (start, end, start, end, ...)
  /// @endcode
  ///
  /// @par References
  /// - R `segments()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/segments.html
  ///
  /// @see abline(), polygon()
  RPlot& segments(std::vector<double> x0,
                  std::vector<double> y0,
                  std::vector<double> x1,
                  std::vector<double> y1,
                  RGB                 color = {0, 0, 0},
                  double              stroke_width = 1.0);
};

/// @brief Tiles independently built plots into one multi-panel figure, like R's
///        `par(mfrow = c(rows, cols))` followed by one plot per panel.
///
/// Create it with create(rows, cols), then add() up to rows * cols panels; they fill the grid
/// row by row. Each panel is copied (and its SVG rendered) at add() time, so later changes to
/// the original plot do not show and panels need not outlive the layout. Each cell is
/// `width / cols` by `(height - band) / rows` pixels, where band is title_band_px when a
/// figure title is set and 0 otherwise; every panel is scaled from its own canvas size into its
/// cell. The figure background is white. Unused cells stay empty.
///
/// @par Example
/// @code
/// const auto x = dm::seql(0.0, 6.0, 30);
/// auto fig = dm::graphics::RLayout::create(1, 2)
///                .size(1200, 500)
///                .title("Two views")
///                .add(dm::Plot(x, dm::sin(x)).set_title("sin"))
///                .add(dm::Plot(x, dm::cos(x)).set_title("cos"));
/// std::print("{} panels in {}x{}\n", fig.panel_count(), fig.rows(), fig.cols());
/// // 2 panels in 1x2
/// fig.save("two_views.svg");
/// @endcode
///
/// @par References
/// - R `par()` (`mfrow`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
///
/// @see RPlot, RPlot::margins(), dm::Splom
class RLayout final {
 public:
  /// @brief Height in pixels of the heading band reserved above the panels when a title is set.
  static constexpr std::size_t title_band_px = 40;
  /// @brief Smallest canvas side fit_panels() gives a panel; smaller cells get a larger canvas
  ///        with the cell's aspect ratio, scaled down when drawn, so the margins still fit.
  static constexpr std::size_t min_panel_px = 320;

  /// @brief Creates an empty layout with @p rows by @p cols cells, like R's
  ///        `par(mfrow = c(rows, cols))`.
  ///
  /// The figure starts at 960 by 640 pixels with no title.
  ///
  /// @param rows  Number of panel rows; must be positive.
  /// @param cols  Number of panel columns; must be positive.
  /// @return The empty layout.
  /// @throws std::invalid_argument  If @p rows or @p cols is 0.
  ///
  /// @par Example
  /// See RLayout.
  ///
  /// @par References
  /// - R `par()` (`mfrow`): https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
  ///
  /// @see add()
  static RLayout create(std::size_t rows, std::size_t cols);

  /// @brief Copies @p panel into the next free cell, filling rows left to right, top to bottom.
  ///
  /// The panel's SVG is rendered immediately and stored with the copy, so rendering errors
  /// surface here and later edits to the original plot have no effect on the layout.
  ///
  /// @param panel  The plot to add; any RPlot (or other Plot) converts.
  /// @return This layout, for chaining.
  /// @throws std::invalid_argument  If all rows * cols cells are already filled.
  /// @throws std::runtime_error  If @p panel has no data series or its canvas is smaller than
  ///         its margins.
  ///
  /// @par Example
  /// @code
  /// auto fig = dm::graphics::RLayout::create(1, 1).add(dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)));
  /// try {
  ///   fig.add(dm::Plot(dm::c(1.0), dm::c(1.0)));
  /// } catch (const std::invalid_argument& e) {
  ///   std::print("{}\n", e.what());  // RLayout::add: layout is already full
  /// }
  /// @endcode
  /// @see panel(), panel_count()
  RLayout& add(const Plot& panel);
  /// @brief Sets the size of the whole figure in pixels (default 960 by 640).
  ///
  /// Panels keep their own canvas sizes and are scaled into their cells, so text in a panel
  /// shrinks or grows with it; give panels a size close to their cell for readable labels.
  ///
  /// @param width  Figure width in pixels.
  /// @param height  Figure height in pixels, including the title band.
  /// @return This layout, for chaining.
  ///
  /// @par Example
  /// See RLayout.
  /// @see title()
  RLayout& size(std::size_t width, std::size_t height);
  /// @brief Sets an optional figure-level heading drawn in a band above the panels, like R's
  ///        `mtext(main, outer = TRUE)`; an empty string (the default) reserves no band.
  /// @param value  Heading text; the band is title_band_px pixels tall.
  /// @return This layout, for chaining.
  ///
  /// @par Example
  /// See RLayout.
  ///
  /// @par References
  /// - R `mtext()`: https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/mtext.html
  ///
  /// @see title_text()
  RLayout& title(std::string value);

  /// @brief Returns the number of panel rows given to create().
  /// @return Row count (at least 1).
  std::size_t rows() const noexcept { return rows_; }
  /// @brief Returns the number of panel columns given to create().
  /// @return Column count (at least 1).
  std::size_t cols() const noexcept { return cols_; }
  /// @brief Returns the figure width in pixels.
  /// @return Width set by size() (default 960).
  std::size_t width() const noexcept { return width_; }
  /// @brief Returns the figure height in pixels.
  /// @return Height set by size() (default 640).
  std::size_t height() const noexcept { return height_; }
  /// @brief Returns how many panels have been added.
  /// @return Panel count, at most rows() * cols().
  std::size_t panel_count() const noexcept { return panels_.size(); }
  /// @brief Borrows the figure-level heading for the lifetime of this layout.
  /// @return The title; empty (the default) when none is drawn.
  const std::string& title_text() const noexcept { return title_; }
  /// @brief Borrows the copy of the @p index-th added panel.
  /// @param index  Zero-based panel index, in the order panels were added.
  /// @return The stored copy, valid until the next add() or the layout's destruction.
  /// @throws std::out_of_range If @p index is not below panel_count().
  ///
  /// @par Example
  /// @code
  /// auto fig = dm::graphics::RLayout::create(1, 2).add(dm::Plot(dm::c(1.0), dm::c(1.0)).set_title("A"));
  /// std::print("{}\n", fig.panel(0).title_text());  // A
  /// @endcode
  const Plot& panel(std::size_t index) const;

  /// @brief Renders the multi-panel figure as a standalone SVG document and returns it as a string.
  ///
  /// Each panel becomes a nested `<svg>` scaled into its cell. A layout with no panels renders
  /// as an empty white figure. The layout is not modified.
  ///
  /// @return The complete SVG text.
  ///
  /// @par Example
  /// @code
  /// auto fig = dm::graphics::RLayout::create(2, 1).add(dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)));
  /// std::print("{}\n", fig.to_svg().starts_with("<?xml"));  // true
  /// @endcode
  /// @see save()
  std::string to_svg() const;
  /// @brief Saves the figure, choosing the format from the extension: `.svg` (or none) writes
  ///        SVG, `.png` / `.pdf` write the same tiled raster the X11 viewer shows.
  ///
  /// The extension is case-sensitive and an existing file is overwritten. Raster panels are
  /// re-rendered from the stored copies and scaled into their cells.
  ///
  /// @param path  Destination file.
  /// @throws std::invalid_argument For any other extension.
  /// @throws std::runtime_error  If the file cannot be written.
  ///
  /// @par Example
  /// See RLayout.
  /// @see save_svg(), to_svg(), show()
  void save(const std::string& path) const;
  /// @brief Writes to_svg() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be opened for writing.
  /// @see save()
  void save_svg(const std::string& path) const;
  /// @brief Renders the tiled figure as PNG image bytes (the raster save() writes for `.png`).
  /// @return The complete PNG file contents.
  /// @throws std::runtime_error  If a panel cannot be rendered.
  /// @see save_png(), to_svg()
  std::vector<std::uint8_t> to_png() const;
  /// @brief Renders the tiled figure as a one-page PDF holding the raster image.
  /// @return The complete PDF file contents.
  /// @throws std::runtime_error  If a panel cannot be rendered.
  /// @see save_pdf(), to_png()
  std::vector<std::uint8_t> to_pdf() const;
  /// @brief Writes to_png() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be written or a panel cannot be rendered.
  void save_png(const std::string& path) const;
  /// @brief Writes to_pdf() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be written or a panel cannot be rendered.
  void save_pdf(const std::string& path) const;
  /// @brief Resizes every stored panel's canvas to its cell, so text keeps its pixel size
  ///        instead of being scaled with the panel.
  ///
  /// Call it after size() and title() (the title band changes the cell height). Panels added
  /// later keep their own size until fit_panels() is called again. A cell whose shorter side is
  /// under min_panel_px gets a canvas scaled up to that side (same aspect ratio), so its text is
  /// drawn proportionally smaller instead of the margins overflowing the panel.
  /// @return This layout, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto fig = dm::graphics::RLayout::create(1, 2)
  ///                .add(dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0)))
  ///                .add(dm::Plot(dm::c(1.0, 2.0), dm::c(1.0, 2.0)))
  ///                .size(1000, 400)
  ///                .fit_panels();
  /// std::print("{}x{}\n", fig.panel(0).width(), fig.panel(0).height());  // 500x400
  /// @endcode
  /// @see size()
  RLayout& fit_panels();
  /// @brief Opens the tiled figure in the interactive X11 viewer; the toolbar zoom/pan buttons
  ///        act on every panel's data bounds at once, keyboard zoom/pan scales the whole image.
  ///
  /// Blocks until the window is closed (q or Escape). Use save() in headless code.
  ///
  /// @param title_hint  Window title; empty (the default) uses title_text().
  /// @throws std::runtime_error If X11 support was not enabled at build time or no display is
  ///         available.
  /// @throws std::invalid_argument  If no panel was added.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::seql(0.0, 6.0, 30);
  /// dm::graphics::RLayout::create(1, 2)
  ///     .add(dm::Plot(x, dm::sin(x)))
  ///     .add(dm::Plot(x, dm::cos(x)))
  ///     .show("Trig");
  /// @endcode
  /// @see save(), Plot::show()
  void show(const std::string& title_hint = "") const;
  /// @brief Same as show(); kept as an alias for older code.
  /// @param title_hint  Window title; empty (the default) uses title_text().
  /// @throws std::runtime_error  As for show().
  /// @throws std::invalid_argument  If no panel was added.
  /// @see show()
  void view(const std::string& title_hint = "") const;

 private:
  /// @brief One stored panel: the plot copy and its pre-rendered SVG body.
  struct Panel {
    Plot        plot;         ///< owned copy, so panels need not outlive the layout
    std::string svg_fragment; ///< rendered eagerly in add()
  };

  std::size_t rows_{1};         ///< Panel rows.
  std::size_t cols_{1};         ///< Panel columns.
  std::size_t width_{960};      ///< Figure width in pixels.
  std::size_t height_{640};     ///< Figure height in pixels, including the title band.
  std::string title_;           ///< Figure heading; empty reserves no band.
  std::vector<Panel> panels_;   ///< Added panels, in row-major cell order.
};

/// @brief @p n colours from green through yellow and brown to near-white, as R's `terrain.colors(n)`.
/// @param n  Number of colours (at least 1).
/// @return The palette, low to high.
/// @throws std::invalid_argument  If @p n is 0.
std::vector<RGB> terrain_colors(std::size_t n);

/// @brief @p n colours from dark blue through white to dark red, a diverging palette for signed
///        values such as correlations (ColorBrewer's RdBu, reversed so that low is blue).
///
/// Interpolates linearly between the seven RdBu anchor colours; an odd @p n puts pure
/// near-white (247, 247, 247) in the middle, so with `z_min = -z_max` in RPlot::image() zero
/// maps to white.
///
/// @param n  Number of colours (at least 1; 1 gives the white midpoint).
/// @return The palette, low (blue) to high (red).
/// @throws std::invalid_argument  If @p n is 0.
///
/// @par Example
/// @code
/// const auto pal = dm::graphics::diverging_colors(3);
/// std::print("{} {} {}\n", pal[0].r, pal[1].g, pal[2].r);  // 33 247 178
/// @endcode
///
/// @par References
/// - C. Brewer, ColorBrewer 2.0 (RdBu scheme): https://colorbrewer2.org
///
/// @see terrain_colors(), RPlot::image()
std::vector<RGB> diverging_colors(std::size_t n);

/// @brief @p n colours of the viridis scale, from dark purple through blue and green to yellow,
///        matching R's `viridisLite::viridis(n)` at n = 10.
///
/// Viridis is perceptually uniform, so equal steps in value look like equal steps in colour,
/// and it stays readable in greyscale and for the common forms of colour blindness. Colours are
/// interpolated linearly between viridis(10)'s ten anchors.
///
/// @param n  Number of colours (at least 1).
/// @return The palette, low (purple) to high (yellow).
/// @throws std::invalid_argument  If @p n is 0.
///
/// @par Example
/// @code
/// const auto pal = dm::graphics::viridis_colors(10);
/// std::print("{} {} {}\n", pal[0].r, pal[5].g, pal[9].b);  // 68 158 37
/// @endcode
///
/// @par References
/// - S. van der Walt and N. Smith (2015). "mpl colourmaps" (viridis): https://bids.github.io/colormap/
/// - viridisLite: https://sjmgarnier.github.io/viridisLite/
///
/// @see value_colors(), terrain_colors()
std::vector<RGB> viridis_colors(std::size_t n);

/// @brief One colour per value on a continuous scale, for colouring scatter points by a
///        numeric variable (ggplot2's `scale_colour_viridis_c()`).
///
/// Each value is mapped linearly from [@p z_min, @p z_max] to the palette and rounded to the
/// nearest entry; values outside are clamped, NaN values get grey (190, 190, 190), and a zero
/// range maps everything to the middle colour. Pass the result to RPlot::points_colored().
///
/// @param values  Values to colour.
/// @param palette  Colours from low to high (default viridis_colors(256)).
/// @param z_min  Value of the first colour; NaN (the default) uses the smallest value.
/// @param z_max  Value of the last colour; NaN (the default) uses the largest value.
/// @return One colour per value.
///
/// @par Example
/// @code
/// const auto cols = dm::graphics::value_colors({0.0, 0.5, 1.0}, dm::graphics::viridis_colors(3));
/// std::print("{} {} {}\n", cols[0].r, cols[1].r, cols[2].r);  // 68 35 253
/// @endcode
///
/// @see viridis_colors(), RPlot::points_colored()
std::vector<RGB> value_colors(const std::vector<double>& values, std::vector<RGB> palette = {},
                              double z_min = std::numeric_limits<double>::quiet_NaN(),
                              double z_max = std::numeric_limits<double>::quiet_NaN());

} // namespace datamunge::graphics
