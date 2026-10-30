#pragma once

/// @file figure.hpp
/// @brief graphics::Figure, the one return type of the free dm::plot() functions, and the
///        generic dm::plot(Figure) overload.
///
/// dm::plot(x) is overloaded per object type (an LDA fit, a plot, a layout, ...) and always
/// returns a Figure, so the caller can title, size, and export the result without knowing
/// whether it is one panel or several:
///
/// @code
/// const dm::stats::LDA petals(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
/// dm::plot(petals).title("LDA on the petals").size(1200, 520).save_svg("petals.svg");
/// @endcode
///
/// To support a new type, add an overload `graphics::Figure plot(const T&)` in namespace
/// datamunge that builds an RPlot (or an RLayout of several) and returns it; both convert to
/// Figure implicitly.

#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace datamunge::graphics {

class Splom;
class GGPlot;

/// @brief A finished figure of any kind (a single plot or a grid of panels) with one output
///        interface: title(), size(), to_svg()/to_png()/to_pdf(), save*(), and show().
///
/// Every plot type in the library converts to a Figure implicitly: RPlot (dm::Plot), its
/// subclasses such as Histogram, RLayout, Splom, and GGPlot. A Figure is an owned copy, so it
/// does not refer back to the object it came from. For a single plot, title() and size() act on
/// that plot; for a grid they act on the whole figure (title() draws a heading band above the
/// panels), and every panel is resized to its cell so text keeps its pixel size.
///
/// @par Example
/// @code
/// const auto x = dm::seql(0.0, 6.0, 30);
/// dm::graphics::Figure fig = dm::graphics::RLayout::create(1, 2)
///                                .add(dm::Plot(x, dm::sin(x)).set_title("sin"))
///                                .add(dm::Plot(x, dm::cos(x)).set_title("cos"));
/// fig.title("Two views").size(1000, 420).save_png("trig.png");
/// std::print("{} panels, {}x{}\n", fig.panel_count(), fig.width(), fig.height());
/// // 2 panels, 1000x420
/// @endcode
/// @see dm::plot(), RPlot, RLayout
class Figure {
 public:
  /// @brief Wraps a copy of a single plot (an RPlot, Histogram, ... converts).
  /// @param plot  The plot; copied (sliced to its Plot base, like RLayout::add()).
  ///
  /// @par Example
  /// See Figure.
  Figure(Plot plot);  // NOLINT(google-explicit-constructor): conversion is the point
  /// @brief Wraps a copy of a multi-panel layout; panel sizes are left as they are until size()
  ///        or title() is called.
  /// @param layout  The layout to copy.
  ///
  /// @par Example
  /// See Figure.
  Figure(RLayout layout);  // NOLINT(google-explicit-constructor)
  /// @brief Wraps the panel grid a pairs plot renders to (Splom::layout()).
  /// @param splom  The pairs plot.
  /// @throws  Whatever Splom::layout() throws for an unusable column selection.
  ///
  /// @par Example
  /// See Figure.
  Figure(const Splom& splom);  // NOLINT(google-explicit-constructor)
  /// @brief Wraps a ggplot-style plot: one panel, or the facet grid when facet_wrap() was used.
  /// @param plot  The grammar-of-graphics plot.
  /// @throws std::runtime_error  If the plot cannot be built (for example an empty facet column).
  ///
  /// @par Example
  /// See Figure.
  Figure(const GGPlot& plot);  // NOLINT(google-explicit-constructor)

  /// @brief Sets the heading: the plot's title, or a band above a grid's panels.
  /// @param value  Heading text; empty removes it.
  /// @return This figure, for chaining.
  ///
  /// @par Example
  /// See Figure.
  Figure& title(std::string value);
  /// @brief Sets the size of the whole figure in pixels.
  ///
  /// For a grid, each panel is resized to its cell (see RLayout::fit_panels()), so labels stay
  /// readable rather than being scaled down with the panel. Any size works: a panel or plot
  /// smaller than RLayout::min_panel_px on a side is drawn on a larger canvas and scaled down.
  /// @param width  Width in pixels.
  /// @param height  Height in pixels, including any title band.
  /// @return This figure, for chaining.
  ///
  /// @par Example
  /// See Figure.
  Figure& size(std::size_t width, std::size_t height);

  /// @brief Returns the figure width in pixels.
  /// @return The current width.
  std::size_t width() const noexcept;
  /// @brief Returns the figure height in pixels.
  /// @return The current height.
  std::size_t height() const noexcept;
  /// @brief Borrows the heading set by title() (or by the wrapped plot or layout).
  /// @return The title; empty when there is none.
  const std::string& title_text() const noexcept;
  /// @brief Returns whether the figure is a grid of panels rather than a single plot.
  /// @return True for a figure built from an RLayout, Splom, or faceted GGPlot.
  bool is_grid() const noexcept;
  /// @brief Returns the number of panels: 1 for a single plot.
  /// @return Panel count.
  std::size_t panel_count() const noexcept;
  /// @brief Borrows one panel.
  /// @param index  Zero-based panel index, in row-major order (0 for a single plot).
  /// @return The panel, valid until the figure is modified or destroyed.
  /// @throws std::out_of_range  If @p index is not below panel_count().
  ///
  /// @par Example
  /// See Figure.
  const Plot& panel(std::size_t index) const;

  /// @brief Renders the figure as a standalone SVG document.
  /// @return The SVG text.
  /// @throws std::runtime_error  If a panel has no data or is smaller than its margins.
  ///
  /// @par Example
  /// See Figure.
  std::string to_svg() const;
  /// @brief Renders the figure as PNG image bytes.
  /// @return The complete PNG file contents.
  /// @throws std::runtime_error  As for to_svg().
  ///
  /// @par Example
  /// See Figure.
  std::vector<std::uint8_t> to_png() const;
  /// @brief Renders the figure as a one-page PDF holding the raster image.
  /// @return The complete PDF file contents.
  /// @throws std::runtime_error  As for to_svg().
  ///
  /// @par Example
  /// See Figure.
  std::vector<std::uint8_t> to_pdf() const;

  /// @brief Saves the figure, choosing the format from the extension: `.svg` (or none), `.png`,
  ///        or `.pdf`.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::invalid_argument  For any other extension.
  /// @throws std::runtime_error  If the file cannot be written or rendering fails.
  ///
  /// @par Example
  /// See Figure.
  void save(const std::string& path) const;
  /// @brief Writes to_svg() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be written or rendering fails.
  ///
  /// @par Example
  /// See Figure.
  void save_svg(const std::string& path) const;
  /// @brief Writes to_png() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be written or rendering fails.
  ///
  /// @par Example
  /// See Figure.
  void save_png(const std::string& path) const;
  /// @brief Writes to_pdf() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be written or rendering fails.
  ///
  /// @par Example
  /// See Figure.
  void save_pdf(const std::string& path) const;
  /// @brief Opens the figure in the interactive X11 viewer and blocks until it is closed.
  /// @param title_hint  Window title; empty uses title_text().
  /// @throws std::runtime_error  If X11 support is missing or no display is available.
  ///
  /// @par Example
  /// @code
  /// dm::plot(dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0))).show("Tiny");
  /// @endcode
  /// @see save(), Plot::show(), RLayout::show()
  void show(const std::string& title_hint = "") const;

 private:
  std::variant<Plot, RLayout> content_;  ///< The single plot or the panel grid.
  bool single_{false};  ///< True when content_ is a 1 x 1 grid standing in for a small single plot.
};

}  // namespace datamunge::graphics

namespace datamunge {

/// @brief Returns @p figure unchanged, so dm::plot() accepts any plot type as well as the
///        objects that have their own overload (any RPlot, RLayout, Splom, or GGPlot converts).
/// @param figure  The figure (or a plot converting to one).
/// @return The same figure.
///
/// @par Example
/// @code
/// dm::plot(dm::Plot(dm::c(1.0, 2.0), dm::c(2.0, 1.0))).size(400, 300).save_svg("tiny.svg");
/// @endcode
/// @see graphics::Figure, plot(const stats::LDA&)
inline graphics::Figure plot(graphics::Figure figure) { return figure; }

}  // namespace datamunge
