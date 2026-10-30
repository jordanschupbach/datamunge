#pragma once

/// @file histogram.hpp
/// @brief Constructor-based histograms using the canonical RPlot binning and renderers.

#include <datamunge/plot/plot.hpp>

#include <string>
#include <utility>
#include <vector>

namespace datamunge {

/// @brief An owned histogram plot with equal-width bins and chainable heading setters,
///        like R's `hist(x, breaks = bins)`.
///
/// Supports the same rendering, export, and plot layers as RPlot (show(), save(), to_svg(),
/// `| dm::legend()`, ...). Bins span [min, max] of the data in equal widths; bin intervals
/// include their left edge, and the maximum value is included in the final bin. Bars show
/// counts. A new histogram is titled "Histogram" with axis labels "x" and "Frequency".
///
/// Every chainable setter and layer verb inherited from RPlot (set_col(), size(), abline(),
/// ...) is retyped to return `Histogram&`, so a whole chain can be stored as a Histogram.
///
/// @par Example
/// @code
/// const auto iris = dm::datasets::iris();
/// dm::Histogram hist = dm::Histogram(iris.get_column("Sepal.Length").values())
///                          .set_title("Iris Sepal Length Distribution")
///                          .set_xlab("Sepal Length");
/// hist.show();
/// @endcode
/// @see dm::Plot, dm::Plot::hist()
class Histogram final : public graphics::RPlot {
 public:
  /// @brief Bins a copy of finite observations; defaults to ten bins and a frequency axis.
  ///
  /// Bin width is `(max - min) / bins`; constant observations use a unit-wide range starting
  /// at their value. The data are copied (or moved from an rvalue) only for binning; the plot
  /// keeps bin centers and counts, not the raw observations.
  ///
  /// @param data  Finite observations; NaN and infinities are rejected, not dropped.
  /// @param bins  Number of equal-width bins (default 10).
  /// @param label  Series label; nonempty adds a row to the automatic legend.
  /// @param color  Bar fill color (default light blue `{96, 165, 250}`).
  /// @throws std::invalid_argument If data is empty or non-finite, bins is zero, or the bin
  ///         range is not a finite positive width.
  ///
  /// @par Example
  /// @code
  /// const auto x = dm::rnorm(0, 1, 500);
  /// auto hist = dm::Histogram(x, 25, "N(0, 1)", dm::RGBA{220, 38, 38, 0.6});
  /// std::print("{} {}\n", hist.series()[0].x.size(), hist.y_label_text());  // 25 Frequency
  /// hist.save("normal_hist.png");
  /// @endcode
  /// @see dm::Plot::hist()
  explicit Histogram(std::vector<double> data, std::size_t bins = 10,
                     std::string label = "", graphics::RGB color = {96, 165, 250})
      : RPlot(RPlot::hist(std::move(data), bins, std::move(label), color)) {}

  /// @brief Owns the main heading and returns this histogram; an empty string clears it.
  /// @param value  Heading text (default "Histogram").
  /// @return This histogram, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto hist = dm::Histogram({1.0, 2.0, 2.0, 3.0, 3.0, 3.0}, 3)
  ///                 .set_title("Counts")
  ///                 .set_subtitle("Toy data")
  ///                 .set_ylab("n");
  /// std::print("{} / {}\n", hist.title_text(), hist.y_label_text());  // Counts / n
  /// @endcode
  /// @see dm::Plot::set_title()
  Histogram& set_title(std::string value) {
    RPlot::set_title(std::move(value));
    return *this;
  }
  /// @brief Owns the subtitle and returns this histogram; an empty string clears it.
  /// @param value  Subtitle text drawn under the title (none by default).
  /// @return This histogram, for chaining.
  /// @see set_title(), dm::Plot::set_subtitle()
  Histogram& set_subtitle(std::string value) {
    RPlot::set_subtitle(std::move(value));
    return *this;
  }
  /// @brief Owns the horizontal-axis label and returns this histogram; an empty string clears it.
  /// @param value  Axis label (default "x"); usually the measured variable's name.
  /// @return This histogram, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto hist = dm::Histogram(dm::rnorm(10, 2, 200)).set_xlab("Length (cm)");
  /// @endcode
  /// @see dm::Plot::set_xlab()
  Histogram& set_xlab(std::string value) {
    RPlot::set_xlab(std::move(value));
    return *this;
  }
  /// @brief Owns the vertical-axis label and returns this histogram; an empty string clears it.
  /// @param value  Axis label (default "Frequency").
  /// @return This histogram, for chaining.
  /// @see set_xlab(), dm::Plot::set_ylab()
  Histogram& set_ylab(std::string value) {
    RPlot::set_ylab(std::move(value));
    return *this;
  }

  /// @name Inherited setters and layer verbs, retyped for chaining
  /// Each forwards to the RPlot member of the same name and returns this histogram, so
  /// `dm::Histogram(x).set_title("t").set_col("red")` still has type Histogram&.
  /// @{

  /// @brief Same as RPlot::with(), returning this histogram so chains keep the Histogram type.
  /// @tparam Options An option group exposing apply(RPlot&), such as dm::PlotArgs.
  template <typename Options>
  Histogram& with(const Options& options) {
    RPlot::with(options);
    return *this;
  }
  /// @brief Same as RPlot::size(), returning this histogram so chains keep the Histogram type.
  Histogram& size(std::size_t width, std::size_t height) {
    RPlot::size(width, height);
    return *this;
  }
  /// @brief Same as RPlot::title(), returning this histogram so chains keep the Histogram type.
  Histogram& title(std::string value) {
    RPlot::title(std::move(value));
    return *this;
  }
  /// @brief Same as RPlot::subtitle(), returning this histogram so chains keep the Histogram type.
  Histogram& subtitle(std::string value) {
    RPlot::subtitle(std::move(value));
    return *this;
  }
  /// @brief Same as RPlot::x_label(), returning this histogram so chains keep the Histogram type.
  Histogram& x_label(std::string value) {
    RPlot::x_label(std::move(value));
    return *this;
  }
  /// @brief Same as RPlot::y_label(), returning this histogram so chains keep the Histogram type.
  Histogram& y_label(std::string value) {
    RPlot::y_label(std::move(value));
    return *this;
  }
  /// @brief Same as RPlot::xlab(), returning this histogram so chains keep the Histogram type.
  Histogram& xlab(std::string value) {
    RPlot::xlab(std::move(value));
    return *this;
  }
  /// @brief Same as RPlot::ylab(), returning this histogram so chains keep the Histogram type.
  Histogram& ylab(std::string value) {
    RPlot::ylab(std::move(value));
    return *this;
  }
  /// @brief Same as RPlot::set_col(), returning this histogram so chains keep the Histogram type.
  Histogram& set_col(graphics::RGB color) {
    RPlot::set_col(color);
    return *this;
  }
  /// @brief Same as RPlot::set_col(), returning this histogram so chains keep the Histogram type.
  Histogram& set_col(std::string color) {
    RPlot::set_col(std::move(color));
    return *this;
  }
  /// @brief Same as RPlot::background(), returning this histogram so chains keep the Histogram type.
  Histogram& background(graphics::RGB color) {
    RPlot::background(color);
    return *this;
  }
  /// @brief Same as RPlot::axis_color(), returning this histogram so chains keep the Histogram type.
  Histogram& axis_color(graphics::RGB color) {
    RPlot::axis_color(color);
    return *this;
  }
  /// @brief Same as RPlot::grid_color(), returning this histogram so chains keep the Histogram type.
  Histogram& grid_color(graphics::RGB color) {
    RPlot::grid_color(color);
    return *this;
  }
  /// @brief Same as RPlot::show_grid(), returning this histogram so chains keep the Histogram type.
  Histogram& show_grid(bool enabled = true) {
    RPlot::show_grid(enabled);
    return *this;
  }
  /// @brief Same as RPlot::x_limits(), returning this histogram so chains keep the Histogram type.
  Histogram& x_limits(double min_x, double max_x) {
    RPlot::x_limits(min_x, max_x);
    return *this;
  }
  /// @brief Same as RPlot::y_limits(), returning this histogram so chains keep the Histogram type.
  Histogram& y_limits(double min_y, double max_y) {
    RPlot::y_limits(min_y, max_y);
    return *this;
  }
  /// @brief Same as RPlot::hide_axes(), returning this histogram so chains keep the Histogram type.
  Histogram& hide_axes(bool enabled = true) {
    RPlot::hide_axes(enabled);
    return *this;
  }
  /// @brief Same as RPlot::x_tick_labels(), returning this histogram so chains keep the Histogram type.
  Histogram& x_tick_labels(std::vector<std::string> labels) {
    RPlot::x_tick_labels(std::move(labels));
    return *this;
  }
  /// @brief Same as RPlot::x_ticks(), returning this histogram so chains keep the Histogram type.
  Histogram& x_ticks(std::vector<double> positions, std::vector<std::string> labels) {
    RPlot::x_ticks(std::move(positions), std::move(labels));
    return *this;
  }
  /// @brief Same as RPlot::y_tick_labels(), returning this histogram so chains keep the Histogram type.
  Histogram& y_tick_labels(std::vector<std::string> labels) {
    RPlot::y_tick_labels(std::move(labels));
    return *this;
  }
  /// @brief Same as RPlot::y_ticks(), returning this histogram so chains keep the Histogram type.
  Histogram& y_ticks(std::vector<double> positions, std::vector<std::string> labels) {
    RPlot::y_ticks(std::move(positions), std::move(labels));
    return *this;
  }
  /// @brief Same as RPlot::margins(), returning this histogram so chains keep the Histogram type.
  Histogram& margins(double left, double right, double top, double bottom) {
    RPlot::margins(left, right, top, bottom);
    return *this;
  }
  /// @brief Same as RPlot::tick_font_size(), returning this histogram so chains keep the Histogram type.
  Histogram& tick_font_size(double pixels) {
    RPlot::tick_font_size(pixels);
    return *this;
  }
  /// @brief Same as RPlot::pretty_ticks(), returning this histogram so chains keep the Histogram type.
  Histogram& pretty_ticks(bool enabled = true) {
    RPlot::pretty_ticks(enabled);
    return *this;
  }
  /// @brief Same as RPlot::points(), returning this histogram so chains keep the Histogram type.
  Histogram& points(std::vector<double> x, std::vector<double> y, std::string label = "", graphics::RGB color = {37, 99, 235}, double marker_size = 4.0) {
    RPlot::points(std::move(x), std::move(y), std::move(label), color, marker_size);
    return *this;
  }
  /// @brief Same as RPlot::points_colored(), returning this histogram so chains keep the Histogram type.
  Histogram& points_colored(std::vector<double> x, std::vector<double> y, std::vector<graphics::RGB> colors, double marker_size = 4.0) {
    RPlot::points_colored(std::move(x), std::move(y), std::move(colors), marker_size);
    return *this;
  }
  /// @brief Same as RPlot::line(), returning this histogram so chains keep the Histogram type.
  Histogram& line(std::vector<double> x, std::vector<double> y, std::string label = "", graphics::RGB color = {220, 38, 38}, double stroke_width = 2.0) {
    RPlot::line(std::move(x), std::move(y), std::move(label), color, stroke_width);
    return *this;
  }
  /// @brief Same as RPlot::lines(), returning this histogram so chains keep the Histogram type.
  Histogram& lines(std::vector<double> x, std::vector<double> y, std::string label = "", graphics::RGB color = {220, 38, 38}, double stroke_width = 2.0) {
    RPlot::lines(std::move(x), std::move(y), std::move(label), color, stroke_width);
    return *this;
  }
  /// @brief Same as RPlot::points(), returning this histogram so chains keep the Histogram type.
  Histogram& points(std::vector<double> x, std::vector<double> y, graphics::RGB color, double marker_size = 4.0) {
    RPlot::points(std::move(x), std::move(y), color, marker_size);
    return *this;
  }
  /// @brief Same as RPlot::lines(), returning this histogram so chains keep the Histogram type.
  Histogram& lines(std::vector<double> x, std::vector<double> y, graphics::RGB color, double stroke_width = 2.0) {
    RPlot::lines(std::move(x), std::move(y), color, stroke_width);
    return *this;
  }
  /// @brief Same as RPlot::bars(), returning this histogram so chains keep the Histogram type.
  Histogram& bars(std::vector<double> x, std::vector<double> y, std::string label = "", graphics::RGB color = {22, 163, 74}, double bar_width = 0.8) {
    RPlot::bars(std::move(x), std::move(y), std::move(label), color, bar_width);
    return *this;
  }
  /// @brief Same as RPlot::box(), returning this histogram so chains keep the Histogram type.
  Histogram& box(double position, double whisker_lo, double q1, double median, double q3, double whisker_hi, std::vector<double> outliers = {}, graphics::RGB color = {96, 165, 250}, double width = 0.6) {
    RPlot::box(position, whisker_lo, q1, median, q3, whisker_hi, std::move(outliers), color, width);
    return *this;
  }
  /// @brief Same as RPlot::abline(), returning this histogram so chains keep the Histogram type.
  Histogram& abline(double intercept, double slope, graphics::RGB color = {0, 0, 0}, double stroke_width = 1.0) {
    RPlot::abline(intercept, slope, color, stroke_width);
    return *this;
  }
  /// @brief Same as RPlot::abline_h(), returning this histogram so chains keep the Histogram type.
  Histogram& abline_h(double y_value, graphics::RGB color = {0, 0, 0}, double stroke_width = 1.0) {
    RPlot::abline_h(y_value, color, stroke_width);
    return *this;
  }
  /// @brief Same as RPlot::abline_v(), returning this histogram so chains keep the Histogram type.
  Histogram& abline_v(double x_value, graphics::RGB color = {0, 0, 0}, double stroke_width = 1.0) {
    RPlot::abline_v(x_value, color, stroke_width);
    return *this;
  }
  /// @brief Same as RPlot::qqline(), returning this histogram so chains keep the Histogram type.
  Histogram& qqline(std::vector<double> data, graphics::RGB color = {220, 38, 38}, double stroke_width = 1.5) {
    RPlot::qqline(std::move(data), color, stroke_width);
    return *this;
  }
  /// @brief Same as RPlot::legend(), returning this histogram so chains keep the Histogram type.
  Histogram& legend(std::vector<std::string> labels, std::vector<graphics::RGB> colors) {
    RPlot::legend(std::move(labels), std::move(colors));
    return *this;
  }
  /// @brief Same as RPlot::text(), returning this histogram so chains keep the Histogram type.
  Histogram& text(double x, double y, std::string label, graphics::RGB color = {17, 24, 39}, double font_size = 14.0) {
    RPlot::text(x, y, std::move(label), color, font_size);
    return *this;
  }
  /// @brief Same as RPlot::polygon(), returning this histogram so chains keep the Histogram type.
  Histogram& polygon(std::vector<double> x, std::vector<double> y, graphics::RGB color = {37, 99, 235}, bool filled = true) {
    RPlot::polygon(std::move(x), std::move(y), color, filled);
    return *this;
  }
  /// @brief Same as RPlot::segments(), returning this histogram so chains keep the Histogram type.
  Histogram& segments(std::vector<double> x0, std::vector<double> y0, std::vector<double> x1, std::vector<double> y1, graphics::RGB color = {0, 0, 0}, double stroke_width = 1.0) {
    RPlot::segments(std::move(x0), std::move(y0), std::move(x1), std::move(y1), color, stroke_width);
    return *this;
  }
  /// @}
};

} // namespace datamunge
