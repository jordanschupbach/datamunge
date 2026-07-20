#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge {
class Callback;
} // namespace datamunge

namespace datamunge::plot {

struct RGB {
  int r{31};
  int g{41};
  int b{55};
};

struct DataSeries {
  enum class Kind {
    Scatter,
    Line,
    Bar,
    Box,     // one box-and-whisker: x = {position}, y = {whisker_lo, q1, median, q3, whisker_hi, outlier...}
    Polygon, // closed polygon / pie wedge: x/y are vertices, `filled` selects fill vs. stroke-only
    Text,    // single label drawn at (x[0], y[0]); `label` is the text, `marker_size` is the font size
    Segment, // disconnected line segments: consecutive pairs (x[2i],y[2i])-(x[2i+1],y[2i+1])
  };

  Kind                kind{Kind::Line};
  std::vector<double> x;
  std::vector<double> y;
  std::string         label;
  RGB                 color{};
  double              stroke_width{2.0};
  double              marker_size{4.0};
  double              bar_width{0.8};
  bool                filled{false};
};

/// A straight reference line spanning the full plotting area, as drawn by R's `abline()`.
struct ABLine {
  bool   vertical{false};
  double value{0.0}; ///< y-intercept for a sloped/horizontal line; x-position for a vertical line
  double slope{0.0};
  RGB    color{0, 0, 0};
  double stroke_width{1.0};
};

/// One manually-specified legend row, as drawn by R's `legend()`. When a plot has any manual
/// entries, they replace the default auto-generated (one-row-per-labeled-series) legend.
struct LegendEntry {
  std::string label;
  RGB         color{0, 0, 0};
};

class Plot {
 public:
  virtual ~Plot() = default;

  Plot& size(std::size_t width, std::size_t height);
  Plot& title(std::string value);
  Plot& x_label(std::string value);
  Plot& y_label(std::string value);
  Plot& background(RGB color);
  Plot& axis_color(RGB color);
  Plot& grid_color(RGB color);
  Plot& show_grid(bool enabled = true);
  Plot& x_limits(double min_x, double max_x);
  Plot& y_limits(double min_y, double max_y);
  Plot& hide_axes(bool enabled = true);
  Plot& x_tick_labels(std::vector<std::string> labels);

  std::size_t width() const noexcept { return width_; }
  std::size_t height() const noexcept { return height_; }
  const std::string& title_text() const noexcept { return title_; }
  const std::string& x_label_text() const noexcept { return x_label_; }
  const std::string& y_label_text() const noexcept { return y_label_; }
  const std::vector<DataSeries>& series() const noexcept { return series_; }
  RGB background_color() const noexcept { return background_; }
  RGB axes_color() const noexcept { return axis_color_; }
  RGB major_grid_color() const noexcept { return grid_color_; }
  bool grid_visible() const noexcept { return show_grid_; }
  bool has_x_limits() const noexcept { return has_x_limits_; }
  bool has_y_limits() const noexcept { return has_y_limits_; }
  double x_min() const noexcept { return x_min_; }
  double x_max() const noexcept { return x_max_; }
  double y_min() const noexcept { return y_min_; }
  double y_max() const noexcept { return y_max_; }
  bool axes_hidden() const noexcept { return hide_axes_; }
  const std::vector<std::string>& x_tick_label_list() const noexcept { return x_tick_labels_; }
  const std::vector<ABLine>& reference_lines() const noexcept { return ablines_; }
  const std::vector<LegendEntry>& legend_entries() const noexcept { return legend_entries_; }

  void save(const std::string& path) const;
  void save_svg(const std::string& path) const;
  void view(const std::string& title_hint = "") const;
  void show(const std::string& title_hint = "") const;

 protected:
  DataSeries& add_series(DataSeries series);
  Plot&       add_reference_line(ABLine line);
  Plot&       add_legend_entry(LegendEntry entry);

 private:
  friend class RPlot;

  std::size_t width_{960};
  std::size_t height_{640};
  std::string title_{"datamunge plot"};
  std::string x_label_{"x"};
  std::string y_label_{"y"};
  RGB         background_{255, 255, 255};
  RGB         axis_color_{55, 65, 81};
  RGB         grid_color_{229, 231, 235};
  bool        show_grid_{true};
  bool        has_x_limits_{false};
  bool        has_y_limits_{false};
  bool        hide_axes_{false};
  double      x_min_{0.0};
  double      x_max_{1.0};
  double      y_min_{0.0};
  double      y_max_{1.0};
  std::vector<DataSeries>  series_;
  std::vector<ABLine>      ablines_;
  std::vector<LegendEntry> legend_entries_;
  std::vector<std::string> x_tick_labels_;
};

/// A single R-base-graphics-style plot (`plot()`, `hist()`, `barplot()`, `boxplot()`, `pie()`,
/// `curve()`, `qqnorm()`, ...) plus the chainable "add to current plot" verbs R exposes as
/// separate top-level functions (`points()`, `lines()`, `abline()`, `legend()`, `text()`, ...).
class RPlot final : public Plot {
 public:
  static RPlot create();

  /// Mimics R's `plot(x, y, type = "p"|"l"|"b")`.
  static RPlot plot(std::vector<double> x,
                     std::vector<double> y,
                     std::string         type = "p",
                     std::string         label = "",
                     RGB                 color = {37, 99, 235});

  /// Mimics R's `hist(x, breaks = bins)`. Bin edges are equal-width over the data range.
  static RPlot hist(std::vector<double> data,
                     std::size_t         bins = 10,
                     std::string         label = "",
                     RGB                 color = {96, 165, 250});

  /// Mimics R's `barplot(heights, names.arg = names)`.
  static RPlot barplot(std::vector<double>      heights,
                        std::vector<std::string> names = {},
                        std::string              label = "",
                        RGB                       color = {22, 163, 74});

  /// Mimics R's `boxplot(...)` over one or more groups.
  static RPlot boxplot(std::vector<std::vector<double>> groups,
                        std::vector<std::string>         names = {},
                        RGB                               color = {96, 165, 250});

  /// Mimics R's `pie(x, labels = names)`.
  static RPlot pie(std::vector<double>      values,
                    std::vector<std::string> names = {},
                    std::vector<RGB>         colors = {});

  /// Mimics R's `curve(expr, from, to)`; `f` is sampled at `n` evenly-spaced points.
  static RPlot curve(datamunge::Callback& f,
                      double               from,
                      double               to,
                      std::size_t          n = 101,
                      std::string          label = "",
                      RGB                  color = {220, 38, 38});

  /// Mimics R's `qqnorm(y)`: plots sample quantiles of `data` against standard-normal quantiles.
  static RPlot qqnorm(std::vector<double> data, std::string label = "sample", RGB color = {37, 99, 235});

  RPlot& points(std::vector<double> x,
                std::vector<double> y,
                std::string         label = "",
                RGB                 color = {37, 99, 235},
                double              marker_size = 4.0);
  RPlot& line(std::vector<double> x,
              std::vector<double> y,
              std::string         label = "",
              RGB                 color = {220, 38, 38},
              double              stroke_width = 2.0);
  RPlot& lines(std::vector<double> x,
               std::vector<double> y,
               std::string         label = "",
               RGB                 color = {220, 38, 38},
               double              stroke_width = 2.0);
  /// Lower-level bar primitive at arbitrary x positions (`barplot()`/`hist()` build on this).
  RPlot& bars(std::vector<double> x,
              std::vector<double> y,
              std::string         label = "",
              RGB                 color = {22, 163, 74},
              double              bar_width = 0.8);
  /// Lower-level box-and-whisker primitive at an arbitrary x position (`boxplot()` builds on this).
  RPlot& box(double              position,
             double              whisker_lo,
             double              q1,
             double              median,
             double              q3,
             double              whisker_hi,
             std::vector<double> outliers = {},
             RGB                 color = {96, 165, 250},
             double              width = 0.6);

  RPlot& abline(double intercept, double slope, RGB color = {0, 0, 0}, double stroke_width = 1.0);
  RPlot& abline_h(double y_value, RGB color = {0, 0, 0}, double stroke_width = 1.0);
  RPlot& abline_v(double x_value, RGB color = {0, 0, 0}, double stroke_width = 1.0);

  /// Mimics R's `qqline()`: draws the line through the 1st and 3rd sample/theoretical quartiles
  /// of `data`, the same data that was passed to `qqnorm()`.
  RPlot& qqline(std::vector<double> data, RGB color = {220, 38, 38}, double stroke_width = 1.5);

  RPlot& legend(std::vector<std::string> labels, std::vector<RGB> colors);
  RPlot& text(double x, double y, std::string label, RGB color = {17, 24, 39}, double font_size = 14.0);
  RPlot& polygon(std::vector<double> x,
                 std::vector<double> y,
                 RGB                 color = {37, 99, 235},
                 bool                filled = true);
  RPlot& segments(std::vector<double> x0,
                  std::vector<double> y0,
                  std::vector<double> x1,
                  std::vector<double> y1,
                  RGB                 color = {0, 0, 0},
                  double              stroke_width = 1.0);
};

/// Tiles independently-built `Plot` objects into one multi-panel figure, mimicking R's
/// `par(mfrow = c(rows, cols))`.
class RLayout final {
 public:
  static RLayout create(std::size_t rows, std::size_t cols);

  RLayout& add(const Plot& panel);
  RLayout& size(std::size_t width, std::size_t height);

  void save(const std::string& path) const;
  void save_svg(const std::string& path) const;

 private:
  struct Panel {
    std::size_t width;
    std::size_t height;
    std::string svg_fragment; ///< rendered eagerly in add(), so panels need not outlive save()
  };

  std::size_t rows_{1};
  std::size_t cols_{1};
  std::size_t width_{960};
  std::size_t height_{640};
  std::vector<Panel> panels_;
};

} // namespace datamunge::plot
