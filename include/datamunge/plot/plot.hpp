#pragma once

#include <cstddef>
#include <string>
#include <vector>

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
  };

  Kind                kind{Kind::Line};
  std::vector<double> x;
  std::vector<double> y;
  std::string         label;
  RGB                 color{};
  double              stroke_width{2.0};
  double              marker_size{4.0};
  double              bar_width{0.8};
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

  void save(const std::string& path) const;
  void save_svg(const std::string& path) const;
  void view(const std::string& title_hint = "") const;
  void show(const std::string& title_hint = "") const;

 protected:
  DataSeries& add_series(DataSeries series);

 private:
  friend class ScatterPlot;
  friend class LinePlot;
  friend class BarChart;

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
  double      x_min_{0.0};
  double      x_max_{1.0};
  double      y_min_{0.0};
  double      y_max_{1.0};
  std::vector<DataSeries> series_;
};

class ScatterPlot final : public Plot {
 public:
  static ScatterPlot create();

  ScatterPlot& points(std::vector<double> x,
                      std::vector<double> y,
                      std::string         label = "",
                      RGB                 color = {37, 99, 235},
                      double              marker_size = 4.0);
};

class LinePlot final : public Plot {
 public:
  static LinePlot create();

  LinePlot& line(std::vector<double> x,
                 std::vector<double> y,
                 std::string         label = "",
                 RGB                 color = {220, 38, 38},
                 double              stroke_width = 2.0);
};

class BarChart final : public Plot {
 public:
  static BarChart create();

  BarChart& bars(std::vector<double> x,
                 std::vector<double> y,
                 std::string         label = "",
                 RGB                 color = {22, 163, 74},
                 double              bar_width = 0.8);
};

} // namespace datamunge::plot
