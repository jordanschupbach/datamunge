#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::plot {

/// Aesthetic mapping, mimicking ggplot2's `aes()`. Column names refer to columns of the
/// `dstruct::DataFrame` passed to `GGPlot::create()`. `color` and `fill` are treated
/// identically (both select a discrete grouping column); leave a field empty to omit it.
struct Aes {
  std::string x;
  std::string y;
  std::string color;
  std::string fill;
  std::string group;
};

enum class GGTheme { Gray, Minimal, BW, Classic };

/// A ggplot2-style grammar-of-graphics builder: layers of geoms are accumulated against one
/// `Aes` mapping and replayed at `save()` time (once per facet panel, if `facet_wrap()` was
/// used), each replay producing an `RPlot` so rendering reuses the exact same SVG/PNG/PDF
/// engine as the R-base graphics library.
class GGPlot final {
 public:
  static GGPlot create(const dstruct::DataFrame& data, Aes mapping);

  GGPlot& geom_point(RGB color = {37, 99, 235}, double size = 3.0);
  GGPlot& geom_line(RGB color = {37, 99, 235}, double width = 1.5);
  GGPlot& geom_bar(RGB color = {37, 99, 235});
  GGPlot& geom_col(RGB color = {37, 99, 235});
  GGPlot& geom_histogram(std::size_t bins = 30, RGB color = {96, 165, 250});
  GGPlot& geom_boxplot(RGB color = {96, 165, 250});
  GGPlot& geom_smooth(RGB color = {220, 38, 38});
  GGPlot& geom_area(RGB color = {96, 165, 250});
  GGPlot& geom_ribbon(std::string ymin_col, std::string ymax_col, RGB color = {96, 165, 250});
  GGPlot& geom_density(RGB color = {37, 99, 235});

  GGPlot& facet_wrap(std::string column, std::size_t ncol = 0);
  GGPlot& theme_minimal();
  GGPlot& theme_bw();
  GGPlot& theme_classic();
  GGPlot& scale_color_manual(std::vector<RGB> values);
  GGPlot& labs(std::string title = "", std::string x = "", std::string y = "");

  void save(const std::string& path) const;
  void save_svg(const std::string& path) const;
  void show(const std::string& title_hint = "") const;

 private:
  enum class GeomKind { Point, Line, Bar, Col, Histogram, Boxplot, Smooth, Area, Ribbon, Density };

  struct Layer {
    GeomKind    kind;
    RGB         color{37, 99, 235};
    double      size{3.0};
    std::size_t bins{30};
    std::string ymin_col;
    std::string ymax_col;
  };

  RPlot build_panel(const dstruct::DataFrame& panel_data, const std::string& panel_title) const;
  void  apply_theme(Plot& panel) const;

  dstruct::DataFrame data_;
  Aes                mapping_;
  std::vector<Layer> layers_;
  GGTheme            theme_{GGTheme::Gray};
  std::vector<RGB>   palette_override_;
  std::string        facet_column_;
  std::size_t        facet_ncol_{0};
  std::string        title_override_;
  std::string        x_label_override_;
  std::string        y_label_override_;
};

} // namespace datamunge::plot
