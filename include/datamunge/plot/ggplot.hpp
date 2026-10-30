#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::graphics {

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

  /// Renders the plot (or faceted layout) as a standalone SVG document and returns it as a string.
  std::string to_svg() const;
  /// @brief Saves the plot; `.svg` (or no extension) writes SVG, `.png` / `.pdf` write a raster
  ///        (facets tiled like RLayout::save()).
  void save(const std::string& path) const;
  void save_svg(const std::string& path) const;
  void show(const std::string& title_hint = "") const;

  /// @brief The plotted data frame.
  /// @return The frame passed to create().
  [[nodiscard]] const dstruct::DataFrame& data() const noexcept { return data_; }
  /// @brief The aesthetic mapping.
  /// @return The mapping passed to create().
  [[nodiscard]] const Aes& mapping() const noexcept { return mapping_; }
  /// @brief Number of geom layers added so far.
  /// @return The layer count.
  [[nodiscard]] std::size_t layer_count() const noexcept { return layers_.size(); }
  /// @brief The geom of each layer, in drawing order, as ggplot2 suffixes ("point", "line", "bar", "col",
  ///        "histogram", "boxplot", "smooth", "area", "ribbon", "density").
  /// @return One name per layer.
  [[nodiscard]] std::vector<std::string> layer_kinds() const {
    std::vector<std::string> out;
    out.reserve(layers_.size());
    for (const Layer& layer : layers_) {
      switch (layer.kind) {
        case GeomKind::Point: out.emplace_back("point"); break;
        case GeomKind::Line: out.emplace_back("line"); break;
        case GeomKind::Bar: out.emplace_back("bar"); break;
        case GeomKind::Col: out.emplace_back("col"); break;
        case GeomKind::Histogram: out.emplace_back("histogram"); break;
        case GeomKind::Boxplot: out.emplace_back("boxplot"); break;
        case GeomKind::Smooth: out.emplace_back("smooth"); break;
        case GeomKind::Area: out.emplace_back("area"); break;
        case GeomKind::Ribbon: out.emplace_back("ribbon"); break;
        case GeomKind::Density: out.emplace_back("density"); break;
      }
    }
    return out;
  }
  /// @brief The selected theme.
  /// @return GGTheme::Gray unless a theme_*() call changed it.
  [[nodiscard]] GGTheme theme() const noexcept { return theme_; }
  /// @brief Colors set by scale_color_manual().
  /// @return The manual palette; empty for the default palette.
  [[nodiscard]] const std::vector<RGB>& palette() const noexcept { return palette_override_; }
  /// @brief The facet_wrap() column.
  /// @return The column name; empty when not faceted.
  [[nodiscard]] const std::string& facet_column() const noexcept { return facet_column_; }
  /// @brief The facet_wrap() column count.
  /// @return The requested column count; 0 means automatic.
  [[nodiscard]] std::size_t facet_ncol() const noexcept { return facet_ncol_; }
  /// @brief The title set by labs().
  /// @return The title; empty when unset.
  [[nodiscard]] const std::string& title_text() const noexcept { return title_override_; }
  /// @brief The x-axis label set by labs().
  /// @return The label; empty when unset (the x column name is then used).
  [[nodiscard]] const std::string& x_label() const noexcept { return x_label_override_; }
  /// @brief The y-axis label set by labs().
  /// @return The label; empty when unset (the y column name is then used).
  [[nodiscard]] const std::string& y_label() const noexcept { return y_label_override_; }

 private:
  friend class Figure;  ///< Figure(const GGPlot&) renders through build_panel()/facet_layout().

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
  /// @brief Builds the facet_wrap grid; requires a facet column.
  RLayout facet_layout() const;
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

} // namespace datamunge::graphics
