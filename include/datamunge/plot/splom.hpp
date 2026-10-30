#pragma once

/// @file splom.hpp
/// @brief Generalized pairs plots (R's `pairs()` / GGally's `ggpairs()`) built on RPlot and RLayout.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::graphics {

/// @brief A generalized pairs plot: every pair of variables drawn as one small panel in an
///        n-by-n grid, choosing the panel type from the two variables' kinds.
///
/// Numeric columns are quantitative variables and string columns are categorical variables.
/// Panel (row i, column j) puts variable j on the horizontal axis and variable i on the
/// vertical axis:
/// - diagonal: kernel density curves, one per group_by() group (quantitative), or a single
///   stacked bar of level counts (categorical), with the variable name above it;
/// - quantitative vs quantitative, above the diagonal: the points with a nonparametric smooth of
///   the mean (local linear regression, as R's `loess(degree = 1)`) and a 95% prediction band,
///   one per group;
/// - quantitative vs quantitative, below the diagonal: contours of a bivariate Gaussian kernel
///   density estimate (MASS's `kde2d()`), one set per group in the group's color, drawn at the
///   levels enclosing 25%, 50%, 75% and 95% of the estimated probability;
/// - categorical vs quantitative: one violin (kernel density) per level, with the interquartile
///   range and median marked; vertical when the categorical variable is on x, horizontal otherwise;
/// - categorical vs categorical: a mosaic plot whose tiles are shaded by the Pearson residuals of
///   a chi-squared test of independence (blue: more rows than expected, red: fewer; darker past
///   |residual| 2 and 4, as in R's `mosaicplot(shade = TRUE)`), with the test's p-value on top.
///
/// Rows with a missing value in either variable of a panel are dropped from that panel only.
/// Rendering reuses the RPlot SVG and raster engines through an RLayout, so to_svg()/save()/
/// show() behave like any other plot.
class Splom final {
 public:
  /// @brief What the diagonal panels show.
  enum class Diagonal {
    Density,   ///< kernel density per group, or a stacked bar for a categorical variable (default)
    Histogram, ///< histogram (bins()), or a stacked bar for a categorical variable
    Names,     ///< only the variable name, centred in an empty panel (R's pairs())
  };

  /// @brief Most distinct values a string column may have to be picked when @p columns is empty.
  static constexpr std::size_t max_default_levels = 12;

  /// @brief Builds a matrix over @p columns of @p data. With @p columns empty it takes every
  ///        numeric column plus every string column with at most max_default_levels distinct
  ///        values. Values are copied, so @p data need not outlive the matrix.
  /// @throws std::invalid_argument If a requested column is missing, or if fewer than two
  ///         variables remain.
  static Splom create(const dstruct::DataFrame& data, std::vector<std::string> columns = {});

  /// @brief Sets the figure heading drawn above the panels; an empty string removes it.
  Splom& title(std::string value);
  /// @brief Sets the overall figure size in pixels (default 960 by 960); panels share it equally.
  /// @throws std::invalid_argument If either dimension is zero.
  Splom& size(std::size_t width, std::size_t height);
  /// @brief Sets one color for every point, density and smooth, clearing group_by() and
  ///        point_colors().
  Splom& color(RGB value);
  /// @brief Splits the data into groups by the levels of string column @p column (GGally's
  ///        `ggpairs(mapping = aes(colour = g))`), recycling @p palette over the sorted levels;
  ///        an empty palette uses the default level palette.
  ///
  /// Each group gets its own color for points, its own density on the diagonal, its own smooth
  /// and prediction band above the diagonal, and its own density contours below it. When @p column
  /// is also a panel variable, its violins, bars and mosaic axis share these colors. The column
  /// must have been present in the frame passed to create(); rows with a missing group value are
  /// left out of the quantitative panels. Pass "" to clear the grouping. Clears point_colors().
  /// @throws std::invalid_argument If the column was not a string column of the source frame.
  Splom& group_by(const std::string& column, std::vector<RGB> palette = {});
  /// @brief Sets one color per data row for the points, recycled when shorter than the frame
  ///        (R's `col = cols`). Densities and smooths are then drawn for the data as a whole. Clears group_by(); an empty vector restores color().
  Splom& point_colors(std::vector<RGB> colors);
  /// @brief Colors scatter points by a factor (R's `col = palette[f]`): @p factor holds one
  ///        one-based level code per data row, as returned by `factor()`, and each point takes
  ///        `palette[code - 1]` (recycled); a code below 1 marks a missing value and uses color().
  ///        Categorical variables whose levels encode exactly to @p factor (e.g. the column the
  ///        factor was built from) also fill their violins and stacked bar with @p palette.
  ///        Clears group_by() and point_colors().
  /// @throws std::invalid_argument If @p factor does not have one code per data row or
  ///         @p palette is empty.
  Splom& color_by_factor(std::vector<int> factor, std::vector<RGB> palette);
  /// @brief Sets the colors cycled over the levels of categorical variables in violins and
  ///        stacked bars; an empty vector restores the default. group_by() overrides it for its
  ///        own column.
  Splom& level_palette(std::vector<RGB> colors);
  /// @brief Sets the point radius in pixels (default 2.5).
  /// @throws std::invalid_argument If @p value is not positive.
  Splom& marker_size(double value);
  /// @brief Chooses the diagonal panel style (default Diagonal::Density).
  Splom& diagonal(Diagonal kind);
  /// @brief Sets the bin count used by Diagonal::Histogram (default 10).
  /// @throws std::invalid_argument If @p value is zero.
  Splom& bins(std::size_t value);
  /// @brief Sets the smoother's span: the fraction of a group's points in each local fit
  ///        (R's `loess(span =)`, default 0.75). Smaller spans follow the data more closely.
  /// @throws std::invalid_argument If @p value is not in (0, 1].
  Splom& span(double value);
  /// @brief Sets the coverage of the prediction band above the diagonal (default 0.95).
  /// @throws std::invalid_argument If @p value is not in (0, 1).
  Splom& prediction_level(double value);
  /// @brief Toggles the light grid drawn behind scatter and violin panels (default on).
  Splom& show_grid(bool enabled = true);

  /// @brief Variable names in panel order.
  const std::vector<std::string>& variables() const noexcept { return names_; }
  /// @brief True when variable @p index is categorical (a string column).
  /// @throws std::out_of_range If @p index is not below panel_count().
  bool is_categorical(std::size_t index) const;
  /// @brief Sorted levels of categorical variable @p index; empty for a quantitative variable.
  /// @throws std::out_of_range If @p index is not below panel_count().
  const std::vector<std::string>& levels(std::size_t index) const;
  /// @brief Sorted unique group labels set by group_by(), in palette order; empty without grouping.
  const std::vector<std::string>& groups() const noexcept { return group_names_; }
  /// @brief Borrows the figure heading set by title(); empty when none.
  const std::string& title_text() const noexcept { return title_; }
  /// @brief Figure width in pixels (default 960).
  std::size_t width() const noexcept { return width_; }
  /// @brief Figure height in pixels (default 960).
  std::size_t height() const noexcept { return height_; }
  /// @brief Number of variables, i.e. the grid is panel_count() by panel_count().
  std::size_t panel_count() const noexcept { return names_.size(); }

  /// @brief Builds one panel: variable @p col on x against variable @p row on y, or the diagonal
  ///        panel when they are equal. Two quantitative variables give the smooth panel when
  ///        @p row < @p col and the bivariate density panel when @p row > @p col. The panel is sized to
  ///        its grid cell with tight margins.
  /// @throws std::out_of_range If either index is not below panel_count().
  /// @throws std::invalid_argument If the panel has no complete rows to draw.
  RPlot panel(std::size_t row, std::size_t col) const;
  /// @brief Builds the full grid as an RLayout (row-major), titled and sized like this matrix.
  RLayout layout() const;

  /// @brief Renders the matrix as a standalone SVG document and returns it as a string.
  std::string to_svg() const;
  /// @brief Saves the matrix; `.svg` (or no extension) writes SVG, `.png` / `.pdf` write the
  ///        tiled raster the viewer shows (see RLayout::save()).
  void save(const std::string& path) const;
  /// @brief Writes to_svg() to @p path regardless of its extension.
  /// @throws std::runtime_error If the file cannot be opened for writing.
  void save_svg(const std::string& path) const;
  /// @brief Opens the matrix in the interactive X11 viewer (see RLayout::show()) and blocks
  ///        until it closes; an empty @p title_hint uses the title, or "datamunge splom".
  void show(const std::string& title_hint = "") const;
  /// @brief Same as show().
  void view(const std::string& title_hint = "") const;

 private:
  Splom() = default;

  /// @brief Scatter color of data row @p index: per-row colors, then group colors, then color().
  RGB point_color(std::size_t index) const;
  /// @brief Fill color of level @p level of variable @p var.
  RGB level_color(std::size_t var, std::size_t level) const;
  /// @brief True when categorical variable @p var encodes exactly to the color_by_factor() codes.
  bool matches_factor(std::size_t var) const;
  /// @brief Pixel size of one grid cell, and the panel margins shared by every panel.
  void panel_geometry(std::size_t& cell_w, std::size_t& cell_h, double& left) const;

  RPlot diagonal_panel(std::size_t var, RPlot p) const;
  /// @brief Points plus a local-linear smooth and prediction band per group (upper triangle).
  RPlot smooth_panel(std::size_t row, std::size_t col, RPlot p) const;
  /// @brief Bivariate kernel density contours per group (lower triangle).
  RPlot density2d_panel(std::size_t row, std::size_t col, RPlot p) const;
  /// @brief Complete (x, y) pairs of two quantitative variables, split by group: one list per
  ///        group_by() group, or a single list without grouping.
  std::vector<std::pair<std::vector<double>, std::vector<double>>> grouped_pairs(std::size_t xvar,
                                                                                 std::size_t yvar) const;
  /// @brief Axis range shared by every panel of quantitative variable @p var: its finite values
  ///        padded by 6% on each side, so panels in one row or column line up.
  std::pair<double, double> axis_range(std::size_t var) const;
  /// @brief Color of group @p g, or color() without grouping.
  RGB group_color(std::size_t g) const;
  RPlot violin_panel(std::size_t cat, std::size_t num, bool vertical, RPlot p) const;
  RPlot mosaic_panel(std::size_t row, std::size_t col, RPlot p) const;

  std::vector<std::string>              names_;        ///< variable names in panel order
  std::vector<bool>                     categorical_;  ///< per variable: string column?
  std::vector<std::vector<double>>      values_;       ///< quantitative: values_[var][row]; NaN is null
  std::vector<std::vector<int>>         codes_;        ///< categorical: level index per row; -1 is null
  std::vector<std::vector<std::string>> levels_;       ///< categorical: sorted levels; empty otherwise
  std::vector<std::vector<std::string>> string_cols_;  ///< candidates for group_by(), rows per column; "" is null
  std::vector<std::string>              string_names_; ///< names of string_cols_, same order
  std::size_t                           rows_{0};      ///< row count of the source frame
  std::string                           group_column_; ///< column set by group_by(); empty without grouping
  std::vector<int>                      group_index_;  ///< per row; -1 without grouping or when null
  std::vector<std::string>              group_names_;  ///< sorted unique labels of the group_by() column
  std::vector<RGB>                      palette_;      ///< colors cycled over group_names_; empty without grouping
  std::vector<RGB>                      point_colors_; ///< per-row scatter colors; empty when unset
  std::vector<int>                      factor_codes_;   ///< color_by_factor() codes, one-based; empty when unset
  std::vector<RGB>                      factor_palette_; ///< color_by_factor() palette, indexed by code - 1
  std::vector<RGB>                      level_palette_; ///< fills for categorical levels; empty uses the default
  std::string                           title_;        ///< figure heading; empty draws no band
  std::size_t                           width_{960};   ///< figure width in pixels
  std::size_t                           height_{960};  ///< figure height in pixels
  RGB                                   color_{37, 99, 235}; ///< point/bar color without grouping
  double                                marker_size_{2.5};   ///< point radius in pixels
  Diagonal                              diagonal_{Diagonal::Density}; ///< diagonal panel style
  std::size_t                           bins_{10};     ///< histogram bins on the diagonal
  double                                span_{0.75};   ///< smoother span, fraction of a group's points
  double                                level_{0.95};  ///< prediction band coverage
  bool                                  show_grid_{true}; ///< whether scatter/violin panels draw a grid
};

} // namespace datamunge::graphics

namespace datamunge {

/// @brief R's `pairs(df, col = df$group)` as one call: builds the pairs plot over the columns of
///        @p data (or the named @p columns), opens it in the interactive viewer, and returns it so
///        it can also be saved. Pass a string column as @p group_by to split every panel by group.
///        Showing requires X11 support; see graphics::Splom for a non-interactive API.
/// @throws std::invalid_argument For missing columns or fewer than two variables;
///         std::runtime_error if the viewer cannot be opened.
graphics::Splom splom(const dstruct::DataFrame& data,
                  std::vector<std::string>  columns = {},
                  const std::string&        group_by = "");

} // namespace datamunge
