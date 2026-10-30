#pragma once

#include <datamunge/algebra/algebra.hpp>
#include <datamunge/autodiff/autodiff.hpp>
#include <datamunge/bayes/bayes.hpp>
#include <datamunge/cv/cv.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/fem/fem.hpp>
#include <datamunge/filter/filter.hpp>
#include <datamunge/geometry/geometry.hpp>
#include <datamunge/gis/gis.hpp>
#include <datamunge/image/imaging.hpp>
#include <datamunge/linalg/tensor.hpp>
#include <datamunge/ode/ode.hpp>
#include <datamunge/optim/optim.hpp>
#include <datamunge/plot/ggplot.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/plot/figure.hpp>
#include <datamunge/plot/histogram.hpp>
#include <datamunge/plot/splom.hpp>
#include <datamunge/plot/class_regions.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/random/mt19937_64_analysis.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/quality.hpp>
#include <datamunge/stats/stats.hpp>
#include <datamunge/vector.hpp>
#include <datamunge/plot/compose.hpp>
#include <datamunge/bspline_regression.hpp>
#include <datamunge/pspline_regression.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace datamunge {

/// @brief Prints a hello message to standard output.
void hello();

/// @brief Base class for user-defined callbacks invoked by datamunge functions.
///
/// Derive from this class and override @c call() to supply custom behavior
/// wherever a @c Callback* is accepted.
class Callback {
 public:
  /// @brief Virtual destructor — ensures proper cleanup of derived objects.
  virtual ~Callback() = default;

  /// @brief Applies the callback to a single value.
  /// @param x Input value.
  /// @return Transformed value; the default implementation returns @p x unchanged.
  virtual double call(double x) { return x; }
};

/// @brief Invokes a callback with the given value.
/// @param x    Input value passed to the callback.
/// @param cb   Pointer to a @c Callback instance; must not be null.
/// @return     Result of @c cb->call(x).
double call_with_callback(double x, Callback* cb);

/// @brief Applies a callback to every element of a vector.
/// @param values Source vector of doubles.
/// @param cb     Pointer to a @c Callback instance; must not be null.
/// @return       New vector where each element is the result of @c cb->call(v)
///               for the corresponding element @c v in @p values.
std::vector<double> map_dvector_with_callback(const std::vector<double>& values, Callback* cb);

/// @brief Constructs a three-element vector from individual values.
/// @param a First element.
/// @param b Second element.
/// @param c Third element.
/// @return  @c std::vector<double>{a, b, c}.
std::vector<double> make_dvector(double a, double b, double c);

/// @brief Computes the sum of all elements in a vector.
/// @param values Vector of doubles to sum.
/// @return       Sum of all elements, or 0.0 if the vector is empty.
double sum_dvector(const std::vector<double>& values);

/// @brief Constructs a pair of doubles.
/// @param a First element.
/// @param b Second element.
/// @return  @c std::pair<double, double>{a, b}.
std::pair<double, double> make_dpair(double a, double b);

/// @brief Computes the sum of both elements in a pair.
/// @param values Pair of doubles.
/// @return       @c values.first + values.second.
double sum_dpair(const std::pair<double, double>& values);

/// @brief SWIG-friendly facade for the C++ dataframe API exposed to bindings.
class DataFrame {
 public:
  DataFrame() = default;

  /// @brief Equivalent to the default constructor, exposed as a static factory (like iris()/
  ///        penguins()) as a workaround for a SWIG R-backend bug where the plain no-argument
  ///        constructor's ownership-flagged pointer breaks method dispatch on the result.
  [[nodiscard]] static DataFrame* empty();

  [[nodiscard]] std::size_t nrows() const;
  [[nodiscard]] std::size_t ncols() const;
  [[nodiscard]] std::vector<std::size_t> shape() const;
  [[nodiscard]] std::vector<std::string> columns() const;

  void add_numeric_column(const std::string& column_name, const std::vector<double>& values,
                          const std::vector<int>& valid_mask = {});
  void add_string_column(const std::string& column_name, const std::vector<std::string>& values,
                         const std::vector<int>& valid_mask = {});
  void add_string_column_encoded(const std::string& column_name, const std::string& encoded_values,
                                 const std::vector<int>& valid_mask = {});
  void fill_null_numeric(const std::string& column_name, double value);
  void fill_null_string(const std::string& column_name, const std::string& value);

  /// @brief dplyr::mutate()-style upsert: adds column_name if absent, replaces it (same type) if
  ///        present. Always returns a new DataFrame, so it composes into a pipe.
  [[nodiscard]] DataFrame* mutate_numeric(const std::string& column_name, const std::vector<double>& values,
                                          const std::vector<int>& valid_mask = {}) const;
  [[nodiscard]] DataFrame* mutate_string(const std::string& column_name, const std::vector<std::string>& values,
                                         const std::vector<int>& valid_mask = {}) const;
  [[nodiscard]] DataFrame* mutate_string_encoded(const std::string& column_name, const std::string& encoded_values,
                                                 const std::vector<int>& valid_mask = {}) const;

  /// @brief Non-mutating, chainable single-pair rename.
  [[nodiscard]] DataFrame* rename(const std::string& old_name, const std::string& new_name) const;

  [[nodiscard]] DataFrame* select(const std::vector<std::string>& selected_columns) const;
  [[nodiscard]] DataFrame* select_encoded(const std::string& encoded_columns) const;

  /// @brief dplyr::relocate()-style column reorder: moves `columns` to the front (default) or
  ///        immediately after the column named `after`.
  [[nodiscard]] DataFrame* relocate(const std::vector<std::string>& columns, const std::string& after = "") const;
  [[nodiscard]] DataFrame* relocate_encoded(const std::string& encoded_columns, const std::string& after = "") const;

  [[nodiscard]] DataFrame* sort_by(const std::string& column_name, bool ascending = true) const;
  /// @brief dplyr::arrange()-style multi-key sort. `ascending` defaults to all-true; when
  ///        provided it must have the same length as `columns`.
  [[nodiscard]] DataFrame* arrange(const std::vector<std::string>& columns, const std::vector<int>& ascending = {}) const;
  [[nodiscard]] DataFrame* arrange_encoded(const std::string& encoded_columns, const std::vector<int>& ascending = {}) const;

  [[nodiscard]] DataFrame* drop_duplicates(const std::vector<std::string>& subset = {}) const;
  [[nodiscard]] DataFrame* drop_duplicates_encoded(const std::string& encoded_subset) const;
  /// @brief dplyr::distinct() alias for drop_duplicates().
  [[nodiscard]] DataFrame* distinct(const std::vector<std::string>& subset = {}) const;
  [[nodiscard]] DataFrame* distinct_encoded(const std::string& encoded_subset) const;

  /// @brief dplyr::pull()-style column extraction. Nulls come back as NaN (numeric) / "" (string)
  ///        in the value vector; check pull_numeric_valid()/pull_string_valid() (1 = present, 0 =
  ///        null, same convention as add_numeric_column's valid_mask) if nulls matter.
  [[nodiscard]] std::vector<double> pull_numeric(const std::string& column_name) const;
  [[nodiscard]] std::vector<int> pull_numeric_valid(const std::string& column_name) const;
  [[nodiscard]] std::vector<std::string> pull_string(const std::string& column_name) const;
  [[nodiscard]] std::vector<int> pull_string_valid(const std::string& column_name) const;

  /// @brief Number of distinct values in column_name; a null counts as one additional distinct
  ///        value if present.
  [[nodiscard]] std::size_t n_distinct(const std::string& column_name) const;

  [[nodiscard]] DataFrame* group_by_sum(const std::vector<std::string>& key_columns,
                                        const std::vector<std::string>& value_columns) const;
  [[nodiscard]] DataFrame* group_by_sum_encoded(const std::string& encoded_key_columns,
                                                const std::string& encoded_value_columns) const;

  /// @brief dplyr::count()-style grouped row counts, default result column name "n".
  [[nodiscard]] DataFrame* count(const std::vector<std::string>& key_columns, const std::string& count_column_name = "n") const;
  [[nodiscard]] DataFrame* count_encoded(const std::string& encoded_key_columns, const std::string& count_column_name = "n") const;

  /// @brief General dplyr::summarise()-style aggregation: one output row per distinct
  ///        combination of `key_columns`, with one output column per (agg_columns[i],
  ///        agg_funcs[i], result_names[i]) triple -- all three arrays must have the same length.
  ///        agg_funcs entries are one of "sum", "mean", "min", "max", "median", "stddev",
  ///        "count", "n_distinct" ("count" ignores the corresponding agg_columns entry, which may
  ///        be ""); a "" result_names entry defaults to the agg_columns entry (or "n" for count).
  [[nodiscard]] DataFrame* summarise(const std::vector<std::string>& key_columns,
                                     const std::vector<std::string>& agg_columns,
                                     const std::vector<std::string>& agg_funcs,
                                     const std::vector<std::string>& result_names) const;
  [[nodiscard]] DataFrame* summarise_encoded(const std::string& encoded_key_columns, const std::string& encoded_agg_columns,
                                             const std::string& encoded_agg_funcs,
                                             const std::string& encoded_result_names) const;

  /// @brief dplyr::pivot_longer()-style reshape: stacks `value_columns` into two new columns
  ///        (`names_to` holding the source column name, `values_to` holding its value).
  [[nodiscard]] DataFrame* pivot_longer(const std::vector<std::string>& value_columns, const std::string& names_to = "name",
                                        const std::string& values_to = "value") const;
  [[nodiscard]] DataFrame* pivot_longer_encoded(const std::string& encoded_value_columns,
                                                const std::string& names_to = "name",
                                                const std::string& values_to = "value") const;

  /// @brief dplyr::pivot_wider()-style reshape: `names_from` (a string column) supplies new
  ///        column names, `values_from` supplies their values; `id_columns` defaults to every
  ///        other column.
  [[nodiscard]] DataFrame* pivot_wider(const std::string& names_from, const std::string& values_from,
                                       const std::vector<std::string>& id_columns = {}) const;
  [[nodiscard]] DataFrame* pivot_wider_encoded(const std::string& names_from, const std::string& values_from,
                                               const std::string& encoded_id_columns) const;

  /// @brief dplyr::bind_rows()-style row union: aligns columns by name (unlike concat_rows,
  ///        which isn't exposed here), null-filling any column present in only one frame.
  [[nodiscard]] DataFrame* bind_rows(const DataFrame& other) const;
  /// @brief dplyr::bind_cols()-style column union: both frames must have the same row count and
  ///        disjoint column names.
  [[nodiscard]] DataFrame* bind_cols(const DataFrame& other) const;

  /// @param join_type One of "inner" (default), "left", "right", "full", "semi", "anti". Any
  ///        column name present in both frames (other than the key column when left_key ==
  ///        right_key) is suffixed on both sides so the result never has duplicate names.
  [[nodiscard]] DataFrame* join(const DataFrame& right, const std::string& left_key, const std::string& right_key,
                                const std::string& join_type = "inner", const std::string& left_suffix = "_x",
                                const std::string& right_suffix = "_y") const;

  [[nodiscard]] std::size_t numeric_count(const std::string& column_name) const;
  [[nodiscard]] std::size_t numeric_null_count(const std::string& column_name) const;
  [[nodiscard]] double numeric_sum(const std::string& column_name) const;
  [[nodiscard]] double numeric_mean(const std::string& column_name) const;
  [[nodiscard]] double numeric_min(const std::string& column_name) const;
  [[nodiscard]] double numeric_max(const std::string& column_name) const;
  [[nodiscard]] std::string to_string(std::size_t max_rows = 10) const;
  /// @brief The frame as a Markdown table (numeric columns right-aligned, at most @p digits
  ///        decimals each), like knitr::kable(); see dstruct::DataFrame::to_md().
  [[nodiscard]] std::string to_md(int digits = 3) const;

  /// @brief true if column_name holds numeric values, false if it holds strings.
  [[nodiscard]] bool is_numeric_column(const std::string& column_name) const;
  [[nodiscard]] bool is_null(const std::string& column_name, std::size_t row_index) const;
  /// @brief The value at (column_name, row_index); throws if the cell is null or the column isn't numeric.
  [[nodiscard]] double numeric_at(const std::string& column_name, std::size_t row_index) const;
  /// @brief The value at (column_name, row_index); throws if the cell is null or the column isn't string-typed.
  [[nodiscard]] std::string string_at(const std::string& column_name, std::size_t row_index) const;

  // Bundled sample datasets.
  [[nodiscard]] static DataFrame* iris();
  [[nodiscard]] static DataFrame* penguins();
  [[nodiscard]] static DataFrame* tooth_growth();
  /// @brief UCI wine recognition data, 178 x 14; see datasets::wine().
  [[nodiscard]] static DataFrame* wine();
  /// @brief Kaggle Titanic training set, 891 x 12 with missing values; see datasets::titanic().
  [[nodiscard]] static DataFrame* titanic();
  /// @brief Rubin's eight schools coaching experiment, 8 x 3; see datasets::eight_schools().
  [[nodiscard]] static DataFrame* eight_schools();
  /// @brief Monthly airline passengers 1949-1960, 144 x 3; see datasets::air_passengers().
  [[nodiscard]] static DataFrame* air_passengers();
  /// @brief Maunga Whau (Mt Eden) elevations on a 10 m grid, 5307 x 3; see datasets::volcano().
  [[nodiscard]] static DataFrame* volcano();
  /// @brief Old Faithful eruptions, 272 x 2; see datasets::faithful().
  [[nodiscard]] static DataFrame* faithful();
  /// @brief Motor Trend cars, 32 x 12 (row names as `model`); see datasets::mtcars().
  [[nodiscard]] static DataFrame* mtcars();
  /// @brief New York air quality 1973, 153 x 6 with missing values; see datasets::airquality().
  [[nodiscard]] static DataFrame* airquality();
  /// @brief Wisconsin diagnostic breast cancer, 569 x 31; see datasets::breast_cancer().
  [[nodiscard]] static DataFrame* breast_cancer();
  /// @brief Efron et al.'s diabetes progression data, 442 x 11; see datasets::diabetes().
  [[nodiscard]] static DataFrame* diabetes();
  /// @brief 8x8 handwritten digits, 1797 x 65; see datasets::digits().
  [[nodiscard]] static DataFrame* digits();
  /// @brief ggplot2's diamonds, 53940 x 10, downloaded once and cached; see datasets::diamonds().
  [[nodiscard]] static DataFrame* diamonds();
  /// @brief California housing, 20640 x 10, downloaded once and cached; see
  ///        datasets::california_housing().
  [[nodiscard]] static DataFrame* california_housing();
  /// @brief Any table-shaped dataset by name, e.g. "wine"; see datasets::load().
  /// @throws std::invalid_argument for an unknown or image dataset name.
  [[nodiscard]] static DataFrame* dataset(const std::string& name);
  /// @brief One row per available dataset (name, rows, columns, storage, size_mb,
  ///        description); see datasets::catalog().
  [[nodiscard]] static DataFrame* dataset_catalog();
  /// @brief Reads a whole CSV file, typing columns automatically; see datasets::read_csv().
  /// @param path The file to read.
  /// @param delimiter The field separator.
  /// @throws std::runtime_error if the file cannot be read or is malformed.
  [[nodiscard]] static DataFrame* read_csv(const std::string& path, char delimiter = ',');
  /// @brief Parses CSV text, typing columns automatically; see datasets::parse_csv().
  /// @throws std::runtime_error if the text is malformed.
  [[nodiscard]] static DataFrame* parse_csv(const std::string& text, char delimiter = ',');
  /// @brief An ImageLoader batch as a table (`pixel_0`.., `label`, `index`); see
  ///        datasets::to_dataframe().
  [[nodiscard]] static DataFrame* from_image_batch(const datasets::ImageBatch& batch);

 private:
  friend class LM;
  friend class GGPlot;
  friend class Splom;
  friend class CsvReader;
  friend class LDA;
  friend class SVM;
  friend class DecisionTreeClassifier;
  friend class DecisionTreeRegressor;
  friend class RandomForestClassifier;
  friend class RandomForestRegressor;
  friend class ElasticNet;
  friend class Ridge;
  friend class Lasso;
  friend class KNNClassifier;
  friend class KNNRegressor;
  friend class KMeans;
  friend class AgglomerativeClustering;
  friend class DBSCAN;
  friend class GBMClassifier;
  friend class GBMRegressor;
  friend class XGBoostClassifier;
  friend class XGBoostRegressor;
  friend class KernelRegression;
  friend class GaussianProcessRegression;
  friend class NaiveBayesClassifier;
  friend class GLM;
  friend class LMM;
  friend class GLMM;
  friend class INLAMixedModel;
  friend class ShapeLayer;
  friend class PCA;
  friend class MDS;
  friend class Isomap;
  friend class LLE;
  friend class TSNE;
  friend class LaplacianEigenmaps;
  friend class DiffusionMaps;
  friend class KernelPCA;
  friend class SammonMapping;
  friend class UMAP;

  explicit DataFrame(dstruct::DataFrame frame);

  template <typename T>
  static std::vector<std::optional<T>> apply_valid_mask(const std::vector<T>& values, const std::vector<int>& valid_mask);
  static std::vector<std::string> split_encoded_strings(const std::string& encoded_values);

  dstruct::DataFrame frame_;
};

/// @brief SWIG-friendly streaming CSV reader: returns a big file as DataFrames of at most
///        batch_size() rows each, so only one chunk is in memory at a time.
///
/// Wraps datasets::CsvReader; column types are fixed from the first 1000 rows.
///
/// @par Example
/// @code
/// dm::CsvReader reader(dm::datasets::fetch("diamonds"), 10000);
/// while (reader.has_next()) {
///   std::unique_ptr<dm::DataFrame> chunk(reader.next());
///   // ... use chunk ...
/// }
/// @endcode
class CsvReader {
 public:
  /// @brief Opens @p path and infers its column types.
  /// @param path The CSV file.
  /// @param batch_size Rows per chunk; must be positive.
  /// @param delimiter The field separator.
  /// @throws std::invalid_argument if batch_size is not positive.
  /// @throws std::runtime_error if the file cannot be opened or is malformed.
  explicit CsvReader(const std::string& path, int batch_size = 1000, char delimiter = ',');

  /// @brief true while rows remain.
  [[nodiscard]] bool has_next();
  /// @brief The next chunk (caller owns it).
  /// @throws std::out_of_range when no rows remain.
  [[nodiscard]] DataFrame* next();
  /// @brief Rewinds to the first data row.
  void reset();
  /// @brief Rows per chunk.
  [[nodiscard]] int batch_size() const;
  /// @brief Changes the rows per chunk.
  /// @throws std::invalid_argument if @p rows is not positive.
  void set_batch_size(int rows);
  /// @brief Column names, in file order.
  [[nodiscard]] std::vector<std::string> column_names() const;
  /// @brief Rows returned since opening or the last reset().
  [[nodiscard]] double rows_read() const;

 private:
  datasets::CsvReader impl_;  ///< The core reader.
};

/// @brief SWIG-friendly facade for datamunge::gis::ShapeLayer — reads a shapefile (.shp geometry
///        + .dbf attributes) and draws it as a map. Geometry access is flattened into per-shape/
///        per-part coordinate vectors rather than exposing gis::Shape's nested
///        vector<vector<Point2D>> directly, matching this facade layer's usual convention for
///        types SWIG can't bind cleanly.
class ShapeLayer {
 public:
  /// @brief Reads "<path>.shp" and "<path>.dbf" (`path` may already end in one of those
  ///        extensions, or in neither).
  [[nodiscard]] static ShapeLayer* read(const std::string& path);

  [[nodiscard]] std::size_t size() const;
  /// @brief One of "point", "polyline", "polygon", "multipoint", or "null" (an empty layer).
  [[nodiscard]] std::string shape_type() const;
  /// @brief {xmin, ymin, xmax, ymax}, from the shapefile's own declared bounding box.
  [[nodiscard]] std::vector<double> bounds() const;
  [[nodiscard]] DataFrame* attributes() const;

  [[nodiscard]] std::string shape_kind(std::size_t shape_index) const;
  /// @brief Number of parts (rings for Polygon, lines for PolyLine; 0 for Point/MultiPoint,
  ///        which use point_x()/point_y() instead).
  [[nodiscard]] std::size_t num_parts(std::size_t shape_index) const;
  [[nodiscard]] std::vector<double> part_x(std::size_t shape_index, std::size_t part_index) const;
  [[nodiscard]] std::vector<double> part_y(std::size_t shape_index, std::size_t part_index) const;
  /// @brief Every point's x/y in a Point or MultiPoint shape (a single element for Point).
  [[nodiscard]] std::vector<double> point_x(std::size_t shape_index) const;
  [[nodiscard]] std::vector<double> point_y(std::size_t shape_index) const;

  [[nodiscard]] datamunge::graphics::RPlot plot(datamunge::graphics::RGB fill_color = {148, 163, 184},
                                            datamunge::graphics::RGB border_color = {51, 65, 85},
                                            std::size_t width = 800, std::size_t height = 800) const;

 private:
  explicit ShapeLayer(gis::ShapeLayer layer);

  gis::ShapeLayer layer_;
};

/// @brief SWIG-friendly facade for datamunge::graphics::GGPlot — a ggplot2-style grammar-of-graphics
///        builder. Geom/theme/scale/facet calls are chainable, mirroring ggplot2's own layered
///        `ggplot(df, aes(...)) + geom_point() + ...` style as closely as C++ method chaining allows.
class GGPlot {
 public:
  /// @param color_column,fill_column,group_column Optional discrete grouping columns; pass "" to omit.
  GGPlot(const DataFrame& data, const std::string& x_column, const std::string& y_column = "",
         const std::string& color_column = "", const std::string& fill_column = "",
         const std::string& group_column = "");

  GGPlot& geom_point(datamunge::graphics::RGB color = {37, 99, 235}, double size = 3.0);
  GGPlot& geom_line(datamunge::graphics::RGB color = {37, 99, 235}, double width = 1.5);
  GGPlot& geom_bar(datamunge::graphics::RGB color = {37, 99, 235});
  GGPlot& geom_col(datamunge::graphics::RGB color = {37, 99, 235});
  GGPlot& geom_histogram(std::size_t bins = 30, datamunge::graphics::RGB color = {96, 165, 250});
  GGPlot& geom_boxplot(datamunge::graphics::RGB color = {96, 165, 250});
  GGPlot& geom_smooth(datamunge::graphics::RGB color = {220, 38, 38});
  GGPlot& geom_area(datamunge::graphics::RGB color = {96, 165, 250});
  GGPlot& geom_ribbon(const std::string& ymin_column, const std::string& ymax_column,
                      datamunge::graphics::RGB color = {96, 165, 250});
  GGPlot& geom_density(datamunge::graphics::RGB color = {37, 99, 235});

  GGPlot& facet_wrap(const std::string& column, std::size_t ncol = 0);
  GGPlot& theme_minimal();
  GGPlot& theme_bw();
  GGPlot& theme_classic();
  GGPlot& scale_color_manual(const std::vector<datamunge::graphics::RGB>& values);
  GGPlot& labs(const std::string& title = "", const std::string& x = "", const std::string& y = "");

  /// @brief Renders the plot as a standalone SVG document and returns it as a string.
  [[nodiscard]] std::string to_svg() const;
  void save(const std::string& path) const;
  void save_svg(const std::string& path) const;
  void show(const std::string& title_hint = "") const;

  /// @brief Number of data rows being plotted.
  /// @return The row count of the frame passed to the constructor.
  [[nodiscard]] int rows() const { return static_cast<int>(impl_.data().nrows()); }
  /// @brief The column mapped to x.
  /// @return The x column name.
  [[nodiscard]] std::string x_column() const { return impl_.mapping().x; }
  /// @brief The column mapped to y.
  /// @return The y column name; empty when unmapped.
  [[nodiscard]] std::string y_column() const { return impl_.mapping().y; }
  /// @brief The discrete color grouping column.
  /// @return The color column name; empty when unmapped.
  [[nodiscard]] std::string color_column() const { return impl_.mapping().color; }
  /// @brief Number of geom layers added so far.
  /// @return The layer count.
  [[nodiscard]] int layer_count() const { return static_cast<int>(impl_.layer_count()); }
  /// @brief The geom of each layer in drawing order ("point", "line", "smooth", ...).
  /// @return One name per layer.
  [[nodiscard]] std::vector<std::string> layer_kinds() const { return impl_.layer_kinds(); }
  /// @brief The selected theme's name.
  /// @return "gray" (the default), "minimal", "bw" or "classic".
  [[nodiscard]] std::string theme_name() const {
    switch (impl_.theme()) {
      case datamunge::graphics::GGTheme::Minimal: return "minimal";
      case datamunge::graphics::GGTheme::BW: return "bw";
      case datamunge::graphics::GGTheme::Classic: return "classic";
      case datamunge::graphics::GGTheme::Gray: break;
    }
    return "gray";
  }
  /// @brief The facet_wrap() column.
  /// @return The column name; empty when not faceted.
  [[nodiscard]] std::string facet_column() const { return impl_.facet_column(); }
  /// @brief The title set by labs().
  /// @return The title; empty when unset.
  [[nodiscard]] std::string title_text() const { return impl_.title_text(); }
  /// @brief The x-axis label set by labs().
  /// @return The label; empty when unset.
  [[nodiscard]] std::string x_label() const { return impl_.x_label(); }
  /// @brief The y-axis label set by labs().
  /// @return The label; empty when unset.
  [[nodiscard]] std::string y_label() const { return impl_.y_label(); }

 private:
  datamunge::graphics::GGPlot impl_;
};

/// @brief A scatterplot matrix (R's `pairs()`, GGally's `ggpairs()`) over the columns of a
///        data frame, with chainable setters; the SWIG-friendly facade for graphics::Splom.
///
/// Panel (row i, column j) puts variable j on x and variable i on y. The panel type follows
/// the two variables' kinds (numeric columns are quantitative, string columns categorical):
/// - diagonal: kernel density curves, one per group_by() group (numeric), or a stacked bar of
///   level counts (string), titled with the variable name;
/// - numeric vs numeric above the diagonal: the points with a local linear smooth of the mean and
///   a 95% prediction band, one per group; point colors follow set_col() or group_by();
/// - numeric vs numeric below the diagonal: bivariate kernel density contours, one set per group;
/// - string vs numeric: one violin per level with the median and interquartile range marked;
/// - string vs string: a mosaic shaded by chi-squared Pearson residuals, with its p-value.
///
/// Values are copied at construction, so the frame need not outlive the matrix. Rows with a
/// missing value in either variable are dropped from that panel only. Setters return
/// `Splom&` for chaining; nothing is drawn until to_svg(), save(), or show(). The default
/// figure is 960 by 960 pixels with blue `{37, 99, 235}` points of radius 2.5.
///
/// @par Example
/// @code
/// const auto iris = dm::datasets::iris();
/// const auto species = dm::factor(dm::c(iris.get_column<std::string>("Species").values()));
/// const auto palette = dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255},
///                            dm::RGB{220, 100, 255});
/// auto pairs = dm::Splom(iris).set_title("Iris Pairs").set_col(species, palette);
/// pairs.show();  // or pairs.save("iris_pairs.svg") without a display
/// @endcode
/// @see dm::splom(), dm::Plot, graphics::Splom
class Splom {
 public:
  /// @brief Builds the matrix from a facade DataFrame (the overload SWIG bindings use).
  ///
  /// @param data  Source frame; its values are copied.
  /// @param columns Columns to include in order; empty selects every numeric column plus every
  ///        string column with at most graphics::Splom::max_default_levels distinct values.
  /// @throws std::invalid_argument For a missing column, or fewer than two variables.
  /// @see Splom(const dstruct::DataFrame&, const std::vector<std::string>&)
  explicit Splom(const DataFrame& data, const std::vector<std::string>& columns = std::vector<std::string>());
#ifndef SWIG
  /// @brief Builds the matrix straight from a core frame, e.g. `dm::Splom(dm::datasets::iris())`.
  ///
  /// With @p columns empty, every numeric column is used, plus each string column that has
  /// between 1 and graphics::Splom::max_default_levels (12) distinct values; so iris gives a 5 by 5
  /// grid including Species. Naming columns fixes both the selection and the panel order.
  ///
  /// @param data  Source frame; its values are copied.
  /// @param columns  Columns to include, in panel order; empty selects them automatically.
  /// @throws std::invalid_argument For a missing column, or fewer than two variables.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// std::print("{}\n", dm::Splom(iris).variables());
  /// // ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"]
  /// auto petals = dm::Splom(iris, {"Petal.Length", "Petal.Width", "Species"});
  /// @endcode
  explicit Splom(const dstruct::DataFrame& data, const std::vector<std::string>& columns = std::vector<std::string>());
  /// @brief R's `col = palette[f]`: colors scatter points by @p factor (one-based codes from
  ///        `dm::factor()`, one per row) using @p palette, e.g.
  ///        `dm::Splom(iris).set_col(dm::factor(species), palette)`. The categorical column the
  ///        factor came from fills its violins and stacked bar with the same palette.
  ///
  /// Row i takes `palette[(factor[i] - 1) % palette.size()]`; a code below 1 (missing) uses
  /// the single color(). Replaces any earlier group_by()/point_colors()/set_col() choice.
  ///
  /// @param factor  One level code per data row, as returned by `dm::factor()`.
  /// @param palette  Nonempty colors indexed by level code - 1, recycled when shorter.
  /// @return This matrix, for chaining.
  /// @throws std::invalid_argument If @p factor's length is not the row count or @p palette is empty.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto supp = dm::factor(dm::c(tooth.get_column<std::string>("supp").values()));
  /// auto pairs = dm::Splom(tooth)
  ///                  .set_title("ToothGrowth Pairs")
  ///                  .set_col(supp, dm::c(dm::RGB{100, 255, 255}, dm::RGB{220, 100, 255}));
  /// pairs.save("tooth_pairs.svg");
  /// @endcode
  /// @see group_by()
  Splom& set_col(const Vector<int>& factor, const Vector<datamunge::graphics::RGB>& palette);
  /// @brief Sets one scatter color per row (recycled), e.g. `set_col(palette.select(f))`.
  ///
  /// Row i takes `colors[i % colors.size()]`. Same as point_colors(): clears group_by()
  /// grouping, and an empty vector restores the single color(). Unlike the factor overload,
  /// violins and stacked bars keep their default level colors.
  ///
  /// @param colors  Per-row colors, recycled when shorter than the frame.
  /// @return This matrix, for chaining.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// const auto species = dm::factor(dm::c(iris.get_column<std::string>("Species").values()));
  /// const auto palette = dm::c(dm::RGB{230, 159, 0}, dm::RGB{86, 180, 233},
  ///                            dm::RGB{0, 158, 115});
  /// auto pairs = dm::Splom(iris).set_col(palette.select(species));
  /// @endcode
  Splom& set_col(const Vector<datamunge::graphics::RGB>& colors);
  /// @brief Sets one color for every scatter point; same as color().
  /// @param color  Point color, alpha included; also used for bars without grouping.
  /// @return This matrix, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto pairs = dm::Splom(dm::datasets::iris()).set_col(dm::RGBA{220, 38, 38, 0.5});
  /// @endcode
  Splom& set_col(datamunge::graphics::RGB color);
  /// @brief Same as title(), matching the `set_*` style of dm::Plot.
  /// @param value  Figure heading; empty removes the heading band.
  /// @return This matrix, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto pairs = dm::Splom(dm::datasets::iris()).set_title("Iris Pairs");
  /// @endcode
  Splom& set_title(const std::string& value);
#endif

  /// @brief Sets the heading above the grid; "" removes it.
  ///
  /// A nonempty title reserves a band above the panels (none by default). show() also uses
  /// it as the window title when no hint is passed.
  ///
  /// @param value  Figure heading.
  /// @return This matrix, for chaining.
  /// @see set_title()
  Splom& title(const std::string& value);
  /// @brief Sets the figure size in pixels (default 960 by 960); panels share it equally.
  /// @param width  Figure width in pixels.
  /// @param height  Figure height in pixels.
  /// @return This matrix, for chaining.
  /// @throws std::invalid_argument  If either dimension is zero.
  ///
  /// @par Example
  /// @code
  /// auto pairs = dm::Splom(dm::datasets::iris()).size(600, 600);
  /// pairs.save("small_pairs.png");
  /// @endcode
  Splom& size(std::size_t width, std::size_t height);
  /// @brief Uses one color for all points and clears any grouping.
  ///
  /// Clears group_by(), point_colors(), and the factor form of set_col().
  ///
  /// @param value  Point color (default `{37, 99, 235}`).
  /// @return This matrix, for chaining.
  /// @see set_col(datamunge::graphics::RGB)
  Splom& color(datamunge::graphics::RGB value);
  /// @brief Splits every numeric panel by the groups of string column @p column (GGally's
  ///        `ggpairs(aes(colour = group))`); an empty palette uses the default level palette.
  ///
  /// The column need not be a panel variable but must be a string column of the source
  /// frame. Groups are its sorted distinct non-missing values, and group k takes
  /// `palette[k % palette.size()]`. Each group gets its own points, diagonal density, smooth
  /// and prediction band, and density contours; rows with a missing group are left out of
  /// those panels. When the column is also a panel variable its violins, bars, and mosaic axis
  /// share the colors. Pass "" to clear the grouping. Clears point_colors().
  ///
  /// @param column  String column to group by, or "" to clear.
  /// @param palette  Group colors in sorted-group order; empty uses the default.
  /// @return This matrix, for chaining.
  /// @throws std::invalid_argument  If @p column is not a string column of the source frame.
  ///
  /// @par Example
  /// @code
  /// auto pairs = dm::Splom(dm::datasets::iris()).group_by("Species");
  /// std::print("{}\n", pairs.groups());  // ["setosa", "versicolor", "virginica"]
  /// @endcode
  /// @see groups(), set_col()
  Splom& group_by(const std::string& column,
                  const std::vector<datamunge::graphics::RGB>& palette = std::vector<datamunge::graphics::RGB>());
  /// @brief Sets one color per data row for the points (recycled); clears group_by().
  ///
  /// Same as set_col(const Vector<RGB>&) for callers holding a std::vector; an empty vector
  /// restores the single color().
  ///
  /// @param colors  Per-row colors, recycled when shorter than the frame.
  /// @return This matrix, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto pairs = dm::Splom(dm::datasets::iris())
  ///                  .point_colors({dm::RGB{0, 0, 0}, dm::RGB{255, 0, 0}});  // alternate rows
  /// @endcode
  Splom& point_colors(const std::vector<datamunge::graphics::RGB>& colors);
  /// @brief Sets the fills cycled over categorical levels in violins and stacked bars.
  ///
  /// An empty vector restores the default palette. group_by() and the factor form of
  /// set_col() override it for their own column. Scatter points are unaffected.
  ///
  /// @param colors  Fill colors in sorted-level order, recycled.
  /// @return This matrix, for chaining.
  Splom& level_palette(const std::vector<datamunge::graphics::RGB>& colors);
  /// @brief Sets the point radius in pixels (default 2.5).
  /// @param value  Positive marker radius.
  /// @return This matrix, for chaining.
  /// @throws std::invalid_argument  If @p value is not positive.
  Splom& marker_size(double value);
  /// @brief Shows only the variable names on the diagonal, like R's `pairs()`.
  /// @return This matrix, for chaining.
  /// @see diagonal_density(), diagonal_histogram()
  Splom& diagonal_names();
  /// @brief Shows kernel densities, one per group (numeric), or a stacked bar (string) on the
  ///        diagonal (the default).
  /// @return This matrix, for chaining.
  /// @see diagonal_histogram(), diagonal_names()
  Splom& diagonal_density();
  /// @brief Shows a histogram (numeric) or stacked bar (string) on the diagonal.
  /// @return This matrix, for chaining.
  /// @see diagonal_density(), bins()
  Splom& diagonal_histogram();
  /// @brief Sets the fraction of each group's points used in each local fit of the smooths
  ///        above the diagonal (R's `loess(span =)`, default 0.75).
  /// @param value  Span in (0, 1]; smaller values follow the data more closely.
  /// @return This matrix, for chaining.
  /// @throws std::invalid_argument  If @p value is not in (0, 1].
  ///
  /// @par Example
  /// @code
  /// auto pairs = dm::Splom(dm::datasets::iris()).group_by("Species").span(0.5).prediction_level(0.9);
  /// @endcode
  Splom& span(double value);
  /// @brief Sets the coverage of the prediction bands above the diagonal (default 0.95).
  /// @param value  Coverage in (0, 1).
  /// @return This matrix, for chaining.
  /// @throws std::invalid_argument  If @p value is not in (0, 1).
  /// @see span()
  Splom& prediction_level(double value);
  /// @brief Sets the histogram bin count used by diagonal_histogram() (default 10).
  /// @param value  Positive number of equal-width bins.
  /// @return This matrix, for chaining.
  /// @throws std::invalid_argument  If @p value is zero.
  Splom& bins(std::size_t value);
  /// @brief Toggles the light background grid of the scatter and violin panels (default on).
  /// @param enabled  True draws the grid.
  /// @return This matrix, for chaining.
  ///
  /// @par Example
  /// @code
  /// auto pairs = dm::Splom(dm::datasets::iris(), {"Sepal.Length", "Sepal.Width", "Species"})
  ///                  .marker_size(1.5)
  ///                  .bins(15)
  ///                  .show_grid(false);
  /// @endcode
  Splom& show_grid(bool enabled = true);

  /// @brief Variable names in panel order.
  /// @return A copy of the selected column names.
  [[nodiscard]] std::vector<std::string> variables() const;
  /// @brief Sorted group labels set by group_by(), in palette order; empty without grouping.
  /// @return A copy of the group labels; also empty after the factor form of set_col().
  [[nodiscard]] std::vector<std::string> groups() const;
  /// @brief The figure title set by set_title() or title().
  /// @return The title; empty when unset.
  [[nodiscard]] std::string title_text() const { return impl_.title_text(); }
  /// @brief Figure width in pixels.
  /// @return The width (960 by default).
  [[nodiscard]] int width() const { return static_cast<int>(impl_.width()); }
  /// @brief Figure height in pixels.
  /// @return The height (960 by default).
  [[nodiscard]] int height() const { return static_cast<int>(impl_.height()); }
  /// @brief Builds the panel with variable @p col on x and variable @p row on y.
  ///
  /// The panel is a standalone dm::Plot sized to one grid cell, useful for inspecting or
  /// saving a single pair. Indices are zero-based positions in variables().
  ///
  /// @param row  Zero-based index of the y variable.
  /// @param col  Zero-based index of the x variable; equal to @p row gives the diagonal panel.
  /// @return The rendered-on-demand panel plot.
  /// @throws std::out_of_range  If either index is not below variables().size().
  /// @throws std::invalid_argument  If the panel has no complete rows to draw.
  ///
  /// @par Example
  /// @code
  /// const auto pairs = dm::Splom(dm::datasets::iris(), {"Sepal.Length", "Sepal.Width"});
  /// pairs.panel(1, 0).save("sepal_scatter.svg");  // Sepal.Length on x, Sepal.Width on y
  /// @endcode
  [[nodiscard]] datamunge::graphics::RPlot panel(std::size_t row, std::size_t col) const;

  /// @brief Renders the matrix as a standalone SVG document and returns it as a string.
  /// @return The complete SVG text of the full grid, including the title band.
  ///
  /// @par Example
  /// @code
  /// const auto svg = dm::Splom(dm::datasets::iris()).set_title("Iris Pairs").to_svg();
  /// std::print("{}\n", svg.contains("Iris Pairs"));  // true
  /// @endcode
  [[nodiscard]] std::string to_svg() const;
  /// @brief Writes the matrix to @p path, choosing the format from the extension.
  ///
  /// `.svg` or no extension writes to_svg(); `.png` and `.pdf` write the tiled raster the
  /// viewer shows. Needs no display.
  ///
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::invalid_argument  For any other extension.
  /// @throws std::runtime_error  If the file cannot be written.
  ///
  /// @par Example
  /// @code
  /// const auto pairs = dm::Splom(dm::datasets::iris()).group_by("Species");
  /// pairs.save("iris_pairs.svg");
  /// pairs.save("iris_pairs.png");
  /// @endcode
  void save(const std::string& path) const;
  /// @brief Writes to_svg() to @p path regardless of its extension.
  /// @param path  Destination file; an existing file is overwritten.
  /// @throws std::runtime_error  If the file cannot be opened for writing.
  void save_svg(const std::string& path) const;
  /// @brief Opens the matrix in the interactive X11 viewer and blocks until the window closes.
  ///
  /// Zoom and pan act on every panel at once; q or Escape closes the window. Requires an
  /// X11-enabled build and a reachable display; use save() in headless code.
  ///
  /// @param title_hint  Window title; empty uses the figure title, or "datamunge splom".
  /// @throws std::runtime_error  If X11 is unavailable or the display cannot be opened.
  ///
  /// @par Example
  /// @code
  /// dm::Splom(dm::datasets::iris()).set_title("Iris Pairs").group_by("Species").show();
  /// @endcode
  void show(const std::string& title_hint = "") const;

 private:
  /// @brief The wrapped core matrix; every facade call forwards to it.
  datamunge::graphics::Splom impl_;
};

/// @brief SWIG-friendly facade for datamunge::stats::LM — R-`lm()`-style linear models fit from a DataFrame.
class LM {
 public:
  /// @param weights_column Optional column name enabling weighted least squares; pass "" (the default) for OLS.
  LM(const DataFrame& data, const std::string& formula, const std::string& weights_column = "");

  [[nodiscard]] std::string  formula_text() const;
  [[nodiscard]] bool         has_intercept() const;
  [[nodiscard]] std::size_t  observations() const;
  [[nodiscard]] std::size_t  rank() const;
  [[nodiscard]] std::size_t  degrees_of_freedom() const;

  [[nodiscard]] std::vector<double>      coefficients() const;
  [[nodiscard]] std::vector<std::string> coefficient_names() const;
  [[nodiscard]] std::vector<double>      fitted_values() const;
  [[nodiscard]] std::vector<double>      residuals() const;
  [[nodiscard]] std::vector<double>      standard_errors() const;
  [[nodiscard]] std::vector<double>      t_values() const;
  [[nodiscard]] std::vector<double>      p_values() const;

  [[nodiscard]] double r_squared() const;
  [[nodiscard]] double adjusted_r_squared() const;
  [[nodiscard]] double sigma() const;
  [[nodiscard]] double f_statistic() const;
  [[nodiscard]] double f_p_value() const;

  [[nodiscard]] std::vector<double> confidence_interval_lower(double level = 0.95) const;
  [[nodiscard]] std::vector<double> confidence_interval_upper(double level = 0.95) const;

  [[nodiscard]] std::vector<double> leverage() const;
  [[nodiscard]] std::vector<double> standardized_residuals() const;
  [[nodiscard]] std::vector<double> studentized_residuals() const;
  [[nodiscard]] std::vector<double> cooks_distance() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  /// @param interval_kind One of "none", "confidence", "prediction". Returned DataFrame has a "fit" column,
  ///                      plus "se_fit"/"lwr"/"upr" when an interval is requested.
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata, const std::string& interval_kind = "none",
                                        double level = 0.95) const;

  /// @brief Sequential (Type I) analysis-of-variance table as a DataFrame with columns
  ///        term/df/sum_sq/mean_sq/f_value/p_value.
  [[nodiscard]] DataFrame* anova() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_normal_qq() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_scale_location() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_leverage() const;

  /// @brief Saves all four diagnostic plots as "<path_prefix>_<name>.svg".
  void save_diagnostic_plots(const std::string& path_prefix) const;

 private:
  stats::LM lm_;
};

/// @brief SWIG-friendly facade for datamunge::stats::LMM — a linear mixed model fit by
///        (RE)ML, with a single grouping factor, e.g. "score ~ x1 + (1 + x1 | school)".
class LMM {
 public:
  /// @param reml REML (default) or maximum likelihood.
  /// @param theta_bound Box-constraint magnitude (relative to the residual SD) for the
  ///                     DifferentialEvolution search over variance-component parameters.
  LMM(const DataFrame& data, const std::string& formula, bool reml = true,
      std::size_t de_population_size = 40, std::size_t de_max_generations = 300, double theta_bound = 5.0,
      std::size_t seed = 42);

  [[nodiscard]] std::string formula_text() const;
  [[nodiscard]] std::string group_variable() const;
  [[nodiscard]] bool        has_random_intercept() const;
  [[nodiscard]] std::vector<std::string> random_effect_names() const;
  [[nodiscard]] bool         is_reml() const;
  [[nodiscard]] std::size_t  observations() const;
  [[nodiscard]] std::size_t  num_groups() const;
  [[nodiscard]] std::size_t  rank() const;

  [[nodiscard]] std::vector<double>      coefficients() const;
  [[nodiscard]] std::vector<std::string> coefficient_names() const;
  [[nodiscard]] std::vector<double>      standard_errors() const;
  [[nodiscard]] std::vector<double>      z_values() const;
  [[nodiscard]] std::vector<double>      p_values() const;
  [[nodiscard]] std::vector<double>      fitted_values() const;
  [[nodiscard]] std::vector<double>      residuals() const;

  [[nodiscard]] double residual_variance() const;
  [[nodiscard]] double residual_std_dev() const;
  [[nodiscard]] std::vector<double> random_effect_std_devs() const;
  [[nodiscard]] double              random_effect_correlation(std::size_t i, std::size_t j) const;

  [[nodiscard]] std::vector<std::string> group_labels() const;
  /// @brief The BLUP random-effect vector for the group at @p group_index (see group_labels()),
  ///        in the same order as random_effect_names().
  [[nodiscard]] std::vector<double> random_effects_for_group(std::size_t group_index) const;

  [[nodiscard]] double log_likelihood() const;
  [[nodiscard]] double deviance() const;
  [[nodiscard]] double aic() const;
  [[nodiscard]] double bic() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

 private:
  stats::LMM lmm_;
};

/// @brief SWIG-friendly facade for datamunge::stats::GLMM — a generalized linear mixed model
///        fit by penalized quasi-likelihood, with a single grouping factor, e.g.
///        "success ~ x1 + (1 | school)" with family="binomial".
class GLMM {
 public:
  /// @param family One of "binomial" (logit link) or "poisson" (log link).
  GLMM(const DataFrame& data, const std::string& formula, const std::string& family = "binomial",
      std::size_t max_iterations = 20, double tol = 1e-6, std::size_t de_population_size = 30,
      std::size_t de_max_generations = 150, double theta_bound = 5.0, std::size_t seed = 42);

  [[nodiscard]] std::string formula_text() const;
  [[nodiscard]] std::string family() const;
  [[nodiscard]] std::string group_variable() const;
  [[nodiscard]] bool        has_random_intercept() const;
  [[nodiscard]] std::vector<std::string> random_effect_names() const;
  [[nodiscard]] std::size_t  observations() const;
  [[nodiscard]] std::size_t  num_groups() const;
  [[nodiscard]] std::size_t  rank() const;
  [[nodiscard]] std::size_t  iterations() const;

  [[nodiscard]] std::vector<double>      coefficients() const;
  [[nodiscard]] std::vector<std::string> coefficient_names() const;
  [[nodiscard]] std::vector<double>      standard_errors() const;
  [[nodiscard]] std::vector<double>      z_values() const;
  [[nodiscard]] std::vector<double>      p_values() const;
  [[nodiscard]] std::vector<double>      fitted_values() const;

  [[nodiscard]] std::vector<double> random_effect_std_devs() const;
  [[nodiscard]] double              random_effect_correlation(std::size_t i, std::size_t j) const;

  [[nodiscard]] std::vector<std::string> group_labels() const;
  /// @brief The BLUP random-effect vector for the group at @p group_index (see group_labels()),
  ///        in the same order as random_effect_names().
  [[nodiscard]] std::vector<double> random_effects_for_group(std::size_t group_index) const;

  [[nodiscard]] double deviance() const;
  [[nodiscard]] double aic() const;
  [[nodiscard]] double bic() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

 private:
  stats::GLMM glmm_;
};

/// @brief SWIG-friendly facade for datamunge::stats::INLAMixedModel — a mixed model (same
///        formula grammar as GLMM/LMM) fit by Integrated Nested Laplace Approximation instead
///        of penalized quasi-likelihood/REML: a genuinely Bayesian alternative returning real
///        posterior means/sds (including for the variance components) rather than point
///        estimates + asymptotic standard errors. See datamunge::bayes::INLA (C++-only, not
///        exposed to bindings -- same reasoning as AutodiffModel) for the underlying algorithm.
class INLAMixedModel {
 public:
  /// @param family One of "gaussian" (identity link), "binomial" (logit link), or "poisson" (log link).
  /// @param strategy One of "grid" (integrate over hyperparameter uncertainty) or "eb"
  ///                 (empirical Bayes: fix hyperparameters at their posterior mode).
  INLAMixedModel(const DataFrame& data, const std::string& formula, const std::string& family = "gaussian",
      const std::string& strategy = "grid", double fixed_effect_prior_sd = 1000.0, std::size_t grid_points_per_dim = 7,
      double grid_span = 4.0, std::size_t mode_population_size = 40, std::size_t mode_max_generations = 200,
      std::size_t seed = 42);

  [[nodiscard]] std::string formula_text() const;
  [[nodiscard]] std::string family() const;
  [[nodiscard]] std::string group_variable() const;
  [[nodiscard]] std::vector<std::string> random_effect_names() const;
  [[nodiscard]] std::size_t  observations() const;
  [[nodiscard]] std::size_t  num_groups() const;

  [[nodiscard]] std::vector<double>      fixed_effects_mean() const;
  [[nodiscard]] std::vector<double>      fixed_effects_sd() const;
  [[nodiscard]] std::vector<std::string> coefficient_names() const;

  [[nodiscard]] std::vector<double> random_effect_std_devs() const;
  /// @brief Gaussian family only; throws for binomial/poisson (dispersion fixed at 1).
  [[nodiscard]] double residual_std_dev() const;

  [[nodiscard]] std::vector<std::string> group_labels() const;
  /// @brief The posterior-mean/sd BLUP-like random-effect vector for the group at
  ///        @p group_index (see group_labels()), in random_effect_names() order.
  [[nodiscard]] std::vector<double> random_effects_mean_for_group(std::size_t group_index) const;
  [[nodiscard]] std::vector<double> random_effects_sd_for_group(std::size_t group_index) const;

  /// @brief log p(y), approximated by the same INLA machinery used to fit the model.
  [[nodiscard]] double log_marginal_likelihood() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

 private:
  stats::INLAMixedModel model_;
};

/// @brief SWIG-friendly facade for datamunge::stats::LDA — R-`MASS::lda()`-style linear discriminant analysis.
class LDA {
 public:
  /// @param priors Optional class prior probabilities (must sum to 1, ordered as classes() once sorted
  ///               alphabetically); pass an empty vector (the default) to use observed class proportions.
  LDA(const DataFrame& data, const std::string& formula, const std::vector<double>& priors = {});

  [[nodiscard]] std::vector<std::string> classes() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              num_discriminants() const;
  [[nodiscard]] std::vector<double>      priors() const;

  /// @brief One row per class, one numeric column per predictor.
  [[nodiscard]] DataFrame* group_means() const;
  /// @brief One row per predictor, one numeric column per linear discriminant (LD1, LD2, ...).
  [[nodiscard]] DataFrame* scaling() const;
  [[nodiscard]] std::vector<double> proportion_of_trace() const;

  [[nodiscard]] double training_accuracy() const;
  /// @brief "actual" column plus one numeric column per class (counts), both ordered as classes().
  [[nodiscard]] DataFrame* confusion_matrix() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;
  /// @brief The fit as Markdown tables (priors, group means, coefficients, proportion of trace).
  [[nodiscard]] std::string to_md(int digits = 4) const;

  [[nodiscard]] std::vector<std::string> predict(const DataFrame& newdata) const;
  /// @brief "class" column, LD1/LD2/... discriminant scores, and posterior_<class> probability columns.
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata) const;

  /// @brief LD1/LD2 scatter; with @p decision_regions, class regions shaded and boundaries drawn.
  [[nodiscard]] datamunge::graphics::RPlot plot_discriminants(bool decision_regions = true) const;
  void                                       save_discriminant_plot(const std::string& path) const;
  /// @brief Decision regions in predictor space, for a model with exactly two predictors.
  [[nodiscard]] datamunge::graphics::RPlot plot_decision_regions() const;

#ifndef SWIG
  /// @brief Same figure as dm::plot() of the wrapped stats::LDA.
  friend graphics::Figure plot(const LDA& model) { return datamunge::plot(model.lda_); }
#endif

 private:
  stats::LDA lda_;
};

#ifndef SWIG
/// @brief dm::plot() for the LDA facade; see plot(const stats::LDA&).
/// @param model  The fitted model.
/// @return The figure.
graphics::Figure plot(const LDA& model);
#endif

/// @brief SWIG-friendly facade for datamunge::stats::SVM — R-`e1071::svm()`-style multi-class SVM classification.
class SVM {
 public:
  /// @param kernel One of "linear", "polynomial", "radial" (default), "sigmoid".
  /// @param gamma  <= 0 means "auto" = 1 / number of predictors.
  SVM(const DataFrame& data, const std::string& formula, const std::string& kernel = "radial", double cost = 1.0,
      double gamma = -1.0, double coef0 = 0.0, int degree = 3, bool scale = true);

  [[nodiscard]] std::vector<std::string> classes() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              num_support_vectors() const;

  [[nodiscard]] double     training_accuracy() const;
  /// @brief "actual" column plus one numeric column per class (counts), both ordered as classes().
  [[nodiscard]] DataFrame* confusion_matrix() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<std::string> predict(const DataFrame& newdata) const;
  /// @brief "class" column plus one numeric vote-count column per class (votes_<class>).
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata) const;

 private:
  stats::SVM svm_;
};

/// @brief SWIG-friendly facade for datamunge::stats::DecisionTreeClassifier — a CART-style classification tree.
class DecisionTreeClassifier {
 public:
  /// @param criterion "gini" (default) or "entropy".
  DecisionTreeClassifier(const DataFrame& data, const std::string& formula, std::size_t max_depth = 5,
                          std::size_t min_samples_split = 2, std::size_t min_samples_leaf = 1,
                          const std::string& criterion = "gini");

  [[nodiscard]] std::vector<std::string> classes() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              node_count() const;
  [[nodiscard]] std::size_t              leaf_count() const;
  [[nodiscard]] std::size_t              depth() const;
  [[nodiscard]] std::vector<double>      feature_importance() const;

  [[nodiscard]] double     training_accuracy() const;
  /// @brief "actual" column plus one numeric column per class (counts), both ordered as classes().
  [[nodiscard]] DataFrame* confusion_matrix() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<std::string> predict(const DataFrame& newdata) const;
  /// @brief "class" column plus one numeric probability column per class (prob_<class>).
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata) const;

  /// @brief Scatter of `data` in the (x_feature, y_feature) plane, colored by true class, with misclassified
  ///        points overlaid in a distinct marker.
  [[nodiscard]] datamunge::graphics::RPlot plot_classification(const DataFrame& data, const std::string& x_feature,
                                                                  const std::string& y_feature) const;
  /// @brief Background grid of predicted class regions plus training points; requires exactly 2 predictors.
  [[nodiscard]] datamunge::graphics::RPlot plot_decision_regions(const std::string& x_feature,
                                                                   const std::string& y_feature,
                                                                   std::size_t grid_resolution = 60) const;

 private:
  stats::DecisionTreeClassifier tree_;
};

/// @brief SWIG-friendly facade for datamunge::stats::DecisionTreeRegressor — a CART-style regression tree.
class DecisionTreeRegressor {
 public:
  DecisionTreeRegressor(const DataFrame& data, const std::string& formula, std::size_t max_depth = 5,
                         std::size_t min_samples_split = 2, std::size_t min_samples_leaf = 1);

  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              node_count() const;
  [[nodiscard]] std::size_t              leaf_count() const;
  [[nodiscard]] std::size_t              depth() const;
  [[nodiscard]] std::vector<double>      feature_importance() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;

 private:
  stats::DecisionTreeRegressor tree_;
};

/// @brief SWIG-friendly facade for datamunge::stats::RandomForestClassifier — a bagged ensemble of CART trees,
///        each fit on a bootstrap sample with a random subset of predictors considered at every split.
class RandomForestClassifier {
 public:
  /// @param criterion "gini" (default) or "entropy".
  /// @param max_features 0 = auto (floor(sqrt(number of predictors))).
  RandomForestClassifier(const DataFrame& data, const std::string& formula, std::size_t n_trees = 100,
                          std::size_t max_depth = 10, std::size_t min_samples_split = 2,
                          std::size_t min_samples_leaf = 1, std::size_t max_features = 0,
                          const std::string& criterion = "gini", bool bootstrap = true,
                          double sample_fraction = 1.0, std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> classes() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_trees() const;
  [[nodiscard]] std::size_t              max_features_used() const;
  [[nodiscard]] std::vector<double>      feature_importance() const;

  [[nodiscard]] double     training_accuracy() const;
  /// @brief Out-of-bag accuracy estimate (majority vote among trees that did not train on each row).
  [[nodiscard]] double     oob_accuracy() const;
  /// @brief "actual" column plus one numeric column per class (counts), both ordered as classes().
  [[nodiscard]] DataFrame* confusion_matrix() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<std::string> predict(const DataFrame& newdata) const;
  /// @brief "class" column plus one numeric vote-share column per class (votes_<class>).
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata) const;

  /// @brief Scatter of `data` in the (x_feature, y_feature) plane, colored by true class, with misclassified
  ///        points overlaid in a distinct marker.
  [[nodiscard]] datamunge::graphics::RPlot plot_classification(const DataFrame& data, const std::string& x_feature,
                                                                  const std::string& y_feature) const;
  /// @brief Background grid of majority-vote predicted class regions plus training points; requires exactly 2
  ///        predictors.
  [[nodiscard]] datamunge::graphics::RPlot plot_decision_regions(const std::string& x_feature,
                                                                   const std::string& y_feature,
                                                                   std::size_t grid_resolution = 60) const;

 private:
  stats::RandomForestClassifier forest_;
};

/// @brief SWIG-friendly facade for datamunge::stats::RandomForestRegressor — a bagged ensemble of CART regression
///        trees, each fit on a bootstrap sample with a random subset of predictors considered at every split.
class RandomForestRegressor {
 public:
  /// @param max_features 0 = auto (floor(number of predictors / 3)).
  RandomForestRegressor(const DataFrame& data, const std::string& formula, std::size_t n_trees = 100,
                         std::size_t max_depth = 10, std::size_t min_samples_split = 2,
                         std::size_t min_samples_leaf = 1, std::size_t max_features = 0, bool bootstrap = true,
                         double sample_fraction = 1.0, std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_trees() const;
  [[nodiscard]] std::size_t              max_features_used() const;
  [[nodiscard]] std::vector<double>      feature_importance() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;
  [[nodiscard]] double              oob_r_squared() const;
  [[nodiscard]] double              oob_rmse() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;

 private:
  stats::RandomForestRegressor forest_;
};

/// @brief SWIG-friendly facade for datamunge::stats::ElasticNet — regularized linear regression fit by
///        coordinate descent (alpha=0 is ridge, alpha=1 is lasso); see also the Ridge and Lasso convenience
///        facades below.
class ElasticNet {
 public:
  /// @param alpha L1/L2 mixing: 0 = ridge, 1 = lasso, in between = elastic net.
  /// @param lambda Regularization strength; pass a negative value (the default) to select it automatically via
  ///                cross-validation.
  ElasticNet(const DataFrame& data, const std::string& formula, double alpha = 0.5, double lambda = -1.0,
             std::size_t n_lambda = 100, std::size_t cv_folds = 5, bool standardize = true,
             std::uint64_t seed = 42);

  [[nodiscard]] std::string              formula_text() const;
  [[nodiscard]] bool                     has_intercept() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;

  [[nodiscard]] double alpha() const;
  /// @brief The lambda actually used to produce coefficients() -- either the caller-supplied value or the
  ///        cross-validation-selected one.
  [[nodiscard]] double lambda() const;
  [[nodiscard]] bool   lambda_was_selected() const;
  [[nodiscard]] std::vector<double> lambda_path() const;
  [[nodiscard]] std::vector<double> cv_mean_squared_error() const;

  [[nodiscard]] std::vector<double> coefficients() const;
  [[nodiscard]] double              intercept() const;
  [[nodiscard]] std::size_t         non_zero_coefficients() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] std::vector<double> residuals() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  /// @brief Coefficient trace (one series per predictor) across the lambda path; throws unless lambda was
  ///        auto-selected.
  [[nodiscard]] datamunge::graphics::RPlot plot_coefficient_path() const;
  /// @brief Cross-validated MSE across the lambda path with the selected lambda marked; throws unless lambda
  ///        was auto-selected.
  [[nodiscard]] datamunge::graphics::RPlot plot_cv_curve() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;

 private:
  stats::ElasticNet net_;
};

/// @brief SWIG-friendly facade for datamunge::stats::Ridge — pure L2-penalized ("ridge") regression, a special
///        case of ElasticNet with alpha fixed to 0. Shrinks coefficients toward zero without ever zeroing them.
class Ridge {
 public:
  Ridge(const DataFrame& data, const std::string& formula, double lambda = -1.0, std::size_t n_lambda = 100,
        std::size_t cv_folds = 5, bool standardize = true, std::uint64_t seed = 42);

  [[nodiscard]] std::string              formula_text() const;
  [[nodiscard]] bool                     has_intercept() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;

  [[nodiscard]] double lambda() const;
  [[nodiscard]] bool   lambda_was_selected() const;
  [[nodiscard]] std::vector<double> lambda_path() const;
  [[nodiscard]] std::vector<double> cv_mean_squared_error() const;

  [[nodiscard]] std::vector<double> coefficients() const;
  [[nodiscard]] double              intercept() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] std::vector<double> residuals() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_coefficient_path() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_cv_curve() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;

 private:
  stats::Ridge ridge_;
};

/// @brief SWIG-friendly facade for datamunge::stats::Lasso — pure L1-penalized ("lasso") regression, a special
///        case of ElasticNet with alpha fixed to 1. Can shrink coefficients exactly to zero, performing
///        variable selection.
class Lasso {
 public:
  Lasso(const DataFrame& data, const std::string& formula, double lambda = -1.0, std::size_t n_lambda = 100,
        std::size_t cv_folds = 5, bool standardize = true, std::uint64_t seed = 42);

  [[nodiscard]] std::string              formula_text() const;
  [[nodiscard]] bool                     has_intercept() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;

  [[nodiscard]] double lambda() const;
  [[nodiscard]] bool   lambda_was_selected() const;
  [[nodiscard]] std::vector<double> lambda_path() const;
  [[nodiscard]] std::vector<double> cv_mean_squared_error() const;

  [[nodiscard]] std::vector<double> coefficients() const;
  [[nodiscard]] double              intercept() const;
  [[nodiscard]] std::size_t         non_zero_coefficients() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] std::vector<double> residuals() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_coefficient_path() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_cv_curve() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;

 private:
  stats::Lasso lasso_;
};

/// @brief SWIG-friendly facade for datamunge::stats::KNNClassifier — k-nearest-neighbors classification.
///        Since a training point's nearest neighbor is always itself, training_accuracy()/confusion_matrix()
///        report leave-one-out performance rather than a trivial resubstitution fit.
class KNNClassifier {
 public:
  /// @param metric "euclidean" (default) or "manhattan".
  KNNClassifier(const DataFrame& data, const std::string& formula, std::size_t k = 5,
                const std::string& metric = "euclidean", bool weighted = false, bool standardize = true);

  [[nodiscard]] std::vector<std::string> classes() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              k() const;

  [[nodiscard]] double     training_accuracy() const;
  /// @brief "actual" column plus one numeric column per class (counts), both ordered as classes(); leave-one-out.
  [[nodiscard]] DataFrame* confusion_matrix() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<std::string> predict(const DataFrame& newdata) const;
  /// @brief "class" column plus one numeric vote-share column per class (votes_<class>).
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata) const;

  /// @brief Scatter of `data` in the (x_feature, y_feature) plane, colored by true class, with misclassified
  ///        points overlaid in a distinct marker.
  [[nodiscard]] datamunge::graphics::RPlot plot_classification(const DataFrame& data, const std::string& x_feature,
                                                                  const std::string& y_feature) const;
  /// @brief Background grid of predicted class regions plus training points; requires exactly 2 predictors.
  [[nodiscard]] datamunge::graphics::RPlot plot_decision_regions(const std::string& x_feature,
                                                                   const std::string& y_feature,
                                                                   std::size_t grid_resolution = 60) const;

 private:
  stats::KNNClassifier knn_;
};

/// @brief SWIG-friendly facade for datamunge::stats::KNNRegressor — k-nearest-neighbors regression. As with
///        KNNClassifier, fitted_values()/r_squared()/rmse() report leave-one-out performance.
class KNNRegressor {
 public:
  /// @param metric "euclidean" (default) or "manhattan".
  KNNRegressor(const DataFrame& data, const std::string& formula, std::size_t k = 5,
              const std::string& metric = "euclidean", bool weighted = false, bool standardize = true);

  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              k() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;

 private:
  stats::KNNRegressor knn_;
};

/// @brief SWIG-friendly facade for datamunge::stats::KMeans — k-means clustering (Lloyd's
///        algorithm with k-means++ initialization) fit from a DataFrame and a list of numeric
///        feature columns.
class KMeans {
 public:
  KMeans(const DataFrame& data, const std::vector<std::string>& feature_columns, std::size_t n_clusters = 8,
        std::size_t max_iterations = 300, std::size_t n_init = 10, double tolerance = 1e-4, std::size_t seed = 42);

  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- a workaround for SWIG-bound languages whose
  ///        overload resolution can't pass a real string vector to a constructor with other
  ///        default arguments (see DataFrame::add_string_column_encoded()).
  KMeans(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_clusters = 8,
        std::size_t max_iterations = 300, std::size_t n_init = 10, double tolerance = 1e-4, std::size_t seed = 42);

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              n_clusters() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              iterations_used() const;

  [[nodiscard]] std::vector<std::size_t> labels() const;
  [[nodiscard]] double                   inertia() const;
  /// @brief The feature vector of cluster @p cluster_index's center.
  [[nodiscard]] std::vector<double> cluster_center(std::size_t cluster_index) const;

  [[nodiscard]] std::vector<std::size_t> predict(const DataFrame& newdata) const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

 private:
  stats::KMeans kmeans_;
};

/// @brief SWIG-friendly facade for datamunge::stats::PCA — principal component analysis fit
///        from a DataFrame and a list of numeric feature columns via eigendecomposition of the
///        correlation (default) or covariance matrix.
class PCA {
 public:
  /// @param center Mean-center each feature before fitting; almost always left true.
  /// @param scale Standardize each feature to unit variance before fitting (i.e. fit on the
  ///              correlation matrix rather than the covariance matrix); recommended whenever
  ///              features are on different scales, and the default here.
  PCA(const DataFrame& data, const std::vector<std::string>& feature_columns, bool center = true, bool scale = true);

  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  PCA(const DataFrame& data, const std::string& encoded_feature_columns, bool center = true, bool scale = true);

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              num_components() const;
  /// @brief Row indices (into the DataFrame passed to the constructor) that survived
  ///        null-dropping, in fitted order -- use to align an external label vector for
  ///        plot_scores_grouped().
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> explained_variance() const;
  [[nodiscard]] std::vector<double> explained_variance_ratio() const;
  [[nodiscard]] std::vector<double> cumulative_explained_variance_ratio() const;

  [[nodiscard]] std::vector<double> component_loadings(std::size_t component_index) const;
  [[nodiscard]] std::vector<double> component_scores(std::size_t component_index) const;

  /// @brief The fitted training scores as a DataFrame ("PC1", "PC2", ... columns), one row per
  ///        kept observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
  [[nodiscard]] DataFrame* scores_frame() const;
  /// @brief Projects new data onto the already-fitted components, as a DataFrame with the same
  ///        "PC1", "PC2", ... columns as scores_frame().
  [[nodiscard]] DataFrame* transform(const DataFrame& newdata) const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  /// @brief Scatter of scores in the (component_x, component_y) plane, single series.
  [[nodiscard]] datamunge::graphics::RPlot plot_scores(std::size_t component_x = 0, std::size_t component_y = 1) const;
  /// @brief Same, colored by an external grouping vector (e.g. a withheld response column);
  ///        `group_labels` must have one entry per kept_row_indices() entry.
  [[nodiscard]] datamunge::graphics::RPlot plot_scores_grouped(const std::vector<std::string>& group_labels,
                                                            std::size_t component_x = 0,
                                                            std::size_t component_y = 1) const;
  /// @brief Scree plot: percent of variance explained by each component, as a bar chart.
  [[nodiscard]] datamunge::graphics::RPlot plot_scree() const;

 private:
  stats::PCA pca_;
};

/// @brief SWIG-friendly facade for datamunge::stats::MDS — classical (metric/Torgerson)
///        multidimensional scaling fit from a DataFrame and a list of numeric feature columns:
///        embeds the rows in a low-dimensional space that best reproduces their original
///        pairwise distances.
class MDS {
 public:
  /// @param metric One of "euclidean" (default) or "manhattan".
  MDS(const DataFrame& data, const std::vector<std::string>& feature_columns, std::size_t n_components = 2,
      const std::string& metric = "euclidean");

  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  MDS(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_components = 2,
      const std::string& metric = "euclidean");

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  /// @brief Row indices (into the DataFrame passed to the constructor) that survived
  ///        null-dropping, in fitted order -- use to align an external label vector for
  ///        plot_embedding_grouped().
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> eigenvalues() const;
  [[nodiscard]] double              goodness_of_fit() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  /// @brief The fitted embedding as a DataFrame ("Dim1", "Dim2", ... columns), one row per kept
  ///        observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
  [[nodiscard]] DataFrame* embedding_frame() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  /// @brief Scatter of the embedding in the (dimension_x, dimension_y) plane, single series.
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
  /// @brief Same, colored by an external grouping vector (e.g. a withheld response column);
  ///        `group_labels` must have one entry per kept_row_indices() entry.
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::MDS mds_;
};

/// @brief SWIG-friendly facade for datamunge::stats::Isomap — nonlinear manifold learning via
///        geodesic (shortest-path, over a k-nearest-neighbor graph) distances embedded with
///        classical MDS. Throws if the k-nearest-neighbor graph is disconnected.
class Isomap {
 public:
  /// @param metric One of "euclidean" (default) or "manhattan".
  Isomap(const DataFrame& data, const std::vector<std::string>& feature_columns, std::size_t n_components = 2,
         std::size_t n_neighbors = 10, const std::string& metric = "euclidean");
  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  Isomap(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_components = 2,
         std::size_t n_neighbors = 10, const std::string& metric = "euclidean");

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> eigenvalues() const;
  [[nodiscard]] double              goodness_of_fit() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  /// @brief The fitted embedding as a DataFrame ("Dim1", "Dim2", ... columns), one row per kept
  ///        observation -- ready to bind_cols()/plot with the rest of the DataFrame API.
  [[nodiscard]] DataFrame* embedding_frame() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0,
                                                       std::size_t dimension_y = 1) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::Isomap isomap_;
};

/// @brief SWIG-friendly facade for datamunge::stats::LLE — locally linear embedding: preserves
///        each point's local reconstruction weights from its k nearest neighbors rather than
///        global distances, letting it unfold nonlinear manifolds PCA/MDS cannot.
class LLE {
 public:
  /// @param metric One of "euclidean" (default) or "manhattan".
  LLE(const DataFrame& data, const std::vector<std::string>& feature_columns, std::size_t n_components = 2,
      std::size_t n_neighbors = 10, double regularization = 1e-3, const std::string& metric = "euclidean");
  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  LLE(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_components = 2,
      std::size_t n_neighbors = 10, double regularization = 1e-3, const std::string& metric = "euclidean");

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> eigenvalues() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  [[nodiscard]] DataFrame*          embedding_frame() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0,
                                                       std::size_t dimension_y = 1) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::LLE lle_;
};

/// @brief SWIG-friendly facade for datamunge::stats::TSNE — t-distributed Stochastic Neighbor
///        Embedding, a nonlinear method that preserves local neighborhood structure (via a
///        perplexity-calibrated probability distribution) primarily for 2D/3D visualization.
class TSNE {
 public:
  /// @param metric One of "euclidean" (default) or "manhattan".
  TSNE(const DataFrame& data, const std::vector<std::string>& feature_columns, std::size_t n_components = 2,
       double perplexity = 30.0, std::size_t max_iterations = 1000, double learning_rate = 200.0,
       double early_exaggeration = 12.0, std::size_t early_exaggeration_iterations = 250,
       double initial_momentum = 0.5, double final_momentum = 0.8, std::size_t momentum_switch_iteration = 250,
       const std::string& metric = "euclidean", std::uint64_t seed = 42);
  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  TSNE(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_components = 2,
       double perplexity = 30.0, std::size_t max_iterations = 1000, double learning_rate = 200.0,
       double early_exaggeration = 12.0, std::size_t early_exaggeration_iterations = 250,
       double initial_momentum = 0.5, double final_momentum = 0.8, std::size_t momentum_switch_iteration = 250,
       const std::string& metric = "euclidean", std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  /// @brief The achieved perplexity for each fitted point after the per-point binary search --
  ///        should be close to the constructor's perplexity argument if calibration converged.
  [[nodiscard]] std::vector<double> achieved_perplexity() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  [[nodiscard]] DataFrame*          embedding_frame() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0,
                                                       std::size_t dimension_y = 1) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::TSNE tsne_;
};

/// @brief SWIG-friendly facade for datamunge::stats::LaplacianEigenmaps — embeds points via the
///        smallest non-trivial eigenvectors of a heat-kernel-weighted, sparse k-nearest-neighbor
///        graph Laplacian. Distinct from DiffusionMaps (also heat-kernel-based, but dense and
///        alpha-normalized).
class LaplacianEigenmaps {
 public:
  LaplacianEigenmaps(const DataFrame& data, const std::vector<std::string>& feature_columns,
                     std::size_t n_components = 2, std::size_t n_neighbors = 10, double heat_kernel_t = 1.0);
  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  LaplacianEigenmaps(const DataFrame& data, const std::string& encoded_feature_columns,
                     std::size_t n_components = 2, std::size_t n_neighbors = 10, double heat_kernel_t = 1.0);

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> eigenvalues() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  [[nodiscard]] DataFrame*          embedding_frame() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0,
                                                       std::size_t dimension_y = 1) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::LaplacianEigenmaps laplacian_eigenmaps_;
};

/// @brief SWIG-friendly facade for datamunge::stats::DiffusionMaps — embeds points via the
///        leading eigenvectors of a dense, alpha-normalized heat-kernel Markov transition matrix,
///        scaled by eigenvalue^diffusion_time. Distinct from LaplacianEigenmaps (sparse k-NN
///        graph Laplacian, no density normalization).
class DiffusionMaps {
 public:
  DiffusionMaps(const DataFrame& data, const std::vector<std::string>& feature_columns,
               std::size_t n_components = 2, double heat_kernel_epsilon = 1.0, double alpha = 0.5,
               double diffusion_time = 1.0);
  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  DiffusionMaps(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_components = 2,
               double heat_kernel_epsilon = 1.0, double alpha = 0.5, double diffusion_time = 1.0);

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> eigenvalues() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  [[nodiscard]] DataFrame*          embedding_frame() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0,
                                                       std::size_t dimension_y = 1) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::DiffusionMaps diffusion_maps_;
};

/// @brief SWIG-friendly facade for datamunge::stats::KernelPCA — principal component analysis
///        generalized to a nonlinear feature space via the kernel trick. With a linear kernel it
///        reproduces plain unscaled PCA's scores (up to a sign flip per component).
class KernelPCA {
 public:
  /// @param kernel One of "linear", "rbf" (default), "polynomial".
  KernelPCA(const DataFrame& data, const std::vector<std::string>& feature_columns, std::size_t n_components = 2,
            const std::string& kernel = "rbf", double gamma = 1.0, double degree = 3.0, double coef0 = 1.0);
  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  KernelPCA(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_components = 2,
            const std::string& kernel = "rbf", double gamma = 1.0, double degree = 3.0, double coef0 = 1.0);

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> eigenvalues() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  [[nodiscard]] DataFrame*          embedding_frame() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0,
                                                       std::size_t dimension_y = 1) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::KernelPCA kernel_pca_;
};

/// @brief SWIG-friendly facade for datamunge::stats::SammonMapping — iteratively minimizes a
///        weighted distance-preservation "stress" (small high-dimensional distances weighted much
///        more heavily than large ones) via Sammon's original pseudo-Newton update.
class SammonMapping {
 public:
  /// @param metric One of "euclidean" (default) or "manhattan".
  SammonMapping(const DataFrame& data, const std::vector<std::string>& feature_columns,
               std::size_t n_components = 2, double learning_rate = 0.3, std::size_t max_iterations = 500,
               double tolerance = 1e-9, const std::string& metric = "euclidean", std::uint64_t seed = 42);
  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  SammonMapping(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_components = 2,
               double learning_rate = 0.3, std::size_t max_iterations = 500, double tolerance = 1e-9,
               const std::string& metric = "euclidean", std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  [[nodiscard]] DataFrame*          embedding_frame() const;

  /// @brief Final Sammon stress -- lower is better, 0 is a perfect distance-preserving embedding.
  [[nodiscard]] double      stress() const;
  [[nodiscard]] std::size_t iterations_run() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0,
                                                       std::size_t dimension_y = 1) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::SammonMapping sammon_mapping_;
};

/// @brief SWIG-friendly facade for datamunge::stats::UMAP — builds a fuzzy simplicial set (a
///        smoothly-calibrated, fuzzy-union-symmetrized k-nearest-neighbor graph) and optimizes a
///        low-dimensional embedding via cross-entropy minimization (attractive + negative-sampled
///        repulsive forces). A deliberately simplified but honest implementation -- see the real
///        class's own documentation for the two named simplifications.
class UMAP {
 public:
  /// @param metric One of "euclidean" (default) or "manhattan".
  UMAP(const DataFrame& data, const std::vector<std::string>& feature_columns, std::size_t n_components = 2,
       std::size_t n_neighbors = 15, double min_dist = 0.1, std::size_t max_iterations = 500,
       double learning_rate = 1.0, double negative_sample_rate = 5.0, const std::string& metric = "euclidean",
       std::uint64_t seed = 42);
  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  UMAP(const DataFrame& data, const std::string& encoded_feature_columns, std::size_t n_components = 2,
       std::size_t n_neighbors = 15, double min_dist = 0.1, std::size_t max_iterations = 500,
       double learning_rate = 1.0, double negative_sample_rate = 5.0, const std::string& metric = "euclidean",
       std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_components() const;
  [[nodiscard]] std::vector<std::size_t> kept_row_indices() const;

  [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
  [[nodiscard]] DataFrame*          embedding_frame() const;

  /// @brief Per-point sigma_i / rho_i from the smooth k-NN calibration -- mainly for diagnostics.
  [[nodiscard]] std::vector<double> sigmas() const;
  [[nodiscard]] std::vector<double> rhos() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] datamunge::graphics::RPlot plot_embedding(std::size_t dimension_x = 0,
                                                       std::size_t dimension_y = 1) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_embedding_grouped(const std::vector<std::string>& group_labels,
                                                               std::size_t dimension_x = 0,
                                                               std::size_t dimension_y = 1) const;

 private:
  stats::UMAP umap_;
};

/// @brief SWIG-friendly facade for datamunge::stats::AgglomerativeClustering — bottom-up
///        hierarchical clustering fit from a DataFrame and a list of numeric feature columns.
class AgglomerativeClustering {
 public:
  /// @param linkage One of "single", "complete", "average", "ward" (default).
  /// @param metric One of "euclidean" (default) or "manhattan"; ward linkage requires euclidean.
  AgglomerativeClustering(const DataFrame& data, const std::vector<std::string>& feature_columns,
                          std::size_t n_clusters = 2, const std::string& linkage = "ward",
                          const std::string& metric = "euclidean");

  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  AgglomerativeClustering(const DataFrame& data, const std::string& encoded_feature_columns,
                          std::size_t n_clusters = 2, const std::string& linkage = "ward",
                          const std::string& metric = "euclidean");

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::vector<std::size_t> labels() const;

  /// @brief Re-cuts the already-built dendrogram to produce @p n_clusters clusters, without refitting.
  [[nodiscard]] std::vector<std::size_t> cut(std::size_t n_clusters) const;

  // Merge history, as parallel flat arrays (length num_merges()) in merge order.
  [[nodiscard]] std::size_t              num_merges() const;
  [[nodiscard]] std::size_t              merge_cluster_a(std::size_t merge_index) const;
  [[nodiscard]] std::size_t              merge_cluster_b(std::size_t merge_index) const;
  [[nodiscard]] double                   merge_distance(std::size_t merge_index) const;
  [[nodiscard]] std::size_t              merge_size(std::size_t merge_index) const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

 private:
  stats::AgglomerativeClustering clustering_;
};

/// @brief SWIG-friendly facade for datamunge::stats::DBSCAN — density-based clustering fit
///        from a DataFrame and a list of numeric feature columns.
class DBSCAN {
 public:
  /// @param metric One of "euclidean" (default) or "manhattan".
  DBSCAN(const DataFrame& data, const std::vector<std::string>& feature_columns, double eps = 0.5,
        std::size_t min_samples = 5, const std::string& metric = "euclidean");

  /// @brief Same as the vector<string> constructor, but feature_columns is a single "<count>\x1e
  ///        col1\x1fcol2\x1f..."-encoded string -- see KMeans's encoded constructor for why.
  DBSCAN(const DataFrame& data, const std::string& encoded_feature_columns, double eps = 0.5,
        std::size_t min_samples = 5, const std::string& metric = "euclidean");

  [[nodiscard]] std::vector<std::string> feature_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_clusters() const;
  [[nodiscard]] std::size_t              n_noise() const;
  /// @brief Cluster index (0-based) assigned to each fitted row, or -1 for noise.
  [[nodiscard]] std::vector<int> labels() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

 private:
  stats::DBSCAN dbscan_;
};

/// @brief SWIG-friendly facade for datamunge::stats::GBMClassifier — multiclass gradient boosting (a sequence
///        of shallow trees, one per class per round, fit to the current multinomial-deviance gradient).
class GBMClassifier {
 public:
  GBMClassifier(const DataFrame& data, const std::string& formula, std::size_t n_trees = 100,
                double learning_rate = 0.1, std::size_t max_depth = 3, std::size_t min_samples_split = 2,
                std::size_t min_samples_leaf = 1, double subsample = 1.0, std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> classes() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_trees() const;
  [[nodiscard]] std::vector<double>      feature_importance() const;

  [[nodiscard]] double     training_accuracy() const;
  /// @brief "actual" column plus one numeric column per class (counts), both ordered as classes().
  [[nodiscard]] DataFrame* confusion_matrix() const;
  /// @brief Multinomial deviance on the training set after each boosting round (length n_trees()).
  [[nodiscard]] std::vector<double> training_deviance() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<std::string> predict(const DataFrame& newdata) const;
  /// @brief "class" column plus one numeric probability column per class (prob_<class>).
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_classification(const DataFrame& data, const std::string& x_feature,
                                                                  const std::string& y_feature) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_decision_regions(const std::string& x_feature,
                                                                   const std::string& y_feature,
                                                                   std::size_t grid_resolution = 60) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_training_deviance() const;

 private:
  stats::GBMClassifier gbm_;
};

/// @brief SWIG-friendly facade for datamunge::stats::GBMRegressor — a sequence of shallow regression trees,
///        each fit to the residuals of the current ensemble (gradient boosting on squared error).
class GBMRegressor {
 public:
  GBMRegressor(const DataFrame& data, const std::string& formula, std::size_t n_trees = 100,
              double learning_rate = 0.1, std::size_t max_depth = 3, std::size_t min_samples_split = 2,
              std::size_t min_samples_leaf = 1, double subsample = 1.0, std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_trees() const;
  [[nodiscard]] std::vector<double>      feature_importance() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;
  /// @brief Mean squared error on the training set after each boosting round (length n_trees()).
  [[nodiscard]] std::vector<double> training_deviance() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_training_deviance() const;

 private:
  stats::GBMRegressor gbm_;
};

/// @brief SWIG-friendly facade for datamunge::stats::XGBoostClassifier — regularized, second-order (gradient +
///        Hessian) multiclass gradient boosting, using the same regularized-gain tree-growing objective as the
///        XGBoost algorithm (L1/L2 leaf regularization plus a per-split complexity penalty).
class XGBoostClassifier {
 public:
  XGBoostClassifier(const DataFrame& data, const std::string& formula, std::size_t n_trees = 100,
                    double learning_rate = 0.3, std::size_t max_depth = 6, double lambda = 1.0, double alpha = 0.0,
                    double gamma = 0.0, double min_child_weight = 1.0, std::size_t min_samples_leaf = 1,
                    double subsample = 1.0, double colsample_bytree = 1.0, std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> classes() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_trees() const;
  /// @brief Gain-based importance (XGBoost's default "gain" metric), normalized to sum to 1.
  [[nodiscard]] std::vector<double>      feature_importance() const;

  [[nodiscard]] double     training_accuracy() const;
  /// @brief "actual" column plus one numeric column per class (counts), both ordered as classes().
  [[nodiscard]] DataFrame* confusion_matrix() const;
  /// @brief Multinomial deviance on the training set after each boosting round (length n_trees()).
  [[nodiscard]] std::vector<double> training_deviance() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<std::string> predict(const DataFrame& newdata) const;
  /// @brief "class" column plus one numeric probability column per class (prob_<class>).
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_classification(const DataFrame& data, const std::string& x_feature,
                                                                  const std::string& y_feature) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_decision_regions(const std::string& x_feature,
                                                                   const std::string& y_feature,
                                                                   std::size_t grid_resolution = 60) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_training_deviance() const;

 private:
  stats::XGBoostClassifier xgb_;
};

/// @brief SWIG-friendly facade for datamunge::stats::XGBoostRegressor — regularized, second-order gradient
///        boosting regression with the same tree-growing objective as the XGBoost algorithm.
class XGBoostRegressor {
 public:
  XGBoostRegressor(const DataFrame& data, const std::string& formula, std::size_t n_trees = 100,
                   double learning_rate = 0.3, std::size_t max_depth = 6, double lambda = 1.0, double alpha = 0.0,
                   double gamma = 0.0, double min_child_weight = 1.0, std::size_t min_samples_leaf = 1,
                   double subsample = 1.0, double colsample_bytree = 1.0, std::uint64_t seed = 42);

  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::size_t              n_trees() const;
  [[nodiscard]] std::vector<double>      feature_importance() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;
  [[nodiscard]] std::vector<double> training_deviance() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_training_deviance() const;

 private:
  stats::XGBoostRegressor xgb_;
};

/// @brief SWIG-friendly facade for datamunge::stats::KernelRegression — Nadaraya-Watson kernel regression, a
///        nonparametric fit where each prediction is a kernel-weighted average of training responses.
class KernelRegression {
 public:
  /// @param kernel "gaussian" (default), "epanechnikov", "uniform", or "triangular".
  /// @param bandwidth Bandwidth in standardized-predictor units; pass a negative value (the default) to select
  ///                   it automatically via leave-one-out cross-validation.
  KernelRegression(const DataFrame& data, const std::string& formula, const std::string& kernel = "gaussian",
                   double bandwidth = -1.0, std::size_t n_bandwidth = 50, bool standardize = true);

  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;

  /// @brief The bandwidth actually used -- either the caller-supplied value or the cross-validation-selected one.
  [[nodiscard]] double bandwidth() const;
  [[nodiscard]] bool   bandwidth_was_selected() const;
  [[nodiscard]] std::vector<double> bandwidth_grid() const;
  [[nodiscard]] std::vector<double> cv_mean_squared_error() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  /// @brief Scatter of `data` plus the fitted kernel-regression curve; only valid for a single-predictor model.
  [[nodiscard]] datamunge::graphics::RPlot plot_fit(const DataFrame& data, std::size_t grid_resolution = 200) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_cv_curve() const;

 private:
  stats::KernelRegression kernel_regression_;
};

/// @brief SWIG-friendly facade for datamunge::stats::GaussianProcessRegression — exact Gaussian process
///        regression with an RBF kernel, fit via Cholesky decomposition. Unlike every other regressor here,
///        predictions come with a principled posterior confidence interval; see predict_frame().
class GaussianProcessRegression {
 public:
  /// @param length_scale RBF kernel length scale in standardized-predictor units; pass a negative value (the
  ///                       default) to select it automatically by maximizing the log marginal likelihood.
  /// @param noise_ratio noise_variance / signal_variance; pass a negative value (the default) to select it
  ///                     automatically the same way (0 is a legal fixed value: a noiseless/interpolating GP).
  GaussianProcessRegression(const DataFrame& data, const std::string& formula, double length_scale = -1.0,
                            double noise_ratio = -1.0, std::size_t n_length_scale_grid = 20,
                            std::size_t n_noise_grid = 15, bool standardize = true);

  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;

  [[nodiscard]] double length_scale() const;
  [[nodiscard]] double signal_variance() const;
  [[nodiscard]] double noise_variance() const;
  [[nodiscard]] double log_marginal_likelihood() const;
  [[nodiscard]] bool   length_scale_was_selected() const;
  [[nodiscard]] bool   noise_ratio_was_selected() const;
  [[nodiscard]] std::vector<double> length_scale_grid() const;
  [[nodiscard]] std::vector<double> length_scale_profile_log_likelihood() const;

  [[nodiscard]] std::vector<double> fitted_values() const;
  [[nodiscard]] double              r_squared() const;
  [[nodiscard]] double              rmse() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;
  /// @param interval_kind One of "none" or "confidence". Returned DataFrame has a "fit" column, plus
  ///                      "se_fit"/"lwr"/"upr" when an interval is requested.
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata, const std::string& interval_kind = "none",
                                        double level = 0.95) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_fit(const DataFrame& data, std::size_t grid_resolution = 200,
                                                       double level = 0.95) const;
  [[nodiscard]] datamunge::graphics::RPlot plot_predicted_vs_actual() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_length_scale_profile() const;

 private:
  stats::GaussianProcessRegression gpr_;
};

/// @brief SWIG-friendly facade for datamunge::stats::NaiveBayesClassifier — a Naive Bayes classifier that
///        models numeric predictors with per-class Gaussians and categorical predictors with per-class
///        frequency tables (each categorical predictor's levels modeled jointly as one variable, not as
///        separate independent dummy features).
class NaiveBayesClassifier {
 public:
  NaiveBayesClassifier(const DataFrame& data, const std::string& formula, double laplace_smoothing = 1.0,
                       double var_smoothing = 1e-9);

  [[nodiscard]] std::vector<std::string> classes() const;
  [[nodiscard]] std::vector<std::string> predictor_names() const;
  [[nodiscard]] std::size_t              observations() const;
  [[nodiscard]] std::vector<double>      class_priors() const;

  [[nodiscard]] double     training_accuracy() const;
  /// @brief "actual" column plus one numeric column per class (counts), both ordered as classes().
  [[nodiscard]] DataFrame* confusion_matrix() const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  [[nodiscard]] std::vector<std::string> predict(const DataFrame& newdata) const;
  /// @brief "class" column plus one numeric probability column per class (prob_<class>).
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_classification(const DataFrame& data, const std::string& x_feature,
                                                                  const std::string& y_feature) const;
  /// @brief Background grid of predicted class regions; requires exactly two predictors, both numeric.
  [[nodiscard]] datamunge::graphics::RPlot plot_decision_regions(const std::string& x_feature,
                                                                   const std::string& y_feature,
                                                                   std::size_t grid_resolution = 60) const;

 private:
  stats::NaiveBayesClassifier nb_;
};

/// @brief SWIG-friendly facade for datamunge::stats::GLM — a generalized linear model (gaussian, binomial,
///        poisson, or Gamma family with its canonical link, matching R's glm() defaults) fit by iteratively
///        reweighted least squares.
class GLM {
 public:
  /// @param family One of "gaussian" (identity link), "binomial" (logit link), "poisson" (log link), or
  ///                "Gamma" (inverse link).
  /// @param weights_column Optional column name enabling prior weights; pass "" (the default) for none.
  GLM(const DataFrame& data, const std::string& formula, const std::string& family = "gaussian",
      const std::string& weights_column = "", std::size_t max_iter = 25, double tol = 1e-8);

  [[nodiscard]] std::string  formula_text() const;
  [[nodiscard]] std::string  family() const;
  [[nodiscard]] bool         has_intercept() const;
  [[nodiscard]] std::size_t  observations() const;
  [[nodiscard]] std::size_t  rank() const;
  [[nodiscard]] std::size_t  degrees_of_freedom() const;

  [[nodiscard]] std::vector<double>      coefficients() const;
  [[nodiscard]] std::vector<std::string> coefficient_names() const;
  [[nodiscard]] std::vector<double>      fitted_values() const;      // response scale
  [[nodiscard]] std::vector<double>      linear_predictors() const;
  [[nodiscard]] std::vector<double>      residuals() const;          // deviance residuals
  [[nodiscard]] std::vector<double>      pearson_residuals() const;
  [[nodiscard]] std::vector<double>      standardized_residuals() const;
  [[nodiscard]] std::vector<double>      leverage() const;
  [[nodiscard]] std::vector<double>      standard_errors() const;
  [[nodiscard]] std::vector<double>      test_statistics() const;
  [[nodiscard]] std::vector<double>      p_values() const;

  [[nodiscard]] double deviance() const;
  [[nodiscard]] double null_deviance() const;
  [[nodiscard]] double dispersion() const;
  [[nodiscard]] double aic() const;

  [[nodiscard]] std::vector<double> confidence_interval_lower(double level = 0.95) const;
  [[nodiscard]] std::vector<double> confidence_interval_upper(double level = 0.95) const;

  [[nodiscard]] std::string summary() const;
  void                      print_summary() const;

  /// @brief Response-scale predictions (back-transformed through the inverse link).
  [[nodiscard]] std::vector<double> predict(const DataFrame& newdata) const;

  /// @param interval_kind One of "none" or "confidence". Returned DataFrame has a "fit" column, plus
  ///                      "se_fit"/"lwr"/"upr" when an interval is requested.
  [[nodiscard]] DataFrame* predict_frame(const DataFrame& newdata, const std::string& interval_kind = "none",
                                        double level = 0.95) const;

  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_fitted() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_normal_qq() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_scale_location() const;
  [[nodiscard]] datamunge::graphics::RPlot plot_residuals_vs_leverage() const;
  void                                        save_diagnostic_plots(const std::string& path_prefix) const;

 private:
  stats::GLM glm_;
};

/// @brief SWIG-friendly facade for datamunge::linalg::Tensor -- a dense, row-major, N-dimensional
///        array that can hold float64, bool, or string elements (chosen at construction).
class Tensor {
 public:
  /// @param dtype One of "float64", "bool", or "string".
  explicit Tensor(const std::vector<std::size_t>& shape, const std::string& dtype = "float64");

  [[nodiscard]] static Tensor* zeros(const std::vector<std::size_t>& shape);
  [[nodiscard]] static Tensor* ones(const std::vector<std::size_t>& shape);
  [[nodiscard]] static Tensor* full(const std::vector<std::size_t>& shape, double value);
  [[nodiscard]] static Tensor* from_values(const std::vector<std::size_t>& shape, const std::vector<double>& values);
  [[nodiscard]] static Tensor* from_bool_values(const std::vector<std::size_t>& shape, const std::vector<int>& values);
  [[nodiscard]] static Tensor* from_string_values(const std::vector<std::size_t>& shape,
                                                  const std::vector<std::string>& values);
  [[nodiscard]] static Tensor* arange(double start, double stop, double step = 1.0);
  [[nodiscard]] static Tensor* eye(std::size_t n);

  [[nodiscard]] std::size_t              ndim() const;
  [[nodiscard]] std::vector<std::size_t> shape() const;
  [[nodiscard]] std::size_t              size() const;
  [[nodiscard]] std::string              dtype_name() const;

  [[nodiscard]] double      at(const std::vector<std::size_t>& index) const;
  void                      set(const std::vector<std::size_t>& index, double value);
  [[nodiscard]] std::string string_at(const std::vector<std::size_t>& index) const;
  void                      set_string(const std::vector<std::size_t>& index, const std::string& value);
  [[nodiscard]] double      at_flat(std::size_t i) const;
  void                      set_flat(std::size_t i, double value);
  [[nodiscard]] std::string string_at_flat(std::size_t i) const;
  void                      set_string_flat(std::size_t i, const std::string& value);

  [[nodiscard]] Tensor* reshape(const std::vector<std::size_t>& new_shape) const;
  [[nodiscard]] Tensor* flatten() const;
  [[nodiscard]] Tensor* transpose(const std::vector<std::size_t>& permutation = {}) const;
  [[nodiscard]] Tensor* squeeze() const;
  [[nodiscard]] Tensor* squeeze_axis(std::size_t axis) const;
  [[nodiscard]] Tensor* expand_dims(std::size_t axis) const;
  [[nodiscard]] Tensor* slice(std::size_t axis, std::size_t start, std::size_t stop, std::size_t step = 1) const;
  [[nodiscard]] Tensor* index_select(std::size_t axis, const std::vector<std::size_t>& indices) const;

  /// @brief Concatenates two tensors along an existing axis (shapes must match on every other axis).
  [[nodiscard]] static Tensor* concatenate2(const Tensor& a, const Tensor& b, std::size_t axis);
  /// @brief Stacks two same-shaped tensors along a new axis inserted at position @p axis.
  [[nodiscard]] static Tensor* stack2(const Tensor& a, const Tensor& b, std::size_t axis);

  [[nodiscard]] Tensor* add(const Tensor& other) const;
  [[nodiscard]] Tensor* subtract(const Tensor& other) const;
  [[nodiscard]] Tensor* multiply(const Tensor& other) const;
  [[nodiscard]] Tensor* divide(const Tensor& other) const;
  [[nodiscard]] Tensor* power(const Tensor& other) const;

  [[nodiscard]] Tensor* add_scalar(double scalar) const;
  [[nodiscard]] Tensor* subtract_scalar(double scalar) const;
  [[nodiscard]] Tensor* multiply_scalar(double scalar) const;
  [[nodiscard]] Tensor* divide_scalar(double scalar) const;
  [[nodiscard]] Tensor* power_scalar(double exponent) const;

  [[nodiscard]] Tensor* negate() const;
  [[nodiscard]] Tensor* abs() const;
  [[nodiscard]] Tensor* sqrt() const;
  [[nodiscard]] Tensor* exp() const;
  [[nodiscard]] Tensor* log() const;

  /// @brief Applies a user-supplied Callback elementwise. Requires a numeric dtype; result is float64.
  [[nodiscard]] Tensor* apply(Callback* callback) const;

  [[nodiscard]] Tensor* equal(const Tensor& other) const;
  [[nodiscard]] Tensor* not_equal(const Tensor& other) const;
  [[nodiscard]] Tensor* less(const Tensor& other) const;
  [[nodiscard]] Tensor* less_equal(const Tensor& other) const;
  [[nodiscard]] Tensor* greater(const Tensor& other) const;
  [[nodiscard]] Tensor* greater_equal(const Tensor& other) const;

  [[nodiscard]] double      sum() const;
  [[nodiscard]] double      mean() const;
  [[nodiscard]] double      max() const;
  [[nodiscard]] double      min() const;
  [[nodiscard]] double      prod() const;
  [[nodiscard]] std::size_t argmax() const;
  [[nodiscard]] std::size_t argmin() const;
  [[nodiscard]] bool        all() const;
  [[nodiscard]] bool        any() const;

  [[nodiscard]] Tensor* sum_axis(std::size_t axis, bool keepdims = false) const;
  [[nodiscard]] Tensor* mean_axis(std::size_t axis, bool keepdims = false) const;
  [[nodiscard]] Tensor* max_axis(std::size_t axis, bool keepdims = false) const;
  [[nodiscard]] Tensor* min_axis(std::size_t axis, bool keepdims = false) const;
  [[nodiscard]] Tensor* prod_axis(std::size_t axis, bool keepdims = false) const;
  [[nodiscard]] Tensor* argmax_axis(std::size_t axis, bool keepdims = false) const;
  [[nodiscard]] Tensor* argmin_axis(std::size_t axis, bool keepdims = false) const;

  [[nodiscard]] Tensor* matmul(const Tensor& other) const;
  [[nodiscard]] double  dot(const Tensor& other) const;
  [[nodiscard]] Tensor* outer(const Tensor& other) const;

  /// @brief The Tensor's own image_to_tensor() bridge: @p img (already directly SWIG-bindable,
  ///        no facade needed) as a [channels, height, width] tensor normalized to [0, 1].
  [[nodiscard]] static Tensor* from_image(const datamunge::image::Image& img);

  /// @brief Basic (inference-only) neural-network building blocks -- see
  ///        datamunge::cv::conv2d/max_pool2d/avg_pool2d/relu/sigmoid/softmax for the underlying
  ///        implementation and full documentation of shapes/semantics.
  [[nodiscard]] static Tensor* conv2d(const Tensor& input, const Tensor& kernel, const Tensor& bias, int stride = 1,
                                       int padding = 0);
  [[nodiscard]] Tensor* max_pool2d(int pool_size, int stride = -1) const;
  [[nodiscard]] Tensor* avg_pool2d(int pool_size, int stride = -1) const;
  [[nodiscard]] Tensor* relu() const;
  [[nodiscard]] Tensor* sigmoid() const;
  [[nodiscard]] Tensor* softmax() const;

  [[nodiscard]] std::string to_string(std::size_t max_elements = 100) const;

 private:
  explicit Tensor(linalg::Tensor tensor);

  linalg::Tensor tensor_;
};

/// @brief SWIG-friendly facade for datamunge::autodiff::Dual -- a first-order forward-mode
///        dual number. Build an expression out of named operations (starting from a seed
///        with derivative=1) to read off an exact derivative alongside the value.
class Dual {
 public:
  explicit Dual(double value, double derivative = 0.0);

  [[nodiscard]] double value() const;
  [[nodiscard]] double derivative() const;

  [[nodiscard]] Dual add(const Dual& other) const;
  [[nodiscard]] Dual subtract(const Dual& other) const;
  [[nodiscard]] Dual multiply(const Dual& other) const;
  [[nodiscard]] Dual divide(const Dual& other) const;
  [[nodiscard]] Dual negate() const;

  [[nodiscard]] Dual add_scalar(double scalar) const;
  [[nodiscard]] Dual subtract_scalar(double scalar) const;
  [[nodiscard]] Dual multiply_scalar(double scalar) const;
  [[nodiscard]] Dual divide_scalar(double scalar) const;

  [[nodiscard]] Dual pow(double exponent) const;
  [[nodiscard]] Dual exp() const;
  [[nodiscard]] Dual log() const;
  [[nodiscard]] Dual sqrt() const;
  [[nodiscard]] Dual sin() const;
  [[nodiscard]] Dual cos() const;
  [[nodiscard]] Dual tan() const;
  [[nodiscard]] Dual tanh() const;
  [[nodiscard]] Dual abs() const;

 private:
  explicit Dual(autodiff::Dual dual);

  autodiff::Dual dual_;
};

/// @brief SWIG-friendly facade for datamunge::autodiff::HyperDual -- a second-order
///        forward-mode dual number carrying two independent derivative directions plus
///        their exact mixed second partial, for computing Hessian entries.
class HyperDual {
 public:
  explicit HyperDual(double value, double eps1 = 0.0, double eps2 = 0.0, double eps1eps2 = 0.0);

  [[nodiscard]] double value() const;
  [[nodiscard]] double eps1() const;
  [[nodiscard]] double eps2() const;
  [[nodiscard]] double eps1eps2() const;

  [[nodiscard]] HyperDual add(const HyperDual& other) const;
  [[nodiscard]] HyperDual subtract(const HyperDual& other) const;
  [[nodiscard]] HyperDual multiply(const HyperDual& other) const;
  [[nodiscard]] HyperDual divide(const HyperDual& other) const;
  [[nodiscard]] HyperDual negate() const;

  [[nodiscard]] HyperDual add_scalar(double scalar) const;
  [[nodiscard]] HyperDual subtract_scalar(double scalar) const;
  [[nodiscard]] HyperDual multiply_scalar(double scalar) const;
  [[nodiscard]] HyperDual divide_scalar(double scalar) const;

  [[nodiscard]] HyperDual pow(double exponent) const;
  [[nodiscard]] HyperDual exp() const;
  [[nodiscard]] HyperDual log() const;
  [[nodiscard]] HyperDual sqrt() const;
  [[nodiscard]] HyperDual sin() const;
  [[nodiscard]] HyperDual cos() const;
  [[nodiscard]] HyperDual tan() const;
  [[nodiscard]] HyperDual tanh() const;

 private:
  explicit HyperDual(autodiff::HyperDual dual);

  autodiff::HyperDual dual_;
};

class Var;

/// @brief SWIG-friendly facade for datamunge::autodiff::Tape -- a reverse-mode
///        ("backpropagation") computation tape. Create leaf Vars against a Tape, build an
///        expression out of named Var operations, then call backward() on the output Var
///        to get the derivative of that output with respect to every node on the tape
///        (index a leaf's own index() into the result to read its gradient).
class Tape {
 public:
  Tape() = default;

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] double value_at(std::size_t index) const;
  [[nodiscard]] std::vector<double> backward(const Var& output) const;

 private:
  friend class Var;
  autodiff::Tape tape_;
};

/// @brief SWIG-friendly facade for datamunge::autodiff::Var -- a handle into a Tape.
class Var {
 public:
  /// @brief Records a new independent leaf variable on @p tape.
  explicit Var(Tape& tape, double value);

  [[nodiscard]] double      value() const;
  [[nodiscard]] std::size_t index() const;

  [[nodiscard]] Var add(const Var& other) const;
  [[nodiscard]] Var subtract(const Var& other) const;
  [[nodiscard]] Var multiply(const Var& other) const;
  [[nodiscard]] Var divide(const Var& other) const;
  [[nodiscard]] Var negate() const;

  [[nodiscard]] Var add_scalar(double scalar) const;
  [[nodiscard]] Var subtract_scalar(double scalar) const;
  [[nodiscard]] Var multiply_scalar(double scalar) const;
  [[nodiscard]] Var divide_scalar(double scalar) const;

  [[nodiscard]] Var pow(double exponent) const;
  [[nodiscard]] Var exp() const;
  [[nodiscard]] Var log() const;
  [[nodiscard]] Var sqrt() const;
  [[nodiscard]] Var sin() const;
  [[nodiscard]] Var cos() const;
  [[nodiscard]] Var tan() const;
  [[nodiscard]] Var tanh() const;
  [[nodiscard]] Var abs() const;

 private:
  friend class Tape;
  explicit Var(autodiff::Var var);

  autodiff::Var var_;
};

} // namespace datamunge
