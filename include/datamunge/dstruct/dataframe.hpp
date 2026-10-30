#pragma once

/// @file dataframe.hpp
/// @brief Column-oriented, null-aware DataFrame with dplyr/pandas-style verbs.
///
/// Defines dm::dstruct::NullableColumn (a typed vector plus validity mask),
/// dm::dstruct::DataFrame (named numeric and string columns of equal length with
/// select/filter/mutate/arrange/join/group_by/summarise operations), and
/// dm::dstruct::GroupedDataFrame, plus the free dm::dstruct::head()/tail() helpers. The
/// bundled datasets (dm::datasets::iris(), tooth_growth(), penguins()) return this type,
/// and pipe.hpp layers the `frame | dm::verb(...)` syntax on top. Frames have value
/// semantics: verbs return new frames, and copying a frame copies every column.
///
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// #include <print>
///
/// const auto tooth = dm::datasets::tooth_growth();
/// const auto high = tooth.filter([](const auto& row) { return row.get_double("dose") == 2; })
///                       .arrange({"len"}, {false});
/// std::print("{} {}\n", high.nrows(), high.double_at("len", 0));   // 20 33.9
/// @endcode
///
/// @par References
/// - H. Wickham (2014). "Tidy Data". Journal of Statistical Software 59(10).
///   doi:10.18637/jss.v059.i10
/// - dplyr reference: https://dplyr.tidyverse.org/reference/index.html
/// - pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
///
/// @see pipe.hpp, format.hpp, info.hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#ifdef DATAMUNGE_HAVE_ARROW
#include <memory>

#include <arrow/api.h>
#endif

namespace datamunge::dstruct {

/// @brief One typed DataFrame column: a contiguous `std::vector<T>` plus a validity mask, in the
///        style of an Arrow array or pandas' nullable dtypes.
///
/// DataFrame stores `NullableColumn<double>` (numeric) and `NullableColumn<std::string>`
/// (string) columns. A null slot holds a default-constructed `T` in values() and a 0 in
/// validity(); use is_null(), optional_at() or materialize() when nulls matter. Columns
/// borrowed from a DataFrame via DataFrame::get_column() are references into the frame.
///
/// @tparam T  Element type: `double` or `std::string` inside a DataFrame.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// const auto& len = tooth.get_column("len");   // const NullableColumn<double>&
/// std::print("{} {}\n", len.size(), len.values()[0]);   // 60 4.2
/// const auto lens = dm::c(len.values());   // copy into a dm::Vector<double>
/// @endcode
///
/// @par References
/// - Apache Arrow columnar format (validity bitmaps):
///   https://arrow.apache.org/docs/format/Columnar.html
/// - pandas nullable integer data type: https://pandas.pydata.org/docs/user_guide/integer_na.html
///
/// @see DataFrame::get_column()
template <typename T>
class NullableColumn {
 public:
  /// @brief Element type stored in the column (`double` or `std::string` in a DataFrame).
  using value_type = T;
  /// @brief Unsigned index and length type.
  using size_type = std::size_t;

  /// @brief Creates an empty column (size() == 0); grow it with push_back()/push_null().
  NullableColumn() = default;

  /// @brief Creates a column with no nulls by copying @p values.
  ///
  /// Every slot is valid. Pass a named `std::vector` (or an explicitly typed temporary):
  /// a bare braced list such as `{1.0, 2.0}` is ambiguous with the optional overload.
  ///
  /// @param values  Values in row order.
  ///
  /// @par Example
  /// @code
  /// const std::vector<double> xs{1.5, 2.5};
  /// const dm::dstruct::NullableColumn<double> col(xs);
  /// std::print("{} {}\n", col.size(), col.empty());   // 2 false
  /// @endcode
  /// @see NullableColumn(const std::vector<std::optional<T>>&)
  explicit NullableColumn(const std::vector<T>& values) : values_(values), valid_(values.size(), 1) {}

  /// @brief Creates a column with no nulls by moving @p values in (no element copies).
  /// @param values  Values in row order; left moved-from.
  explicit NullableColumn(std::vector<T>&& values) : values_(std::move(values)), valid_(values_.size(), 1) {}

  /// @brief Creates a column from optionals: `std::nullopt` entries become nulls.
  ///
  /// A null slot stores `T{}` as its placeholder value.
  ///
  /// @param values  Optional values in row order.
  ///
  /// @par Example
  /// @code
  /// const dm::dstruct::NullableColumn<double> col(
  ///     std::vector<std::optional<double>>{1.0, std::nullopt, 3.0});
  /// std::print("{} {}\n", col.is_null(1), col.values()[1]);   // true 0
  /// @endcode
  /// @see materialize()
  explicit NullableColumn(const std::vector<std::optional<T>>& values) { assign(values); }

  /// @brief Same as the const-reference optional overload; the values are copied out of
  ///        @p values, which is left unchanged.
  /// @param values  Optional values in row order.
  explicit NullableColumn(std::vector<std::optional<T>>&& values) { assign(values); }

  /// @brief Whether the column has no slots (nulls count as slots).
  /// @return  true when size() == 0.
  [[nodiscard]] bool empty() const { return values_.empty(); }

  /// @brief Number of slots, null or not.
  /// @return  The column length.
  [[nodiscard]] size_type size() const { return values_.size(); }

  /// @brief Whether slot @p index is null (R's `NA`).
  /// @param index  Zero-based slot index.
  /// @return  true if the slot is null.
  /// @throws std::out_of_range  if @p index >= size().
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// const auto& mass = penguins.get_column("body_mass_g");
  /// std::print("{} {}\n", mass.is_null(0), mass.is_null(3));   // false true
  /// @endcode
  /// @see optional_at(), validity(), DataFrame::is_null()
  [[nodiscard]] bool is_null(const size_type index) const {
    require_index(index);
    return valid_[index] == 0;
  }

  /// @brief Checked access to a non-null value.
  /// @param index  Zero-based slot index.
  /// @return  A reference to the value, valid until the column is modified or destroyed.
  /// @throws std::out_of_range  if @p index >= size().
  /// @throws std::invalid_argument  if the slot is null.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// std::print("{}\n", tooth.get_column("len").at(0));   // 4.2
  /// @endcode
  /// @see optional_at(), is_null()
  [[nodiscard]] const T& at(const size_type index) const {
    require_index(index);
    if (valid_[index] == 0) {
      throw std::invalid_argument("NullableColumn::at cannot access null value");
    }
    return values_[index];
  }

  /// @brief Null-safe access: a copy of the value, or `std::nullopt` for a null slot.
  /// @param index  Zero-based slot index.
  /// @return  The value, or `std::nullopt` if the slot is null.
  /// @throws std::out_of_range  if @p index >= size().
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// const auto mass = penguins.get_column("body_mass_g").optional_at(3);
  /// std::print("{}\n", mass.value_or(-1));   // -1
  /// @endcode
  /// @see at(), materialize()
  [[nodiscard]] std::optional<T> optional_at(const size_type index) const {
    require_index(index);
    if (valid_[index] == 0) {
      return std::nullopt;
    }
    return values_[index];
  }

  /// @brief Overwrites slot @p index with @p value and marks it valid (clearing any null).
  ///
  /// For a `std::string` column pass a `std::string`, not a string literal: a `const char*`
  /// argument is ambiguous with the `std::optional<T>` overload.
  ///
  /// @param index  Zero-based slot index.
  /// @param value  New value.
  /// @throws std::out_of_range  if @p index >= size().
  ///
  /// @par Example
  /// @code
  /// auto col = dm::dstruct::NullableColumn<double>(
  ///     std::vector<std::optional<double>>{1.0, std::nullopt});
  /// col.set(1, 2.0);
  /// std::print("{} {}\n", col.is_null(1), col.at(1));   // false 2
  /// @endcode
  /// @see set_null(), DataFrame::set_value()
  void set(const size_type index, const T& value) {
    require_index(index);
    values_[index] = value;
    valid_[index] = 1;
  }

  /// @brief Move overload of set(size_type, const T&).
  /// @param index  Zero-based slot index.
  /// @param value  New value, moved into the slot.
  /// @throws std::out_of_range  if @p index >= size().
  void set(const size_type index, T&& value) {
    require_index(index);
    values_[index] = std::move(value);
    valid_[index] = 1;
  }

  /// @brief Sets slot @p index from an optional: a value marks it valid, `std::nullopt`
  ///        makes it null (storing `T{}`).
  /// @param index  Zero-based slot index.
  /// @param value  New value or `std::nullopt`.
  /// @throws std::out_of_range  if @p index >= size().
  void set(const size_type index, const std::optional<T>& value) {
    require_index(index);
    if (value.has_value()) {
      values_[index] = *value;
      valid_[index] = 1;
    } else {
      values_[index] = T{};
      valid_[index] = 0;
    }
  }

  /// @brief Makes slot @p index null, resetting its stored value to `T{}`.
  /// @param index  Zero-based slot index.
  /// @throws std::out_of_range  if @p index >= size().
  ///
  /// @par Example
  /// @code
  /// auto tooth = dm::datasets::tooth_growth();
  /// tooth.get_column("len").set_null(0);
  /// std::print("{}\n", tooth.null_count("len"));   // 1
  /// @endcode
  /// @see set(), DataFrame::set_null()
  void set_null(const size_type index) {
    require_index(index);
    values_[index] = T{};
    valid_[index] = 0;
  }

  /// @brief Appends a valid value.
  ///
  /// Like `std::vector::push_back`, this may reallocate, invalidating references from
  /// values()/validity()/at(). For a `std::string` column pass a `std::string`, not a
  /// string literal (ambiguous with the optional overload). Growing a column that lives
  /// inside a DataFrame breaks the frame's equal-length invariant; use
  /// DataFrame::append_row() there instead.
  ///
  /// @param value  Value to append.
  ///
  /// @par Example
  /// @code
  /// dm::dstruct::NullableColumn<std::string> col;
  /// col.push_back(std::string("a"));
  /// col.push_null();
  /// std::print("{} {}\n", col.size(), col.is_null(1));   // 2 true
  /// @endcode
  /// @see push_null(), DataFrame::append_row()
  void push_back(const T& value) {
    values_.push_back(value);
    valid_.push_back(1);
  }

  /// @brief Move overload of push_back(const T&).
  /// @param value  Value to append, moved into the column.
  void push_back(T&& value) {
    values_.push_back(std::move(value));
    valid_.push_back(1);
  }

  /// @brief Appends an optional: a value is appended as valid, `std::nullopt` as a null.
  /// @param value  Value or `std::nullopt`.
  void push_back(const std::optional<T>& value) {
    if (value.has_value()) {
      values_.push_back(*value);
      valid_.push_back(1);
    } else {
      values_.push_back(T{});
      valid_.push_back(0);
    }
  }

  /// @brief Appends a null slot (stored value `T{}`); may reallocate like push_back().
  void push_null() {
    values_.push_back(T{});
    valid_.push_back(0);
  }

  /// @brief Borrows the raw values, including placeholder values in null slots.
  ///
  /// The vector has size() elements in row order. A null slot holds `T{}` (0.0 or ""), so
  /// check validity()/is_null() first when the column may contain nulls -- or use
  /// materialize() to get `std::optional`s. The reference is valid while this column lives
  /// and is not modified; wrap it as `dm::c(col.values())` to get an owning dm::Vector.
  ///
  /// @return  A read-only reference to the underlying value storage.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// const auto& species = iris.get_column<std::string>("Species").values();
  /// std::print("{}\n", species.front());   // setosa
  /// const auto widths = dm::c(iris.get_column("Sepal.Width").values());
  /// @endcode
  /// @see materialize(), validity(), DataFrame::get_column()
  [[nodiscard]] const std::vector<T>& values() const { return values_; }

  /// @brief Borrows the validity mask: one byte per slot, 1 = valid, 0 = null.
  ///
  /// This is a byte mask rather than Arrow's packed bitmap. The reference is valid while
  /// the column lives and is not modified.
  ///
  /// @return  A read-only reference to the mask, size() elements long.
  ///
  /// @par Example
  /// @code
  /// const dm::dstruct::NullableColumn<double> col(
  ///     std::vector<std::optional<double>>{1.0, std::nullopt});
  /// std::print("{} {}\n", col.validity()[0], col.validity()[1]);   // 1 0
  /// @endcode
  /// @see is_null(), values()
  [[nodiscard]] const std::vector<std::uint8_t>& validity() const { return valid_; }

  /// @brief Copies the column out as `std::optional`s, with `std::nullopt` for each null.
  ///
  /// O(size()) time and a full copy; use values() plus validity() to avoid the copy.
  ///
  /// @return  One optional per slot, in row order.
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// const auto sex = penguins.get_column<std::string>("sex").materialize();
  /// std::print("{} {}\n", sex[0].value(), sex[3].has_value());   // male false
  /// @endcode
  /// @see DataFrame::pull_numeric(), DataFrame::pull_string()
  [[nodiscard]] std::vector<std::optional<T>> materialize() const {
    std::vector<std::optional<T>> result;
    result.reserve(values_.size());
    for (size_type index = 0; index < values_.size(); ++index) {
      if (valid_[index] == 0) {
        result.emplace_back(std::nullopt);
      } else {
        result.emplace_back(values_[index]);
      }
    }
    return result;
  }

  /// @brief Copies the half-open slot range [@p begin_index, @p end_index), nulls included.
  /// @param begin_index  First slot to copy.
  /// @param end_index  One past the last slot to copy.
  /// @return  A new column of length `end_index - begin_index`.
  /// @throws std::out_of_range  if @p begin_index > @p end_index or @p end_index > size().
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto part = tooth.get_column("len").slice(10, 15);
  /// std::print("{} {}\n", part.size(), part.at(0));   // 5 16.5
  /// @endcode
  /// @see DataFrame::slice()
  [[nodiscard]] NullableColumn slice(const size_type begin_index, const size_type end_index) const {
    if (begin_index > end_index || end_index > size()) {
      throw std::out_of_range("NullableColumn::slice range out of bounds");
    }

    NullableColumn result;
    result.values_.assign(values_.begin() + static_cast<std::ptrdiff_t>(begin_index),
                          values_.begin() + static_cast<std::ptrdiff_t>(end_index));
    result.valid_.assign(valid_.begin() + static_cast<std::ptrdiff_t>(begin_index),
                         valid_.begin() + static_cast<std::ptrdiff_t>(end_index));
    return result;
  }

 private:
  /// @brief Appends @p values (to an empty column, from the constructors), turning each
  ///        `std::nullopt` into a null slot.
  /// @param values  Optional values in row order.
  void assign(const std::vector<std::optional<T>>& values) {
    values_.reserve(values.size());
    valid_.reserve(values.size());
    for (const auto& value : values) {
      if (value.has_value()) {
        values_.push_back(*value);
        valid_.push_back(1);
      } else {
        values_.push_back(T{});
        valid_.push_back(0);
      }
    }
  }

  /// @brief Bounds check shared by the indexed accessors and setters.
  /// @param index  Slot index to validate.
  /// @throws std::out_of_range  if @p index >= size().
  void require_index(const size_type index) const {
    if (index >= values_.size()) {
      throw std::out_of_range("NullableColumn index out of range");
    }
  }

  std::vector<T> values_;  ///< Values in row order; null slots hold `T{}`.

  std::vector<std::uint8_t> valid_;  ///< Validity mask parallel to values_ (1 = valid, 0 = null).
};

/// @brief Forward declaration -- see the full definition after DataFrame, since it holds a
///        DataFrame by value (see DataFrame::group_by()).
class GroupedDataFrame;

/// @brief Column-oriented table of named numeric (double) and string columns with nulls, in the
///        spirit of R's data.frame/tibble and pandas' DataFrame.
///
/// Every column is a NullableColumn of the same length (nrows()); column order is preserved.
/// Most operations are non-mutating and return a new frame (head(), filter(), mutate(),
/// arrange(), select(), join(), group_by() + summarise_each(), ...), so they chain either as
/// methods or through the dplyr-style `|` verbs in pipe.hpp. Copies are deep (value semantics).
/// `std::print("{}", frame)` prints a Markdown-style table (see format.hpp) with nulls as `NA`,
/// and dm::info(frame) gives a per-column summary.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// std::print("{}\n", tooth.head(3));   // DataFrame[3 x 3]: len, supp, dose
/// const auto summary = tooth
///   | dm::filter_rows([](const auto& row) { return row.get_double("dose") >= 1; })
///   | dm::group_by("supp", "dose")
///   | dm::summarise_each(dm::mean);   // 4 x 3
/// @endcode
///
/// @par References
/// - H. Wickham (2014). "Tidy Data". Journal of Statistical Software 59(10).
///   doi:10.18637/jss.v059.i10
/// - R data.frame: https://stat.ethz.ch/R-manual/R-devel/library/base/html/data.frame.html
/// - tibble: https://tibble.tidyverse.org/reference/tibble.html
/// - pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
///
/// @see GroupedDataFrame, NullableColumn, dm::datasets::iris()
class DataFrame {
 public:
  /// @brief Unsigned row/column count and index type.
  using size_type = std::size_t;
  /// @brief The null alternative of cell_type (R's `NA`); test with
  ///        `std::holds_alternative<null_type>(cell)`.
  using null_type = std::monostate;
  /// @brief Storage type of a numeric column.
  using numeric_column_type = NullableColumn<double>;
  /// @brief Storage type of a string column.
  using string_column_type = NullableColumn<std::string>;
  /// @brief One column of either kind; also the value type of the initializer-list constructor.
  using column_data_type = std::variant<numeric_column_type, string_column_type>;
  /// @brief One cell: null, a number, or a string (see value(), Row and append_row()).
  using cell_type = std::variant<null_type, double, std::string>;

  /// @brief Storage kind of a column, as reported by column_type().
  enum class ColumnType {
    Numeric,  ///< NullableColumn<double>.
    String    ///< NullableColumn<std::string>.
  };

  /// @brief Join flavour for join(), following dplyr's mutating and filtering joins.
  enum class JoinType {
    Inner,  ///< Rows with a key match on both sides.
    Left,   ///< All left rows; unmatched right columns are null.
    Right,  ///< All right rows; unmatched left columns are null.
    Full,   ///< All rows from both sides.
    Semi,   ///< Left rows that have a match, left columns only.
    Anti    ///< Left rows without a match, left columns only.
  };

  /// @brief Aggregation functions usable with summarise()/GroupedDataFrame::summarise().
  ///        Median/StdDev/Min/Max/Sum/Mean require a numeric column; Min/Max also work on
  ///        string columns (lexicographic); Count and NDistinct work on any column (Count
  ///        ignores AggSpec::column entirely).
  enum class AggFunc {
    Sum,        ///< Sum of the non-null values (null if the group has none).
    Mean,       ///< Arithmetic mean of the non-null values (null if none).
    Min,        ///< Smallest non-null value (lexicographic for strings).
    Max,        ///< Largest non-null value (lexicographic for strings).
    Median,     ///< Median of the non-null values; mean of the two middle values for even n.
    StdDev,     ///< Sample standard deviation (n - 1 denominator); 0 for a single value.
    Count,      ///< Number of rows in the group, nulls included.
    NDistinct   ///< Number of distinct values in the group; nulls count as one value.
  };

  /// @brief One aggregation to compute per group in summarise(): result_name defaults to
  ///        `column` (or "n" for Count when both are left empty).
  ///
  /// @par Example
  /// @code
  /// using DF = dm::dstruct::DataFrame;
  /// const auto means = dm::datasets::tooth_growth().summarise(
  ///     {"supp"}, {DF::AggSpec{"len", DF::AggFunc::Mean, "mean_len"}});   // 2 x 2
  /// @endcode
  /// @see summarise(), GroupedDataFrame::summarise()
  struct AggSpec {
    std::string column;  ///< Input column (ignored for AggFunc::Count).

    AggFunc func{AggFunc::Sum};  ///< Aggregation to apply; defaults to Sum.

    std::string result_name;  ///< Output column name; empty means use `column`.
  };

  /// @brief Formatting choices for to_md(), in the spirit of knitr::kable().
  ///
  /// Numbers are printed with at most `digits` decimals. Within a column every number gets the
  /// same number of decimals: the fewest (up to `digits`) that show each value exactly, so a
  /// column of counts prints as `50`, a column of 5.1, 4.9, ... prints with one decimal, and
  /// a column of means prints with `digits` decimals.
  ///
  /// @par Example
  /// @code
  /// DataFrame::MarkdownOptions opts;
  /// opts.digits = 2;
  /// opts.column_digits = {{"p.value", 4}};
  /// opts.scientific = {"eigenvalue"};
  /// std::print("{}\n", frame.to_md(opts));
  /// @endcode
  /// @see to_md()
  struct MarkdownOptions {
    int digits{3};  ///< Maximum decimals for numeric columns (0 or more).

    std::unordered_map<std::string, int> column_digits;  ///< Per-column overrides of `digits`.

    /// Numeric columns printed in scientific notation with `digits` (or the column's
    /// `column_digits`) decimals in the mantissa, e.g. `1.81e-08`: for columns spanning many
    /// orders of magnitude, such as eigenvalues or p-values.
    std::unordered_set<std::string> scientific;

    size_type max_rows{std::numeric_limits<size_type>::max()};  ///< Rows to print; the rest are cut.

    std::string na{"NA"};  ///< Text for null cells.
  };

  /// @brief A snapshot of one row: each column name mapped to its cell (null, double or
  ///        string).
  ///
  /// Rows are what filter()/dm::filter_rows() predicates and dm::mutate()/mutate_with() row
  /// callables receive; DataFrame::row(i) also returns one. A Row is an independent copy, so
  /// it stays valid after the frame changes. The typed getters must match the column's type:
  /// get_double() for numeric columns and get_string() for string columns.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto row = tooth.row(0);
  /// std::print("{} {} {}\n", row.get_double("len"), row.get_string("supp"),
  ///            row.get_double("dose"));   // 4.2 VC 0.5
  /// @endcode
  /// @see DataFrame::filter(), dm::filter_rows(), dm::mutate()
  struct Row {
    /// @brief Cells keyed by column name.
    std::unordered_map<std::string, cell_type> values;

    /// @brief Whether the row has a column named @p column_name.
    /// @param column_name  Column name.
    /// @return  true if the column exists (even if its cell is null).
    [[nodiscard]] bool contains(const std::string& column_name) const {
      return values.find(column_name) != values.end();
    }

    /// @brief The raw cell variant for @p column_name.
    /// @param column_name  Column name.
    /// @return  The cell: `null_type` (std::monostate), `double` or `std::string`.
    /// @throws std::out_of_range  if the row has no such column.
    [[nodiscard]] const cell_type& at(const std::string& column_name) const {
      const auto it = values.find(column_name);
      if (it == values.end()) {
        throw std::out_of_range("DataFrame::Row missing column: " + column_name);
      }
      return it->second;
    }

    /// @brief Whether the cell in @p column_name is null (R's `NA`).
    /// @param column_name  Column name.
    /// @return  true for a null cell.
    /// @throws std::out_of_range  if the row has no such column.
    [[nodiscard]] bool is_null(const std::string& column_name) const {
      return std::holds_alternative<null_type>(at(column_name));
    }

    /// @brief The value of a numeric cell.
    ///
    /// Use this inside filter/mutate callbacks on columns without nulls; for nullable
    /// columns (e.g. penguins' measurements) use get_optional_double() or check is_null()
    /// first.
    ///
    /// @param column_name  Name of a numeric column.
    /// @return  The cell's value.
    /// @throws std::out_of_range  if the row has no such column.
    /// @throws std::bad_variant_access  if the cell is null or the column is a string column.
    ///
    /// @par Example
    /// @code
    /// const auto tooth = dm::datasets::tooth_growth();
    /// const auto high_dose = tooth
    ///   | dm::filter_rows([](const auto& row) { return row.get_double("dose") >= 1; });
    /// std::print("{}\n", high_dose.nrows());   // 40
    /// @endcode
    /// @see get_optional_double(), get_string()
    [[nodiscard]] double get_double(const std::string& column_name) const { return std::get<double>(at(column_name)); }

    /// @brief The value of a string cell.
    /// @param column_name  Name of a string column.
    /// @return  A reference to the cell's string, valid while this Row lives.
    /// @throws std::out_of_range  if the row has no such column.
    /// @throws std::bad_variant_access  if the cell is null or the column is numeric.
    ///
    /// @par Example
    /// @code
    /// const auto oj = dm::datasets::tooth_growth()
    ///   | dm::filter_rows([](const auto& row) { return row.get_string("supp") == "OJ"; });
    /// @endcode
    /// @see get_optional_string(), get_double()
    [[nodiscard]] const std::string& get_string(const std::string& column_name) const {
      return std::get<std::string>(at(column_name));
    }

    /// @brief Null-safe get_double(): the numeric value, or `std::nullopt` for a null cell.
    /// @param column_name  Name of a numeric column.
    /// @return  The value, or `std::nullopt` if the cell is null.
    /// @throws std::out_of_range  if the row has no such column.
    /// @throws std::bad_variant_access  if the column is a string column.
    ///
    /// @par Example
    /// @code
    /// const auto heavy = dm::datasets::penguins() | dm::filter_rows([](const auto& row) {
    ///   return row.get_optional_double("body_mass_g").value_or(0) > 5000; });
    /// @endcode
    [[nodiscard]] std::optional<double> get_optional_double(const std::string& column_name) const {
      if (is_null(column_name)) {
        return std::nullopt;
      }
      return std::get<double>(at(column_name));
    }

    /// @brief Null-safe get_string(): the string value, or `std::nullopt` for a null cell.
    /// @param column_name  Name of a string column.
    /// @return  A copy of the value, or `std::nullopt` if the cell is null.
    /// @throws std::out_of_range  if the row has no such column.
    /// @throws std::bad_variant_access  if the column is numeric.
    [[nodiscard]] std::optional<std::string> get_optional_string(const std::string& column_name) const {
      if (is_null(column_name)) {
        return std::nullopt;
      }
      return std::get<std::string>(at(column_name));
    }
  };

  /// @brief Summary statistics of one numeric column, returned by describe_numeric().
  ///
  /// The statistics cover non-null values only. With no non-null values every statistic
  /// except null_count is 0.
  ///
  /// @par Example
  /// @code
  /// const auto stats = dm::datasets::penguins().describe_numeric("body_mass_g");
  /// std::print("{} {} {} {}\n", stats.count, stats.null_count, stats.min, stats.max);
  /// // 342 2 2700 6300
  /// @endcode
  /// @see describe_numeric(), dm::info()
  struct NumericSummary {
    size_type count{0};  ///< Number of non-null values.

    size_type null_count{0};  ///< Number of null values.

    double sum{0.0};  ///< Sum of the non-null values.

    double mean{0.0};  ///< Mean of the non-null values (0 when count == 0).

    double min{0.0};  ///< Smallest non-null value (0 when count == 0).

    double max{0.0};  ///< Largest non-null value (0 when count == 0).
  };

  /// @brief Creates an empty frame with no columns and no rows (`DataFrame[0 x 0]`).
  ///
  /// Add columns with add_column(); the first column fixes nrows().
  ///
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame frame;
  /// frame.add_column("x", std::vector<double>{1, 2, 3});
  /// std::print("{} {}\n", frame.nrows(), frame.ncols());   // 3 1
  /// @endcode
  /// @see add_column()
  DataFrame() = default;

  /// @brief Builds a frame from `{name, column}` pairs, in the order given (like R's
  ///        `data.frame(x = ..., y = ...)` or `tibble()`).
  ///
  /// Each column is copied in through add_column(), so the same rules apply.
  ///
  /// @param columns  Name and column pairs; wrap values as numeric_column_type(...) or
  ///                 string_column_type(...).
  /// @throws std::invalid_argument  if a name is empty or repeated, or the columns'
  ///         lengths differ.
  ///
  /// @par Example
  /// @code
  /// using DF = dm::dstruct::DataFrame;
  /// const DF frame{{"x", DF::numeric_column_type(std::vector<double>{3, 1, 2})},
  ///                {"g", DF::string_column_type(std::vector<std::string>{"b", "a", "b"})}};
  /// std::print("{} {}\n", frame.nrows(), frame.columns());   // 3 ["x", "g"]
  /// @endcode
  ///
  /// @par References
  /// - R data.frame: https://stat.ethz.ch/R-manual/R-devel/library/base/html/data.frame.html
  /// - tibble: https://tibble.tidyverse.org/reference/tibble.html
  ///
  /// @see add_column(), DataFrame()
  DataFrame(std::initializer_list<std::pair<const std::string, column_data_type>> columns) {
    for (const auto& column : columns) {
      std::visit(
          [&](const auto& values) {
            add_column(column.first, values);
          },
          column.second);
    }
  }

  /// @brief Whether the frame has no rows or no columns (pandas' `DataFrame.empty`).
  ///
  /// A frame with columns but zero rows (e.g. a filter() with no matches) is empty.
  ///
  /// @return  true if nrows() == 0 or ncols() == 0.
  [[nodiscard]] bool empty() const { return row_count_ == 0 || column_order_.empty(); }

  /// @brief Number of rows (R's `nrow()`).
  /// @return  The shared length of every column; 0 for a frame without columns.
  [[nodiscard]] size_type nrows() const { return row_count_; }

  /// @brief Number of columns (R's `ncol()`).
  /// @return  The column count.
  [[nodiscard]] size_type ncols() const { return column_order_.size(); }

  /// @brief Rows and columns as a pair, like R's `dim()` or pandas' `shape`.
  /// @return  `{nrows(), ncols()}`.
  ///
  /// @par Example
  /// @code
  /// const auto [rows, cols] = dm::datasets::iris().shape();
  /// std::print("{} {}\n", rows, cols);   // 150 5
  /// @endcode
  [[nodiscard]] std::pair<size_type, size_type> shape() const { return {row_count_, column_order_.size()}; }

  /// @brief Whether a column named @p column_name exists (case-sensitive, O(1) average).
  /// @param column_name  Column name.
  /// @return  true if the column exists.
  [[nodiscard]] bool has_column(const std::string& column_name) const {
    return columns_.find(column_name) != columns_.end();
  }

  /// @brief Column names in column order (R's `names()`).
  /// @return  A copy of the names.
  ///
  /// @par Example
  /// @code
  /// std::print("{}\n", dm::datasets::tooth_growth().columns());   // ["len", "supp", "dose"]
  /// @endcode
  /// @see select(), relocate()
  [[nodiscard]] std::vector<std::string> columns() const { return column_order_; }

  /// @brief Storage kind of a column.
  /// @param column_name  Column name.
  /// @return  ColumnType::Numeric or ColumnType::String.
  /// @throws std::out_of_range  if the column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// const bool text = iris.column_type("Species") == dm::dstruct::DataFrame::ColumnType::String;
  /// std::print("{}\n", text);   // true
  /// @endcode
  /// @see get_column(), dm::info()
  [[nodiscard]] ColumnType column_type(const std::string& column_name) const {
    const auto& data = column_data(column_name);
    if (std::holds_alternative<numeric_column_type>(data)) {
      return ColumnType::Numeric;
    }
    return ColumnType::String;
  }

  /// @brief Number of null cells in a column, like R's `sum(is.na(x))` or pandas'
  ///        `isna().sum()`.
  ///
  /// O(nrows()).
  ///
  /// @param column_name  Column name (numeric or string).
  /// @return  The count of null cells.
  /// @throws std::out_of_range  if the column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// std::print("{} {}\n", penguins.null_count("body_mass_g"), penguins.null_count("sex"));
  /// // 2 11
  /// @endcode
  ///
  /// @par References
  /// - pandas DataFrame.isna: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html
  ///
  /// @see is_null(), drop_nulls(), fill_null(), describe_numeric()
  [[nodiscard]] size_type null_count(const std::string& column_name) const {
    const auto& data = column_data(column_name);
    return std::visit(
        [](const auto& column) {
          size_type count = 0;
          for (size_type index = 0; index < column.size(); ++index) {
            if (column.is_null(index)) {
              ++count;
            }
          }
          return count;
        },
        data);
  }

  /// @brief Removes every column and row, leaving a `DataFrame[0 x 0]`.
  ///
  /// Invalidates all column references previously obtained from get_column() and friends.
  void clear() {
    column_order_.clear();
    columns_.clear();
    row_count_ = 0;
  }

  /// @brief Appends a new column in place (mutating; pandas' `df[name] = values` for a new
  ///        name). Use mutate() for a chainable, non-mutating upsert.
  ///
  /// The column goes last in column order. On a frame without columns, its length sets
  /// nrows(); otherwise it must equal nrows(). The column is copied (or moved, for rvalue
  /// overloads). Existing column references stay valid.
  ///
  /// @param column_name  New, non-empty column name.
  /// @param values  The column, with its nulls.
  /// @throws std::invalid_argument  if @p column_name is empty or already exists, or the
  ///         length differs from nrows() on a frame that already has columns.
  ///
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame frame;
  /// frame.add_column("x", std::vector<double>{1, 2, 3});
  /// frame.add_column("y", std::vector<std::optional<double>>{1, std::nullopt, 3});
  /// frame.add_column("g", std::vector<std::string>{"a", "b", "c"});
  /// std::print("{} {}\n", frame.columns(), frame.null_count("y"));   // ["x", "y", "g"] 1
  /// @endcode
  /// @see replace_column(), mutate(), remove_column()
  void add_column(const std::string& column_name, const numeric_column_type& values) { add_column_impl(column_name, values); }

  /// @brief Move overload of add_column() for a numeric NullableColumn.
  /// @param column_name  New, non-empty column name.
  /// @param values  The column, moved into the frame.
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, numeric_column_type&& values) {
    add_column_impl(column_name, std::move(values));
  }

  /// @brief add_column() for a string NullableColumn (copied).
  /// @param column_name  New, non-empty column name.
  /// @param values  The column, with its nulls.
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, const string_column_type& values) { add_column_impl(column_name, values); }

  /// @brief Move overload of add_column() for a string NullableColumn.
  /// @param column_name  New, non-empty column name.
  /// @param values  The column, moved into the frame.
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, string_column_type&& values) {
    add_column_impl(column_name, std::move(values));
  }

  /// @brief add_column() for plain doubles: a numeric column without nulls.
  /// @param column_name  New, non-empty column name.
  /// @param values  One value per row (copied).
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, const std::vector<double>& values) {
    add_column_impl(column_name, numeric_column_type(values));
  }

  /// @brief Move overload of add_column(const std::string&, const std::vector<double>&).
  /// @param column_name  New, non-empty column name.
  /// @param values  One value per row (moved).
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, std::vector<double>&& values) {
    add_column_impl(column_name, numeric_column_type(std::move(values)));
  }

  /// @brief add_column() for optional doubles: `std::nullopt` entries become nulls.
  /// @param column_name  New, non-empty column name.
  /// @param values  One optional value per row.
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, const std::vector<std::optional<double>>& values) {
    add_column_impl(column_name, numeric_column_type(values));
  }

  /// @brief Rvalue overload of the optional-double add_column(); values are copied out.
  /// @param column_name  New, non-empty column name.
  /// @param values  One optional value per row.
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, std::vector<std::optional<double>>&& values) {
    add_column_impl(column_name, numeric_column_type(std::move(values)));
  }

  /// @brief add_column() for strings: a string column without nulls.
  /// @param column_name  New, non-empty column name.
  /// @param values  One string per row (copied).
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, const std::vector<std::string>& values) {
    add_column_impl(column_name, string_column_type(values));
  }

  /// @brief Move overload of add_column(const std::string&, const std::vector<std::string>&).
  /// @param column_name  New, non-empty column name.
  /// @param values  One string per row (moved).
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, std::vector<std::string>&& values) {
    add_column_impl(column_name, string_column_type(std::move(values)));
  }

  /// @brief add_column() for optional strings: `std::nullopt` entries become nulls.
  /// @param column_name  New, non-empty column name.
  /// @param values  One optional string per row.
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, const std::vector<std::optional<std::string>>& values) {
    add_column_impl(column_name, string_column_type(values));
  }

  /// @brief Rvalue overload of the optional-string add_column(); values are copied out.
  /// @param column_name  New, non-empty column name.
  /// @param values  One optional string per row.
  /// @throws std::invalid_argument  on an empty or duplicate name or a length mismatch.
  void add_column(const std::string& column_name, std::vector<std::optional<std::string>>&& values) {
    add_column_impl(column_name, string_column_type(std::move(values)));
  }

  /// @brief Overwrites an existing column in place, keeping its position (mutating).
  ///
  /// The new column may change the stored type (numeric to string or back). Do not keep
  /// references from get_column() and friends across this call: they either see the new
  /// contents or dangle (if the type changed). Use mutate() for a non-mutating upsert.
  ///
  /// @param column_name  Existing column name.
  /// @param values  Replacement column, exactly nrows() long.
  /// @throws std::invalid_argument  if @p column_name is empty or the length differs from
  ///         nrows().
  /// @throws std::out_of_range  if the column does not exist.
  ///
  /// @par Example
  /// @code
  /// using DF = dm::dstruct::DataFrame;
  /// auto tooth = dm::datasets::tooth_growth();
  /// tooth.replace_column("dose", DF::numeric_column_type(std::vector<double>(60, 1.0)));
  /// std::print("{}\n", tooth.double_at("dose", 0));   // 1
  /// @endcode
  /// @see add_column(), mutate()
  void replace_column(const std::string& column_name, const numeric_column_type& values) {
    replace_column_impl(column_name, values);
  }

  /// @brief Move overload of replace_column() for a numeric column.
  /// @param column_name  Existing column name.
  /// @param values  Replacement column, moved in.
  /// @throws std::invalid_argument  on an empty name or a length mismatch.
  /// @throws std::out_of_range  if the column does not exist.
  void replace_column(const std::string& column_name, numeric_column_type&& values) {
    replace_column_impl(column_name, std::move(values));
  }

  /// @brief replace_column() with a string column (copied).
  /// @param column_name  Existing column name.
  /// @param values  Replacement column.
  /// @throws std::invalid_argument  on an empty name or a length mismatch.
  /// @throws std::out_of_range  if the column does not exist.
  void replace_column(const std::string& column_name, const string_column_type& values) {
    replace_column_impl(column_name, values);
  }

  /// @brief Move overload of replace_column() for a string column.
  /// @param column_name  Existing column name.
  /// @param values  Replacement column, moved in.
  /// @throws std::invalid_argument  on an empty name or a length mismatch.
  /// @throws std::out_of_range  if the column does not exist.
  void replace_column(const std::string& column_name, string_column_type&& values) {
    replace_column_impl(column_name, std::move(values));
  }

  /// @brief dplyr::mutate()-style upsert: adds `column_name` if absent, replaces it in place
  ///        if present.
  ///
  /// Unlike add_column()/replace_column(), it does not care whether the column exists and
  /// returns a new DataFrame (this frame is unchanged), so it composes into a chain:
  /// `df.mutate(...).filter(...)`. A new column is appended last; a replaced column keeps its
  /// position and takes the new values' type (a numeric column may become a string column).
  /// For a column computed from each row, use dm::mutate() with a row callable or
  /// mutate_with().
  ///
  /// @param column_name  Column to add or replace (must not be empty).
  /// @param values  One value per row.
  /// @return  A new DataFrame with the column added or replaced.
  /// @throws std::invalid_argument  if @p values.size() differs from nrows() (unless the frame
  ///         has no columns yet) or @p column_name is empty.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto& len = tooth.get_column("len").values();
  /// std::vector<double> log_len(len.size());
  /// std::ranges::transform(len, log_len.begin(), [](double v) { return std::log(v); });
  /// const auto logged = tooth.mutate("log_len", log_len);   // 60 x 4
  /// @endcode
  ///
  /// @par References
  /// - dplyr mutate: https://dplyr.tidyverse.org/reference/mutate.html
  ///
  /// @see dm::mutate(), mutate_with(), add_column(), replace_column()
  [[nodiscard]] DataFrame mutate(const std::string& column_name, const std::vector<double>& values) const {
    return mutate_impl(column_name, numeric_column_type(values));
  }

  /// @brief Same as mutate(const std::string&, const std::vector<double>&) but
  ///        `std::nullopt` entries become nulls.
  /// @param column_name  Column to add or replace.
  /// @param values  One optional value per row.
  /// @return  A new DataFrame with the numeric column added or replaced.
  /// @throws std::invalid_argument  on a length mismatch or empty name.
  [[nodiscard]] DataFrame mutate(const std::string& column_name, const std::vector<std::optional<double>>& values) const {
    return mutate_impl(column_name, numeric_column_type(values));
  }

  /// @brief Same as mutate(const std::string&, const std::vector<double>&) but adds or
  ///        replaces a string column.
  /// @param column_name  Column to add or replace.
  /// @param values  One string per row.
  /// @return  A new DataFrame with the string column added or replaced.
  /// @throws std::invalid_argument  on a length mismatch or empty name.
  [[nodiscard]] DataFrame mutate(const std::string& column_name, const std::vector<std::string>& values) const {
    return mutate_impl(column_name, string_column_type(values));
  }

  /// @brief Same as mutate(const std::string&, const std::vector<std::string>&) but
  ///        `std::nullopt` entries become nulls.
  /// @param column_name  Column to add or replace.
  /// @param values  One optional string per row.
  /// @return  A new DataFrame with the string column added or replaced.
  /// @throws std::invalid_argument  on a length mismatch or empty name.
  [[nodiscard]] DataFrame mutate(const std::string& column_name, const std::vector<std::optional<std::string>>& values) const {
    return mutate_impl(column_name, string_column_type(values));
  }

  /// @brief Row-wise mutate for C++ callers: `df.mutate_with("z", [](const Row& r){ return
  ///        r.get_double("x") + r.get_double("y"); })`. Not SWIG-exposable (arbitrary callable).
  ///
  /// Calls @p fn once per row, in row order, with a Row snapshot, and stores the results as a
  /// numeric column without nulls through mutate() (added last, or replacing an existing
  /// column in place). This frame is unchanged. Exceptions thrown by @p fn (for example
  /// std::bad_variant_access from Row::get_double() on a null cell) propagate.
  ///
  /// @tparam NumericFn  Callable type.
  /// @param column_name  Column to add or replace (must not be empty).
  /// @param fn  Callable taking `const Row&` and returning a value convertible to `double`.
  /// @return  A new DataFrame with the computed column.
  /// @throws std::invalid_argument  if @p column_name is empty.
  ///
  /// @par Example
  /// @code
  /// using DF = dm::dstruct::DataFrame;
  /// const auto ratio = dm::datasets::tooth_growth().mutate_with(
  ///     "len_per_dose", [](const DF::Row& r) { return r.get_double("len") / r.get_double("dose"); });
  /// std::print("{}\n", ratio.double_at("len_per_dose", 0));   // 8.4
  /// @endcode
  ///
  /// @par References
  /// - dplyr mutate: https://dplyr.tidyverse.org/reference/mutate.html
  ///
  /// @see mutate(), dm::mutate(), Row
  template <typename NumericFn>
  [[nodiscard]] DataFrame mutate_with(const std::string& column_name, NumericFn fn) const {
    std::vector<double> values;
    values.reserve(row_count_);
    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      values.push_back(fn(row(row_index)));
    }
    return mutate(column_name, std::move(values));
  }

  /// @brief Deletes a column in place (mutating), like pandas' `df.drop(columns=...)`.
  ///
  /// The other columns keep their order. Removing the last column resets nrows() to 0.
  /// References to the removed column dangle; references to other columns stay valid.
  ///
  /// @param column_name  Column to delete.
  /// @return  true if the column existed and was removed, false if there was no such column.
  ///
  /// @par Example
  /// @code
  /// auto tooth = dm::datasets::tooth_growth();
  /// const bool removed = tooth.remove_column("dose");
  /// std::print("{} {}\n", removed, tooth.columns());   // true ["len", "supp"]
  /// @endcode
  /// @see select(), add_column()
  bool remove_column(const std::string& column_name) {
    const auto it = columns_.find(column_name);
    if (it == columns_.end()) {
      return false;
    }

    columns_.erase(it);
    column_order_.erase(std::remove(column_order_.begin(), column_order_.end(), column_name), column_order_.end());
    if (column_order_.empty()) {
      row_count_ = 0;
    }
    return true;
  }

  /// @brief Renames one column in place (mutating), keeping its position and data.
  ///
  /// Renaming a column to its own name is a no-op. Column references stay valid.
  ///
  /// @param old_name  Existing column name.
  /// @param new_name  New, non-empty name not used by another column.
  /// @throws std::invalid_argument  if @p new_name is empty or names another existing column.
  /// @throws std::out_of_range  if @p old_name does not exist.
  ///
  /// @par Example
  /// @code
  /// auto tooth = dm::datasets::tooth_growth();
  /// tooth.rename_column("len", "length");
  /// std::print("{}\n", tooth.columns());   // ["length", "supp", "dose"]
  /// @endcode
  /// @see rename()
  void rename_column(const std::string& old_name, const std::string& new_name) {
    require_column_name(new_name);
    if (!has_column(old_name)) {
      throw std::out_of_range("DataFrame::rename_column missing column: " + old_name);
    }
    if (old_name != new_name && has_column(new_name)) {
      throw std::invalid_argument("DataFrame::rename_column duplicate column: " + new_name);
    }

    auto node = columns_.extract(old_name);
    node.key() = new_name;
    columns_.insert(std::move(node));

    for (auto& name : column_order_) {
      if (name == old_name) {
        name = new_name;
        break;
      }
    }
  }

  /// @brief Non-mutating, chainable rename: renames every (old_name, new_name) pair in order.
  ///
  /// Like dplyr::rename() but with the pair order `{old, new}` (dplyr writes `new = old`).
  /// Pairs apply one after another to a copy of this frame, so a later pair sees earlier
  /// renames (swapping two names therefore fails on the duplicate). Positions are kept.
  ///
  /// @param mapping  `{old_name, new_name}` pairs.
  /// @return  A renamed copy; this frame is unchanged.
  /// @throws std::invalid_argument  if a new name is empty or already in use.
  /// @throws std::out_of_range  if an old name does not exist (at that point in the sequence).
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto renamed = tooth.rename({{"len", "length"}, {"supp", "supplement"}});
  /// std::print("{}\n", renamed.columns());   // ["length", "supplement", "dose"]
  /// @endcode
  ///
  /// @par References
  /// - dplyr rename: https://dplyr.tidyverse.org/reference/rename.html
  ///
  /// @see rename_column()
  [[nodiscard]] DataFrame rename(const std::vector<std::pair<std::string, std::string>>& mapping) const {
    DataFrame result = select(column_order_);
    for (const auto& [old_name, new_name] : mapping) {
      result.rename_column(old_name, new_name);
    }
    return result;
  }

  /// @brief Single-pair convenience overload of rename(); see the vector<pair> overload for
  ///        renaming several columns at once.
  /// @param old_name  Existing column name.
  /// @param new_name  New, non-empty, unused name.
  /// @return  A renamed copy; this frame is unchanged.
  /// @throws std::invalid_argument  if @p new_name is empty or already in use.
  /// @throws std::out_of_range  if @p old_name does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto renamed = dm::datasets::tooth_growth().rename("len", "length");
  /// std::print("{}\n", renamed.columns());   // ["length", "supp", "dose"]
  /// @endcode
  [[nodiscard]] DataFrame rename(const std::string& old_name, const std::string& new_name) const {
    return rename(std::vector<std::pair<std::string, std::string>>{{old_name, new_name}});
  }

  /// @brief Borrows a read-only typed column; double (numeric) is the default element type.
  ///
  /// The reference shares this frame's storage. Removing the column or destroying
  /// the frame invalidates it; values() borrows the column's underlying vector.
  /// Calling it on a temporary (`dm::datasets::iris().get_column("x")`) therefore dangles --
  /// keep the frame in a variable first.
  ///
  /// @tparam T  double or std::string; use get_column<std::string>() for a text column.
  /// @param column_name  Column name.
  /// @return  A read-only NullableColumn<T> reference into this frame.
  /// @throws std::out_of_range  If the named column does not exist.
  /// @throws std::invalid_argument  If its stored type differs from T.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// const auto& length = iris.get_column("Sepal.Length");   // numeric (double)
  /// const auto species = dm::factor(dm::c(iris.get_column<std::string>("Species").values()));
  /// std::print("{}\n", dm::mean(dm::c(length.values())));   // 5.843333333333335
  /// @endcode
  /// @see NullableColumn::values(), numeric_column(), string_column(), pull()
  template <typename T = double>
  [[nodiscard]] const NullableColumn<T>& get_column(const std::string& column_name) const {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::string>,
                  "DataFrame columns contain double or std::string elements");
    return require_column_type<NullableColumn<T>>(
        column_name, std::is_same_v<T, double> ? "numeric" : "string");
  }

  /// @brief Borrows a mutable typed column; see the const overload for types, lifetime, and errors.
  ///
  /// Cell updates through set()/set_null() update this frame and its validity mask.
  /// Callers must preserve the frame's row count and synchronize concurrent mutation.
  ///
  /// @tparam T  double or std::string.
  /// @param column_name  Column name.
  /// @return  A mutable NullableColumn<T> reference into this frame.
  /// @throws std::out_of_range  If the named column does not exist.
  /// @throws std::invalid_argument  If its stored type differs from T.
  ///
  /// @par Example
  /// @code
  /// auto tooth = dm::datasets::tooth_growth();
  /// tooth.get_column("len").set_null(0);   // first len is now NA
  /// @endcode
  template <typename T = double>
  [[nodiscard]] NullableColumn<T>& get_column(const std::string& column_name) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::string>,
                  "DataFrame columns contain double or std::string elements");
    return require_column_type<NullableColumn<T>>(
        column_name, std::is_same_v<T, double> ? "numeric" : "string");
  }

  /// @brief Borrows a read-only numeric column; the non-template spelling of
  ///        `get_column<double>()`.
  ///
  /// Same lifetime rules as get_column(): the reference points into this frame.
  ///
  /// @param column_name  Name of a numeric column.
  /// @return  A read-only reference to the column.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a string column.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// std::print("{}\n", tooth.numeric_column("dose").values()[0]);   // 0.5
  /// @endcode
  /// @see get_column(), string_column(), pull_numeric()
  [[nodiscard]] const numeric_column_type& numeric_column(const std::string& column_name) const {
    return require_column_type<numeric_column_type>(column_name, "numeric");
  }

  /// @brief Borrows a mutable numeric column; edits through it change this frame.
  ///
  /// Keep the column's length unchanged (use set()/set_null(), not push_back()).
  ///
  /// @param column_name  Name of a numeric column.
  /// @return  A mutable reference to the column.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a string column.
  [[nodiscard]] numeric_column_type& numeric_column(const std::string& column_name) {
    return require_column_type<numeric_column_type>(column_name, "numeric");
  }

  /// @brief Borrows a read-only string column; the non-template spelling of
  ///        `get_column<std::string>()`.
  /// @param column_name  Name of a string column.
  /// @return  A read-only reference to the column, valid while the column exists.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a numeric column.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// std::print("{}\n", tooth.string_column("supp").at(0));   // VC
  /// @endcode
  /// @see get_column(), numeric_column(), pull_string()
  [[nodiscard]] const string_column_type& string_column(const std::string& column_name) const {
    return require_column_type<string_column_type>(column_name, "string");
  }

  /// @brief Borrows a mutable string column; edits through it change this frame.
  ///
  /// Keep the column's length unchanged (use set()/set_null(), not push_back()).
  ///
  /// @param column_name  Name of a string column.
  /// @return  A mutable reference to the column.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a numeric column.
  [[nodiscard]] string_column_type& string_column(const std::string& column_name) {
    return require_column_type<string_column_type>(column_name, "string");
  }

  /// @brief dplyr::pull()-style column extraction, null-aware (see materialize()).
  ///
  /// Returns an owning copy, so it is safe on temporaries (unlike get_column()).
  ///
  /// @param column_name  Name of a numeric column.
  /// @return  One optional per row; `std::nullopt` marks a null.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a string column.
  ///
  /// @par Example
  /// @code
  /// const auto mass = dm::datasets::penguins().pull_numeric("body_mass_g");
  /// std::print("{} {} {}\n", mass.size(), mass[0].value(), mass[3].has_value());
  /// // 344 3750 false
  /// @endcode
  ///
  /// @par References
  /// - dplyr pull: https://dplyr.tidyverse.org/reference/pull.html
  ///
  /// @see pull_string(), get_column(), NullableColumn::materialize()
  [[nodiscard]] std::vector<std::optional<double>> pull_numeric(const std::string& column_name) const {
    return numeric_column(column_name).materialize();
  }

  /// @brief String counterpart of pull_numeric(): an owning, null-aware copy of a column.
  /// @param column_name  Name of a string column.
  /// @return  One optional per row; `std::nullopt` marks a null.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a numeric column.
  ///
  /// @par Example
  /// @code
  /// const auto sex = dm::datasets::penguins().pull_string("sex");
  /// std::print("{}\n", sex[3].value_or("NA"));   // NA
  /// @endcode
  ///
  /// @par References
  /// - dplyr pull: https://dplyr.tidyverse.org/reference/pull.html
  ///
  /// @see pull_numeric()
  [[nodiscard]] std::vector<std::optional<std::string>> pull_string(const std::string& column_name) const {
    return string_column(column_name).materialize();
  }

  /// @brief Whether one cell is null (R's `is.na(df$col[i])`).
  /// @param column_name  Column name (numeric or string).
  /// @param row_index  Zero-based row index.
  /// @return  true if the cell is null.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// std::print("{}\n", penguins.is_null("body_mass_g", 3));   // true
  /// @endcode
  /// @see null_count(), value(), NullableColumn::is_null()
  [[nodiscard]] bool is_null(const std::string& column_name, const size_type row_index) const {
    const auto& data = column_data(column_name);
    return std::visit(
        [&](const auto& column) {
          if (row_index >= column.size()) {
            throw std::out_of_range("DataFrame::is_null row index out of range");
          }
          return column.is_null(row_index);
        },
        data);
  }

  /// @brief One cell as a cell_type variant (null, double or string), whatever the column type.
  ///
  /// Returns a copy. Use the typed accessors (double_at(), string_at(), ...) when the type is
  /// known.
  ///
  /// @param column_name  Column name.
  /// @param row_index  Zero-based row index.
  /// @return  The cell: `null_type`, `double` or `std::string`.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// const auto cell = penguins.value("body_mass_g", 3);
  /// const bool na = std::holds_alternative<dm::dstruct::DataFrame::null_type>(cell);
  /// std::print("{} {}\n", na, std::get<std::string>(penguins.value("species", 0)));
  /// // true Adelie
  /// @endcode
  /// @see row(), double_at(), string_at()
  [[nodiscard]] cell_type value(const std::string& column_name, const size_type row_index) const {
    const auto& data = column_data(column_name);
    return std::visit(
        [row_index](const auto& column) -> cell_type {
          if (row_index >= column.size()) {
            throw std::out_of_range("DataFrame::value row index out of range");
          }
          const auto value = column.optional_at(row_index);
          if (!value.has_value()) {
            return null_type{};
          }
          return *value;
        },
        data);
  }

  /// @brief One non-null numeric cell (R's `df$col[i]`, pandas' `df.at[i, col]`).
  /// @param column_name  Name of a numeric column.
  /// @param row_index  Zero-based row index.
  /// @return  The cell's value.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  /// @throws std::invalid_argument  if the column is a string column or the cell is null.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// std::print("{} {}\n", tooth.double_at("len", 0), tooth.string_at("supp", 0));   // 4.2 VC
  /// @endcode
  /// @see optional_double_at(), string_at(), value()
  [[nodiscard]] double double_at(const std::string& column_name, const size_type row_index) const {
    return numeric_column(column_name).at(row_index);
  }

  /// @brief Null-safe double_at(): the value, or `std::nullopt` for a null cell.
  /// @param column_name  Name of a numeric column.
  /// @param row_index  Zero-based row index.
  /// @return  The value or `std::nullopt`.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  /// @throws std::invalid_argument  if the column is a string column.
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// std::print("{}\n", penguins.optional_double_at("body_mass_g", 3).value_or(-1));   // -1
  /// @endcode
  /// @see double_at(), optional_string_at()
  [[nodiscard]] std::optional<double> optional_double_at(const std::string& column_name, const size_type row_index) const {
    return numeric_column(column_name).optional_at(row_index);
  }

  /// @brief One non-null string cell.
  /// @param column_name  Name of a string column.
  /// @param row_index  Zero-based row index.
  /// @return  A reference into the frame, valid until the column is modified or removed.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  /// @throws std::invalid_argument  if the column is numeric or the cell is null.
  ///
  /// @par Example
  /// See double_at().
  /// @see optional_string_at(), double_at()
  [[nodiscard]] const std::string& string_at(const std::string& column_name, const size_type row_index) const {
    return string_column(column_name).at(row_index);
  }

  /// @brief Null-safe string_at(): a copy of the value, or `std::nullopt` for a null cell.
  /// @param column_name  Name of a string column.
  /// @param row_index  Zero-based row index.
  /// @return  The value or `std::nullopt`.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  /// @throws std::invalid_argument  if the column is numeric.
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// std::print("{}\n", penguins.optional_string_at("sex", 3).value_or("NA"));   // NA
  /// @endcode
  /// @see string_at(), optional_double_at()
  [[nodiscard]] std::optional<std::string> optional_string_at(const std::string& column_name,
                                                              const size_type row_index) const {
    return string_column(column_name).optional_at(row_index);
  }

  /// @brief Overwrites one numeric cell in place (pandas' `df.at[i, col] = v`), clearing any
  ///        null.
  ///
  /// For a string column pass a `std::string`: a string literal argument is ambiguous
  /// between the `std::string` and `std::optional<std::string>` overloads.
  ///
  /// @param column_name  Name of a numeric column.
  /// @param row_index  Zero-based row index.
  /// @param value  New value.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  /// @throws std::invalid_argument  if the column is a string column.
  ///
  /// @par Example
  /// @code
  /// auto tooth = dm::datasets::tooth_growth();
  /// tooth.set_value("len", 0, 99.0);
  /// tooth.set_value("supp", 0, std::string("OJ"));
  /// tooth.set_value("dose", 1, std::optional<double>{});   // null
  /// std::print("{} {} {}\n", tooth.double_at("len", 0), tooth.string_at("supp", 0),
  ///            tooth.is_null("dose", 1));   // 99 OJ true
  /// @endcode
  /// @see set_null(), fill_null(), double_at()
  void set_value(const std::string& column_name, const size_type row_index, const double value) {
    numeric_column(column_name).set(row_index, value);
  }

  /// @brief Sets one numeric cell from an optional; `std::nullopt` makes it null.
  /// @param column_name  Name of a numeric column.
  /// @param row_index  Zero-based row index.
  /// @param value  New value or `std::nullopt`.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  /// @throws std::invalid_argument  if the column is a string column.
  void set_value(const std::string& column_name, const size_type row_index, std::optional<double> value) {
    numeric_column(column_name).set(row_index, value);
  }

  /// @brief Overwrites one string cell in place, clearing any null.
  /// @param column_name  Name of a string column.
  /// @param row_index  Zero-based row index.
  /// @param value  New value (moved into the cell).
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  /// @throws std::invalid_argument  if the column is numeric.
  void set_value(const std::string& column_name, const size_type row_index, std::string value) {
    string_column(column_name).set(row_index, std::move(value));
  }

  /// @brief Sets one string cell from an optional; `std::nullopt` makes it null.
  /// @param column_name  Name of a string column.
  /// @param row_index  Zero-based row index.
  /// @param value  New value or `std::nullopt`.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  /// @throws std::invalid_argument  if the column is numeric.
  void set_value(const std::string& column_name, const size_type row_index, std::optional<std::string> value) {
    string_column(column_name).set(row_index, value);
  }

  /// @brief Makes one cell null in place (R's `df$col[i] <- NA`), for either column type.
  /// @param column_name  Column name.
  /// @param row_index  Zero-based row index.
  /// @throws std::out_of_range  if the column does not exist or @p row_index >= nrows().
  ///
  /// @par Example
  /// @code
  /// auto tooth = dm::datasets::tooth_growth();
  /// tooth.set_null("supp", 0);
  /// std::print("{}\n", tooth.null_count("supp"));   // 1
  /// @endcode
  /// @see set_value(), fill_null(), is_null()
  void set_null(const std::string& column_name, const size_type row_index) {
    auto& data = column_data(column_name);
    std::visit(
        [row_index](auto& column) {
          column.set_null(row_index);
        },
        data);
  }

  /// @brief Replaces every null in a numeric column with @p value, in place (tidyr's
  ///        `replace_na()`, pandas' `fillna()`).
  ///
  /// Non-null cells are untouched. O(nrows()).
  ///
  /// @param column_name  Name of a numeric column.
  /// @param value  Fill value.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a string column.
  ///
  /// @par Example
  /// @code
  /// auto penguins = dm::datasets::penguins();
  /// penguins.fill_null("body_mass_g", 0.0);
  /// penguins.fill_null("sex", "unknown");
  /// std::print("{} {}\n", penguins.null_count("body_mass_g"), penguins.string_at("sex", 3));
  /// // 0 unknown
  /// @endcode
  ///
  /// @par References
  /// - tidyr replace_na: https://tidyr.tidyverse.org/reference/replace_na.html
  /// - pandas DataFrame.fillna:
  ///   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
  ///
  /// @see drop_nulls(), set_null(), null_count()
  void fill_null(const std::string& column_name, const double value) {
    auto& column = numeric_column(column_name);
    for (size_type index = 0; index < column.size(); ++index) {
      if (column.is_null(index)) {
        column.set(index, value);
      }
    }
  }

  /// @brief Replaces every null in a string column with @p value, in place.
  /// @param column_name  Name of a string column.
  /// @param value  Fill value.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a numeric column.
  void fill_null(const std::string& column_name, const std::string& value) {
    auto& column = string_column(column_name);
    for (size_type index = 0; index < column.size(); ++index) {
      if (column.is_null(index)) {
        column.set(index, value);
      }
    }
  }

  /// @brief Copies one row into a Row snapshot (pandas' `df.iloc[i]`).
  ///
  /// The Row owns its cells, so it stays valid after this frame changes. O(ncols()).
  ///
  /// @param row_index  Zero-based row index.
  /// @return  The row's cells keyed by column name.
  /// @throws std::out_of_range  if @p row_index >= nrows().
  ///
  /// @par Example
  /// @code
  /// const auto row = dm::datasets::tooth_growth().row(0);
  /// std::print("{} {}\n", row.values.size(), row.get_string("supp"));   // 3 VC
  /// @endcode
  /// @see Row, append_row(), value()
  [[nodiscard]] Row row(const size_type row_index) const {
    if (row_index >= row_count_) {
      throw std::out_of_range("DataFrame::row row index out of range");
    }

    Row result;
    result.values.reserve(column_order_.size());
    for (const auto& name : column_order_) {
      result.values.emplace(name, value(name, row_index));
    }
    return result;
  }

  /// @brief Appends one row in place (R's `rbind(df, row)` for a single row).
  ///
  /// The row must have exactly this frame's columns; each cell must be null or match its
  /// column's type. The frame must already have columns: to start from scratch, create
  /// (possibly empty) columns with add_column() first. Appending may reallocate column
  /// storage, invalidating references from values()/at(). Amortized O(ncols()).
  ///
  /// @param row_value  Cells keyed by column name.
  /// @throws std::invalid_argument  if the frame has no columns, the row has a different
  ///         number of cells, a column is missing from the row, or a cell has the wrong type.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// auto frame = tooth.head(2);
  /// frame.append_row(tooth.row(59));
  /// frame.append_row({{"len", 1.5}, {"supp", std::string("VC")},
  ///                   {"dose", dm::dstruct::DataFrame::null_type{}}});
  /// std::print("{} {}\n", frame.nrows(), frame.is_null("dose", 3));   // 4 true
  /// @endcode
  /// @see row(), bind_rows()
  void append_row(const Row& row_value) {
    require_complete_row(row_value);
    for (const auto& name : column_order_) {
      append_cell(columns_.at(name), row_value.at(name));
    }
    ++row_count_;
  }

  /// @brief append_row() from an inline `{name, cell}` list.
  ///
  /// Numeric cells must be `double` (write `2.0`, not `2`), string cells may be literals,
  /// and `null_type{}` is a null. If a name repeats, the first cell is kept and the width
  /// check then fails.
  ///
  /// @param row_value  `{column_name, cell}` pairs, one per column.
  /// @throws std::invalid_argument  as for append_row(const Row&).
  void append_row(std::initializer_list<std::pair<const std::string, cell_type>> row_value) {
    Row row;
    for (const auto& [name, value] : row_value) {
      row.values.emplace(name, value);
    }
    append_row(row);
  }

  /// @brief The named columns, in the order given (dplyr::select() by name).
  ///
  /// Returns a deep copy; every row is kept. Columns not listed are dropped. An empty
  /// list gives a `DataFrame[0 x 0]`. No tidyselect helpers: pass explicit names.
  ///
  /// @param selected_columns  Column names, in output order, without repeats.
  /// @return  A new DataFrame with just those columns.
  /// @throws std::out_of_range  if a name does not exist.
  /// @throws std::invalid_argument  if a name is repeated.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto picked = tooth.select({"dose", "len"});
  /// std::print("{} {}\n", picked.columns(), picked.nrows());   // ["dose", "len"] 60
  /// @endcode
  ///
  /// @par References
  /// - dplyr select: https://dplyr.tidyverse.org/reference/select.html
  ///
  /// @see relocate(), remove_column(), rename()
  [[nodiscard]] DataFrame select(const std::vector<std::string>& selected_columns) const {
    DataFrame result;
    for (const auto& name : selected_columns) {
      const auto& data = column_data(name);
      std::visit(
          [&](const auto& values) {
            result.add_column(name, values);
          },
          data);
    }
    return result;
  }

  /// @brief dplyr::relocate()-style column reorder: moves `columns` to the front (default) or
  ///        immediately after the column named `after`.
  ///
  /// The moved columns keep the order given in @p columns; the others keep their relative
  /// order. Every column and row is kept (a deep copy). There is no `before` option.
  ///
  /// @param columns  Columns to move, without repeats.
  /// @param after  Column to place them after; empty (the default) moves them to the front.
  /// @return  A new DataFrame with the columns reordered.
  /// @throws std::out_of_range  if a name in @p columns or @p after does not exist.
  /// @throws std::invalid_argument  if @p after is one of @p columns, or @p columns repeats
  ///         a name.
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// std::print("{}\n", penguins.relocate({"year"}, "species").columns());
  /// // ["species", "year", "island", "bill_length_mm", "bill_depth_mm",
  /// //  "flipper_length_mm", "body_mass_g", "sex"]
  /// @endcode
  ///
  /// @par References
  /// - dplyr relocate: https://dplyr.tidyverse.org/reference/relocate.html
  ///
  /// @see select(), columns()
  [[nodiscard]] DataFrame relocate(const std::vector<std::string>& columns, const std::string& after = "") const {
    for (const auto& name : columns) {
      (void)column_data(name);
    }
    if (!after.empty()) {
      (void)column_data(after);
    }

    const std::unordered_set<std::string> moving(columns.begin(), columns.end());
    if (!after.empty() && moving.find(after) != moving.end()) {
      throw std::invalid_argument("DataFrame::relocate: 'after' column cannot be one of the columns being moved");
    }

    std::vector<std::string> remaining;
    remaining.reserve(column_order_.size());
    for (const auto& name : column_order_) {
      if (moving.find(name) == moving.end()) {
        remaining.push_back(name);
      }
    }

    std::vector<std::string> new_order;
    new_order.reserve(column_order_.size());
    if (after.empty()) {
      new_order = columns;
      new_order.insert(new_order.end(), remaining.begin(), remaining.end());
    } else {
      for (const auto& name : remaining) {
        new_order.push_back(name);
        if (name == after) {
          new_order.insert(new_order.end(), columns.begin(), columns.end());
        }
      }
    }

    return select(new_order);
  }

  /// @brief The first @p count rows (all rows if there are fewer), like R's `head()`.
  ///
  /// Unlike the free function dm::head(frame, n), @p count is unsigned and has no default.
  ///
  /// @param count  Maximum number of rows to keep.
  /// @return  A new DataFrame with the same columns and min(count, nrows()) rows.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// std::print("{}\n", iris.head(10));   // DataFrame[10 x 5], all setosa
  /// @endcode
  ///
  /// @par References
  /// - R head/tail: https://stat.ethz.ch/R-manual/R-devel/library/utils/html/head.html
  ///
  /// @see tail(), slice(), dm::head()
  [[nodiscard]] DataFrame head(const size_type count) const { return slice_rows(0, std::min(count, row_count_)); }

  /// @brief The last @p count rows (all rows if there are fewer), like R's `tail()`.
  ///
  /// Rows keep their original order. Unlike the free function dm::tail(frame, n), @p count is
  /// unsigned and has no default.
  ///
  /// @param count  Maximum number of rows to keep.
  /// @return  A new DataFrame with the same columns and min(count, nrows()) rows.
  ///
  /// @par Example
  /// @code
  /// const auto last = dm::datasets::tooth_growth().tail(3);
  /// std::print("{} {}\n", last.nrows(), last.double_at("len", 2));   // 3 23
  /// @endcode
  ///
  /// @par References
  /// - R head/tail: https://stat.ethz.ch/R-manual/R-devel/library/utils/html/head.html
  ///
  /// @see head(), slice(), dm::tail()
  [[nodiscard]] DataFrame tail(const size_type count) const {
    const auto kept = std::min(count, row_count_);
    return slice_rows(row_count_ - kept, row_count_);
  }

  /// @brief Rows [@p begin_index, @p end_index) as a new frame (pandas' `df.iloc[b:e]`).
  ///
  /// Zero-based and half-open, unlike dplyr::slice()'s one-based row numbers. Columns and
  /// nulls are copied.
  ///
  /// @param begin_index  First row to keep.
  /// @param end_index  One past the last row to keep.
  /// @return  A new DataFrame with `end_index - begin_index` rows.
  /// @throws std::out_of_range  if @p begin_index > @p end_index or @p end_index > nrows()
  ///         (only checked when the frame has at least one column).
  ///
  /// @par Example
  /// @code
  /// const auto rows = dm::datasets::tooth_growth().slice(10, 15);
  /// std::print("{} {}\n", rows.nrows(), rows.double_at("len", 0));   // 5 16.5
  /// @endcode
  ///
  /// @par References
  /// - dplyr slice: https://dplyr.tidyverse.org/reference/slice.html
  ///
  /// @see head(), tail(), take_rows()
  [[nodiscard]] DataFrame slice(const size_type begin_index, const size_type end_index) const {
    return slice_rows(begin_index, end_index);
  }

  /// @brief The rows at @p indices, in that order (R's `df[idx, ]`, pandas' `df.take()`).
  ///
  /// Indices are zero-based and may repeat or appear in any order, so this also reorders or
  /// resamples rows. O(indices.size() * ncols()).
  ///
  /// @param indices  Zero-based row indices.
  /// @return  A new DataFrame with one row per index and the same columns.
  /// @throws std::out_of_range  if an index is >= nrows().
  ///
  /// @par Example
  /// @code
  /// const auto picked = dm::datasets::tooth_growth().take_rows({59, 0, 0});
  /// std::print("{} {} {}\n", picked.nrows(), picked.double_at("len", 0),
  ///            picked.double_at("len", 2));   // 3 23 4.2
  /// @endcode
  ///
  /// @par References
  /// - pandas DataFrame.take: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.take.html
  ///
  /// @see slice(), filter(), arrange()
  [[nodiscard]] DataFrame take_rows(const std::vector<size_type>& indices) const {
    return take_rows_impl(indices);
  }

  /// @brief The rows for which @p predicate returns true, like dplyr::filter().
  ///
  /// Each row is passed as a Row snapshot; matching rows keep their order and every column.
  /// If nothing matches, the result has the same columns and 0 rows. dm::filter_rows() is the
  /// pipe form.
  ///
  /// @tparam Predicate  Callable type.
  /// @param predicate  Callable taking `const Row&` and returning something testable as bool.
  /// @return  A new filtered DataFrame.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto vc =
  ///   tooth.filter([](const auto& row) { return row.get_string("supp") == "VC"; });
  /// std::print("{}\n", vc.nrows());   // 30
  /// @endcode
  ///
  /// @par References
  /// - dplyr filter: https://dplyr.tidyverse.org/reference/filter.html
  ///
  /// @see dm::filter_rows(), drop_nulls(), Row
  template <typename Predicate>
  [[nodiscard]] DataFrame filter(Predicate&& predicate) const {
    DataFrame result = clone_schema();
    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      const Row current = row(row_index);
      if (std::forward<Predicate>(predicate)(current)) {
        result.append_row(current);
      }
    }
    return result;
  }

  /// @brief Drops every row with a null in @p subset (tidyr's `drop_na()`, pandas'
  ///        `dropna()`).
  ///
  /// Kept rows keep their order. With the default empty @p subset, a null in any column
  /// drops the row.
  ///
  /// @param subset  Columns to inspect; empty (default) means all columns.
  /// @return  A new DataFrame without those rows.
  /// @throws std::out_of_range  if a @p subset column does not exist (only detected when the
  ///         frame has at least one row).
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// std::print("{} {}\n", penguins.drop_nulls().nrows(),
  ///            penguins.drop_nulls({"body_mass_g"}).nrows());   // 333 342
  /// @endcode
  ///
  /// @par References
  /// - tidyr drop_na: https://tidyr.tidyverse.org/reference/drop_na.html
  /// - pandas DataFrame.dropna:
  ///   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
  ///
  /// @see fill_null(), null_count(), filter()
  [[nodiscard]] DataFrame drop_nulls(const std::vector<std::string>& subset = {}) const {
    const auto inspected_columns = subset.empty() ? column_order_ : subset;
    return filter([&](const Row& current) {
      for (const auto& name : inspected_columns) {
        if (current.is_null(name)) {
          return false;
        }
      }
      return true;
    });
  }

  /// @brief Keeps the first row of each distinct combination of @p subset values (pandas'
  ///        `drop_duplicates(keep="first")`).
  ///
  /// All columns are kept, unlike dplyr::distinct() with columns, which keeps only the
  /// named ones. Kept rows stay in their original order. Nulls compare equal to each other,
  /// and so do NaNs (as in R's unique()), so repeated NaN rows are duplicates too.
  /// Average O(nrows() * subset size) using a hash set.
  ///
  /// @param subset  Columns that define a duplicate; empty (default) means all columns.
  /// @return  A new DataFrame without the duplicate rows.
  /// @throws std::invalid_argument  if there are no columns to inspect (empty @p subset on a
  ///         frame without columns).
  /// @throws std::out_of_range  if a @p subset column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto cells = tooth.drop_duplicates({"supp", "dose"});
  /// std::print("{} {}\n", cells.nrows(), cells.ncols());   // 6 3
  /// @endcode
  ///
  /// @par References
  /// - pandas DataFrame.drop_duplicates:
  ///   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
  ///
  /// @see distinct(), n_distinct()
  [[nodiscard]] DataFrame drop_duplicates(const std::vector<std::string>& subset = {}) const {
    const auto inspected_columns = subset.empty() ? column_order_ : subset;
    if (inspected_columns.empty()) {
      throw std::invalid_argument("DataFrame::drop_duplicates requires at least one column");
    }
    for (const auto& column_name : inspected_columns) {
      (void)column_data(column_name);
    }

    std::unordered_map<group_key_type, size_type, GroupKeyHash, GroupKeyEqual> seen;
    std::vector<size_type> kept_indices;
    kept_indices.reserve(row_count_);

    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      auto key = group_key_for_row(inspected_columns, row_index);
      if (seen.find(key) == seen.end()) {
        seen.emplace(std::move(key), row_index);
        kept_indices.push_back(row_index);
      }
    }

    return take_rows_impl(kept_indices);
  }

  /// @brief dplyr::distinct() alias for drop_duplicates().
  ///
  /// Behaves like `distinct(..., .keep_all = TRUE)`: every column is kept and the first row
  /// of each combination wins. Follow with select() for dplyr's default column subset.
  ///
  /// @param subset  Columns that define a duplicate; empty (default) means all columns.
  /// @return  A new DataFrame without the duplicate rows.
  /// @throws std::invalid_argument  if there are no columns to inspect.
  /// @throws std::out_of_range  if a @p subset column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// std::print("{} {}\n", tooth.distinct({"supp"}).nrows(), tooth.distinct().nrows());   // 2 55
  /// @endcode
  ///
  /// @par References
  /// - dplyr distinct: https://dplyr.tidyverse.org/reference/distinct.html
  ///
  /// @see drop_duplicates(), n_distinct()
  [[nodiscard]] DataFrame distinct(const std::vector<std::string>& subset = {}) const { return drop_duplicates(subset); }

  /// @brief Sorts rows by one column (pandas' `sort_values(by=col)`); nulls always go last.
  ///
  /// The sort is stable: ties keep their input order. Numbers compare by value, strings
  /// lexicographically (byte order). NaN values break the ordering (the comparison is not
  /// a strict weak ordering), so replace them first. O(n log n). Use arrange() for several
  /// keys.
  ///
  /// @param column_name  Sort key.
  /// @param ascending  true (default) for ascending, false for descending.
  /// @return  A new DataFrame with the rows reordered.
  /// @throws std::out_of_range  if the column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// const auto heavy_first = penguins.sort_by("body_mass_g", false);
  /// std::print("{} {}\n", heavy_first.double_at("body_mass_g", 0),
  ///            heavy_first.is_null("body_mass_g", 343));   // 6300 true
  /// @endcode
  ///
  /// @par References
  /// - pandas DataFrame.sort_values:
  ///   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
  ///
  /// @see arrange(), dm::arrange()
  [[nodiscard]] DataFrame sort_by(const std::string& column_name, const bool ascending = true) const {
    std::vector<size_type> indices(row_count_);
    for (size_type i = 0; i < row_count_; ++i) {
      indices[i] = i;
    }

    const auto& data = column_data(column_name);
    std::visit(
        [&](const auto& column) {
          std::stable_sort(indices.begin(), indices.end(), [&](const size_type lhs, const size_type rhs) {
            const bool lhs_null = column.is_null(lhs);
            const bool rhs_null = column.is_null(rhs);
            if (lhs_null != rhs_null) {
              return rhs_null;
            }
            if (lhs_null) {
              return false;
            }
            if (ascending) {
              return column.at(lhs) < column.at(rhs);
            }
            return column.at(rhs) < column.at(lhs);
          });
        },
        data);

    DataFrame result = clone_schema();
    for (const auto row_index : indices) {
      result.append_row(row(row_index));
    }
    return result;
  }

  /// @brief dplyr::arrange()-style multi-key sort. `ascending` defaults to all-true; when
  ///        provided it must have the same length as `columns`. Nulls always sort last,
  ///        regardless of direction (matching sort_by()).
  ///
  /// The sort is stable: rows tied on every key keep their input order. Numbers compare by
  /// value, strings lexicographically. dm::arrange() with dm::desc() is the pipe form.
  ///
  /// @param columns  Sort keys, most significant first.
  /// @param ascending  Per-key direction (true = ascending); empty means all ascending.
  /// @return  A new DataFrame with the rows reordered.
  /// @throws std::invalid_argument  if @p columns is empty or the lengths differ.
  /// @throws std::out_of_range  if a column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto sorted = tooth.arrange({"supp", "len"}, {true, false});
  /// std::print("{}\n", sorted.head(1));   // OJ, len 30.9
  /// @endcode
  ///
  /// @par References
  /// - dplyr arrange: https://dplyr.tidyverse.org/reference/arrange.html
  ///
  /// @see dm::arrange(), dm::desc(), sort_by()
  [[nodiscard]] DataFrame arrange(const std::vector<std::string>& columns, std::vector<bool> ascending = {}) const {
    if (columns.empty()) {
      throw std::invalid_argument("DataFrame::arrange requires at least one column");
    }
    if (ascending.empty()) {
      ascending.assign(columns.size(), true);
    }
    if (ascending.size() != columns.size()) {
      throw std::invalid_argument("DataFrame::arrange ascending flags must match column count");
    }
    for (const auto& name : columns) {
      (void)column_data(name);
    }

    std::vector<size_type> indices(row_count_);
    for (size_type i = 0; i < row_count_; ++i) {
      indices[i] = i;
    }

    std::stable_sort(indices.begin(), indices.end(), [&](const size_type lhs, const size_type rhs) {
      for (size_type key_index = 0; key_index < columns.size(); ++key_index) {
        const auto& data = column_data(columns[key_index]);
        const int cmp = std::visit(
            [&](const auto& column) -> int {
              const bool lhs_null = column.is_null(lhs);
              const bool rhs_null = column.is_null(rhs);
              if (lhs_null != rhs_null) {
                return lhs_null ? 1 : -1;
              }
              if (lhs_null) {
                return 0;
              }
              const auto& lhs_value = column.at(lhs);
              const auto& rhs_value = column.at(rhs);
              if (lhs_value < rhs_value) {
                return ascending[key_index] ? -1 : 1;
              }
              if (rhs_value < lhs_value) {
                return ascending[key_index] ? 1 : -1;
              }
              return 0;
            },
            data);
        if (cmp != 0) {
          return cmp < 0;
        }
      }
      return false;
    });

    DataFrame result = clone_schema();
    for (const auto row_index : indices) {
      result.append_row(row(row_index));
    }
    return result;
  }

  /// @brief Count, null count, sum, mean, min and max of a numeric column (a subset of
  ///        pandas' `describe()`).
  ///
  /// Statistics use non-null values only. No standard deviation or quantiles are computed;
  /// see dm::info() for a whole-frame summary. With no non-null values, every field except
  /// null_count is 0. NaN values count as non-null and propagate into sum/mean. O(nrows()).
  ///
  /// @param column_name  Name of a numeric column.
  /// @return  A NumericSummary.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a string column.
  ///
  /// @par Example
  /// @code
  /// const auto stats = dm::datasets::tooth_growth().describe_numeric("len");
  /// std::print("{} {:.4f} {} {}\n", stats.count, stats.mean, stats.min, stats.max);
  /// // 60 18.8133 4.2 33.9
  /// @endcode
  ///
  /// @par References
  /// - pandas DataFrame.describe:
  ///   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
  ///
  /// @see NumericSummary, null_count(), dm::info()
  [[nodiscard]] NumericSummary describe_numeric(const std::string& column_name) const {
    const auto& values = numeric_column(column_name);
    NumericSummary summary;
    if (values.empty()) {
      return summary;
    }

    summary.min = std::numeric_limits<double>::infinity();
    summary.max = -std::numeric_limits<double>::infinity();

    for (size_type index = 0; index < values.size(); ++index) {
      const auto value = values.optional_at(index);
      if (!value.has_value()) {
        ++summary.null_count;
        continue;
      }

      ++summary.count;
      summary.sum += *value;
      if (*value < summary.min) {
        summary.min = *value;
      }
      if (*value > summary.max) {
        summary.max = *value;
      }
    }

    if (summary.count == 0) {
      summary.min = 0.0;
      summary.max = 0.0;
      return summary;
    }

    summary.mean = summary.sum / static_cast<double>(summary.count);
    return summary;
  }

  /// @brief Occurrences of each distinct value in a string column (pandas'
  ///        `value_counts()`, R's `table()`).
  ///
  /// Nulls are not counted. The map is unordered: iteration order is unspecified, so sort
  /// the entries yourself if order matters. O(nrows()) average.
  ///
  /// @param column_name  Name of a string column.
  /// @return  Value-to-count map.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a numeric column.
  ///
  /// @par Example
  /// @code
  /// auto counts = dm::datasets::penguins().value_counts("species");
  /// std::print("{} {} {}\n", counts["Adelie"], counts["Gentoo"], counts["Chinstrap"]);
  /// // 152 124 68
  /// @endcode
  ///
  /// @par References
  /// - pandas Series.value_counts:
  ///   https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
  ///
  /// @see numeric_value_counts(), n_distinct(), group_by_count()
  [[nodiscard]] std::unordered_map<std::string, size_type> value_counts(const std::string& column_name) const {
    const auto& values = string_column(column_name);
    std::unordered_map<std::string, size_type> counts;
    for (size_type index = 0; index < values.size(); ++index) {
      const auto value = values.optional_at(index);
      if (value.has_value()) {
        ++counts[*value];
      }
    }
    return counts;
  }

  /// @brief Occurrences of each distinct value in a numeric column; numeric counterpart of
  ///        value_counts().
  ///
  /// Nulls are not counted; values are compared exactly (no tolerance), and each NaN is its
  /// own key because NaN != NaN. Iteration order is unspecified.
  ///
  /// @param column_name  Name of a numeric column.
  /// @return  Value-to-count map.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if it is a string column.
  ///
  /// @par Example
  /// @code
  /// auto counts = dm::datasets::tooth_growth().numeric_value_counts("dose");
  /// std::print("{} {}\n", counts.size(), counts[0.5]);   // 3 20
  /// @endcode
  ///
  /// @par References
  /// - pandas Series.value_counts:
  ///   https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
  ///
  /// @see value_counts(), n_distinct()
  [[nodiscard]] std::unordered_map<double, size_type> numeric_value_counts(const std::string& column_name) const {
    const auto& values = numeric_column(column_name);
    std::unordered_map<double, size_type> counts;
    for (size_type index = 0; index < values.size(); ++index) {
      const auto value = values.optional_at(index);
      if (value.has_value()) {
        ++counts[*value];
      }
    }
    return counts;
  }

  /// @brief Number of distinct values in `column_name`; a null is counted as a single
  ///        additional distinct value if present (matches dplyr::n_distinct(na.rm = FALSE)).
  ///
  /// Values compare exactly. As in dplyr, all NaNs count as one value, separate from null.
  /// O(nrows()) average.
  ///
  /// @param column_name  Column name (numeric or string).
  /// @return  The number of distinct values, plus one if any cell is null.
  /// @throws std::out_of_range  if the column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// std::print("{} {}\n", penguins.n_distinct("species"), penguins.n_distinct("sex"));   // 3 3
  /// @endcode
  ///
  /// @par References
  /// - dplyr n_distinct: https://dplyr.tidyverse.org/reference/n_distinct.html
  ///
  /// @see distinct(), value_counts()
  [[nodiscard]] size_type n_distinct(const std::string& column_name) const {
    const auto& data = column_data(column_name);
    return std::visit(
        [](const auto& column) -> size_type {
          using column_type = std::decay_t<decltype(column)>;
          using value_type = typename column_type::value_type;
          std::unordered_set<value_type> seen;
          bool has_null = false;
          bool has_nan = false;  // NaN != NaN, so it is counted once here rather than in `seen`
          for (size_type index = 0; index < column.size(); ++index) {
            const auto value = column.optional_at(index);
            if (!value.has_value()) {
              has_null = true;
            } else if constexpr (std::is_floating_point_v<value_type>) {
              if (std::isnan(*value)) {
                has_nan = true;
              } else {
                seen.insert(*value);
              }
            } else {
              seen.insert(*value);
            }
          }
          return seen.size() + (has_null ? 1 : 0) + (has_nan ? 1 : 0);
        },
        data);
  }

  /// @brief Number of rows in each distinct combination of `key_columns`, like
  ///        dplyr::count() or pandas `groupby(...).size()`.
  ///
  /// Groups appear in order of their first row (pandas `sort=False`), not sorted as dplyr
  /// does. Null key cells form a group of their own, and keys of either column type may be
  /// mixed. The counts are stored as a numeric (double) column. Runs in O(rows x keys)
  /// expected time; this frame is not modified. count() is the same with the dplyr default
  /// name "n".
  ///
  /// @param key_columns  Grouping columns (numeric or string), at least one.
  /// @param count_column_name  Name of the count column (default "count"); it must not already
  ///        be a column of this frame, even a non-key one.
  /// @return  One row per group: the key columns, then the count column.
  /// @throws std::invalid_argument  if @p key_columns is empty or @p count_column_name is
  ///         already a column (or is empty).
  /// @throws std::out_of_range  if a key column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto sizes = tooth.group_by_count({"supp"});
  /// std::print("{}\n", sizes);   // 2 x 2: supp, count (VC 30, OJ 30)
  /// @endcode
  ///
  /// @par References
  /// - dplyr count(): https://dplyr.tidyverse.org/reference/count.html
  /// - H. Wickham (2011). "The Split-Apply-Combine Strategy for Data Analysis". Journal of
  ///   Statistical Software 40(1). doi:10.18637/jss.v040.i01
  ///
  /// @see count(), GroupedDataFrame::count(), summarise(), value_counts()
  [[nodiscard]] DataFrame group_by_count(const std::vector<std::string>& key_columns,
                                         const std::string& count_column_name = "count") const {
    require_group_by_keys(key_columns);
    if (has_column(count_column_name)) {
      throw std::invalid_argument("DataFrame::group_by_count count column already exists: " + count_column_name);
    }

    const auto groups = build_groups(key_columns);
    DataFrame result = make_group_result_schema(key_columns);

    std::vector<double> counts;
    counts.reserve(groups.order.size());
    for (const auto& key : groups.order) {
      append_group_key_row(result, key_columns, key);
      counts.push_back(static_cast<double>(groups.groups.at(key).size()));
    }
    result.add_column(count_column_name, std::move(counts));
    return result;
  }

  /// @brief dplyr::count()-style alias for group_by_count(), with the more familiar default
  ///        result column name "n".
  ///
  /// Groups are in first-appearance order (dplyr sorts them); null keys form their own
  /// group. The count column is numeric. This frame is not modified.
  ///
  /// @param key_columns  Grouping columns (numeric or string), at least one.
  /// @param count_column_name  Name of the count column (default "n"); it must not already
  ///        be a column of this frame.
  /// @return  One row per group: the key columns, then the count column.
  /// @throws std::invalid_argument  if @p key_columns is empty or @p count_column_name is
  ///         already a column (or is empty).
  /// @throws std::out_of_range  if a key column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto cells = tooth.count({"supp", "dose"});
  /// std::print("{} {}\n", cells.nrows(), cells.get_column("n").at(0));   // 6 10
  /// @endcode
  ///
  /// @par References
  /// - dplyr count(): https://dplyr.tidyverse.org/reference/count.html
  ///
  /// @see group_by_count(), GroupedDataFrame::count(), dm::count()
  [[nodiscard]] DataFrame count(const std::vector<std::string>& key_columns,
                                const std::string& count_column_name = "n") const {
    return group_by_count(key_columns, count_column_name);
  }

  /// @brief Per-group sums of numeric `value_columns`, like pandas
  ///        `groupby(keys)[cols].sum()` or dplyr `summarise(across(cols, sum, na.rm = TRUE))`.
  ///
  /// Nulls are skipped (R `na.rm = TRUE`), but a group whose values are all null gets a null
  /// sum rather than R's 0. Groups appear in first-appearance order; null keys form their own
  /// group. Each output column keeps its source name. This frame is not modified.
  ///
  /// @param key_columns  Grouping columns, at least one.
  /// @param value_columns  Numeric columns to sum, at least one; none may also be a key.
  /// @return  One row per group: the key columns, then one numeric column per value column.
  /// @throws std::invalid_argument  if either list is empty, a value column is not numeric, or
  ///         a value column is also a key column (duplicate output name).
  /// @throws std::out_of_range  if a column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto totals = tooth.group_by_sum({"supp"}, {"len"});
  /// std::print("{}\n", totals.get_column("len").at(0));   // 508.9 (VC)
  /// @endcode
  ///
  /// @par References
  /// - pandas DataFrameGroupBy.sum():
  ///   https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.sum.html
  ///
  /// @see group_by_mean(), summarise(), summarise_each()
  [[nodiscard]] DataFrame group_by_sum(const std::vector<std::string>& key_columns,
                                       const std::vector<std::string>& value_columns) const {
    require_group_by_keys(key_columns);
    require_numeric_columns(value_columns);

    const auto groups = build_groups(key_columns);
    DataFrame result = make_group_result_schema(key_columns);
    std::unordered_map<std::string, std::vector<std::optional<double>>> aggregated;
    for (const auto& column_name : value_columns) {
      aggregated.emplace(column_name, std::vector<std::optional<double>>{});
      aggregated.at(column_name).reserve(groups.order.size());
    }

    for (const auto& key : groups.order) {
      append_group_key_row(result, key_columns, key);
      const auto& rows = groups.groups.at(key);
      for (const auto& column_name : value_columns) {
        double sum = 0.0;
        size_type present = 0;
        for (const auto row_index : rows) {
          const auto value = optional_double_at(column_name, row_index);
          if (value.has_value()) {
            sum += *value;
            ++present;
          }
        }
        if (present == 0) {
          aggregated.at(column_name).push_back(std::nullopt);
        } else {
          aggregated.at(column_name).push_back(sum);
        }
      }
    }

    for (const auto& column_name : value_columns) {
      result.add_column(column_name, std::move(aggregated.at(column_name)));
    }
    return result;
  }

  /// @brief Per-group arithmetic means of numeric `value_columns`, like pandas
  ///        `groupby(keys)[cols].mean()` or dplyr `summarise(across(cols, mean, na.rm = TRUE))`.
  ///
  /// Nulls are skipped (R `na.rm = TRUE`): the mean is over the non-null values, and a group
  /// with no non-null values gets a null. Groups appear in first-appearance order; null keys
  /// form their own group. Each output column keeps its source name. This frame is not
  /// modified.
  ///
  /// @param key_columns  Grouping columns, at least one.
  /// @param value_columns  Numeric columns to average, at least one; none may also be a key.
  /// @return  One row per group: the key columns, then one numeric column per value column.
  /// @throws std::invalid_argument  if either list is empty, a value column is not numeric, or
  ///         a value column is also a key column.
  /// @throws std::out_of_range  if a column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto penguins = dm::datasets::penguins();
  /// const auto means = penguins.group_by_mean({"species"}, {"body_mass_g", "flipper_length_mm"});
  /// std::print("{}\n", means);   // 3 x 3: Adelie 3700.66 g, ...
  /// @endcode
  ///
  /// @par References
  /// - pandas DataFrameGroupBy.mean():
  ///   https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.mean.html
  ///
  /// @see group_by_sum(), summarise(), summarise_each()
  [[nodiscard]] DataFrame group_by_mean(const std::vector<std::string>& key_columns,
                                        const std::vector<std::string>& value_columns) const {
    require_group_by_keys(key_columns);
    require_numeric_columns(value_columns);

    const auto groups = build_groups(key_columns);
    DataFrame result = make_group_result_schema(key_columns);
    std::unordered_map<std::string, std::vector<std::optional<double>>> aggregated;
    for (const auto& column_name : value_columns) {
      aggregated.emplace(column_name, std::vector<std::optional<double>>{});
      aggregated.at(column_name).reserve(groups.order.size());
    }

    for (const auto& key : groups.order) {
      append_group_key_row(result, key_columns, key);
      const auto& rows = groups.groups.at(key);
      for (const auto& column_name : value_columns) {
        double sum = 0.0;
        size_type present = 0;
        for (const auto row_index : rows) {
          const auto value = optional_double_at(column_name, row_index);
          if (value.has_value()) {
            sum += *value;
            ++present;
          }
        }
        if (present == 0) {
          aggregated.at(column_name).push_back(std::nullopt);
        } else {
          aggregated.at(column_name).push_back(sum / static_cast<double>(present));
        }
      }
    }

    for (const auto& column_name : value_columns) {
      result.add_column(column_name, std::move(aggregated.at(column_name)));
    }
    return result;
  }

  /// @brief General dplyr::summarise()-style aggregation: one output row per distinct
  ///        combination of `key_columns`, with one output column per AggSpec. See also
  ///        group_by(), which returns a GroupedDataFrame wrapping this same method.
  ///
  /// Groups appear in order of first appearance (dplyr sorts them); null keys form their own
  /// group. Per AggSpec:
  /// - Sum, Mean, Min, Max, Median, StdDev skip nulls (R `na.rm = TRUE`); a group with no
  ///   non-null values gets a null. Median averages the two middle values for an even count.
  ///   StdDev is the sample standard deviation (n - 1 denominator, like R `sd()`), but a group
  ///   with one non-null value gets 0 where R gives NA.
  /// - Min and Max also work on string columns (lexicographic) and give a string column.
  /// - Count is the group's row count (nulls included); `column` is ignored.
  /// - NDistinct counts distinct values, with a null counted once (dplyr::n_distinct()).
  ///
  /// Output columns are named by AggSpec::result_name, else `column`, else "n" for Count.
  /// This frame is not modified.
  ///
  /// @param key_columns  Grouping columns, at least one.
  /// @param specs  Aggregations, at least one; one output column each, in this order.
  /// @return  One row per group: the key columns, then one column per spec.
  /// @throws std::invalid_argument  if @p key_columns or @p specs is empty, a numeric-only
  ///         aggregation is applied to a string column (only raised for a group that has a
  ///         non-null value), or two output names collide (including with a key column).
  /// @throws std::out_of_range  if a key or spec column does not exist.
  ///
  /// @par Example
  /// @code
  /// using DataFrame = dm::dstruct::DataFrame;
  /// const auto penguins = dm::datasets::penguins();
  /// const auto stats = penguins.summarise({"species"}, {
  ///     {"body_mass_g", DataFrame::AggFunc::Median, "median_mass"},
  ///     {"body_mass_g", DataFrame::AggFunc::StdDev, "sd_mass"},
  ///     {"island", DataFrame::AggFunc::NDistinct, "islands"},
  ///     {"", DataFrame::AggFunc::Count, ""}});
  /// std::print("{}\n", stats);   // 3 x 5: species, median_mass, sd_mass, islands, n
  ///                              // Adelie: 3700, 458.566, 3, 152
  /// @endcode
  ///
  /// @par References
  /// - dplyr summarise(): https://dplyr.tidyverse.org/reference/summarise.html
  /// - H. Wickham (2011). "The Split-Apply-Combine Strategy for Data Analysis". Journal of
  ///   Statistical Software 40(1). doi:10.18637/jss.v040.i01
  ///
  /// @see group_by(), GroupedDataFrame::summarise(), summarise_each(), AggSpec, AggFunc
  [[nodiscard]] DataFrame summarise(const std::vector<std::string>& key_columns, const std::vector<AggSpec>& specs) const {
    require_group_by_keys(key_columns);
    if (specs.empty()) {
      throw std::invalid_argument("DataFrame::summarise requires at least one aggregation");
    }
    for (const auto& spec : specs) {
      if (spec.func != AggFunc::Count && !has_column(spec.column)) {
        throw std::out_of_range("DataFrame::summarise missing column: " + spec.column);
      }
    }

    const auto groups = build_groups(key_columns);
    DataFrame result = make_group_result_schema(key_columns);

    std::vector<bool> spec_is_numeric(specs.size(), true);
    for (size_type spec_index = 0; spec_index < specs.size(); ++spec_index) {
      const auto& spec = specs[spec_index];
      spec_is_numeric[spec_index] =
          spec.func == AggFunc::Count || spec.func == AggFunc::NDistinct || column_type(spec.column) == ColumnType::Numeric;
    }

    std::vector<std::vector<std::optional<double>>> numeric_cells(specs.size());
    std::vector<std::vector<std::optional<std::string>>> string_cells(specs.size());

    for (const auto& key : groups.order) {
      append_group_key_row(result, key_columns, key);
      const auto& rows = groups.groups.at(key);

      for (size_type spec_index = 0; spec_index < specs.size(); ++spec_index) {
        const auto& spec = specs[spec_index];

        if (spec.func == AggFunc::Count) {
          numeric_cells[spec_index].push_back(static_cast<double>(rows.size()));
          continue;
        }
        if (spec.func == AggFunc::NDistinct) {
          numeric_cells[spec_index].push_back(static_cast<double>(group_n_distinct(spec.column, rows)));
          continue;
        }
        if (spec_is_numeric[spec_index]) {
          numeric_cells[spec_index].push_back(aggregate_numeric(spec.column, rows, spec.func));
        } else {
          string_cells[spec_index].push_back(aggregate_string(spec.column, rows, spec.func));
        }
      }
    }

    for (size_type spec_index = 0; spec_index < specs.size(); ++spec_index) {
      const auto& spec = specs[spec_index];
      const std::string default_name = spec.func == AggFunc::Count ? std::string("n") : spec.column;
      const std::string result_name = spec.result_name.empty() ? default_name : spec.result_name;
      if (spec_is_numeric[spec_index]) {
        result.add_column(result_name, std::move(numeric_cells[spec_index]));
      } else {
        result.add_column(result_name, std::move(string_cells[spec_index]));
      }
    }

    return result;
  }

  /// @brief Ungrouped dplyr::summarise(): one output row that aggregates the whole frame.
  ///
  /// Same rules as the grouped overload, with every row in one group. A frame with no rows
  /// gives a frame with no rows (dplyr instead gives one row).
  ///
  /// @param specs  Aggregations, at least one; one output column each, in this order.
  /// @return  A one-row frame with one column per spec.
  /// @throws std::invalid_argument  as the grouped overload.
  /// @throws std::out_of_range  if a spec column does not exist.
  ///
  /// @par Example
  /// @code
  /// using DF = dm::dstruct::DataFrame;
  /// const auto all = dm::datasets::iris().summarise(
  ///     {{"", DF::AggFunc::Count, "n"}, {"Petal.Length", DF::AggFunc::Mean, "mean"}});
  /// std::print("{}\n", all.to_md());   // | n | mean | ... | 150 | 3.758 |
  /// @endcode
  /// @see summarise(const std::vector<std::string>&, const std::vector<AggSpec>&) const
  [[nodiscard]] DataFrame summarise(const std::vector<AggSpec>& specs) const {
    static const std::string key = "\x1f.summarise_all";
    DataFrame tagged = *this;
    tagged.add_column(key, std::vector<std::string>(row_count_, ""));
    auto result = tagged.summarise({key}, specs);
    result.remove_column(key);
    return result;
  }

  /// @brief dplyr::summarise_each()-style aggregation: applies `func` to every numeric column
  ///        that is not a key, once per group (one output row per distinct combination of
  ///        `key_columns`, in first-appearance order). String columns that are not keys are
  ///        skipped. Like R with `na.rm = FALSE`, a group containing a null gives a null result.
  /// @tparam Func Callable taking a `std::vector<double>` (or anything implicitly constructible
  ///         from one, such as `dm::Vector<>`) and returning something convertible to double.
  /// @param key_columns Grouping columns; at least one is required.
  /// @param func        Reduction applied to each group's values, e.g. `dm::mean`.
  /// @return  One row per group: the key columns, then each numeric non-key column in this
  ///          frame's column order.
  /// @throws std::invalid_argument  if @p key_columns is empty.
  /// @throws std::out_of_range  if a key column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto means = tooth.summarise_each({"supp", "dose"}, dm::mean);
  /// std::print("{}\n", means);   // 6 x 3 frame: supp, dose, len (VC 0.5 -> 7.98)
  /// @endcode
  /// @see dm::summarise_each(), group_by(), summarise()
  template <typename Func>
  [[nodiscard]] DataFrame summarise_each(const std::vector<std::string>& key_columns, Func&& func) const {
    require_group_by_keys(key_columns);
    const auto groups = build_groups(key_columns);
    DataFrame result = make_group_result_schema(key_columns);

    std::vector<std::string> value_columns;
    for (const auto& name : column_order_) {
      const bool is_key = std::find(key_columns.begin(), key_columns.end(), name) != key_columns.end();
      if (!is_key && column_type(name) == ColumnType::Numeric) {
        value_columns.push_back(name);
      }
    }

    std::vector<std::vector<std::optional<double>>> cells(value_columns.size());
    for (const auto& key : groups.order) {
      append_group_key_row(result, key_columns, key);
      const auto& rows = groups.groups.at(key);
      for (size_type column_index = 0; column_index < value_columns.size(); ++column_index) {
        std::vector<double> values;
        values.reserve(rows.size());
        bool has_null = false;
        for (const auto row_index : rows) {
          const auto value = optional_double_at(value_columns[column_index], row_index);
          if (!value) {
            has_null = true;
            break;
          }
          values.push_back(*value);
        }
        cells[column_index].push_back(has_null ? std::nullopt
                                               : std::optional<double>(static_cast<double>(func(values))));
      }
    }

    for (size_type column_index = 0; column_index < value_columns.size(); ++column_index) {
      result.add_column(value_columns[column_index], std::move(cells[column_index]));
    }
    return result;
  }

  /// @brief dplyr::group_by()-style entry point: returns a lightweight GroupedDataFrame that
  ///        pairs `key_columns` with a copy of `*this`, deferring aggregation to
  ///        GroupedDataFrame::summarise()/count(). Defined out-of-line below, after
  ///        GroupedDataFrame itself is complete.
  ///
  /// Groups are formed lazily by the aggregation and appear in order of first appearance.
  /// dm::group_by() is the pipe form.
  ///
  /// @param key_columns  Grouping columns (numeric or string), at least one.
  /// @return  A GroupedDataFrame holding a copy of this frame and the keys.
  /// @throws std::invalid_argument  if @p key_columns is empty.
  /// @throws std::out_of_range  if a key column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// std::print("{}\n", tooth.group_by({"supp"}).count());   // VC 30, OJ 30
  /// @endcode
  /// @see dm::group_by(), GroupedDataFrame
  [[nodiscard]] GroupedDataFrame group_by(const std::vector<std::string>& key_columns) const;

  /// @brief dplyr::pivot_longer()-style reshape: stacks `value_columns` into two new columns
  ///        (`names_to` holding the source column name, `values_to` holding its value), one row
  ///        per (original row, pivoted column) pair. All of `value_columns` must share the same
  ///        column type.
  ///
  /// Every other column is an id column, repeated on each of a row's output rows and kept in
  /// its original order. Output rows run source row by source row, and within a row follow
  /// the order of @p value_columns, as in tidyr. Null values are kept (there is no
  /// `values_drop_na`). The result has nrows() * value_columns.size() rows; this frame is not
  /// modified.
  ///
  /// @param value_columns  Columns to stack, at least one, all numeric or all string.
  /// @param names_to  Name of the new string column holding the source column names
  ///        (default "name").
  /// @param values_to  Name of the new column holding the values (default "value"); it has
  ///        the type of @p value_columns.
  /// @return  The id columns, then @p names_to, then @p values_to.
  /// @throws std::invalid_argument  if @p value_columns is empty, the value columns differ in
  ///         type, or @p names_to / @p values_to is empty or clashes with an id column or with
  ///         each other.
  /// @throws std::out_of_range  if a value column does not exist.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// const auto sepals = iris.select({"Species", "Sepal.Length", "Sepal.Width"})
  ///                         .pivot_longer({"Sepal.Length", "Sepal.Width"}, "measure", "cm");
  /// std::print("{}\n", sepals.to_string(2));   // 300 x 3: setosa | Sepal.Length | 5.1, ...
  /// @endcode
  ///
  /// @par References
  /// - tidyr pivot_longer(): https://tidyr.tidyverse.org/reference/pivot_longer.html
  /// - H. Wickham (2014). "Tidy Data". Journal of Statistical Software 59(10).
  ///   doi:10.18637/jss.v059.i10
  ///
  /// @see pivot_wider()
  [[nodiscard]] DataFrame pivot_longer(const std::vector<std::string>& value_columns, const std::string& names_to = "name",
                                       const std::string& values_to = "value") const {
    if (value_columns.empty()) {
      throw std::invalid_argument("DataFrame::pivot_longer requires at least one value column");
    }
    for (const auto& name : value_columns) {
      (void)column_data(name);
    }

    const std::unordered_set<std::string> pivoted(value_columns.begin(), value_columns.end());
    std::vector<std::string> id_columns;
    for (const auto& name : column_order_) {
      if (pivoted.find(name) == pivoted.end()) {
        id_columns.push_back(name);
      }
    }

    const bool values_numeric = column_type(value_columns.front()) == ColumnType::Numeric;
    for (const auto& name : value_columns) {
      if ((column_type(name) == ColumnType::Numeric) != values_numeric) {
        throw std::invalid_argument("DataFrame::pivot_longer requires all value columns to share the same type");
      }
    }

    DataFrame result;
    for (const auto& name : id_columns) {
      append_schema_column(result, name, column_type(name));
    }
    append_schema_column(result, names_to, ColumnType::String);
    append_schema_column(result, values_to, values_numeric ? ColumnType::Numeric : ColumnType::String);

    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      for (const auto& value_column : value_columns) {
        Row out;
        for (const auto& id_name : id_columns) {
          out.values[id_name] = value(id_name, row_index);
        }
        out.values[names_to] = cell_type(value_column);
        out.values[values_to] = value(value_column, row_index);
        result.append_row(out);
      }
    }

    return result;
  }

  /// @brief dplyr::pivot_wider()-style reshape: `names_from` (a string column) supplies new
  ///        column names, `values_from` supplies their values, and the remaining columns (or an
  ///        explicit `id_columns`) define one output row per distinct combination. If a given
  ///        (id, name) combination has more than one source row, the first occurrence wins.
  ///
  /// Output rows are one per distinct id combination in first-appearance order (null ids form
  /// their own row, as do NaN ids). With no id columns at all (a frame holding only the names
  /// and values columns), the result is a single row, as in tidyr. New columns are created in the order their names first appear in
  /// @p names_from; rows whose name is null are ignored. A missing (id, name) combination is
  /// null. tidyr instead warns about duplicates and builds list-columns. This frame is not
  /// modified.
  ///
  /// @param names_from  String column whose values become the new column names.
  /// @param values_from  Column (numeric or string) holding the values; the new columns take
  ///        its type.
  /// @param id_columns  Columns identifying an output row (default empty: every column other
  ///        than @p names_from and @p values_from).
  /// @return  The id columns, then one column per distinct name.
  /// @throws std::invalid_argument  if @p names_from is not a string column, or a new column
  ///         name is empty or clashes with an id column.
  /// @throws std::out_of_range  if a named column does not exist.
  ///
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame scores;
  /// scores.add_column("id", std::vector<double>{1, 1, 2});
  /// scores.add_column("test", std::vector<std::string>{"math", "art", "math"});
  /// scores.add_column("score", std::vector<double>{90, 75, 60});
  /// std::print("{}\n", scores.pivot_wider("test", "score"));   // 2 x 3: id, math, art (id 2 art NA)
  /// @endcode
  ///
  /// @par References
  /// - tidyr pivot_wider(): https://tidyr.tidyverse.org/reference/pivot_wider.html
  /// - H. Wickham (2014). "Tidy Data". Journal of Statistical Software 59(10).
  ///   doi:10.18637/jss.v059.i10
  ///
  /// @see pivot_longer()
  [[nodiscard]] DataFrame pivot_wider(const std::string& names_from, const std::string& values_from,
                                      const std::vector<std::string>& id_columns = {}) const {
    (void)column_data(names_from);
    (void)column_data(values_from);
    if (column_type(names_from) != ColumnType::String) {
      throw std::invalid_argument("DataFrame::pivot_wider requires names_from to be a string column");
    }

    std::vector<std::string> ids = id_columns;
    if (ids.empty()) {
      for (const auto& name : column_order_) {
        if (name != names_from && name != values_from) {
          ids.push_back(name);
        }
      }
    } else {
      for (const auto& name : ids) {
        (void)column_data(name);
      }
    }

    const bool values_numeric = column_type(values_from) == ColumnType::Numeric;
    const auto groups = build_groups(ids);

    std::vector<std::string> new_column_names;
    std::unordered_set<std::string> seen_names;
    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      const auto name_value = optional_string_at(names_from, row_index);
      if (!name_value.has_value()) {
        continue;
      }
      if (seen_names.insert(*name_value).second) {
        new_column_names.push_back(*name_value);
      }
    }

    DataFrame result = make_group_result_schema(ids);

    std::vector<std::vector<std::optional<double>>> numeric_cells(new_column_names.size());
    std::vector<std::vector<std::optional<std::string>>> string_cells(new_column_names.size());

    for (const auto& key : groups.order) {
      // With no id columns there is a single group and no key cells to write; the result's
      // one row comes from the value columns added below.
      if (!ids.empty()) append_group_key_row(result, ids, key);
      const auto& rows = groups.groups.at(key);

      std::unordered_map<std::string, size_type> row_for_name;
      for (const auto row_index : rows) {
        const auto name_value = optional_string_at(names_from, row_index);
        if (!name_value.has_value() || row_for_name.count(*name_value) > 0) {
          continue;
        }
        row_for_name.emplace(*name_value, row_index);
      }

      for (size_type column_index = 0; column_index < new_column_names.size(); ++column_index) {
        const auto found = row_for_name.find(new_column_names[column_index]);
        if (found == row_for_name.end()) {
          if (values_numeric) {
            numeric_cells[column_index].push_back(std::nullopt);
          } else {
            string_cells[column_index].push_back(std::nullopt);
          }
          continue;
        }
        if (values_numeric) {
          numeric_cells[column_index].push_back(optional_double_at(values_from, found->second));
        } else {
          string_cells[column_index].push_back(optional_string_at(values_from, found->second));
        }
      }
    }

    for (size_type column_index = 0; column_index < new_column_names.size(); ++column_index) {
      if (values_numeric) {
        result.add_column(new_column_names[column_index], std::move(numeric_cells[column_index]));
      } else {
        result.add_column(new_column_names[column_index], std::move(string_cells[column_index]));
      }
    }

    return result;
  }

  /// @brief Single-key relational join with @p right, like dplyr's inner_join(), left_join(),
  ///        right_join(), full_join(), semi_join() and anti_join() (or pandas merge()).
  ///
  /// Keys match on cell equality: a numeric key never matches a string key, and a null key
  /// matches a null key (dplyr's default `na_matches = "na"`). Each left row is paired with
  /// every matching right row, in right's row order, so duplicates multiply. Output rows
  /// follow left's order; Right/Full joins then append unmatched right rows in right's order,
  /// with nulls on the left side. Columns are left's (key included) followed by right's. When
  /// @p left_key equals @p right_key the key appears once and is filled from right for
  /// unmatched right rows; otherwise both key columns are kept, and the left key stays null
  /// on unmatched right rows (dplyr would coalesce them). Uses a hash table on right's key:
  /// expected O(rows of both + output rows). Neither frame is modified.
  ///
  /// @param right  Frame to join with.
  /// @param left_key  Key column in this frame.
  /// @param right_key  Key column in @p right.
  /// @param join_type One of Inner/Left/Right/Full/Semi/Anti (default Inner). Semi/Anti keep only left's rows
  ///        (matched / unmatched respectively) and never merge in right's columns. For the
  ///        others, any column name present in both frames (other than the key column when
  ///        left_key == right_key) is suffixed on both sides (left_suffix/right_suffix) so the
  ///        output has no duplicate names.
  /// @param left_suffix  Suffix for clashing left columns (default "_x", as in dplyr).
  /// @param right_suffix  Suffix for clashing right columns (default "_y").
  /// @return  The joined frame (for Semi/Anti, a subset of this frame's rows).
  /// @throws std::out_of_range  if @p left_key is not in this frame or @p right_key is not in
  ///         @p right.
  /// @throws std::invalid_argument  if a suffixed name still clashes with another column
  ///         (for example left already has "a" and "a_x").
  ///
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame people;
  /// people.add_column("id", std::vector<double>{1, 2, 3});
  /// people.add_column("name", std::vector<std::string>{"ann", "bob", "cy"});
  /// dm::dstruct::DataFrame pets;
  /// pets.add_column("id", std::vector<double>{1, 1, 3});
  /// pets.add_column("pet", std::vector<std::string>{"cat", "dog", "fish"});
  /// std::print("{}\n", people.join(pets, "id", "id").nrows());   // 3: ann twice, cy once
  /// const auto all = people.join(pets, "id", "id", dm::dstruct::DataFrame::JoinType::Left);
  /// std::print("{}\n", all.nrows());   // 4: bob kept with a null pet
  /// const auto lonely = people.join(pets, "id", "id", dm::dstruct::DataFrame::JoinType::Anti);
  /// std::print("{}\n", lonely.get_column<std::string>("name").at(0));   // bob
  /// @endcode
  ///
  /// @par References
  /// - dplyr mutating joins: https://dplyr.tidyverse.org/reference/mutate-joins.html
  /// - dplyr filtering joins: https://dplyr.tidyverse.org/reference/filter-joins.html
  ///
  /// @see bind_cols(), JoinType
  [[nodiscard]] DataFrame join(const DataFrame& right, const std::string& left_key, const std::string& right_key,
                               const JoinType join_type = JoinType::Inner, const std::string& left_suffix = "_x",
                               const std::string& right_suffix = "_y") const {
    if (!has_column(left_key)) {
      throw std::out_of_range("DataFrame::join missing left key: " + left_key);
    }
    if (!right.has_column(right_key)) {
      throw std::out_of_range("DataFrame::join missing right key: " + right_key);
    }

    if (join_type == JoinType::Semi || join_type == JoinType::Anti) {
      const auto right_lookup = right.build_row_lookup(right_key);
      std::vector<size_type> kept_indices;
      for (size_type row_index = 0; row_index < row_count_; ++row_index) {
        const bool matched = right_lookup.find(value(left_key, row_index)) != right_lookup.end();
        if (matched == (join_type == JoinType::Semi)) {
          kept_indices.push_back(row_index);
        }
      }
      return take_rows_impl(kept_indices);
    }

    const bool merged_key = left_key == right_key;
    const auto [left_plan, right_plan] = build_join_plans(right, left_key, right_key, merged_key, left_suffix, right_suffix);

    DataFrame result;
    for (const auto& [source_name, output_name] : left_plan) {
      append_schema_column(result, output_name, column_type(source_name));
    }
    for (const auto& [source_name, output_name] : right_plan) {
      append_schema_column(result, output_name, right.column_type(source_name));
    }

    const auto right_lookup = right.build_row_lookup(right_key);
    const bool include_unmatched_left = join_type == JoinType::Left || join_type == JoinType::Full;
    std::vector<bool> right_matched(right.row_count_, false);

    for (size_type left_row = 0; left_row < row_count_; ++left_row) {
      const auto match_it = right_lookup.find(value(left_key, left_row));
      if (match_it == right_lookup.end()) {
        if (include_unmatched_left) {
          append_join_row(result, left_plan, right_plan, this, left_row, nullptr, 0, merged_key, left_key, right_key);
        }
        continue;
      }
      for (const auto right_row : match_it->second) {
        right_matched[right_row] = true;
        append_join_row(result, left_plan, right_plan, this, left_row, &right, right_row, merged_key, left_key, right_key);
      }
    }

    if (join_type == JoinType::Right || join_type == JoinType::Full) {
      for (size_type right_row = 0; right_row < right.row_count_; ++right_row) {
        if (!right_matched[right_row]) {
          append_join_row(result, left_plan, right_plan, nullptr, 0, &right, right_row, merged_key, left_key, right_key);
        }
      }
    }

    return result;
  }

  /// @brief Strict row concatenation (like R `rbind()`): this frame's rows, then @p other's.
  ///
  /// Both frames must have exactly the same column names in the same order with the same
  /// types; use bind_rows() to align columns by name instead. Nulls are copied. Both inputs
  /// are unchanged; runs in O(total rows x columns).
  ///
  /// @param other  Frame whose rows are appended.
  /// @return  A new frame with nrows() + other.nrows() rows.
  /// @throws std::invalid_argument  if the column order or a column type differs.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto ends = tooth.head(2).concat_rows(tooth.tail(3));
  /// std::print("{}\n", ends.nrows());   // 5
  /// @endcode
  ///
  /// @see bind_rows(), append_row()
  [[nodiscard]] DataFrame concat_rows(const DataFrame& other) const {
    require_same_schema(other);
    DataFrame result = select(column_order_);
    for (size_type row_index = 0; row_index < other.nrows(); ++row_index) {
      result.append_row(other.row(row_index));
    }
    return result;
  }

  /// @brief dplyr::bind_rows()-style row union: unlike concat_rows() (which requires identical
  ///        column order/types), aligns columns by name -- a column present in only one frame is
  ///        null-filled for the rows coming from the other. A shared column name must have the
  ///        same type in both frames.
  ///
  /// The result's columns are this frame's, in order, followed by @p other's new columns in
  /// their order. Rows are this frame's, then @p other's. Both inputs are unchanged.
  ///
  /// @param other  Frame whose rows are appended.
  /// @return  A new frame with nrows() + other.nrows() rows and the union of the columns.
  /// @throws std::invalid_argument  if a shared column is numeric in one frame and string in
  ///         the other.
  ///
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame a;
  /// a.add_column("x", std::vector<double>{1, 2});
  /// dm::dstruct::DataFrame b;
  /// b.add_column("x", std::vector<double>{3});
  /// b.add_column("label", std::vector<std::string>{"new"});
  /// const auto both = a.bind_rows(b);
  /// std::print("{}\n", both);   // 3 x 2: x 1,2,3; label NA,NA,new
  /// @endcode
  ///
  /// @par References
  /// - dplyr bind_rows(): https://dplyr.tidyverse.org/reference/bind_rows.html
  ///
  /// @see concat_rows(), bind_cols()
  [[nodiscard]] DataFrame bind_rows(const DataFrame& other) const {
    std::vector<std::string> combined_columns = column_order_;
    for (const auto& name : other.column_order_) {
      if (!has_column(name)) {
        combined_columns.push_back(name);
      }
    }

    for (const auto& name : combined_columns) {
      if (has_column(name) && other.has_column(name) && column_type(name) != other.column_type(name)) {
        throw std::invalid_argument("DataFrame::bind_rows column type mismatch: " + name);
      }
    }

    DataFrame result;
    for (const auto& name : combined_columns) {
      const auto type = has_column(name) ? column_type(name) : other.column_type(name);
      append_schema_column(result, name, type);
    }

    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      result.append_row(build_bind_row(combined_columns, this, row_index));
    }
    for (size_type row_index = 0; row_index < other.row_count_; ++row_index) {
      result.append_row(build_bind_row(combined_columns, &other, row_index));
    }

    return result;
  }

  /// @brief dplyr::bind_cols()-style column union: both frames must have the same row count and
  ///        disjoint column names.
  ///
  /// Rows are matched by position, not by key (use join() for that). The result has this
  /// frame's columns, then @p other's, all copied. Unlike dplyr, duplicate names are an error
  /// rather than being renamed. Both inputs are unchanged.
  ///
  /// @param other  Frame whose columns are appended.
  /// @return  A new frame with ncols() + other.ncols() columns.
  /// @throws std::invalid_argument  if the row counts differ (including binding onto an empty
  ///         frame with no columns) or a column name appears in both frames.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto glued = tooth.select({"len"}).bind_cols(tooth.select({"supp", "dose"}));
  /// std::print("{} {}\n", glued.nrows(), glued.ncols());   // 60 3
  /// @endcode
  ///
  /// @par References
  /// - dplyr bind_cols(): https://dplyr.tidyverse.org/reference/bind_cols.html
  ///
  /// @see bind_rows(), join()
  [[nodiscard]] DataFrame bind_cols(const DataFrame& other) const {
    if (row_count_ != other.row_count_) {
      throw std::invalid_argument("DataFrame::bind_cols requires equal row counts");
    }
    for (const auto& name : other.column_order_) {
      if (has_column(name)) {
        throw std::invalid_argument("DataFrame::bind_cols duplicate column: " + name);
      }
    }

    DataFrame result = select(column_order_);
    for (const auto& name : other.column_order_) {
      const auto& data = other.column_data(name);
      std::visit(
          [&](const auto& values) {
            result.add_column(name, values);
          },
          data);
    }
    return result;
  }

  /// @brief Renders a plain-text preview: a `DataFrame[rows x cols]` header, then a
  ///        Markdown-style table of the first @p max_rows rows.
  ///
  /// Columns are left-aligned and padded to their widest shown cell. Numbers use the default
  /// stream format (6 significant digits, e.g. 0.333333 and 1e+10), nulls print as `NA`, and
  /// strings print unquoted and unescaped. When rows are cut, a `... (k more rows)` line
  /// follows. A frame with no columns renders only the header. `std::print("{}", frame)`
  /// uses this with the default @p max_rows.
  ///
  /// @param max_rows  Maximum number of rows to show (default 10); 0 shows only the header,
  ///        the column names and the `... (k more rows)` line.
  /// @return  The rendered text, without a trailing newline.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// std::print("{}\n", tooth.to_string(2));
  /// // DataFrame[60 x 3]
  /// // | len  | supp | dose |
  /// // | ---- | ---- | ---- |
  /// // | 4.2  | VC   | 0.5  |
  /// // | 11.5 | VC   | 0.5  |
  /// // ... (58 more rows)
  /// @endcode
  ///
  /// @see head(), dm::info()
  [[nodiscard]] std::string to_string(const size_type max_rows = 10) const {
    std::ostringstream out;
    out << "DataFrame[" << row_count_ << " x " << column_order_.size() << "]";
    if (column_order_.empty()) {
      return out.str();
    }

    const auto displayed_rows = std::min(max_rows, row_count_);
    std::vector<std::vector<std::string>> rendered_rows;
    rendered_rows.reserve(displayed_rows);
    std::vector<size_type> column_widths(column_order_.size(), 0);

    for (size_type column_index = 0; column_index < column_order_.size(); ++column_index) {
      column_widths[column_index] = column_order_[column_index].size();
    }

    for (size_type row_index = 0; row_index < displayed_rows; ++row_index) {
      for (size_type column_index = 0; column_index < column_order_.size(); ++column_index) {
        const auto rendered = cell_to_string(value(column_order_[column_index], row_index));
        column_widths[column_index] = std::max(column_widths[column_index], rendered.size());
        if (rendered_rows.size() <= row_index) {
          rendered_rows.emplace_back();
          rendered_rows.back().reserve(column_order_.size());
        }
        rendered_rows[row_index].push_back(rendered);
      }
    }

    const auto write_row = [&](const std::vector<std::string>& cells) {
      out << '\n' << '|';
      for (size_type column_index = 0; column_index < cells.size(); ++column_index) {
        out << ' ' << std::left << std::setw(static_cast<int>(column_widths[column_index])) << cells[column_index] << ' '
            << '|';
      }
    };

    write_row(column_order_);

    out << '\n' << '|';
    for (size_type column_index = 0; column_index < column_order_.size(); ++column_index) {
      out << ' ' << std::string(column_widths[column_index], '-') << ' ' << '|';
    }

    for (const auto& rendered_row : rendered_rows) {
      write_row(rendered_row);
    }

    if (displayed_rows < row_count_) {
      out << "\n... (" << (row_count_ - displayed_rows) << " more rows)";
    }

    return out.str();
  }

  /// @brief Renders the frame as a GitHub-flavoured Markdown table, like knitr::kable().
  ///
  /// Numeric columns are right-aligned and string columns left-aligned. Each numeric column
  /// uses one decimal count for all its cells: the fewest (up to `options.digits`, or the
  /// column's entry in `options.column_digits`) that show every value exactly. Columns named in
  /// `options.scientific` print as `d.dde-xx` with that many decimals instead. Nulls print as
  /// `options.na`, and `|` inside strings is escaped. When rows are cut, a final line
  /// `... (k more rows)` follows the table.
  ///
  /// @param options  Decimals, per-column overrides, row limit and null text.
  /// @return  The table, one line per row, ending in a newline.
  /// @throws std::invalid_argument  if a digits setting is negative.
  ///
  /// @par Example
  /// @code
  /// const auto iris = dm::datasets::iris();
  /// std::print("{}", iris.head(2).to_md());
  /// // | Sepal.Length | Sepal.Width | Petal.Length | Petal.Width | Species |
  /// // | ---: | ---: | ---: | ---: | :--- |
  /// // | 5.1 | 3.5 | 1.4 | 0.2 | setosa |
  /// // | 4.9 | 3.0 | 1.4 | 0.2 | setosa |
  /// @endcode
  ///
  /// @par References
  /// - knitr kable(): https://bookdown.org/yihui/rmarkdown-cookbook/kable.html
  /// - GitHub Flavored Markdown tables: https://github.github.com/gfm/#tables-extension-
  ///
  /// @see to_string(), dm::to_md()
  [[nodiscard]] std::string to_md(const MarkdownOptions& options) const {
    if (options.digits < 0) {
      throw std::invalid_argument("DataFrame::to_md: digits must not be negative");
    }
    const auto shown = std::min(options.max_rows, row_count_);
    const auto escape = [](const std::string& text) {
      std::string escaped;
      escaped.reserve(text.size());
      for (const char c : text) {
        if (c == '|') escaped += '\\';
        escaped += c == '\n' ? ' ' : c;
      }
      return escaped;
    };

    std::ostringstream out;
    out << '|';
    for (const auto& name : column_order_) out << ' ' << escape(name) << " |";
    out << "\n|";
    for (const auto& name : column_order_) out << (column_type(name) == ColumnType::Numeric ? " ---: |" : " :--- |");
    out << '\n';

    // One decimal count per numeric column.
    std::vector<int> decimals(column_order_.size(), 0);
    for (size_type c = 0; c < column_order_.size(); ++c) {
      const auto& name = column_order_[c];
      if (column_type(name) != ColumnType::Numeric) continue;
      const auto override_it = options.column_digits.find(name);
      const int max_decimals = override_it == options.column_digits.end() ? options.digits : override_it->second;
      if (max_decimals < 0) {
        throw std::invalid_argument("DataFrame::to_md: digits for column '" + name + "' must not be negative");
      }
      if (options.scientific.contains(name)) {
        decimals[c] = max_decimals;
        continue;
      }
      const auto& column = numeric_column(name);
      int needed = 0;
      for (size_type r = 0; r < shown && needed < max_decimals; ++r) {
        if (column.is_null(r) || !std::isfinite(column.at(r))) continue;
        const double v = column.at(r);
        while (needed < max_decimals) {
          const double scale = std::pow(10.0, needed);
          if (std::abs(std::round(v * scale) / scale - v) <= 1e-9 * std::max(1.0, std::abs(v))) break;
          ++needed;
        }
      }
      decimals[c] = needed;
    }

    for (size_type r = 0; r < shown; ++r) {
      out << '|';
      for (size_type c = 0; c < column_order_.size(); ++c) {
        const auto cell = value(column_order_[c], r);
        out << ' ';
        if (std::holds_alternative<null_type>(cell)) {
          out << options.na;
        } else if (const auto* number = std::get_if<double>(&cell)) {
          const bool scientific = options.scientific.contains(column_order_[c]);
          std::ostringstream num;
          num << (scientific ? std::scientific : std::fixed) << std::setprecision(decimals[c])
              << (*number == 0.0 ? 0.0 : *number);  // print -0.0 as 0
          auto text = num.str();
          if (text.find_first_not_of("-0.") == std::string::npos && text.front() == '-') text.erase(0, 1);  // "-0.0"
          out << text;
        } else {
          out << escape(std::get<std::string>(cell));
        }
        out << " |";
      }
      out << '\n';
    }
    if (shown < row_count_) out << "... (" << (row_count_ - shown) << " more rows)\n";
    return out.str();
  }

  /// @brief to_md() with at most @p digits decimals in numeric columns.
  ///
  /// @param digits  Maximum decimals (default 3); see MarkdownOptions::digits.
  /// @return  The Markdown table, ending in a newline.
  /// @throws std::invalid_argument  if @p digits is negative.
  ///
  /// @par Example
  /// @code
  /// const auto means = dm::datasets::iris() | dm::group_by("Species")
  ///                  | dm::summarise(dm::agg::mean("Petal.Length"));
  /// std::print("{}", means.to_md(2));   // | Species | Petal.Length | ... | setosa | 1.46 |
  /// @endcode
  /// @see to_md(const MarkdownOptions&) const
  [[nodiscard]] std::string to_md(const int digits = 3) const {
    MarkdownOptions options;
    options.digits = digits;
    return to_md(options);
  }

#ifdef DATAMUNGE_HAVE_ARROW
  /// @brief Converts this frame to an Apache Arrow table (only when built with Arrow).
  ///
  /// Numeric columns become nullable `float64` fields and string columns nullable `utf8`
  /// fields, in column order; nulls become Arrow nulls. Data is copied, so the table does
  /// not depend on this frame's lifetime.
  ///
  /// @return  A new single-chunk Arrow table.
  /// @throws std::runtime_error  if an Arrow builder reports an error.
  ///
  /// @par References
  /// - Apache Arrow C++ Table: https://arrow.apache.org/docs/cpp/tables.html
  ///
  /// @see from_arrow()
  [[nodiscard]] std::shared_ptr<arrow::Table> to_arrow() const;

  /// @brief Builds a frame from an Apache Arrow table (only when built with Arrow).
  ///
  /// `double` columns become numeric columns and `utf8`/`large_utf8` columns string columns;
  /// Arrow nulls become nulls and all chunks are concatenated. Data is copied.
  ///
  /// @param table  Source table; must not be null.
  /// @return  A new frame with the table's columns in order.
  /// @throws std::invalid_argument  if @p table is null or has a column of another Arrow type
  ///         (for example int64 or float32), or duplicate/empty column names.
  ///
  /// @par References
  /// - Apache Arrow C++ Table: https://arrow.apache.org/docs/cpp/tables.html
  ///
  /// @see to_arrow()
  static DataFrame from_arrow(const std::shared_ptr<arrow::Table>& table);
#endif

 private:
  /// @brief Hash for a cell variant, used by join lookups and group keys.
  ///
  /// Combines the variant index with the held value's std::hash (null hashes as 0), so a
  /// numeric 1 and a string "1" hash differently.
  struct CellHash {
    /// @brief Hashes one cell.
    /// @param value  Cell to hash.
    /// @return  A hash consistent with CellEqual (every NaN hashes alike).
    [[nodiscard]] std::size_t operator()(const cell_type& value) const {
      const auto index_hash = std::hash<std::size_t>{}(value.index());
      const auto value_hash = std::visit(
          [](const auto& item) -> std::size_t {
            using item_type = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<item_type, null_type>) {
              return 0u;
            } else if constexpr (std::is_floating_point_v<item_type>) {
              // Every NaN (any sign or payload) hashes alike, matching CellEqual.
              return std::isnan(item) ? std::size_t{0x7ff8000000000000ULL} : std::hash<item_type>{}(item);
            } else {
              return std::hash<item_type>{}(item);
            }
          },
          value);
      return index_hash ^ (value_hash + 0x9e3779b97f4a7c15ULL + (index_hash << 6U) + (index_hash >> 2U));
    }
  };

  /// @brief Equality for cell variants: same alternative and equal value; null equals null,
  ///        and NaN equals NaN.
  ///
  /// Treating NaN as one value is what keeps grouping, distinct() and joins well defined: with
  /// plain `==` a NaN key would never find its own group. It matches R/dplyr, where all NaNs
  /// form one group (separate from NA) and `na_matches = "na"` joins NaN to NaN.
  struct CellEqual {
    /// @brief Compares two cells.
    /// @param lhs  First cell.
    /// @param rhs  Second cell.
    /// @return  true if both are null, both are NaN, or both hold equal values of the same type.
    [[nodiscard]] bool operator()(const cell_type& lhs, const cell_type& rhs) const {
      if (const auto* a = std::get_if<double>(&lhs)) {
        if (const auto* b = std::get_if<double>(&rhs)) {
          return *a == *b || (std::isnan(*a) && std::isnan(*b));
        }
      }
      return lhs == rhs;
    }
  };

  /// @brief A group key: one cell per key column, in key-column order.
  using group_key_type = std::vector<cell_type>;

  /// @brief Hash for a whole group key, combining each cell's CellHash (boost-style
  ///        hash_combine).
  struct GroupKeyHash {
    /// @brief Hashes a group key.
    /// @param key  Key cells.
    /// @return  The combined hash; order-sensitive.
    [[nodiscard]] std::size_t operator()(const group_key_type& key) const {
      std::size_t seed = 0;
      for (const auto& value : key) {
        const auto value_hash = CellHash{}(value);
        seed ^= value_hash + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
      }
      return seed;
    }
  };

  /// @brief Equality for whole group keys: same length and CellEqual cell by cell (so NaN keys
  ///        match each other).
  struct GroupKeyEqual {
    /// @brief Compares two group keys.
    /// @param lhs  First key.
    /// @param rhs  Second key.
    /// @return  true if every cell pair is CellEqual.
    [[nodiscard]] bool operator()(const group_key_type& lhs, const group_key_type& rhs) const {
      if (lhs.size() != rhs.size()) return false;
      for (size_type index = 0; index < lhs.size(); ++index) {
        if (!CellEqual{}(lhs[index], rhs[index])) return false;
      }
      return true;
    }
  };

  /// @brief Result of build_groups(): the distinct keys and the rows belonging to each.
  struct GroupPlan {
    /// @brief Distinct keys in order of first appearance.
    std::vector<group_key_type> order;

    /// @brief Row indices (ascending) of each key's group.
    std::unordered_map<group_key_type, std::vector<size_type>, GroupKeyHash, GroupKeyEqual> groups;
  };

  /// @brief Key cell to the ascending row indices holding it; see build_row_lookup().
  using row_lookup_type = std::unordered_map<cell_type, std::vector<size_type>, CellHash, CellEqual>;

  /// @brief Join column plan: (source column name, output column name) pairs in output order.
  using join_plan_type = std::vector<std::pair<std::string, std::string>>;

  /// @brief Shared body of the add_column() overloads: validates and appends a column.
  /// @tparam ColumnValues  numeric_column_type or string_column_type (forwarded).
  /// @param column_name  New column name.
  /// @param values  Column contents; moved in when given an rvalue.
  /// @throws std::invalid_argument  if the name is empty or taken, or the length differs from
  ///         nrows() while the frame already has columns.
  template <typename ColumnValues>
  void add_column_impl(const std::string& column_name, ColumnValues&& values) {
    require_column_name(column_name);
    if (has_column(column_name)) {
      throw std::invalid_argument("DataFrame::add_column duplicate column: " + column_name);
    }

    const size_type incoming_size = values.size();
    if (!column_order_.empty() && incoming_size != row_count_) {
      throw std::invalid_argument("DataFrame::add_column column length mismatch");
    }

    if (column_order_.empty()) {
      row_count_ = incoming_size;
    }

    column_order_.push_back(column_name);
    columns_.emplace(column_name, column_data_type(std::forward<ColumnValues>(values)));
  }

  /// @brief Shared body of the replace_column() overloads: swaps in new contents in place,
  ///        keeping the column's position (the type may change).
  /// @tparam ColumnValues  numeric_column_type or string_column_type (forwarded).
  /// @param column_name  Existing column name.
  /// @param values  New contents, one per row.
  /// @throws std::invalid_argument  if the name is empty or the length differs from nrows().
  /// @throws std::out_of_range  if the column does not exist.
  template <typename ColumnValues>
  void replace_column_impl(const std::string& column_name, ColumnValues&& values) {
    require_column_name(column_name);
    if (!has_column(column_name)) {
      throw std::out_of_range("DataFrame::replace_column missing column: " + column_name);
    }
    if (values.size() != row_count_) {
      throw std::invalid_argument("DataFrame::replace_column column length mismatch");
    }

    columns_.at(column_name) = column_data_type(std::forward<ColumnValues>(values));
  }

  /// @brief Shared body of the mutate() overloads: copies the frame, then adds or replaces
  ///        one column in the copy.
  /// @tparam ColumnValues  numeric_column_type or string_column_type (forwarded).
  /// @param column_name  Column to add or replace.
  /// @param values  One value per row.
  /// @return  The modified copy; this frame is unchanged.
  /// @throws std::invalid_argument  on a length mismatch or an empty name.
  template <typename ColumnValues>
  [[nodiscard]] DataFrame mutate_impl(const std::string& column_name, ColumnValues&& values) const {
    DataFrame result = select(column_order_);
    if (result.has_column(column_name)) {
      result.replace_column(column_name, std::forward<ColumnValues>(values));
    } else {
      result.add_column(column_name, std::forward<ColumnValues>(values));
    }
    return result;
  }

  /// @brief The stored column variant for @p column_name (read-only).
  /// @param column_name  Column name.
  /// @return  A reference valid until the column set changes.
  /// @throws std::out_of_range  if the column does not exist.
  [[nodiscard]] const column_data_type& column_data(const std::string& column_name) const {
    const auto it = columns_.find(column_name);
    if (it == columns_.end()) {
      throw std::out_of_range("DataFrame missing column: " + column_name);
    }
    return it->second;
  }

  /// @brief Mutable overload of column_data(const std::string&) const.
  /// @param column_name  Column name.
  /// @return  A mutable reference valid until the column set changes.
  /// @throws std::out_of_range  if the column does not exist.
  [[nodiscard]] column_data_type& column_data(const std::string& column_name) {
    const auto it = columns_.find(column_name);
    if (it == columns_.end()) {
      throw std::out_of_range("DataFrame missing column: " + column_name);
    }
    return it->second;
  }

  /// @brief Typed read-only access to a column, checking that it holds @p ColumnValues.
  /// @tparam ColumnValues  numeric_column_type or string_column_type.
  /// @param column_name  Column name.
  /// @param expected_type  Type word used in the error message ("numeric" or "string").
  /// @return  The column, valid until the column set changes.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if the column has the other type.
  template <typename ColumnValues>
  [[nodiscard]] const ColumnValues& require_column_type(const std::string& column_name,
                                                        const char* expected_type) const {
    const auto& data = column_data(column_name);
    const auto* values = std::get_if<ColumnValues>(&data);
    if (values == nullptr) {
      throw std::invalid_argument(std::string("DataFrame column has wrong type, expected ") + expected_type +
                                  ": " + column_name);
    }
    return *values;
  }

  /// @brief Mutable overload of the const require_column_type().
  /// @tparam ColumnValues  numeric_column_type or string_column_type.
  /// @param column_name  Column name.
  /// @param expected_type  Type word used in the error message.
  /// @return  The mutable column, valid until the column set changes.
  /// @throws std::out_of_range  if the column does not exist.
  /// @throws std::invalid_argument  if the column has the other type.
  template <typename ColumnValues>
  [[nodiscard]] ColumnValues& require_column_type(const std::string& column_name, const char* expected_type) {
    auto& data = column_data(column_name);
    auto* values = std::get_if<ColumnValues>(&data);
    if (values == nullptr) {
      throw std::invalid_argument(std::string("DataFrame column has wrong type, expected ") + expected_type +
                                  ": " + column_name);
    }
    return *values;
  }

  /// @brief Rejects an empty column name.
  /// @param column_name  Candidate name.
  /// @throws std::invalid_argument  if @p column_name is empty.
  static void require_column_name(const std::string& column_name) {
    if (column_name.empty()) {
      throw std::invalid_argument("DataFrame column names must be non-empty");
    }
  }

  /// @brief Checks that @p row_value fits this frame's schema before append_row() adds it.
  /// @param row_value  Row to check: exactly one cell per column, each null or of the
  ///        column's type.
  /// @throws std::invalid_argument  if the frame has no columns, the row's width differs, a
  ///         column is missing from the row, or a cell has the wrong type.
  void require_complete_row(const Row& row_value) const {
    if (column_order_.empty()) {
      throw std::invalid_argument("DataFrame::append_row requires at least one column");
    }
    if (row_value.values.size() != column_order_.size()) {
      throw std::invalid_argument("DataFrame::append_row row width mismatch");
    }

    for (const auto& name : column_order_) {
      const auto cell_it = row_value.values.find(name);
      if (cell_it == row_value.values.end()) {
        throw std::invalid_argument("DataFrame::append_row missing column: " + name);
      }

      const auto& data = columns_.at(name);
      const bool type_matches = std::holds_alternative<null_type>(cell_it->second) ||
                                (std::holds_alternative<numeric_column_type>(data) &&
                                 std::holds_alternative<double>(cell_it->second)) ||
                                (std::holds_alternative<string_column_type>(data) &&
                                 std::holds_alternative<std::string>(cell_it->second));
      if (!type_matches) {
        throw std::invalid_argument("DataFrame::append_row cell type mismatch for column: " + name);
      }
    }
  }

  /// @brief Appends one cell to a stored column, pushing a null for a null cell.
  /// @param column  Column to grow by one.
  /// @param cell  Value to append; its type must already match (see require_complete_row()),
  ///        otherwise std::get throws std::bad_variant_access.
  static void append_cell(column_data_type& column, const cell_type& cell) {
    if (auto* values = std::get_if<numeric_column_type>(&column)) {
      if (std::holds_alternative<null_type>(cell)) {
        values->push_null();
      } else {
        values->push_back(std::get<double>(cell));
      }
      return;
    }

    auto& values = std::get<string_column_type>(column);
    if (std::holds_alternative<null_type>(cell)) {
      values.push_null();
    } else {
      values.push_back(std::get<std::string>(cell));
    }
  }

  /// @brief An empty frame with this frame's column names, order and types.
  /// @return  A zero-row frame.
  [[nodiscard]] DataFrame clone_schema() const {
    DataFrame result;
    for (const auto& name : column_order_) {
      append_schema_column(result, name, column_type(name));
    }
    return result;
  }

  /// @brief Adds an empty column of the given type to @p target.
  /// @param target  Frame to extend; normally still empty (zero rows).
  /// @param name  New column name.
  /// @param type  Column type.
  /// @throws std::invalid_argument  if @p name is empty or already in @p target, or
  ///         @p target already has rows.
  static void append_schema_column(DataFrame& target, const std::string& name, const ColumnType type) {
    if (type == ColumnType::Numeric) {
      target.add_column(name, numeric_column_type{});
    } else {
      target.add_column(name, string_column_type{});
    }
  }

  /// @brief Copies the contiguous rows [@p begin_index, @p end_index) of every column; used
  ///        by head(), tail() and slice().
  /// @param begin_index  First row to copy.
  /// @param end_index  One past the last row; must satisfy begin_index <= end_index <= nrows().
  /// @return  A new frame with end_index - begin_index rows (no columns if this frame has
  ///          none).
  /// @throws std::out_of_range  if the range is invalid (from NullableColumn::slice()).
  [[nodiscard]] DataFrame slice_rows(const size_type begin_index, const size_type end_index) const {
    DataFrame result;
    for (const auto& name : column_order_) {
      const auto& data = column_data(name);
      std::visit(
          [&](const auto& column) {
            result.add_column(name, column.slice(begin_index, end_index));
          },
          data);
    }
    return result;
  }

  /// @brief Copies the rows at @p indices, in that order (repeats allowed).
  /// @param indices  Row indices into this frame.
  /// @return  A new frame with indices.size() rows and this frame's schema.
  /// @throws std::out_of_range  if an index is >= nrows().
  [[nodiscard]] DataFrame take_rows_impl(const std::vector<size_type>& indices) const {
    DataFrame result = clone_schema();
    for (const auto row_index : indices) {
      result.append_row(row(row_index));
    }
    return result;
  }

  /// @brief Validates grouping keys for the group_by family.
  /// @param key_columns  Key column names.
  /// @throws std::invalid_argument  if @p key_columns is empty.
  /// @throws std::out_of_range  if a key column does not exist.
  void require_group_by_keys(const std::vector<std::string>& key_columns) const {
    if (key_columns.empty()) {
      throw std::invalid_argument("DataFrame group_by requires at least one key column");
    }
    for (const auto& column_name : key_columns) {
      (void)column_data(column_name);
    }
  }

  /// @brief Validates the value columns of group_by_sum()/group_by_mean().
  /// @param value_columns  Column names that must exist and be numeric.
  /// @throws std::invalid_argument  if the list is empty or a column is a string column.
  /// @throws std::out_of_range  if a column does not exist.
  void require_numeric_columns(const std::vector<std::string>& value_columns) const {
    if (value_columns.empty()) {
      throw std::invalid_argument("DataFrame aggregation requires at least one value column");
    }
    for (const auto& column_name : value_columns) {
      (void)numeric_column(column_name);
    }
  }

  /// @brief Applies one numeric AggFunc to a group's non-null values (summarise() helper).
  ///
  /// Median averages the two middle values for an even count; StdDev uses the n - 1
  /// denominator and returns 0 for a single value.
  ///
  /// @param column_name  Numeric column to aggregate.
  /// @param rows  The group's row indices.
  /// @param func  Sum, Mean, Min, Max, Median or StdDev.
  /// @return  The aggregate, or std::nullopt if every value in the group is null.
  /// @throws std::invalid_argument  for Count/NDistinct (handled by the caller), or if the
  ///         column is a string column.
  [[nodiscard]] std::optional<double> aggregate_numeric(const std::string& column_name,
                                                         const std::vector<size_type>& rows, const AggFunc func) const {
    std::vector<double> values;
    values.reserve(rows.size());
    for (const auto row_index : rows) {
      const auto value = optional_double_at(column_name, row_index);
      if (value.has_value()) {
        values.push_back(*value);
      }
    }
    if (values.empty()) {
      return std::nullopt;
    }

    switch (func) {
      case AggFunc::Sum: {
        double sum = 0.0;
        for (const auto v : values) sum += v;
        return sum;
      }
      case AggFunc::Mean: {
        double sum = 0.0;
        for (const auto v : values) sum += v;
        return sum / static_cast<double>(values.size());
      }
      case AggFunc::Min:
        return *std::min_element(values.begin(), values.end());
      case AggFunc::Max:
        return *std::max_element(values.begin(), values.end());
      case AggFunc::Median: {
        std::sort(values.begin(), values.end());
        const auto mid = values.size() / 2;
        if (values.size() % 2 == 0) {
          return (values[mid - 1] + values[mid]) / 2.0;
        }
        return values[mid];
      }
      case AggFunc::StdDev: {
        if (values.size() < 2) {
          return 0.0;
        }
        double mean = 0.0;
        for (const auto v : values) mean += v;
        mean /= static_cast<double>(values.size());
        double variance = 0.0;
        for (const auto v : values) variance += (v - mean) * (v - mean);
        variance /= static_cast<double>(values.size() - 1);
        return std::sqrt(variance);
      }
      default:
        throw std::invalid_argument("DataFrame::summarise unsupported aggregation for numeric column");
    }
  }

  /// @brief Applies Min or Max (lexicographic) to a group's non-null strings (summarise()
  ///        helper).
  /// @param column_name  String column to aggregate.
  /// @param rows  The group's row indices.
  /// @param func  Min or Max.
  /// @return  The aggregate, or std::nullopt if every value in the group is null (checked
  ///          before @p func, so an all-null group never throws).
  /// @throws std::invalid_argument  if @p func is not Min or Max.
  [[nodiscard]] std::optional<std::string> aggregate_string(const std::string& column_name,
                                                             const std::vector<size_type>& rows, const AggFunc func) const {
    std::vector<std::string> values;
    values.reserve(rows.size());
    for (const auto row_index : rows) {
      const auto value = optional_string_at(column_name, row_index);
      if (value.has_value()) {
        values.push_back(*value);
      }
    }
    if (values.empty()) {
      return std::nullopt;
    }

    switch (func) {
      case AggFunc::Min:
        return *std::min_element(values.begin(), values.end());
      case AggFunc::Max:
        return *std::max_element(values.begin(), values.end());
      default:
        throw std::invalid_argument("DataFrame::summarise: aggregation not supported for string columns");
    }
  }

  /// @brief n_distinct() restricted to one group's rows: distinct values, plus one if any
  ///        cell is null.
  /// @param column_name  Column of either type.
  /// @param rows  The group's row indices.
  /// @return  The distinct count.
  /// @throws std::out_of_range  if the column does not exist.
  [[nodiscard]] size_type group_n_distinct(const std::string& column_name, const std::vector<size_type>& rows) const {
    const auto& data = column_data(column_name);
    return std::visit(
        [&](const auto& column) -> size_type {
          using column_type = std::decay_t<decltype(column)>;
          using value_type = typename column_type::value_type;
          std::unordered_set<value_type> seen;
          bool has_null = false;
          bool has_nan = false;  // counted once, as in n_distinct()
          for (const auto row_index : rows) {
            const auto value = column.optional_at(row_index);
            if (!value.has_value()) {
              has_null = true;
            } else if constexpr (std::is_floating_point_v<value_type>) {
              if (std::isnan(*value)) {
                has_nan = true;
              } else {
                seen.insert(*value);
              }
            } else {
              seen.insert(*value);
            }
          }
          return seen.size() + (has_null ? 1 : 0) + (has_nan ? 1 : 0);
        },
        data);
  }

  /// @brief The key cells of one row.
  /// @param key_columns  Key column names.
  /// @param row_index  Row to read.
  /// @return  One cell per key column, in order.
  [[nodiscard]] group_key_type group_key_for_row(const std::vector<std::string>& key_columns,
                                                 const size_type row_index) const {
    group_key_type key;
    key.reserve(key_columns.size());
    for (const auto& column_name : key_columns) {
      key.push_back(value(column_name, row_index));
    }
    return key;
  }

  /// @brief Splits the rows into groups by key (the "split" of split-apply-combine).
  ///
  /// One hash-table pass: expected O(rows x keys). Groups are ordered by first appearance
  /// and each group's rows stay ascending. All NaN key cells belong to one group (see
  /// CellEqual), separate from the null group.
  ///
  /// @param key_columns  Key column names (already validated).
  /// @return  The group plan.
  [[nodiscard]] GroupPlan build_groups(const std::vector<std::string>& key_columns) const {
    GroupPlan plan;
    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      auto key = group_key_for_row(key_columns, row_index);
      auto it = plan.groups.find(key);
      if (it == plan.groups.end()) {
        plan.order.push_back(key);
        plan.groups.emplace(std::move(key), std::vector<size_type>{row_index});
      } else {
        it->second.push_back(row_index);
      }
    }
    return plan;
  }

  /// @brief An empty frame holding just the key columns, with this frame's types.
  /// @param key_columns  Key column names.
  /// @return  A zero-row frame to which group rows are appended.
  /// @throws std::out_of_range  if a key column does not exist.
  [[nodiscard]] DataFrame make_group_result_schema(const std::vector<std::string>& key_columns) const {
    DataFrame result;
    for (const auto& column_name : key_columns) {
      append_schema_column(result, column_name, column_type(column_name));
    }
    return result;
  }

  /// @brief Appends one row of key cells to a frame built by make_group_result_schema().
  /// @param result  Frame containing exactly the key columns.
  /// @param key_columns  Key column names, matching @p key.
  /// @param key  Cells to append.
  /// @throws std::invalid_argument  if @p result has no columns (an empty key list).
  static void append_group_key_row(DataFrame& result, const std::vector<std::string>& key_columns,
                                   const group_key_type& key) {
    Row row;
    for (size_type index = 0; index < key_columns.size(); ++index) {
      row.values.emplace(key_columns[index], key[index]);
    }
    result.append_row(row);
  }

  /// @brief Hash index from each key cell to the rows holding it, used by join().
  /// @param key_column  Column to index (null cells are indexed too).
  /// @return  The lookup; row lists are ascending.
  /// @throws std::out_of_range  if the column does not exist.
  [[nodiscard]] row_lookup_type build_row_lookup(const std::string& key_column) const {
    row_lookup_type lookup;
    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      lookup[value(key_column, row_index)].push_back(row_index);
    }
    return lookup;
  }

  /// @brief Computes the output column-name plan for both sides of a join: any column name
  ///        present on both sides (other than the merged key, when `merged_key` is true) is
  ///        suffixed on both sides so join() never produces duplicate output names.
  /// @param right  Right-hand frame.
  /// @param left_key  Left key column.
  /// @param right_key  Right key column.
  /// @param merged_key  Whether the keys share a name, in which case right's key is dropped.
  /// @param left_suffix  Suffix for clashing left names.
  /// @param right_suffix  Suffix for clashing right names.
  /// @return  (left plan, right plan), each a list of (source name, output name) pairs.
  [[nodiscard]] std::pair<join_plan_type, join_plan_type> build_join_plans(const DataFrame& right,
                                                                           const std::string& left_key,
                                                                           const std::string& right_key,
                                                                           const bool merged_key,
                                                                           const std::string& left_suffix,
                                                                           const std::string& right_suffix) const {
    std::vector<std::string> right_output_names;
    right_output_names.reserve(right.column_order_.size());
    for (const auto& name : right.column_order_) {
      if (merged_key && name == right_key) {
        continue;
      }
      right_output_names.push_back(name);
    }

    const std::unordered_set<std::string> right_name_set(right_output_names.begin(), right_output_names.end());
    const std::unordered_set<std::string> left_name_set(column_order_.begin(), column_order_.end());

    join_plan_type left_plan;
    left_plan.reserve(column_order_.size());
    for (const auto& name : column_order_) {
      const bool collides = right_name_set.find(name) != right_name_set.end();
      left_plan.emplace_back(name, collides ? name + left_suffix : name);
    }

    join_plan_type right_plan;
    right_plan.reserve(right_output_names.size());
    for (const auto& name : right_output_names) {
      const bool collides = left_name_set.find(name) != left_name_set.end();
      right_plan.emplace_back(name, collides ? name + right_suffix : name);
    }

    return {std::move(left_plan), std::move(right_plan)};
  }

  /// @brief Appends one merged row to a join result: `left_source`/`right_source` are nullptr
  ///        for the unmatched side of an outer join (producing nulls), except the merged key
  ///        column itself, which is filled from whichever side is present.
  /// @param result  Join output frame being built.
  /// @param left_plan  Left (source, output) column names.
  /// @param right_plan  Right (source, output) column names.
  /// @param left_source  Left frame, or nullptr for an unmatched right row.
  /// @param left_row  Row in @p left_source (ignored when it is nullptr).
  /// @param right_source  Right frame, or nullptr for an unmatched left row.
  /// @param right_row  Row in @p right_source (ignored when it is nullptr).
  /// @param merged_key  Whether the key column is shared (see build_join_plans()).
  /// @param left_key  Left key column name.
  /// @param right_key  Right key column name.
  /// @throws std::invalid_argument  if the merged row does not fit @p result's schema.
  static void append_join_row(DataFrame& result, const join_plan_type& left_plan, const join_plan_type& right_plan,
                              const DataFrame* left_source, const size_type left_row, const DataFrame* right_source,
                              const size_type right_row, const bool merged_key, const std::string& left_key,
                              const std::string& right_key) {
    Row merged;
    merged.values.reserve(left_plan.size() + right_plan.size());
    for (const auto& [source_name, output_name] : left_plan) {
      if (left_source != nullptr) {
        merged.values[output_name] = left_source->value(source_name, left_row);
      } else if (merged_key && source_name == left_key && right_source != nullptr) {
        merged.values[output_name] = right_source->value(right_key, right_row);
      } else {
        merged.values[output_name] = cell_type(null_type{});
      }
    }
    for (const auto& [source_name, output_name] : right_plan) {
      merged.values[output_name] = right_source != nullptr ? right_source->value(source_name, right_row) : cell_type(null_type{});
    }
    result.append_row(merged);
  }

  /// @brief One output row for bind_rows(): @p source's cells, with nulls for columns it
  ///        lacks.
  /// @param columns  The combined column names.
  /// @param source  Frame the row comes from (not null).
  /// @param row_index  Row in @p source.
  /// @return  A Row with one cell per entry of @p columns.
  static Row build_bind_row(const std::vector<std::string>& columns, const DataFrame* source, const size_type row_index) {
    Row row;
    for (const auto& name : columns) {
      row.values[name] = source->has_column(name) ? source->value(name, row_index) : cell_type(null_type{});
    }
    return row;
  }

  /// @brief Checks that @p other has the same column names, order and types (concat_rows()
  ///        helper).
  /// @param other  Frame to compare.
  /// @throws std::invalid_argument  if the column order or a column type differs.
  void require_same_schema(const DataFrame& other) const {
    if (column_order_ != other.column_order_) {
      throw std::invalid_argument("DataFrame::concat_rows requires identical column order");
    }
    for (const auto& column_name : column_order_) {
      if (column_type(column_name) != other.column_type(column_name)) {
        throw std::invalid_argument("DataFrame::concat_rows requires identical column types");
      }
    }
  }

  /// @brief Text for one cell in to_string(): "NA" for null, the string itself, or the
  ///        number in default stream format (6 significant digits).
  /// @param cell  Cell to render.
  /// @return  The rendered text.
  static std::string cell_to_string(const cell_type& cell) {
    return std::visit(
        [](const auto& value) -> std::string {
          using value_type = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<value_type, null_type>) {
            return "NA";
          } else if constexpr (std::is_same_v<value_type, std::string>) {
            return value;
          } else {
            std::ostringstream cell_stream;
            cell_stream << value;
            return cell_stream.str();
          }
        },
        cell);
  }

  /// @brief Column names in display order.
  std::vector<std::string> column_order_;

  /// @brief Column storage keyed by name; holds exactly the names in column_order_.
  std::unordered_map<std::string, column_data_type> columns_;

  /// @brief Number of rows (set by the first column added; every column has this length).
  size_type row_count_{0};
};

/// @brief dplyr-style grouped-DataFrame handle returned by DataFrame::group_by(): pairs a copy
///        of the source DataFrame with a set of key columns, deferring aggregation until
///        summarise()/count() is called. Holding the source by value (rather than a reference or
///        pointer) keeps chains like `df.select(...).group_by(...).summarise(...)` safe, since
///        the intermediate select() result is a temporary that would otherwise dangle.
///
/// All aggregations return one row per group with the key columns first, in order of each
/// group's first appearance in the source.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// const auto grouped = tooth | dm::group_by("supp");   // GroupedDataFrame
/// const auto sizes = grouped.count();                   // VC 30, OJ 30
/// const auto means = grouped | dm::summarise_each(dm::mean);
/// @endcode
/// @see DataFrame::group_by(), dm::group_by()
class GroupedDataFrame {
 public:
  /// @brief Pairs a source frame with grouping keys; prefer DataFrame::group_by(), which also
  ///        validates the keys.
  /// @param source  Frame to group (stored by value).
  /// @param keys  Grouping column names.
  GroupedDataFrame(DataFrame source, std::vector<std::string> keys) : source_(std::move(source)), keys_(std::move(keys)) {}

  /// @brief dplyr::summarise() with explicit aggregations; see DataFrame::summarise().
  ///
  /// Groups appear in first-appearance order; numeric aggregates skip nulls. The stored
  /// source frame is not modified, so one handle can be summarised several times.
  ///
  /// @param specs  One AggSpec per output column.
  /// @return  One row per group: the keys, then one column per spec.
  /// @throws std::invalid_argument  if @p specs is empty, an aggregation does not suit the
  ///         column type, or output names collide.
  /// @throws std::out_of_range  if a spec column does not exist.
  ///
  /// @par Example
  /// @code
  /// using DataFrame = dm::dstruct::DataFrame;
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto stats = tooth.group_by({"supp"}).summarise(
  ///     {{"len", DataFrame::AggFunc::Max, "longest"}, {"", DataFrame::AggFunc::Count, ""}});
  /// std::print("{}\n", stats);   // 2 x 3: supp, longest, n (VC 33.9 30, OJ 30.9 30)
  /// @endcode
  ///
  /// @par References
  /// - dplyr summarise(): https://dplyr.tidyverse.org/reference/summarise.html
  ///
  /// @see DataFrame::summarise(), count(), summarise_each()
  [[nodiscard]] DataFrame summarise(const std::vector<DataFrame::AggSpec>& specs) const {
    return source_.summarise(keys_, specs);
  }

  /// @brief Number of rows in each group, like dplyr::count().
  /// @param count_column_name  Name of the count column (default "n").
  /// @return  One row per group: the keys, then the count column.
  /// @throws std::invalid_argument  if @p count_column_name is already a column.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// std::print("{}\n", tooth.group_by({"dose"}).count("rows"));   // 0.5/1/2 -> 20 each
  /// @endcode
  /// @see dm::count(), DataFrame::count()
  [[nodiscard]] DataFrame count(const std::string& count_column_name = "n") const {
    return source_.count(keys_, count_column_name);
  }

  /// @brief Applies `func` to each non-key numeric column per group; see DataFrame::summarise_each().
  ///
  /// Non-key string columns are dropped; a group with a null in a column gets a null result.
  ///
  /// @tparam Func  Reduction callable type.
  /// @param func  Callable taking a group's values (std::vector<double> or dm::Vector<>) and
  ///              returning a number, e.g. `dm::mean`.
  /// @return  One row per group: the keys, then the summarised numeric columns.
  ///
  /// @par Example
  /// @code
  /// const auto tooth = dm::datasets::tooth_growth();
  /// const auto totals = tooth.group_by({"supp"}).summarise_each(dm::sum);   // VC len 508.9
  /// @endcode
  /// @see dm::summarise_each()
  template <typename Func>
  [[nodiscard]] DataFrame summarise_each(Func&& func) const {
    return source_.summarise_each(keys_, std::forward<Func>(func));
  }

  /// @brief The grouping column names, in order.
  /// @return  A reference valid while this object lives.
  [[nodiscard]] const std::vector<std::string>& keys() const { return keys_; }

 private:
  /// @brief Copy of the grouped frame.
  DataFrame source_;
  /// @brief Grouping column names.
  std::vector<std::string> keys_;
};

/// @brief Out-of-line definition of DataFrame::group_by(); it needs the complete
///        GroupedDataFrame type. Validates the keys, then copies this frame into the handle.
///        Parameters, return value and exceptions are documented on the declaration.
inline GroupedDataFrame DataFrame::group_by(const std::vector<std::string>& key_columns) const {
  require_group_by_keys(key_columns);
  return GroupedDataFrame(*this, key_columns);
}

} // namespace datamunge::dstruct

namespace datamunge {

/// @brief R-style `head(x, n)`: the first `n` rows of `frame`.
/// @param frame Source frame; it is not modified.
/// @param n     Rows to keep (default 6, as in R). A negative `n` keeps every row except the
///              last `|n|`. Values past either end are clamped, so the result may be empty.
/// @return      A new frame with the same columns.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// std::print("{}\n", dm::head(tooth));   // first 6 rows
/// std::print("{}\n", dm::head(tooth, -57).nrows());   // 3: all but the last 57
/// @endcode
/// @see dm::tail(), DataFrame::head()
[[nodiscard]] inline dstruct::DataFrame head(const dstruct::DataFrame& frame, const std::ptrdiff_t n = 6) {
  const auto rows = static_cast<std::ptrdiff_t>(frame.nrows());
  const auto kept = n >= 0 ? std::min(n, rows) : std::max<std::ptrdiff_t>(rows + n, 0);
  return frame.head(static_cast<dstruct::DataFrame::size_type>(kept));
}

/// @brief R-style `tail(x, n)`: the last `n` rows of `frame`.
/// @param frame Source frame; it is not modified.
/// @param n     Rows to keep (default 6, as in R). A negative `n` keeps every row except the
///              first `|n|`. Values past either end are clamped, so the result may be empty.
/// @return      A new frame with the same columns.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// std::print("{}\n", dm::tail(tooth).nrows());        // 6
/// std::print("{}\n", dm::tail(tooth, -57).nrows());   // 3: all but the first 57
/// @endcode
///
/// @par References
/// - R utils::head/tail: https://stat.ethz.ch/R-manual/R-devel/library/utils/html/head.html
///
/// @see dm::head(), DataFrame::tail()
[[nodiscard]] inline dstruct::DataFrame tail(const dstruct::DataFrame& frame, const std::ptrdiff_t n = 6) {
  const auto rows = static_cast<std::ptrdiff_t>(frame.nrows());
  const auto kept = n >= 0 ? std::min(n, rows) : std::max<std::ptrdiff_t>(rows + n, 0);
  return frame.tail(static_cast<dstruct::DataFrame::size_type>(kept));
}

} // namespace datamunge
