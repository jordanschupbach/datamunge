#pragma once

/// @file pipe.hpp
/// @brief dplyr-style `|` pipelines over DataFrames.
///
/// Each verb (group_by(), summarise_each(), count(), filter_rows(), mutate(), arrange()) builds
/// a small step object; `frame | step` applies it and returns a new DataFrame (or a
/// GroupedDataFrame after group_by()), leaving the input untouched. Every verb is a thin wrapper
/// over a DataFrame member (DataFrame::group_by(), DataFrame::filter(), ...), so pipelines and
/// method chains are interchangeable.
///
/// @code
/// auto tooth = dm::datasets::tooth_growth();
/// auto means = tooth | dm::group_by("supp", "dose") | dm::summarise_each(dm::mean);
/// auto sizes = tooth | dm::group_by("supp") | dm::count();
/// // any callable is a step, including generic lambdas
/// auto top = tooth | [](const auto& df) { return dm::head(df, 3); };
///
/// auto high = tooth
///   | dm::filter_rows([](const auto& row) { return row.get_double("len") > 10; })
///   | dm::mutate("len_per_dose", [](const auto& row) { return row.get_double("len") / row.get_double("dose"); })
///   | dm::arrange("supp", dm::desc("len"));
/// @endcode

#include <datamunge/dstruct/dataframe.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge {

/// @brief Pipeline step created by group_by(); `frame | step` yields a GroupedDataFrame.
///
/// Normally built with dm::group_by() rather than directly.
/// @see dm::group_by(), DataFrame::group_by()
struct GroupBy {
  /// @brief Grouping column names, in order.
  std::vector<std::string> keys;
};

/// @brief Pipeline step created by summarise_each(); reduces each group's numeric columns.
///
/// Normally built with dm::summarise_each(); `grouped | step` yields a DataFrame.
/// @tparam Func Reduction callable, e.g. a pointer to `dm::mean`.
/// @see dm::summarise_each(), GroupedDataFrame::summarise_each()
template <typename Func>
struct SummariseEach {
  /// @brief Reduction applied to each non-key numeric column of each group.
  Func func;
};

/// @brief Pipeline step created by count(); counts rows per group.
///
/// Normally built with dm::count(); `grouped | step` yields a DataFrame.
/// @see dm::count(), GroupedDataFrame::count()
struct Count {
  /// @brief Name of the result column.
  std::string column_name{"n"};
};

/// @brief Pipeline step created by filter_rows(); keeps rows where the predicate is true.
///
/// Normally built with dm::filter_rows(); `frame | step` yields a DataFrame.
/// @tparam Predicate Callable taking a `const dstruct::DataFrame::Row&` and returning bool.
/// @see dm::filter_rows(), DataFrame::filter()
template <typename Predicate>
struct Filter {
  /// @brief Row predicate.
  Predicate predicate;
};

/// @brief Pipeline step created by mutate(); adds or replaces one column.
///
/// Normally built with dm::mutate(); `frame | step` yields a DataFrame.
/// @tparam Value A row callable or a ready column vector; see mutate().
/// @see dm::mutate(), DataFrame::mutate()
template <typename Value>
struct Mutate {
  /// @brief Name of the column to add or replace.
  std::string column_name;
  /// @brief Row callable or column values.
  Value value;
};

/// @brief One sort key for arrange(): a column name and its direction.
///
/// A plain column name converts implicitly to an ascending key; dm::desc() makes a
/// descending one.
///
/// @par Example
/// @code
/// const auto sorted = dm::datasets::tooth_growth() | dm::arrange("supp", dm::desc("len"));
/// @endcode
/// @see dm::desc(), dm::arrange()
struct SortKey {
  /// @brief Builds an ascending key from a column name.
  /// @param name Column name.
  SortKey(std::string name) : column(std::move(name)) {}  // NOLINT: implicit so "col" works in arrange()
  /// @brief Builds an ascending key from a string literal.
  /// @param name Column name.
  SortKey(const char* name) : column(name) {}  // NOLINT: implicit so "col" works in arrange()
  /// @brief Builds a key with an explicit direction.
  /// @param name      Column name.
  /// @param ascending false for descending order.
  SortKey(std::string name, bool ascending) : column(std::move(name)), ascending(ascending) {}

  /// @brief Column to sort by.
  std::string column;
  /// @brief true for ascending (the default), false for descending.
  bool ascending{true};
};

/// @brief Pipeline step created by arrange(); sorts rows by one or more keys.
///
/// Normally built with dm::arrange(); `frame | step` yields a DataFrame.
/// @see dm::arrange(), DataFrame::arrange()
struct Arrange {
  /// @brief Sort keys, most significant first.
  std::vector<SortKey> keys;
};

/// @brief Groups rows by one or more key columns, like dplyr::group_by().
///
/// `frame | dm::group_by(...)` returns a GroupedDataFrame; follow it with summarise_each(),
/// count() or GroupedDataFrame::summarise(). Groups are formed from the distinct combinations
/// of the key values (a null is its own key value) and appear in the result in order of first
/// appearance -- they are not sorted; add arrange() for that. Key columns may be numeric or
/// string.
///
/// @tparam Keys  Types convertible to std::string.
/// @param keys  Column names (string literals or std::string), at least one.
/// @return  A GroupBy step; nothing is computed until it is piped.
/// @throws std::out_of_range  (when piped) if a key column does not exist.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// const auto means = tooth | dm::group_by("supp", "dose") | dm::summarise_each(dm::mean);
/// std::print("{}\n", means);   // 6 x 3 frame: supp, dose, len (VC rows first)
/// @endcode
/// @see dm::summarise_each(), dm::count(), DataFrame::group_by()
template <typename... Keys,
          std::enable_if_t<(sizeof...(Keys) > 0) && (std::is_convertible_v<Keys, std::string> && ...), int> = 0>
[[nodiscard]] GroupBy group_by(Keys&&... keys) {
  return GroupBy{std::vector<std::string>{std::string(std::forward<Keys>(keys))...}};
}

/// @brief Same as group_by(Keys&&...) but takes the key names as a vector, for keys chosen
///        at run time.
/// @param keys  Column names, at least one.
/// @return  A GroupBy step.
/// @throws std::invalid_argument  (when piped) if @p keys is empty.
/// @throws std::out_of_range  (when piped) if a key column does not exist.
/// @see dm::group_by()
[[nodiscard]] inline GroupBy group_by(std::vector<std::string> keys) { return GroupBy{std::move(keys)}; }

/// @brief Reduces every non-key numeric column of each group with one function, like
///        dplyr's `summarise(across(where(is.numeric), func))`.
///
/// `grouped | dm::summarise_each(func)` returns one row per group: the key columns first,
/// then each numeric non-key column (in the frame's column order) holding `func` of that
/// group's values. Non-key string columns are dropped. Like R with `na.rm = FALSE`, a group
/// with a null in a column gets a null result for that column.
///
/// @tparam Func  Reduction callable type.
/// @param func  Callable taking the group's values (std::vector<double> or dm::Vector<>) and
///              returning a number. Function names such as `dm::mean` work directly.
/// @return  A SummariseEach step to pipe a GroupedDataFrame into.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// const auto means = tooth | dm::group_by("supp") | dm::summarise_each(dm::mean);
/// std::print("{}\n", means);   // 2 x 3: VC len 16.9633, OJ len 20.6633 (dose 1.16667)
/// const auto spread = tooth | dm::group_by("supp") | dm::summarise_each(dm::sd);
/// @endcode
/// @see dm::group_by(), DataFrame::summarise_each(), DataFrame::summarise()
template <typename Func>
[[nodiscard]] SummariseEach<std::decay_t<Func>> summarise_each(Func&& func) {
  return SummariseEach<std::decay_t<Func>>{std::forward<Func>(func)};
}

/// @brief Keeps the rows where `predicate(row)` is true, like dplyr::filter().
///
/// Not named `filter` because `dm::filter` is the Kalman/particle filter namespace. The
/// predicate sees each row as a DataFrame::Row; read cells with Row::get_double(),
/// Row::get_string() or the null-safe Row::get_optional_double(). Row order is preserved and
/// all columns are kept; if no row matches, the result has the same columns and 0 rows.
///
/// @tparam Predicate  Callable type.
/// @param predicate  Callable taking a `const dstruct::DataFrame::Row&` and returning bool.
/// @return  A Filter step to pipe a DataFrame into.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// const auto long_teeth = tooth
///   | dm::filter_rows([](const auto& row) { return row.get_double("len") > 25; });
/// std::print("{}\n", long_teeth.nrows());   // 16
/// @endcode
/// @see DataFrame::filter(), DataFrame::drop_nulls()
template <typename Predicate>
[[nodiscard]] Filter<std::decay_t<Predicate>> filter_rows(Predicate&& predicate) {
  return Filter<std::decay_t<Predicate>>{std::forward<Predicate>(predicate)};
}

/// @brief Adds `column_name`, or replaces it if it exists, like dplyr::mutate().
///
/// A new column is appended at the end; an existing column keeps its position but takes the
/// new values (and type). With a row callable, the column is numeric when the callable returns
/// a number or `std::optional<double>`, and string when it returns a string or
/// `std::optional<std::string>`.
///
/// @tparam Value  Row callable or column vector type.
/// @param column_name  Column to add or replace (must not be empty).
/// @param value  Either a callable taking a `const dstruct::DataFrame::Row&` and returning
///               a number, a string, or a `std::optional` of either (`std::nullopt` gives a
///               null), or a ready `std::vector` of doubles/strings (optionally of
///               `std::optional`s) with one value per row.
/// @return  A Mutate step to pipe a DataFrame into.
/// @throws std::invalid_argument  (when piped) if a ready vector's length differs from the
///         frame's row count.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth() | dm::mutate("len_per_dose",
///   [](const auto& row) { return row.get_double("len") / row.get_double("dose"); });
/// std::print("{}\n", tooth.head(3));   // 3 x 4; len_per_dose 8.4, 23, 14.6
/// @endcode
/// @see DataFrame::mutate(), DataFrame::mutate_with()
template <typename Value>
[[nodiscard]] Mutate<std::decay_t<Value>> mutate(std::string column_name, Value&& value) {
  return Mutate<std::decay_t<Value>>{std::move(column_name), std::forward<Value>(value)};
}

/// @brief Marks an arrange() key as descending, like dplyr::desc().
///
/// Nulls still sort last under a descending key.
///
/// @param column  Column name.
/// @return  A descending SortKey for dm::arrange().
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// const auto longest = dm::head(tooth | dm::arrange(dm::desc("len")), 1);   // len 33.9
/// @endcode
/// @see dm::arrange()
[[nodiscard]] inline SortKey desc(std::string column) { return SortKey(std::move(column), false); }

/// @brief Sorts rows by one or more columns, like dplyr::arrange().
///
/// Keys are compared most significant first; numeric columns compare by value and string
/// columns lexicographically (so "OJ" < "VC"). The sort is stable (ties keep their input
/// order) and nulls go last regardless of direction.
///
/// @tparam Keys  Types convertible to SortKey (strings or desc() results).
/// @param keys  Column names (ascending) or desc() keys, most significant first.
/// @return  An Arrange step to pipe a DataFrame into.
/// @throws std::out_of_range  (when piped) if a key column does not exist.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// const auto sorted = tooth | dm::arrange("supp", dm::desc("len"));
/// std::print("{}\n", dm::head(sorted, 3));   // OJ rows, len 30.9, 29.4, 27.3
/// @endcode
/// @see dm::desc(), DataFrame::arrange(), DataFrame::sort_by()
template <typename... Keys,
          std::enable_if_t<(sizeof...(Keys) > 0) && (std::is_constructible_v<SortKey, Keys> && ...), int> = 0>
[[nodiscard]] Arrange arrange(Keys&&... keys) {
  return Arrange{std::vector<SortKey>{SortKey(std::forward<Keys>(keys))...}};
}

/// @brief Implementation helpers for the pipe steps; not part of the public API.
namespace pipe_detail {

/// @brief Builds a column by calling `func` on every row, choosing a numeric or string column
///        from the callable's return type.
/// @param frame Source frame.
/// @param func  Row callable.
template <typename Func>
auto column_from_rows(const dstruct::DataFrame& frame, const Func& func) {
  using Row = dstruct::DataFrame::Row;
  using Result = std::decay_t<std::invoke_result_t<const Func&, const Row&>>;
  constexpr bool is_string = std::is_convertible_v<Result, std::string> && !std::is_arithmetic_v<Result>;
  constexpr bool is_optional_string = std::is_same_v<Result, std::optional<std::string>>;
  static_assert(std::is_arithmetic_v<Result> || is_string || is_optional_string ||
                    std::is_same_v<Result, std::optional<double>>,
                "dm::mutate: the row callable must return a number, a string, or a std::optional of either");

  using Cell = std::conditional_t<is_string || is_optional_string, std::optional<std::string>, std::optional<double>>;
  std::vector<Cell> values;
  values.reserve(frame.nrows());
  for (std::size_t i = 0; i < frame.nrows(); ++i) {
    const Result value = func(frame.row(i));
    if constexpr (std::is_arithmetic_v<Result>) {
      values.emplace_back(static_cast<double>(value));
    } else if constexpr (is_string) {
      values.emplace_back(std::string(value));
    } else {
      values.push_back(value);
    }
  }
  return values;
}

} // namespace pipe_detail

/// @brief Counts rows per group, like dplyr::count().
///
/// `grouped | dm::count()` returns one row per group (first-appearance order) with the key
/// columns followed by the count column.
///
/// @param column_name  Name of the result column (default "n", as in dplyr).
/// @return  A Count step to pipe a GroupedDataFrame into.
/// @throws std::invalid_argument  (when piped) if @p column_name is already a column.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// std::print("{}\n", tooth | dm::group_by("supp") | dm::count());   // VC 30, OJ 30
/// const auto per_dose = tooth | dm::group_by("dose") | dm::count("rows");
/// @endcode
/// @see dm::group_by(), GroupedDataFrame::count()
[[nodiscard]] inline Count count(std::string column_name = "n") { return Count{std::move(column_name)}; }

/// @brief Column names in column order, like R's colnames().
/// @param frame  Source frame.
/// @return  A copy of the names; see DataFrame::columns().
///
/// @par Example
/// @code
/// std::print("{}\n", dm::col_names(dm::datasets::tooth_growth()));   // ["len", "supp", "dose"]
/// @endcode
/// @see col_names(const dstruct::DataFrame&, dstruct::DataFrame::ColumnType)
[[nodiscard]] inline std::vector<std::string> col_names(const dstruct::DataFrame& frame) { return frame.columns(); }

/// @brief Names of the columns of one storage kind, in column order, like R's
///        `names(Filter(is.numeric, df))`.
/// @param frame  Source frame.
/// @param type  ColumnType::Numeric or ColumnType::String.
/// @return  The matching column names.
///
/// @par Example
/// @code
/// using ColumnType = dm::dstruct::DataFrame::ColumnType;
/// const auto iris = dm::datasets::iris();
/// std::print("{}\n", dm::col_names(iris, ColumnType::Numeric));   // the four measurements
/// std::print("{}\n", dm::col_names(iris, ColumnType::String));    // ["Species"]
/// @endcode
/// @see DataFrame::column_type()
[[nodiscard]] inline std::vector<std::string> col_names(const dstruct::DataFrame& frame,
                                                        dstruct::DataFrame::ColumnType type) {
  std::vector<std::string> names;
  for (const auto& name : frame.columns()) {
    if (frame.column_type(name) == type) names.push_back(name);
  }
  return names;
}

/// @brief Distinct non-null values of a column in order of first appearance, like R's
///        `unique(df$x)` (but nulls are skipped rather than kept as one NA).
/// @tparam T  double or std::string.
/// @param column  Column, e.g. from DataFrame::get_column<std::string>().
/// @return  The distinct values.
///
/// @par Example
/// @code
/// const auto iris = dm::datasets::iris();
/// std::print("{}\n", dm::unique(iris.get_column<std::string>("Species")));
/// // ["setosa", "versicolor", "virginica"]
/// @endcode
/// @see dm::unique(const std::vector<T>&)
template <typename T>
[[nodiscard]] std::vector<T> unique(const dstruct::NullableColumn<T>& column) {
  std::vector<T> result;
  for (std::size_t i = 0; i < column.size(); ++i) {
    if (column.is_null(i)) continue;
    const auto& value = column.values()[i];
    if (std::find(result.begin(), result.end(), value) == result.end()) result.push_back(value);
  }
  return result;
}

/// @brief DataFrame types; the pipe operators live here so they are found for any right-hand
///        side, including plain lambdas.
namespace dstruct {

/// @brief `frame | dm::group_by(...)`: groups @p frame by the step's keys.
/// @param frame  Frame to group (copied into the result).
/// @param step  Step from dm::group_by().
/// @return  A GroupedDataFrame; see DataFrame::group_by().
[[nodiscard]] inline GroupedDataFrame operator|(const DataFrame& frame, const GroupBy& step) {
  return frame.group_by(step.keys);
}

/// @brief `grouped | dm::summarise_each(fn)`: one summary row per group.
/// @tparam Func  Reduction callable type.
/// @param grouped  Grouped frame.
/// @param step  Step from dm::summarise_each().
/// @return  A DataFrame; see GroupedDataFrame::summarise_each().
template <typename Func>
[[nodiscard]] DataFrame operator|(const GroupedDataFrame& grouped, const SummariseEach<Func>& step) {
  return grouped.summarise_each(step.func);
}

/// @brief `grouped | dm::count()`: row counts per group.
/// @param grouped  Grouped frame.
/// @param step  Step from dm::count().
/// @return  A DataFrame of keys plus the count column; see GroupedDataFrame::count().
[[nodiscard]] inline DataFrame operator|(const GroupedDataFrame& grouped, const Count& step) {
  return grouped.count(step.column_name);
}

/// @brief `frame | dm::filter_rows(predicate)`: the rows for which the predicate is true.
/// @tparam Predicate  Row predicate type.
/// @param frame  Source frame.
/// @param step  Step from dm::filter_rows().
/// @return  A new DataFrame; see DataFrame::filter().
template <typename Predicate>
[[nodiscard]] DataFrame operator|(const DataFrame& frame, const Filter<Predicate>& step) {
  return frame.filter(step.predicate);
}

/// @brief `frame | dm::mutate(name, value)`: @p frame with one column added or replaced.
///
/// A row callable is evaluated once per row (in order) to build the column; a ready vector
/// is used as-is.
/// @tparam Value  Row callable or column vector type.
/// @param frame  Source frame.
/// @param step  Step from dm::mutate().
/// @return  A new DataFrame; see DataFrame::mutate().
template <typename Value>
[[nodiscard]] DataFrame operator|(const DataFrame& frame, const Mutate<Value>& step) {
  if constexpr (std::is_invocable_v<const Value&, const DataFrame::Row&>) {
    return frame.mutate(step.column_name, pipe_detail::column_from_rows(frame, step.value));
  } else {
    return frame.mutate(step.column_name, step.value);
  }
}

/// @brief `frame | dm::arrange(keys...)`: @p frame with its rows sorted.
/// @param frame  Source frame.
/// @param step  Step from dm::arrange().
/// @return  A new DataFrame; see DataFrame::arrange().
[[nodiscard]] inline DataFrame operator|(const DataFrame& frame, const Arrange& step) {
  std::vector<std::string> columns;
  std::vector<bool> ascending;
  for (const auto& key : step.keys) {
    columns.push_back(key.column);
    ascending.push_back(key.ascending);
  }
  return frame.arrange(columns, ascending);
}

/// @brief `frame | fn`: any callable taking the DataFrame is a pipeline step.
///
/// Generic lambdas (`[](const auto& df) { ... }`) work: the overload first checks that the
/// left side is a DataFrame, and only then whether @p func accepts it, so the lambda's body
/// is never instantiated with a GroupedDataFrame.
///
/// @code
/// const auto top = dm::datasets::iris() | [](const auto& df) { return dm::head(df, 3); };
/// std::print("{}\n", top.nrows());   // 3
/// @endcode
/// @tparam Frame  `DataFrame` with any cv/ref qualification.
/// @tparam Func  Callable type.
/// @param frame  Source frame.
/// @param func  Callable invoked as `func(frame)` with a `const DataFrame&`.
/// @return  Whatever @p func returns (not necessarily a DataFrame).
template <typename Frame, typename Func,
          std::enable_if_t<std::conjunction_v<std::is_same<std::remove_cv_t<std::remove_reference_t<Frame>>, DataFrame>,
                                              std::is_invocable<Func, const DataFrame&>>,
                           int> = 0>
decltype(auto) operator|(Frame&& frame, Func&& func) {
  return std::forward<Func>(func)(static_cast<const DataFrame&>(frame));
}

/// @brief `grouped | fn`: any callable taking the GroupedDataFrame is a pipeline step.
///
/// Like the DataFrame overload, generic lambdas work: @p func is only checked once the left
/// side is known to be a GroupedDataFrame.
///
/// @code
/// const auto n = dm::datasets::tooth_growth() | dm::group_by("supp")
///   | [](const auto& g) { return g.count().nrows(); };   // 2
/// @endcode
/// @tparam Grouped  `GroupedDataFrame` with any cv/ref qualification.
/// @tparam Func  Callable type.
/// @param grouped  Grouped frame.
/// @param func  Callable invoked as `func(grouped)` with a `const GroupedDataFrame&`.
/// @return  Whatever @p func returns.
template <typename Grouped, typename Func,
          std::enable_if_t<
              std::conjunction_v<std::is_same<std::remove_cv_t<std::remove_reference_t<Grouped>>, GroupedDataFrame>,
                                 std::is_invocable<Func, const GroupedDataFrame&>>,
              int> = 0>
decltype(auto) operator|(Grouped&& grouped, Func&& func) {
  return std::forward<Func>(func)(static_cast<const GroupedDataFrame&>(grouped));
}

} // namespace dstruct

} // namespace datamunge

// ---------------------------------------------------------------------------------------------
// summarise(), select(), rename(), pivots, slice_head(), replace_na() and to_md() verbs.
// ---------------------------------------------------------------------------------------------

namespace datamunge {

/// @brief Builders for DataFrame::AggSpec, named after their dplyr counterparts, for use with
///        dm::summarise().
///
/// Each takes the input column and an optional output name (default: the input column's name,
/// or "n" for n()). across() applies one aggregation to several columns.
///
/// @par Example
/// @code
/// namespace agg = dm::agg;
/// const auto stats = dm::datasets::iris() | dm::group_by("Species")
///   | dm::summarise(agg::n(), agg::mean("Petal.Length", "mean"), agg::sd("Petal.Length", "sd"));
/// @endcode
/// @see dm::summarise(), DataFrame::summarise()
namespace agg {

/// @brief Aggregation function enum, re-exported for across().
using Func = dstruct::DataFrame::AggFunc;
/// @brief One aggregation, as built by the helpers below.
using Spec = dstruct::DataFrame::AggSpec;

/// @brief Row count per group, like dplyr's n().
/// @param name  Output column name (default "n").
/// @return  A Count AggSpec.
[[nodiscard]] inline Spec n(std::string name = "n") { return Spec{"", Func::Count, std::move(name)}; }
/// @brief Sum of the non-null values.
/// @param column  Input column.
/// @param name  Output column name (default: @p column).
/// @return  A Sum AggSpec.
[[nodiscard]] inline Spec sum(std::string column, std::string name = "") { return Spec{std::move(column), Func::Sum, std::move(name)}; }
/// @brief Mean of the non-null values.
/// @param column  Input column.
/// @param name  Output column name (default: @p column).
/// @return  A Mean AggSpec.
[[nodiscard]] inline Spec mean(std::string column, std::string name = "") { return Spec{std::move(column), Func::Mean, std::move(name)}; }
/// @brief Sample standard deviation (n - 1 denominator) of the non-null values.
/// @param column  Input column.
/// @param name  Output column name (default: @p column).
/// @return  A StdDev AggSpec.
[[nodiscard]] inline Spec sd(std::string column, std::string name = "") { return Spec{std::move(column), Func::StdDev, std::move(name)}; }
/// @brief Median of the non-null values.
/// @param column  Input column.
/// @param name  Output column name (default: @p column).
/// @return  A Median AggSpec.
[[nodiscard]] inline Spec median(std::string column, std::string name = "") { return Spec{std::move(column), Func::Median, std::move(name)}; }
/// @brief Smallest non-null value.
/// @param column  Input column.
/// @param name  Output column name (default: @p column).
/// @return  A Min AggSpec.
[[nodiscard]] inline Spec min(std::string column, std::string name = "") { return Spec{std::move(column), Func::Min, std::move(name)}; }
/// @brief Largest non-null value.
/// @param column  Input column.
/// @param name  Output column name (default: @p column).
/// @return  A Max AggSpec.
[[nodiscard]] inline Spec max(std::string column, std::string name = "") { return Spec{std::move(column), Func::Max, std::move(name)}; }
/// @brief Number of distinct values, like dplyr's n_distinct().
/// @param column  Input column.
/// @param name  Output column name (default: @p column).
/// @return  An NDistinct AggSpec.
[[nodiscard]] inline Spec n_distinct(std::string column, std::string name = "") { return Spec{std::move(column), Func::NDistinct, std::move(name)}; }

/// @brief dplyr's short name for an aggregation function ("mean", "sd", ...), as used by the
///        `{fn}` placeholder of across().
/// @param func  Aggregation function.
/// @return  Its short name.
[[nodiscard]] inline std::string name_of(Func func) {
  switch (func) {
    case Func::Sum: return "sum";
    case Func::Mean: return "mean";
    case Func::Min: return "min";
    case Func::Max: return "max";
    case Func::Median: return "median";
    case Func::StdDev: return "sd";
    case Func::Count: return "n";
    case Func::NDistinct: return "n_distinct";
  }
  return "";
}

/// @brief Applies one aggregation to several columns, like dplyr's `across(cols, fn)`.
///
/// @param columns  Input columns.
/// @param func  Aggregation applied to each.
/// @param names  Output-name pattern: `{col}` becomes the column name and `{fn}` the function's
///        short name (default "{col}", so outputs keep their input names).
/// @return  One AggSpec per column, in order.
///
/// @par Example
/// @code
/// const auto means = dm::datasets::iris() | dm::group_by("Species")
///   | dm::summarise(dm::agg::n(), dm::agg::across({"Sepal.Length", "Petal.Length"},
///                                               dm::agg::Func::Mean, "{col}_{fn}"));
/// // columns: Species, n, Sepal.Length_mean, Petal.Length_mean
/// @endcode
[[nodiscard]] inline std::vector<Spec> across(const std::vector<std::string>& columns, Func func,
                                              const std::string& names = "{col}") {
  std::vector<Spec> specs;
  specs.reserve(columns.size());
  for (const auto& column : columns) {
    std::string name = names;
    for (const auto& [placeholder, text] : {std::pair<std::string, std::string>{"{col}", column}, {"{fn}", name_of(func)}}) {
      for (auto at = name.find(placeholder); at != std::string::npos; at = name.find(placeholder, at + text.size())) {
        name.replace(at, placeholder.size(), text);
      }
    }
    specs.push_back(Spec{column, func, name});
  }
  return specs;
}

} // namespace agg

/// @brief Pipeline step created by summarise(); `grouped | step` or `frame | step` yields a DataFrame.
/// @see dm::summarise()
struct Summarise {
  /// @brief Aggregations, one output column each.
  std::vector<dstruct::DataFrame::AggSpec> specs;
};

namespace pipe_detail {

/// @brief Appends one aggregation spec to @p out (used when expanding summarise() arguments).
inline void append_specs(std::vector<dstruct::DataFrame::AggSpec>& out, dstruct::DataFrame::AggSpec spec) {
  out.push_back(std::move(spec));
}

/// @brief Appends every spec in @p specs to @p out (for verbs such as across() that yield several).
inline void append_specs(std::vector<dstruct::DataFrame::AggSpec>& out,
                         const std::vector<dstruct::DataFrame::AggSpec>& specs) {
  out.insert(out.end(), specs.begin(), specs.end());
}

} // namespace pipe_detail

/// @brief Summarises each group (or the whole frame) into one row, like dplyr::summarise().
///
/// Arguments are AggSpecs (built with dm::agg::n(), dm::agg::mean(), ...) or vectors of them
/// (from dm::agg::across()); the output has the group keys followed by one column per spec, in
/// argument order. Piping an ungrouped frame gives a single row.
///
/// @tparam Specs  `DataFrame::AggSpec` or `std::vector<DataFrame::AggSpec>`.
/// @param specs  Aggregations, at least one in total.
/// @return  A Summarise step.
/// @throws std::invalid_argument  (when piped) as DataFrame::summarise().
///
/// @par Example
/// @code
/// namespace agg = dm::agg;
/// const auto iris = dm::datasets::iris();
/// const auto by_species = iris | dm::group_by("Species")
///   | dm::summarise(agg::n(), agg::across({"Petal.Length", "Petal.Width"}, agg::Func::Mean));
/// std::print("{}", by_species.to_md());
/// const auto overall = iris | dm::summarise(agg::mean("Petal.Length"), agg::sd("Petal.Length", "sd"));
/// @endcode
/// @see dm::agg, DataFrame::summarise(), dm::group_by()
template <typename... Specs>
[[nodiscard]] Summarise summarise(Specs&&... specs) {
  Summarise step;
  (pipe_detail::append_specs(step.specs, std::forward<Specs>(specs)), ...);
  return step;
}

/// @brief Pipeline step created by select().
/// @see dm::select()
struct Select {
  /// @brief Columns to keep, in output order.
  std::vector<std::string> columns;
};

/// @brief Keeps the named columns, in the given order, like dplyr::select().
/// @tparam Columns  Types convertible to std::string.
/// @param columns  Column names, at least one.
/// @return  A Select step; see DataFrame::select().
///
/// @par Example
/// @code
/// const auto petals = dm::datasets::iris() | dm::select("Species", "Petal.Length");
/// @endcode
template <typename... Columns,
          std::enable_if_t<(sizeof...(Columns) > 0) && (std::is_convertible_v<Columns, std::string> && ...), int> = 0>
[[nodiscard]] Select select(Columns&&... columns) {
  return Select{{std::string(std::forward<Columns>(columns))...}};
}

/// @brief select() taking the column names as one vector.
/// @param columns  Column names.
/// @return  A Select step.
[[nodiscard]] inline Select select(std::vector<std::string> columns) { return Select{std::move(columns)}; }

/// @brief Pipeline step created by rename().
/// @see dm::rename()
struct Rename {
  /// @brief (old name, new name) pairs.
  std::vector<std::pair<std::string, std::string>> mapping;
};

/// @brief Renames one column, like dplyr's `rename(new = old)` (note the argument order here
///        is old, then new).
/// @param old_name  Existing column.
/// @param new_name  Its new name.
/// @return  A Rename step; see DataFrame::rename().
///
/// @par Example
/// @code
/// const auto renamed = dm::datasets::iris() | dm::rename("Species", "species");
/// @endcode
[[nodiscard]] inline Rename rename(std::string old_name, std::string new_name) {
  return Rename{{{std::move(old_name), std::move(new_name)}}};
}

/// @brief rename() for several (old, new) pairs at once.
/// @param mapping  (old name, new name) pairs.
/// @return  A Rename step.
[[nodiscard]] inline Rename rename(std::vector<std::pair<std::string, std::string>> mapping) {
  return Rename{std::move(mapping)};
}

/// @brief Pipeline step created by pivot_longer().
/// @see dm::pivot_longer()
struct PivotLonger {
  std::vector<std::string> columns;  ///< Columns to stack.
  std::string names_to;              ///< New column of source column names.
  std::string values_to;             ///< New column of values.
};

/// @brief Stacks columns into name/value pairs, like tidyr::pivot_longer().
/// @param columns  Columns to stack, all numeric or all string.
/// @param names_to  Name of the new names column (default "name").
/// @param values_to  Name of the new values column (default "value").
/// @return  A PivotLonger step; see DataFrame::pivot_longer().
///
/// @par Example
/// @code
/// const auto long_iris = dm::datasets::iris()
///   | dm::pivot_longer({"Sepal.Length", "Petal.Length"}, "measurement", "cm");   // 300 x 5
/// @endcode
[[nodiscard]] inline PivotLonger pivot_longer(std::vector<std::string> columns, std::string names_to = "name",
                                              std::string values_to = "value") {
  return PivotLonger{std::move(columns), std::move(names_to), std::move(values_to)};
}

/// @brief Pipeline step created by pivot_wider().
/// @see dm::pivot_wider()
struct PivotWider {
  std::string names_from;            ///< String column supplying the new column names.
  std::string values_from;           ///< Column supplying their values.
  std::optional<double> values_fill; ///< Value for missing numeric cells (tidyr's values_fill).
};

/// @brief Spreads name/value pairs into columns, like tidyr::pivot_wider().
///
/// Every column other than @p names_from and @p values_from identifies an output row.
/// Missing combinations are null unless @p values_fill is given (numeric values only).
///
/// @param names_from  String column whose values become column names.
/// @param values_from  Column holding the values.
/// @param values_fill  Optional fill for missing numeric cells, e.g. 0 for counts.
/// @return  A PivotWider step; see DataFrame::pivot_wider().
///
/// @par Example
/// @code
/// // A confusion table: rows are true classes, columns predicted classes.
/// const auto table = frame | dm::group_by("truth", "predicted") | dm::count()
///                  | dm::pivot_wider("predicted", "n", 0.0);
/// @endcode
[[nodiscard]] inline PivotWider pivot_wider(std::string names_from, std::string values_from,
                                            std::optional<double> values_fill = std::nullopt) {
  return PivotWider{std::move(names_from), std::move(values_from), values_fill};
}

/// @brief Pipeline step created by slice_head().
/// @see dm::slice_head()
struct SliceHead {
  std::size_t n;  ///< Rows to keep.
};

/// @brief Keeps the first @p n rows, like dplyr::slice_head().
/// @param n  Rows to keep (all of them if the frame is shorter).
/// @return  A SliceHead step; see DataFrame::head().
///
/// @par Example
/// @code
/// std::print("{}\n", (dm::datasets::iris() | dm::slice_head(3)).nrows());   // 3
/// @endcode
[[nodiscard]] inline SliceHead slice_head(std::size_t n) { return SliceHead{n}; }

/// @brief Pipeline step created by replace_na().
/// @see dm::replace_na()
struct ReplaceNa {
  double value;  ///< Replacement for nulls in numeric columns.
};

/// @brief Replaces nulls in every numeric column with @p value, like tidyr::replace_na().
/// @param value  Replacement value.
/// @return  A ReplaceNa step; see DataFrame::fill_null().
///
/// @par Example
/// @code
/// const auto filled = dm::datasets::penguins() | dm::replace_na(0);
/// std::print("{}\n", filled.null_count("body_mass_g"));   // 0
/// @endcode
[[nodiscard]] inline ReplaceNa replace_na(double value) { return ReplaceNa{value}; }

/// @brief Pipeline step created by to_md(); `frame | step` yields a std::string.
/// @see dm::to_md()
struct ToMd {
  dstruct::DataFrame::MarkdownOptions options;  ///< Formatting choices.
};

/// @brief Ends a pipeline with a Markdown table, like `knitr::kable()`.
/// @param digits  Maximum decimals in numeric columns (default 3).
/// @return  A ToMd step; `frame | step` returns DataFrame::to_md().
///
/// @par Example
/// @code
/// std::cout << (dm::datasets::iris() | dm::slice_head(3) | dm::to_md(1));
/// @endcode
[[nodiscard]] inline ToMd to_md(int digits = 3) {
  ToMd step;
  step.options.digits = digits;
  return step;
}

/// @brief to_md() with full options (per-column digits, row limit, null text).
/// @param options  Formatting choices.
/// @return  A ToMd step.
[[nodiscard]] inline ToMd to_md(dstruct::DataFrame::MarkdownOptions options) { return ToMd{std::move(options)}; }

namespace dstruct {

/// @brief `grouped | dm::summarise(...)`: one row per group.
/// @param grouped  Grouped frame.
/// @param step  Step from dm::summarise().
/// @return  Keys plus one column per spec; see GroupedDataFrame::summarise().
[[nodiscard]] inline DataFrame operator|(const GroupedDataFrame& grouped, const Summarise& step) {
  return grouped.summarise(step.specs);
}

/// @brief `frame | dm::summarise(...)`: one row for the whole frame.
/// @param frame  Source frame.
/// @param step  Step from dm::summarise().
/// @return  A one-row DataFrame; see DataFrame::summarise().
[[nodiscard]] inline DataFrame operator|(const DataFrame& frame, const Summarise& step) {
  return frame.summarise(step.specs);
}

/// @brief `frame | dm::select(...)`.
/// @param frame  Source frame.
/// @param step  Step from dm::select().
/// @return  The selected columns.
[[nodiscard]] inline DataFrame operator|(const DataFrame& frame, const Select& step) { return frame.select(step.columns); }

/// @brief `frame | dm::rename(...)`.
/// @param frame  Source frame.
/// @param step  Step from dm::rename().
/// @return  The renamed frame.
[[nodiscard]] inline DataFrame operator|(const DataFrame& frame, const Rename& step) { return frame.rename(step.mapping); }

/// @brief `frame | dm::pivot_longer(...)`.
/// @param frame  Source frame.
/// @param step  Step from dm::pivot_longer().
/// @return  The longer frame.
[[nodiscard]] inline DataFrame operator|(const DataFrame& frame, const PivotLonger& step) {
  return frame.pivot_longer(step.columns, step.names_to, step.values_to);
}

/// @brief `frame | dm::pivot_wider(...)`.
/// @param frame  Source frame.
/// @param step  Step from dm::pivot_wider().
/// @return  The wider frame, with missing numeric cells filled when the step asks for it.
[[nodiscard]] inline DataFrame operator|(const DataFrame& frame, const PivotWider& step) {
  auto wide = frame.pivot_wider(step.names_from, step.values_from);
  if (step.values_fill) {
    for (const auto& name : wide.columns()) {
      const bool is_id = frame.has_column(name) && name != step.names_from && name != step.values_from;
      if (!is_id && wide.column_type(name) == DataFrame::ColumnType::Numeric) wide.fill_null(name, *step.values_fill);
    }
  }
  return wide;
}

/// @brief `frame | dm::slice_head(n)`.
/// @param frame  Source frame.
/// @param step  Step from dm::slice_head().
/// @return  The first rows.
[[nodiscard]] inline DataFrame operator|(const DataFrame& frame, const SliceHead& step) { return frame.head(step.n); }

/// @brief `frame | dm::replace_na(value)`.
/// @param frame  Source frame.
/// @param step  Step from dm::replace_na().
/// @return  A copy with numeric nulls replaced.
[[nodiscard]] inline DataFrame operator|(const DataFrame& frame, const ReplaceNa& step) {
  DataFrame filled = frame;
  for (const auto& name : filled.columns()) {
    if (filled.column_type(name) == DataFrame::ColumnType::Numeric) filled.fill_null(name, step.value);
  }
  return filled;
}

/// @brief `frame | dm::to_md(...)`: the frame as a Markdown table.
/// @param frame  Source frame.
/// @param step  Step from dm::to_md().
/// @return  The Markdown text; see DataFrame::to_md().
[[nodiscard]] inline std::string operator|(const DataFrame& frame, const ToMd& step) { return frame.to_md(step.options); }

} // namespace dstruct

} // namespace datamunge
