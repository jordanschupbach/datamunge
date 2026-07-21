#pragma once

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

template <typename T>
class NullableColumn {
 public:
  using value_type = T;
  using size_type = std::size_t;

  NullableColumn() = default;

  explicit NullableColumn(const std::vector<T>& values) : values_(values), valid_(values.size(), 1) {}

  explicit NullableColumn(std::vector<T>&& values) : values_(std::move(values)), valid_(values_.size(), 1) {}

  explicit NullableColumn(const std::vector<std::optional<T>>& values) { assign(values); }

  explicit NullableColumn(std::vector<std::optional<T>>&& values) { assign(values); }

  [[nodiscard]] bool empty() const { return values_.empty(); }

  [[nodiscard]] size_type size() const { return values_.size(); }

  [[nodiscard]] bool is_null(const size_type index) const {
    require_index(index);
    return valid_[index] == 0;
  }

  [[nodiscard]] const T& at(const size_type index) const {
    require_index(index);
    if (valid_[index] == 0) {
      throw std::invalid_argument("NullableColumn::at cannot access null value");
    }
    return values_[index];
  }

  [[nodiscard]] std::optional<T> optional_at(const size_type index) const {
    require_index(index);
    if (valid_[index] == 0) {
      return std::nullopt;
    }
    return values_[index];
  }

  void set(const size_type index, const T& value) {
    require_index(index);
    values_[index] = value;
    valid_[index] = 1;
  }

  void set(const size_type index, T&& value) {
    require_index(index);
    values_[index] = std::move(value);
    valid_[index] = 1;
  }

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

  void set_null(const size_type index) {
    require_index(index);
    values_[index] = T{};
    valid_[index] = 0;
  }

  void push_back(const T& value) {
    values_.push_back(value);
    valid_.push_back(1);
  }

  void push_back(T&& value) {
    values_.push_back(std::move(value));
    valid_.push_back(1);
  }

  void push_back(const std::optional<T>& value) {
    if (value.has_value()) {
      values_.push_back(*value);
      valid_.push_back(1);
    } else {
      values_.push_back(T{});
      valid_.push_back(0);
    }
  }

  void push_null() {
    values_.push_back(T{});
    valid_.push_back(0);
  }

  [[nodiscard]] const std::vector<T>& values() const { return values_; }

  [[nodiscard]] const std::vector<std::uint8_t>& validity() const { return valid_; }

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

  void require_index(const size_type index) const {
    if (index >= values_.size()) {
      throw std::out_of_range("NullableColumn index out of range");
    }
  }

  std::vector<T> values_;
  std::vector<std::uint8_t> valid_;
};

/// @brief Forward declaration -- see the full definition after DataFrame, since it holds a
///        DataFrame by value (see DataFrame::group_by()).
class GroupedDataFrame;

class DataFrame {
 public:
  using size_type = std::size_t;
  using null_type = std::monostate;
  using numeric_column_type = NullableColumn<double>;
  using string_column_type = NullableColumn<std::string>;
  using column_data_type = std::variant<numeric_column_type, string_column_type>;
  using cell_type = std::variant<null_type, double, std::string>;

  enum class ColumnType { Numeric, String };
  enum class JoinType { Inner, Left, Right, Full, Semi, Anti };

  /// @brief Aggregation functions usable with summarise()/GroupedDataFrame::summarise().
  ///        Median/StdDev/Min/Max/Sum/Mean require a numeric column; Min/Max also work on
  ///        string columns (lexicographic); Count and NDistinct work on any column (Count
  ///        ignores AggSpec::column entirely).
  enum class AggFunc { Sum, Mean, Min, Max, Median, StdDev, Count, NDistinct };

  /// @brief One aggregation to compute per group in summarise(): result_name defaults to
  ///        `column` (or "n" for Count when both are left empty).
  struct AggSpec {
    std::string column;
    AggFunc func{AggFunc::Sum};
    std::string result_name;
  };

  struct Row {
    std::unordered_map<std::string, cell_type> values;

    [[nodiscard]] bool contains(const std::string& column_name) const {
      return values.find(column_name) != values.end();
    }

    [[nodiscard]] const cell_type& at(const std::string& column_name) const {
      const auto it = values.find(column_name);
      if (it == values.end()) {
        throw std::out_of_range("DataFrame::Row missing column: " + column_name);
      }
      return it->second;
    }

    [[nodiscard]] bool is_null(const std::string& column_name) const {
      return std::holds_alternative<null_type>(at(column_name));
    }

    [[nodiscard]] double get_double(const std::string& column_name) const { return std::get<double>(at(column_name)); }

    [[nodiscard]] const std::string& get_string(const std::string& column_name) const {
      return std::get<std::string>(at(column_name));
    }

    [[nodiscard]] std::optional<double> get_optional_double(const std::string& column_name) const {
      if (is_null(column_name)) {
        return std::nullopt;
      }
      return std::get<double>(at(column_name));
    }

    [[nodiscard]] std::optional<std::string> get_optional_string(const std::string& column_name) const {
      if (is_null(column_name)) {
        return std::nullopt;
      }
      return std::get<std::string>(at(column_name));
    }
  };

  struct NumericSummary {
    size_type count{0};
    size_type null_count{0};
    double sum{0.0};
    double mean{0.0};
    double min{0.0};
    double max{0.0};
  };

  DataFrame() = default;

  DataFrame(std::initializer_list<std::pair<const std::string, column_data_type>> columns) {
    for (const auto& column : columns) {
      std::visit(
          [&](const auto& values) {
            add_column(column.first, values);
          },
          column.second);
    }
  }

  [[nodiscard]] bool empty() const { return row_count_ == 0 || column_order_.empty(); }

  [[nodiscard]] size_type nrows() const { return row_count_; }

  [[nodiscard]] size_type ncols() const { return column_order_.size(); }

  [[nodiscard]] std::pair<size_type, size_type> shape() const { return {row_count_, column_order_.size()}; }

  [[nodiscard]] bool has_column(const std::string& column_name) const {
    return columns_.find(column_name) != columns_.end();
  }

  [[nodiscard]] std::vector<std::string> columns() const { return column_order_; }

  [[nodiscard]] ColumnType column_type(const std::string& column_name) const {
    const auto& data = column_data(column_name);
    if (std::holds_alternative<numeric_column_type>(data)) {
      return ColumnType::Numeric;
    }
    return ColumnType::String;
  }

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

  void clear() {
    column_order_.clear();
    columns_.clear();
    row_count_ = 0;
  }

  void add_column(const std::string& column_name, const numeric_column_type& values) { add_column_impl(column_name, values); }

  void add_column(const std::string& column_name, numeric_column_type&& values) {
    add_column_impl(column_name, std::move(values));
  }

  void add_column(const std::string& column_name, const string_column_type& values) { add_column_impl(column_name, values); }

  void add_column(const std::string& column_name, string_column_type&& values) {
    add_column_impl(column_name, std::move(values));
  }

  void add_column(const std::string& column_name, const std::vector<double>& values) {
    add_column_impl(column_name, numeric_column_type(values));
  }

  void add_column(const std::string& column_name, std::vector<double>&& values) {
    add_column_impl(column_name, numeric_column_type(std::move(values)));
  }

  void add_column(const std::string& column_name, const std::vector<std::optional<double>>& values) {
    add_column_impl(column_name, numeric_column_type(values));
  }

  void add_column(const std::string& column_name, std::vector<std::optional<double>>&& values) {
    add_column_impl(column_name, numeric_column_type(std::move(values)));
  }

  void add_column(const std::string& column_name, const std::vector<std::string>& values) {
    add_column_impl(column_name, string_column_type(values));
  }

  void add_column(const std::string& column_name, std::vector<std::string>&& values) {
    add_column_impl(column_name, string_column_type(std::move(values)));
  }

  void add_column(const std::string& column_name, const std::vector<std::optional<std::string>>& values) {
    add_column_impl(column_name, string_column_type(values));
  }

  void add_column(const std::string& column_name, std::vector<std::optional<std::string>>&& values) {
    add_column_impl(column_name, string_column_type(std::move(values)));
  }

  void replace_column(const std::string& column_name, const numeric_column_type& values) {
    replace_column_impl(column_name, values);
  }

  void replace_column(const std::string& column_name, numeric_column_type&& values) {
    replace_column_impl(column_name, std::move(values));
  }

  void replace_column(const std::string& column_name, const string_column_type& values) {
    replace_column_impl(column_name, values);
  }

  void replace_column(const std::string& column_name, string_column_type&& values) {
    replace_column_impl(column_name, std::move(values));
  }

  /// @brief dplyr::mutate()-style upsert: adds `column_name` if absent, replaces it in place
  ///        (same type) if present. Unlike add_column()/replace_column(), always succeeds and
  ///        returns a new DataFrame, so it composes into a pipe: df.mutate(...).filter(...).
  [[nodiscard]] DataFrame mutate(const std::string& column_name, const std::vector<double>& values) const {
    return mutate_impl(column_name, numeric_column_type(values));
  }

  [[nodiscard]] DataFrame mutate(const std::string& column_name, const std::vector<std::optional<double>>& values) const {
    return mutate_impl(column_name, numeric_column_type(values));
  }

  [[nodiscard]] DataFrame mutate(const std::string& column_name, const std::vector<std::string>& values) const {
    return mutate_impl(column_name, string_column_type(values));
  }

  [[nodiscard]] DataFrame mutate(const std::string& column_name, const std::vector<std::optional<std::string>>& values) const {
    return mutate_impl(column_name, string_column_type(values));
  }

  /// @brief Row-wise mutate for C++ callers: `df.mutate_with("z", [](const Row& r){ return
  ///        r.get_double("x") + r.get_double("y"); })`. Not SWIG-exposable (arbitrary callable).
  template <typename NumericFn>
  [[nodiscard]] DataFrame mutate_with(const std::string& column_name, NumericFn fn) const {
    std::vector<double> values;
    values.reserve(row_count_);
    for (size_type row_index = 0; row_index < row_count_; ++row_index) {
      values.push_back(fn(row(row_index)));
    }
    return mutate(column_name, std::move(values));
  }

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
  [[nodiscard]] DataFrame rename(const std::vector<std::pair<std::string, std::string>>& mapping) const {
    DataFrame result = select(column_order_);
    for (const auto& [old_name, new_name] : mapping) {
      result.rename_column(old_name, new_name);
    }
    return result;
  }

  /// @brief Single-pair convenience overload of rename(); see the vector<pair> overload for
  ///        renaming several columns at once.
  [[nodiscard]] DataFrame rename(const std::string& old_name, const std::string& new_name) const {
    return rename(std::vector<std::pair<std::string, std::string>>{{old_name, new_name}});
  }

  [[nodiscard]] const numeric_column_type& numeric_column(const std::string& column_name) const {
    return require_column_type<numeric_column_type>(column_name, "numeric");
  }

  [[nodiscard]] numeric_column_type& numeric_column(const std::string& column_name) {
    return require_column_type<numeric_column_type>(column_name, "numeric");
  }

  [[nodiscard]] const string_column_type& string_column(const std::string& column_name) const {
    return require_column_type<string_column_type>(column_name, "string");
  }

  [[nodiscard]] string_column_type& string_column(const std::string& column_name) {
    return require_column_type<string_column_type>(column_name, "string");
  }

  /// @brief dplyr::pull()-style column extraction, null-aware (see materialize()).
  [[nodiscard]] std::vector<std::optional<double>> pull_numeric(const std::string& column_name) const {
    return numeric_column(column_name).materialize();
  }

  [[nodiscard]] std::vector<std::optional<std::string>> pull_string(const std::string& column_name) const {
    return string_column(column_name).materialize();
  }

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

  [[nodiscard]] double double_at(const std::string& column_name, const size_type row_index) const {
    return numeric_column(column_name).at(row_index);
  }

  [[nodiscard]] std::optional<double> optional_double_at(const std::string& column_name, const size_type row_index) const {
    return numeric_column(column_name).optional_at(row_index);
  }

  [[nodiscard]] const std::string& string_at(const std::string& column_name, const size_type row_index) const {
    return string_column(column_name).at(row_index);
  }

  [[nodiscard]] std::optional<std::string> optional_string_at(const std::string& column_name,
                                                              const size_type row_index) const {
    return string_column(column_name).optional_at(row_index);
  }

  void set_value(const std::string& column_name, const size_type row_index, const double value) {
    numeric_column(column_name).set(row_index, value);
  }

  void set_value(const std::string& column_name, const size_type row_index, std::optional<double> value) {
    numeric_column(column_name).set(row_index, value);
  }

  void set_value(const std::string& column_name, const size_type row_index, std::string value) {
    string_column(column_name).set(row_index, std::move(value));
  }

  void set_value(const std::string& column_name, const size_type row_index, std::optional<std::string> value) {
    string_column(column_name).set(row_index, value);
  }

  void set_null(const std::string& column_name, const size_type row_index) {
    auto& data = column_data(column_name);
    std::visit(
        [row_index](auto& column) {
          column.set_null(row_index);
        },
        data);
  }

  void fill_null(const std::string& column_name, const double value) {
    auto& column = numeric_column(column_name);
    for (size_type index = 0; index < column.size(); ++index) {
      if (column.is_null(index)) {
        column.set(index, value);
      }
    }
  }

  void fill_null(const std::string& column_name, const std::string& value) {
    auto& column = string_column(column_name);
    for (size_type index = 0; index < column.size(); ++index) {
      if (column.is_null(index)) {
        column.set(index, value);
      }
    }
  }

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

  void append_row(const Row& row_value) {
    require_complete_row(row_value);
    for (const auto& name : column_order_) {
      append_cell(columns_.at(name), row_value.at(name));
    }
    ++row_count_;
  }

  void append_row(std::initializer_list<std::pair<const std::string, cell_type>> row_value) {
    Row row;
    for (const auto& [name, value] : row_value) {
      row.values.emplace(name, value);
    }
    append_row(row);
  }

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

  [[nodiscard]] DataFrame head(const size_type count) const { return slice_rows(0, std::min(count, row_count_)); }

  [[nodiscard]] DataFrame tail(const size_type count) const {
    const auto kept = std::min(count, row_count_);
    return slice_rows(row_count_ - kept, row_count_);
  }

  [[nodiscard]] DataFrame slice(const size_type begin_index, const size_type end_index) const {
    return slice_rows(begin_index, end_index);
  }

  [[nodiscard]] DataFrame take_rows(const std::vector<size_type>& indices) const {
    return take_rows_impl(indices);
  }

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

  [[nodiscard]] DataFrame drop_duplicates(const std::vector<std::string>& subset = {}) const {
    const auto inspected_columns = subset.empty() ? column_order_ : subset;
    if (inspected_columns.empty()) {
      throw std::invalid_argument("DataFrame::drop_duplicates requires at least one column");
    }
    for (const auto& column_name : inspected_columns) {
      (void)column_data(column_name);
    }

    std::unordered_map<group_key_type, size_type, GroupKeyHash> seen;
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
  [[nodiscard]] DataFrame distinct(const std::vector<std::string>& subset = {}) const { return drop_duplicates(subset); }

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
  [[nodiscard]] size_type n_distinct(const std::string& column_name) const {
    const auto& data = column_data(column_name);
    return std::visit(
        [](const auto& column) -> size_type {
          using column_type = std::decay_t<decltype(column)>;
          using value_type = typename column_type::value_type;
          std::unordered_set<value_type> seen;
          bool has_null = false;
          for (size_type index = 0; index < column.size(); ++index) {
            const auto value = column.optional_at(index);
            if (value.has_value()) {
              seen.insert(*value);
            } else {
              has_null = true;
            }
          }
          return seen.size() + (has_null ? 1 : 0);
        },
        data);
  }

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
  [[nodiscard]] DataFrame count(const std::vector<std::string>& key_columns,
                                const std::string& count_column_name = "n") const {
    return group_by_count(key_columns, count_column_name);
  }

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

  /// @brief dplyr::group_by()-style entry point: returns a lightweight GroupedDataFrame that
  ///        pairs `key_columns` with a copy of `*this`, deferring aggregation to
  ///        GroupedDataFrame::summarise()/count(). Defined out-of-line below, after
  ///        GroupedDataFrame itself is complete.
  [[nodiscard]] GroupedDataFrame group_by(const std::vector<std::string>& key_columns) const;

  /// @brief dplyr::pivot_longer()-style reshape: stacks `value_columns` into two new columns
  ///        (`names_to` holding the source column name, `values_to` holding its value), one row
  ///        per (original row, pivoted column) pair. All of `value_columns` must share the same
  ///        column type.
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
      append_group_key_row(result, ids, key);
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

  /// @param join_type One of Inner/Left/Right/Full/Semi/Anti. Semi/Anti keep only left's rows
  ///        (matched / unmatched respectively) and never merge in right's columns. For the
  ///        others, any column name present in both frames (other than the key column when
  ///        left_key == right_key) is suffixed on both sides (left_suffix/right_suffix) so the
  ///        output has no duplicate names.
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

#ifdef DATAMUNGE_HAVE_ARROW
  [[nodiscard]] std::shared_ptr<arrow::Table> to_arrow() const;
  static DataFrame from_arrow(const std::shared_ptr<arrow::Table>& table);
#endif

 private:
  struct CellHash {
    [[nodiscard]] std::size_t operator()(const cell_type& value) const {
      const auto index_hash = std::hash<std::size_t>{}(value.index());
      const auto value_hash = std::visit(
          [](const auto& item) -> std::size_t {
            using item_type = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<item_type, null_type>) {
              return 0u;
            } else {
              return std::hash<item_type>{}(item);
            }
          },
          value);
      return index_hash ^ (value_hash + 0x9e3779b97f4a7c15ULL + (index_hash << 6U) + (index_hash >> 2U));
    }
  };

  struct CellEqual {
    [[nodiscard]] bool operator()(const cell_type& lhs, const cell_type& rhs) const { return lhs == rhs; }
  };

  using group_key_type = std::vector<cell_type>;

  struct GroupKeyHash {
    [[nodiscard]] std::size_t operator()(const group_key_type& key) const {
      std::size_t seed = 0;
      for (const auto& value : key) {
        const auto value_hash = CellHash{}(value);
        seed ^= value_hash + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
      }
      return seed;
    }
  };

  struct GroupPlan {
    std::vector<group_key_type> order;
    std::unordered_map<group_key_type, std::vector<size_type>, GroupKeyHash> groups;
  };

  using row_lookup_type = std::unordered_map<cell_type, std::vector<size_type>, CellHash, CellEqual>;
  using join_plan_type = std::vector<std::pair<std::string, std::string>>;

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

  [[nodiscard]] const column_data_type& column_data(const std::string& column_name) const {
    const auto it = columns_.find(column_name);
    if (it == columns_.end()) {
      throw std::out_of_range("DataFrame missing column: " + column_name);
    }
    return it->second;
  }

  [[nodiscard]] column_data_type& column_data(const std::string& column_name) {
    const auto it = columns_.find(column_name);
    if (it == columns_.end()) {
      throw std::out_of_range("DataFrame missing column: " + column_name);
    }
    return it->second;
  }

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

  static void require_column_name(const std::string& column_name) {
    if (column_name.empty()) {
      throw std::invalid_argument("DataFrame column names must be non-empty");
    }
  }

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

  [[nodiscard]] DataFrame clone_schema() const {
    DataFrame result;
    for (const auto& name : column_order_) {
      append_schema_column(result, name, column_type(name));
    }
    return result;
  }

  static void append_schema_column(DataFrame& target, const std::string& name, const ColumnType type) {
    if (type == ColumnType::Numeric) {
      target.add_column(name, numeric_column_type{});
    } else {
      target.add_column(name, string_column_type{});
    }
  }

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

  [[nodiscard]] DataFrame take_rows_impl(const std::vector<size_type>& indices) const {
    DataFrame result = clone_schema();
    for (const auto row_index : indices) {
      result.append_row(row(row_index));
    }
    return result;
  }

  void require_group_by_keys(const std::vector<std::string>& key_columns) const {
    if (key_columns.empty()) {
      throw std::invalid_argument("DataFrame group_by requires at least one key column");
    }
    for (const auto& column_name : key_columns) {
      (void)column_data(column_name);
    }
  }

  void require_numeric_columns(const std::vector<std::string>& value_columns) const {
    if (value_columns.empty()) {
      throw std::invalid_argument("DataFrame aggregation requires at least one value column");
    }
    for (const auto& column_name : value_columns) {
      (void)numeric_column(column_name);
    }
  }

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

  [[nodiscard]] size_type group_n_distinct(const std::string& column_name, const std::vector<size_type>& rows) const {
    const auto& data = column_data(column_name);
    return std::visit(
        [&](const auto& column) -> size_type {
          using column_type = std::decay_t<decltype(column)>;
          using value_type = typename column_type::value_type;
          std::unordered_set<value_type> seen;
          bool has_null = false;
          for (const auto row_index : rows) {
            const auto value = column.optional_at(row_index);
            if (value.has_value()) {
              seen.insert(*value);
            } else {
              has_null = true;
            }
          }
          return seen.size() + (has_null ? 1 : 0);
        },
        data);
  }

  [[nodiscard]] group_key_type group_key_for_row(const std::vector<std::string>& key_columns,
                                                 const size_type row_index) const {
    group_key_type key;
    key.reserve(key_columns.size());
    for (const auto& column_name : key_columns) {
      key.push_back(value(column_name, row_index));
    }
    return key;
  }

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

  [[nodiscard]] DataFrame make_group_result_schema(const std::vector<std::string>& key_columns) const {
    DataFrame result;
    for (const auto& column_name : key_columns) {
      append_schema_column(result, column_name, column_type(column_name));
    }
    return result;
  }

  static void append_group_key_row(DataFrame& result, const std::vector<std::string>& key_columns,
                                   const group_key_type& key) {
    Row row;
    for (size_type index = 0; index < key_columns.size(); ++index) {
      row.values.emplace(key_columns[index], key[index]);
    }
    result.append_row(row);
  }

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

  static Row build_bind_row(const std::vector<std::string>& columns, const DataFrame* source, const size_type row_index) {
    Row row;
    for (const auto& name : columns) {
      row.values[name] = source->has_column(name) ? source->value(name, row_index) : cell_type(null_type{});
    }
    return row;
  }

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

  std::vector<std::string> column_order_;
  std::unordered_map<std::string, column_data_type> columns_;
  size_type row_count_{0};
};

/// @brief dplyr-style grouped-DataFrame handle returned by DataFrame::group_by(): pairs a copy
///        of the source DataFrame with a set of key columns, deferring aggregation until
///        summarise()/count() is called. Holding the source by value (rather than a reference or
///        pointer) keeps chains like `df.select(...).group_by(...).summarise(...)` safe, since
///        the intermediate select() result is a temporary that would otherwise dangle.
class GroupedDataFrame {
 public:
  GroupedDataFrame(DataFrame source, std::vector<std::string> keys) : source_(std::move(source)), keys_(std::move(keys)) {}

  [[nodiscard]] DataFrame summarise(const std::vector<DataFrame::AggSpec>& specs) const {
    return source_.summarise(keys_, specs);
  }

  [[nodiscard]] DataFrame count(const std::string& count_column_name = "n") const {
    return source_.count(keys_, count_column_name);
  }

  [[nodiscard]] const std::vector<std::string>& keys() const { return keys_; }

 private:
  DataFrame source_;
  std::vector<std::string> keys_;
};

inline GroupedDataFrame DataFrame::group_by(const std::vector<std::string>& key_columns) const {
  require_group_by_keys(key_columns);
  return GroupedDataFrame(*this, key_columns);
}

} // namespace datamunge::dstruct
