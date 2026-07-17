#pragma once

#include <algorithm>
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

class DataFrame {
 public:
  using size_type = std::size_t;
  using null_type = std::monostate;
  using numeric_column_type = NullableColumn<double>;
  using string_column_type = NullableColumn<std::string>;
  using column_data_type = std::variant<numeric_column_type, string_column_type>;
  using cell_type = std::variant<null_type, double, std::string>;

  enum class ColumnType { Numeric, String };
  enum class JoinType { Inner, Left };

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

  [[nodiscard]] DataFrame join(const DataFrame& right, const std::string& left_key, const std::string& right_key,
                               const JoinType join_type = JoinType::Inner, const std::string& left_suffix = "_x",
                               const std::string& right_suffix = "_y") const {
    if (!has_column(left_key)) {
      throw std::out_of_range("DataFrame::join missing left key: " + left_key);
    }
    if (!right.has_column(right_key)) {
      throw std::out_of_range("DataFrame::join missing right key: " + right_key);
    }

    const auto right_plan = build_join_right_plan(right, right_key, left_key, right_suffix);
    DataFrame result = make_join_result_schema(right, right_key, left_key, right_suffix);

    const auto right_lookup = right.build_row_lookup(right_key);

    for (size_type left_row = 0; left_row < row_count_; ++left_row) {
      const auto left_value = value(left_key, left_row);
      const auto match_it = right_lookup.find(left_value);
      if (match_it == right_lookup.end()) {
        if (join_type == JoinType::Left) {
          Row merged = row(left_row);
          for (const auto& [source_name, output_name] : right_plan) {
            (void)source_name;
            merged.values[output_name] = null_type{};
          }
          result.append_row(merged);
        }
        continue;
      }

      for (const auto right_row : match_it->second) {
        Row merged = row(left_row);
        for (const auto& [source_name, output_name] : right_plan) {
          merged.values[output_name] = right.value(source_name, right_row);
        }
        result.append_row(merged);
      }
    }

    (void)left_suffix;
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
      if (column_type(name) == ColumnType::Numeric) {
        result.add_column(name, numeric_column_type{});
      } else {
        result.add_column(name, string_column_type{});
      }
    }
    return result;
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
      if (column_type(column_name) == ColumnType::Numeric) {
        result.add_column(column_name, numeric_column_type{});
      } else {
        result.add_column(column_name, string_column_type{});
      }
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

  [[nodiscard]] std::vector<std::pair<std::string, std::string>> build_join_right_plan(
      const DataFrame& right, const std::string& right_key, const std::string& left_key,
      const std::string& right_suffix) const {
    std::vector<std::pair<std::string, std::string>> plan;
    for (const auto& column_name : right.column_order_) {
      if (column_name == right_key && left_key == right_key) {
        continue;
      }

      std::string output_name = column_name;
      if (has_column(output_name)) {
        output_name += right_suffix;
      }
      plan.emplace_back(column_name, std::move(output_name));
    }
    return plan;
  }

  [[nodiscard]] DataFrame make_join_result_schema(const DataFrame& right, const std::string& right_key,
                                                  const std::string& left_key,
                                                  const std::string& right_suffix) const {
    DataFrame result = clone_schema();
    for (const auto& [source_name, output_name] : build_join_right_plan(right, right_key, left_key, right_suffix)) {
      if (right.column_type(source_name) == ColumnType::Numeric) {
        result.add_column(output_name, numeric_column_type{});
      } else {
        result.add_column(output_name, string_column_type{});
      }
    }
    return result;
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

} // namespace datamunge::dstruct
