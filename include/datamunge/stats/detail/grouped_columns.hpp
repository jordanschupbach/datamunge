#pragma once

/// @file grouped_columns.hpp
/// @brief Extracts a (value, group) pair of DataFrame columns for group-comparison routines.

#include <datamunge/dstruct/dataframe.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::stats::detail {

/// @brief Copies a numeric value column and a string group column, dropping rows where either
///        is null (as R drops NA in aov() and pairwise.t.test()).
/// @param data Source frame; unchanged.
/// @param value_column Numeric column name.
/// @param group_column String column name.
/// @param caller Prefix for error messages.
/// @return The non-null values and their group labels, in row order.
/// @throws std::invalid_argument A missing column, or a column of the wrong type.
inline std::pair<std::vector<double>, std::vector<std::string>> grouped_columns(
    const dstruct::DataFrame& data, const std::string& value_column, const std::string& group_column,
    const std::string& caller) {
  using ColumnType = dstruct::DataFrame::ColumnType;
  if (!data.has_column(value_column) || !data.has_column(group_column))
    throw std::invalid_argument(caller + ": column not found");
  if (data.column_type(value_column) != ColumnType::Numeric)
    throw std::invalid_argument(caller + ": value column must be numeric");
  if (data.column_type(group_column) != ColumnType::String)
    throw std::invalid_argument(caller + ": group column must be a string column");
  const auto& value = data.get_column<double>(value_column);
  const auto& group = data.get_column<std::string>(group_column);
  std::pair<std::vector<double>, std::vector<std::string>> rows;
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    if (value.is_null(i) || group.is_null(i)) continue;
    rows.first.push_back(value.values()[i]);
    rows.second.push_back(group.values()[i]);
  }
  return rows;
}

} // namespace datamunge::stats::detail
