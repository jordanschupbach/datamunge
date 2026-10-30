#pragma once

/// @file info.hpp
/// @brief `info(x)`: a short structural summary of a value, returned as a std::string.
///
/// @code
/// auto iris = dm::datasets::iris();
/// std::print("{}\n", dm::info(dm::head(iris, 10)));
/// @endcode
///
/// Supported out of the box: dstruct::DataFrame, Vector<T>, std::vector<T>,
/// linalg::DenseMatrix<T>, linalg::SparseMatrix<T>, linalg::Tensor, arithmetic
/// scalars and strings. To support another type, either add an overload
/// `std::string info(const MyType&)` in namespace datamunge, or give the type a
/// const member `info()` returning something convertible to std::string.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/sparse_matrix.hpp>
#include <datamunge/linalg/tensor.hpp>
#include <datamunge/vector.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge {

/// @brief Implementation helpers for info(); not part of the public API.
namespace info_detail {

/// @brief Values shown in a "first few values" preview.
inline constexpr std::size_t preview_count = 6;

/// @brief Readable name of an element type, e.g. "double" or "string".
/// @tparam T Element type.
template <typename T>
std::string type_name() {
  if constexpr (std::is_same_v<T, bool>) return "bool";
  else if constexpr (std::is_same_v<T, double>) return "double";
  else if constexpr (std::is_same_v<T, float>) return "float";
  else if constexpr (std::is_same_v<T, long double>) return "long double";
  else if constexpr (std::is_same_v<T, char>) return "char";
  else if constexpr (std::is_same_v<T, int>) return "int";
  else if constexpr (std::is_same_v<T, long>) return "long";
  else if constexpr (std::is_same_v<T, long long>) return "long long";
  else if constexpr (std::is_same_v<T, unsigned>) return "unsigned";
  else if constexpr (std::is_same_v<T, unsigned long>) return "unsigned long";
  else if constexpr (std::is_same_v<T, unsigned long long>) return "unsigned long long";
  else if constexpr (std::is_same_v<T, std::string>) return "string";
  else if constexpr (std::is_integral_v<T>) return std::is_signed_v<T> ? "integer" : "unsigned integer";
  else return "object";
}

/// @brief Formats one value for a preview (bools as true/false, strings quoted).
/// @tparam T Value type.
/// @param value Value to format.
template <typename T>
std::string format_value(const T& value) {
  std::ostringstream out;
  if constexpr (std::is_same_v<T, bool>) {
    out << (value ? "true" : "false");
  } else if constexpr (std::is_same_v<T, std::string>) {
    out << '"' << value << '"';
  } else {
    out << value;
  }
  return out.str();
}

/// @brief One-line summary of a sequence of values: a preview of the first few, then
///        min/mean/max (numeric), true count (bool) or distinct count (string).
/// @tparam Range Iterable sequence of values.
/// @param values Values to summarize.
/// @param size   Number of values in @p values.
template <typename Range>
std::string summarize_values(const Range& values, std::size_t size) {
  using T = std::decay_t<decltype(*std::begin(values))>;
  std::ostringstream out;

  out << '[';
  std::size_t shown = 0;
  for (const auto& value : values) {
    if (shown == preview_count) break;
    out << (shown == 0 ? "" : ", ") << format_value(static_cast<T>(value));
    ++shown;
  }
  if (size > shown) out << (shown == 0 ? "..." : ", ...");
  out << ']';

  if (size == 0) return out.str();

  if constexpr (std::is_same_v<T, bool>) {
    std::size_t trues = 0;
    for (const bool value : values) trues += value ? 1 : 0;
    out << "  true: " << trues << ", false: " << (size - trues);
  } else if constexpr (std::is_arithmetic_v<T>) {
    std::size_t nan_count = 0;
    std::size_t counted = 0;
    long double sum = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    for (const auto& raw : values) {
      const auto value = static_cast<double>(raw);
      if (std::isnan(value)) {
        ++nan_count;
        continue;
      }
      ++counted;
      sum += value;
      min = std::min(min, value);
      max = std::max(max, value);
    }
    if (counted > 0) {
      out << "  min: " << min << ", mean: " << static_cast<double>(sum / counted) << ", max: " << max;
    }
    if (nan_count > 0) out << "  NaN: " << nan_count;
  } else if constexpr (std::is_same_v<T, std::string>) {
    std::unordered_set<std::string> distinct(std::begin(values), std::end(values));
    out << "  distinct: " << distinct.size();
  }
  return out.str();
}

/// @brief True when `T` has a const `info()` member convertible to std::string.
/// @tparam T Type to inspect.
template <typename T, typename = void>
struct has_info_member : std::false_type {};

/// @brief Specialization selected when `T::info()` exists.
/// @tparam T Type to inspect.
template <typename T>
struct has_info_member<T, std::void_t<decltype(std::string(std::declval<const T&>().info()))>> : std::true_type {};

} // namespace info_detail

/// @brief Describes a DataFrame, like pandas' `DataFrame.info()` or R's `str()`.
///
/// The first line gives the size ("DataFrame: 60 rows x 3 columns"), followed by an aligned
/// table with one line per column: index, name, type ("numeric" or "string"), non-null count,
/// null count and a summary. Numeric columns show min/mean/max over their non-null values
/// (blank if every value is null); string columns show the number of distinct values, where
/// nulls, if any, count as one extra distinct value (like dplyr::n_distinct()). The result has
/// no trailing newline.
///
/// @param frame  Frame to describe; it is not modified.
/// @return  A multi-line description.
///
/// @par Example
/// @code
/// const auto tooth = dm::datasets::tooth_growth();
/// std::print("{}\n", dm::info(tooth));
/// // DataFrame: 60 rows x 3 columns
/// // #  Column  Type     Non-null  Nulls  Summary
/// // 0  len     numeric        60      0  min: 4.2, mean: 18.8133, max: 33.9
/// // 1  supp    string         60      0  distinct: 2
/// // 2  dose    numeric        60      0  min: 0.5, mean: 1.16667, max: 2
/// @endcode
/// @see DataFrame::describe_numeric(), DataFrame::null_count(), DataFrame::n_distinct()
inline std::string info(const dstruct::DataFrame& frame) {
  using ColumnType = dstruct::DataFrame::ColumnType;
  const auto names = frame.columns();

  std::vector<std::vector<std::string>> table;
  table.push_back({"#", "Column", "Type", "Non-null", "Nulls", "Summary"});
  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto& name = names[i];
    const auto nulls = frame.null_count(name);
    std::ostringstream summary;
    std::string type;
    if (frame.column_type(name) == ColumnType::Numeric) {
      type = "numeric";
      const auto stats = frame.describe_numeric(name);
      if (stats.count > 0) {
        summary << "min: " << stats.min << ", mean: " << stats.sum / static_cast<double>(stats.count)
                << ", max: " << stats.max;
      }
    } else {
      type = "string";
      summary << "distinct: " << frame.n_distinct(name);
    }
    table.push_back({std::to_string(i), name, type, std::to_string(frame.nrows() - nulls), std::to_string(nulls),
                     summary.str()});
  }

  std::vector<std::size_t> width(table.front().size(), 0);
  for (const auto& row : table) {
    for (std::size_t j = 0; j < row.size(); ++j) width[j] = std::max(width[j], row[j].size());
  }

  std::ostringstream out;
  out << "DataFrame: " << frame.nrows() << (frame.nrows() == 1 ? " row" : " rows") << " x " << frame.ncols()
      << (frame.ncols() == 1 ? " column" : " columns");
  for (const auto& row : table) {
    out << "\n";
    for (std::size_t j = 0; j < row.size(); ++j) {
      // Counts are right-aligned, text left-aligned; the last column is not padded.
      const bool right = j == 0 || j == 3 || j == 4;
      const auto pad = std::string(width[j] - row[j].size(), ' ');
      if (j > 0) out << "  ";
      if (right) out << pad << row[j];
      else out << row[j] << (j + 1 < row.size() ? pad : "");
    }
  }
  return out.str();
}

/// @brief Describes a Vector<T>: element type, length, a preview and a summary, like R's `str()`.
///
/// Two lines: "Vector<type> of length n", then the first 6 values (followed by "..." when
/// there are more) and a summary that depends on T -- min/mean/max for arithmetic types
/// (NaNs are skipped and counted separately), true/false counts for bool, the number of
/// distinct values for strings. An empty vector shows just "[]".
///
/// @tparam T  Element type.
/// @param values  Vector to describe.
/// @return  A two-line description with no trailing newline.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::info(dm::c(1.0, 2.0, 3.0, 4.0)));
/// // Vector<double> of length 4
/// //   [1, 2, 3, 4]  min: 1, mean: 2.5, max: 4
/// @endcode
template <typename T>
std::string info(const Vector<T>& values) {
  return "Vector<" + info_detail::type_name<T>() + "> of length " + std::to_string(values.size()) + "\n  " +
         info_detail::summarize_values(values.values(), values.size());
}

/// @brief Describes a std::vector<T> the same way as Vector<T> (header "std::vector<type>").
/// @tparam T  Element type.
/// @tparam Allocator  Allocator type.
/// @param values  Vector to describe.
/// @return  A two-line description.
template <typename T, typename Allocator>
std::string info(const std::vector<T, Allocator>& values) {
  return "std::vector<" + info_detail::type_name<T>() + "> of length " + std::to_string(values.size()) + "\n  " +
         info_detail::summarize_values(values, values.size());
}

/// @brief Describes a dense matrix: element type, dimensions and a min/mean/max summary.
/// @tparam T  Element type.
/// @param matrix  Matrix to describe.
/// @return  A two-line description ("DenseMatrix<double>: 3 x 4", then the value summary).
template <typename T>
std::string info(const linalg::DenseMatrix<T>& matrix) {
  return "DenseMatrix<" + info_detail::type_name<T>() + ">: " + std::to_string(matrix.rows()) + " x " +
         std::to_string(matrix.cols()) + "\n  " + info_detail::summarize_values(matrix.data(), matrix.data().size());
}

/// @brief Describes a sparse matrix: storage format, dimensions, stored entries and density.
/// @tparam T  Element type.
/// @param matrix  Matrix to describe.
/// @return  A two-line description; density is stored entries / (rows x cols).
template <typename T>
std::string info(const linalg::SparseMatrix<T>& matrix) {
  using Format = typename linalg::SparseMatrix<T>::Format;
  const char* format = "?";
  switch (matrix.format()) {
    case Format::COO: format = "COO"; break;
    case Format::CSR: format = "CSR"; break;
    case Format::CSC: format = "CSC"; break;
    case Format::ELL: format = "ELL"; break;
    case Format::DIA: format = "DIA"; break;
  }
  const double cells = static_cast<double>(matrix.rows()) * static_cast<double>(matrix.cols());
  std::ostringstream out;
  out << "SparseMatrix<" << info_detail::type_name<T>() << "> (" << format << "): " << matrix.rows() << " x "
      << matrix.cols() << "\n  stored entries: " << matrix.nnz()
      << ", density: " << (cells > 0 ? static_cast<double>(matrix.nnz()) / cells : 0.0);
  return out.str();
}

/// @brief Describes a tensor: dtype, shape, element count and a value summary.
/// @param tensor  Tensor to describe.
/// @return  A two-line description ("Tensor<dtype>: shape [...], n elements", then values).
inline std::string info(const linalg::Tensor& tensor) {
  std::ostringstream out;
  out << "Tensor<" << tensor.dtype_name() << ">: shape [";
  const auto& shape = tensor.shape();
  for (std::size_t i = 0; i < shape.size(); ++i) out << (i == 0 ? "" : ", ") << shape[i];
  out << "], " << tensor.size() << (tensor.size() == 1 ? " element" : " elements") << "\n  ";

  if (tensor.dtype() == linalg::TensorDType::String) {
    std::vector<std::string> values;
    values.reserve(tensor.size());
    for (std::size_t i = 0; i < tensor.size(); ++i) values.push_back(tensor.string_at_flat(i));
    out << info_detail::summarize_values(values, values.size());
  } else if (tensor.dtype() == linalg::TensorDType::Bool) {
    std::vector<bool> values;
    values.reserve(tensor.size());
    for (std::size_t i = 0; i < tensor.size(); ++i) values.push_back(tensor.at_flat(i) != 0.0);
    out << info_detail::summarize_values(values, values.size());
  } else {
    std::vector<double> values;
    values.reserve(tensor.size());
    for (std::size_t i = 0; i < tensor.size(); ++i) values.push_back(tensor.at_flat(i));
    out << info_detail::summarize_values(values, values.size());
  }
  return out.str();
}

/// @brief Describes a string: its length and value.
/// @param value  String to describe.
/// @return  e.g. `string of length 3: "abc"`.
inline std::string info(const std::string& value) {
  return "string of length " + std::to_string(value.size()) + ": \"" + value + "\"";
}

/// @brief Describes a C string the same way as std::string.
/// @param value  Null-terminated string (must not be null).
/// @return  e.g. `string of length 3: "abc"`.
inline std::string info(const char* value) { return info(std::string(value)); }

/// @brief Describes a scalar: its type and value.
/// @tparam T  Arithmetic type.
/// @param value  Value to describe.
/// @return  e.g. `double: 3.5`.
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
std::string info(const T& value) {
  return info_detail::type_name<T>() + ": " + info_detail::format_value(value);
}

/// @brief Fallback for any type with a const `info()` member returning a string.
/// @tparam T  Type with an `info()` member.
/// @param value  Object to describe.
/// @return  `std::string(value.info())`.
template <typename T,
          std::enable_if_t<info_detail::has_info_member<T>::value && !std::is_arithmetic_v<T>, int> = 0>
std::string info(const T& value) {
  return std::string(value.info());
}

} // namespace datamunge
