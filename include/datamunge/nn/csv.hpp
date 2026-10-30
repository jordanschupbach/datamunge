// CSV tables and fitted numeric/categorical feature columns (port of tflow's
// common/data/csv.h; Transform returns a float32 nn::Tensor).
//
// Quoting follows RFC 4180: doubled quotes, embedded delimiters and embedded
// newlines. Accept CRLF or LF records and an optional UTF-8 BOM; preserve field
// whitespace. Require a unique, nonempty header and rectangular records.
//
// Fit feature statistics and categorical vocabularies ONLY on training rows.
// Numeric columns emit a standardized value plus a missing indicator. Empty
// numeric fields (after ASCII trimming) are imputed to the training mean.
// Use population variance of observed training values; constant/all-missing
// columns use scale 1 (all-missing mean 0). Categorical columns preserve exact
// strings, with separate missing and unknown buckets followed by sorted known
// values. Transform never changes the fitted state or sees target columns
// unless the caller explicitly includes them in the feature specification.
//
// nn::read_csv (the library) reads purely numeric CSV into a float64 matrix;
// this header covers quoted text fields and categorical features.
#pragma once

/// @file csv.hpp
/// @brief Quoted CSV tables and training-only fitted feature transformations.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/ops.hpp>
#include <limits>
#include <locale>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "datamunge/nn/binary.hpp"

namespace datamunge::nn::data {

/// @brief Owned rectangular string table with a unique nonempty header.
///
/// ParseCsv validates the structure. Direct caller mutation must preserve it.
/// Row/column positions are zero-based; references expire on storage mutation.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto t = d::ParseCsv("a,b\n1,x\n");
/// const auto& x = t.At(0, t.Column("b")); // "x"
/// @endcode
/// @see ParseCsv()
struct CsvTable {
  std::vector<std::string> header; ///< Unique nonempty column names in source order.
  std::vector<std::vector<std::string>>
      rows; ///< Data cells, excluding header; every row has header width.

            /// @brief Find a column by its exact header name.
            /// @param name Case-sensitive name.
            /// @return Zero-based header index.
            /// @throws std::invalid_argument If the name is absent.
  size_t Column(const std::string& name) const {
    for (size_t i = 0; i < header.size(); ++i)
      if (header[i] == name)
        return i;
    throw std::invalid_argument("missing CSV column: " + name);
  }
  /// @brief Access one cell without numeric coercion.
  /// @param row Zero-based data row.
  /// @param column Zero-based column.
  /// @return Reference into rows, invalidated by mutation.
  /// @throws std::out_of_range On an invalid position.
  const std::string& At(size_t row, size_t column) const { return rows.at(row).at(column); }
};

/// @brief Parse quoted CSV with preserved field whitespace.
///
/// Doubled quotes escape a quote inside a quoted field. Embedded newlines are retained.
/// This parser does not validate UTF-8 or interpret missing-value markers.
/// @param text UTF-8-compatible bytes; optional UTF-8 BOM, LF or CRLF records.
/// @param delimiter Single delimiter character, default comma.
/// @return Owned table; header-only input has zero rows.
/// @throws std::invalid_argument On bad quoting, NUL, invalid delimiter, bare CR, absent/duplicate
/// headers or ragged rows.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto t = d::ParseCsv("a,b\n1,\"x,y\"\n");
/// @endcode
///
/// @par References
/// - Shafranovich (2005), RFC 4180, https://www.rfc-editor.org/rfc/rfc4180
/// @see LoadCsv()
inline CsvTable ParseCsv(std::string_view text, char delimiter = ',') {
  if (delimiter == '"' || delimiter == '\n' || delimiter == '\r' || delimiter == '\0')
    throw std::invalid_argument("invalid CSV delimiter");
  if (text.substr(0, 3) == "\xef\xbb\xbf")
    text.remove_prefix(3);
  std::vector<std::vector<std::string>> records;
  std::vector<std::string>              row;
  std::string                           field;
  bool                                  quoted = false, closed = false, started = false;
  auto                                  end_field = [&] {
    row.push_back(std::move(field));
    field.clear();
    closed = started = false;
  };
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '\0')
      throw std::invalid_argument("NUL in CSV");
    if (quoted) {
      if (c == '"') {
        if (i + 1 < text.size() && text[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else {
          quoted = false;
          closed = true;
        }
      } else
        field.push_back(c);
      continue;
    }
    if (c == delimiter)
      end_field();
    else if (c == '\n' || c == '\r') {
      if (c == '\r') {
        if (i + 1 == text.size() || text[i + 1] != '\n')
          throw std::invalid_argument("bare CR outside quoted field");
        ++i;
      }
      end_field();
      records.push_back(std::move(row));
      row.clear();
    } else if (c == '"' && !started && !closed) {
      quoted = started = true;
    } else {
      if (closed || c == '"')
        throw std::invalid_argument("invalid CSV quote placement");
      field.push_back(c);
      started = true;
    }
  }
  if (quoted)
    throw std::invalid_argument("unterminated CSV quote");
  if (started || closed || !row.empty()) {
    end_field();
    records.push_back(std::move(row));
  }
  if (records.empty())
    throw std::invalid_argument("CSV requires a header");
  CsvTable result;
  result.header = std::move(records[0]);
  std::set<std::string> names;
  for (const auto& name : result.header)
    if (name.empty() || !names.insert(name).second)
      throw std::invalid_argument("empty or duplicate CSV header");
  for (size_t i = 1; i < records.size(); ++i) {
    if (records[i].size() != result.header.size())
      throw std::invalid_argument("CSV record width mismatch at record " + std::to_string(i + 1));
    result.rows.push_back(std::move(records[i]));
  }
  return result;
}

/// @brief Read at most 128 MiB and parse a CSV table.
/// @param path File path.
/// @param delimiter Field separator, default comma.
/// @return Parsed string table.
/// @throws std::runtime_error On file errors; std::invalid_argument on malformed CSV.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto table = d::LoadCsv("features.csv");
/// @endcode
/// @see ParseCsv()
inline CsvTable LoadCsv(const std::filesystem::path& path, char delimiter = ',') {
  const auto bytes = ReadBinary(path, 128ull * 1024 * 1024);
  return ParseCsv(std::string(bytes.begin(), bytes.end()), delimiter);
}

/// @brief Parse one finite number or recognise an empty numeric cell.
/// @param text Number with optional ASCII whitespace and leading plus.
/// @return Finite double, or nullopt for whitespace-only input.
/// @throws std::invalid_argument On trailing text, bad signs, nonfinite values or overflow.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto value = d::Numeric(" +1.25e2 "); // 125
/// @endcode
/// @see FeatureColumns
inline std::optional<double> Numeric(std::string_view text) {
  const auto start = text.find_first_not_of(" \t\r\n");
  if (start == std::string_view::npos)
    return std::nullopt;
  text = text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1);
  if (text.front() == '+') {
    text.remove_prefix(1);
    if (text.empty() || text.front() == '+' || text.front() == '-')
      throw std::invalid_argument("invalid numeric sign");
  }
  double value = 0;
  // std::from_chars for double is unavailable before macOS 26, so parse with
  // the classic locale and no whitespace skipping (strict, like from_chars).
  std::istringstream input{std::string(text)};
  input.imbue(std::locale::classic());
  input >> std::noskipws >> value;
  if (!input || input.peek() != std::char_traits<char>::eof() || !std::isfinite(value))
    throw std::invalid_argument("invalid finite numeric field: " + std::string(text));
  return value;
}

/// @brief Numeric standardisation or exact-string categorical one-hot encoding.
enum class FeatureKind {
  kNumeric,    ///< Standardised value and missingness indicator.
  kCategorical ///< Missing, unknown and sorted known-category indicators.
};
/// @brief Named input column and its encoding kind.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// d::FeatureSpec age{"age", d::FeatureKind::kNumeric};
/// @endcode
/// @see FeatureColumns
struct FeatureSpec {
  std::string name; ///< Exact input header name.
  FeatureKind kind; ///< Encoding type.
};
/// @brief Training-only moments or sorted categorical vocabulary for one column.
///
/// Numeric scale is observed population standard deviation; constant or all-missing
/// columns use scale 1 and all-missing mean 0. Vocabulary excludes empty strings.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto t = d::ParseCsv("x\n1\n3\n");
/// const auto f = d::FeatureColumns::Fit(t, {0,1}, {{"x",d::FeatureKind::kNumeric}});
/// const auto mean = f.features()[0].mean; // 2
/// @endcode
/// @see FeatureColumns
struct FittedFeature {
  FeatureSpec              spec;       ///< Input column name and encoding.
  double                   mean  = 0;  ///< Observed training mean, zero if entirely missing.
  double                   scale = 1;  ///< Population deviation, with unit fallback.
  std::vector<std::string> vocabulary; ///< Sorted nonempty training categories.
};

/// @brief Fit immutable numeric/categorical feature state on selected training rows.
///
/// Numeric features emit value then missing indicator. Categories emit missing, unknown,
/// then known categories in sorted order. No target column is inferred. Header lookup
/// makes column order irrelevant. Transform never refits the state; concurrent const calls
/// are safe when the table and fitted object are not being modified.
///
/// @par Example
/// @code
/// namespace d = dm::nn::data;
/// const auto t = d::ParseCsv("x,city\n1,A\n3,B\n9,C\n");
/// const auto f = d::FeatureColumns::Fit(t, {0,1},
/// {{"x",d::FeatureKind::kNumeric},{"city",d::FeatureKind::kCategorical}}); const auto validation =
/// f.Transform(t, {2});
/// @endcode
/// @see FeatureSpec
class FeatureColumns {
 public:
  /// @brief Estimate feature state using only explicitly selected rows.
  /// @param table Rectangular string table.
  /// @param training_rows Nonempty zero-based training indices; duplicates act as repeated
  /// observations.
  /// @param specs Nonempty unique input-column specifications in output order.
  /// @return Fitted object with fixed output names.
  /// @throws std::invalid_argument On invalid specifications, missing columns, invalid numbers or
  /// overflow; std::out_of_range on invalid rows.
  ///
  /// @par Example
  /// @code
  /// namespace d = dm::nn::data;
  /// const auto t = d::ParseCsv("x\n1\n3\n");
  /// const auto f = d::FeatureColumns::Fit(t,{0,1},{{"x",d::FeatureKind::kNumeric}});
  /// @endcode
  /// @see Transform()
  static FeatureColumns Fit(
      const CsvTable&                 table,
      const std::vector<size_t>&      training_rows,
      const std::vector<FeatureSpec>& specs) {
    if (training_rows.empty() || specs.empty())
      throw std::invalid_argument("feature fitting needs training rows and columns");
    FeatureColumns        result;
    std::set<std::string> unique;
    for (const auto& spec : specs) {
      if (!unique.insert(spec.name).second)
        throw std::invalid_argument("duplicate feature column");
      const size_t  column = table.Column(spec.name);
      FittedFeature fit{spec, 0, 1, {}};
      if (spec.kind == FeatureKind::kNumeric) {
        size_t count = 0;
        double m2    = 0;
        for (size_t row : training_rows)
          if (const auto value = Numeric(table.At(row, column))) {
            ++count;
            const double delta = *value - fit.mean;
            fit.mean += delta / count;
            m2 += delta * (*value - fit.mean);
          }
        if (count && m2 > 0)
          fit.scale = std::sqrt(m2 / count);
        if (!std::isfinite(fit.mean) || !std::isfinite(fit.scale))
          throw std::invalid_argument("numeric statistics overflow");
        result.names_.push_back(spec.name + ":value");
        result.names_.push_back(spec.name + ":missing");
      } else if (spec.kind == FeatureKind::kCategorical) {
        std::set<std::string> values;
        for (size_t row : training_rows)
          if (!table.At(row, column).empty())
            values.insert(table.At(row, column));
        fit.vocabulary.assign(values.begin(), values.end());
        result.names_.push_back(spec.name + ":missing");
        result.names_.push_back(spec.name + ":unknown");
        for (const auto& value : values)
          result.names_.push_back(spec.name + ":category=" + value);
      } else
        throw std::invalid_argument("invalid feature kind");
      result.features_.push_back(std::move(fit));
    }
    return result;
  }
  /// @brief Apply fitted preprocessing without changing moments or vocabulary.
  /// @param table Compatible named columns; may be reordered.
  /// @param rows Zero-based rows, in desired output order; may be empty.
  /// @return Owned Float32 [rows, names().size()] tensor without gradient history.
  /// @throws std::logic_error If unfitted; std::invalid_argument on absent columns, invalid numbers
  /// or overflow; std::out_of_range on invalid rows.
  ///
  /// @par Example
  /// @code
  /// namespace d = dm::nn::data;
  /// const auto t = d::ParseCsv("x\n1\n3\n");
  /// const auto x =
  /// d::FeatureColumns::Fit(t,{0,1},{{"x",d::FeatureKind::kNumeric}}).Transform(t,{1});
  /// @endcode
  /// @see Fit()
  datamunge::nn::Tensor Transform(const CsvTable& table, const std::vector<size_t>& rows) const {
    if (features_.empty())
      throw std::logic_error("feature columns are not fitted");
    const std::size_t  width = names_.size();
    std::vector<float> result(rows.size() * width, 0.f);
    size_t             offset = 0;
    for (const auto& feature : features_) {
      const size_t column = table.Column(feature.spec.name);
      for (size_t r = 0; r < rows.size(); ++r) {
        const auto& value = table.At(rows[r], column);
        if (feature.spec.kind == FeatureKind::kNumeric) {
          const auto   numeric    = Numeric(value);
          const double normalized = numeric ? (*numeric - feature.mean) / feature.scale : 0;
          if (!std::isfinite(normalized) ||
              std::abs(normalized) > std::numeric_limits<float>::max())
            throw std::invalid_argument("normalized feature overflows float32");
          result[r * width + offset]     = static_cast<float>(normalized);
          result[r * width + offset + 1] = numeric ? 0.f : 1.f;
        } else {
          size_t     index = value.empty() ? 0 : 1;
          const auto it =
              std::lower_bound(feature.vocabulary.begin(), feature.vocabulary.end(), value);
          if (!value.empty() && it != feature.vocabulary.end() && *it == value)
            index = 2 + (it - feature.vocabulary.begin());
          result[r * width + offset + index] = 1;
        }
      }
      offset += feature.spec.kind == FeatureKind::kNumeric ? 2 : feature.vocabulary.size() + 2;
    }
    return datamunge::nn::from_vector(result, {std::int64_t(rows.size() * width)})
        .reshape({std::int64_t(rows.size()), std::int64_t(width)});
  }
  /// @brief Inspect fitted state without modifying it.
  /// @return Borrowed state in specification order, valid for the object lifetime.
  const std::vector<FittedFeature>& features() const { return features_; }
  /// @brief Inspect the stable encoded output-column names.
  /// @return Borrowed output names, valid for the object lifetime.
  const std::vector<std::string>& names() const { return names_; }

 private:
  std::vector<FittedFeature> features_; ///< Immutable fitted states in specification order.
  std::vector<std::string>   names_;    ///< Flattened encoded output names.
};

} // namespace datamunge::nn::data
