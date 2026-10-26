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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <locale>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "common/binary.hpp"

namespace data {

struct CsvTable {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
  size_t Column(const std::string& name) const {
    for (size_t i = 0; i < header.size(); ++i) if (header[i] == name) return i;
    throw std::invalid_argument("missing CSV column: " + name);
  }
  const std::string& At(size_t row, size_t column) const { return rows.at(row).at(column); }
};

inline CsvTable ParseCsv(std::string_view text, char delimiter = ',') {
  if (delimiter == '"' || delimiter == '\n' || delimiter == '\r' || delimiter == '\0') throw std::invalid_argument("invalid CSV delimiter");
  if (text.substr(0, 3) == "\xef\xbb\xbf") text.remove_prefix(3);
  std::vector<std::vector<std::string>> records;
  std::vector<std::string> row;
  std::string field;
  bool quoted = false, closed = false, started = false;
  auto end_field = [&] { row.push_back(std::move(field)); field.clear(); closed = started = false; };
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '\0') throw std::invalid_argument("NUL in CSV");
    if (quoted) {
      if (c == '"') {
        if (i + 1 < text.size() && text[i + 1] == '"') { field.push_back('"'); ++i; }
        else { quoted = false; closed = true; }
      } else field.push_back(c);
      continue;
    }
    if (c == delimiter) end_field();
    else if (c == '\n' || c == '\r') {
      if (c == '\r') {
        if (i + 1 == text.size() || text[i + 1] != '\n') throw std::invalid_argument("bare CR outside quoted field");
        ++i;
      }
      end_field(); records.push_back(std::move(row)); row.clear();
    } else if (c == '"' && !started && !closed) { quoted = started = true; }
    else {
      if (closed || c == '"') throw std::invalid_argument("invalid CSV quote placement");
      field.push_back(c); started = true;
    }
  }
  if (quoted) throw std::invalid_argument("unterminated CSV quote");
  if (started || closed || !row.empty()) { end_field(); records.push_back(std::move(row)); }
  if (records.empty()) throw std::invalid_argument("CSV requires a header");
  CsvTable result;
  result.header = std::move(records[0]);
  std::set<std::string> names;
  for (const auto& name : result.header) if (name.empty() || !names.insert(name).second) throw std::invalid_argument("empty or duplicate CSV header");
  for (size_t i = 1; i < records.size(); ++i) {
    if (records[i].size() != result.header.size()) throw std::invalid_argument("CSV record width mismatch at record " + std::to_string(i + 1));
    result.rows.push_back(std::move(records[i]));
  }
  return result;
}

inline CsvTable LoadCsv(const std::filesystem::path& path, char delimiter = ',') {
  const auto bytes = ReadBinary(path, 128ull * 1024 * 1024);
  return ParseCsv(std::string(bytes.begin(), bytes.end()), delimiter);
}

inline std::optional<double> Numeric(std::string_view text) {
  const auto start = text.find_first_not_of(" \t\r\n");
  if (start == std::string_view::npos) return std::nullopt;
  text = text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1);
  if (text.front() == '+') {
    text.remove_prefix(1);
    if (text.empty() || text.front() == '+' || text.front() == '-') throw std::invalid_argument("invalid numeric sign");
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

enum class FeatureKind { kNumeric, kCategorical };
struct FeatureSpec { std::string name; FeatureKind kind; };
struct FittedFeature {
  FeatureSpec spec;
  double mean = 0, scale = 1;
  std::vector<std::string> vocabulary;
};

class FeatureColumns {
 public:
  static FeatureColumns Fit(const CsvTable& table, const std::vector<size_t>& training_rows,
                            const std::vector<FeatureSpec>& specs) {
    if (training_rows.empty() || specs.empty()) throw std::invalid_argument("feature fitting needs training rows and columns");
    FeatureColumns result;
    std::set<std::string> unique;
    for (const auto& spec : specs) {
      if (!unique.insert(spec.name).second) throw std::invalid_argument("duplicate feature column");
      const size_t column = table.Column(spec.name);
      FittedFeature fit{spec, 0, 1, {}};
      if (spec.kind == FeatureKind::kNumeric) {
        size_t count = 0;
        double m2 = 0;
        for (size_t row : training_rows) if (const auto value = Numeric(table.At(row, column))) {
          ++count;
          const double delta = *value - fit.mean;
          fit.mean += delta / count;
          m2 += delta * (*value - fit.mean);
        }
        if (count && m2 > 0) fit.scale = std::sqrt(m2 / count);
        if (!std::isfinite(fit.mean) || !std::isfinite(fit.scale)) throw std::invalid_argument("numeric statistics overflow");
        result.names_.push_back(spec.name + ":value"); result.names_.push_back(spec.name + ":missing");
      } else if (spec.kind == FeatureKind::kCategorical) {
        std::set<std::string> values;
        for (size_t row : training_rows) if (!table.At(row, column).empty()) values.insert(table.At(row, column));
        fit.vocabulary.assign(values.begin(), values.end());
        result.names_.push_back(spec.name + ":missing"); result.names_.push_back(spec.name + ":unknown");
        for (const auto& value : values) result.names_.push_back(spec.name + ":category=" + value);
      } else throw std::invalid_argument("invalid feature kind");
      result.features_.push_back(std::move(fit));
    }
    return result;
  }
  datamunge::nn::Tensor Transform(const CsvTable& table, const std::vector<size_t>& rows) const {
    if (features_.empty()) throw std::logic_error("feature columns are not fitted");
    const std::size_t width = names_.size();
    std::vector<float> result(rows.size() * width, 0.f);
    size_t offset = 0;
    for (const auto& feature : features_) {
      const size_t column = table.Column(feature.spec.name);
      for (size_t r = 0; r < rows.size(); ++r) {
        const auto& value = table.At(rows[r], column);
        if (feature.spec.kind == FeatureKind::kNumeric) {
          const auto numeric = Numeric(value);
          const double normalized = numeric ? (*numeric - feature.mean) / feature.scale : 0;
          if (!std::isfinite(normalized) || std::abs(normalized) > std::numeric_limits<float>::max()) throw std::invalid_argument("normalized feature overflows float32");
          result[r * width + offset] = static_cast<float>(normalized);
          result[r * width + offset + 1] = numeric ? 0.f : 1.f;
        } else {
          size_t index = value.empty() ? 0 : 1;
          const auto it = std::lower_bound(feature.vocabulary.begin(), feature.vocabulary.end(), value);
          if (!value.empty() && it != feature.vocabulary.end() && *it == value) index = 2 + (it - feature.vocabulary.begin());
          result[r * width + offset + index] = 1;
        }
      }
      offset += feature.spec.kind == FeatureKind::kNumeric ? 2 : feature.vocabulary.size() + 2;
    }
    return datamunge::nn::from_vector(result, {std::int64_t(rows.size() * width)})
        .reshape({std::int64_t(rows.size()), std::int64_t(width)});
  }
  const std::vector<FittedFeature>& features() const { return features_; }
  const std::vector<std::string>& names() const { return names_; }
 private:
  std::vector<FittedFeature> features_;
  std::vector<std::string> names_;
};

}  // namespace data
