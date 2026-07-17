#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <datamunge/datamunge.hpp>

namespace datamunge {
void hello() {
  std::cout << "Hello datamunge" << std::endl;
}

double call_with_callback(double x, Callback* cb) {
  return cb ? cb->call(x) : x;
}

std::vector<double> map_dvector_with_callback(const std::vector<double>& values, Callback* cb) {
  if (!cb) {
    return values;
  }
  std::vector<double> out;
  out.reserve(values.size());
  for (double v : values) {
    out.push_back(cb->call(v));
  }
  return out;
}

std::vector<double> make_dvector(double a, double b, double c) {
  return {a, b, c};
}

double sum_dvector(const std::vector<double>& values) {
  double sum = 0.0;
  for (double v : values) {
    sum += v;
  }
  return sum;
}

std::pair<double, double> make_dpair(double a, double b) {
  return {a, b};
}

double sum_dpair(const std::pair<double, double>& values) {
  return values.first + values.second;
}

DataFrame::DataFrame(dstruct::DataFrame frame) : frame_(std::move(frame)) {}

std::size_t DataFrame::nrows() const { return frame_.nrows(); }

std::size_t DataFrame::ncols() const { return frame_.ncols(); }

std::vector<std::size_t> DataFrame::shape() const {
  const auto [rows, cols] = frame_.shape();
  return {rows, cols};
}

std::vector<std::string> DataFrame::columns() const { return frame_.columns(); }

void DataFrame::add_numeric_column(const std::string& column_name, const std::vector<double>& values,
                                   const std::vector<int>& valid_mask) {
  frame_.add_column(column_name, apply_valid_mask(values, valid_mask));
}

void DataFrame::add_string_column(const std::string& column_name, const std::vector<std::string>& values,
                                  const std::vector<int>& valid_mask) {
  frame_.add_column(column_name, apply_valid_mask(values, valid_mask));
}

void DataFrame::add_string_column_encoded(const std::string& column_name, const std::string& encoded_values,
                                          const std::vector<int>& valid_mask) {
  frame_.add_column(column_name, apply_valid_mask(split_encoded_strings(encoded_values), valid_mask));
}

void DataFrame::fill_null_numeric(const std::string& column_name, const double value) { frame_.fill_null(column_name, value); }

void DataFrame::fill_null_string(const std::string& column_name, const std::string& value) {
  frame_.fill_null(column_name, value);
}

DataFrame* DataFrame::select(const std::vector<std::string>& selected_columns) const {
  return new DataFrame(frame_.select(selected_columns));
}

DataFrame* DataFrame::select_encoded(const std::string& encoded_columns) const {
  return new DataFrame(frame_.select(split_encoded_strings(encoded_columns)));
}

DataFrame* DataFrame::sort_by(const std::string& column_name, const bool ascending) const {
  return new DataFrame(frame_.sort_by(column_name, ascending));
}

DataFrame* DataFrame::drop_duplicates(const std::vector<std::string>& subset) const {
  return new DataFrame(frame_.drop_duplicates(subset));
}

DataFrame* DataFrame::drop_duplicates_encoded(const std::string& encoded_subset) const {
  return new DataFrame(frame_.drop_duplicates(split_encoded_strings(encoded_subset)));
}

DataFrame* DataFrame::group_by_sum(const std::vector<std::string>& key_columns,
                                   const std::vector<std::string>& value_columns) const {
  return new DataFrame(frame_.group_by_sum(key_columns, value_columns));
}

DataFrame* DataFrame::group_by_sum_encoded(const std::string& encoded_key_columns,
                                           const std::string& encoded_value_columns) const {
  return new DataFrame(
      frame_.group_by_sum(split_encoded_strings(encoded_key_columns), split_encoded_strings(encoded_value_columns)));
}

DataFrame* DataFrame::join(const DataFrame& right, const std::string& left_key, const std::string& right_key,
                           const bool left_join) const {
  return new DataFrame(frame_.join(right.frame_, left_key, right_key,
                                   left_join ? dstruct::DataFrame::JoinType::Left : dstruct::DataFrame::JoinType::Inner));
}

std::size_t DataFrame::numeric_count(const std::string& column_name) const {
  return frame_.describe_numeric(column_name).count;
}

std::size_t DataFrame::numeric_null_count(const std::string& column_name) const {
  return frame_.describe_numeric(column_name).null_count;
}

double DataFrame::numeric_sum(const std::string& column_name) const { return frame_.describe_numeric(column_name).sum; }

double DataFrame::numeric_mean(const std::string& column_name) const { return frame_.describe_numeric(column_name).mean; }

double DataFrame::numeric_min(const std::string& column_name) const { return frame_.describe_numeric(column_name).min; }

double DataFrame::numeric_max(const std::string& column_name) const { return frame_.describe_numeric(column_name).max; }

std::string DataFrame::to_string(const std::size_t max_rows) const { return frame_.to_string(max_rows); }

template <typename T>
std::vector<std::optional<T>> DataFrame::apply_valid_mask(const std::vector<T>& values, const std::vector<int>& valid_mask) {
  if (!valid_mask.empty() && valid_mask.size() != values.size()) {
    throw std::invalid_argument("DataFrame valid mask length mismatch");
  }

  std::vector<std::optional<T>> result;
  result.reserve(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!valid_mask.empty() && valid_mask[index] == 0) {
      result.emplace_back(std::nullopt);
    } else {
      result.emplace_back(values[index]);
    }
  }
  return result;
}

std::vector<std::string> DataFrame::split_encoded_strings(const std::string& encoded_values) {
  const auto header_end = encoded_values.find('\x1e');
  if (header_end == std::string::npos) {
    std::vector<std::string> values;
    if (encoded_values.empty()) {
      return values;
    }
    std::stringstream stream(encoded_values);
    std::string item;
    while (std::getline(stream, item, '\x1f')) {
      values.push_back(item);
    }
    return values;
  }

  const auto expected_count = static_cast<std::size_t>(std::stoull(encoded_values.substr(0, header_end)));
  const auto payload = encoded_values.substr(header_end + 1);

  std::vector<std::string> values;
  values.reserve(expected_count);

  std::size_t start = 0;
  while (start <= payload.size() && values.size() < expected_count) {
    const auto delimiter = payload.find('\x1f', start);
    if (delimiter == std::string::npos) {
      values.push_back(payload.substr(start));
      break;
    }
    values.push_back(payload.substr(start, delimiter - start));
    start = delimiter + 1;
  }

  while (values.size() < expected_count) {
    values.push_back("");
  }
  return values;
}

} // namespace datamunge
