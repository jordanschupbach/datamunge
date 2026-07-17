#include <datamunge/dstruct/dataframe.hpp>

#ifdef DATAMUNGE_HAVE_ARROW

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::dstruct {

namespace {

void require_arrow_status(const arrow::Status& status, const std::string& context) {
  if (!status.ok()) {
    throw std::runtime_error(context + ": " + status.ToString());
  }
}

} // namespace

std::shared_ptr<arrow::Table> DataFrame::to_arrow() const {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  fields.reserve(column_order_.size());
  arrays.reserve(column_order_.size());

  for (const auto& column_name : column_order_) {
    const auto& data = column_data(column_name);
    std::visit(
        [&](const auto& column) {
          using column_type = std::decay_t<decltype(column)>;
          if constexpr (std::is_same_v<column_type, numeric_column_type>) {
            arrow::DoubleBuilder builder;
            for (size_type row_index = 0; row_index < column.size(); ++row_index) {
              if (column.is_null(row_index)) {
                require_arrow_status(builder.AppendNull(), "DataFrame::to_arrow append null");
              } else {
                require_arrow_status(builder.Append(column.at(row_index)), "DataFrame::to_arrow append double");
              }
            }

            std::shared_ptr<arrow::Array> array;
            require_arrow_status(builder.Finish(&array), "DataFrame::to_arrow finalize double column");
            fields.push_back(arrow::field(column_name, arrow::float64(), true));
            arrays.push_back(std::move(array));
          } else {
            arrow::StringBuilder builder;
            for (size_type row_index = 0; row_index < column.size(); ++row_index) {
              if (column.is_null(row_index)) {
                require_arrow_status(builder.AppendNull(), "DataFrame::to_arrow append null");
              } else {
                require_arrow_status(builder.Append(column.at(row_index)), "DataFrame::to_arrow append string");
              }
            }

            std::shared_ptr<arrow::Array> array;
            require_arrow_status(builder.Finish(&array), "DataFrame::to_arrow finalize string column");
            fields.push_back(arrow::field(column_name, arrow::utf8(), true));
            arrays.push_back(std::move(array));
          }
        },
        data);
  }

  return arrow::Table::Make(arrow::schema(fields), arrays);
}

DataFrame DataFrame::from_arrow(const std::shared_ptr<arrow::Table>& table) {
  if (table == nullptr) {
    throw std::invalid_argument("DataFrame::from_arrow requires a non-null table");
  }

  DataFrame result;
  const auto& fields = table->schema()->fields();
  for (int column_index = 0; column_index < table->num_columns(); ++column_index) {
    const auto& field = fields[static_cast<std::size_t>(column_index)];
    const auto& chunked = table->column(column_index);
    const auto type_id = chunked->type()->id();

    if (type_id == arrow::Type::DOUBLE) {
      std::vector<std::optional<double>> values;
      values.reserve(static_cast<std::size_t>(chunked->length()));
      for (const auto& chunk : chunked->chunks()) {
        const auto& array = static_cast<const arrow::DoubleArray&>(*chunk);
        for (int64_t row_index = 0; row_index < array.length(); ++row_index) {
          if (array.IsNull(row_index)) {
            values.emplace_back(std::nullopt);
          } else {
            values.emplace_back(array.Value(row_index));
          }
        }
      }
      result.add_column(field->name(), std::move(values));
      continue;
    }

    if (type_id == arrow::Type::STRING) {
      std::vector<std::optional<std::string>> values;
      values.reserve(static_cast<std::size_t>(chunked->length()));
      for (const auto& chunk : chunked->chunks()) {
        const auto& array = static_cast<const arrow::StringArray&>(*chunk);
        for (int64_t row_index = 0; row_index < array.length(); ++row_index) {
          if (array.IsNull(row_index)) {
            values.emplace_back(std::nullopt);
          } else {
            values.emplace_back(array.GetString(row_index));
          }
        }
      }
      result.add_column(field->name(), std::move(values));
      continue;
    }

    if (type_id == arrow::Type::LARGE_STRING) {
      std::vector<std::optional<std::string>> values;
      values.reserve(static_cast<std::size_t>(chunked->length()));
      for (const auto& chunk : chunked->chunks()) {
        const auto& array = static_cast<const arrow::LargeStringArray&>(*chunk);
        for (int64_t row_index = 0; row_index < array.length(); ++row_index) {
          if (array.IsNull(row_index)) {
            values.emplace_back(std::nullopt);
          } else {
            values.emplace_back(array.GetString(row_index));
          }
        }
      }
      result.add_column(field->name(), std::move(values));
      continue;
    }

    throw std::invalid_argument("DataFrame::from_arrow unsupported Arrow column type for column: " + field->name());
  }

  return result;
}

} // namespace datamunge::dstruct

#endif
