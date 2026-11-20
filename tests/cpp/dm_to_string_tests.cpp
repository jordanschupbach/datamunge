#include <gtest/gtest.h>

#include <dm/data.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

using datamunge::dstruct::DataFrame;

TEST(DmToString, FormatsCellsAndNumbers) {
  const DataFrame frame{{{"year", DataFrame::numeric_column_type(std::vector<std::optional<double>>{2007, std::nullopt})},
                         {"g", DataFrame::string_column_type(std::vector<std::optional<std::string>>{"a", "b"})}}};
  EXPECT_EQ(dm::to_string(frame.row(0)["year"]), "2007");
  EXPECT_EQ(dm::to_string(frame.row(1)["year"]), "NA");
  EXPECT_EQ(dm::to_string(frame.row(1)["g"]), "b");

  const auto labelled = frame.mutate({{"year", [](auto& row) { return dm::to_string(row["year"]); }}});
  EXPECT_EQ(labelled.column_type("year"), DataFrame::ColumnType::String);
  EXPECT_EQ(labelled.string_at("year", 0), "2007");
  EXPECT_EQ(labelled.string_at("year", 1), "NA");

  EXPECT_EQ(dm::to_string(2007), "2007");
  EXPECT_EQ(dm::to_string(2007.0), "2007");
  EXPECT_EQ(dm::to_string(0.1), "0.1");
  EXPECT_EQ(dm::to_string(std::size_t{18446744073709551615U}), "18446744073709551615");
}
