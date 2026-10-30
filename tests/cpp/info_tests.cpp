#include <gtest/gtest.h>

#include <datamunge/info.hpp>

#include <optional>
#include <string>
#include <vector>

namespace dm = datamunge;
using datamunge::dstruct::DataFrame;

namespace {

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

struct HasInfo {
  std::string info() const { return "custom info"; }
};

} // namespace

TEST(Info, DescribesDataFrameColumns) {
  DataFrame frame{{{"score", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1.0, 3.0, std::nullopt})},
                   {"label", DataFrame::string_column_type(std::vector<std::optional<std::string>>{"a", "b", "a"})}}};
  const auto text = dm::info(frame);
  EXPECT_TRUE(contains(text, "DataFrame: 3 rows x 2 columns")) << text;
  EXPECT_TRUE(contains(text, "score")) << text;
  EXPECT_TRUE(contains(text, "numeric")) << text;
  EXPECT_TRUE(contains(text, "min: 1, mean: 2, max: 3")) << text;
  EXPECT_TRUE(contains(text, "distinct: 2")) << text;
}

TEST(Info, DescribesHeadOfAFrame) {
  DataFrame frame{{{"x", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1, 2, 3, 4, 5, 6, 7, 8})}}};
  EXPECT_TRUE(contains(dm::info(dm::head(frame, 2)), "DataFrame: 2 rows x 1 column"));
}

TEST(Info, DescribesVectorsAndScalars) {
  const auto vec = dm::info(std::vector<double>{1, 2, 3, 4, 5, 6, 7});
  EXPECT_TRUE(contains(vec, "std::vector<double> of length 7")) << vec;
  EXPECT_TRUE(contains(vec, "[1, 2, 3, 4, 5, 6, ...]")) << vec;
  EXPECT_TRUE(contains(vec, "min: 1, mean: 4, max: 7")) << vec;

  const auto dm_vec = dm::info(dm::Vector<std::string>{"a", "b", "a"});
  EXPECT_TRUE(contains(dm_vec, "Vector<string> of length 3")) << dm_vec;
  EXPECT_TRUE(contains(dm_vec, "distinct: 2")) << dm_vec;

  EXPECT_TRUE(contains(dm::info(std::vector<bool>{true, false, true}), "true: 2, false: 1"));
  EXPECT_EQ(dm::info(std::vector<int>{}), "std::vector<int> of length 0\n  []");
  EXPECT_EQ(dm::info(2.5), "double: 2.5");
  EXPECT_EQ(dm::info(3), "int: 3");
  EXPECT_EQ(dm::info("hi"), "string of length 2: \"hi\"");
}

TEST(Info, DescribesMatrices) {
  const dm::linalg::DenseMatrix<double> dense(2, 3, std::vector<double>{1, 2, 3, 4, 5, 6});
  const auto text = dm::info(dense);
  EXPECT_TRUE(contains(text, "DenseMatrix<double>: 2 x 3")) << text;
  EXPECT_TRUE(contains(text, "mean: 3.5")) << text;
}

TEST(Info, UsesInfoMemberAsFallback) { EXPECT_EQ(dm::info(HasInfo{}), "custom info"); }
