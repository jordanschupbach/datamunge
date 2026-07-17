#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::dstruct::DataFrame;

TEST(DataFrame, SupportsTypedColumnsCellAccessAndNullTracking) {
  DataFrame frame;
  frame.add_column("score", std::vector<std::optional<double>>{1.5, std::nullopt, 3.5});
  frame.add_column("label", std::vector<std::optional<std::string>>{"a", "b", std::nullopt});

  EXPECT_EQ(frame.nrows(), 3u);
  EXPECT_EQ(frame.ncols(), 2u);
  EXPECT_EQ(frame.shape(), (std::pair<std::size_t, std::size_t>{3u, 2u}));
  EXPECT_EQ(frame.columns(), (std::vector<std::string>{"score", "label"}));
  EXPECT_DOUBLE_EQ(frame.double_at("score", 0), 1.5);
  EXPECT_EQ(frame.optional_double_at("score", 1), std::nullopt);
  EXPECT_EQ(frame.optional_string_at("label", 2), std::nullopt);
  EXPECT_TRUE(frame.is_null("score", 1));
  EXPECT_EQ(frame.null_count("label"), 1u);
}

TEST(DataFrame, RejectsLengthAndTypeMismatches) {
  DataFrame frame;
  frame.add_column("score", std::vector<double>{1.0, 2.0});

  EXPECT_THROW(frame.add_column("label", std::vector<std::string>{"x"}), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(frame.string_column("score")), std::invalid_argument);
  EXPECT_THROW(frame.append_row({{"score", std::string("oops")}}), std::invalid_argument);
}

TEST(DataFrame, SupportsAppendingRowsSettingValuesAndFillingNulls) {
  DataFrame frame{{{"score", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1.0, std::nullopt})},
                   {"label",
                    DataFrame::string_column_type(std::vector<std::optional<std::string>>{"x", std::nullopt})}}};

  frame.append_row({{"score", 4.0}, {"label", std::string("z")}});
  frame.set_value("score", 1, 9.0);
  frame.set_null("label", 0);
  frame.fill_null("label", "filled");

  EXPECT_EQ(frame.nrows(), 3u);
  EXPECT_DOUBLE_EQ(frame.double_at("score", 1), 9.0);
  EXPECT_EQ(frame.string_at("label", 0), "filled");

  const auto row = frame.row(2);
  EXPECT_DOUBLE_EQ(row.get_double("score"), 4.0);
  EXPECT_EQ(row.get_string("label"), "z");
}

TEST(DataFrame, SupportsProjectionFilteringSlicingAndDroppingNulls) {
  DataFrame frame{{{"score", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1.0, 4.0, std::nullopt, 5.0})},
                   {"label",
                    DataFrame::string_column_type(std::vector<std::optional<std::string>>{"a", "b", "c", std::nullopt})}}};

  const auto selected = frame.select({"label"});
  EXPECT_EQ(selected.ncols(), 1u);
  EXPECT_EQ(selected.columns(), (std::vector<std::string>{"label"}));
  EXPECT_EQ(selected.optional_string_at("label", 3), std::nullopt);

  const auto filtered = frame.filter([](const DataFrame::Row& row) {
    const auto score = row.get_optional_double("score");
    return score.has_value() && *score >= 4.0;
  });
  EXPECT_EQ(filtered.nrows(), 2u);
  EXPECT_EQ(filtered.string_at("label", 0), "b");
  EXPECT_TRUE(filtered.is_null("label", 1));

  const auto head = frame.head(2);
  EXPECT_EQ(head.optional_double_at("score", 1), std::optional<double>(4.0));

  const auto tail = frame.tail(2);
  EXPECT_EQ(tail.optional_string_at("label", 1), std::nullopt);

  const auto sliced = frame.slice(1, 3);
  EXPECT_EQ(sliced.nrows(), 2u);
  EXPECT_EQ(sliced.optional_double_at("score", 0), std::optional<double>(4.0));

  const auto taken = frame.take_rows({3, 0});
  EXPECT_EQ(taken.nrows(), 2u);
  EXPECT_EQ(taken.optional_double_at("score", 0), std::optional<double>(5.0));
  EXPECT_EQ(taken.optional_string_at("label", 1), std::optional<std::string>("a"));

  const auto dropped = frame.drop_nulls();
  EXPECT_EQ(dropped.nrows(), 2u);
  EXPECT_EQ(dropped.string_at("label", 1), "b");
}

TEST(DataFrame, SupportsSortingSummariesAndValueCounts) {
  DataFrame frame{{{"score", DataFrame::numeric_column_type(std::vector<std::optional<double>>{3.0, 1.0, std::nullopt, 2.0})},
                   {"label",
                    DataFrame::string_column_type(std::vector<std::optional<std::string>>{"c", "a", "c", std::nullopt})}}};

  const auto ascending = frame.sort_by("score");
  EXPECT_EQ(ascending.optional_double_at("score", 0), std::optional<double>(1.0));
  EXPECT_EQ(ascending.optional_double_at("score", 3), std::nullopt);

  const auto descending = frame.sort_by("label", false);
  EXPECT_EQ(descending.optional_string_at("label", 0), std::optional<std::string>("c"));
  EXPECT_EQ(descending.optional_string_at("label", 3), std::nullopt);

  const auto summary = frame.describe_numeric("score");
  EXPECT_EQ(summary.count, 3u);
  EXPECT_EQ(summary.null_count, 1u);
  EXPECT_DOUBLE_EQ(summary.sum, 6.0);
  EXPECT_DOUBLE_EQ(summary.mean, 2.0);
  EXPECT_DOUBLE_EQ(summary.min, 1.0);
  EXPECT_DOUBLE_EQ(summary.max, 3.0);

  const auto label_counts = frame.value_counts("label");
  EXPECT_EQ(label_counts.at("c"), 2u);
  EXPECT_EQ(label_counts.at("a"), 1u);

  const auto score_counts = frame.numeric_value_counts("score");
  EXPECT_EQ(score_counts.at(3.0), 1u);
  EXPECT_EQ(score_counts.at(1.0), 1u);
}

TEST(DataFrame, SupportsRenamingReplacingAndRemovingColumns) {
  DataFrame frame{{{"score", DataFrame::numeric_column_type(std::vector<double>{1.0, 2.0})},
                   {"label", DataFrame::string_column_type(std::vector<std::string>{"x", "y"})}}};

  frame.rename_column("label", "group");
  EXPECT_TRUE(frame.has_column("group"));
  EXPECT_FALSE(frame.has_column("label"));

  frame.replace_column("score", DataFrame::numeric_column_type(std::vector<std::optional<double>>{7.0, std::nullopt}));
  EXPECT_EQ(frame.optional_double_at("score", 0), std::optional<double>(7.0));
  EXPECT_EQ(frame.optional_double_at("score", 1), std::nullopt);

  EXPECT_TRUE(frame.remove_column("group"));
  EXPECT_EQ(frame.ncols(), 1u);
  EXPECT_FALSE(frame.remove_column("group"));
}

TEST(DataFrame, SupportsGroupByCountSumAndMean) {
  DataFrame frame{{{"group", DataFrame::string_column_type(std::vector<std::string>{"a", "a", "b", "b"})},
                   {"score", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1.0, 3.0, std::nullopt, 5.0})},
                   {"weight", DataFrame::numeric_column_type(std::vector<std::optional<double>>{10.0, 20.0, 30.0, 40.0})}}};

  const auto counts = frame.group_by_count({"group"});
  EXPECT_EQ(counts.nrows(), 2u);
  EXPECT_EQ(counts.string_at("group", 0), "a");
  EXPECT_DOUBLE_EQ(counts.double_at("count", 0), 2.0);

  const auto sums = frame.group_by_sum({"group"}, {"score", "weight"});
  EXPECT_DOUBLE_EQ(sums.double_at("score", 0), 4.0);
  EXPECT_DOUBLE_EQ(sums.double_at("weight", 1), 70.0);

  const auto means = frame.group_by_mean({"group"}, {"score"});
  EXPECT_DOUBLE_EQ(means.double_at("score", 0), 2.0);
  EXPECT_DOUBLE_EQ(means.double_at("score", 1), 5.0);
}

TEST(DataFrame, SupportsDropDuplicatesConcatAndPrinting) {
  DataFrame frame{{{"id", DataFrame::numeric_column_type(std::vector<double>{1.0, 1.0, 2.0, 2.0})},
                   {"group", DataFrame::string_column_type(std::vector<std::string>{"a", "a", "b", "b"})},
                   {"score", DataFrame::numeric_column_type(std::vector<std::optional<double>>{10.0, 10.0, 20.0, std::nullopt})}}};

  const auto distinct_all = frame.drop_duplicates();
  EXPECT_EQ(distinct_all.nrows(), 3u);

  const auto distinct_groups = frame.drop_duplicates({"group"});
  EXPECT_EQ(distinct_groups.nrows(), 2u);
  EXPECT_EQ(distinct_groups.string_at("group", 0), "a");
  EXPECT_EQ(distinct_groups.string_at("group", 1), "b");

  const auto first_half = frame.head(2);
  const auto second_half = frame.tail(2);
  const auto rebound = first_half.concat_rows(second_half);
  EXPECT_EQ(rebound.nrows(), 4u);
  EXPECT_TRUE(rebound.is_null("score", 3));

  const auto printed = frame.to_string(2);
  EXPECT_NE(printed.find("DataFrame[4 x 3]"), std::string::npos);
  EXPECT_NE(printed.find("| id | group | score |"), std::string::npos);
  EXPECT_NE(printed.find("| -- | ----- | ----- |"), std::string::npos);
  EXPECT_NE(printed.find("| 1  | a     | 10    |"), std::string::npos);
  EXPECT_NE(printed.find("... (2 more rows)"), std::string::npos);
}

TEST(DataFrame, SupportsInnerAndLeftJoin) {
  DataFrame left{{{"id", DataFrame::numeric_column_type(std::vector<double>{1.0, 2.0, 3.0})},
                  {"name", DataFrame::string_column_type(std::vector<std::string>{"alice", "bob", "cara"})}}};
  DataFrame right{{{"id", DataFrame::numeric_column_type(std::vector<double>{2.0, 3.0, 4.0})},
                   {"team", DataFrame::string_column_type(std::vector<std::string>{"blue", "red", "green"})}}};

  const auto inner = left.join(right, "id", "id");
  EXPECT_EQ(inner.nrows(), 2u);
  EXPECT_DOUBLE_EQ(inner.double_at("id", 0), 2.0);
  EXPECT_EQ(inner.string_at("team", 1), "red");

  const auto left_join = left.join(right, "id", "id", DataFrame::JoinType::Left);
  EXPECT_EQ(left_join.nrows(), 3u);
  EXPECT_TRUE(left_join.is_null("team", 0));
  EXPECT_EQ(left_join.string_at("team", 1), "blue");
}

#ifdef DATAMUNGE_HAVE_ARROW
TEST(DataFrame, SupportsArrowRoundTrip) {
  DataFrame frame{{{"score", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1.0, std::nullopt, 3.0})},
                   {"label",
                    DataFrame::string_column_type(std::vector<std::optional<std::string>>{"x", "y", std::nullopt})}}};

  const auto table = frame.to_arrow();
  ASSERT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);

  const auto restored = DataFrame::from_arrow(table);
  EXPECT_EQ(restored.nrows(), 3u);
  EXPECT_EQ(restored.optional_double_at("score", 1), std::nullopt);
  EXPECT_EQ(restored.optional_string_at("label", 2), std::nullopt);
  EXPECT_EQ(restored.string_at("label", 0), "x");
}
#endif
