#include <gtest/gtest.h>

#include <datamunge/pipe.hpp>

#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace dm = datamunge;
using datamunge::dstruct::DataFrame;

namespace {

DataFrame sample_frame() {
  return DataFrame{{{"len", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1, 3, 10, 20, 5, 7})},
                    {"supp", DataFrame::string_column_type(std::vector<std::optional<std::string>>{"VC", "VC", "OJ", "OJ", "VC", "VC"})},
                    {"dose", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1, 1, 1, 1, 2, 2})}}};
}

double mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

} // namespace

TEST(Pipe, GroupBySummariseEach) {
  const auto result = sample_frame() | dm::group_by("supp", "dose") | dm::summarise_each(mean);
  ASSERT_EQ(result.nrows(), 3u);
  EXPECT_EQ(result.columns(), (std::vector<std::string>{"supp", "dose", "len"}));
  EXPECT_EQ(result.string_at("supp", 0), "VC");
  EXPECT_EQ(result.optional_double_at("dose", 0), std::optional<double>(1.0));
  EXPECT_EQ(result.optional_double_at("len", 0), std::optional<double>(2.0));
  EXPECT_EQ(result.string_at("supp", 1), "OJ");
  EXPECT_EQ(result.optional_double_at("len", 1), std::optional<double>(15.0));
  EXPECT_EQ(result.optional_double_at("len", 2), std::optional<double>(6.0));
}

TEST(Pipe, SummariseEachSummarisesEveryNonKeyNumericColumn) {
  const auto result =
      sample_frame() | dm::group_by("supp") | dm::summarise_each([](const std::vector<double>& v) { return v.size(); });
  EXPECT_EQ(result.columns(), (std::vector<std::string>{"supp", "len", "dose"}));
  EXPECT_EQ(result.optional_double_at("dose", 0), std::optional<double>(4.0));
}

TEST(Pipe, NullInGroupGivesNull) {
  DataFrame frame{{{"g", DataFrame::string_column_type(std::vector<std::optional<std::string>>{"a", "a", "b"})},
                   {"x", DataFrame::numeric_column_type(std::vector<std::optional<double>>{1, std::nullopt, 4})}}};
  const auto result = frame | dm::group_by("g") | dm::summarise_each(mean);
  EXPECT_TRUE(result.is_null("x", 0));
  EXPECT_EQ(result.optional_double_at("x", 1), std::optional<double>(4.0));
}

TEST(Pipe, CountAndCallableSteps) {
  const auto counts = sample_frame() | dm::group_by(std::vector<std::string>{"supp"}) | dm::count();
  EXPECT_EQ(counts.optional_double_at("n", 0), std::optional<double>(4.0));

  const auto rows = sample_frame() | [](const DataFrame& df) { return dm::head(df, 2); } |
                    [](const DataFrame& df) { return df.nrows(); };
  EXPECT_EQ(rows, 2u);
}

TEST(Pipe, UnknownKeyThrows) {
  EXPECT_THROW((void)(sample_frame() | dm::group_by("nope")), std::out_of_range);
}

TEST(Pipe, FilterRowsKeepsMatchingRows) {
  const auto result = sample_frame() | dm::filter_rows([](const DataFrame::Row& row) { return row.get_double("len") > 4; });
  EXPECT_EQ(result.nrows(), 4u);
  EXPECT_EQ(result.optional_double_at("len", 0), std::optional<double>(10.0));
}

TEST(Pipe, MutateFromRowsAndVectors) {
  const auto doubled = sample_frame() | dm::mutate("len2", [](const auto& row) { return row.get_double("len") * 2; });
  EXPECT_EQ(doubled.columns().back(), "len2");
  EXPECT_EQ(doubled.optional_double_at("len2", 1), std::optional<double>(6.0));

  // Replacing an existing column keeps its position.
  const auto replaced = sample_frame() | dm::mutate("len", [](const auto& row) { return row.get_double("len") + 1; });
  EXPECT_EQ(replaced.columns(), sample_frame().columns());
  EXPECT_EQ(replaced.optional_double_at("len", 0), std::optional<double>(2.0));

  const auto labelled = sample_frame() | dm::mutate("tag", [](const auto& row) {
                          return row.get_string("supp") + "-" + std::to_string(static_cast<int>(row.get_double("dose")));
                        });
  EXPECT_EQ(labelled.string_at("tag", 4), "VC-2");

  const auto with_nulls = sample_frame() | dm::mutate("big", [](const auto& row) -> std::optional<double> {
                            if (row.get_double("len") < 10) return std::nullopt;
                            return row.get_double("len");
                          });
  EXPECT_TRUE(with_nulls.is_null("big", 0));
  EXPECT_EQ(with_nulls.optional_double_at("big", 3), std::optional<double>(20.0));

  const auto from_vector = sample_frame() | dm::mutate("id", std::vector<double>{1, 2, 3, 4, 5, 6});
  EXPECT_EQ(from_vector.optional_double_at("id", 5), std::optional<double>(6.0));
}

TEST(Pipe, ArrangeWithDescKeys) {
  const auto result = sample_frame() | dm::arrange("supp", dm::desc("len"));
  ASSERT_EQ(result.nrows(), 6u);
  EXPECT_EQ(result.string_at("supp", 0), "OJ");
  EXPECT_EQ(result.optional_double_at("len", 0), std::optional<double>(20.0));
  EXPECT_EQ(result.optional_double_at("len", 1), std::optional<double>(10.0));
  EXPECT_EQ(result.string_at("supp", 2), "VC");
  EXPECT_EQ(result.optional_double_at("len", 2), std::optional<double>(7.0));
  EXPECT_EQ(result.optional_double_at("len", 5), std::optional<double>(1.0));
}

TEST(Pipe, StepsChain) {
  const auto result = sample_frame() | dm::filter_rows([](const auto& row) { return row.get_string("supp") == "VC"; }) |
                      dm::mutate("len2", [](const auto& row) { return row.get_double("len") * 2; }) |
                      dm::arrange(dm::desc("len2")) | dm::group_by("dose") | dm::summarise_each(mean);
  ASSERT_EQ(result.nrows(), 2u);
  EXPECT_EQ(result.optional_double_at("dose", 0), std::optional<double>(2.0));
  EXPECT_EQ(result.optional_double_at("len2", 0), std::optional<double>(12.0));
  EXPECT_EQ(result.optional_double_at("len2", 1), std::optional<double>(4.0));
}

TEST(Pipe, GenericLambdaStepsCompileForFramesAndGroups) {
  // A `const auto&` lambda used to be a hard error: checking it against the grouped overload
  // instantiated its body with a GroupedDataFrame.
  const auto top = sample_frame() | [](const auto& df) { return dm::head(df, 2); };
  EXPECT_EQ(top.nrows(), 2u);
  const auto groups = sample_frame() | dm::group_by("supp") | [](const auto& g) { return g.count().nrows(); };
  EXPECT_EQ(groups, 2u);
}

// --- to_md() and the summarise/select/rename/pivot/slice/replace_na verbs ----

TEST(DataFrameToMd, AlignsColumnsAndPicksDecimalsPerColumn) {
  DataFrame frame;
  frame.add_column("name", std::vector<std::string>{"a|b", "c"});
  frame.add_column("count", std::vector<double>{50, 3});
  frame.add_column("tenths", std::vector<double>{5.1, 3.0});
  frame.add_column("mean", std::vector<double>{1.23456, 2.0});
  frame.add_column("missing", std::vector<std::optional<double>>{std::nullopt, -0.0001});
  EXPECT_EQ(frame.to_md(),
            "| name | count | tenths | mean | missing |\n"
            "| :--- | ---: | ---: | ---: | ---: |\n"
            "| a\\|b | 50 | 5.1 | 1.235 | NA |\n"
            "| c | 3 | 3.0 | 2.000 | 0.000 |\n");
}

TEST(DataFrameToMd, OptionsControlDigitsRowsAndNulls) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.23456, std::nullopt, 3});
  frame.add_column("p", std::vector<double>{0.000123, 0.5, 1});
  DataFrame::MarkdownOptions options;
  options.digits = 1;
  options.column_digits = {{"p", 4}};
  options.max_rows = 2;
  options.na = "-";
  EXPECT_EQ(frame.to_md(options),
            "| x | p |\n"
            "| ---: | ---: |\n"
            "| 1.2 | 0.0001 |\n"
            "| - | 0.5000 |\n"
            "... (1 more rows)\n");
  options.digits = -1;
  EXPECT_THROW((void)frame.to_md(options), std::invalid_argument);
}

TEST(DataFrameToMd, ScientificColumns) {
  DataFrame frame;
  frame.add_column("eigenvalue", std::vector<double>{1.80533e-8, 0.25, -0.0});
  frame.add_column("k", std::vector<double>{5, 10, 12});
  DataFrame::MarkdownOptions options;
  options.digits = 2;
  options.scientific = {"eigenvalue"};
  EXPECT_EQ(frame.to_md(options),
            "| eigenvalue | k |\n"
            "| ---: | ---: |\n"
            "| 1.81e-08 | 5 |\n"
            "| 2.50e-01 | 10 |\n"
            "| 0.00e+00 | 12 |\n");
}

TEST(PipeVerbs, SummariseGroupedAndUngrouped) {
  namespace agg = dm::agg;
  const auto frame = sample_frame();
  const auto grouped = frame | dm::group_by("supp")
                     | dm::summarise(agg::n(), agg::mean("len", "mean_len"), agg::max("dose"));
  EXPECT_EQ(grouped.columns(), (std::vector<std::string>{"supp", "n", "mean_len", "dose"}));
  EXPECT_DOUBLE_EQ(grouped.double_at("n", 0), 4);
  EXPECT_DOUBLE_EQ(grouped.double_at("mean_len", 0), 4);
  EXPECT_DOUBLE_EQ(grouped.double_at("mean_len", 1), 15);

  const auto overall = frame | dm::summarise(agg::n("rows"), agg::sum("len"));
  ASSERT_EQ(overall.nrows(), 1U);
  EXPECT_EQ(overall.columns(), (std::vector<std::string>{"rows", "len"}));
  EXPECT_DOUBLE_EQ(overall.double_at("len", 0), 46);
}

TEST(PipeVerbs, AcrossNamesOutputsFromAPattern) {
  namespace agg = dm::agg;
  const auto specs = agg::across({"len", "dose"}, agg::Func::StdDev, "{col}_{fn}");
  ASSERT_EQ(specs.size(), 2U);
  EXPECT_EQ(specs[0].result_name, "len_sd");
  EXPECT_EQ(specs[1].result_name, "dose_sd");
  const auto out = sample_frame() | dm::group_by("supp") | dm::summarise(agg::across({"len", "dose"}, agg::Func::Mean));
  EXPECT_EQ(out.columns(), (std::vector<std::string>{"supp", "len", "dose"}));
}

TEST(PipeVerbs, SelectRenameSliceHeadAndToMd) {
  const auto text = sample_frame() | dm::select("supp", "len") | dm::rename("len", "length") | dm::slice_head(2)
                  | dm::to_md();
  EXPECT_EQ(text, "| supp | length |\n| :--- | ---: |\n| VC | 1 |\n| VC | 3 |\n");
}

TEST(PipeVerbs, PivotLongerThenWiderWithFill) {
  DataFrame frame;
  frame.add_column("truth", std::vector<std::string>{"a", "a", "b"});
  frame.add_column("pred", std::vector<std::string>{"a", "b", "b"});
  const auto table = frame | dm::group_by("truth", "pred") | dm::count() | dm::pivot_wider("pred", "n", 0.0);
  EXPECT_EQ(table.columns(), (std::vector<std::string>{"truth", "a", "b"}));
  EXPECT_DOUBLE_EQ(table.double_at("a", 1), 0);  // b was never predicted as a
  EXPECT_DOUBLE_EQ(table.double_at("b", 1), 1);

  const auto unfilled = frame | dm::group_by("truth", "pred") | dm::count() | dm::pivot_wider("pred", "n");
  EXPECT_TRUE(unfilled.is_null("a", 1));
  EXPECT_FALSE((unfilled | dm::replace_na(-1)).is_null("a", 1));

  const auto longer = sample_frame() | dm::pivot_longer({"len", "dose"}, "what", "value");
  EXPECT_EQ(longer.nrows(), 12U);
  EXPECT_EQ(longer.columns(), (std::vector<std::string>{"supp", "what", "value"}));
}
