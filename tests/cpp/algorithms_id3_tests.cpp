#include <gtest/gtest.h>

#include <datamunge/algorithms/id3_diagnostics.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <limits>
namespace a = datamunge::algorithms;
using DF    = datamunge::dstruct::DataFrame;
TEST(ID3Diagnostics, ExactFourRowSplitAndBranchAccounting) {
  const std::vector<std::vector<int>> x{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const std::vector<int>              y{0, 0, 1, 0};
  const auto                          s = a::id3_split_table(x, y);
  const double                        h = -.75 * std::log2(.75) - .25 * std::log2(.25);
  EXPECT_NEAR(s.double_at("entropy", 0), h, 1e-15);
  EXPECT_NEAR(s.double_at("conditional_entropy", 0), .5, 1e-15);
  EXPECT_NEAR(s.double_at("gain", 0), h - .5, 1e-15);
  EXPECT_EQ(s.double_at("gain", 0), s.double_at("gain", 1));
  const auto b = a::id3_branch_table(x, y, 0);
  EXPECT_EQ(b.double_at("rows", 0), 2);
  EXPECT_EQ(b.double_at("weighted_entropy", 1), .5);
  const auto t = a::id3_train(x, y);
  EXPECT_EQ(t.nodes[0].feature, 0);
  EXPECT_EQ(t.nodes.size(), 5);
  EXPECT_EQ(a::id3_evaluate(t, x, y).double_at("correct", 0), 4);
}
TEST(ID3Diagnostics, TennisKnownGainsTreeAndDepth) {
  const auto                d = a::id3_tennis();
  const auto                s = a::id3_split_table(d.features, d.labels, d.feature_names);
  const std::vector<double> gain{
      .2467498197744391, .029222565658954647, .15183550136234159, .04812703040826949};
  for (std::size_t j = 0; j < 4; ++j)
    EXPECT_NEAR(s.double_at("gain", j), gain[j], 1e-14);
  const auto t = a::id3_train(d.features, d.labels);
  const auto n = a::id3_node_table(t);
  EXPECT_EQ(n.nrows(), 8);
  EXPECT_EQ(a::id3_evaluate(t, d.features, d.labels).double_at("correct", 0), 14);
  EXPECT_EQ(
      a::id3_evaluate(a::id3_train(d.features, d.labels, 1), d.features, d.labels)
          .double_at("correct", 0),
      10);
  EXPECT_EQ(a::id3_train(d.features, d.labels, 2).nodes.size(), 8);
}
TEST(ID3Diagnostics, UnseenFallsBackLocallyAndBaselineIsFitted) {
  const auto d = a::id3_tennis();
  const auto t = a::id3_train(d.features, d.labels);
  EXPECT_EQ(a::id3_predict(t, {99, 0, 0, 0}), 1); // root majority yes
  EXPECT_EQ(a::id3_predict(t, {0, 0, 99, 0}), 0); // sunny majority no
  const auto trace = a::id3_trace(t, {0, 0, 99, 0});
  EXPECT_EQ(trace.nrows(), 2);
  EXPECT_EQ(trace.double_at("unseen", 1), 1);
  EXPECT_EQ(trace.double_at("prediction", 1), 0);
  const auto e = a::id3_evaluate(t, {{0, 0, 99, 0}, {99, 0, 0, 0}}, {1, 1});
  EXPECT_EQ(e.double_at("fallbacks", 0), 2);
  EXPECT_EQ(e.double_at("baseline_accuracy", 0), 1);
  EXPECT_EQ(e.double_at("accuracy", 0), .5);
}
TEST(ID3Diagnostics, XorStopsAtZeroGain) {
  const std::vector<std::vector<int>> x{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const std::vector<int>              y{0, 1, 1, 0};
  const auto                          t = a::id3_train(x, y);
  EXPECT_EQ(t.nodes.size(), 1);
  EXPECT_EQ(t.nodes[0].prediction, 0);
  EXPECT_EQ(a::id3_evaluate(t, x, y).double_at("accuracy", 0), .5);
  EXPECT_DOUBLE_EQ(a::id3_split_table(x, y).double_at("gain", 0), 0);
}
TEST(ID3Diagnostics, IdentifierOverfitsUnseenCategories) {
  const std::vector<std::vector<int>> x{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 4}, {1, 5}};
  const std::vector<int>              y{0, 0, 0, 1, 1, 1};
  const auto                          t = a::id3_train(x, y);
  EXPECT_EQ(t.nodes[0].feature, 1);
  EXPECT_EQ(t.nodes.size(), 7);
  EXPECT_EQ(a::id3_predict(t, {1, 90}), 0);
  EXPECT_EQ(a::id3_evaluate(t, x, y).double_at("accuracy", 0), 1);
}
TEST(ID3Diagnostics, ArbitraryLabelsTiesConstantAndZeroWidth) {
  const auto t = a::id3_train({{}, {}}, {-5, 9});
  EXPECT_EQ(a::id3_predict(t, {}), -5);
  const auto c = a::id3_train({{7}, {7}, {7}}, {9, -5, 9});
  EXPECT_EQ(c.nodes.size(), 1);
  EXPECT_EQ(a::id3_predict(c, {}), 9);
  EXPECT_TRUE(a::id3_train({}, {}).nodes.empty());
  EXPECT_EQ(a::id3_node_table({}).nrows(), 0);
  EXPECT_EQ(a::id3_split_table({}, {}).nrows(), 0);
}
TEST(ID3Diagnostics, ShapeAndEmptyErrors) {
  EXPECT_THROW(a::id3_train({{0}}, {}), std::invalid_argument);
  EXPECT_THROW(a::id3_train({{0}, {}}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(a::id3_predict({}, {}), std::invalid_argument);
  const auto t = a::id3_train({{0}, {1}}, {0, 1});
  EXPECT_THROW(a::id3_predict(t, {}), std::invalid_argument);
  EXPECT_THROW(a::id3_split_table({{0}}, {0}, {"a", "b"}), std::invalid_argument);
  EXPECT_THROW(a::id3_branch_table({}, {}, 0), std::invalid_argument);
  EXPECT_THROW(a::id3_branch_table({{0}}, {0}, 1), std::invalid_argument);
  EXPECT_THROW(a::id3_evaluate(t, {}, {}), std::invalid_argument);
  EXPECT_THROW(a::id3_evaluate(t, {{0}, {}}, {0, 1}), std::invalid_argument);
}
TEST(ID3Diagnostics, MalformedVisitedPathsAreRejected) {
  auto t                 = a::id3_train({{0}, {1}}, {0, 1});
  t.nodes[0].children[0] = 99;
  EXPECT_THROW(a::id3_predict(t, {0}), std::invalid_argument);
  EXPECT_THROW(a::id3_trace(t, {0}), std::invalid_argument);
  EXPECT_THROW(a::id3_node_table(t), std::invalid_argument);
  t.nodes[0].children[0] = 0;
  EXPECT_THROW(a::id3_predict(t, {0}), std::invalid_argument);
  EXPECT_THROW(a::id3_node_table(t), std::invalid_argument);
  t = a::id3_train({{0}, {1}}, {0, 1});
  t.nodes.push_back({});
  EXPECT_THROW(a::id3_node_table(t), std::invalid_argument);
  t                      = a::id3_train({{0}, {1}}, {0, 1});
  t.nodes[0].children[1] = t.nodes[0].children[0];
  EXPECT_THROW(a::id3_node_table(t), std::invalid_argument);
  t = {};
  t.nodes.push_back({});
  t.nodes[0].children[1] = 0;
  EXPECT_THROW(a::id3_node_table(t), std::invalid_argument);
}
TEST(ID3Diagnostics, PlotTopologyAndNameGuards) {
  const auto d = a::id3_tennis();
  const auto t = a::id3_train(d.features, d.labels);
  EXPECT_NO_THROW(a::plot_id3_tree(t, d.feature_names));
  EXPECT_THROW(a::plot_id3_tree({}), std::invalid_argument);
  EXPECT_THROW(a::plot_id3_tree(t, {"one"}), std::invalid_argument);
  a::DecisionTree wide;
  wide.nodes.resize(33);
  wide.nodes[0].leaf = false;
  for (int i = 1; i < 33; ++i)
    wide.nodes[0].children[i] = i;
  EXPECT_THROW(a::plot_id3_tree(wide), std::invalid_argument);
}
TEST(ID3Frame, FrozenLevelsNumericCategoriesAndMissingQueries) {
  DF f;
  f.add_column("s", std::vector<std::string>{"a", "a", "b", "b"});
  f.add_column("p", std::vector<double>{100., 100., -2., -2.});
  f.add_column("y", std::vector<double>{3., 3., 7., 7.});
  const a::ID3FrameClassifier m(f, {"s", "p"}, "y");
  EXPECT_EQ(m.predict(f), (std::vector<int>{3, 3, 7, 7}));
  auto q = f.head(1).mutate("s", std::vector<std::string>{"new"});
  EXPECT_EQ(m.encode(q)[0][0], -1);
  EXPECT_EQ(m.predict(q)[0], 3);
  EXPECT_EQ(m.evaluate(q).double_at("fallbacks", 0), 1);
  EXPECT_EQ(m.encode(q)[0][0], -1);
  EXPECT_EQ(m.encode(f)[2][0], 1); // inference never changes codes
  EXPECT_TRUE(m.predict(f.head(0)).empty());
  EXPECT_EQ(m.encode(f.select({"y", "p", "s"})), m.encode(f));
}
TEST(ID3Frame, SchemaAndCellGuards) {
  DF f;
  f.add_column("s", std::vector<std::string>{"a", "b"});
  f.add_column("y", std::vector<double>{0., 1.});
  const a::ID3FrameClassifier m(f, {"s"}, "y");
  EXPECT_THROW((a::ID3FrameClassifier(f, {}, "y")), std::invalid_argument);
  EXPECT_THROW((a::ID3FrameClassifier(f, {"s", "s"}, "y")), std::invalid_argument);
  EXPECT_THROW((a::ID3FrameClassifier(f, {"y"}, "y")), std::invalid_argument);
  EXPECT_THROW((a::ID3FrameClassifier(f, {"missing"}, "y")), std::invalid_argument);
  EXPECT_THROW((a::ID3FrameClassifier(f.head(0), {"s"}, "y")), std::invalid_argument);
  EXPECT_THROW(m.encode(f.select({"y"})), std::invalid_argument);
  EXPECT_THROW(m.encode(f.mutate("s", std::vector<double>{0, 1})), std::invalid_argument);
  EXPECT_THROW(m.evaluate(f.head(0)), std::invalid_argument);
  auto bad = f;
  bad.set_value("s", 0, std::optional<std::string>{});
  EXPECT_THROW(m.predict(bad), std::invalid_argument);
  EXPECT_THROW((a::ID3FrameClassifier(bad, {"s"}, "y")), std::invalid_argument);
  for (double v :
       {.5,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        2147483648.}) {
    bad = f;
    bad.set_value("y", 0, v);
    EXPECT_THROW(m.evaluate(bad), std::invalid_argument);
  }
  bad = f;
  bad.set_value("y", 0, std::optional<double>{});
  EXPECT_THROW(m.evaluate(bad), std::invalid_argument);
  DF numeric;
  numeric.add_column("p", std::vector<double>{0., std::numeric_limits<double>::infinity()});
  numeric.add_column("y", std::vector<double>{0., 1.});
  EXPECT_THROW((a::ID3FrameClassifier(numeric, {"p"}, "y")), std::invalid_argument);
}
TEST(ID3Frame, TitanicDeclaredSplit) {
  const auto f =
      datamunge::datasets::titanic().select({"Sex", "Pclass", "Embarked", "Survived"}).drop_nulls();
  EXPECT_EQ(f.nrows(), 889);
  std::vector<std::size_t> fit, test;
  for (std::size_t i = 0; i < f.nrows(); ++i)
    (i % 5 == 0 ? test : fit).push_back(i);
  const a::ID3FrameClassifier m(f.take_rows(fit), {"Sex", "Pclass", "Embarked"}, "Survived", 2);
  const auto                  e = m.evaluate(f.take_rows(test));
  EXPECT_EQ(e.double_at("rows", 0), 178);
  EXPECT_EQ(e.double_at("fallbacks", 0), 0);
  EXPECT_GT(e.double_at("accuracy", 0), .7);
  EXPECT_LT(e.double_at("accuracy", 0), .95);
}
