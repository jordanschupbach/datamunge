/// @file vector_expression_tests.cpp
/// @brief Lazy vector math, recycling, lifetime, snapshot, range and API-boundary regressions.
#include <dm/dm.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <type_traits>

namespace {
/// @brief Scalar conversion counter to detect accidental eager element evaluation.
struct Counted {
  double value;
  int* reads;
  operator double() const { ++*reads; return value; }
};
/// @brief Scalar which can throw during a later evaluation pass.
struct Throwing {
  double value;
  bool* fail;
  operator double() const {
    if (*fail) throw std::runtime_error("evaluation failure");
    return value;
  }
};
}

TEST(VectorExpressions, FormationIsLazyAndSnapshotsDoNotReevaluate) {
  int reads = 0;
  dm::Vector<Counted> x{{1, &reads}, {2, &reads}, {3, &reads}};
  const auto expression = dm::sin((x + 1) * 2);
  static_assert(dm::vector_expr::is_node<decltype(expression)>);
  EXPECT_EQ(reads, 0);
  EXPECT_EQ(expression.size(), 3u);
  EXPECT_FALSE(expression.empty());
  EXPECT_EQ(reads, 0);
  const dm::Vector<> owned = expression;
  EXPECT_EQ(reads, 3);
  EXPECT_DOUBLE_EQ(owned[0], std::sin(4.0));
  EXPECT_EQ(reads, 3);
  const auto again = expression.eval();
  EXPECT_EQ(reads, 6);
  EXPECT_EQ(again.values(), owned.values());
}

TEST(VectorExpressions, BorrowsLiveLeavesAndTracksTheirCurrentLength) {
  auto x = dm::c(1, 2);
  const auto expression = x + 10;
  const auto snapshot = expression.eval();
  x[0] = 7;
  EXPECT_DOUBLE_EQ(expression[0], 17);
  EXPECT_DOUBLE_EQ(snapshot[0], 11);
  x = dm::c(3, 4, 5);
  EXPECT_EQ(expression.values(), (std::vector<double>{13, 14, 15}));
  x = dm::c();
  EXPECT_TRUE(expression.empty());
  EXPECT_TRUE(expression.eval().empty());
}

TEST(VectorExpressions, OwnsTemporaryLeavesAndCopiedExpressionChildren) {
  const auto expression = [] {
    auto inner = dm::c(1, 2) + 1;
    return dm::sin(inner * 0) + dm::c(3, 4);
  }();
  const auto copied = expression;
  EXPECT_EQ(copied.values(), (std::vector<double>{3, 4}));
  const auto const_temporary = []() -> const dm::Vector<> { return dm::c(2, 3); }() * 2;
  EXPECT_EQ(const_temporary.values(), (std::vector<double>{4, 6}));
  const auto doppler = [](auto&& x) {
    return dm::sin(20 / (std::forward<decltype(x)>(x) + 0.25));
  };
  EXPECT_DOUBLE_EQ(doppler(0.5), std::sin(20 / 0.75));
  const auto temporary_result = doppler(dm::c(0, 0.5));
  EXPECT_DOUBLE_EQ(temporary_result[1], std::sin(20 / 0.75));
}

TEST(VectorExpressions, RecyclingIsPreservedAtEachNodeAndForScalarBroadcasts) {
  const auto expression = (dm::c(1, 2) + dm::c(10, 20, 30)) + dm::c(100, 200, 300, 400, 500);
  EXPECT_EQ(expression.values(), (std::vector<double>{111, 222, 331, 411, 522}));
  EXPECT_EQ((10 - dm::c(1, 2)).values(), (std::vector<double>{9, 8}));
  EXPECT_EQ((12 / (dm::c(1, 2) * 2)).values(), (std::vector<double>{6, 3}));
  EXPECT_TRUE((dm::c() + 1).empty());
  EXPECT_TRUE((dm::c(1, 2) * dm::sin(dm::c())).eval().empty());
  EXPECT_EQ((dm::Vector<int>{1, 2} + dm::Vector<bool>{true, false}).values(),
            (std::vector<double>{2, 2}));
}

TEST(VectorExpressions, AliasSafeAssignmentSupportsRecyclingAndStrongExceptionGuarantee) {
  auto x = dm::c(1, 2);
  x = x + dm::c(10, 20, 30);
  EXPECT_EQ(x.values(), (std::vector<double>{11, 22, 31}));
  x += x * 2;
  EXPECT_EQ(x.values(), (std::vector<double>{33, 66, 93}));
  bool fail = false;
  dm::Vector<Throwing> source{{2, &fail}, {3, &fail}};
  const auto expression = source + 1;
  fail = true;
  EXPECT_THROW(x = expression, std::runtime_error);
  EXPECT_EQ(x.values(), (std::vector<double>{33, 66, 93}));
  dm::Vector<int> integral = (dm::c(1.5, 2.5) + 1);
  EXPECT_EQ(integral.values(), (std::vector<int>{2, 3}));
}

TEST(VectorExpressions, SupportsMathRangeFormattingAndOwnedStandardVectorBoundaries) {
  const auto expression = dm::round(dm::sqrt(dm::pow(dm::c(2, 3), 2)) + 0.125, 2);
  EXPECT_EQ(expression.values(), (std::vector<double>{2.12, 3.12}));
  EXPECT_EQ(std::format("{::.2f}", expression), "[2.12, 3.12]");
  std::ostringstream stream;
  stream << (dm::c(1, 2) + 1);
  EXPECT_EQ(stream.str(), "[2, 3]");
  const std::vector<double> direct = dm::abs(-dm::c(1, 2));
  EXPECT_EQ(direct, (std::vector<double>{1, 2}));
  dm::Vector inferred = dm::exp(dm::log(dm::c(1, 2)));
  EXPECT_NEAR(inferred[1], 2, 1e-14);
  const auto plot = dm::plot(dm::c(0, 1), dm::cos(dm::c(0, 0)));
  EXPECT_EQ(plot.series()[0].y, (std::vector<double>{1, 1}));
  const dm::LM model(dm::c(1, 3, 5, 7) + 0, dm::SpMat<>{{0}, {1}, {2}, {3}});
  EXPECT_NEAR(model.predict(dm::SpMat<>{{4}})[0], 9, 1e-10);
}

TEST(VectorExpressions, PreservesSelectionConcatenationAndEagerApplySemantics) {
  const auto expression = dm::c(1, 1, 2) + 1;
  EXPECT_EQ(expression.select({3, 1}).values(), (std::vector<double>{3, 2}));
  EXPECT_EQ(expression.select(dm::c(true, false)).values(), (std::vector<double>{2, 3}));
  EXPECT_EQ(expression.recycled(5).values(), (std::vector<double>{2, 2, 3, 2, 2}));
  EXPECT_EQ(dm::unique(expression).values(), (std::vector<double>{2, 3}));
  EXPECT_EQ(dm::c(expression).values(), expression.values());
  int calls = 0;
  const auto mapped = dm::apply(expression, [&](double value) { ++calls; return value * value; });
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(mapped.values(), (std::vector<double>{4, 4, 9}));
  auto base = dm::c(1, 2);
  const auto frozen_input = dm::apply(base + 1, [&](double value) {
    base = base + 100;
    return value;
  });
  EXPECT_EQ(frozen_input.values(), (std::vector<double>{2, 3}));
  EXPECT_EQ(expression.select(dm::c(0, 2) + 1).values(), (std::vector<double>{2, 3}));
}

TEST(VectorExpressions, ReductionsStreamValuesAndRetainNonfiniteBehavior) {
  const auto expression = dm::c(1, 2, 3) * 2;
  EXPECT_DOUBLE_EQ(dm::sum(expression), 12);
  EXPECT_DOUBLE_EQ(dm::mean(expression), 4);
  EXPECT_DOUBLE_EQ(dm::sd(expression), 2);
  EXPECT_DOUBLE_EQ(dm::sum(dm::c() + 1), 0);
  EXPECT_TRUE(std::isnan(dm::mean(dm::c() + 1)));
  EXPECT_TRUE(std::isnan(dm::sd(dm::c(1) + 1)));
  EXPECT_TRUE(std::isnan(dm::sd(dm::c(std::numeric_limits<double>::infinity()) + 1)));
  EXPECT_TRUE(std::isinf((dm::c(1) / 0)[0]));
  EXPECT_TRUE(std::isnan(dm::log(dm::c(-1))[0]));
}

TEST(VectorExpressions, RandomSamplesAreOwnedAndNeverRedrawnDuringEvaluation) {
  dm::set_seed(123);
  const auto x = dm::c(0, 0, 0);
  const auto expression = x + dm::rnorm(0, 1, 3);
  const auto first = expression.eval();
  const auto second = expression.eval();
  const auto next = dm::rnorm(0, 1, 3);
  dm::set_seed(123);
  EXPECT_EQ(first.values(), dm::rnorm(0, 1, 3).values());
  EXPECT_EQ(second.values(), first.values());
  EXPECT_EQ(next.values(), dm::rnorm(0, 1, 3).values());
  static_assert(sizeof(dm::Vector<double>) == sizeof(std::vector<double>));
  static_assert(alignof(dm::Vector<double>) == alignof(std::vector<double>));
  static_assert(std::is_standard_layout_v<dm::Vector<double>>);
}
