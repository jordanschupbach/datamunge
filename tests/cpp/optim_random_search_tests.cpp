#include <datamunge/optim/random_search.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

/// @brief Records trials for a spherical objective.
struct RecordingObjective final : datamunge::optim::ArbitraryFunction {
  std::vector<std::vector<double>> points; ///< Calls in evaluation order.

  /// @brief Score a two-dimensional point and record it.
  /// @param x Trial point.
  /// @return Squared distance from the origin.
  double evaluate(const std::vector<double>& x) override {
    points.push_back(x);
    return x[0] * x[0] + x[1] * x[1];
  }
};

} // namespace

TEST(RandomSearch, ReplayAndBestObservedPoint) {
  const datamunge::optim::RandomSearch search({31, 7});
  RecordingObjective first;
  std::vector<double> point{30, 40};
  const auto value = search.optimize(first, point, {-2, 1}, {3, 1});
  ASSERT_TRUE(first.points.size() == 31);
  for (const auto& trial : first.points) {
    EXPECT_TRUE(trial[0] >= -2);
    EXPECT_TRUE(trial[0] <= 3);
    EXPECT_TRUE(trial[1] == 1);
    EXPECT_TRUE(value <= trial[0] * trial[0] + 1);
  }
  EXPECT_TRUE(value == point[0] * point[0] + 1);
  EXPECT_TRUE(std::ranges::find(first.points, point) != first.points.end());

  RecordingObjective second;
  std::vector<double> replay{0, 0};
  EXPECT_TRUE(search.optimize(second, replay, {-2, 1}, {3, 1}) == value);
  EXPECT_TRUE(second.points == first.points);
  EXPECT_TRUE(replay == point);
}

TEST(RandomSearch, ValidationAndDuplicates) {
  RecordingObjective objective;
  std::vector<double> point{4, 5};
  EXPECT_THROW(datamunge::optim::RandomSearch({0, 7}).optimize(
                      objective, point, {0, 0}, {1, 1}),
                  std::invalid_argument);
  EXPECT_THROW(datamunge::optim::RandomSearch({1, 7}).optimize(
                      objective, point, {0, 2}, {1, 1}),
                  std::invalid_argument);
  EXPECT_TRUE(objective.points.empty());
  EXPECT_EQ(point, (std::vector<double>{4, 5}));
  EXPECT_TRUE(datamunge::optim::RandomSearch({3, 7}).optimize(
            objective, point, {1, 2}, {1, 2}) == 5);
  EXPECT_TRUE(objective.points.size() == 3);
  EXPECT_EQ(point, (std::vector<double>{1, 2}));
}

TEST(RandomSearch, NonfiniteObjectivePreservesInput) {
  struct NonfiniteObjective final : datamunge::optim::ArbitraryFunction {
    double evaluate(const std::vector<double>&) override {
      return std::numeric_limits<double>::quiet_NaN();
    }
  } objective;
  std::vector<double> point{4};
  EXPECT_THROW(datamunge::optim::RandomSearch({1, 7}).optimize(
                      objective, point, {0}, {1}),
                  std::domain_error);
  EXPECT_EQ(point, (std::vector<double>{4}));
}

TEST(RandomSearch, LatinHypercubeUsesEachStratumOnEveryAxis) {
  using namespace datamunge::optim;
  RecordingObjective f;
  std::vector<double> x(2);
  RandomSearch({16, 7, RandomDesign::latin_hypercube}).optimize(f, x, {0, 0}, {1, 1});
  ASSERT_EQ(f.points.size(), 16);
  for (std::size_t axis = 0; axis < 2; ++axis) {
    std::vector<unsigned> counts(16);
    for (const auto& point : f.points) {
      const auto stratum = static_cast<std::size_t>(point[axis] * 16);
      ASSERT_LT(stratum, counts.size());
      ++counts[stratum];
    }
    EXPECT_EQ(counts, std::vector<unsigned>(16, 1));
  }
}

TEST(RandomSearch, SobolPreservesDyadicBalanceAndHaltonPreservesRadicalInverse) {
  using namespace datamunge::optim;
  RecordingObjective f;
  std::vector<double> x(2);
  RandomSearch({16, 7, RandomDesign::sobol}).optimize(f, x, {0, 0}, {1, 1});
  ASSERT_EQ(f.points.size(), 16);
  std::vector<unsigned> cells(16);
  for (const auto& point : f.points) {
    const auto row = static_cast<std::size_t>(point[0] * 4);
    const auto column = static_cast<std::size_t>(point[1] * 4);
    ASSERT_LT(row, 4);
    ASSERT_LT(column, 4);
    ++cells[row * 4 + column];
  }
  EXPECT_EQ(cells, std::vector<unsigned>(16, 1));

  f.points.clear();
  RandomSearch({3, 7, RandomDesign::halton}).optimize(f, x, {0, 0}, {1, 1});
  ASSERT_EQ(f.points.size(), 3);
  const auto difference = [&](std::size_t i, std::size_t axis) {
    const auto diff = f.points[i][axis] - f.points[0][axis];
    return diff < 0 ? 1 + diff : diff;
  };
  EXPECT_NEAR(difference(1, 0), 0.5, 1e-15);
  EXPECT_NEAR(difference(2, 0), 0.25, 1e-15);
  EXPECT_NEAR(difference(1, 1), 1.0 / 3, 1e-15);
  EXPECT_NEAR(difference(2, 1), 2.0 / 3, 1e-15);
}

TEST(SolisWets, OppositeTrialAndPartialPairConsumeExactBudget) {
  using namespace datamunge::optim;
  const std::vector<double> lower{-1, -1}, upper{1, 1};
  for (const std::size_t budget : {1, 2, 3, 25}) {
    RecordingObjective f;
    std::vector<double> x{0, 0};
    EXPECT_EQ(SolisWets({budget, 7, 0.01}).optimize(f, x, lower, upper), 0);
    ASSERT_EQ(f.points.size(), budget);
    EXPECT_EQ(x, (std::vector<double>{0, 0}));
    if (budget >= 3) {
      EXPECT_NE(f.points[1], f.points[0]);
      EXPECT_NEAR(f.points[1][0], -f.points[2][0], 1e-15);
      EXPECT_NEAR(f.points[1][1], -f.points[2][1], 1e-15);
    }
  }
}

TEST(SolisWets, ReplayAndBestObservedPoint) {
  using namespace datamunge::optim;
  const SolisWets search({100, 7, 0.2});
  RecordingObjective first, second;
  const std::vector<double> lower{-1, -1}, upper{1, 1};
  std::vector<double> point{4, 5}, replay = point;
  const auto value = search.optimize(first, point, lower, upper);
  ASSERT_EQ(first.points.size(), 100);
  EXPECT_EQ(first.points.front(), (std::vector<double>{1, 1}));
  EXPECT_EQ(value, point[0] * point[0] + point[1] * point[1]);
  EXPECT_LT(value, 2);
  EXPECT_EQ(search.optimize(second, replay, lower, upper), value);
  EXPECT_EQ(first.points, second.points);
  EXPECT_EQ(point, replay);
  for (const auto& trial : first.points) {
    EXPECT_LE(value, trial[0] * trial[0] + trial[1] * trial[1]);
  }
}
