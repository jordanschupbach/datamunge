#include <datamunge/optim/stochastic_local_search.hpp>

#include <gtest/gtest.h>

#include <limits>

namespace o = datamunge::optim;

namespace {

/// @brief Sphere in one variable with a trial history.
struct Sphere final : o::ArbitraryFunction {
  std::vector<double> points; ///< Recorded callback coordinates.

  /// @brief Record an in-box coordinate and return its squared distance from zero.
  /// @param x Candidate coordinates.
  /// @return Squared distance from zero.
  double evaluate(const std::vector<double>& x) override {
    points.push_back(x.at(0));
    return x[0] * x[0];
  }
};

/// @brief Check that a native optimizer returns an evaluated incumbent using exactly the budget.
template <class Optimizer>
void check_incumbent(const Optimizer& optimizer, std::size_t budget) {
  Sphere first, second;
  std::vector<double> x{10}, replay = x;
  const auto value = optimizer.optimize(first, x, {-1}, {1});
  EXPECT_EQ(first.points.size(), budget);
  ASSERT_FALSE(first.points.empty());
  EXPECT_EQ(first.points.front(), 1);
  EXPECT_EQ(value, x[0] * x[0]);
  EXPECT_EQ(optimizer.optimize(second, replay, {-1}, {1}), value);
  EXPECT_EQ(first.points, second.points);
  EXPECT_EQ(x, replay);
  for (const auto point : first.points) {
    EXPECT_LE(value, point * point);
    EXPECT_GE(point, -1);
    EXPECT_LE(point, 1);
  }
}

} // namespace

TEST(StochasticLocalSearch, EveryRuleReplaysAndReturnsItsBestAtExactBudget) {
  for (const auto method : {o::StochasticLocalMethod::hill_climbing, o::StochasticLocalMethod::luus_jaakola,
                            o::StochasticLocalMethod::one_plus_one}) {
    check_incumbent(o::StochasticLocalSearch({1, 7, method}), 1);
    check_incumbent(o::StochasticLocalSearch({65, 7, method}), 65);
  }
}

TEST(HybridRandomSearch, CountsNestedSearchAndTruncatesPartialRuns) {
  for (const auto method : {o::HybridSearchMethod::multistart, o::HybridSearchMethod::basin_hopping,
                            o::HybridSearchMethod::mlsl}) {
    for (const auto budget : {1u, 2u, 53u}) {
      o::HybridSearchOptions options;
      options.evaluations = budget;
      options.method = method;
      options.local_evaluations = 7;
      options.sample_batch = 3;
      check_incumbent(o::HybridRandomSearch(options), budget);
    }
  }
}

TEST(StochasticLocalSearch, ValidatesActiveOptionsAndDoesNotPublishPartialResult) {
  Sphere f;
  std::vector<double> x{1};
  o::StochasticLocalOptions invalid;
  invalid.initial_scale = 0;
  EXPECT_THROW(o::StochasticLocalSearch(invalid).optimize(f, x, {-1}, {1}), std::invalid_argument);
  EXPECT_TRUE(f.points.empty());

  struct Fails final : o::ArbitraryFunction {
    unsigned calls = 0;
    double evaluate(const std::vector<double>&) override {
      return calls++ == 0 ? 0.0 : std::numeric_limits<double>::quiet_NaN();
    }
  } fails;
  EXPECT_THROW(o::HybridRandomSearch({2}).optimize(fails, x, {2}, {3}), std::domain_error);
  EXPECT_EQ(x, std::vector<double>{1});
}
