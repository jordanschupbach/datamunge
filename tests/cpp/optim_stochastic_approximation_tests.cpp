#include <datamunge/optim/stochastic_approximation.hpp>

#include <gtest/gtest.h>

namespace o = datamunge::optim;

namespace {

/// @brief One-dimensional exact linear function with a fixed dummy axis.
struct Linear final : o::ArbitraryFunction {
  std::vector<std::vector<double>> points; ///< Trial history.

  /// @brief Record the trial and return the first coordinate.
  /// @param x Feasible point.
  /// @return Linear loss equal to the first coordinate.
  double evaluate(const std::vector<double>& x) override {
    points.push_back(x);
    return x[0];
  }
};

} // namespace

TEST(StochasticApproximation, PairedGradientAndTrailingCenterUseExactBudget) {
  for (const auto method : {o::StochasticApproximationMethod::spsa, o::StochasticApproximationMethod::rdsa}) {
    Linear f;
    o::StochasticApproximationOptions options{4, 42, method, 0.25, 0.1, 0, 0, 0};
    std::vector<double> x{0.5, 5};
    const auto value = o::StochasticApproximation(options).optimize(f, x, {0, 2}, {1, 2});
    ASSERT_EQ(f.points.size(), 4);
    EXPECT_NEAR(f.points[1][0] + f.points[2][0], 1, 1e-15);
    EXPECT_EQ(f.points.back(), (std::vector<double>{0.25, 2}));
    EXPECT_EQ(x, f.points.back());
    EXPECT_EQ(value, 0.25);
  }
}

TEST(StochasticApproximation, StartsWithTheClampedPointAndAllowsPartialBudget) {
  const o::StochasticApproximation search({1});
  Linear f;
  std::vector<double> x{-1, 0};
  EXPECT_EQ(search.optimize(f, x, {1, 2}, {2, 3}), 1);
  ASSERT_EQ(f.points.size(), 1);
  EXPECT_EQ(x, (std::vector<double>{1, 2}));
}
