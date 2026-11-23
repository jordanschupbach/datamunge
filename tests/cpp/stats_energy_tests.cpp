#include <datamunge/stats/energy_test.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {

using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
using datamunge::random::SplitMix64;

TEST(EnergyDistance, MatchesDirectDistanceMeansAndExactLabelsInR) {
  const Matrix first(3, 2, {0, 1, 1, 2, 2, 1});
  const Matrix second(4, 2, {1, 2, 3, 4, 4, 1, 2, 4});
  const auto exact = energy_distance_test_exact(first, second, 35);
  EXPECT_EQ(exact.alternative, Alternative::Greater);
  EXPECT_NEAR(exact.statistic, 3.609090115252068, 1e-14);
  EXPECT_NEAR(exact.estimate1, 2.1053025672303729, 1e-14);
  EXPECT_DOUBLE_EQ(exact.parameter1, 35);
  EXPECT_DOUBLE_EQ(exact.parameter2, 8);
  EXPECT_DOUBLE_EQ(exact.p_value, 8. / 35);
  EXPECT_DOUBLE_EQ(energy_distance_test_exact(second, first).statistic, exact.statistic);
  const Matrix reordered(4, 2, {4, 1, 3, 4, 1, 2, 2, 4});
  EXPECT_DOUBLE_EQ(energy_distance_test_exact(first, reordered).p_value, exact.p_value);
  EXPECT_THROW(energy_distance_test_exact(first, second, 34), std::length_error);
}

TEST(EnergyDistance, SimulatedLabelsAreReproducibleAndUseThePlusOneCorrection) {
  const Matrix first(3, 2, {0, 1, 1, 2, 2, 1});
  const Matrix second(4, 2, {1, 2, 3, 4, 4, 1, 2, 4});
  SplitMix64 one(21), two(21);
  const auto simulated = energy_distance_test(first, second, 1999, one);
  const auto repeated = energy_distance_test(first, second, 1999, two);
  EXPECT_DOUBLE_EQ(simulated.parameter1, 1999);
  EXPECT_DOUBLE_EQ(simulated.parameter2, repeated.parameter2);
  EXPECT_DOUBLE_EQ(simulated.p_value, (1 + simulated.parameter2) / 2000);
  EXPECT_NEAR(simulated.p_value, 8. / 35, .035);
}

TEST(EnergyDistance, IncludesTiesAndRejectsInvalidOrUnrepresentableInputs) {
  const Matrix first(2, 1, {0, 1}), second(2, 1, {4, 5});
  const auto exact = energy_distance_test_exact(first, second);
  EXPECT_DOUBLE_EQ(exact.statistic, 7);
  EXPECT_DOUBLE_EQ(exact.parameter2, 2);
  EXPECT_DOUBLE_EQ(exact.p_value, 1. / 3);
  const auto identical = energy_distance_test_exact(first, first);
  EXPECT_DOUBLE_EQ(identical.statistic, 0);
  EXPECT_DOUBLE_EQ(identical.p_value, 1);
  const Matrix constant(5, 2, 7.);
  EXPECT_DOUBLE_EQ(energy_distance_test_exact(constant, constant).p_value, 1);
  const Matrix far(2, 1, {4e200, 5e200}), near(2, 1, {0, 1e200});
  EXPECT_NEAR(energy_distance_test_exact(near, far).statistic / 1e200, 7, 1e-14);
  EXPECT_DOUBLE_EQ(energy_distance_test_exact(near, far).p_value, exact.p_value);

  SplitMix64 rng(42);
  EXPECT_THROW(energy_distance_test(first, second, 0, rng), std::invalid_argument);
  EXPECT_THROW(energy_distance_test_exact(first, Matrix(1, 2)), std::invalid_argument);
  EXPECT_THROW(energy_distance_test_exact(first, Matrix(0, 1)), std::invalid_argument);
  auto malformed = first;
  malformed.data().clear();
  EXPECT_THROW(energy_distance_test_exact(malformed, second), std::invalid_argument);
  EXPECT_THROW(energy_distance_test_exact(first, Matrix(1, 1, {std::numeric_limits<double>::infinity()})),
               std::invalid_argument);
  constexpr auto largest = std::numeric_limits<double>::max();
  EXPECT_THROW(energy_distance_test_exact(Matrix(1, 1, {-largest}), Matrix(1, 1, {largest})),
               std::overflow_error);
}

} // namespace
