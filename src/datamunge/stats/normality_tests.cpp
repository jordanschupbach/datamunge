/// @file normality_tests.cpp
/// @brief Normal-reference transformations for standardized sample moments.

#include <datamunge/stats/normality_tests.hpp>

#include <datamunge/stats/sample_moments.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Moment components and their chi-square calibration.
namespace {

/// @brief Transforms sample skewness toward a standard normal reference.
/// @param skewness Uncorrected standardized third sample moment.
/// @param n Sample size, at least eight.
/// @return Signed skewness z-score.
long double skewness_z(long double skewness, long double n) {
  const auto y = skewness * std::sqrt((n + 1) * (n + 3) / (6 * (n - 2)));
  // Compute beta_2 - 3 directly to retain precision as it tends to zero.
  const auto excess = 36 * (n * (n * (n - 5) - 19) + 35) /
                      ((n - 2) * (n + 5) * (n + 7) * (n + 9));
  const auto w_squared_minus_one = (2 * excess) / (std::sqrt(4 + 2 * excess) + 2);
  return std::asinh(y * std::sqrt(w_squared_minus_one / 2)) *
         std::sqrt(2 / std::log1p(w_squared_minus_one));
}

/// @brief Transforms sample kurtosis using the Anscombe–Glynn reference.
/// @param excess Uncorrected standardized fourth sample moment minus three.
/// @param n Sample size, at least eight.
/// @return Signed kurtosis z-score.
long double kurtosis_z(long double excess, long double n) {
  const auto variance = 24 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5));
  const auto standardized = (excess + 6 / (n + 1)) / std::sqrt(variance);
  const auto root_beta = 6 * (n * (n - 5) + 2) / ((n + 7) * (n + 9)) *
                         std::sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)));
  const auto reciprocal = 2 / root_beta;
  const auto a = 6 + 4 * reciprocal * (reciprocal + std::hypot(1.0L, reciprocal));
  const auto denominator = 1 + standardized * std::sqrt(2 / (a - 4));
  const auto center = 1 - 2 / (9 * a);
  const auto root = std::cbrt((1 - 2 / a) / denominator);
  return (center - root) * std::sqrt(9 * a / 2);
}

/// @brief Calibrates the sum of the two normality components using chi-square with two degrees of freedom.
/// @param skewness Nonnegative contribution from skewness.
/// @param kurtosis Nonnegative contribution from kurtosis.
/// @return Both contributions, the combined statistic and its upper tail.
MomentNormalityResult combine_moments(long double skewness, long double kurtosis) {
  const auto statistic = skewness + kurtosis;
  return {static_cast<double>(statistic), static_cast<double>(std::exp(-statistic / 2)),
          static_cast<double>(skewness), static_cast<double>(kurtosis)};
}

} // namespace

MomentNormalityResult dagostino_pearson_test(const std::vector<double>& values) {
  if (values.size() < 8)
    throw std::invalid_argument("dagostino_pearson_test: need at least eight observations");
  const auto shape = sample_shape(values);
  const auto skewness = skewness_z(shape.skewness, values.size());
  const auto kurtosis = kurtosis_z(shape.excess_kurtosis, values.size());
  return combine_moments(skewness * skewness, kurtosis * kurtosis);
}

MomentNormalityResult jarque_bera_test(const std::vector<double>& values) {
  const auto shape = sample_shape(values);
  const long double skewness = shape.skewness, kurtosis = shape.excess_kurtosis;
  return combine_moments(values.size() * skewness * skewness / 6,
                         values.size() * kurtosis * kurtosis / 24);
}

} // namespace datamunge::stats
