#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/matrix_distributions.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace {

using namespace datamunge::random;
using Matrix = datamunge::linalg::DenseMatrix<double>;

TEST(WishartLaws, MatchesNormalizedDeterminantAndTraceFormulas) {
  const Matrix x(2, 2, {5, 1, 1, 3}), scale(2, 2, {2, .4, .4, 1});
  EXPECT_NEAR(wishart_log_pdf(x, 4.5, scale), -5.8964165186867161, 1e-13);
  EXPECT_NEAR(inverse_wishart_log_pdf(x, 4.5, scale), -12.6207760135122875, 1e-13);
  const Matrix x3(3, 3, {1, .1, .3, .1, 3, .25, .3, .25, 1.5});
  const Matrix s3(3, 3, {2, .3, .2, .3, 4, -.3, .2, -.3, 1});
  EXPECT_NEAR(wishart_log_pdf(x3, 2.75, s3), -10.4951225937672046, 1e-13);
  EXPECT_NEAR(inverse_wishart_log_pdf(x3, 2.75, s3), -9.5166123746322775, 1e-13);
  const Matrix scalar(1, 1, {2}), scalar_scale(1, 1, {4});
  EXPECT_NEAR(wishart_log_pdf(scalar, 2, scalar_scale), -.25 - std::log(8.), 1e-14);
  EXPECT_NEAR(inverse_wishart_log_pdf(scalar, 2, scalar_scale), -1 - std::log(2.), 1e-14);
  constexpr double huge = 1e200;
  EXPECT_NEAR(wishart_log_pdf(Matrix(1, 1, {huge}), huge, Matrix(1, 1, {1})),
              -.5 * (std::log(huge) + std::log(4 * std::numbers::pi)), 1e-12);
}

TEST(WishartLaws, VerifiesDomainsAndReproducesSeededDraws) {
  const auto identity = Matrix::identity(2);
  const Matrix indefinite(2, 2, {1, 2, 2, 1}), asymmetric(2, 2, {1, 0, .1, 1});
  EXPECT_THROW(wishart_log_pdf(identity, 1, identity), std::invalid_argument);
  EXPECT_THROW(wishart_log_pdf(identity, 3, indefinite), std::invalid_argument);
  EXPECT_THROW(inverse_wishart_log_pdf(asymmetric, 3, identity), std::invalid_argument);
  EXPECT_THROW(inverse_wishart_log_pdf(Matrix(1, 1, {1}), 3, identity), std::invalid_argument);
  constexpr auto infinity = std::numeric_limits<double>::infinity();
  EXPECT_THROW(wishart_log_pdf(Matrix(2, 2, {infinity, 0, 0, 1}), 3, identity), std::invalid_argument);

  SplitMix64 first(731), second(731), copy(731);
  const auto draw = wishart_sample(first, 6.5, identity);
  const auto inverse = inverse_wishart_sample(second, 6.5, identity);
  EXPECT_EQ(draw(0, 1), draw(1, 0));
  EXPECT_EQ(inverse(0, 1), inverse(1, 0));
  EXPECT_TRUE(datamunge::linalg::cholesky(draw).ok);
  EXPECT_TRUE(datamunge::linalg::cholesky(inverse).ok);
  EXPECT_DOUBLE_EQ(draw(1, 0), wishart_sample(copy, 6.5, identity)(1, 0));
  const auto product = draw * inverse;
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(product(i, j), i == j ? 1 : 0, 1e-14);
  EXPECT_THROW(inverse_wishart_sample(first, 4, asymmetric), std::invalid_argument);
  EXPECT_THROW(wishart_sample(first, 40, Matrix(1, 1, {std::numeric_limits<double>::max()})),
               std::overflow_error);
}

TEST(WishartLaws, EmpiricalMeansFollowTheScaleParameterizations) {
  const Matrix scale(2, 2, {2, .4, .4, 1});
  constexpr double df = 9.5;
  constexpr std::size_t draws = 10000;
  SplitMix64 rng(42);
  Matrix sums(2, 2), inverse_sums(2, 2);
  for (std::size_t i = 0; i < draws; ++i) {
    const auto wishart = wishart_sample(rng, df, scale);
    const auto inverse = inverse_wishart_sample(rng, df, scale);
    for (std::size_t r = 0; r < 2; ++r) {
      for (std::size_t c = 0; c < 2; ++c) {
        sums(r, c) += wishart(r, c);
        inverse_sums(r, c) += inverse(r, c);
      }
    }
  }
  for (std::size_t r = 0; r < 2; ++r) {
    for (std::size_t c = 0; c < 2; ++c) {
      EXPECT_NEAR(sums(r, c) / draws, df * scale(r, c), .25);
      EXPECT_NEAR(inverse_sums(r, c) / draws, scale(r, c) / (df - 3), .007);
    }
  }
}

} // namespace
