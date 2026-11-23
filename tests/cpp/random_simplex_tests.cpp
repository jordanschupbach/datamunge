#include <datamunge/random/count_distributions.hpp>
#include <datamunge/random/simplex.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace {

using namespace datamunge::random;

TEST(DirichletLaws, DensitiesAndMassesMatchDirectGammaFormulas) {
  EXPECT_NEAR(dirichlet_log_pdf({.2, .3, .5}, {.5, 3.25, 7}), .36895120214101329, 1e-13);
  EXPECT_NEAR(dirichlet_log_pdf({.1, .2, .3, .4}, {23, 47, 7, 120}), -39.91245953307495142, 1e-11);
  EXPECT_NEAR(dirichlet_log_pdf({.5, .5}, {.002, 250}), -178.10430509980167813, 1e-11);
  EXPECT_NEAR(dirichlet_log_pdf({.5, .5}, {1e200, 1e200}),
              .5 * std::log(1e200) + std::log(2 / std::sqrt(std::numbers::pi)), 1e-12);
  EXPECT_NEAR(dirichlet_multinomial_log_pmf({2, 0, 6}, {.5, 3.25, 7}), -4.6957105917556206, 1e-13);
  EXPECT_NEAR(dirichlet_multinomial_log_pmf({20, 50, 150, 200}, {23, 47, 7, 120}),
              -59.758086422619954, 1e-11);
  // A stable product of the 1,000 predictive probabilities gives the reference.
  EXPECT_NEAR(dirichlet_multinomial_log_pmf({0, 1000}, {.002, 250}), -.003222071969503343, 1e-14);
  for (std::size_t x = 0; x <= 250; ++x)
    EXPECT_NEAR(std::exp(dirichlet_multinomial_log_pmf({x, 250 - x}, {2.75, 8.25})),
                beta_binomial_pmf(x, 250, 2.75, 8.25), 1e-14);
  const auto n = std::numeric_limits<std::size_t>::max();
  EXPECT_NEAR(dirichlet_multinomial_log_pmf({n - 7, 5, 2}, {1, 1, 1}),
              -std::log(.5L * (static_cast<long double>(n) + 1) * (static_cast<long double>(n) + 2)),
              1e-12);
}

TEST(DirichletLaws, NormalizesAndChecksTheSimplexSupport) {
  double normalization = 0;
  for (std::size_t a = 0; a <= 12; ++a)
    for (std::size_t b = 0; b <= 12 - a; ++b)
      normalization += std::exp(dirichlet_multinomial_log_pmf({a, b, 12 - a - b}, {.25, 3, 6}));
  EXPECT_NEAR(normalization, 1, 1e-14);
  const auto infinity = std::numeric_limits<double>::infinity();
  EXPECT_EQ(dirichlet_log_pdf({0, .4, .6}, {.5, 2, 3}), -infinity);
  EXPECT_EQ(dirichlet_log_pdf({.1, .2, .3}, {1, 1, 1}), -infinity);
  EXPECT_DOUBLE_EQ(dirichlet_log_pdf({1}, {7}), 0);
  EXPECT_DOUBLE_EQ(dirichlet_multinomial_log_pmf({0, 0}, {2, 3}), 0);
  EXPECT_THROW(dirichlet_log_pdf({.5, .5}, {2}), std::invalid_argument);
  EXPECT_THROW(dirichlet_log_pdf({infinity, 1}, {2, 3}), std::invalid_argument);
  EXPECT_THROW(dirichlet_multinomial_log_pmf({1, 2}, {0, 3}), std::invalid_argument);
  EXPECT_THROW(dirichlet_multinomial_log_pmf({std::numeric_limits<std::size_t>::max(), 1}, {2, 3}),
               std::overflow_error);
}

TEST(DirichletLaws, SeededSamplersReproduceTheirDrawsAndMoments) {
  SplitMix64 rng(2026), copy(2026);
  const std::vector<double> alpha{.5, 1.5, 3.0};
  EXPECT_EQ(dirichlet_sample(rng, alpha), dirichlet_sample(copy, alpha));
  EXPECT_EQ(dirichlet_multinomial_sample(rng, 20, alpha), dirichlet_multinomial_sample(copy, 20, alpha));
  EXPECT_EQ(dirichlet_multinomial_sample(rng, 0, alpha), (std::vector<std::size_t>{0, 0, 0}));
  constexpr std::size_t repetitions = 12000, trials = 20;
  std::array<double, 3> sum{}, counts{};
  double squares = 0;
  for (std::size_t trial = 0; trial < repetitions; ++trial) {
    const auto point = dirichlet_sample(rng, alpha);
    const auto draw = dirichlet_multinomial_sample(rng, trials, alpha);
    ASSERT_NEAR(std::accumulate(point.begin(), point.end(), 0.), 1, 3e-16);
    ASSERT_EQ(std::accumulate(draw.begin(), draw.end(), std::size_t{0}), trials);
    for (std::size_t i = 0; i < alpha.size(); ++i) {
      sum[i] += point[i];
      counts[i] += draw[i];
    }
    squares += draw[0] * draw[0];
  }
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    EXPECT_NEAR(sum[i] / repetitions, alpha[i] / 5, .005);
    EXPECT_NEAR(counts[i] / repetitions, trials * alpha[i] / 5, .12);
  }
  const auto mean = counts[0] / repetitions;
  EXPECT_NEAR(squares / repetitions - mean * mean, 20 * .1 * .9 * 25 / 6, .45);

  const auto tiny = std::numeric_limits<double>::denorm_min();
  const auto one_vertex = dirichlet_sample(rng, {tiny, tiny, tiny});
  EXPECT_DOUBLE_EQ(*std::max_element(one_vertex.begin(), one_vertex.end()), 1);
  const auto huge = std::numeric_limits<std::size_t>::max();
  const auto large = dirichlet_multinomial_sample(rng, huge, {1e200, 1e200});
  EXPECT_EQ(large[0] + large[1], huge);
  EXPECT_NEAR(static_cast<double>(large[0]) / huge, .5, 1e-8);
  EXPECT_THROW(dirichlet_sample(rng, {}), std::invalid_argument);
}

} // namespace
