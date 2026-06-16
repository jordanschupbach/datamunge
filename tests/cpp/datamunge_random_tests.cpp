#include <gtest/gtest.h>

#include <datamunge/random/distributions.hpp>
#include <datamunge/random/random.hpp>

#include <cmath>

TEST(SplitMix64, SameSeedProducesSameSequence) {
  datamunge::random::SplitMix64 a(0x12345678ULL);
  datamunge::random::SplitMix64 b(0x12345678ULL);

  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(a.next_u64(), b.next_u64());
  }
}

TEST(SplitMix64, UniformIsBounded) {
  datamunge::random::SplitMix64 rng(0xCAFEBABEULL);

  for (int i = 0; i < 64; ++i) {
    const double u = rng.uniform01();
    EXPECT_GE(u, 0.0);
    EXPECT_LT(u, 1.0);
  }
}

TEST(SplitMix64, NormalIsDeterministicAndFinite) {
  datamunge::random::SplitMix64 a(0xDEADBEEFULL);
  datamunge::random::SplitMix64 b(0xDEADBEEFULL);

  for (int i = 0; i < 16; ++i) {
    const double x = a.normal();
    const double y = b.normal();
    EXPECT_TRUE(std::isfinite(x));
    EXPECT_DOUBLE_EQ(x, y);
  }
}

TEST(RandomDistributions, NormalRoundTrip) {
  const double p = datamunge::random::normal_cdf(1.25, 0.5, 2.0);
  const double x = datamunge::random::normal_quantile(p, 0.5, 2.0);
  EXPECT_NEAR(x, 1.25, 1e-6);
}

TEST(RandomDistributions, LogisticRoundTrip) {
  const double x = 1.1;
  const double p = datamunge::random::logistic_cdf(x, -0.5, 0.75);
  EXPECT_NEAR(datamunge::random::logistic_quantile(p, -0.5, 0.75), x, 1e-12);
}

TEST(RandomDistributions, ExponentialRoundTrip) {
  const double x = 0.8;
  const double p = datamunge::random::exponential_cdf(x, 3.0);
  EXPECT_NEAR(datamunge::random::exponential_quantile(p, 3.0), x, 1e-12);
}

TEST(RandomDistributions, WeibullRoundTrip) {
  const double x = 1.7;
  const double p = datamunge::random::weibull_cdf(x, 1.5, 2.0);
  EXPECT_NEAR(datamunge::random::weibull_quantile(p, 1.5, 2.0), x, 1e-12);
}

TEST(RandomDistributions, CauchyRoundTrip) {
  const double x = -0.7;
  const double p = datamunge::random::cauchy_cdf(x, 0.25, 1.5);
  EXPECT_NEAR(datamunge::random::cauchy_quantile(p, 0.25, 1.5), x, 1e-12);
}

TEST(RandomDistributions, PoissonRoundTripOrdering) {
  const double lambda = 4.0;
  const auto   k      = datamunge::random::poisson_quantile(0.70, lambda);
  const double cdf_k  = datamunge::random::poisson_cdf(k, lambda);
  EXPECT_GE(cdf_k, 0.70);
  if (k > 0) {
    EXPECT_LT(datamunge::random::poisson_cdf(k - 1, lambda), 0.70);
  }
}

TEST(RandomDistributions, BernoulliAndGeometricBasicValues) {
  EXPECT_DOUBLE_EQ(datamunge::random::bernoulli_pdf(1, 0.3), 0.3);
  EXPECT_DOUBLE_EQ(datamunge::random::bernoulli_cdf(0, 0.3), 0.7);
  EXPECT_EQ(datamunge::random::bernoulli_quantile(0.69, 0.3), 0);
  EXPECT_EQ(datamunge::random::bernoulli_quantile(0.71, 0.3), 1);

  EXPECT_NEAR(datamunge::random::geometric_pdf(2, 0.25), 0.140625, 1e-15);
  EXPECT_NEAR(datamunge::random::geometric_cdf(2, 0.25), 0.578125, 1e-15);
  EXPECT_EQ(datamunge::random::geometric_quantile(0.57, 0.25), 2u);
}
