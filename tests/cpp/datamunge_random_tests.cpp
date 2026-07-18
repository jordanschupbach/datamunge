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

TEST(RandomDistributions, StudentTMatchesKnownRValues) {
  using namespace datamunge::random;

  // Reference values from R: qt(), pt(), dt().
  EXPECT_NEAR(student_t_quantile(0.975, 10.0), 2.228139, 1e-5);
  EXPECT_NEAR(student_t_quantile(0.95, 1.0), 6.313752, 1e-4);
  EXPECT_NEAR(student_t_quantile(0.5, 7.0), 0.0, 1e-12);

  EXPECT_NEAR(student_t_cdf(2.228139, 10.0), 0.975, 1e-5);
  EXPECT_NEAR(student_t_cdf(0.0, 5.0), 0.5, 1e-12);
  EXPECT_NEAR(student_t_cdf(-1.812461, 10.0), 0.05, 1e-5);

  EXPECT_NEAR(student_t_pdf(0.0, 1.0), 1.0 / M_PI, 1e-12);

  // Large df should approach the standard normal distribution.
  EXPECT_NEAR(student_t_cdf(1.959964, 1.0e7), normal_cdf(1.959964), 1e-4);
}

TEST(RandomDistributions, FDistributionMatchesKnownRValues) {
  using namespace datamunge::random;

  // Reference values from R: qf(), pf().
  EXPECT_NEAR(f_quantile(0.95, 1.0, 10.0), 4.964603, 1e-4);
  EXPECT_NEAR(f_quantile(0.95, 5.0, 20.0), 2.710894, 1e-4);

  EXPECT_NEAR(f_cdf(4.964603, 1.0, 10.0), 0.95, 1e-5);
  EXPECT_NEAR(f_cdf(1.0, 5.0, 5.0), 0.5, 1e-9);

  // F(1, df2) is the square of a t(df2) variable: relate the two CDFs.
  const double t_stat = student_t_quantile(0.975, 12.0);
  EXPECT_NEAR(f_cdf(t_stat * t_stat, 1.0, 12.0), 0.95, 1e-5);
}
