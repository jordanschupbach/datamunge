#include <datamunge/random/noncentral_distributions.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using namespace datamunge::random;

TEST(NoncentralDistributions, MatchesRForOrdinaryAndSmallTailProbabilities) {
  EXPECT_NEAR(noncentral_chi_square_cdf(4, 3, 2), .48388135838807439, 1e-13);
  EXPECT_NEAR(noncentral_chi_square_cdf(27, 7, 3.5, false), .0081210153434700168, 1e-13);
  EXPECT_NEAR(noncentral_chi_square_cdf(.1, .5, 14), .0006109787373035242, 1e-14);
  EXPECT_NEAR(noncentral_chi_square_cdf(950, 10, 1000, false), .82763658833748799, 1e-12);
  // Independently sum R's dpois weights against directed central pbeta tails.
  EXPECT_NEAR(noncentral_f_cdf(2, 4, 12, 3), .60702476819298812, 1e-12);
  EXPECT_NEAR(noncentral_f_cdf(4.2, 6.5, 9, 17, false), .42321801145201082, 1e-12);
  EXPECT_NEAR(noncentral_f_cdf(.1, 1.25, 3.5, 14), .00023024861519783943, 1e-13);
  EXPECT_NEAR(noncentral_f_cdf(135, 8, 50, 1000), .60562397019705683, 1e-12);
  EXPECT_NEAR(noncentral_f_cdf(2, 5, 3, 0, false), .30154736269507576, 1e-13);
  EXPECT_NEAR(noncentral_f_cdf(1e50, 2, 8, 4, false) / 6.912e-197, 1, 1e-12);
  EXPECT_NEAR(noncentral_chi_square_cdf(3e6 + 4000, 7, 3e6), .87545949957658875, 1e-8);
  EXPECT_NEAR(noncentral_f_cdf(510000, 6, 12, 3e6), .46475539918844722, 1e-8);
  EXPECT_NEAR(noncentral_f_cdf(500001, 6, 3e6, 3e6), .50004179162868889, 1e-8);
}

TEST(NoncentralDistributions, InvertsBothTailsAndHandlesTheChiSquareAtom) {
  EXPECT_NEAR(noncentral_chi_square_quantile(.97, 7, 12), 35.96911811248996571, 1e-9);
  EXPECT_NEAR(noncentral_f_quantile(.05, 5, 25, 11, false), 7.2138299752430859, 1e-9);
  const auto f = noncentral_f_quantile(1e-35, 3, 11, 19, false);
  EXPECT_NEAR(noncentral_f_cdf(f, 3, 11, 19, false) / 1e-35, 1, 1e-12);
  const auto x = noncentral_chi_square_quantile(1e-35, 4, 8, false);
  EXPECT_NEAR(noncentral_chi_square_cdf(x, 4, 8, false) / 1e-35, 1, 1e-12);
  EXPECT_NEAR(noncentral_chi_square_noncentrality(.48388135838807439, 4, 3), 2, 1e-12);
  EXPECT_NEAR(noncentral_f_noncentrality(.42321801145201082, 4.2, 6.5, 9, false), 17, 1e-11);
  EXPECT_DOUBLE_EQ(noncentral_chi_square_cdf(0, 0, 2), std::exp(-1.));
  EXPECT_NEAR(noncentral_chi_square_noncentrality(std::exp(-1.), 0, 0), 2, 1e-14);
  EXPECT_DOUBLE_EQ(noncentral_chi_square_quantile(.3, 0, 2), 0);
  EXPECT_DOUBLE_EQ(noncentral_chi_square_quantile(std::exp(-1.), 0, 2), 0);
  EXPECT_DOUBLE_EQ(noncentral_chi_square_quantile(-std::expm1(-1.), 0, 2, false), 0);
  EXPECT_NEAR(noncentral_chi_square_quantile(.5, 0, 2), .79344513204025657, 1e-10);
  EXPECT_DOUBLE_EQ(noncentral_f_quantile(1, 4, 5, 6, false), 0);
  EXPECT_TRUE(std::isinf(noncentral_f_quantile(0, 4, 5, 6, false)));
  EXPECT_TRUE(std::isinf(noncentral_f_noncentrality(1, 1, 4, 5, false)));
  EXPECT_DOUBLE_EQ(noncentral_f_noncentrality(noncentral_f_cdf(3, 4, 5, 0), 3, 4, 5), 0);
  EXPECT_THROW(noncentral_f_noncentrality(.99, .1, 4, 5), std::invalid_argument);
  EXPECT_THROW(noncentral_f_cdf(1, 1, -1, 1), std::invalid_argument);
  EXPECT_THROW(noncentral_chi_square_cdf(std::numeric_limits<double>::quiet_NaN(), 1, 3),
               std::invalid_argument);
}

TEST(NoncentralDistributions, NoncentralTMatchesOrdinaryExtremeAndLimitingTails) {
  EXPECT_NEAR(noncentral_t_cdf(1, 10, -2), .99818356658941032, 1e-11);
  EXPECT_NEAR(noncentral_t_cdf(2, 7, 1.5), .64680078500369331, 1e-11);
  EXPECT_NEAR(noncentral_t_cdf(4, .5, 4), .26620141962829735, 1e-11);
  EXPECT_NEAR(noncentral_t_cdf(5, 100, 15) / 2.640405806735035e-21, 1, 1e-12);
  EXPECT_DOUBLE_EQ(noncentral_t_cdf(-5, 100, -15, false), noncentral_t_cdf(5, 100, 15));
  EXPECT_NEAR(noncentral_t_cdf(1, 1e-100, 5), .5 * std::erfc(5 / std::sqrt(2.)), 1e-20);
  EXPECT_NEAR(noncentral_t_cdf(1, 1e308, 1), .5, 1e-13);
  EXPECT_NEAR(noncentral_t_cdf(1, 1e308, 0, false), .5 * std::erfc(1 / std::sqrt(2.)), 1e-13);
  EXPECT_DOUBLE_EQ(noncentral_t_cdf(0, 12, 0), .5);
}

TEST(NoncentralDistributions, NoncentralTInvertsCutoffsAndSignedDisplacements) {
  EXPECT_NEAR(noncentral_t_quantile(.9, 7, 2), 3.97212736576244652, 1e-9);
  EXPECT_NEAR(noncentral_t_quantile(.03, 8, -3), -6.67475716816110776, 1e-8);
  const auto tail = noncentral_t_cdf(5, 100, 15);
  EXPECT_NEAR(noncentral_t_quantile(tail, 100, 15), 5, 1e-12);
  EXPECT_NEAR(noncentral_t_noncentrality(tail, 5, 100), 15, 1e-12);
  EXPECT_NEAR(noncentral_t_noncentrality(noncentral_t_cdf(1, 10, -2, false), 1, 10, false), -2, 1e-12);
  const auto upper = noncentral_t_quantile(1e-35, 8, -3, false);
  EXPECT_NEAR(noncentral_t_cdf(upper, 8, -3, false) / 1e-35, 1, 1e-12);
  EXPECT_DOUBLE_EQ(noncentral_t_quantile(noncentral_t_cdf(0, 2, -1), 2, -1), 0);
  constexpr auto infinity = std::numeric_limits<double>::infinity();
  EXPECT_EQ(noncentral_t_quantile(1, 5, -2), infinity);
  EXPECT_EQ(noncentral_t_quantile(1, 5, -2, false), -infinity);
  EXPECT_EQ(noncentral_t_noncentrality(1, 1, 5), -infinity);
  EXPECT_EQ(noncentral_t_noncentrality(1, 1, 5, false), infinity);
  EXPECT_THROW(noncentral_t_cdf(2, 0, 3), std::invalid_argument);
  EXPECT_THROW(noncentral_t_noncentrality(.5, infinity, 6), std::invalid_argument);
}

} // namespace
