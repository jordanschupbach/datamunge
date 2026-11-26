#include <datamunge/stats/tolerance_limits.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

// Reference values: tolerance 3.x K.factor(n, alpha, P, side, method = "EXACT", m = 200) and
// sqrt((n-1)(1+1/n) qnorm((1+P)/2)^2 / qchisq(alpha, n-1)) for Howe's factor, in R 4.5.3.
TEST(ToleranceLimits, NormalFactorsMatchRTolerancePackage) {
    struct Row { std::size_t n; double one, two, howe, two99; };
    const Row rows[]{{5,3.40663326280349,4.29060407068658,4.27462167840056,10.22009030504328},
                     {10,2.35464013183444,2.85631084699453,2.83819127023668,5.61016827112153},
                     {30,1.77732879783457,2.14511108878345,2.13972139067339,3.74246349781938},
                     {100,1.52674874785104,1.87480754376836,1.87382728072877,3.09757020287023}};
    for (const auto& r : rows) {
        EXPECT_NEAR(normal_tolerance_factor(r.n,.9,.95,ToleranceSide::upper),r.one,1e-6) << r.n;
        EXPECT_NEAR(normal_tolerance_factor(r.n,.9,.95,ToleranceSide::lower),r.one,1e-6) << r.n;
        EXPECT_NEAR(normal_tolerance_factor(r.n,.9,.95),r.two,2e-5) << r.n;
        EXPECT_NEAR(normal_tolerance_factor(r.n,.9,.95,ToleranceSide::two_sided,NormalToleranceMethod::howe),
                    r.howe,1e-7) << r.n;
        EXPECT_NEAR(normal_tolerance_factor(r.n,.99,.99),r.two99,2e-5) << r.n;
    }
}

TEST(ToleranceLimits, NormalIntervalsMatchRNormtolInt) {
    const std::vector<double> x{10.2,9.8,11.1,10.5,9.4,10.9,10.0,10.7,9.9,10.3,11.4,9.6};
    const auto two = normal_tolerance_interval(x,.9,.95);
    EXPECT_NEAR(two.lower,8.66338863313784,1e-4);
    EXPECT_NEAR(two.upper,11.9699447001955,1e-4);
    const auto upper = normal_tolerance_interval(x,.9,.95,ToleranceSide::upper);
    EXPECT_EQ(upper.lower,-std::numeric_limits<double>::infinity());
    EXPECT_NEAR(upper.upper,11.6850457545407,1e-6);
    const auto lower = normal_tolerance_interval(x,.9,.95,ToleranceSide::lower);
    EXPECT_NEAR(lower.lower,8.94828757879267,1e-6);
    EXPECT_EQ(lower.upper,std::numeric_limits<double>::infinity());
}

// Reference values: nptol.int(y, alpha, P, side, method = "WILKS") in R.
TEST(ToleranceLimits, OrderStatisticLimitsMatchWilksMethod) {
    const std::vector<double> y{.755,1.182,.146,.14,.436,2.895,1.23,.54,.957,.147,1.391,.762,1.238,4.424,1.055,1.035,
                                1.876,.655,.337,.588,2.365,.642,.294,.566,.106,.059,.579,3.959,1.173,.997,1.435,.037,
                                .324,1.32,.204,1.023,.302,.725,.752,.235,1.08,1.028,1.292,1.253,.555,.301,1.293,.995,
                                .514,2.008,.422,2.179,3.218,.558,.595,.977,.21,.309,1.106,.774};
    const auto two = nonparametric_tolerance_interval(y,.9,.95);
    EXPECT_TRUE(two.feasible);
    EXPECT_DOUBLE_EQ(two.lower,.037);
    EXPECT_DOUBLE_EQ(two.upper,4.424);
    const auto upper = nonparametric_tolerance_interval(y,.9,.95,ToleranceSide::upper);
    EXPECT_DOUBLE_EQ(upper.upper,3.959);
    EXPECT_EQ(upper.upper_rank,59U);
    const auto lower = nonparametric_tolerance_interval(y,.9,.95,ToleranceSide::lower);
    EXPECT_DOUBLE_EQ(lower.lower,.059);
    EXPECT_EQ(lower.lower_rank,2U);
    // 1 - 0.9^60 - 60 * 0.9^59 * 0.1 for the upper bound X_(59).
    EXPECT_NEAR(upper.achieved_confidence,1-std::pow(.9,60)-60*std::pow(.9,59)*.1,1e-12);

    // n = 1000 at P = 0.9: r with P(Bin(1000, .9) <= 1000 - 2r) >= 0.95 gives an interior pair.
    std::vector<double> many(1000);
    for (std::size_t i = 0; i < many.size(); ++i)
        many[i] = static_cast<double>(i);
    const auto wide = nonparametric_tolerance_interval(many,.9,.95);
    EXPECT_GT(wide.lower_rank,1U);
    EXPECT_EQ(wide.upper_rank,1000U-wide.lower_rank+1);
    EXPECT_GE(wide.achieved_confidence,.95);
}

TEST(ToleranceLimits, SmallSamplesAreInfeasibleAndInputsValidated) {
    const auto small = nonparametric_tolerance_interval({1,2,3,4,5},.9,.95);
    EXPECT_FALSE(small.feasible);
    EXPECT_EQ(small.upper,std::numeric_limits<double>::infinity());
    EXPECT_NEAR(small.achieved_confidence,1-std::pow(.9,5)-5*std::pow(.9,4)*.1,1e-12);
    EXPECT_THROW((void)normal_tolerance_factor(1,.9,.95),std::invalid_argument);
    EXPECT_THROW((void)normal_tolerance_factor(5,1,.95),std::invalid_argument);
    EXPECT_THROW((void)normal_tolerance_interval({2,2,2},.9,.95),std::invalid_argument);
    EXPECT_THROW((void)nonparametric_tolerance_interval({},.9,.95),std::invalid_argument);
}

} // namespace
