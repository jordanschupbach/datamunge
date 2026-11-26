#include <datamunge/stats/hodges_lehmann.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::stats::Alternative;
using datamunge::stats::HodgesLehmannInterval;
using datamunge::stats::hodges_lehmann_location;
using datamunge::stats::hodges_lehmann_shift;

constexpr double inf = std::numeric_limits<double>::infinity();

// Reference values: wilcox.test(x, conf.int = TRUE, exact = TRUE) in R 4.5.2, and
// 1 - 2 * psignrank(qsignrank(alpha / 2, n) - 1, n) for the achieved coverage.
TEST(HodgesLehmann, LocationMatchesRExactInterval) {
    const auto hl = hodges_lehmann_location({1.1,2.3,.4,3.8,2.9,1.7});
    EXPECT_TRUE(hl.exact);
    EXPECT_EQ(hl.pair_count,21U);
    EXPECT_DOUBLE_EQ(hl.estimate,2.0);
    EXPECT_DOUBLE_EQ(hl.conf_int_lower,.4);
    EXPECT_DOUBLE_EQ(hl.conf_int_upper,3.8);
    EXPECT_NEAR(hl.achieved_conf_level,.96875,1e-12);

    const std::vector<double> x{-.62,1.41,.27,2.33,-1.05,.88,1.73,.12,2.91,.49,-.31,1.12};
    const auto two = hodges_lehmann_location(x,{.conf_level=.9});
    EXPECT_DOUBLE_EQ(two.estimate,.7375);
    EXPECT_DOUBLE_EQ(two.conf_int_lower,.12);
    EXPECT_DOUBLE_EQ(two.conf_int_upper,1.41);
    const auto greater = hodges_lehmann_location(x,{.conf_level=.9,.alternative=Alternative::Greater});
    EXPECT_DOUBLE_EQ(greater.conf_int_lower,.25);
    EXPECT_EQ(greater.conf_int_upper,inf);
    const auto less = hodges_lehmann_location(x,{.conf_level=.9,.alternative=Alternative::Less});
    EXPECT_EQ(less.conf_int_lower,-inf);
    EXPECT_DOUBLE_EQ(less.conf_int_upper,1.3);
}

TEST(HodgesLehmann, LocationExactBeyondAutomaticSizeMatchesR) {
    std::vector<double> x;
    for (int i = 1; i <= 60; ++i)
        x.push_back(std::sin(i*1.7)*2+i/30.0);
    const auto automatic = hodges_lehmann_location(x);
    EXPECT_FALSE(automatic.exact);
    const auto hl = hodges_lehmann_location(x,{.method=HodgesLehmannInterval::exact});
    EXPECT_TRUE(hl.exact);
    EXPECT_NEAR(hl.estimate,1.04528268722816708,1e-12);
    EXPECT_NEAR(hl.conf_int_lower,.63046851971343654,1e-12);
    EXPECT_NEAR(hl.conf_int_upper,1.45860275510486392,1e-12);
    EXPECT_EQ(hl.lower_rank,649U);
    EXPECT_EQ(hl.upper_rank,1830U-649U+1U);
    EXPECT_NEAR(hl.achieved_conf_level,.95071849401563868,1e-10);
    // The continuity-corrected normal rank lies within one order statistic of the exact one.
    EXPECT_LE(std::abs(static_cast<double>(automatic.lower_rank)-649.0),1.0);
    EXPECT_NEAR(automatic.achieved_conf_level,.95,.003);
}

// Reference values: wilcox.test(x, y, conf.int = TRUE, exact = TRUE) and pwilcox in R 4.5.2.
TEST(HodgesLehmann, ShiftMatchesRExactInterval) {
    const auto hl = hodges_lehmann_shift({5.1,6.0,7.2,5.8},{4.0,4.9,3.6,5.5,4.2});
    EXPECT_TRUE(hl.exact);
    EXPECT_EQ(hl.pair_count,20U);
    EXPECT_NEAR(hl.estimate,1.65,1e-12);
    EXPECT_NEAR(hl.conf_int_lower,.2,1e-12);
    EXPECT_NEAR(hl.conf_int_upper,3.2,1e-12);
    EXPECT_NEAR(hl.achieved_conf_level,.96825396825396826,1e-12);

    const std::vector<double> a{.038,.707,1.259,-.152,1.196,1.03,1.085,2.117,-.219,2.267,.255,-.131,.284};
    const std::vector<double> b{.253,.152,-.308,-.953,-.648,1.224,.2,-.578,-.942};
    const auto two = hodges_lehmann_shift(a,b,{.conf_level=.9});
    EXPECT_NEAR(two.estimate,.885,1e-12);
    EXPECT_NEAR(two.conf_int_lower,.359,1e-12);
    EXPECT_NEAR(two.conf_int_upper,1.608,1e-12);
    const auto greater = hodges_lehmann_shift(a,b,{.conf_level=.9,.alternative=Alternative::Greater});
    EXPECT_NEAR(greater.conf_int_lower,.496,1e-12);
    EXPECT_EQ(greater.conf_int_upper,inf);
    EXPECT_NEAR(greater.achieved_conf_level,.90712476378111051,1e-12);
    EXPECT_EQ(greater.lower_rank,39U);
    const auto less = hodges_lehmann_shift(a,b,{.conf_level=.9,.alternative=Alternative::Less});
    EXPECT_EQ(less.conf_int_lower,-inf);
    EXPECT_NEAR(less.conf_int_upper,1.338,1e-12);
}

TEST(HodgesLehmann, ShiftLargerExactSamplesMatchR) {
    std::vector<double> x, y;
    for (int j = 1; j <= 35; ++j)
        x.push_back(std::cos(j*2.3)+.8);
    for (int k = 1; k <= 28; ++k)
        y.push_back(std::cos(k*1.1+.4));
    const auto hl = hodges_lehmann_shift(x,y,{.conf_level=.99});
    EXPECT_TRUE(hl.exact);
    EXPECT_NEAR(hl.estimate,.79082536264828884,1e-12);
    EXPECT_NEAR(hl.conf_int_lower,.30652645792074418,1e-12);
    EXPECT_NEAR(hl.conf_int_upper,1.29857345118210676,1e-12);
    EXPECT_EQ(hl.lower_rank,305U);
    EXPECT_NEAR(hl.achieved_conf_level,.99039465278537864,1e-10);
}

TEST(HodgesLehmann, TiesSelectNormalIntervalAndShiftIsEquivariant) {
    const std::vector<double> x{1,2,2,3,5,8,8,9}, y{0,1,1,4};
    const auto hl = hodges_lehmann_shift(x,y);
    EXPECT_FALSE(hl.exact);
    std::vector<double> shifted(x);
    for (auto& v : shifted)
        v += 10;
    const auto moved = hodges_lehmann_shift(shifted,y);
    EXPECT_NEAR(moved.estimate-hl.estimate,10,1e-12);
    EXPECT_NEAR(moved.conf_int_lower-hl.conf_int_lower,10,1e-12);
    EXPECT_NEAR(moved.conf_int_upper-hl.conf_int_upper,10,1e-12);
    EXPECT_LE(hl.conf_int_lower,hl.estimate);
    EXPECT_GE(hl.conf_int_upper,hl.estimate);
}

TEST(HodgesLehmann, RejectsInvalidInput) {
    EXPECT_THROW((void)hodges_lehmann_location({1}),std::invalid_argument);
    EXPECT_THROW((void)hodges_lehmann_location({1,std::nan("")}),std::invalid_argument);
    EXPECT_THROW((void)hodges_lehmann_location({1,2},{.conf_level=1}),std::invalid_argument);
    EXPECT_THROW((void)hodges_lehmann_location({1,2},{.alternative=static_cast<Alternative>(9)}),
                 std::invalid_argument);
    EXPECT_THROW((void)hodges_lehmann_shift({},{1}),std::invalid_argument);
    EXPECT_THROW((void)hodges_lehmann_shift({1},{inf}),std::invalid_argument);
    EXPECT_THROW((void)hodges_lehmann_location(std::vector<double>(1001,1.0),{.method=HodgesLehmannInterval::exact}),
                 std::invalid_argument);
}

} // namespace
