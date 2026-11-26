#include <datamunge/stats/dunnett.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::stats::Alternative;
using datamunge::stats::dunnett_test;

// Control first, then treatments A, B and D with unequal sizes 6, 5, 7 and 4.
const std::vector<std::vector<double>> groups{{10.1,11.3,9.6,10.4,12.0,10.8},
                                              {12.4,13.1,11.9,12.8,13.5},
                                              {10.9,11.6,10.2,11.1,12.3,10.7,11.5},
                                              {9.8,10.6,11.0,10.3}};

// Reference values: statistics and standard errors from multcomp 1.4 glht(aov(y ~ g), mcp(g = "Dunnett"))
// in R 4.5.3; p-values from mvtnorm pmvt(corr, df = 18, GenzBretz(abseps = 1e-9)) and critical values
// by uniroot on pmvt, because multcomp's default integration (abseps = 1e-3) is too coarse here.
TEST(Dunnett, TwoSidedMatchesMultcomp) {
    const auto d = dunnett_test(groups);
    ASSERT_EQ(d.comparisons.size(),3U);
    EXPECT_DOUBLE_EQ(d.df,18);
    const double t[]{4.810583812469,1.246628867580,-.608333380901};
    const double se[]{.424064953346,.389622203004,.452054759173};
    const double p[]{.000398890143393,.481555068514021,.879531753953470};
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_EQ(d.comparisons[j].treatment,j+1);
        EXPECT_NEAR(d.comparisons[j].statistic,t[j],1e-10);
        EXPECT_NEAR(d.comparisons[j].std_error,se[j],1e-10);
        EXPECT_NEAR(d.comparisons[j].p_value,p[j],2e-5);
    }
    EXPECT_NEAR(d.critical_value,2.57177087361,1e-4);
    EXPECT_NEAR(d.comparisons[0].conf_low,2.04-2.57177087361*se[0],1e-4);
    EXPECT_NEAR(d.comparisons[2].conf_high,-.275+2.57177087361*se[2],1e-4);
    EXPECT_LT(d.integration_error,1e-4);
}

TEST(Dunnett, OneSidedAlternativesMatchMultcomp) {
    const auto greater = dunnett_test(groups,{.alternative=Alternative::Greater});
    EXPECT_NEAR(greater.comparisons[0].p_value,.000199452704286,2e-5);
    EXPECT_NEAR(greater.comparisons[1].p_value,.247113802141011,2e-5);
    EXPECT_NEAR(greater.comparisons[2].p_value,.918901482663181,2e-5);
    EXPECT_NEAR(greater.critical_value,2.2176938587,1e-4);
    EXPECT_EQ(greater.comparisons[0].conf_high,std::numeric_limits<double>::infinity());
    EXPECT_NEAR(greater.comparisons[0].conf_low,2.04-2.2176938587*.424064953346,1e-4);
    const auto less = dunnett_test(groups,{.alternative=Alternative::Less});
    EXPECT_NEAR(less.comparisons[2].p_value,.505977981916,2e-5);
    EXPECT_NEAR(less.comparisons[1].p_value,.981635994552,2e-5);
    EXPECT_EQ(less.comparisons[0].conf_low,-std::numeric_limits<double>::infinity());
    EXPECT_NEAR(less.comparisons[1].conf_high,.485714285714+2.2176938587*.389622203004,1e-4);
}

TEST(Dunnett, TwoGroupsReduceToPooledTTest) {
    // One comparison: the adjusted p-value is the two-sided pooled t-test p-value, and c = qt(.975, df).
    const auto d = dunnett_test({{1,2,3,4},{3,4,5,6,7}});
    EXPECT_NEAR(d.comparisons[0].statistic,2.5458753860865779,1e-12);
    EXPECT_NEAR(d.comparisons[0].p_value,.0383337288360308,1e-9); // t.test(var.equal = TRUE)
    EXPECT_NEAR(d.critical_value,2.364624251592785,1e-8);       // qt(.975, 7)
}

TEST(Dunnett, ControlIndexAndValidation) {
    const auto moved = dunnett_test({groups[1],groups[0],groups[2],groups[3]},{.control=1});
    EXPECT_EQ(moved.comparisons[0].treatment,0U);
    EXPECT_NEAR(moved.comparisons[0].statistic,4.810583812469,1e-10);
    EXPECT_THROW((void)dunnett_test({{1,2}}),std::invalid_argument);
    EXPECT_THROW((void)dunnett_test({{1},{2}}),std::invalid_argument);
    EXPECT_THROW((void)dunnett_test({{1,1},{2,2}}),std::invalid_argument);
    EXPECT_THROW((void)dunnett_test(groups,{.control=4}),std::invalid_argument);
    EXPECT_THROW((void)dunnett_test({{1,2},{}}),std::invalid_argument);
    EXPECT_THROW((void)dunnett_test(groups,{.conf_level=1}),std::invalid_argument);
}

} // namespace
