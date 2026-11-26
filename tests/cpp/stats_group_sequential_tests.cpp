#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/group_sequential.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

const std::vector<double> looks{.25,.5,.75,1};

// Reference values: boundaries solved with uniroot() on mvtnorm::pmvnorm(algorithm = Miwa(steps = 4096))
// crossing probabilities, in R 4.5.3 (ldbounds::ldBounds agrees to about 1e-4).
TEST(GroupSequential, OBrienFlemingBoundaryMatchesMiwa) {
    const auto b = group_sequential_boundaries(looks,{});
    const std::vector<double> miwa{4.33263364604956,2.96313159767999,2.35904428530116,2.01409014403423};
    for (std::size_t k = 0; k < 4; ++k)
        EXPECT_NEAR(b.z[k],miwa[k],2e-7) << k;
    EXPECT_NEAR(b.cumulative_alpha[3],.025,1e-15);
    EXPECT_NEAR(b.nominal_alpha[0],7.36680843593795e-06,1e-12);
    const auto p = boundary_crossing_probabilities(looks,b.z,3);
    const std::vector<double> stage{.00230831314872482,.19770789273962872,.39799795278359051,.24616992639063184};
    for (std::size_t k = 0; k < 4; ++k)
        EXPECT_NEAR(p.stage[k],stage[k],1e-7) << k;
    EXPECT_NEAR(p.total,.844184085062576,1e-7);
    EXPECT_NEAR(boundary_crossing_probabilities(looks,b.z,0).total,.025,1e-9);
    EXPECT_NEAR(drift_for_power(looks,b.z,.9),3.27100885734655,1e-6);
}

// Reference values: ldbounds 2.0 ldBounds(..., sides = 1, alpha = 0.025) upper.bounds, which use a coarser grid.
TEST(GroupSequential, OtherSpendingFamiliesMatchLdbounds) {
    const auto pocock = group_sequential_boundaries(looks,{.function=SpendingFunction::pocock});
    const std::vector<double> ld_pocock{2.36832770352390,2.36748521765386,2.35811194779948,2.34996485176195};
    for (std::size_t k = 0; k < 4; ++k)
        EXPECT_NEAR(pocock.z[k],ld_pocock[k],5e-4) << k;
    const auto power = group_sequential_boundaries(looks,{.function=SpendingFunction::power,.parameter=2});
    const std::vector<double> ld_power{2.95516684749783,2.55932126787437,2.30081405815768,2.09191342374891};
    for (std::size_t k = 0; k < 4; ++k)
        EXPECT_NEAR(power.z[k],ld_power[k],5e-4) << k;
    const auto hsd = group_sequential_boundaries({.3,.6,1},{.function=SpendingFunction::hwang_shih_decani,
                                                            .parameter=-4});
    const std::vector<double> ld_hsd{3.06669954932178,2.65495657535978,1.99208652025447};
    for (std::size_t k = 0; k < 3; ++k)
        EXPECT_NEAR(hsd.z[k],ld_hsd[k],5e-4) << k;
    // Each family spends exactly alpha in total.
    EXPECT_NEAR(boundary_crossing_probabilities({.3,.6,1},hsd.z,0).total,.025,1e-9);
}

TEST(GroupSequential, SpendingFunctionsAndSingleLook) {
    EXPECT_NEAR(alpha_spent(.5,{}),.00152532275798900,1e-15);
    EXPECT_NEAR(alpha_spent(.5,{.function=SpendingFunction::pocock}),.01550286267395694,1e-15);
    EXPECT_NEAR(alpha_spent(.4,{.function=SpendingFunction::hwang_shih_decani,.parameter=0}),.01,1e-15);
    EXPECT_EQ(alpha_spent(0,{}),0.0);
    EXPECT_NEAR(alpha_spent(2,{}),.025,1e-15);
    const auto one = group_sequential_boundaries({1},{});
    EXPECT_NEAR(one.z[0],1.959963984540054,1e-12);
    EXPECT_NEAR(boundary_crossing_probabilities({1},one.z,1.959963984540054+1.2815515655446004).total,.9,1e-12);
    // A look with an infinite boundary spends nothing there.
    const auto skip = boundary_crossing_probabilities({.5,1},{INFINITY,1.959963984540054},0);
    EXPECT_EQ(skip.stage[0],0.0);
    EXPECT_NEAR(skip.total,.025,1e-9);
    EXPECT_THROW((void)group_sequential_boundaries({.5,.5},{}),std::invalid_argument);
    EXPECT_THROW((void)group_sequential_boundaries({.5,1.2},{}),std::invalid_argument);
    EXPECT_THROW((void)alpha_spent(.5,{.function=SpendingFunction::power,.parameter=0}),std::invalid_argument);
    EXPECT_THROW((void)boundary_crossing_probabilities(looks,{1,2},0),std::invalid_argument);
    EXPECT_THROW((void)drift_for_power({1},one.z,.01),std::invalid_argument);
}

// Reference values: qnorm() in R 4.5.3. Spending boundaries need the quantile to full precision.
TEST(GroupSequential, NormalQuantileIsFullPrecision) {
    using datamunge::random::normal_quantile;
    EXPECT_NEAR(normal_quantile(.975),1.9599639845400536,2e-15);
    EXPECT_NEAR(normal_quantile(.75),.6744897501960817,1e-15);
    EXPECT_NEAR(normal_quantile(1e-10),-6.3613409024040557,5e-15);
    EXPECT_NEAR(normal_quantile(1-1e-6),4.7534243088170891,4e-11); // 1 - 1e-6 is itself rounded
    EXPECT_NEAR(normal_quantile(.02),-2.0537489106318225,2e-15);
}

} // namespace
