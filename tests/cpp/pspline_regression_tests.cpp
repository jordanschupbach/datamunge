#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>

#include <cmath>
#include <stdexcept>

namespace dm = datamunge;

TEST(PSplineRegression, SecondOrderPenaltyLeavesLinesUnpenalized) {
    // A second-order difference penalty vanishes on lines, so any lambda reproduces one exactly.
    dm::Vector x = dm::seql(0.0, 2.0, 60);
    dm::Vector y = 1.5 - 0.75 * x;
    dm::PSR fit(x, y, 20, 1e4);
    EXPECT_EQ(fit.basis_size(), 20u);
    EXPECT_EQ(fit.degree(), 3u);
    EXPECT_EQ(fit.penalty_order(), 2u);
    EXPECT_DOUBLE_EQ(fit.lambda(), 1e4);
    EXPECT_NEAR(fit.r_squared(), 1.0, 1e-10);
    EXPECT_NEAR(fit.predict({1.25})[0], 1.5 - 0.75 * 1.25, 1e-8);
}

TEST(PSplineRegression, LargerLambdaMeansFewerEffectiveDf) {
    dm::set_seed(11);
    dm::Vector x = dm::unif(0.0, 1.0, 300);
    dm::Vector y = dm::sin(6.0 * x) + dm::rnorm(0.0, 0.2, 300);
    dm::PSR rough(x, y, 40, 1e-3);
    dm::PSR smooth(x, y, 40, 1e3);
    EXPECT_GT(rough.effective_df(), smooth.effective_df());
    EXPECT_LT(smooth.effective_df(), 10.0);
    EXPECT_GT(smooth.effective_df(), 2.0);  // never below the unpenalized line
}

TEST(PSplineRegression, GcvTracksNoisySignalWithRichBasis) {
    // More basis functions than BSplineRegression could support on this data: the penalty keeps it
    // well posed and GCV picks a lambda that is no worse than any of a few fixed choices.
    dm::set_seed(7);
    dm::Vector x = dm::unif(0.0, 1.0, 400);
    dm::Vector y = dm::sin(6.0 * x) + dm::rnorm(0.0, 0.1, 400);
    dm::PSR fit(x, y, 80);
    EXPECT_GT(fit.lambda(), 0.0);
    for (double lambda : {1e-4, 1e-1, 1e2, 1e5}) {
        EXPECT_LE(fit.gcv(), dm::PSR(x, y, 80, lambda).gcv() * (1.0 + 1e-9));
    }
    dm::Vector grid = dm::seql(0.05, 0.95, 50);
    dm::Vector err = dm::abs(fit.predict(grid) - dm::sin(6.0 * grid));
    for (double e : err) EXPECT_LT(e, 0.06);
    EXPECT_EQ(fit.fitted().size(), 400u);
}

TEST(PSplineRegression, PredictClampsOutsideDomain) {
    dm::Vector x = {0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7};
    dm::Vector y = 2.0 * x;
    dm::PSR fit(x, y, 6, 1.0);
    dm::Vector pred = fit.predict({0.0, 0.2, 0.8, 1.0});
    EXPECT_NEAR(pred[0], pred[1], 1e-12);
    EXPECT_NEAR(pred[2], pred[3], 1e-12);
    EXPECT_NEAR(pred[1], 0.4, 1e-8);
}

TEST(PSplineRegression, KnotConstructor) {
    dm::Vector x = dm::seql(0.0, 1.0, 200);
    dm::Vector y = 3.0 * x - 1.0;
    dm::PSR fit(x, y, dm::seql(0.0, 1.0, 11));
    EXPECT_EQ(fit.basis_size(), 13u);  // 9 interior knots + degree + 1
    EXPECT_NEAR(fit.predict({0.6})[0], 0.8, 1e-6);
    dm::PSR fixed(x, y, dm::seql(0.0, 1.0, 11), 2.0, 2, 1);
    EXPECT_EQ(fixed.degree(), 2u);
    EXPECT_EQ(fixed.penalty_order(), 1u);
}

TEST(PSplineRegression, RejectsBadInputs) {
    dm::Vector x = {0.0, 0.5, 1.0};
    EXPECT_THROW(dm::PSR(x, dm::Vector{1.0, 2.0}, 6), std::invalid_argument);
    EXPECT_THROW(dm::PSR(x, x, 3), std::invalid_argument);         // < degree + 1
    EXPECT_THROW(dm::PSR(x, x, 6, -1.0), std::invalid_argument);   // negative lambda
    EXPECT_THROW(dm::PSR(x, x, 2, 1.0, 1, 2), std::invalid_argument);  // basis <= penalty order
    EXPECT_NO_THROW(dm::PSR(x, x, 20, 1.0));  // rich basis on 3 points is fine when penalized
}
