#include <datamunge/stats/tweedie_regression.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::TweedieRegressionOptions;
using datamunge::stats::TweedieRegressionStatus;
using datamunge::stats::tweedie_regression;

TEST(TweedieRegression, ProfilesPowerAndDispersionWithWeightsAndOffsets) {
    const std::vector<double> y{
        .156679,3.619702,0,0,2.789960,0, 0,1.078499,2.336380,3.148337,0,.624602,
        .075665,.519148,.977899,1.989796,2.967952,.719423, 0,.515524,.829289,1.300996,0,1.340719,
        .729317,4.442495,4.120408,2.744605,6.978793,2.709877, 0,.541217,3.381231,4.310448,2.240227,1.192585,
        4.094531,2.203604,8.776602,4.847056,3.398759,0, .717664,.216298,9.212592,.035759,9.782466,2.963284};
    const double exposure[]{.6, 1., 1.5, 2., 1.3, .9}, weight[]{.8, 1., 1.1, 1.25, .9, 1.};
    DenseMatrix<double> x(y.size(), 2, 1);
    TweedieRegressionOptions options;
    for (std::size_t i = 0; i < y.size(); ++i) {
        x(i, 1) = -1 + 2. * (i / 6) / 7;
        options.offset.push_back(std::log(exposure[i % 6]));
        options.weights.push_back(weight[i % 6]);
    }
    const auto fit = tweedie_regression(x, y, options);
    ASSERT_EQ(fit.status, TweedieRegressionStatus::Converged);
    ASSERT_EQ(fit.coefficients.size(), 2u);
    // R dpois()/dgamma() mixture, jointly optimized over coefficients, log-phi and power.
    EXPECT_NEAR(fit.coefficients[0], .449010350391279, 3e-8);
    EXPECT_NEAR(fit.coefficients[1], .737961109610782, 3e-8);
    EXPECT_NEAR(fit.dispersion, std::exp(.326962446200602), 1e-7);
    EXPECT_NEAR(fit.power, 1.463626255504162, 6e-8);
    EXPECT_NEAR(fit.log_likelihood, -93.82921888249088, 1e-10);
    ASSERT_EQ(fit.covariance.rows(), 4u);
    EXPECT_NEAR(fit.covariance(0, 0), .0207783824024, 2e-8);
    EXPECT_NEAR(fit.covariance(2, 2), .0290881788777 * std::exp(2 * .326962446200602), 3e-7);
    EXPECT_NEAR(fit.covariance(2, 3), .00436606312620 * std::exp(.326962446200602), 1e-7);
    EXPECT_NEAR(fit.covariance(3, 3), .00507446151693, 2e-8);
    const auto best = std::ranges::max_element(fit.profile, {}, &datamunge::stats::TweedieProfilePoint::log_likelihood);
    ASSERT_NE(best, fit.profile.end());
    EXPECT_DOUBLE_EQ(best->power, fit.power);
    EXPECT_TRUE(std::ranges::is_sorted(fit.profile, {}, &datamunge::stats::TweedieProfilePoint::power));
    const auto predicted = fit.predict(DenseMatrix<double>(2, 2, {1, .4, 1, .4}), {0, std::log(2.)});
    EXPECT_NEAR(predicted[1], 2 * predicted[0], 1e-14);

    options.minimum_power = 1.25;
    options.maximum_power = 1.4;
    const auto bounded = tweedie_regression(x, y, options);
    EXPECT_EQ(bounded.status, TweedieRegressionStatus::PowerBoundary);
    EXPECT_DOUBLE_EQ(bounded.power, options.maximum_power);
    EXPECT_EQ(bounded.covariance.rows(), 3u);
    EXPECT_LT(bounded.log_likelihood, fit.log_likelihood);
}

TEST(TweedieRegression, FitsKnownPowerAndUsesCaseWeightsForTheMean) {
    const DenseMatrix<double> intercept(5, 1, 1);
    const auto fit = tweedie_regression(intercept, {0,.5,1,2,4},
        {.minimum_power = 1.5, .maximum_power = 1.5, .weights = {1,2,1,1,1}});
    ASSERT_EQ(fit.status, TweedieRegressionStatus::Converged);
    EXPECT_DOUBLE_EQ(fit.power, 1.5);
    EXPECT_NEAR(fit.fitted_mean[0], 8. / 6, 1e-10);
    EXPECT_EQ(fit.covariance.rows(), 2u);
    EXPECT_EQ(fit.profile.size(), 1u);
}

TEST(TweedieRegression, ComparesNearbyDispersionPeaksAtLowPower) {
    const std::vector<double> y{
        .6653733188,.8767824811,1.3946619764,1.6616132781,2.2419938923,2.7517235235,
        1.1934833731,2.0056682282,1.0714280460,1.5932128766,1.1175783873,.9877026153,
        2.0465017704,1.6513356595,.3221771570,1.6538760611,.5380969269,.5255240939};
    const std::vector<double> w{
        .6799287562,.6879920951,1.3998915135,1.4704514530,1.2369193153,.6415428340,
        1.1754261040,.6194826822,1.0855959678,1.1368041300,1.0273385064,1.3684479251,
        1.4089878065,1.1662864288,.9638032277,1.0392866149,.7254360990,.7318449139};
    const DenseMatrix<double> intercept(y.size(), 1, 1);
    const auto fit = tweedie_regression(intercept, y,
        {.minimum_power = 1.08, .maximum_power = 1.08, .weights = w});
    ASSERT_EQ(fit.status, TweedieRegressionStatus::Converged);
    // The log likelihood at the competing maximum near phi = .2134 is lower by .05148.
    EXPECT_NEAR(fit.dispersion, .31248393665, 1e-7);
    EXPECT_NEAR(fit.log_likelihood, -16.4796519738, 3e-9);
}

TEST(TweedieRegression, RejectsInvalidSupportPowerAndRank) {
    const DenseMatrix<double> intercept(4, 1, 1), aliased(4, 2, 1);
    EXPECT_THROW((void)tweedie_regression(intercept, {0,-.5,1,2}), std::invalid_argument);
    EXPECT_THROW((void)tweedie_regression(intercept, {0,0,0,0}), std::invalid_argument);
    EXPECT_THROW((void)tweedie_regression(aliased, {0,.5,1,2}), std::invalid_argument);
    EXPECT_THROW((void)tweedie_regression(intercept, {0,.5,1,2}, {.maximum_power = 2}), std::invalid_argument);
}
}
