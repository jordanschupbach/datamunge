#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <algorithm>
#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GaussianProcessRegression;
using datamunge::stats::GaussianProcessRegressionOptions;

namespace {
constexpr const char* kFormula = "Petal.Length ~ Petal.Width";
}

// Reference values computed with an independent numpy implementation of
// exact GP regression (RBF kernel, Cholesky-based, same standardization
// convention -- population mean/std, ddof=0 -- for both predictors and
// response, signal variance profiled analytically, virtual leave-one-out
// via Rasmussen & Williams eq. 5.12) on the embedded iris dataset. This is
// closed-form linear algebra with no tie-breaking ambiguity, so agreement
// is expected to near machine precision.

TEST(GaussianProcessRegression, FixedHyperparametersMatchOracle) {
    GaussianProcessRegressionOptions options;
    options.length_scale = 0.5;
    options.noise_ratio   = 0.05;
    GaussianProcessRegression model(datamunge::datasets::iris(), kFormula, options);

    EXPECT_NEAR(model.signal_variance(), 0.9108624311214876, 1e-6);
    EXPECT_NEAR(model.noise_variance(), 0.04554312155607438, 1e-6);
    EXPECT_NEAR(model.log_marginal_likelihood(), -1.1850587093879028, 1e-5);
    EXPECT_FALSE(model.length_scale_was_selected());
    EXPECT_FALSE(model.noise_ratio_was_selected());
}

TEST(GaussianProcessRegression, PredictionsAndStandardErrorsMatchOracle) {
    GaussianProcessRegressionOptions options;
    options.length_scale = 0.5;
    options.noise_ratio   = 0.05;
    GaussianProcessRegression model(datamunge::datasets::iris(), kFormula, options);

    DataFrame newdata;
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0, 1.8});
    const auto detail = model.predict_detail(newdata);
    ASSERT_EQ(detail.fit.size(), 4u);

    const double expected_mean[4] = {1.4321521650742546, 4.69690444280166, 5.604043155717242, 5.2676965935375835};
    const double expected_se[4]   = {0.06090310301372005, 0.07671561402418392, 0.08813273609365545,
                                    0.08193300972772583};
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(detail.fit[i], expected_mean[i], 1e-6);
        EXPECT_NEAR(detail.se_fit[i], expected_se[i], 1e-6);
        // A Gaussian 95% interval is +/- ~1.96 standard errors.
        EXPECT_NEAR(detail.upper[i] - detail.fit[i], 1.959963985 * detail.se_fit[i], 1e-6);
        EXPECT_NEAR(detail.fit[i] - detail.lower[i], 1.959963985 * detail.se_fit[i], 1e-6);
    }
}

TEST(GaussianProcessRegression, LeaveOneOutFittedValuesMatchOracle) {
    GaussianProcessRegressionOptions options;
    options.length_scale = 0.5;
    options.noise_ratio   = 0.05;
    GaussianProcessRegression model(datamunge::datasets::iris(), kFormula, options);

    const auto& fitted = model.fitted_values();
    ASSERT_GE(fitted.size(), 5u);
    EXPECT_NEAR(fitted[0], 1.43302095422355, 1e-6);
    EXPECT_NEAR(fitted[1], 1.4330209542235495, 1e-6);
    EXPECT_NEAR(fitted[2], 1.435723071319838, 1e-6);
    EXPECT_NEAR(fitted[3], 1.4303188371272602, 1e-6);
    EXPECT_NEAR(fitted[4], 1.4330209542235495, 1e-6);
}

TEST(GaussianProcessRegression, LeaveOneOutRSquaredAndRmseMatchOracle) {
    GaussianProcessRegressionOptions options;
    options.length_scale = 0.5;
    options.noise_ratio   = 0.05;
    GaussianProcessRegression model(datamunge::datasets::iris(), kFormula, options);
    EXPECT_NEAR(model.r_squared(), 0.9499784630617563, 1e-6);
    EXPECT_NEAR(model.rmse(), 0.3934994294571452, 1e-6);
}

TEST(GaussianProcessRegression, PredictiveVarianceShrinksNearTrainingPointsAndGrowsFarAway) {
    const auto iris = datamunge::datasets::iris();
    GaussianProcessRegressionOptions options;
    options.length_scale = 0.5;
    options.noise_ratio   = 0.01; // low noise -> near-interpolation at training points
    GaussianProcessRegression model(iris, kFormula, options);

    DataFrame near;
    near.add_column("Petal.Width", std::vector<double>{iris.double_at("Petal.Width", 0)});
    DataFrame far;
    far.add_column("Petal.Width", std::vector<double>{50.0});      // well outside the training range
    DataFrame farther;
    farther.add_column("Petal.Width", std::vector<double>{500.0}); // even further outside

    const auto near_detail    = model.predict_detail(near);
    const auto far_detail     = model.predict_detail(far);
    const auto farther_detail = model.predict_detail(farther);
    EXPECT_LT(near_detail.se_fit[0], far_detail.se_fit[0]);
    // Far from any training data, the kernel between the query and every training point vanishes,
    // so predictive variance saturates at the (profiled, original-scale) signal variance -- it
    // should stop growing well before 50 standardized units away.
    EXPECT_NEAR(far_detail.se_fit[0], farther_detail.se_fit[0], 1e-6);
}

TEST(GaussianProcessRegression, AutoSelectedLengthScaleLiesWithinSweptGrid) {
    GaussianProcessRegression model(datamunge::datasets::iris(), kFormula); // both hyperparameters auto
    EXPECT_TRUE(model.length_scale_was_selected());
    EXPECT_TRUE(model.noise_ratio_was_selected());
    ASSERT_FALSE(model.length_scale_grid().empty());
    ASSERT_EQ(model.length_scale_grid().size(), model.length_scale_profile_log_likelihood().size());

    const auto& grid = model.length_scale_grid();
    const double lo   = *std::min_element(grid.begin(), grid.end());
    const double hi   = *std::max_element(grid.begin(), grid.end());
    EXPECT_GE(model.length_scale(), lo);
    EXPECT_LE(model.length_scale(), hi);
}

TEST(GaussianProcessRegression, PredictorNamesReflectFormula) {
    GaussianProcessRegression model(datamunge::datasets::iris(), kFormula);
    ASSERT_EQ(model.predictor_names().size(), 1u);
    EXPECT_EQ(model.predictor_names()[0], "Petal.Width");
}

TEST(GaussianProcessRegression, PlotFitRequiresExactlyOnePredictor) {
    GaussianProcessRegression model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Petal.Width");
    EXPECT_THROW(model.plot_fit(datamunge::datasets::iris()), std::invalid_argument);
}

TEST(GaussianProcessRegression, PlotLengthScaleProfileThrowsWhenFixed) {
    GaussianProcessRegressionOptions options;
    options.length_scale = 0.5;
    GaussianProcessRegression model(datamunge::datasets::iris(), kFormula, options);
    EXPECT_THROW(model.plot_length_scale_profile(), std::invalid_argument);
}

TEST(GaussianProcessRegression, RejectsZeroLengthScale) {
    GaussianProcessRegressionOptions options;
    options.length_scale = 0.0;
    EXPECT_THROW(GaussianProcessRegression(datamunge::datasets::iris(), kFormula, options), std::invalid_argument);
}

TEST(GaussianProcessRegression, SummaryContainsKeyInformation) {
    GaussianProcessRegressionOptions options;
    options.length_scale = 0.5;
    options.noise_ratio   = 0.05;
    GaussianProcessRegression model(datamunge::datasets::iris(), kFormula, options);
    const auto text = model.summary();
    EXPECT_NE(text.find("Length scale"), std::string::npos);
    EXPECT_NE(text.find("Log marginal likelihood"), std::string::npos);
    EXPECT_NE(text.find("R-squared"), std::string::npos);
}
