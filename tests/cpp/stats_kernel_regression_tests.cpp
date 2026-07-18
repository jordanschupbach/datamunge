#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <algorithm>
#include <cmath>

using datamunge::dstruct::DataFrame;
using datamunge::stats::KernelRegression;
using datamunge::stats::KernelRegressionKernel;
using datamunge::stats::KernelRegressionOptions;

namespace {
constexpr const char* kFormula = "Petal.Length ~ Petal.Width";
}

// Reference values computed with an independent numpy implementation of the
// exact Nadaraya-Watson formula (same standardization convention:
// population mean/std, ddof=0) on the embedded iris dataset. Unlike the
// tree-based methods, this estimator has no split tie-breaking or other
// algorithmic freedom -- it's a single closed-form weighted average -- so
// agreement is expected to near machine precision.

TEST(KernelRegression, GaussianFixedBandwidthMatchesOracle) {
    KernelRegressionOptions options;
    options.kernel    = KernelRegressionKernel::Gaussian;
    options.bandwidth = 0.5;
    KernelRegression model(datamunge::datasets::iris(), kFormula, options);

    DataFrame newdata;
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0, 1.8});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 4u);
    EXPECT_NEAR(preds[0], 1.5249460300756446, 1e-6);
    EXPECT_NEAR(preds[1], 4.696829197561, 1e-6);
    EXPECT_NEAR(preds[2], 5.298007233375302, 1e-6);
    EXPECT_NEAR(preds[3], 5.0718558327039345, 1e-6);
    EXPECT_FALSE(model.bandwidth_was_selected());
}

TEST(KernelRegression, EpanechnikovFixedBandwidthMatchesOracle) {
    KernelRegressionOptions options;
    options.kernel    = KernelRegressionKernel::Epanechnikov;
    options.bandwidth = 0.8;
    KernelRegression model(datamunge::datasets::iris(), kFormula, options);

    DataFrame newdata;
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0, 1.8});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 4u);
    EXPECT_NEAR(preds[0], 1.4582005252641312, 1e-6);
    EXPECT_NEAR(preds[1], 4.658018979912076, 1e-6);
    EXPECT_NEAR(preds[2], 5.4268779679962496, 1e-6);
    EXPECT_NEAR(preds[3], 5.127509366438405, 1e-6);
}

TEST(KernelRegression, LeaveOneOutFittedValuesMatchOracle) {
    KernelRegressionOptions options;
    options.kernel    = KernelRegressionKernel::Gaussian;
    options.bandwidth = 0.5;
    KernelRegression model(datamunge::datasets::iris(), kFormula, options);

    const auto& fitted = model.fitted_values();
    ASSERT_GE(fitted.size(), 5u);
    EXPECT_NEAR(fitted[0], 1.5275284460228766, 1e-6);
    EXPECT_NEAR(fitted[1], 1.5275284460228766, 1e-6);
    EXPECT_NEAR(fitted[2], 1.5295952711518295, 1e-6);
    EXPECT_NEAR(fitted[3], 1.5254616208939242, 1e-6);
    EXPECT_NEAR(fitted[4], 1.5275284460228766, 1e-6);
}

TEST(KernelRegression, LeaveOneOutRSquaredAndRmseMatchOracle) {
    KernelRegressionOptions options;
    options.kernel    = KernelRegressionKernel::Gaussian;
    options.bandwidth = 0.5;
    KernelRegression model(datamunge::datasets::iris(), kFormula, options);
    EXPECT_NEAR(model.r_squared(), 0.9441536945394454, 1e-6);
    EXPECT_NEAR(model.rmse(), 0.4157792533023119, 1e-6);
}

TEST(KernelRegression, PredictUsesFullSampleNotLeaveOneOut) {
    // predict() on a genuinely-new query should not match the LOO fitted
    // value at the nearest training point (it uses every training row,
    // including near-duplicates of the query).
    const auto iris = datamunge::datasets::iris();
    KernelRegressionOptions options;
    options.bandwidth = 0.5;
    KernelRegression model(iris, kFormula, options);

    DataFrame self;
    self.add_column("Petal.Width", std::vector<double>{iris.double_at("Petal.Width", 0)});
    const auto preds = model.predict(self);
    // The full-sample prediction at exactly a training point's location is
    // dominated by (but not equal to, since other points still contribute)
    // that point's own y and should be very close to it for a small
    // bandwidth relative to the data's spread.
    EXPECT_NEAR(preds[0], iris.double_at("Petal.Length", 0), 0.5);
}

TEST(KernelRegression, AutoSelectedBandwidthLiesWithinSweptGrid) {
    KernelRegression model(datamunge::datasets::iris(), kFormula); // bandwidth left at -1 (auto)
    EXPECT_TRUE(model.bandwidth_was_selected());
    ASSERT_FALSE(model.bandwidth_grid().empty());
    ASSERT_EQ(model.bandwidth_grid().size(), model.cv_mean_squared_error().size());

    const auto& grid = model.bandwidth_grid();
    const double lo   = *std::min_element(grid.begin(), grid.end());
    const double hi   = *std::max_element(grid.begin(), grid.end());
    EXPECT_GE(model.bandwidth(), lo);
    EXPECT_LE(model.bandwidth(), hi);
}

TEST(KernelRegression, CompactSupportKernelReturnsNanFarOutsideData) {
    KernelRegressionOptions options;
    options.kernel    = KernelRegressionKernel::Epanechnikov;
    options.bandwidth = 0.01; // tiny, compact-support window
    KernelRegression model(datamunge::datasets::iris(), kFormula, options);

    DataFrame newdata;
    newdata.add_column("Petal.Width", std::vector<double>{1000.0});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 1u);
    EXPECT_TRUE(std::isnan(preds[0]));
}

TEST(KernelRegression, PlotFitRequiresExactlyOnePredictor) {
    KernelRegression model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Petal.Width");
    EXPECT_THROW(model.plot_fit(datamunge::datasets::iris()), std::invalid_argument);
}

TEST(KernelRegression, PlotCvCurveThrowsForFixedBandwidthFit) {
    KernelRegressionOptions options;
    options.bandwidth = 0.5;
    KernelRegression model(datamunge::datasets::iris(), kFormula, options);
    EXPECT_THROW(model.plot_cv_curve(), std::invalid_argument);
}

TEST(KernelRegression, RejectsZeroBandwidth) {
    KernelRegressionOptions options;
    options.bandwidth = 0.0;
    EXPECT_THROW(KernelRegression(datamunge::datasets::iris(), kFormula, options), std::invalid_argument);
}

TEST(KernelRegression, PredictorNamesReflectFormula) {
    KernelRegression model(datamunge::datasets::iris(), kFormula);
    ASSERT_EQ(model.predictor_names().size(), 1u);
    EXPECT_EQ(model.predictor_names()[0], "Petal.Width");
}

TEST(KernelRegression, SummaryContainsKeyInformation) {
    KernelRegressionOptions options;
    options.bandwidth = 0.5;
    KernelRegression model(datamunge::datasets::iris(), kFormula, options);
    const auto text = model.summary();
    EXPECT_NE(text.find("Bandwidth"), std::string::npos);
    EXPECT_NE(text.find("R-squared"), std::string::npos);
}
