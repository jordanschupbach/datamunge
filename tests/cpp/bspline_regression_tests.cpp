#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>

#include <cmath>
#include <stdexcept>

namespace dm = datamunge;

TEST(BSplineRegression, ReproducesACubicExactly) {
    // A cubic is in the span of any cubic B-spline basis, so least squares recovers it exactly.
    dm::Vector x = dm::seql(0.0, 2.0, 50);
    dm::Vector y = 1.0 - 2.0 * x + 0.5 * dm::pow(x, 3.0);
    dm::BSR fit(x, y, 8);
    EXPECT_EQ(fit.basis_size(), 8u);
    EXPECT_EQ(fit.degree(), 3u);
    EXPECT_NEAR(fit.r_squared(), 1.0, 1e-10);

    dm::Vector grid = dm::seql(0.0, 2.0, 17);
    dm::Vector truth = 1.0 - 2.0 * grid + 0.5 * dm::pow(grid, 3.0);
    dm::Vector pred = fit.predict(grid);
    for (std::size_t i = 0; i < grid.size(); ++i) EXPECT_NEAR(pred[i], truth[i], 1e-8);
}

TEST(BSplineRegression, TracksNoisySmoothSignal) {
    dm::set_seed(7);
    dm::Vector x = dm::unif(0.0, 1.0, 2000);
    dm::Vector y = dm::sin(6.0 * x) + dm::rnorm(0.0, 0.1, 2000);
    dm::BSR fit(x, y, 12);
    dm::Vector grid = dm::seql(0.05, 0.95, 50);
    dm::Vector err = dm::abs(fit.predict(grid) - dm::sin(6.0 * grid));
    for (double e : err) EXPECT_LT(e, 0.05);
    EXPECT_EQ(fit.residuals().size(), 2000u);
}

TEST(BSplineRegression, PredictClampsOutsideDomain) {
    dm::Vector x = {0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7};
    dm::Vector y = 2.0 * x;
    dm::BSR fit(x, y, 4);
    dm::Vector pred = fit.predict({0.0, 0.2, 0.8, 1.0});
    EXPECT_NEAR(pred[0], pred[1], 1e-12);
    EXPECT_NEAR(pred[2], pred[3], 1e-12);
    EXPECT_NEAR(pred[1], 0.4, 1e-10);
}

TEST(BSplineRegression, RejectsBadInputs) {
    dm::Vector x = {0.0, 0.5, 1.0};
    EXPECT_THROW(dm::BSR(x, dm::Vector{1.0, 2.0}, 4), std::invalid_argument);
    EXPECT_THROW(dm::BSR(x, x, 3), std::invalid_argument);  // < degree + 1
    // More basis functions than data can support -> singular normal equations.
    EXPECT_THROW(dm::BSR(x, x, 20), std::runtime_error);
}

TEST(BSplineRegression, KnotConstructor) {
    // Interior knots at 0.25/0.5/0.75 (+ boundaries from the data) -> 3 + 3 + 1 basis functions.
    dm::Vector x = dm::seql(0.0, 1.0, 200);
    dm::Vector y = dm::pow(x, 3.0) - x;
    dm::BSplineRegression fit = dm::BSR(x, y, dm::Vector{0.25, 0.75, 0.5, 0.5});
    EXPECT_EQ(fit.basis_size(), 7u);
    EXPECT_NEAR(fit.r_squared(), 1.0, 1e-10);
    EXPECT_NEAR(fit.predict({0.6})[0], 0.216 - 0.6, 1e-8);

    // Knots that include the boundaries (e.g. a grid) are treated the same way.
    dm::BSR grid_fit(x, y, dm::seql(0.0, 1.0, 5));
    EXPECT_EQ(grid_fit.basis_size(), 7u);
    EXPECT_EQ(grid_fit.knots().size(), 11u);
}
