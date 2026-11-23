#include <datamunge/stats/mean_shift.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_mean_shift;

TEST(MeanShift, ModesAndNormalizedDensityMatchDirectRMaximization) {
    const DenseMatrix<double> x(7,1,{-2.85,-2.33,-2.1,-1.95,.7,1.3,1.65});
    const auto narrow = fit_mean_shift(x,.42);
    ASSERT_EQ(narrow.modes.rows(),2u);
    EXPECT_NEAR(narrow.modes(0,0),-2.20716866503565,4e-8);
    EXPECT_NEAR(narrow.modes(1,0),1.33544436529265,4e-8);
    EXPECT_NEAR(narrow.log_density[0],-.877256503426948,1e-13);
    EXPECT_NEAR(narrow.log_density[1],-1.26967493998172,1e-13);
    EXPECT_EQ(narrow.labels,(std::vector<std::size_t>{0,0,0,0,1,1,1}));
    EXPECT_TRUE(std::ranges::all_of(narrow.converged,[](auto x) { return x == 1; }));

    const auto broad = fit_mean_shift(x,4);
    ASSERT_EQ(broad.modes.rows(),1u);
    EXPECT_NEAR(broad.modes(0,0),-.85810957361105,8e-8);
    EXPECT_NEAR(broad.log_density[0],-2.40310880695787,1e-13);
    EXPECT_EQ(broad.labels,std::vector<std::size_t>(7,0));
}

TEST(MeanShift, EscapesAnExactlyStationaryValleyInTwoDimensions) {
    DenseMatrix<double> x(11,2);
    for (std::size_t i = 0; i < 5; ++i) {
        x(i,0) = -2;
        x(10-i,0) = 2;
    }
    const auto fit = fit_mean_shift(x,1);
    ASSERT_EQ(fit.modes.rows(),2u);
    EXPECT_NE(fit.labels[0],fit.labels[10]);
    EXPECT_EQ(fit.labels[5],fit.labels[10]);
    EXPECT_GT(fit.iterations[5],0u);
    EXPECT_EQ(fit.converged[5],1);
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(std::abs(fit.modes(i,0)),1.93902267654888,4e-8);
        EXPECT_DOUBLE_EQ(fit.modes(i,1),0);
        EXPECT_NEAR(fit.log_density[i],-1.67872012325679-.5*std::log(2*std::numbers::pi),1e-13);
    }
}

TEST(MeanShift, ReportsUnfinishedTrajectoriesAndHandlesDuplicateRows) {
    const DenseMatrix<double> x(3,1,{1,2,2});
    const auto limited = fit_mean_shift(x,.6,{.max_iterations=1,.tolerance=1e-15});
    EXPECT_EQ(limited.iterations[0],1u);
    EXPECT_EQ(limited.converged[0],0);
    EXPECT_EQ(limited.labels[1],limited.labels[2]);

    const auto repeated = fit_mean_shift(DenseMatrix<double>(3,2,7),1e-200);
    ASSERT_EQ(repeated.modes.rows(),1u);
    EXPECT_DOUBLE_EQ(repeated.modes(0,0),7);
    EXPECT_DOUBLE_EQ(repeated.modes(0,1),7);
    EXPECT_EQ(repeated.converged,(std::vector<std::uint8_t>(3,1)));
    EXPECT_TRUE(std::isfinite(repeated.log_density[0]));

    EXPECT_THROW((void)fit_mean_shift(x,0),std::invalid_argument);
    EXPECT_THROW((void)fit_mean_shift(x,.6,{.merge_tolerance=-1}),std::invalid_argument);
    const DenseMatrix<double> bad(1,1,std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW((void)fit_mean_shift(bad,.4),std::invalid_argument);
}
}
