#include <datamunge/stats/arellano_bond.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::ArellanoBondRegressor;
using datamunge::stats::ArellanoBondStep;
using datamunge::stats::fit_arellano_bond;

struct Observations {
    std::vector<double> y;
    std::vector<double> predictors;
    std::vector<std::int64_t> unit;
    std::vector<std::int64_t> time;
};

Observations sample(int units) {
    Observations data;
    std::uint32_t state = 928147;
    const auto uniform = [&] {
        state = 1664525u*state + 1013904223u;
        return std::ldexp(static_cast<double>(state), -32) - .5;
    };
    for (int s = 1; s <= units; ++s) {
        double y = std::sin(.37*s), previous_error = 0, predetermined = 0, endogenous = 0;
        for (int t = 0; t < 8; ++t) {
            const auto first = uniform();
            const auto error = first + uniform();
            const auto strict = uniform() + .4*std::sin(.8*t);
            predetermined = .35*predetermined + .4*previous_error + uniform();
            endogenous = .4*endogenous + .6*error + uniform();
            y = .62*y + .27*strict - .4*predetermined + .32*endogenous + error + .7*std::cos(.23*s) + .12*std::sin(t);
            previous_error = error;
            if ((s % 6 == 0 && t == 3) || (s % 5 == 0 && t == 0))
                continue;
            data.y.push_back(y);
            data.unit.push_back(13*s-340);
            data.time.push_back(t-3);
            data.predictors.insert(data.predictors.end(), {strict,predetermined,endogenous});
        }
    }
    return data;
}

TEST(ArellanoBond, MixedRegressorTimingAndTwoStepsMatchIndependentDenseR) {
    const auto data = sample(48);
    const DenseMatrix<double> x(data.y.size(),3,data.predictors);
    const std::vector<ArellanoBondRegressor> types{ArellanoBondRegressor::StrictlyExogenous,
        ArellanoBondRegressor::Predetermined, ArellanoBondRegressor::Endogenous};
    const double slopes[2][4]{
        {.173077855914634,.198928505175463,-.106698939760766,1.008962199423933},
        {.272346419811502,.236704845891978,-.197444952658706,.732334058494782}};
    const double covariance[2][3]{
        {.089231689099977,-.165046819558827,.349359551425370},
        {.061479960092415,-.108788751457873,.247638116476668}};
    for (const bool two : {false,true}) {
        const auto fit = fit_arellano_bond(x,data.y,data.unit,data.time,{.step=two ? ArellanoBondStep::Two : ArellanoBondStep::One,
            .time_effects=true,.maximum_lag=3,.regressor_types=types});
        EXPECT_EQ(fit.units,48u);
        EXPECT_EQ(fit.equation_rows.size(),255u);
        EXPECT_EQ(fit.instrument_columns,14u);
        EXPECT_EQ(fit.instrument_rank,14u);
        EXPECT_EQ(fit.time_levels,(std::vector<std::int64_t>{-1,0,1,2,3,4}));
        ASSERT_EQ(fit.coefficients.size(),10u);
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(fit.coefficients[j],slopes[two][j],4e-12);
        EXPECT_NEAR(fit.covariance(0,0),covariance[two][0],4e-12);
        EXPECT_NEAR(fit.covariance(0,3),covariance[two][1],4e-12);
        EXPECT_NEAR(fit.covariance(3,3),covariance[two][2],4e-12);
        ASSERT_EQ(fit.serial_correlation.size(),2u);
        EXPECT_EQ(fit.serial_correlation[0].pairs,200u);
        EXPECT_EQ(fit.serial_correlation[1].pairs,152u);
        EXPECT_NEAR(fit.serial_correlation[0].statistic,-1.54536912704112,4e-12);
        EXPECT_NEAR(fit.serial_correlation[1].statistic,-.66843913055472,4e-12);
        EXPECT_NEAR(fit.serial_correlation[1].p_value,.50385332437293,4e-12);
        EXPECT_NEAR(fit.fitted[0]+fit.residuals[0],data.y[2]-data.y[1],1e-13);
        EXPECT_EQ(fit.equation_rows.front(),2u);
        if (two) {
            ASSERT_TRUE(fit.hansen);
            EXPECT_EQ(fit.hansen->df,4u);
            EXPECT_NEAR(fit.hansen->statistic,3.20494132828376,4e-12);
            EXPECT_NEAR(fit.hansen->p_value,.52413320713273,4e-12);
        } else {
            EXPECT_FALSE(fit.hansen);
        }
    }
}

TEST(ArellanoBond, UncollapsedInstrumentsRespectPeriodLabelsAndOriginalRowOrder) {
    auto data = sample(48);
    std::ranges::reverse(data.y);
    std::ranges::reverse(data.unit);
    std::ranges::reverse(data.time);
    const auto fit = fit_arellano_bond(data.y,data.unit,data.time,
        {.step=ArellanoBondStep::Two,.collapse=false,.maximum_lag=3});
    EXPECT_EQ(fit.instrument_columns,11u);
    EXPECT_EQ(fit.instrument_rank,11u);
    EXPECT_EQ(fit.equation_rows.front(),0u);
    EXPECT_TRUE(std::ranges::is_sorted(fit.equation_rows));
    EXPECT_NEAR(fit.coefficients[0],.129688033174505,4e-12);
    EXPECT_NEAR(fit.covariance(0,0),.0759420300213,4e-12);
    ASSERT_EQ(fit.serial_correlation.size(),2u);
    EXPECT_NEAR(fit.serial_correlation[0].statistic,-1.08616635243376,4e-12);
    EXPECT_NEAR(fit.serial_correlation[1].statistic,-1.48359422294830,4e-12);
    ASSERT_TRUE(fit.hansen);
    EXPECT_NEAR(fit.hansen->statistic,4.14262594358150,4e-12);
}

TEST(ArellanoBond, UsesEarlierLevelInstrumentsAcrossGapsButCountsActualResidualLags) {
    const auto data = sample(8);
    Observations gapped;
    for (std::size_t i = 0; i < data.y.size(); ++i) {
        if (data.time[i] == 0 || data.time[i] == 1)
            continue;
        gapped.y.push_back(data.y[i]);
        gapped.time.push_back(data.time[i]);
        gapped.unit.push_back(data.unit[i]);
    }
    const auto fit = fit_arellano_bond(gapped.y,gapped.unit,gapped.time,{.max_serial_order=5});
    EXPECT_EQ(fit.instrument_columns,4u);
    ASSERT_EQ(fit.serial_correlation.size(),1u);
    EXPECT_EQ(fit.serial_correlation[0].lag,5u);
    EXPECT_EQ(fit.serial_correlation[0].pairs,7u);
}

TEST(ArellanoBond, RejectsDuplicateRowsInvalidBoundsAndSingularSecondStepWeights) {
    const auto data = sample(4);
    EXPECT_THROW((void)fit_arellano_bond(data.y,data.unit,data.time,{.minimum_outcome_lag=1}), std::invalid_argument);
    EXPECT_THROW((void)fit_arellano_bond(data.y,data.unit,data.time,
        {.step=ArellanoBondStep::Two,.collapse=false,.maximum_lag=3}), std::domain_error);
    auto duplicate = data.time;
    duplicate[1] = duplicate[0];
    EXPECT_THROW((void)fit_arellano_bond(data.y,data.unit,duplicate), std::invalid_argument);
    const auto fit = fit_arellano_bond(data.y,data.unit,data.time,{.maximum_lag=2,.max_serial_order=0});
    EXPECT_EQ(fit.instrument_rank,1u);
    EXPECT_TRUE(fit.serial_correlation.empty());
}
} // namespace
