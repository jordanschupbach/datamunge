#include <datamunge/stats/repeated_measures_anova.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::repeated_measures_anova;

TEST(RepeatedMeasuresANOVA, OmnibusAndAdjustedInteractionMatchRSphericalTest) {
    DenseMatrix<double> y(18,4), full(18,3,1), reduced(18,2,1);
    for (std::size_t row = 0; row < y.rows(); ++row) {
        const double i = row + 1;
        const double g = i <= 9 ? -1 : 1;
        const auto z = std::sin(i*.41), s = std::sin(i*.22);
        reduced(row,1) = full(row,1) = z;
        full(row,2) = g;
        y(row,0) = 5 + .9*s + .42*std::cos(2*i);
        y(row,1) = 6 + .8*s + .31*std::cos(3*i) + .33*z;
        y(row,2) = 6 + .7*s + 1.3*std::cos(.9*i) + .6*g + .41*z;
        y(row,3) = 7 + .6*s + 2.1*std::cos(.73*i) + .8*g + .25*z;
    }

    // Base R anova.mlm(..., test = "Spherical") and its residual epsilon calculation.
    const auto omnibus = repeated_measures_anova(y);
    EXPECT_NEAR(omnibus.f_statistic, 11.231998566098, 2e-12);
    EXPECT_DOUBLE_EQ(omnibus.uncorrected.df_numerator, 3);
    EXPECT_DOUBLE_EQ(omnibus.uncorrected.df_denominator, 51);
    EXPECT_NEAR(omnibus.uncorrected.p_value, 9.043643377676e-6, 1e-15);
    EXPECT_NEAR(omnibus.greenhouse_geisser.epsilon, .53926551060036, 2e-12);
    EXPECT_NEAR(omnibus.greenhouse_geisser.p_value, .00057163251131035, 1e-13);
    ASSERT_TRUE(omnibus.huynh_feldt);
    EXPECT_NEAR(omnibus.huynh_feldt->epsilon, .58769945039599, 2e-12);
    EXPECT_NEAR(omnibus.huynh_feldt->p_value, .0003679371926704, 1e-13);

    // A nonorthogonal basis of a two-dimensional contrast subspace.
    const DenseMatrix<double> contrasts(2,4,{1,-1,1,-1, 9,-9,5,-5});
    const auto interaction = repeated_measures_anova(y,full,reduced,contrasts);
    EXPECT_NEAR(interaction.f_statistic, 1.4636969759611, 2e-12);
    EXPECT_DOUBLE_EQ(interaction.uncorrected.df_numerator, 2);
    EXPECT_DOUBLE_EQ(interaction.uncorrected.df_denominator, 30);
    EXPECT_NEAR(interaction.uncorrected.p_value, .24743345828172, 1e-12);
    EXPECT_NEAR(interaction.greenhouse_geisser.epsilon, .54365572923489, 2e-12);
    EXPECT_NEAR(interaction.greenhouse_geisser.p_value, .24662356714964, 1e-12);
    ASSERT_TRUE(interaction.huynh_feldt);
    EXPECT_NEAR(interaction.huynh_feldt->epsilon, .55334320500152, 2e-12);
    EXPECT_NEAR(interaction.huynh_feldt->p_value, .24691017084222, 1e-12);

    // Large subject offsets and a rescaling of the within-subject differences.
    auto transformed = y;
    for (std::size_t i = 0; i < y.rows(); ++i)
        for (std::size_t j = 0; j < y.cols(); ++j)
            transformed(i,j) = 3000.0*i + 11*y(i,j);
    const auto shifted = repeated_measures_anova(transformed,full,reduced,contrasts);
    EXPECT_NEAR(shifted.f_statistic, interaction.f_statistic, 2e-11);
    EXPECT_NEAR(shifted.greenhouse_geisser.epsilon, interaction.greenhouse_geisser.epsilon, 2e-12);
}

TEST(RepeatedMeasuresANOVA, SingleContrastIgnoresUnrelatedVisitsAndNeedsNoCorrection) {
    const DenseMatrix<double> y(4,3,{1e100,2,3, 1e100,1,4, 1e100,3,5, 1e100,2,4});
    const auto test = repeated_measures_anova(y,DenseMatrix<double>(4,1,1),DenseMatrix<double>(4,0),
        DenseMatrix<double>(1,3,{0,-1e-100,1e-100}));
    EXPECT_DOUBLE_EQ(test.f_statistic, 24);
    EXPECT_DOUBLE_EQ(test.greenhouse_geisser.epsilon, 1);
    ASSERT_TRUE(test.huynh_feldt);
    EXPECT_DOUBLE_EQ(test.huynh_feldt->p_value, test.uncorrected.p_value);
}

TEST(RepeatedMeasuresANOVA, HandlesDefinedAndUndefinedEpsilonLimits) {
    const DenseMatrix<double> minimal(2,3,{0,2,3,1,7,2});
    const auto test = repeated_measures_anova(minimal);
    EXPECT_DOUBLE_EQ(test.greenhouse_geisser.epsilon, .5);
    EXPECT_FALSE(test.huynh_feldt);
    const DenseMatrix<double> spherical(4,3,{0,1,1,1,0,1,1,1,0,0,0,0});
    const auto clipped = repeated_measures_anova(spherical);
    EXPECT_DOUBLE_EQ(clipped.greenhouse_geisser.epsilon,1);
    ASSERT_TRUE(clipped.huynh_feldt);
    EXPECT_DOUBLE_EQ(clipped.huynh_feldt->epsilon,1);
    EXPECT_DOUBLE_EQ(clipped.uncorrected.p_value,1);
}

TEST(RepeatedMeasuresANOVA, RejectsIncompleteDataNonNestedModelsAndDegenerateContrasts) {
    const DenseMatrix<double> y(4,3,{2,3,4,1,4,2,3,5,2,2,4,7});
    const DenseMatrix<double> full(4,2,{1,0,1,1,1,2,1,3}), wrong(4,1,{1,0,0,0});
    const DenseMatrix<double> contrasts(1,3,{-1,1,0}), intercept(4,1,1);
    EXPECT_THROW((void)repeated_measures_anova(y,full,wrong,contrasts), std::invalid_argument);
    EXPECT_THROW((void)repeated_measures_anova(y,full,intercept,DenseMatrix<double>(1,3,{1,0,0})), std::invalid_argument);
    EXPECT_THROW((void)repeated_measures_anova(y,full,intercept,DenseMatrix<double>(2,3,{-1,1,0,2,-2,0})),
        std::invalid_argument);
    EXPECT_THROW((void)repeated_measures_anova(DenseMatrix<double>(4,2,{1,3,2,4,3,5,4,6})), std::invalid_argument);
    auto incomplete = y;
    incomplete(2,2) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)repeated_measures_anova(incomplete), std::invalid_argument);
}
} // namespace
