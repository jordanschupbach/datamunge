#include <datamunge/stats/factor_analysis.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::ObliqueRotation;
using datamunge::stats::oblimin;
using datamunge::stats::promax;

DenseMatrix<double> fixture() {
    return DenseMatrix<double>(9,3,{
        .78,.21,.03,
        .68,.30,.08,
        .73,.26,.13,
        .32,.74,.11,
        .25,.81,.17,
        .36,.62,.20,
        .12,.23,.77,
        .07,.13,.68,
        .17,.18,.81
    });
}

void expect_invariant(const DenseMatrix<double>& initial, const ObliqueRotation& rotated) {
    const auto common = initial*initial.transpose();
    const auto structure = rotated.pattern*rotated.factor_correlation;
    const auto reconstructed = structure*rotated.pattern.transpose();
    for (std::size_t i = 0; i < rotated.factor_correlation.rows(); ++i)
        EXPECT_DOUBLE_EQ(rotated.factor_correlation(i,i),1);
    for (std::size_t i = 0; i < common.rows(); ++i)
        for (std::size_t j = 0; j < common.cols(); ++j)
            EXPECT_NEAR(common(i,j),reconstructed(i,j),2e-13);
    for (std::size_t i = 0; i < structure.data().size(); ++i)
        EXPECT_NEAR(structure.data()[i],rotated.structure.data()[i],2e-14);
}

void expect_reference(const ObliqueRotation& rotated, const std::array<double,3>& main_loadings,
    const std::array<double,3>& correlations, double tolerance) {
    const std::array<std::size_t,3> row{0,4,8};
    std::array<std::size_t,3> column{};
    std::array<double,3> sign{};
    for (std::size_t i = 0; i < row.size(); ++i) {
        for (std::size_t j = 1; j < column.size(); ++j)
            if (std::abs(rotated.pattern(row[i],j)) > std::abs(rotated.pattern(row[i],column[i])))
                column[i] = j;
        const auto loading = rotated.pattern(row[i],column[i]);
        sign[i] = std::copysign(1.0,loading);
        EXPECT_NEAR(std::abs(loading),main_loadings[i],tolerance);
    }
    EXPECT_NE(column[0],column[1]);
    EXPECT_NE(column[0],column[2]);
    EXPECT_NE(column[1],column[2]);
    EXPECT_NEAR(sign[0]*sign[1]*rotated.factor_correlation(column[0],column[1]),correlations[0],tolerance);
    EXPECT_NEAR(sign[0]*sign[2]*rotated.factor_correlation(column[0],column[2]),correlations[1],tolerance);
    EXPECT_NEAR(sign[1]*sign[2]*rotated.factor_correlation(column[1],column[2]),correlations[2],tolerance);
}

TEST(ObliqueFactorRotation, DirectObliminMatchesGPArotationAtTwoGammaValues) {
    const auto input = fixture();
    const auto quartimin = oblimin(input,0,2000,1e-8);
    ASSERT_TRUE(quartimin.converged);
    expect_reference(quartimin,{.843142790,.895317904,.844907088},
        {.620763565,.304143664,.439120558},4e-7);
    expect_invariant(input,quartimin);

    const auto biquartimin = oblimin(input,.5,2000,1e-8);
    ASSERT_TRUE(biquartimin.converged);
    expect_reference(biquartimin,{1.561334138,1.619353038,1.473501722},
        {.926017833,.865933162,.890852668},4e-7);
    expect_invariant(input,biquartimin);
}

TEST(ObliqueFactorRotation, PromaxMatchesRAndPreservesCommonCovariance) {
    const auto input = fixture();
    const auto fit = promax(input,4,1000,1e-12);
    ASSERT_TRUE(fit.converged);
    expect_reference(fit,{.866147331,.916221500,.848939990},
        {.657850698,.323456168,.457446786},2e-4);
    expect_invariant(input,fit);
}

TEST(ObliqueFactorRotation, HandlesStationaryStartAndReportsIterationLimit) {
    const DenseMatrix<double> symmetric(4,2,{.6,.6,.8,.8,-.6,.6,-.8,.8});
    const auto rotated = oblimin(symmetric);
    ASSERT_TRUE(rotated.converged);
    for (std::size_t i = 0; i < symmetric.rows(); ++i)
        EXPECT_NEAR(rotated.pattern(i,0)*rotated.pattern(i,1),0,1e-13);
    expect_invariant(symmetric,rotated);
    const auto stopped = oblimin(fixture(),0,1,1e-10);
    EXPECT_FALSE(stopped.converged);
    expect_invariant(fixture(),stopped);
}

TEST(ObliqueFactorRotation, SingleFactorAndDegenerateInput) {
    const DenseMatrix<double> input(3,1,{.6,0,-.7});
    for (const auto& fit : {oblimin(input),promax(input)}) {
        EXPECT_TRUE(fit.converged);
        EXPECT_EQ(fit.pattern.data(),input.data());
        expect_invariant(input,fit);
    }
    const DenseMatrix<double> repeated(3,2,{.2,.2,.6,.6,.1,.1});
    EXPECT_THROW((void)oblimin(repeated),std::invalid_argument);
    EXPECT_THROW((void)promax(repeated),std::invalid_argument);
    EXPECT_THROW((void)promax(fixture(),1),std::invalid_argument);
    EXPECT_THROW((void)oblimin(fixture(),std::numeric_limits<double>::infinity()),std::invalid_argument);
}
}
