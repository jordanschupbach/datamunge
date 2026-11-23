#include <datamunge/stats/canonical_correlation.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::canonical_correlation;

TEST(CanonicalCorrelation, CoefficientsScoresLoadingsAndSequentialTestsMatchR) {
    DenseMatrix<double> x(24,3), y(24,2);
    for (std::size_t row = 0; row < x.rows(); ++row) {
        const double i = row + 1;
        const auto u = std::sin(.3*i), v = std::cos(.61*i);
        x(row,0) = 1+u+.4*std::cos(1.1*i);
        x(row,1) = 3+.7*v+.5*std::sin(1.23*i);
        x(row,2) = 8+.5*u-.3*v+.6*std::cos(2.2*i);
        y(row,0) = 5+.8*u-.4*v+.25*std::cos(2.1*i);
        y(row,1) = 2+.75*v+.43*std::sin(1.9*i);
    }

    const auto fit = canonical_correlation(x,y);
    EXPECT_EQ(fit.x_rank,3u);
    EXPECT_EQ(fit.y_rank,2u);
    ASSERT_EQ(fit.correlations.size(),2u);
    EXPECT_NEAR(fit.correlations[0],.937956283915445,1e-13);
    EXPECT_NEAR(fit.correlations[1],.643824565898660,1e-13);
    EXPECT_NEAR(fit.x_weights(0,0),1.035189015460799,1e-13);
    EXPECT_NEAR(fit.x_weights(2,1),-.215324638423585,1e-13);
    EXPECT_NEAR(fit.y_weights(0,1),.67620547488886,1e-13);
    EXPECT_NEAR(fit.y_weights(1,1),1.77887580564999,1e-13);
    EXPECT_NEAR(fit.x_loadings(1,1),.8850025223895074,1e-13);
    EXPECT_NEAR(fit.y_loadings(0,0),.999732863077513,1e-13);
    EXPECT_NEAR(fit.x_scores(0,0),-.413415322378487,1e-13);
    EXPECT_NEAR(fit.y_scores(0,1),1.57512632606228,1e-13);
    ASSERT_EQ(fit.sequential_tests.size(),2u);
    EXPECT_EQ(fit.sequential_tests[0].first_component,1u);
    EXPECT_EQ(fit.sequential_tests[0].df,6u);
    EXPECT_NEAR(fit.sequential_tests[0].wilks_lambda,.0703981435451956,1e-13);
    EXPECT_NEAR(fit.sequential_tests[0].chi_squared,53.0717677252498,1e-11);
    EXPECT_NEAR(fit.sequential_tests[0].p_value,1.13488671834604e-9,1e-19);
    EXPECT_EQ(fit.sequential_tests[1].first_component,2u);
    EXPECT_EQ(fit.sequential_tests[1].df,2u);
    EXPECT_NEAR(fit.sequential_tests[1].p_value,.00473362976339262,1e-13);

    const auto cross = fit.x_scores.transpose()*fit.y_scores;
    for (std::size_t k = 0; k < fit.correlations.size(); ++k) {
        EXPECT_NEAR(cross(k,k)/23,fit.correlations[k],1e-14);
        EXPECT_NEAR(cross(k,1-k)/23,0,1e-14);
        for (std::size_t i = 0; i < x.rows(); ++i) {
            double score = 0;
            for (std::size_t j = 0; j < x.cols(); ++j)
                score += (x(i,j)-fit.x_means[j])*fit.x_weights(j,k);
            EXPECT_NEAR(score,fit.x_scores(i,k),1e-14);
        }
    }

    for (std::size_t i = 0; i < x.rows(); ++i) {
        x(i,0) *= 1e-140;
        x(i,2) *= 1e140;
        y(i,0) *= 1e-120;
    }
    const auto rescaled = canonical_correlation(x,y);
    EXPECT_NEAR(rescaled.correlations[0],fit.correlations[0],1e-14);
    EXPECT_NEAR(rescaled.correlations[1],fit.correlations[1],1e-14);
    EXPECT_NEAR(rescaled.x_weights(0,0)*1e-140,fit.x_weights(0,0),1e-14);
}

TEST(CanonicalCorrelation, CompletesOrthogonalUnitVarianceScoresAtZeroCorrelation) {
    DenseMatrix<double> x(8,2), y(8,2);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        const auto a = i&1 ? 1.0 : -1.0, b = i&2 ? 1.0 : -1.0, c = i&4 ? 1.0 : -1.0;
        x(i,0) = a;
        x(i,1) = b;
        y(i,0) = .6*a+.8*c;
        y(i,1) = a*b;
    }
    const auto fit = canonical_correlation(x,y);
    ASSERT_EQ(fit.correlations.size(),2u);
    EXPECT_NEAR(fit.correlations[0],.6,1e-14);
    EXPECT_DOUBLE_EQ(fit.correlations[1],0);
    for (const auto& scores : {fit.x_scores,fit.y_scores}) {
        const auto covariance = scores.transpose()*scores;
        EXPECT_NEAR(covariance(0,0)/7,1,1e-14);
        EXPECT_NEAR(covariance(1,1)/7,1,1e-14);
        EXPECT_NEAR(covariance(0,1)/7,0,1e-14);
    }
    ASSERT_EQ(fit.sequential_tests.size(),2u);
    EXPECT_DOUBLE_EQ(fit.sequential_tests[1].wilks_lambda,1);
    EXPECT_DOUBLE_EQ(fit.sequential_tests[1].p_value,1);
}

TEST(CanonicalCorrelation, RetainsEstimableDirectionsAndSkipsTestsWithSingularJointCovariance) {
    DenseMatrix<double> x(6,8), y(6,2);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        const double a = i+1, b = std::sin(a);
        x(i,0) = a;
        x(i,1) = b;
        x(i,2) = 10;
        for (std::size_t j = 3; j < x.cols(); ++j)
            x(i,j) = j*a + (j-1)*b;
        y(i,0) = a;
        y(i,1) = std::cos(1.3*a);
    }
    const auto fit = canonical_correlation(x,y);
    EXPECT_EQ(fit.x_rank,2u);
    EXPECT_EQ(fit.y_rank,2u);
    ASSERT_EQ(fit.correlations.size(),2u);
    EXPECT_NEAR(fit.correlations[0],1,1e-14);
    EXPECT_DOUBLE_EQ(fit.x_loadings(2,0),0);
    EXPECT_DOUBLE_EQ(fit.x_weights(2,0),0);
    EXPECT_TRUE(fit.sequential_tests.empty());
    const auto perfect = canonical_correlation(y,y);
    EXPECT_EQ(perfect.x_rank,2u);
    EXPECT_NEAR(perfect.correlations[1],1,1e-14);
    EXPECT_TRUE(perfect.sequential_tests.empty());
}

TEST(CanonicalCorrelation, ValidatesPairingRankAndTolerance) {
    const DenseMatrix<double> x(3,1,{1,2,4});
    EXPECT_THROW((void)canonical_correlation(x,DenseMatrix<double>(2,1,{1,2})),std::invalid_argument);
    EXPECT_THROW((void)canonical_correlation(x,DenseMatrix<double>(3,1,2)),std::domain_error);
    EXPECT_THROW((void)canonical_correlation(x,x,-.1),std::invalid_argument);
    auto nonfinite = x;
    nonfinite(0,0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)canonical_correlation(x,nonfinite),std::invalid_argument);
}
} // namespace
