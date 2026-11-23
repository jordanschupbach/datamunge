#include <datamunge/stats/mixed_model_inference.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_variance_components;
using datamunge::stats::mixed_model_contrast_test;
using datamunge::stats::MixedModelDF;
using datamunge::stats::VarianceComponentsStatus;

TEST(MixedModelInference, MatchesIndependentObservedAndExpectedInformationForCrossedDesign) {
    const double person[]{-1.4,.8,-.6,1.65,1.35,-1.8,.75};
    const double item[]{.9,-1.2,.25,.68,-.63};
    std::vector<double> values, y;
    std::vector<std::pair<int,int>> labels;
    for (int p = 0; p < 7; ++p)
        for (int i = 0; i < 5; ++i) {
            if ((2*p+i) % 7 == 4)
                continue;
            const double x = ((2*p-i+9) % 9) / 4. - 1;
            values.push_back(x);
            labels.emplace_back(p, i);
            y.push_back(2 + 1.4*x + person[p] + item[i] + .46*std::cos(3*p+2*i));
        }
    DenseMatrix<double> x(y.size(), 2, 1);
    std::vector<DenseMatrix<double>> z{DenseMatrix<double>(y.size(),7),DenseMatrix<double>(y.size(),5)};
    for (std::size_t i = 0; i < y.size(); ++i) {
        x(i,1) = values[i];
        z[0](i,labels[i].first) = 1;
        z[1](i,labels[i].second) = 1;
    }
    const auto fit = fit_variance_components(x,y,z);
    ASSERT_EQ(fit.status, VarianceComponentsStatus::Converged);
    // Reference uses base R REML, dense projectors and moment calculations.
    const DenseMatrix<double> slope(1, 2, {0,1});
    const auto sat = mixed_model_contrast_test(fit,x,y,z,slope,MixedModelDF::Satterthwaite,{1});
    EXPECT_EQ(sat.numerator_df, 1u);
    EXPECT_NEAR(sat.denominator_df, 18.4180789542, 2e-6);
    EXPECT_NEAR(sat.statistic, 10.0598230208, 1e-6);
    EXPECT_NEAR(sat.p_value, .00516844797874, 1e-8);
    EXPECT_NEAR(sat.covariance(1,1), fit.covariance(1,1), 1e-12);

    const auto kr = mixed_model_contrast_test(fit,x,y,z,slope,MixedModelDF::KenwardRoger,{1});
    EXPECT_NEAR(kr.denominator_df, 18.4673446185, 2e-6);
    EXPECT_NEAR(kr.statistic, 9.93651308807, 1e-6);
    EXPECT_NEAR(kr.p_value, .00538316932171, 1e-8);
    EXPECT_DOUBLE_EQ(kr.scale, 1);
    EXPECT_NEAR(kr.covariance(0,0), .405537431154, 1e-6);
    EXPECT_NEAR(kr.covariance(1,0), -.000961973371, 1e-8);
    EXPECT_NEAR(kr.covariance(1,1), .0149709832481, 1e-8);

    const auto all = DenseMatrix<double>::identity(2);
    const auto joint_sat = mixed_model_contrast_test(fit,x,y,z,all,MixedModelDF::Satterthwaite,{2,1});
    EXPECT_NEAR(joint_sat.denominator_df, 12.4270521545, 3e-6);
    EXPECT_NEAR(joint_sat.statistic, 5.04666160286, 1e-6);
    EXPECT_NEAR(joint_sat.p_value, .0248668595462, 1e-8);
    const auto joint_kr = mixed_model_contrast_test(fit,x,y,z,all,MixedModelDF::KenwardRoger,{2,1});
    EXPECT_EQ(joint_kr.numerator_df, 2u);
    EXPECT_NEAR(joint_kr.denominator_df, 16.2035181239, 2e-6);
    EXPECT_NEAR(joint_kr.scale, .961174942896, 1e-8);
    EXPECT_NEAR(joint_kr.statistic, 4.79148581339, 1e-6);
    EXPECT_NEAR(joint_kr.p_value, .0231848078214, 1e-8);
    const auto transformed = mixed_model_contrast_test(fit,x,y,z,
        DenseMatrix<double>(2,2,{3,1,5,2}),MixedModelDF::KenwardRoger,{7,12});
    EXPECT_NEAR(transformed.statistic, joint_kr.statistic, 1e-10);
    EXPECT_NEAR(transformed.denominator_df, joint_kr.denominator_df, 1e-10);
}

TEST(MixedModelInference, UsesExactBalancedDegreesAndActiveBoundaryModel) {
    const DenseMatrix<double> x(6,1,1), z(6,3,{1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1});
    const DenseMatrix<double> l(1,1,1);
    const std::vector<double> grouped{0,1,3,4,6,7}, balanced{0,1,0,1,0,1};
    const auto interior = fit_variance_components(x,grouped,{z});
    const auto boundary = fit_variance_components(x,balanced,{z});
    ASSERT_EQ(boundary.status, VarianceComponentsStatus::Boundary);
    for (const auto method : {MixedModelDF::Satterthwaite,MixedModelDF::KenwardRoger}) {
        const auto a = mixed_model_contrast_test(interior,x,grouped,{z},l,method);
        EXPECT_NEAR(a.denominator_df, 2, 1e-7);
        EXPECT_NEAR(a.covariance(0,0), 3, 1e-7);
        const auto b = mixed_model_contrast_test(boundary,x,balanced,{z},l,method);
        EXPECT_DOUBLE_EQ(b.denominator_df, 5);
        EXPECT_DOUBLE_EQ(b.scale, 1);
        EXPECT_NEAR(b.statistic, 5, 1e-11);
    }
    const auto ml = fit_variance_components(x,grouped,{z},{.reml = false});
    EXPECT_THROW((void)mixed_model_contrast_test(ml,x,grouped,{z},l), std::invalid_argument);
    EXPECT_THROW((void)mixed_model_contrast_test(interior,x,grouped,{z},DenseMatrix<double>(1,1,0)),
                 std::invalid_argument);
}
}
