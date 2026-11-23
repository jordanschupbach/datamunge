#include <datamunge/stats/hotelling_test.hpp>

#include <limits>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::hotelling_t2_one_sample;
using datamunge::stats::hotelling_t2_paired;
using datamunge::stats::hotelling_t2_two_sample;

TEST(HotellingMeanTest, AllThreeFormsMatchIndependentRQuadraticFormsAndFTails) {
    const DenseMatrix<double> a(6,3,{2,1,5,4,3,1,3,2,2,6,1,3,5,4,2,7,3,5});
    const DenseMatrix<double> b(6,3,{1,0,2,2,1,1,4,2,0,3,4,1,2,1,3,4,1,5});
    const DenseMatrix<double> shorter(5,3,{1,0,2,2,1,1,4,2,0,3,4,1,2,1,3});
    const auto one = hotelling_t2_one_sample(a,{4,2,3});
    EXPECT_NEAR(one.t_squared,.6781545302946085,1e-13);
    EXPECT_NEAR(one.f_statistic,.1356309060589217,1e-13);
    EXPECT_NEAR(one.p_value,.9324969452694757,1e-13);
    EXPECT_EQ(one.df_numerator,3u);
    EXPECT_EQ(one.df_denominator,3u);
    EXPECT_DOUBLE_EQ(one.mean_difference[0],4.5);

    const auto paired = hotelling_t2_paired(a,b,{.5,0,.5});
    EXPECT_NEAR(paired.t_squared,22.9892638036809913,1e-12);
    EXPECT_NEAR(paired.f_statistic,4.5978527607361981,1e-13);
    EXPECT_NEAR(paired.p_value,.1210736562448742,1e-13);
    EXPECT_EQ(paired.df_numerator,3u);
    EXPECT_EQ(paired.df_denominator,3u);
    EXPECT_NEAR(paired.mean_difference[0],11.0/6,1e-15);
    EXPECT_NEAR(hotelling_t2_paired(b,a,{-.5,0,-.5}).t_squared,paired.t_squared,1e-13);

    const auto pooled = hotelling_t2_two_sample(a,shorter,{0,.1,0});
    EXPECT_NEAR(pooled.t_squared,9.1886182465770325,1e-12);
    EXPECT_NEAR(pooled.f_statistic,2.3822343602236749,1e-13);
    EXPECT_NEAR(pooled.p_value,.1552426666828575,1e-13);
    EXPECT_EQ(pooled.df_numerator,3u);
    EXPECT_EQ(pooled.df_denominator,7u);
    EXPECT_DOUBLE_EQ(pooled.mean_difference[2],1.6);
}

TEST(HotellingMeanTest, PoolingCanResolveTwoIndividuallySingularCovariances) {
    const DenseMatrix<double> a(2,3,{2,1,5,4,3,1});
    const DenseMatrix<double> b(3,3,{1,0,2,2,1,1,4,2,0});
    const auto pooled = hotelling_t2_two_sample(a,b);
    EXPECT_NEAR(pooled.t_squared,162.6,4e-12);
    EXPECT_NEAR(pooled.p_value,.1708534383464591,1e-13);
    EXPECT_EQ(pooled.df_denominator,1u);
}

TEST(HotellingMeanTest, IsInvariantToOutcomeUnitsAndPairOffsets) {
    const DenseMatrix<double> a(6,3,{2,1,5,4,3,1,3,2,2,6,1,3,5,4,2,7,3,5});
    const DenseMatrix<double> b(6,3,{1,0,2,2,1,1,4,2,0,3,4,1,2,1,3,4,1,5});
    auto modified_a = a, modified_b = b;
    const double scales[]{1e-140,1e140,1e-100};
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) {
            modified_a(i,j) *= scales[j];
            modified_b(i,j) *= scales[j];
        }
    EXPECT_NEAR(hotelling_t2_paired(modified_a,modified_b).t_squared,hotelling_t2_paired(a,b).t_squared,1e-12);

    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) {
            const double shift = (double(i)-3) * 1e12;
            modified_a(i,j) = a(i,j) + shift;
            modified_b(i,j) = b(i,j) + shift;
        }
    EXPECT_DOUBLE_EQ(hotelling_t2_paired(modified_a,modified_b).t_squared,hotelling_t2_paired(a,b).t_squared);
}

TEST(HotellingMeanTest, ValidatesCovarianceSampleSizeAndFiniteInput) {
    const DenseMatrix<double> singular(4,2,{1,2,2,4,3,6,4,8});
    EXPECT_THROW((void)hotelling_t2_one_sample(singular), std::domain_error);
    EXPECT_THROW((void)hotelling_t2_one_sample(DenseMatrix<double>(2,2,{1,2,3,4})), std::invalid_argument);
    EXPECT_THROW((void)hotelling_t2_paired(DenseMatrix<double>(3,1,{1,2,3}),singular), std::invalid_argument);
    EXPECT_THROW((void)hotelling_t2_two_sample(DenseMatrix<double>(3,1,{1,2,3}),singular),std::invalid_argument);
    EXPECT_THROW((void)hotelling_t2_paired(singular,singular), std::domain_error);
    auto nonfinite = singular;
    nonfinite(0,1) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)hotelling_t2_one_sample(nonfinite), std::invalid_argument);

    const auto exact = hotelling_t2_one_sample(DenseMatrix<double>(3,1,{1,2,3}),{2});
    EXPECT_DOUBLE_EQ(exact.t_squared,0);
    EXPECT_DOUBLE_EQ(exact.f_statistic,0);
    EXPECT_DOUBLE_EQ(exact.p_value,1);
    const auto strong = hotelling_t2_one_sample(DenseMatrix<double>(3,1,{1e-200,2e-200,3e-200}),{1});
    EXPECT_EQ(strong.f_statistic,std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(strong.p_value,0);
}
} // namespace
