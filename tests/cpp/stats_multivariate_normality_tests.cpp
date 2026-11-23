#include <datamunge/stats/multivariate_normality.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::henze_zirkler_test;
using datamunge::stats::mardia_test;

DenseMatrix<double> sample(std::size_t n) {
    DenseMatrix<double> x(n,3);
    for (std::size_t row = 0; row < n; ++row) {
        const double i = row+1;
        x(row,0) = std::sin(.3*i)+.2*std::cos(2*i);
        x(row,1) = std::cos(.4*i)+.15*std::sin(i);
        x(row,2) = std::sin(.9*i)+.12*i;
    }
    x(n-1,0) += 2;
    x(n-1,1) -= 1;
    x(n-1,2) += 3;
    return x;
}

TEST(MultivariateNormality, MatchesBaseRWithAndWithoutSmallSampleCorrection) {
    const double reference[][11]{
        {5.603888147321845,12.903314890637166,15.837075198953043,.10439226138514490,-.66303004817457867,
         .50731130975440419,.85342050846001616,1.09237932753088129,-.68550553077059073,.25522975057536851,.01947017311467909},
        {6.145693365456935,17.551614779759468,27.655620144556209,.002049098264715487,1.2103371623083239,
         .22614954325207456,1.126664465880967,1.2265500122518731,-.53242248258128755,.23455981911576035,.002731977456479799}};
    for (const bool large : {false,true}) {
        const auto x = sample(large ? 27 : 12);
        const auto m = mardia_test(x);
        const auto hz = henze_zirkler_test(x);
        EXPECT_DOUBLE_EQ(m.skewness_df,10);
        const double actual[]{m.skewness,m.kurtosis,m.skewness_chi_squared,m.skewness_p_value,m.kurtosis_z,
            m.kurtosis_p_value,hz.statistic,hz.bandwidth,hz.mean_log,hz.standard_deviation_log,hz.p_value};
        for (std::size_t i = 0; i < 11; ++i)
            EXPECT_NEAR(actual[i],reference[large][i],8e-13);
    }
}

TEST(MultivariateNormality, PreservesStatisticsAcrossAffineTransforms) {
    const auto x = sample(12);
    auto y = x;
    for (std::size_t row = 0; row < x.rows(); ++row) {
        y(row,0) = (5 + x(row,0) - .4*x(row,1))*1e120;
        y(row,1) = (2 + x(row,1) + x(row,2))*1e-120;
        y(row,2) = -8-x(row,2);
    }
    EXPECT_NEAR(mardia_test(x).skewness,mardia_test(y).skewness,1e-13);
    EXPECT_NEAR(mardia_test(x).kurtosis,mardia_test(y).kurtosis,1e-13);
    EXPECT_NEAR(henze_zirkler_test(x).p_value,henze_zirkler_test(y).p_value,1e-13);
}

TEST(MultivariateNormality, RejectsSingularAndInsufficientSamples) {
    const DenseMatrix<double> singular(4,2,{1,2,2,4,3,6,5,10});
    EXPECT_THROW((void)mardia_test(singular),std::domain_error);
    EXPECT_THROW((void)henze_zirkler_test(singular),std::domain_error);
    EXPECT_THROW((void)mardia_test(DenseMatrix<double>(3,2)),std::invalid_argument);
    EXPECT_THROW((void)henze_zirkler_test(DenseMatrix<double>(4,1,1)),std::domain_error);
    const auto symmetric = mardia_test(DenseMatrix<double>(4,1,{-2,-1,1,2}));
    EXPECT_NEAR(symmetric.skewness,0,1e-15);
    EXPECT_DOUBLE_EQ(symmetric.skewness_p_value,1);
}
} // namespace
