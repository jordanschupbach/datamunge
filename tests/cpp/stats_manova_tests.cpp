#include <datamunge/stats/hotelling_test.hpp>
#include <datamunge/stats/manova.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::manova;
using datamunge::stats::one_way_manova;

TEST(MANOVA, OneWayAndAdjustedTestsMatchRForAllFourCriteria) {
    DenseMatrix<double> y(18,3), full(18,4,1), reduced(18,2,1);
    std::vector<std::int64_t> group;
    for (std::size_t row = 0; row < 18; ++row) {
        const double i = row+1, code = row < 5 ? 0 : (row < 11 ? 1 : 2);
        const auto z = std::sin(.37*i);
        group.push_back(static_cast<std::int64_t>(code)*14-12);
        reduced(row,1) = full(row,1) = z;
        full(row,2) = code == 1;
        full(row,3) = code == 2;
        y(row,0) = 3+code+.45*z+.4*std::cos(1.8*i);
        y(row,1) = 6+.5*code-.5*z+.55*std::sin(1.4*i);
        y(row,2) = 8-.6*code+.3*code*code+.72*z+.45*std::cos(2.2*i);
    }
    // Base R summary.manova and cross-products from the nested multivariate fits.
    const double roots[2][2]{{8.73579109937228,.0853058211552378},{5.00839263600785,.132301759426143}};
    const double error[2][2]{{2.39116697301244,-2.17242177166553},{1.30932492448444,-.166336284597282}};
    const double hypothesis[2][2]{{6.49412451239208,-3.75104276489802},{6.161850340198367,.0959574014533319}};
    const std::array<double,4> stats[]{
        {.97588693227310142,.0946404117113272,8.82109692052751,8.73579109937227},
        {.95040933513835446,.146987198546515,5.140694395434,5.00839263600785}};
    const std::array<double,4> f[]{
        {4.44691036643322146,9.75254027336653,17.642193841055,40.7670251304039},
        {3.92385392719654735,6.43326491800129,9.42460639162899,21.703034756034}};
    const std::array<double,4> p[]{
        {.00279723358097758,1.18535909682879e-5,9.94892365035402e-8,3.61544837008397e-7},
        {.00635003414476746,3.83792100385039e-4,3.58854380873295e-5,2.43765136386535e-5}};
    for (const bool adjusted : {false,true}) {
        SCOPED_TRACE(adjusted);
        const auto result = adjusted ? manova(y,full,reduced) : one_way_manova(y,group);
        EXPECT_EQ(result.hypothesis_df,2u);
        EXPECT_EQ(result.error_df,adjusted ? 14u : 15u);
        ASSERT_EQ(result.roots.size(),2u);
        EXPECT_NEAR(result.roots[0],roots[adjusted][0],1e-12);
        EXPECT_NEAR(result.roots[1],roots[adjusted][1],1e-12);
        EXPECT_NEAR(result.error(0,0),error[adjusted][0],1e-12);
        EXPECT_NEAR(result.error(2,1),error[adjusted][1],1e-12);
        EXPECT_NEAR(result.hypothesis(0,0),hypothesis[adjusted][0],1e-12);
        EXPECT_NEAR(result.hypothesis(2,1),hypothesis[adjusted][1],1e-12);
        const std::array criteria{result.pillai,result.wilks,result.hotelling_lawley,result.roy};
        const double denominator[]{28.-2*adjusted,26.-2*adjusted,24.-2*adjusted,14.-adjusted};
        for (std::size_t k = 0; k < criteria.size(); ++k) {
            SCOPED_TRACE(k);
            const auto& criterion = criteria[k];
            EXPECT_NEAR(criterion.statistic,stats[adjusted][k],1e-12);
            ASSERT_TRUE(criterion.reference);
            EXPECT_NEAR(criterion.reference->statistic,f[adjusted][k],1e-12);
            EXPECT_NEAR(criterion.reference->p_value,p[adjusted][k],1e-13);
            EXPECT_DOUBLE_EQ(criterion.reference->df_numerator,k == 3 ? 3 : 6);
            EXPECT_DOUBLE_EQ(criterion.reference->df_denominator,denominator[k]);
        }
    }
}

TEST(MANOVA, SingleContrastMatchesHotellingUnderLargeChangesOfScale) {
    const DenseMatrix<double> a(3,2,{1,2,2,1,3,2}), b(3,2,{4,5,5,4,6,5});
    const auto hotelling = datamunge::stats::hotelling_t2_two_sample(a,b);
    const DenseMatrix<double> y(6,2,{1,2,2,1,3,2,4,5,5,4,6,5});
    auto scaled = y;
    for (std::size_t i = 0; i < y.rows(); ++i) {
        scaled(i,0) = (scaled(i,0) + 1000) * 1e-120;
        scaled(i,1) = (scaled(i,1) - 1000) * 1e120;
    }
    const auto fit = one_way_manova(scaled,{-2,-2,-2,7,7,7});
    for (const auto& criterion : {fit.pillai,fit.wilks,fit.hotelling_lawley,fit.roy}) {
        ASSERT_TRUE(criterion.reference);
        EXPECT_NEAR(criterion.reference->statistic,hotelling.f_statistic,1e-11);
        EXPECT_NEAR(criterion.reference->p_value,hotelling.p_value,1e-12);
    }
    const auto one_sample = datamunge::stats::hotelling_t2_one_sample(y);
    const auto intercept = manova(y,DenseMatrix<double>(6,1,1),DenseMatrix<double>(6,0));
    ASSERT_TRUE(intercept.wilks.reference);
    EXPECT_NEAR(intercept.wilks.reference->statistic,one_sample.f_statistic,1e-11);
}

TEST(MANOVA, ReturnsCriteriaWhenHotellingLawleyFReferenceIsUndefined) {
    const DenseMatrix<double> sample(5,2,{0,0,3,1,5,1,1,5,1,7});
    const auto fit = one_way_manova(sample,{2,3,3,7,7});
    EXPECT_EQ(fit.error_df,2u);
    EXPECT_FALSE(fit.hotelling_lawley.reference);
    EXPECT_TRUE(fit.roy.reference);
    EXPECT_TRUE(fit.wilks.reference);
    EXPECT_TRUE(fit.pillai.reference);
    EXPECT_GT(fit.hotelling_lawley.statistic,0);
}

TEST(MANOVA, ValidatesModelsEffectsAndResidualCovariance) {
    const DenseMatrix<double> y(6,2,{1,2,2,1,3,2,4,5,5,4,6,5});
    const DenseMatrix<double> full(6,2,{1,0,1,0,1,0,1,1,1,1,1,1});
    EXPECT_THROW((void)manova(y,full,DenseMatrix<double>(6,1,{1,0,0,0,0,0})),std::invalid_argument);
    EXPECT_THROW((void)manova(y,DenseMatrix<double>(6,2,1),DenseMatrix<double>(6,1,1)),std::invalid_argument);
    EXPECT_THROW((void)one_way_manova(y,{1,2,3,4,5,6}),std::invalid_argument);
    EXPECT_THROW((void)one_way_manova(y,{1,1,2,2,3}),std::invalid_argument);
    auto singular = y;
    for (std::size_t row = 0; row < y.rows(); ++row)
        singular(row,1) = 2 * singular(row,0);
    EXPECT_THROW((void)one_way_manova(singular,{1,1,1,2,2,2}),std::domain_error);
    singular(0,0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)one_way_manova(singular,{1,1,1,2,2,2}),std::invalid_argument);
}
} // namespace
