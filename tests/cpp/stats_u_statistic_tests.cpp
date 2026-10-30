#include <gtest/gtest.h>
#include <datamunge/stats/u_statistic.hpp>
#include <datamunge/stats/jackknife.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

using datamunge::stats::UStatistic;
using datamunge::stats::SymmetricKernel;

namespace {
const SymmetricKernel variance_kernel=[](const auto& x){return 0.5*(x[0]-x[1])*(x[0]-x[1]);};
const SymmetricKernel product_kernel=[](const auto& x){return std::accumulate(x.begin(),x.end(),1.0,std::multiplies<double>());};
}

TEST(UStatisticDiagnostics, ThreePointTermsProjectionsAndJackknifeMatchHandCalculation) {
    const UStatistic u(2);
    const auto terms=u.term_details({1,2,4},variance_kernel);
    ASSERT_EQ(terms.nrows(),3u);
    const double values[]={0.5,4.5,2};
    for (std::size_t i=0;i<3;++i) EXPECT_DOUBLE_EQ(terms.double_at("kernel",i),values[i]);
    const auto fit=u.analyze({1,2,4},variance_kernel);
    EXPECT_NEAR(fit.result.estimate,7.0/3,1e-12);
    EXPECT_NEAR(fit.result.zeta1,49.0/48,1e-12);
    EXPECT_NEAR(fit.result.variance,49.0/36,1e-12);
    EXPECT_NEAR(fit.jackknife_variance,49.0/9,1e-12);
    EXPECT_DOUBLE_EQ(fit.monte_carlo_variance,0);
    const double means[]={2.5,1.25,3.25},loo[]={2,4.5,0.5},pseudo[]={3,-2,6};
    for (std::size_t i=0;i<3;++i) {
        EXPECT_DOUBLE_EQ(fit.projections.double_at("inclusions",i),2);
        EXPECT_NEAR(fit.projections.double_at("conditional_mean",i),means[i],1e-12);
        EXPECT_NEAR(fit.projections.double_at("leave_one_out",i),loo[i],1e-12);
        EXPECT_NEAR(fit.projections.double_at("pseudo_value",i),pseudo[i],1e-12);
    }
    EXPECT_NEAR(fit.kernel_variance,49.0/18,1e-12);
}

TEST(UStatisticDiagnostics, DeleteOneFormulaAgreesWithIndependentRefitsForSeveralDegrees) {
    const std::vector<double> sample{1,2,3,4,5,6};
    for (std::size_t degree=1;degree<=4;++degree) {
        const UStatistic u(degree);
        const auto fit=u.analyze(sample,product_kernel);
        const auto jk=datamunge::stats::jackknife(sample,[&](const auto& x){return u.run(x,product_kernel).estimate;});
        EXPECT_NEAR(fit.jackknife_variance,jk.standard_error*jk.standard_error,1e-9);
        for (std::size_t i=0;i<sample.size();++i)
            EXPECT_NEAR(fit.projections.double_at("leave_one_out",i),jk.leave_one_out[i],1e-10);
        const double correction=std::pow(5.0/(6-degree),2);
        EXPECT_NEAR(fit.jackknife_variance,correction*fit.result.variance,1e-9);
    }
}

TEST(UStatisticDiagnostics, CompleteEnumerationHasEveryDistinctSubsetExactlyOnce) {
    const auto terms=UStatistic(3).term_details({1,2,3,4,5},product_kernel);
    EXPECT_EQ(terms.nrows(),10u);
    std::set<std::vector<int>> unique;
    for (std::size_t r=0;r<terms.nrows();++r) {
        std::vector<int> indices;
        for (int j=1;j<=3;++j) indices.push_back(terms.double_at("i"+std::to_string(j),r));
        EXPECT_TRUE(std::is_sorted(indices.begin(),indices.end()));
        unique.insert(indices);
    }
    EXPECT_EQ(unique.size(),10u);
}

TEST(UStatisticDiagnostics, IncompleteTuplesAreDistinctWithinDrawAndMayRepeatAcrossDraws) {
    const UStatistic u(2,{100,7});
    const auto terms=u.term_details({1,2,4},variance_kernel);
    const auto fit=u.analyze({1,2,4},variance_kernel);
    double mean=0,ss=0;
    for (std::size_t i=0;i<100;++i) {
        EXPECT_NE(terms.double_at("i1",i),terms.double_at("i2",i));
        mean+=terms.double_at("kernel",i)/100;
    }
    for (std::size_t i=0;i<100;++i) ss+=std::pow(terms.double_at("kernel",i)-mean,2);
    EXPECT_NEAR(fit.result.estimate,mean,1e-12);
    EXPECT_NEAR(fit.kernel_variance,ss/99,1e-12);
    EXPECT_NEAR(fit.monte_carlo_variance,ss/9900,1e-12);
    EXPECT_FALSE(fit.result.complete);
    EXPECT_TRUE(std::isnan(fit.jackknife_variance));
    EXPECT_EQ(u.term_details({1,2,4},variance_kernel).to_md(12),terms.to_md(12));
}

TEST(UStatisticDiagnostics, SparseCoverageAndSingleDrawAreExplicitlyUnavailable) {
    const auto fit=UStatistic(2,{1,42}).analyze({1,2,3,4,5},variance_kernel);
    std::size_t uncovered=0;
    for (std::size_t i=0;i<5;++i) if (fit.projections.double_at("inclusions",i)==0) {
        ++uncovered;
        EXPECT_TRUE(std::isnan(fit.projections.double_at("conditional_mean",i)));
    }
    EXPECT_EQ(uncovered,3u);
    EXPECT_TRUE(std::isnan(fit.kernel_variance));
    EXPECT_TRUE(std::isnan(fit.monte_carlo_variance));
    EXPECT_TRUE(std::isnan(fit.projections.double_at("leave_one_out",0)));
    EXPECT_DOUBLE_EQ(fit.summary().double_at("covered_rows",0),2);
}

TEST(UStatisticDiagnostics, EntireSampleKernelCannotSupplyDeleteOneOrProjectionUncertainty) {
    const auto fit=UStatistic(3).analyze({1,2,4},product_kernel);
    EXPECT_DOUBLE_EQ(fit.result.estimate,8);
    EXPECT_EQ(fit.result.num_terms,1u);
    EXPECT_DOUBLE_EQ(fit.result.variance,0);
    EXPECT_TRUE(std::isnan(fit.jackknife_variance));
}

TEST(UStatisticDiagnostics, RademacherProductLawExposesDegeneracyAndExactVariance) {
    for (std::size_t n:{4u,8u,16u,32u,64u,128u}) {
        const auto law=UStatistic(2).two_point_law(n,-1,1,0.5,product_kernel);
        EXPECT_NEAR(law.target,0,1e-14);
        EXPECT_NEAR(law.mean,0,1e-13);
        EXPECT_NEAR(law.variance,2.0/(n*(n-1)),1e-13);
        EXPECT_NEAR(law.first_order_variance,0,1e-14);
        for (std::size_t k=0;k<=n;++k) {
            const double sum=2.0*k-n;
            EXPECT_NEAR(law.distribution.double_at("estimate",k),(sum*sum-n)/(n*(n-1)),1e-12);
        }
    }
}

TEST(UStatisticDiagnostics, NondegenerateProductLawMatchesFullHoeffdingVariance) {
    constexpr double p=0.75,mu=0.5,sigma2=0.75;
    for (std::size_t n:{4u,8u,32u,128u}) {
        const auto law=UStatistic(2).two_point_law(n,-1,1,p,product_kernel);
        const double leading=4*mu*mu*sigma2/n;
        EXPECT_NEAR(law.target,mu*mu,1e-14);
        EXPECT_NEAR(law.mean,law.target,1e-13);
        EXPECT_NEAR(law.first_order_variance,leading,1e-13);
        EXPECT_NEAR(law.variance,leading+2*sigma2*sigma2/(n*(n-1)),1e-13);
    }
}

TEST(UStatisticDiagnostics, TwoPointLawAgreesWithExplicitSequenceEnumerationForGeneralDegree) {
    constexpr std::size_t n=5;
    const SymmetricKernel kernel=[](const auto& x){return std::pow(std::accumulate(x.begin(),x.end(),0.0),2);};
    for (std::size_t m:{1u,2u,3u,5u}) {
        const UStatistic u(m);
        const auto law=u.two_point_law(n,-2,3,0.3,kernel);
        long double mean=0,second=0;
        for (std::size_t bits=0;bits<(1u<<n);++bits) {
            std::vector<double> sample(n,-2);
            double probability=1;
            for (std::size_t i=0;i<n;++i) {
                const bool high=bits&(1u<<i); if (high) sample[i]=3;
                probability*=high ? 0.3 : 0.7;
            }
            const double value=u.run(sample,kernel).estimate;
            mean+=probability*value; second+=probability*value*value;
        }
        EXPECT_NEAR(law.mean,mean,1e-11);
        EXPECT_NEAR(law.mean,law.target,1e-11);
        EXPECT_NEAR(law.variance,second-mean*mean,1e-9);
    }
}

TEST(UStatisticDiagnostics, TwoPointLawHandlesProbabilityEndpoints) {
    for (double p:{0.0,1.0}) {
        const auto law=UStatistic(2).two_point_law(8,-2,3,p,product_kernel);
        EXPECT_DOUBLE_EQ(law.variance,0);
        EXPECT_DOUBLE_EQ(law.first_order_variance,0);
        EXPECT_DOUBLE_EQ(law.mean,p==0 ? 4 : 9);
    }
}

TEST(UStatisticDiagnostics, MonteCarloRmseAgreesWithExactConditionalPrediction) {
    const auto path=UStatistic(2,{0,31}).incomplete_path({1,2,4},variance_kernel,{20,200},1024);
    for (std::size_t i=0;i<2;++i) {
        const double expected=std::sqrt((49.0/18)/path.double_at("terms",i));
        EXPECT_NEAR(path.double_at("expected_rmse",i),expected,1e-12);
        EXPECT_NEAR(path.double_at("rmse_to_complete",i),expected,expected*0.08);
        EXPECT_NEAR(path.double_at("mean_estimated_mc_variance",i),expected*expected,expected*expected*0.04);
    }
}

TEST(UStatisticDiagnostics, InvalidDataCallbacksAndBudgetsFailExplicitly) {
    EXPECT_THROW(UStatistic(0),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(2).run({1},variance_kernel),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(1).run({std::numeric_limits<double>::quiet_NaN()},product_kernel),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(1).run({1},[](const auto&){return std::numeric_limits<double>::infinity();}),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(1).run({1},{}),std::bad_function_call);
    EXPECT_THROW((void)UStatistic(1).incomplete_path({1,2},product_kernel,{}),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(1).incomplete_path({1,2},product_kernel,{0}),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(1).incomplete_path({1,2},product_kernel,{10},0),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(2).two_point_law(1,-1,1,0.5,product_kernel),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(2).two_point_law(4,1,1,0.5,product_kernel),std::invalid_argument);
    EXPECT_THROW((void)UStatistic(2).two_point_law(4,-1,1,-0.1,product_kernel),std::invalid_argument);
}

TEST(UStatisticDiagnostics, ImpossibleCompleteCountIsRejectedBeforeEvaluatingKernel) {
    std::size_t calls=0;
    const auto kernel=[&](const auto&){++calls;return 1.0;};
    EXPECT_THROW((void)UStatistic(35).run(std::vector<double>(70,1),kernel),std::overflow_error);
    EXPECT_EQ(calls,0u);
    const auto incomplete=UStatistic(35,{2,9}).run(std::vector<double>(70,1),kernel);
    EXPECT_DOUBLE_EQ(incomplete.estimate,1);
    EXPECT_EQ(calls,2u);
}
