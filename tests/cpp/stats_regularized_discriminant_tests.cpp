#include <datamunge/stats/regularized_discriminant.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::DiscriminantCovariance;
using datamunge::stats::fit_quadratic_discriminant;
using datamunge::stats::fit_regularized_discriminant;

const DenseMatrix<double> x(15,2,{-3,0,-2,2,-1,-.4,-1.6,1.3,-2.4,-.6, 0,2,.6,2.2,2,3,1.8,1.7,
    2,-2,3,-1.6,3.1,-2.5,4.2,-.5,4.1,-2.8,2.7,-2.1});
const std::vector<std::string> labels{"Zeta","Zeta","Zeta","Zeta","Zeta","Alpha","Alpha","Alpha","Alpha",
    "Beta","Beta","Beta","Beta","Beta","Beta"};
const DenseMatrix<double> query(5,2,{-2,0,.6,2.3,3,-2,1.7,.5,-500,50});

TEST(RegularizedDiscriminant, SampleQDAAndMaximumLikelihoodRDAAgreeWithIndependentGaussianScores) {
    const auto qda = fit_quadratic_discriminant(x,labels);
    EXPECT_EQ(qda.classes(),(std::vector<std::string>{"Alpha","Beta","Zeta"}));
    EXPECT_NEAR(qda.priors()[0],4.0/15,1e-15);
    EXPECT_NEAR(qda.class_means()(1,1),-1.91666666666667,1e-13);
    EXPECT_NEAR(qda.class_covariances()[0](0,0),.92,1e-13);
    EXPECT_NEAR(qda.class_covariances()[1](1,0),.161666666666667,1e-13);
    EXPECT_NEAR(qda.class_covariances()[2](1,1),1.288,1e-13);
    const auto base = qda.predict(query);
    EXPECT_EQ(base.class_label,(std::vector<std::string>{"Zeta","Alpha","Beta","Alpha","Alpha"}));
    EXPECT_NEAR(base.posterior(0,2),.999915713040630,1e-13);
    EXPECT_NEAR(base.posterior(1,2),.00114982978793335,1e-14);
    EXPECT_NEAR(base.posterior(3,0),.721121109992265,1e-13);
    EXPECT_NEAR(base.posterior(3,1),.276256106196697,1e-13);
    EXPECT_DOUBLE_EQ(base.posterior(4,0),1);

    const auto rda = fit_regularized_discriminant(x,labels,
        {.pooling=.37,.shrinkage=.22,.normalization=DiscriminantCovariance::MaximumLikelihood,.priors={2,3,5}});
    EXPECT_EQ(rda.priors(),(std::vector<double>{.2,.3,.5}));
    EXPECT_NEAR(rda.class_covariances()[0](0,0),.5989559768690624,1e-13);
    EXPECT_NEAR(rda.class_covariances()[1](1,0),.0969635584137192,1e-13);
    EXPECT_NEAR(rda.class_covariances()[2](1,1),.7447031130268200,1e-13);
    const auto regularized = rda.predict(query);
    EXPECT_NEAR(regularized.posterior(0,2),.999990719825189,1e-13);
    EXPECT_NEAR(regularized.posterior(3,0),.982857308902583,1e-13);
    EXPECT_NEAR(regularized.posterior(3,1),.0167986241152805,1e-13);
    for (std::size_t i = 0; i < query.rows(); ++i) {
        const auto row = regularized.posterior.row(i);
        EXPECT_NEAR(std::accumulate(row.begin(),row.end(),0.0),1,1e-15);
    }
}

TEST(RegularizedDiscriminant, FullPoolingGivesCommonCovariancesAndIsotropicLimit) {
    const auto pooled = fit_regularized_discriminant(x,labels,{.pooling=1});
    for (const auto& covariance : pooled.class_covariances()) {
        EXPECT_NEAR(covariance(0,0),.719027777777778,1e-13);
        EXPECT_NEAR(covariance(1,0),.146527777777778,1e-13);
        EXPECT_NEAR(covariance(1,1),.778986111111111,1e-13);
    }
    EXPECT_NEAR(pooled.predict(query).posterior(3,0),.970781530824507,1e-13);
    const auto spherical = fit_regularized_discriminant(x,labels,{.pooling=1,.shrinkage=1});
    for (const auto& covariance : spherical.class_covariances()) {
        EXPECT_NEAR(covariance(0,0),(.719027777777778+.778986111111111)/2,1e-13);
        EXPECT_DOUBLE_EQ(covariance(0,0),covariance(1,1));
        EXPECT_DOUBLE_EQ(covariance(0,1),0);
    }
}

TEST(RegularizedDiscriminant, ShrinkageHandlesCollinearityAndPoolingHandlesSingletons) {
    const DenseMatrix<double> collinear(7,2,{0,0,1,1,2,2,4,4,5,5,6,6,9,9});
    auto group = std::vector<std::string>{"A","A","A","B","B","B","B"};
    EXPECT_THROW((void)fit_quadratic_discriminant(collinear,group),std::domain_error);
    const auto regularized = fit_regularized_discriminant(collinear,group,{.shrinkage=.2});
    EXPECT_DOUBLE_EQ(regularized.class_covariances()[0](0,0),1);
    EXPECT_NEAR(regularized.class_covariances()[0](0,1),.8,1e-14);
    EXPECT_EQ(regularized.predict(DenseMatrix<double>(1,2,{1,1})).class_label[0],"A");

    group.back() = "C";
    EXPECT_THROW((void)fit_regularized_discriminant(collinear,group,{.shrinkage=.2}),std::domain_error);
    const auto pooled = fit_regularized_discriminant(collinear,group,{.pooling=.5,.shrinkage=.2});
    EXPECT_DOUBLE_EQ(pooled.class_covariances()[2](0,0),1);
    EXPECT_EQ(pooled.predict(DenseMatrix<double>(1,2,{9,9})).class_label[0],"C");
}

TEST(RegularizedDiscriminant, QuadraticPosteriorsAreInvariantToUnitsAndPriorTiesAreDeterministic) {
    auto scaled = x, scaled_query = query;
    for (std::size_t i = 0; i < scaled.rows(); ++i) {
        scaled(i,0) *= 1e-140;
        scaled(i,1) *= 1e140;
    }
    for (std::size_t i = 0; i < scaled_query.rows(); ++i) {
        scaled_query(i,0) *= 1e-140;
        scaled_query(i,1) *= 1e140;
    }
    const auto base = fit_quadratic_discriminant(x,labels).predict(query);
    const auto transformed = fit_quadratic_discriminant(scaled,labels).predict(scaled_query);
    EXPECT_EQ(base.class_label,transformed.class_label);
    for (std::size_t i = 0; i < query.rows(); ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(base.posterior(i,j),transformed.posterior(i,j),1e-13);

    const DenseMatrix<double> identical(6,1,{0,0,1,1,2,2});
    const std::vector<std::string> alternated{"Z","A","Z","A","Z","A"};
    const auto tied = fit_quadratic_discriminant(identical,alternated).predict(DenseMatrix<double>(1,1,{1}));
    EXPECT_EQ(tied.class_label[0],"A");
    EXPECT_DOUBLE_EQ(tied.posterior(0,0),.5);
    const auto prior = fit_quadratic_discriminant(identical,alternated,{1,3}).predict(DenseMatrix<double>(1,1,{1}));
    EXPECT_EQ(prior.class_label[0],"Z");
    EXPECT_DOUBLE_EQ(prior.posterior(0,1),.75);
}

TEST(RegularizedDiscriminant, RejectsMalformedDataAndOptions) {
    EXPECT_THROW((void)fit_regularized_discriminant(x,labels,{.pooling=1.1}),std::invalid_argument);
    EXPECT_THROW((void)fit_regularized_discriminant(x,labels,{.shrinkage=-.2}),std::invalid_argument);
    EXPECT_THROW((void)fit_quadratic_discriminant(x,labels,{1,0,1}),std::invalid_argument);
    EXPECT_THROW((void)fit_quadratic_discriminant(x,std::vector<std::string>(15,"A")),std::invalid_argument);
    const auto fit = fit_quadratic_discriminant(x,labels);
    EXPECT_THROW((void)fit.predict(DenseMatrix<double>(1,1,{1})),std::invalid_argument);
    EXPECT_THROW((void)fit.predict(DenseMatrix<double>(1,2,{1,std::numeric_limits<double>::infinity()})),
        std::invalid_argument);
}
} // namespace
