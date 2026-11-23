#include <datamunge/nn/gaussian_mixture.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace nn = datamunge::nn;
namespace ml = nn::ml;
using Covariance = ml::GaussianMixtureCovariance;

nn::Tensor correlated_clusters() {
    const std::array quantiles{-1.3829941271,-.6744897502,-.2104283942,
        .2104283942,.6744897502,1.3829941271};
    std::vector<double> values;
    for (const int sign : {-1,1})
        for (const double v : quantiles)
            for (const double u : quantiles) {
                values.push_back(4*sign+.86*u);
                values.push_back(sign+.65*u+.38*v);
            }
    return nn::from_vector(values,{72,2});
}

TEST(GaussianMixtureSelection, ComparesCountsAndCovariancesAgainstMclust) {
    const auto x = correlated_clusters();
    const auto fit = ml::select_gaussian_mixture(x,{1,2,3},
        {.starts=8,.max_iterations=500,.relative_tolerance=1e-10,.covariance_floor=.03,.seed=79});
    ASSERT_EQ(fit.candidates.size(),9u);
    const std::array<std::size_t,9> counts{5,4,5,8,9,11,11,14,17};
    const std::array bic_reference{536.0652722237,655.0529324050,536.0652722237,
        350.2091493655,452.9527055153,363.0391477226};
    for (std::size_t i = 0; i < fit.candidates.size(); ++i) {
        const auto& candidate = fit.candidates[i];
        EXPECT_EQ(candidate.parameters,counts[i]);
        EXPECT_EQ(candidate.components,static_cast<int>(i/3+1));
        EXPECT_TRUE(std::isfinite(candidate.bic));
        EXPECT_NEAR(candidate.bic,-2*candidate.log_likelihood+counts[i]*std::log(72.),1e-12);
        if (i < bic_reference.size()) {
            EXPECT_TRUE(candidate.converged);
            EXPECT_NEAR(candidate.bic,bic_reference[i],1e-6);
        }
    }
    ASSERT_EQ(fit.best_index,3u);
    EXPECT_EQ(fit.candidates[fit.best_index].covariance,Covariance::Tied);
    const auto& model = fit.model;
    const auto left = model.means.at({0,0}) < 0 ? 0 : 1;
    EXPECT_GT(fit.responsibilities.at({0,left}),.99999);
    EXPECT_GT(fit.responsibilities.at({71,1-left}),.99999);
    EXPECT_EQ(fit.responsibilities.shape(),(nn::Shape{72,2}));
    const auto scored = ml::GaussianMixture(2,2,.03,Covariance::Tied).Step(x,model);
    EXPECT_NEAR(scored.log_likelihood,fit.candidates[fit.best_index].log_likelihood,1e-12);
    EXPECT_EQ(scored.responsibilities.to_vector<double>(),fit.responsibilities.to_vector<double>());
    EXPECT_GT(model.covariances.at({0,0,1}),.4);
    EXPECT_DOUBLE_EQ(model.covariances.at({0,0,1}),model.covariances.at({1,0,1}));
}

TEST(GaussianMixtureSelection, TiedAndDiagonalUpdatesUseWeightedScatterAndFloor) {
    const auto x = nn::tensor({-3,-1,-2,0,-1,2,1,0,2,1,4,3},{6,2},nn::DType::Float64);
    const ml::GaussianMixtureState state{nn::tensor({.25,.75},{2},nn::DType::Float64),
        nn::tensor({-1,0,2,1},{2,2},nn::DType::Float64),
        nn::tensor({1,0,0,1,1,0,0,1},{2,2,2},nn::DType::Float64)};
    const auto tied = ml::GaussianMixture(2,2,.00001,Covariance::Tied).Step(x,state);
    double pooled[2][2]{};
    double variance[2][2]{};
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < 6; ++i)
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b) {
                    const double term = tied.responsibilities.at({i,k}) *
                        (x.at({i,a})-tied.next.means.at({k,a})) *
                        (x.at({i,b})-tied.next.means.at({k,b}));
                    pooled[a][b] += term / 6;
                    if (a == b)
                        variance[k][a] += term / (6*tied.next.weights.at({k}));
                }
    const auto diagonal = ml::GaussianMixture(2,2,.5,Covariance::Diagonal).Step(x,state);
    for (int k = 0; k < 2; ++k)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                EXPECT_NEAR(tied.next.covariances.at({k,a,b}),pooled[a][b],1e-13);
                EXPECT_NEAR(diagonal.next.covariances.at({k,a,b}),
                    a == b ? std::max(.5,variance[k][a]) : 0,1e-13);
            }
}

TEST(GaussianMixtureSelection, SupportsOneObservationAndValidatesOptions) {
    const auto x = nn::tensor({1,2},{1,2},nn::DType::Float64);
    const auto fit = ml::select_gaussian_mixture(x,{1},{.covariance_floor=.125});
    ASSERT_EQ(fit.candidates.size(),3u);
    EXPECT_EQ(fit.best_index,0u);
    EXPECT_DOUBLE_EQ(fit.model.covariances.at({0,0,0}),.125);
    EXPECT_DOUBLE_EQ(fit.responsibilities.at({0,0}),1);
    const auto early = ml::select_gaussian_mixture(correlated_clusters(),{2},
        {.starts=1,.max_iterations=1,.relative_tolerance=1e-15});
    EXPECT_TRUE(std::isfinite(early.candidates[early.best_index].bic));
    EXPECT_FALSE(early.candidates[early.best_index].converged);
    EXPECT_THROW((void)ml::select_gaussian_mixture(x,{1,1}),std::invalid_argument);
    EXPECT_THROW((void)ml::select_gaussian_mixture(x,{2}),std::invalid_argument);
    EXPECT_THROW((void)ml::select_gaussian_mixture(x,{1},{.starts=0}),std::invalid_argument);
    const auto nan = nn::tensor({std::numeric_limits<double>::quiet_NaN()},{1,1},nn::DType::Float64);
    EXPECT_THROW((void)ml::select_gaussian_mixture(nan,{1}),std::invalid_argument);
}
}
