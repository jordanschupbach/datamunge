#include <datamunge/stats/mvn_probability.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::mvn_cdf;
using datamunge::stats::mvn_rectangle_probability;

constexpr double inf = std::numeric_limits<double>::infinity();

double phi(double x) {
    return .5*std::erfc(-x/std::numbers::sqrt2);
}

DenseMatrix<double> equicorrelated(std::size_t d, double rho) {
    DenseMatrix<double> sigma(d,d,rho);
    for (std::size_t j = 0; j < d; ++j)
        sigma(j,j) = 1;
    return sigma;
}

TEST(MvnProbability, OrthantProbabilitiesMatchClosedForms) {
    // P(X1<0, X2<0) = 1/4 + asin(rho)/(2 pi); trivariate orthant = 1/8 + sum asin(rho_ij)/(4 pi).
    for (const double rho : {-.7,-.2,.3,.9}) {
        const auto p = mvn_cdf({0,0},{0,0},equicorrelated(2,rho),{.abs_tol=1e-5,.max_points=400000});
        EXPECT_TRUE(p.converged);
        EXPECT_NEAR(p.value,.25+std::asin(rho)/(2*std::numbers::pi),p.error);
    }
    const DenseMatrix<double> sigma(3,3,{1,.2,-.4,.2,1,.6,-.4,.6,1});
    const double exact = .125+(std::asin(.2)+std::asin(-.4)+std::asin(.6))/(4*std::numbers::pi);
    const auto p = mvn_cdf({0,0,0},{0,0,0},sigma,{.abs_tol=1e-5,.max_points=400000});
    EXPECT_TRUE(p.converged);
    EXPECT_LE(p.error,1e-5);
    EXPECT_NEAR(p.value,exact,p.error);
}

TEST(MvnProbability, IndependentBoxIsProductAndMeanScaleAreHandled) {
    const DenseMatrix<double> sigma(3,3,{4,0,0,0,1,0,0,0,.25});
    const std::vector<double> mean{1,-1,2};
    const auto p = mvn_rectangle_probability({-1,-inf,1.5},{3,0,inf},mean,sigma,{.abs_tol=1e-8});
    const double exact = (phi(1)-phi(-1))*phi(1)*(1-phi(-1));
    EXPECT_NEAR(p.value,exact,1e-8);

    const auto one = mvn_rectangle_probability({-1},{2},{.5},DenseMatrix<double>(1,1,{4.0}));
    EXPECT_DOUBLE_EQ(one.value,phi(.75)-phi(-.75));
    EXPECT_EQ(one.error,0);
}

TEST(MvnProbability, ErrorEstimateCoversTheTruthAndSeedOnlyChangesNoise) {
    // Ten-dimensional equicorrelated orthant: P = 1/(d+1) for rho = 1/2.
    const auto sigma = equicorrelated(10,.5);
    const std::vector<double> zero(10,0.0);
    for (std::uint64_t seed : {1U,2U,3U}) {
        const auto p = mvn_cdf(zero,zero,sigma,{.abs_tol=1e-4,.max_points=400000,.seed=seed});
        EXPECT_TRUE(p.converged);
        EXPECT_NEAR(p.value,1.0/11,p.error);
    }
    const auto loose = mvn_cdf(zero,zero,sigma,{.abs_tol=1e-12,.max_points=20000});
    EXPECT_FALSE(loose.converged);
    EXPECT_LE(loose.points,20000U);
}

// Reference values: mvtnorm 1.3 pmvnorm(..., algorithm = GenzBretz(maxpts = 2e7, abseps = 1e-7)) in R 4.5.3.
TEST(MvnProbability, BoxesMatchMvtnorm) {
    const double r[4][4]{{1,.6,.2,-.3},{.6,1,.4,.1},{.2,.4,1,.5},{-.3,.1,.5,1}};
    const double sd[4]{1,2,.5,1.5};
    DenseMatrix<double> sigma(4,4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            sigma(i,j) = r[i][j]*sd[i]*sd[j];
    const auto p = mvn_rectangle_probability({-1,-inf,0,-2},{1.5,2,inf,1},{.2,0,.1,-.5},sigma,
                                             {.abs_tol=2e-5,.max_points=1000000});
    EXPECT_TRUE(p.converged);
    EXPECT_NEAR(p.value,.268569625878,p.error+1e-7);

    DenseMatrix<double> ar(6,6);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            ar(i,j) = std::pow(.4,std::abs(i-j));
    const auto q = mvn_rectangle_probability(std::vector<double>(6,-1.0),{1,2,1,2,1,2},std::vector<double>(6,0.0),ar,
                                             {.abs_tol=2e-5,.max_points=1000000});
    EXPECT_TRUE(q.converged);
    EXPECT_NEAR(q.value,.199187938934,q.error+1e-7);
}

// Reference values: mvtnorm pmvt(..., type = "shifted") and pmvt(corr = ...) in R 4.5.3.
TEST(MvnProbability, MultivariateTMatchesMvtnorm) {
    const double r[4][4]{{1,.6,.2,-.3},{.6,1,.4,.1},{.2,.4,1,.5},{-.3,.1,.5,1}};
    const double sd[4]{1,2,.5,1.5};
    DenseMatrix<double> sigma(4,4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            sigma(i,j) = r[i][j]*sd[i]*sd[j];
    const auto p = datamunge::stats::mvt_rectangle_probability({-1,-inf,0,-2},{1.5,2,inf,1},{.2,0,.1,-.5},sigma,4,
                                                               {.abs_tol=2e-5,.max_points=2000000});
    EXPECT_TRUE(p.converged);
    EXPECT_NEAR(p.value,.244172805672,p.error+1e-7);
    const auto orthant = datamunge::stats::mvt_rectangle_probability({-inf,-inf},{0,0},{0,0},equicorrelated(2,.5),5,
                                                                     {.abs_tol=1e-5,.max_points=1000000});
    EXPECT_NEAR(orthant.value,1.0/3,orthant.error+1e-7);
    const auto cube = datamunge::stats::mvt_rectangle_probability({-2,-2,-2},{2,2,2},{0,0,0},equicorrelated(3,.3),3,
                                                                  {.abs_tol=2e-5,.max_points=2000000});
    EXPECT_NEAR(cube.value,.708237572523,cube.error+1e-7);
    // One dimension is the Student t distribution function; infinite df is the normal law.
    const auto one = datamunge::stats::mvt_rectangle_probability({-inf},{1},{0},DenseMatrix<double>(1,1,{1.0}),4);
    EXPECT_NEAR(one.value,.81304951684997,1e-12); // pt(1, 4)
    const auto normal = datamunge::stats::mvt_rectangle_probability({-inf,-inf},{0,0},{0,0},equicorrelated(2,.5),inf,
                                                                    {.abs_tol=1e-5,.max_points=1000000});
    EXPECT_NEAR(normal.value,1.0/3,normal.error+1e-7);
    EXPECT_THROW((void)datamunge::stats::mvt_rectangle_probability({0},{1},{0},DenseMatrix<double>(1,1,{1.0}),0),
                 std::invalid_argument);
}

TEST(MvnProbability, DegenerateAndInvalidInputs) {
    const auto sigma = equicorrelated(2,.3);
    EXPECT_EQ(mvn_rectangle_probability({1,0},{1,2},{0,0},sigma).value,0.0);
    EXPECT_NEAR(mvn_rectangle_probability({-inf,-inf},{inf,inf},{0,0},sigma).value,1.0,1e-12);
    EXPECT_THROW((void)mvn_cdf({},{},DenseMatrix<double>()),std::invalid_argument);
    EXPECT_THROW((void)mvn_rectangle_probability({1,0},{0,1},{0,0},sigma),std::invalid_argument);
    EXPECT_THROW((void)mvn_rectangle_probability({inf,0},{inf,1},{0,0},sigma),std::invalid_argument);
    EXPECT_THROW((void)mvn_cdf({0,0},{0,0},DenseMatrix<double>(2,2,{1,1,1,1})),std::invalid_argument);
    EXPECT_THROW((void)mvn_cdf({0,0},{0,0},DenseMatrix<double>(2,2,{1,.2,.3,1})),std::invalid_argument);
    EXPECT_THROW((void)mvn_cdf({0,0},{0,0},sigma,{.shifts=1}),std::invalid_argument);
}

} // namespace
