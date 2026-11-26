#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/stats/knockoffs.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::KnockoffScale;
using datamunge::stats::gaussian_knockoffs;
using datamunge::stats::knockoff_diagonal;
using datamunge::stats::knockoff_filter;
using datamunge::stats::knockoff_threshold;

/// AR(1) correlation 0.6 with standard deviations 1, 2, 0.5 and sqrt(2).
DenseMatrix<double> ar_covariance() {
    const double sd[4]{1,2,.5,std::sqrt(2.0)};
    DenseMatrix<double> sigma(4,4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            sigma(i,j) = std::pow(.6,std::abs(i-j))*sd[i]*sd[j];
    return sigma;
}

// Reference values: knockoff 0.3.6 create.solve_equi / create.solve_sdp(gaptol=1e-10) in R 4.5.3.
TEST(Knockoffs, DiagonalMatchesRKnockoffPackage) {
    const auto equi = knockoff_diagonal(ar_covariance(),KnockoffScale::equicorrelated);
    const std::vector<double> r_equi{.56930001627644056,2.27720006510576223,.14232500406911014,1.13860003255288134};
    for (std::size_t j = 0; j < 4; ++j)
        EXPECT_NEAR(equi[j],r_equi[j],1e-12);
    const auto sdp = knockoff_diagonal(ar_covariance(),KnockoffScale::sdp);
    const std::vector<double> r_sdp{.999999899988489,1.577464630993567,.098591539437098,1.999999799976979};
    for (std::size_t j = 0; j < 4; ++j)
        EXPECT_NEAR(sdp[j],r_sdp[j],1e-7);

    const DenseMatrix<double> g(3,3,{1,.9,.1,.9,1,.2,.1,.2,1});
    for (const double s : knockoff_diagonal(g,KnockoffScale::equicorrelated))
        EXPECT_NEAR(s,.18864609289349393,1e-12);
    const auto sdp3 = knockoff_diagonal(g);
    EXPECT_NEAR(sdp3[0],.23999997598918513,1e-7);
    EXPECT_NEAR(sdp3[1],.11999998799790847,1e-7);
    EXPECT_NEAR(sdp3[2],.99999989999652039,1e-7);
}

// Reference values: knockoff.threshold(W, q, offset) in R.
TEST(Knockoffs, ThresholdMatchesRForBothOffsets) {
    const std::vector<double> w{3.1,-.4,2.2,0,1.5,-2.0,.9,2.7,1.1,-.3,4.0,.8,1.9,2.5,-1.2};
    EXPECT_EQ(knockoff_threshold(w,.1,true),std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(knockoff_threshold(w,.1,false),2.2);
    EXPECT_DOUBLE_EQ(knockoff_threshold(w,.2,true),2.2);
    EXPECT_DOUBLE_EQ(knockoff_threshold(w,.2,false),.8);
    EXPECT_DOUBLE_EQ(knockoff_threshold(w,.3,true),.8);
    EXPECT_DOUBLE_EQ(knockoff_threshold(w,.3,false),.4);
    EXPECT_THROW((void)knockoff_threshold(w,1,true),std::invalid_argument);
    EXPECT_THROW((void)knockoff_threshold({std::nan("")},.1,true),std::invalid_argument);
}

TEST(Knockoffs, JointMomentsMatchTheExchangeableCovariance) {
    const auto sigma = ar_covariance();
    const std::vector<double> mean{1,-2,.5,3};
    const std::size_t n = 60000;
    const auto root = datamunge::linalg::cholesky(sigma).L;
    std::mt19937_64 random(99);
    std::normal_distribution<double> normal;
    DenseMatrix<double> x(n,4);
    for (std::size_t i = 0; i < n; ++i) {
        double z[4];
        for (auto& v : z)
            v = normal(random);
        for (std::size_t j = 0; j < 4; ++j) {
            x(i,j) = mean[j];
            for (std::size_t k = 0; k <= j; ++k)
                x(i,j) += root(j,k)*z[k];
        }
    }
    const auto draw = gaussian_knockoffs(x,mean,sigma,{.scale=KnockoffScale::equicorrelated,.seed=5});
    const auto& s = draw.diagonal;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            double knock = 0, cross = 0;
            for (std::size_t r = 0; r < n; ++r) {
                knock += (draw.knockoffs(r,i)-mean[i])*(draw.knockoffs(r,j)-mean[j]);
                cross += (x(r,i)-mean[i])*(draw.knockoffs(r,j)-mean[j]);
            }
            const double scale = std::sqrt(sigma(i,i)*sigma(j,j));
            EXPECT_NEAR(knock/n,sigma(i,j),.03*scale) << i << "," << j;
            EXPECT_NEAR(cross/n,sigma(i,j)-(i == j ? s[i] : 0.0),.03*scale) << i << "," << j;
        }
    // Closed form 2D - D Sigma^{-1} D, checked against an independent solve.
    const auto inverse = datamunge::linalg::cholesky(sigma).solve(DenseMatrix<double>::identity(4));
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(draw.conditional_covariance(i,j),(i == j ? 2*s[i] : 0.0)-s[i]*inverse(i,j)*s[j],1e-12);
}

TEST(Knockoffs, SuppliedDiagonalControlsCopiesAndValidity) {
    const auto sigma = ar_covariance();
    const DenseMatrix<double> x(3,4,{1,2,3,4,0,0,0,0,-1,.5,2,-3});
    const auto copies = gaussian_knockoffs(x,{0,0,0,0},sigma,{.diagonal={0,0,0,0}});
    EXPECT_EQ(copies.knockoffs.data(),x.data());
    const auto first = gaussian_knockoffs(x,{0,0,0,0},sigma,{.seed=3});
    const auto again = gaussian_knockoffs(x,{0,0,0,0},sigma,{.seed=3});
    const auto other = gaussian_knockoffs(x,{0,0,0,0},sigma,{.seed=4});
    EXPECT_EQ(first.knockoffs.data(),again.knockoffs.data());
    EXPECT_NE(first.knockoffs.data(),other.knockoffs.data());
    EXPECT_THROW((void)gaussian_knockoffs(x,{0,0,0,0},sigma,{.diagonal={3,3,3,3}}),std::invalid_argument);
    EXPECT_THROW((void)gaussian_knockoffs(x,{0,0,0,0},sigma,{.diagonal={-1,0,0,0}}),std::invalid_argument);
    EXPECT_THROW((void)gaussian_knockoffs(x,{0,0,0},sigma),std::invalid_argument);
    EXPECT_THROW((void)knockoff_diagonal(DenseMatrix<double>(2,2,{1,2,2,1})),std::invalid_argument);
    EXPECT_THROW((void)knockoff_diagonal(DenseMatrix<double>(2,2,{1,.5,.4,1})),std::invalid_argument);
}

TEST(Knockoffs, FilterPassesDrawAndSelectsAboveThreshold) {
    const auto sigma = ar_covariance();
    DenseMatrix<double> x(5,4,0.0);
    for (std::size_t i = 0; i < 5; ++i)
        x(i,i%4) = 1;
    const std::vector<double> y{1,2,3,4,5};
    const datamunge::stats::KnockoffFilterOptions options{.fdr=.4,.knockoffs={.seed=8}};
    std::size_t calls = 0;
    const auto fit = knockoff_filter(x,y,{0,0,0,0},sigma,[&](const auto& original,const auto& knock,const auto& response) {
        ++calls;
        EXPECT_EQ(original.data(),x.data());
        EXPECT_EQ(response,y);
        EXPECT_EQ(knock.data(),gaussian_knockoffs(x,{0,0,0,0},sigma,{.seed=8}).knockoffs.data());
        return std::vector<double>{5,4,6,-1};
    },options);
    EXPECT_EQ(calls,1u);
    EXPECT_DOUBLE_EQ(fit.threshold,4); // (1+1)/3 > .4 at t=1; (1+0)/3 <= .4 at t=4
    EXPECT_EQ(fit.selected,(std::vector<std::size_t>{0,1,2}));
    const auto plain = knockoff_filter(x,y,{0,0,0,0},sigma,[](const auto&,const auto&,const auto&) {
        return std::vector<double>{5,4,6,-1};
    },{.fdr=.34,.plus=false});
    // R: knockoff.threshold(c(5,4,6,-1), .34, 0) is 0, since 1/3 <= .34 at t=0; at .3 it is 4.
    EXPECT_DOUBLE_EQ(plain.threshold,0);
    EXPECT_EQ(plain.selected,(std::vector<std::size_t>{0,1,2}));
    EXPECT_THROW((void)knockoff_filter(x,y,{0,0,0,0},sigma,[](const auto&,const auto&,const auto&) {
        return std::vector<double>{1};
    }),std::invalid_argument);
    EXPECT_THROW((void)knockoff_filter(x,y,{0,0,0,0},sigma,{}),std::invalid_argument);
    EXPECT_THROW((void)knockoff_filter(x,{1,2},{0,0,0,0},sigma,[](const auto&,const auto&,const auto&) {
        return std::vector<double>(4,1.0);
    }),std::invalid_argument);
}

} // namespace
