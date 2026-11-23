#include <datamunge/fda/functional_pca.hpp>

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::fda::functional_pca;
using datamunge::linalg::DenseMatrix;

TEST(FunctionalPCA, IrregularGridMatchesWeightedRSvd) {
    const std::vector<double> grid{0,.3,1.4,2,3.5};
    const DenseMatrix<double> x(6,5,{
        1,2,0,-1,2,
        2,1,3,0,-1,
        -1,4,2,1,0,
        0,3,1,1,1,
        1,1,-1,2,4,
        -2,2,0,4,3
    });
    const auto fit = functional_pca(grid,x,3);
    EXPECT_EQ(fit.rank,5u);
    const std::array weights{.15,.70,.85,1.05,.75};
    for (std::size_t t = 0; t < grid.size(); ++t)
        EXPECT_NEAR(fit.weights[t],weights[t],2e-16);
    const std::array expected_eigenvalues{5.759700871604175,2.169980031589561,.904410493123648};
    for (std::size_t k = 0; k < expected_eigenvalues.size(); ++k) {
        EXPECT_NEAR(fit.eigenvalues[k],expected_eigenvalues[k],2e-13);
        EXPECT_NEAR(fit.explained_fraction[k],expected_eigenvalues[k]/8.86333333333333,2e-13);
    }
    EXPECT_NEAR(fit.eigenfunctions(4,0),.738063251951934,2e-13);
    EXPECT_NEAR(fit.eigenfunctions(3,1),.722651782268841,2e-13);
    EXPECT_NEAR(fit.scores(1,0),-3.039333209516348,2e-13);
    EXPECT_NEAR(fit.scores(5,1),1.576323739292391,2e-13);
    for (std::size_t k = 0; k < 3; ++k) {
        for (std::size_t l = 0; l < 3; ++l) {
            double dot = 0, covariance = 0;
            for (std::size_t t = 0; t < grid.size(); ++t)
                dot += fit.weights[t]*fit.eigenfunctions(t,k)*fit.eigenfunctions(t,l);
            for (std::size_t n = 0; n < x.rows(); ++n)
                covariance += fit.scores(n,k)*fit.scores(n,l)/(x.rows()-1);
            EXPECT_NEAR(dot,k == l ? 1 : 0,1e-13);
            EXPECT_NEAR(covariance,k == l ? fit.eigenvalues[k] : 0,1e-13);
        }
    }
}

TEST(FunctionalPCA, WideLowRankCurvesReconstructInOriginalUnits) {
    const std::vector<double> grid{0,.2,1,1.4,1.7,4,5};
    DenseMatrix<double> x(4,grid.size());
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            x(i,j) = 5 + (static_cast<double>(i)-1.5)*static_cast<double>(j)
                + (i%2 == 0 ? 1 : -1);
    const auto fit = functional_pca(grid,x,2);
    EXPECT_EQ(fit.rank,2u);
    const auto reconstruction = fit.scores*fit.eigenfunctions.transpose();
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            EXPECT_NEAR(reconstruction(i,j)+fit.mean[j],x(i,j),2e-13);
    EXPECT_THROW((void)functional_pca(grid,x,3),std::domain_error);
}

TEST(FunctionalPCA, ValidatesGridCurveValuesAndComponents) {
    DenseMatrix<double> x(3,2,{-1,-1,0,0,1,1});
    EXPECT_THROW((void)functional_pca({0,0},x,1),std::invalid_argument);
    EXPECT_THROW((void)functional_pca({0,1},x,0),std::invalid_argument);
    EXPECT_THROW((void)functional_pca({0,1},DenseMatrix<double>(3,2,4),1),std::domain_error);
    x(1,1) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)functional_pca({0,1},x,1),std::invalid_argument);
}
}
