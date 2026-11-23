#include <datamunge/stats/variance_components.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_variance_components;
using datamunge::stats::VarianceComponentsStatus;

TEST(VarianceComponents, CrossedAndNestedModelsMatchIndependentMLAndREML) {
    // Base R assembled marginal covariances and optimized their profiled
    // likelihood using separate Cholesky solves and optim().
    const std::array<double,9> reference[][2]{
        {{2.091846098842,1.385693363938,1.788380955491,.725979453132,.142451039046,
          -34.098622046543,.014787473962,-1.335126363663,.874527605299},
         {2.091934378789,1.385327001750,1.611499887646,.662243173884,.135169247326,
          -33.339585656925,.014022781635,-1.333498287748,.873287792944}},
        {{3.194627733443,-.331840012131,2.062401223780,.641927005007,.070623001119,
          -26.561637261070,.003903509845,-1.690765618417,-.153138579013},
         {3.194386852324,-.331468463122,1.490783067979,.643162267061,.066719146804,
          -25.257171682692,.003688427266,-1.629880221275,-.212066083760}},
    };
    for (const bool nested : {false,true}) {
        std::vector<std::array<std::size_t,2>> labels;
        std::vector<double> x, y;
        if (nested) {
            const double site[]{-1.5,.6,-.15,1.6};
            const double unit[]{-.4,.85,-1.2,.6,-.7,.15,.55,.22,-.63,.30,-.5,.93};
            for (int s = 0; s < 4; ++s)
                for (int u = 0; u < 3; ++u)
                    for (int r = 0; r < 3; ++r) {
                        if ((s + u + r) % 7 == 5)
                            continue;
                        labels.push_back({std::size_t(s), std::size_t(3*s + u)});
                        x.push_back(r + s/6. - 1);
                        y.push_back(3 - .35*x.back() + site[s] + unit[3*s + u] +
                                    .42 * std::sin(2*s + 3*u + r));
                    }
        } else {
            const double person[]{-1.4,.8,-.6,1.65,1.35,-1.8,.75};
            const double item[]{.9,-1.2,.25,.68,-.63};
            for (int p = 0; p < 7; ++p)
                for (int i = 0; i < 5; ++i) {
                    if ((2*p + i) % 7 == 4)
                        continue;
                    labels.push_back({std::size_t(p), std::size_t(i)});
                    x.push_back(((2*p - i + 9) % 9) / 4. - 1);
                    y.push_back(2 + 1.4*x.back() + person[p] + item[i] + .46 * std::cos(3*p + 2*i));
                }
        }
        DenseMatrix<double> fixed(y.size(), 2, 1);
        std::vector<DenseMatrix<double>> random{
            DenseMatrix<double>(y.size(), nested ? 4 : 7), DenseMatrix<double>(y.size(), nested ? 12 : 5)};
        for (std::size_t j = 0; j < y.size(); ++j) {
            fixed(j, 1) = x[j];
            random[0](j, labels[j][0]) = 1;
            random[1](j, labels[j][1]) = 1;
        }
        for (const bool reml : {true,false}) {
            SCOPED_TRACE((nested ? 10 : 0) + reml);
            const auto fit = fit_variance_components(fixed, y, random, {.reml = reml});
            ASSERT_EQ(fit.status, VarianceComponentsStatus::Converged);
            const auto& ref = reference[nested][!reml];
            EXPECT_NEAR(fit.coefficients[0], ref[0], 1e-7);
            EXPECT_NEAR(fit.coefficients[1], ref[1], 1e-7);
            EXPECT_NEAR(fit.variances[0], ref[2], 3e-6);
            EXPECT_NEAR(fit.variances[1], ref[3], 3e-6);
            EXPECT_NEAR(fit.residual_variance, ref[4], 3e-7);
            EXPECT_NEAR(fit.log_likelihood, ref[5], 1e-9);
            EXPECT_NEAR(fit.covariance(1,1), ref[6], 1e-7);
            EXPECT_NEAR(fit.effects[0][labels[0][0]], ref[7], 5e-7);
            EXPECT_NEAR(fit.effects[1][labels[0][1]], ref[8], 5e-7);
            EXPECT_NEAR(fit.fitted[0], fit.predict(fixed)[0] + ref[7] + ref[8], 1e-6);
            const DenseMatrix<double> known(1, 2, {1,x[0]});
            DenseMatrix<double> first(1, random[0].cols()), new_group(1, random[1].cols());
            first(0, labels[0][0]) = 1;
            EXPECT_NEAR(fit.predict(known, {first,new_group})[0], fit.predict(known)[0] + ref[7], 1e-6);
        }
    }
}

TEST(VarianceComponents, EstimatesAnExactBoundaryAndRestoresRandomDesignUnits) {
    const DenseMatrix<double> x(24, 1, 1);
    DenseMatrix<double> person(24, 3), item(24, 4);
    std::vector<double> y;
    for (int p = 0; p < 3; ++p)
        for (int i = 0; i < 4; ++i)
            for (int r = 0; r < 2; ++r) {
                person(y.size(), p) = 2;
                item(y.size(), i) = 1;
                y.push_back(2 + (p - 1) + (r ? .2 : -.2));
            }
    const auto fit = fit_variance_components(x, y, {person,item});
    ASSERT_EQ(fit.status, VarianceComponentsStatus::Boundary);
    const auto residual = 24 * .04 / 21;
    EXPECT_NEAR(fit.coefficients[0], 2, 1e-12);
    EXPECT_NEAR(fit.residual_variance, residual, 1e-8);
    EXPECT_NEAR(fit.variances[0], (8 - residual) / 32, 2e-7);
    EXPECT_DOUBLE_EQ(fit.variances[1], 0);
    EXPECT_EQ(fit.effects[1], (std::vector<double>{0,0,0,0}));
    EXPECT_NEAR(fit.effects[0][2], (1 - residual / 8) / 2, 2e-7);
}

TEST(VarianceComponents, RejectsNonidentifiableComponentsAndMissingResidualVariation) {
    const DenseMatrix<double> x(6, 1, 1), z(6, 3, {1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,0,1, 0,0,1});
    const std::vector<double> y{0,1,3,4,6,7};
    EXPECT_THROW((void)fit_variance_components(x, y, {x}), std::invalid_argument);
    EXPECT_THROW((void)fit_variance_components(x, y, {z,z}), std::invalid_argument);
    EXPECT_THROW((void)fit_variance_components(x, y, {DenseMatrix<double>::identity(6)}), std::invalid_argument);
    EXPECT_THROW((void)fit_variance_components(x, {1,1,1,1,1,1}, {z}), std::invalid_argument);
}

TEST(VarianceComponents, SeparatesThreeOrthogonalBlocksIncludingRandomSlopes) {
    DenseMatrix<double> x(32, 2, 1);
    std::vector<DenseMatrix<double>> z(3, DenseMatrix<double>(32, 4));
    const double person[]{-1,0,0,1}, item[]{-.7,-.3,.2,.8}, slope[]{.9,-.7,-.4,.2};
    std::vector<double> y;
    for (int p = 0; p < 4; ++p)
        for (int i = 0; i < 4; ++i)
            for (const double t : {-1.,1.}) {
                const auto row = y.size();
                x(row, 1) = t;
                z[0](row, p) = 1;
                z[1](row, i) = 1;
                z[2](row, p) = t;
                const auto error = (p+i) % 2 ? .3 : -.3;
                y.push_back(2 + .7*t + person[p] + item[i] + slope[p]*t + error);
            }
    const auto fit = fit_variance_components(x, y, z);
    ASSERT_EQ(fit.status, VarianceComponentsStatus::Converged);
    const double residual = 32 * .09 / 21;
    EXPECT_NEAR(fit.residual_variance, residual, 1e-8);
    EXPECT_NEAR(fit.coefficients[0], 2, 1e-12);
    EXPECT_NEAR(fit.coefficients[1], .7, 1e-12);
    const double expected[]{2./3 - residual/8, 1.26/3 - residual/8, 1.5/3 - residual/8};
    for (std::size_t j = 0; j < 3; ++j)
        EXPECT_NEAR(fit.variances[j], expected[j], 1e-7);
    EXPECT_NEAR(fit.effects[2][0], slope[0] * expected[2] / (expected[2] + residual/8), 1e-8);
}
}
