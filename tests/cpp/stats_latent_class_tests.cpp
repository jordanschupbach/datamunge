#include <datamunge/stats/latent_class_analysis.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_latent_class_analysis;

TEST(LatentClassAnalysis, PolytomousFitMatchesPoLCAAndKnownPopulation) {
    const std::array<std::array<int,3>,4> a{{{4,1,0},{3,2,0},{4,1,0},{3,1,1}}};
    const std::array<std::array<int,3>,4> b{{{1,4,0},{1,4,0},{1,4,0},{1,1,3}}};
    const std::vector<std::size_t> levels{2,2,2,3};
    std::vector<std::int32_t> values;
    for (int item4 = 0; item4 < 3; ++item4)
        for (int item3 = 0; item3 < 2; ++item3)
            for (int item2 = 0; item2 < 2; ++item2)
                for (int item1 = 0; item1 < 2; ++item1) {
                    const std::array<std::int32_t,4> row{item1,item2,item3,item4};
                    int count_a = 3, count_b = 2;
                    for (std::size_t j = 0; j < row.size(); ++j) {
                        count_a *= a[j][row[j]];
                        count_b *= b[j][row[j]];
                    }
                    for (int repeat = 0; repeat < count_a + count_b; ++repeat)
                        values.insert(values.end(),row.begin(),row.end());
                }
    values.insert(values.end(),4,-1);
    const DenseMatrix<std::int32_t> responses(3126,4,std::move(values));
    const auto fit = fit_latent_class_analysis(responses,levels,
        {.starts=12,.max_iterations=2000,.relative_tolerance=1e-15,.seed=76});
    ASSERT_TRUE(fit.converged);
    EXPECT_NEAR(fit.log_likelihood,-9205.457128823,1e-8);
    EXPECT_NEAR(fit.class_probabilities[0],.59999937693453,4e-6);
    EXPECT_NEAR(fit.class_probabilities[1],.40000062306547,4e-6);
    ASSERT_EQ(fit.item_probabilities.size(),4u);
    for (std::size_t j = 0; j < levels.size(); ++j)
        for (std::size_t c = 0; c < levels[j]; ++c) {
            EXPECT_NEAR(fit.item_probabilities[j](0,c),a[j][c]/5.0,4e-6);
            EXPECT_NEAR(fit.item_probabilities[j](1,c),b[j][c]/5.0,4e-6);
        }
    EXPECT_NEAR(fit.posterior(0,0),.9953916625960960,1e-6);
    EXPECT_DOUBLE_EQ(fit.posterior(0,0),fit.posterior(1,0));
    EXPECT_DOUBLE_EQ(fit.posterior(3125,0),fit.class_probabilities[0]);
    EXPECT_DOUBLE_EQ(fit.posterior(3125,1),fit.class_probabilities[1]);
    for (std::size_t i = 0; i < responses.rows(); ++i)
        EXPECT_NEAR(fit.posterior(i,0)+fit.posterior(i,1),1,2e-16);
}

TEST(LatentClassAnalysis, MarginalizesPartialResponsesAndKeepsAbsentLevels) {
    const DenseMatrix<std::int32_t> responses(5,2,{0,0,0,1,-1,1,1,-1,-1,-1});
    const auto fit = fit_latent_class_analysis(responses,{3,2},{.classes=1});
    ASSERT_TRUE(fit.converged);
    EXPECT_DOUBLE_EQ(fit.item_probabilities[0](0,0),2.0/3);
    EXPECT_DOUBLE_EQ(fit.item_probabilities[0](0,1),1.0/3);
    EXPECT_DOUBLE_EQ(fit.item_probabilities[0](0,2),0);
    EXPECT_DOUBLE_EQ(fit.item_probabilities[1](0,1),2.0/3);
    EXPECT_NEAR(fit.log_likelihood,4*std::log(2.0/3)+2*std::log(1.0/3),1e-14);
}

TEST(LatentClassAnalysis, ReportsIterationLimitAndRejectsInvalidInput) {
    DenseMatrix<std::int32_t> responses(5,3,{0,0,0,0,1,0,1,0,1,1,1,1,1,-1,0});
    const auto fit = fit_latent_class_analysis(responses,{2,2,2},
        {.starts=1,.max_iterations=1,.relative_tolerance=1e-15,.seed=23});
    EXPECT_EQ(fit.iterations,1u);
    EXPECT_FALSE(fit.converged);
    EXPECT_TRUE(std::isfinite(fit.log_likelihood));
    for (std::size_t i = 0; i < responses.rows(); ++i) {
        double scores[2]{fit.class_probabilities[0],fit.class_probabilities[1]};
        for (std::size_t k = 0; k < 2; ++k)
            for (std::size_t j = 0; j < responses.cols(); ++j)
                if (responses(i,j) != -1)
                    scores[k] *= fit.item_probabilities[j](k,responses(i,j));
        EXPECT_NEAR(fit.posterior(i,0),scores[0]/(scores[0]+scores[1]),1e-15);
    }

    EXPECT_THROW((void)fit_latent_class_analysis(responses,{2,2,2},{.starts=0}),std::invalid_argument);
    EXPECT_THROW((void)fit_latent_class_analysis(responses,{2,2,2},{.classes=6}),std::invalid_argument);
    responses(0,0) = 2;
    EXPECT_THROW((void)fit_latent_class_analysis(responses,{2,2,2}),std::invalid_argument);
    responses(0,0) = 0;
    for (std::size_t i = 0; i < responses.rows(); ++i)
        responses(i,1) = -1;
    EXPECT_THROW((void)fit_latent_class_analysis(responses,{2,2,2}),std::invalid_argument);
}
}
