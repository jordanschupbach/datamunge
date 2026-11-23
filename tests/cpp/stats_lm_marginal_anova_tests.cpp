#include <datamunge/stats/lm.hpp>

#include <array>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::dstruct::DataFrame;
using datamunge::stats::FactorContrast;
using datamunge::stats::LM;
using datamunge::stats::MarginalAnovaType;

TEST(MarginalAnova, MatchesWeightedNestedFitsUnderTreatmentAndSumCoding) {
    DataFrame data;
    data.add_column("method", std::vector<std::string>{
        "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"});
    data.add_column("batch", std::vector<std::string>{
        "Cool", "Cool", "Cool", "Warm", "Warm", "Cool", "Cool", "Warm",
        "Warm", "Warm", "Cool", "Cool", "Warm", "Warm", "Warm"});
    data.add_column("y", std::vector<double>{8, 9, 10, 12, 13, 12, 14, 21, 22, 24, 13, 15, 12, 14, 14});
    data.add_column("w", std::vector<double>{1, 2, 1, 1, 3, 2, 1, 1, 2, 1, 1, 2, 2, 1, 3});
    const LM treatment(data, "y ~ method * batch", {.weights_column = "w"});
    const LM sum(data, "y ~ method * batch", {
        .weights_column = "w",
        .contrasts = {{"method", FactorContrast::Sum}, {"batch", FactorContrast::Sum}},
    });

    // Base R model.matrix() with each contrast coding, followed by nested lm.wfit() fits.
    const auto ii = sum.marginal_anova(MarginalAnovaType::II);
    const auto ii_treatment = treatment.marginal_anova(MarginalAnovaType::II);
    const auto iii_sum = sum.marginal_anova(MarginalAnovaType::III);
    const auto iii_treatment = treatment.marginal_anova(MarginalAnovaType::III);
    ASSERT_EQ(ii.size(), 4u);
    ASSERT_EQ(iii_sum.size(), ii.size());
    EXPECT_EQ(ii[0].term, "method");
    EXPECT_EQ(ii[1].term, "batch");
    EXPECT_EQ(ii[2].term, "method:batch");
    EXPECT_EQ(ii[2].degrees_of_freedom, 2u);
    EXPECT_EQ(ii[3].term, "Residuals");
    EXPECT_EQ(ii[3].degrees_of_freedom, 9u);
    EXPECT_NEAR(ii[3].sum_sq, 18.1666666666667, 1e-12);
    const std::array ii_ss{189.375892857143, 84.1508928571429, 103.414583333333};
    const std::array iii_sum_ss{160.059027777778, 96.0701754385965, 103.414583333333};
    const std::array iii_treatment_ss{52.7666666666667, 28.125, 103.414583333333};
    for (std::size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(ii[j].sum_sq, ii_ss[j], 2e-11);
        EXPECT_NEAR(ii_treatment[j].sum_sq, ii_ss[j], 2e-11);
        EXPECT_NEAR(iii_sum[j].sum_sq, iii_sum_ss[j], 2e-11);
        EXPECT_NEAR(iii_treatment[j].sum_sq, iii_treatment_ss[j], 2e-11);
    }
    EXPECT_NEAR(ii[0].f_value, 46.9096248361729, 1e-11);
    EXPECT_NEAR(iii_sum[0].p_value, 3.44645167914427e-5, 1e-14);
    EXPECT_NEAR(iii_treatment[0].p_value, .0021772683222871, 1e-13);
    EXPECT_NEAR(sum.coefficients()[0], 14.0555555555556, 1e-12);
    const auto centered = sum.predict(data);
    const auto raw = treatment.predict(data);
    for (std::size_t i = 0; i < data.nrows(); ++i)
        EXPECT_NEAR(centered[i], raw[i], 1e-12);
}

TEST(MarginalAnova, AgreesForIndependentTermsAndSparseMatrixFits) {
    const std::vector<double> y{1, 3, 2, 5, 8, 6};
    const datamunge::Mat<> x{{0, 1}, {1, 0}, {2, 0}, {3, 1}, {4, 0}, {5, 1}};
    const datamunge::SpMat<> sparse{{0, 1}, {1, 0}, {2, 0}, {3, 1}, {4, 0}, {5, 1}};
    const LM dense_fit(y, x, false), sparse_fit(y, sparse, false);
    const auto ii = dense_fit.marginal_anova(MarginalAnovaType::II);
    const auto iii = dense_fit.marginal_anova(MarginalAnovaType::III);
    const auto sparse_iii = sparse_fit.marginal_anova(MarginalAnovaType::III);
    ASSERT_EQ(ii.size(), 3u);
    for (std::size_t j = 0; j < 2; ++j) {
        EXPECT_DOUBLE_EQ(ii[j].sum_sq, iii[j].sum_sq);
        EXPECT_NEAR(sparse_iii[j].sum_sq, iii[j].sum_sq, 1e-10);
    }
    const auto only = LM(y, datamunge::Mat<>{{0}, {1}, {2}, {3}, {4}, {5}}, false);
    EXPECT_NEAR(iii[1].sum_sq, only.rss() - dense_fit.rss(), 1e-11);
    EXPECT_THROW((void)dense_fit.marginal_anova(static_cast<MarginalAnovaType>(-1)), std::invalid_argument);
}

TEST(MarginalAnova, DistinguishesUnrelatedInteractionsWithSharedFactors) {
    DataFrame data;
    data.add_column("a", std::vector<double>{1, 2, 1, 0, 3, -2, -2, 2, 1, 3});
    data.add_column("b", std::vector<double>{1, 1, 0, 3, 2, 1, 2, 4, 0, 1});
    data.add_column("c", std::vector<double>{0, 2, 1, 0, 1, 0, 3, 1, 2, 4});
    data.add_column("y", std::vector<double>{3, 1, 4, 2, 6, 0, 1, 2, 3, 1});
    const LM full(data, "y ~ a + b + c + b:a + c:b");
    const LM with_a(data, "y ~ a + b + c + c:b");
    const LM without_a(data, "y ~ b + c + c:b");
    const auto table = full.marginal_anova(MarginalAnovaType::II);
    ASSERT_EQ(table.front().term, "a");
    EXPECT_NEAR(table.front().sum_sq, without_a.rss() - with_a.rss(), 1e-11);
    EXPECT_THROW((void)LM(data, "y ~ a + b", {.contrasts = {{"b", FactorContrast::Sum}}}),
                 std::invalid_argument);
}

} // namespace
