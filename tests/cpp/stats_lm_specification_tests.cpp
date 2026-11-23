#include <datamunge/stats/lm.hpp>
#include <datamunge/random/random.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

using datamunge::stats::LM;
using datamunge::linalg::DenseMatrix;

namespace {

LM weighted_model() {
    datamunge::dstruct::DataFrame data;
    data.add_column("x", std::vector<double>{-2, -1, 0, 1, 3, 4, 6, 8});
    data.add_column("y", std::vector<double>{3, 2, 0, 4, 7, 4, 9, 12});
    data.add_column("w", std::vector<double>{1, 2, .5, 1, 3, 1.25, 2, 1});
    return LM(data, "y ~ x", {.weights_column = "w"});
}

} // namespace

TEST(LmSpecification, GaussianAndStudentizedBreuschPaganAndWhiteMatchR) {
    const auto model = weighted_model();
    const DenseMatrix<double> z(8, 1, {-2, -1, 0, 1, 3, 4, 6, 8});
    const auto koenker = model.breusch_pagan(z);
    EXPECT_NEAR(koenker.statistic, 0.00231847841796013, 1e-13);
    EXPECT_NEAR(koenker.p_value, 0.9615962353868033, 1e-13);
    EXPECT_DOUBLE_EQ(koenker.parameter1, 1);
    EXPECT_NEAR(koenker.estimate1, koenker.statistic / 8, 1e-15);
    const auto original = model.breusch_pagan(z, false);
    EXPECT_NEAR(original.statistic, 0.00172456782102559, 1e-13);
    EXPECT_NEAR(original.estimate1, koenker.estimate1, 1e-15);
    const auto white = model.white_heteroscedasticity(z);
    EXPECT_NEAR(white.statistic, 0.9929236311445161, 1e-13);
    EXPECT_NEAR(white.p_value, 0.6086804780318740, 1e-13);
    EXPECT_DOUBLE_EQ(white.parameter1, 2);
}

TEST(LmSpecification, WhiteExpansionCountsAliasedIndicatorsAndDuplicateColumnsOnce) {
    const LM model({1, 3, 6, 6, 10, 10, 15, 13, 20},
                   datamunge::Mat<>{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
    DenseMatrix<double> z(9, 3);
    for (std::size_t i = 0; i < z.rows(); ++i) {
        z(i, 0) = static_cast<double>(i % 2);
        z(i, 1) = z(i, 0) + 1e12;
        z(i, 2) = 1;
    }
    const auto white = model.white_heteroscedasticity(z);
    EXPECT_DOUBLE_EQ(white.parameter1, 1);
    EXPECT_NEAR(white.statistic, 0.5908842414476503, 1e-12);
    EXPECT_NEAR(white.statistic, model.breusch_pagan(z).statistic, 1e-12);
}

TEST(LmSpecification, ResetMatchesWeightedPartialFTestsAndSparseFits) {
    const auto model = weighted_model();
    const auto default_fit = model.ramsey_reset();
    EXPECT_NEAR(default_fit.statistic, 0.2500904849642658, 1e-12);
    EXPECT_NEAR(default_fit.p_value, 0.7900599101410463, 1e-12);
    EXPECT_DOUBLE_EQ(default_fit.parameter1, 2);
    EXPECT_DOUBLE_EQ(default_fit.parameter2, 4);
    EXPECT_NEAR(model.ramsey_reset({4, 2}).statistic, 0.2435637804316321, 1e-12);
    const std::vector<double> y{3, 4, 5, 7, 10, 12, 14};
    const LM dense(y, datamunge::Mat<>{{0}, {1}, {2}, {3}, {4}, {5}, {6}});
    const LM sparse(y, datamunge::SpMat<>{{0}, {1}, {2}, {3}, {4}, {5}, {6}});
    EXPECT_NEAR(dense.ramsey_reset().statistic, sparse.ramsey_reset().statistic, 1e-10);
}

TEST(LmSpecification, ResetHandlesExactlyFittedAugmentationAndRejectsInvalidDesigns) {
    const LM quadratic({1, 2, 5, 10, 17, 26}, datamunge::Mat<>{{0}, {1}, {2}, {3}, {4}, {5}});
    const auto fit = quadratic.ramsey_reset({2});
    EXPECT_TRUE(std::isinf(fit.statistic));
    EXPECT_DOUBLE_EQ(fit.p_value, 0);
    EXPECT_THROW((void)quadratic.ramsey_reset({2, 2}), std::invalid_argument);
    EXPECT_THROW((void)quadratic.ramsey_reset({0, 2}), std::invalid_argument);
    const LM two_levels({1, 0, 2, 4, 7, 4}, datamunge::Mat<>{{0}, {0}, {0}, {1}, {1}, {1}});
    EXPECT_THROW((void)two_levels.ramsey_reset(), std::domain_error);
    const LM exact({1, 3, 5, 7}, datamunge::Mat<>{{0}, {1}, {2}, {3}});
    EXPECT_THROW((void)exact.ramsey_reset({2}), std::domain_error);
}

TEST(LmSpecification, ValidatesFittedRowsAndVariancePredictorRank) {
    datamunge::dstruct::DataFrame data;
    data.add_column("x", std::vector<std::optional<double>>{0, 1, std::nullopt, 2, 3, 4});
    data.add_column("y", std::vector<double>{3, 2, 200, 4, 9, 8});
    const LM model(data, "y ~ x");
    EXPECT_THROW((void)model.breusch_pagan(DenseMatrix<double>(6, 1)), std::invalid_argument);
    EXPECT_THROW((void)model.white_heteroscedasticity(DenseMatrix<double>(5, 1, 1)), std::domain_error);
    EXPECT_THROW((void)model.breusch_pagan(DenseMatrix<double>(5, 1, std::numeric_limits<double>::infinity())),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(model.white_heteroscedasticity(DenseMatrix<double>(5, 1, {0, 1, 2, 3, 4})).parameter1, 2);
}

TEST(LmSpecification, BreuschGodfreyMatchesWeightedRAuxiliaryRegression) {
    const auto model = weighted_model();
    const auto lm = model.breusch_godfrey(2);
    const auto f = model.breusch_godfrey(2, true);
    EXPECT_NEAR(lm.statistic, 4.9118106584442813, 1e-12);
    EXPECT_NEAR(lm.p_value, std::exp(-lm.statistic / 2), 1e-12);
    EXPECT_DOUBLE_EQ(lm.parameter1, 2);
    EXPECT_DOUBLE_EQ(lm.parameter2, 0);
    EXPECT_NEAR(lm.estimate1, lm.statistic / 8, 1e-15);
    EXPECT_NEAR(f.statistic, 3.1810294740347018, 1e-12);
    EXPECT_NEAR(f.p_value, std::pow(1 + f.statistic / 2, -2), 1e-12);
    EXPECT_DOUBLE_EQ(f.parameter1, 2);
    EXPECT_DOUBLE_EQ(f.parameter2, 4);

    const std::vector<double> y{3, 2, 0, 4, 7, 4, 9, 12};
    const LM sparse(y, datamunge::SpMat<>{{-2}, {-1}, {0}, {1}, {3}, {4}, {6}, {8}});
    EXPECT_NEAR(sparse.breusch_godfrey(2).statistic, 5.2305672106576218, 1e-12);
    EXPECT_THROW((void)model.breusch_godfrey(0), std::invalid_argument);
    EXPECT_THROW((void)model.breusch_godfrey(8), std::invalid_argument);
    EXPECT_THROW((void)model.breusch_godfrey(7), std::domain_error);
}

TEST(LmSpecification, DurbinWatsonUsesTheAnalyticThreeRowConditionalNull) {
    const LM fit({1, 0, 0}, datamunge::Mat<>(3, 0));
    using datamunge::random::SplitMix64;
    using datamunge::stats::Alternative;
    SplitMix64 positive_rng(731), negative_rng(731), both_rng(731);
    const auto positive = fit.durbin_watson(positive_rng, 19999, Alternative::Greater);
    const auto negative = fit.durbin_watson(negative_rng, 19999, Alternative::Less);
    const auto both = fit.durbin_watson(both_rng, 19999);
    // The two residual-space eigenvalues are 1 and 3. The ratio is
    // 1 + 2 * Beta(1/2, 1/2), so P(D <= 1.5) is exactly 1/3.
    EXPECT_NEAR(positive.statistic, 1.5, 1e-14);
    EXPECT_NEAR(positive.p_value, 1.0 / 3, 0.012);
    EXPECT_NEAR(negative.p_value, 2.0 / 3, 0.012);
    EXPECT_DOUBLE_EQ(both.p_value, 2 * positive.p_value);
    EXPECT_DOUBLE_EQ(both.parameter1, 19999);
}

TEST(LmSpecification, DurbinWatsonUsesWeightedCoordinatesAndHandlesTies) {
    const auto model = weighted_model();
    datamunge::random::SplitMix64 rng(31), repeat(31);
    const auto result = model.durbin_watson(rng, 199);
    EXPECT_NEAR(result.statistic, 2.2822983347006618, 1e-14);
    EXPECT_DOUBLE_EQ(result.p_value, model.durbin_watson(repeat, 199).p_value);
    const LM pair({1, 3}, datamunge::Mat<>(2, 0));
    EXPECT_DOUBLE_EQ(pair.durbin_watson(rng, 21).p_value, 1);
    EXPECT_THROW((void)model.durbin_watson(rng, 0), std::invalid_argument);
    const LM exact({1, 3, 5, 7}, datamunge::Mat<>{{0}, {1}, {2}, {3}});
    EXPECT_THROW((void)exact.durbin_watson(rng, 21), std::domain_error);
}
