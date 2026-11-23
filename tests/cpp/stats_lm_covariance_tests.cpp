#include <datamunge/stats/lm.hpp>
#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <stdexcept>

using datamunge::stats::HcCovariance;
using datamunge::stats::LM;

namespace {

LM weighted_model() {
    datamunge::dstruct::DataFrame data;
    data.add_column("x", std::vector<double>{-2, -1, 0, 1, 3, 4, 6, 8});
    data.add_column("y", std::vector<double>{3, 2, 0, 4, 7, 4, 9, 12});
    data.add_column("w", std::vector<double>{1, 2, .5, 1, 3, 1.25, 2, 1});
    return LM(data, "y ~ x", {.weights_column = "w"});
}

void expect_covariance(const datamunge::linalg::DenseMatrix<double>& actual,
                       const std::array<double, 3>& reference, double tolerance = 1e-12) {
    ASSERT_EQ(actual.rows(), 2U);
    ASSERT_EQ(actual.cols(), 2U);
    EXPECT_NEAR(actual(0, 0), reference[0], tolerance);
    EXPECT_NEAR(actual(0, 1), reference[1], tolerance);
    EXPECT_NEAR(actual(1, 1), reference[2], tolerance);
    EXPECT_DOUBLE_EQ(actual(0, 1), actual(1, 0));
}

} // namespace

TEST(LmCovariance, WeightedHc0ThroughHc3MatchRScoreCrossProducts) {
    const auto model = weighted_model();
    expect_covariance(model.heteroscedastic_covariance(HcCovariance::HC0),
                      {0.2713132351857597, -0.02808720315692070, 0.01289700225066962});
    expect_covariance(model.heteroscedastic_covariance(HcCovariance::HC1),
                      {0.3617509802476795, -0.03744960420922760, 0.01719600300089283});
    expect_covariance(model.heteroscedastic_covariance(HcCovariance::HC2),
                      {0.3512890593695578, -0.03945906499421164, 0.01748629764929261});
    expect_covariance(model.heteroscedastic_covariance(),
                      {0.4606203287900116, -0.05590783184807280, 0.02405929928484468});
}

TEST(LmCovariance, NeweyWestUsesBartlettLagsAndOptionalDegreesOfFreedomScaling) {
    const auto model = weighted_model();
    const std::array<double, 3> reference{0.124725202244969732, -0.024685860609929451, 0.008307510859467588};
    expect_covariance(model.newey_west_covariance(2), reference);
    expect_covariance(model.newey_west_covariance(2, true),
                      {reference[0] * 8 / 6, reference[1] * 8 / 6, reference[2] * 8 / 6});
    EXPECT_EQ(model.newey_west_covariance(0).data(),
              model.heteroscedastic_covariance(HcCovariance::HC0).data());
    EXPECT_EQ(model.newey_west_covariance(0, true).data(),
              model.heteroscedastic_covariance(HcCovariance::HC1).data());
}

TEST(LmCovariance, ClusterCovarianceMatchesROneTwoAndThreeWayInclusionExclusion) {
    const auto model = weighted_model();
    const std::vector<std::int64_t> first{2, 2, 2, 3, 3, 4, 4, 4};
    const std::vector<std::int64_t> second{8, 8, 9, 8, 9, 8, 9, 9};
    const std::vector<std::int64_t> third{1, 1, 1, 2, 2, 2, 3, 3};
    const auto one_way = model.cluster_robust_covariance({first});
    expect_covariance(one_way.covariance, {0.10369168870615920, -0.00185185671349177, 0.00085457462037116});
    const auto two_way = model.cluster_robust_covariance({first, second}, true);
    expect_covariance(two_way.covariance, {-0.10803392038535373, -0.01281242302511369, 0.04002180927134112});
    const auto three_way = model.cluster_robust_covariance({first, second, third});
    expect_covariance(three_way.covariance, {-0.19464090931889311, -0.01075280799078273, 0.01194202115797732});
    ASSERT_EQ(three_way.components.size(), 7U);
    EXPECT_EQ(three_way.components.back().dimensions, (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_DOUBLE_EQ(three_way.components.back().multiplier, 1);
    ASSERT_EQ(two_way.components.size(), 3U);
    const auto& joint = two_way.components.back();
    EXPECT_EQ(joint.dimensions, (std::vector<std::size_t>{0, 1}));
    ASSERT_EQ(joint.groups.size(), 6U);
    EXPECT_NEAR(joint.multiplier, -(6.0 / 5) * (7.0 / 6), 1e-14);
    EXPECT_EQ(joint.groups.front().labels, (std::vector<std::int64_t>{2, 8}));
    EXPECT_EQ(joint.groups.front().observations, 2U);
}

TEST(LmCovariance, SparseAndDenseFitsAgreeAndSingletonClustersAreHc0) {
    const std::vector<double> y{3, 1, 6, 5, 9};
    const LM dense(y, datamunge::Mat<>{{0}, {1}, {2}, {3}, {4}});
    const LM sparse(y, datamunge::SpMat<>{{0}, {1}, {2}, {3}, {4}});
    const auto compare = [](const auto& a, const auto& b) {
        for (std::size_t i = 0; i < a.data().size(); ++i)
            EXPECT_NEAR(a.data()[i], b.data()[i], 1e-12);
    };
    compare(dense.heteroscedastic_covariance(), sparse.heteroscedastic_covariance());
    compare(dense.newey_west_covariance(2), sparse.newey_west_covariance(2));
    compare(dense.heteroscedastic_covariance(HcCovariance::HC0),
            sparse.cluster_robust_covariance({{1, 2, 3, 4, 5}}).covariance);
}

TEST(LmCovariance, UsesFittedRowOrderAndRejectsUndefinedCorrections) {
    datamunge::dstruct::DataFrame data;
    data.add_column("x", std::vector<std::optional<double>>{0, std::nullopt, 1, 2, 3});
    data.add_column("y", std::vector<double>{2, 200, 1, 3, 6});
    const LM model(data, "y ~ x");
    const LM direct({2, 1, 3, 6}, datamunge::Mat<>{{0}, {1}, {2}, {3}});
    const auto current = model.newey_west_covariance(1);
    const auto expected = direct.newey_west_covariance(1);
    expect_covariance(current, {expected(0, 0), expected(0, 1), expected(1, 1)});
    EXPECT_THROW((void)model.cluster_robust_covariance({{0, 1, 2, 3, 4}}), std::invalid_argument);
    EXPECT_THROW((void)model.cluster_robust_covariance({}), std::invalid_argument);
    EXPECT_THROW((void)model.cluster_robust_covariance({{1, 1, 1, 1}}, true), std::domain_error);
    EXPECT_THROW((void)model.newey_west_covariance(4), std::invalid_argument);
    EXPECT_THROW((void)model.heteroscedastic_covariance(static_cast<HcCovariance>(12)), std::invalid_argument);
    const LM unit_leverage({0, 1, 5}, datamunge::Mat<>{{0}, {0}, {1}});
    EXPECT_THROW((void)unit_leverage.heteroscedastic_covariance(HcCovariance::HC2), std::domain_error);
    EXPECT_THROW((void)unit_leverage.heteroscedastic_covariance(HcCovariance::HC3), std::domain_error);
}
