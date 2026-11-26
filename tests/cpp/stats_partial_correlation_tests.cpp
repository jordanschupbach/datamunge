#include <datamunge/stats/partial_correlation.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_partial_correlation_data.inc"
} // namespace

TEST(PartialCorrelation, MatchesPpcorTests) {
    const std::vector<std::vector<double>> z{pc_z1, pc_z2};
    for (const auto [method, ref] : {std::pair{PartialMethod::pearson, &pc_pearson},
                                     std::pair{PartialMethod::spearman, &pc_spearman}}) {
        const auto p = partial_correlation(pc_x, pc_y, z, {.method = method});
        EXPECT_NEAR(p.estimate, (*ref)[0], 1e-13);
        EXPECT_NEAR(p.statistic, (*ref)[1], 1e-12);
        EXPECT_NEAR(p.p_value, (*ref)[2], 1e-12);
        EXPECT_EQ(p.df, 56.0);
        const auto s = semipartial_correlation(pc_x, pc_y, z, {.method = method});
        EXPECT_NEAR(s.estimate, (*ref)[3], 1e-13);
        // Coefficient test (= partial test), not ppcor's spcor.test statistic.
        EXPECT_NEAR(s.statistic, (*ref)[1], 1e-12);
        EXPECT_NEAR(s.p_value, (*ref)[2], 1e-12);
        EXPECT_GT(std::abs(s.statistic), std::abs((*ref)[4]));
        EXPECT_TRUE(std::isnan(s.lower) && std::isnan(s.upper));
    }
    const auto lm = semipartial_correlation(pc_x, pc_y, z);
    EXPECT_NEAR(lm.statistic, pc_lm_t[0], 1e-11);
    EXPECT_NEAR(lm.p_value, pc_lm_t[1], 1e-12);
    const auto one = partial_correlation(pc_x, pc_y, {pc_z1});
    EXPECT_NEAR(one.estimate, pc_one[0], 1e-13);
    EXPECT_NEAR(one.lower, pc_one[1], 1e-13);
    EXPECT_NEAR(one.upper, pc_one[2], 1e-13);
}

TEST(PartialCorrelation, MatrixFormMatchesPpcor) {
    datamunge::linalg::DenseMatrix<double> cov(4, 4, 0.0);
    for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t i = 0; i < 4; ++i)
            cov(i, j) = pc_cov[j * 4 + i];
    const auto m = partial_correlation_matrix(cov, 60);
    for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(m.partial(i, j), pc_pcor[j * 4 + i], 1e-12) << i << j;
            EXPECT_NEAR(m.partial_p_value(i, j), pc_pcor_p[j * 4 + i], 1e-12) << i << j;
            EXPECT_NEAR(m.semipartial(i, j), pc_spcor[j * 4 + i], 1e-12) << i << j;
            EXPECT_NEAR(m.semipartial_p_value(i, j), pc_pcor_p[j * 4 + i], 1e-12) << i << j;
        }
    // The data and matrix forms agree.
    EXPECT_NEAR(m.partial(0, 1), partial_correlation(pc_x, pc_y, {pc_z1, pc_z2}).estimate, 1e-12);
    EXPECT_NEAR(m.semipartial(0, 1), semipartial_correlation(pc_x, pc_y, {pc_z1, pc_z2}).estimate, 1e-12);
}

TEST(PartialCorrelation, BootstrapAndErrors) {
    const auto b = partial_correlation(pc_x, pc_y, {pc_z1, pc_z2}, {.bootstrap = 400, .seed = 3});
    EXPECT_LT(b.bootstrap_lower, b.estimate);
    EXPECT_GT(b.bootstrap_upper, b.estimate);
    const auto none = partial_correlation(pc_x, pc_y, {});
    EXPECT_EQ(none.covariates, 0u);
    EXPECT_TRUE(std::isnan(none.bootstrap_lower));
    EXPECT_THROW(partial_correlation({1, 2, 3}, {1, 2, 3}, {}), std::invalid_argument);
    EXPECT_THROW(partial_correlation(pc_x, pc_y, {pc_z1, pc_z1}), std::invalid_argument); // collinear
    EXPECT_THROW(partial_correlation_matrix(datamunge::linalg::DenseMatrix<double>(2, 2, {1, 1, 1, 1}), 10),
                 std::invalid_argument);
}
