#include <datamunge/stats/distance_correlation.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_distance_correlation_data.inc"

Matrix rows(const std::vector<double>& v, std::size_t cols) {
    Matrix m(v.size() / cols, cols, 0.0);
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < cols; ++j)
            m(i, j) = v[i * cols + j];
    return m;
}
} // namespace

TEST(DistanceCorrelation, MatchesEnergyEstimates) {
    const auto x = rows(dc_x, 2), y = rows(dc_y, 3);
    for (const auto& [index, ref] : {std::pair{1.0, &dc_index_10}, std::pair{0.5, &dc_index_5},
                                     std::pair{1.5, &dc_index_15}}) {
        const auto d = distance_correlation(x, y, {.index = index});
        EXPECT_NEAR(d.dcov, (*ref)[0], 1e-13) << index;
        EXPECT_NEAR(d.dcor, (*ref)[1], 1e-13) << index;
        EXPECT_NEAR(d.dvar_x, (*ref)[2], 1e-13) << index;
        EXPECT_NEAR(d.dvar_y, (*ref)[3], 1e-13) << index;
        EXPECT_NEAR(d.permutation_statistic, (*ref)[4], 1e-11) << index;
    }
    const auto d = distance_correlation(x, y);
    EXPECT_NEAR(d.dcor_bias_corrected, dc_t[0], 1e-13);
    EXPECT_NEAR(d.t_statistic, dc_t[1], 1e-11);
    EXPECT_EQ(d.t_df, dc_t[2]);
    EXPECT_NEAR(d.t_p_value, dc_t[3], 1e-15);
    EXPECT_NEAR(d.dcov_squared_unbiased, dc_u[0], 1e-13);
    EXPECT_NEAR(partial_distance_correlation(x, y, rows(dc_z, 1)), dc_pdcor[0], 1e-13);
}

TEST(DistanceCorrelation, PermutationTest) {
    const auto x = rows(dc_x, 2), y = rows(dc_y, 3);
    const auto dep = distance_correlation(x, y, {.permutations = 9999, .seed = 4});
    EXPECT_NEAR(dep.permutation_p_value, dc_perm[0], 1e-12); // nothing exceeds: 1 / 10000
    const auto ind = distance_correlation(dc_x0, dc_y0, {.permutations = 9999, .seed = 4});
    EXPECT_NEAR(ind.dcor, dc_perm0[1], 1e-13);
    // Different RNG: agree within Monte Carlo error (SE ~ 0.005).
    EXPECT_NEAR(ind.permutation_p_value, dc_perm0[0], 0.02);
    EXPECT_TRUE(std::isnan(distance_correlation(dc_x0, dc_y0).permutation_p_value));
}

TEST(DistanceCorrelation, PropertiesAndErrors) {
    // Invariance to shifts, rotations of x and scaling; dcor = 1 for linear similarity.
    const std::vector<double> x{0.3, -1.2, 2.5, 0.7, -0.4, 1.9}, y{1, 2, 3, 4, 5, 6};
    std::vector<double> lin(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        lin[i] = 5 - 3 * x[i];
    EXPECT_NEAR(distance_correlation(x, lin).dcor, 1.0, 1e-14);
    EXPECT_NEAR(distance_correlation(x, y).dcor, distance_correlation(lin, y).dcor, 1e-14);
    const std::vector<double> c(6, 2.0);
    EXPECT_EQ(distance_correlation(x, c).dcor, 0.0);
    EXPECT_THROW(distance_correlation(std::vector<double>{1, 2, 3}, {1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(distance_correlation(x, y, {.index = 2.0}), std::invalid_argument);
    EXPECT_THROW(distance_correlation(x, std::vector<double>{1, 2, 3, 4, 5}), std::invalid_argument);
}
