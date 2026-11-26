#include <datamunge/stats/hsic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_hsic_data.inc"

Matrix rows(const std::vector<double>& v, std::size_t cols) {
    Matrix m(v.size() / cols, cols, 0.0);
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < cols; ++j)
            m(i, j) = v[i * cols + j];
    return m;
}
} // namespace

TEST(Hsic, MatchesDhsicGamma) {
    const auto x = rows(hs_x, 2), y = rows(hs_y, 1), z = rows(hs_z, 1);
    const auto two = hsic({x, y});
    EXPECT_NEAR(two.statistic, hs_gamma_xy[0], 1e-14);
    EXPECT_NEAR(two.scaled_statistic, hs_gamma_xy[1], 1e-13);
    EXPECT_NEAR(two.gamma_p_value / hs_gamma_xy[2], 1.0, 1e-9);
    EXPECT_NEAR(two.bandwidths[0], hs_gamma_xy[3], 1e-14);
    EXPECT_NEAR(two.bandwidths[1], hs_gamma_xy[4], 1e-14);
    const auto three = hsic({x, y, z});
    EXPECT_NEAR(three.statistic, hs_gamma_xyz[0], 1e-14);
    EXPECT_NEAR(three.scaled_statistic, hs_gamma_xyz[1], 1e-13);
    EXPECT_NEAR(three.gamma_p_value / hs_gamma_xyz[2], 1.0, 1e-9);
    EXPECT_NEAR(three.bandwidths[2], hs_gamma_xyz[5], 1e-14);
    const auto fixed = hsic({x, y}, {.kernel = HsicKernel::gaussian_fixed, .bandwidth = 0.8});
    EXPECT_NEAR(fixed.scaled_statistic, hs_fixed[0], 1e-13);
    EXPECT_NEAR(fixed.gamma_p_value / hs_fixed[1], 1.0, 1e-9);
    // A scaled median bandwidth equals the fixed kernel at that bandwidth.
    const Matrix xy0 = rows(hs_y, 1);
    const auto scaled = hsic({xy0, xy0}, {.bandwidth_scale = 0.5});
    const auto same = hsic({xy0, xy0}, {.kernel = HsicKernel::gaussian_fixed, .bandwidth = 0.5 * hs_gamma_xy[4]});
    EXPECT_NEAR(scaled.statistic, same.statistic, 1e-15);
    const auto disc = hsic(hs_a, hs_b, {.kernel = HsicKernel::discrete});
    EXPECT_NEAR(disc.scaled_statistic, hs_discrete[0], 1e-13);
    EXPECT_NEAR(disc.gamma_p_value, hs_discrete[1], 1e-12);
}

TEST(Hsic, PermutationTest) {
    const auto ind = hsic(hs_x0, hs_y0, {.permutations = 9999, .seed = 3});
    EXPECT_NEAR(ind.gamma_p_value, hs_perm0[1], 1e-12);
    EXPECT_NEAR(ind.permutation_p_value, hs_perm0[0], 0.02); // different RNG; SE ~ 0.004
    const auto dep = hsic({rows(hs_x, 2), rows(hs_y, 1)}, {.permutations = 999});
    EXPECT_NEAR(dep.permutation_p_value, 1.0 / 1000, 1e-15);
    EXPECT_TRUE(std::isnan(hsic(hs_x0, hs_y0).permutation_p_value));
}

TEST(Hsic, Errors) {
    const Matrix x(3, 1, {1, 2, 3});
    EXPECT_THROW(hsic({x}), std::invalid_argument);
    EXPECT_THROW(hsic({x, x}), std::invalid_argument); // n < 2d
    EXPECT_THROW(hsic(hs_x0, hs_y0, {.kernel = HsicKernel::gaussian_fixed, .bandwidth = 0}),
                 std::invalid_argument);
    EXPECT_THROW(hsic(hs_x0, std::vector<double>{1, 2, 3, 4}), std::invalid_argument);
    EXPECT_THROW(hsic(hs_x0, hs_y0, {.bandwidth_scale = -1}), std::invalid_argument);
}
