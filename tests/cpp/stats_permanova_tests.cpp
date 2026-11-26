#include <datamunge/stats/permanova.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_permanova_data.inc"

std::vector<std::vector<std::size_t>> permutations() {
    std::vector<std::vector<std::size_t>> p(999, std::vector<std::size_t>(20));
    for (std::size_t r = 0; r < 999; ++r)
        for (std::size_t i = 0; i < 20; ++i)
            p[r][i] = static_cast<std::size_t>(pm_perm[r * 20 + i]);
    return p;
}

Matrix bray() { return bray_curtis_distances(Matrix(20, 30, pm_dune)); }

void expect_table(const Permanova& p, const std::vector<double>& ref) {
    ASSERT_EQ(p.rows.size() * 5, ref.size());
    for (std::size_t k = 0; k < p.rows.size(); ++k) {
        EXPECT_EQ(p.rows[k].df, static_cast<std::size_t>(ref[5 * k])) << p.rows[k].term;
        EXPECT_NEAR(p.rows[k].sum_of_squares, ref[5 * k + 1], 1e-12) << p.rows[k].term;
        EXPECT_NEAR(p.rows[k].r_squared, ref[5 * k + 2], 1e-13) << p.rows[k].term;
        if (ref[5 * k + 3] != 0) {
            EXPECT_NEAR(p.rows[k].f, ref[5 * k + 3], 1e-11) << p.rows[k].term;
            EXPECT_NEAR(p.rows[k].p_value, ref[5 * k + 4], 1e-15) << p.rows[k].term;
        } else {
            EXPECT_TRUE(std::isnan(p.rows[k].f));
        }
    }
}
} // namespace

TEST(Permanova, MatchesAdonis2) {
    const auto d = bray();
    for (std::size_t j = 0; j < 20; ++j)
        EXPECT_NEAR(d(0, j), pm_bray_row1[j], 1e-15);
    const auto m = factor_term("Management", pm_management);
    const auto a1 = numeric_term("A1", pm_a1);
    const auto mxa = interaction_term("Management:A1", m, a1);
    const auto perms = permutations();
    expect_table(permanova(d, {m, a1, mxa}, {.by = PermanovaBy::overall, .permutation_matrix = perms}), pm_overall);
    expect_table(permanova(d, {m, a1, mxa}, {.by = PermanovaBy::terms, .permutation_matrix = perms}), pm_terms);
    expect_table(permanova(d, {m, a1, factor_term("Use", pm_use)},
                           {.by = PermanovaBy::margin, .permutation_matrix = perms}),
                 pm_margin);
}

TEST(Permanova, RandomPermutationsAndValidation) {
    const auto d = bray();
    const auto m = factor_term("Management", pm_management);
    // Random permutations give a p-value close to the replayed one (0.002 with 999).
    const auto p = permanova(d, {m}, {.permutations = 4999, .seed = 3});
    EXPECT_EQ(p.permutations, 4999u);
    EXPECT_LT(p.rows[0].p_value, 0.01);
    // Restricted permutations keep each observation within its stratum.
    std::vector<std::size_t> strata(20);
    for (std::size_t i = 0; i < 20; ++i)
        strata[i] = i % 2;
    const auto ps = permanova(d, {m}, {.permutations = 199, .strata = strata});
    EXPECT_EQ(ps.permutations, 199u);
    EXPECT_NEAR(ps.rows[0].f, p.rows[0].f, 1e-12);
    // Aliased columns are dropped: duplicating a term adds no degrees of freedom.
    const auto dup = permanova(d, {m, m}, {.permutations = 9});
    EXPECT_EQ(dup.rows[1].df, 0u);
    EXPECT_EQ(dup.rows[2].df, 16u);
    Matrix bad = d;
    bad(0, 1) += 0.1;
    EXPECT_THROW(permanova(bad, {m}), std::invalid_argument);
    EXPECT_THROW(permanova(d, {numeric_term("x", {1, 2})}), std::invalid_argument);
}

TEST(Permanova, DispersionMatchesBetadisper) {
    const auto d = bray();
    const auto perms = permutations();
    const auto c = dispersion_test(d, pm_management, {.center = DispersionCenter::centroid,
                                                      .permutation_matrix = perms});
    for (std::size_t i = 0; i < 20; ++i)
        EXPECT_NEAR(c.distances[i], pm_bd_centroid[i], 1e-12) << i;
    for (std::size_t g = 0; g < 4; ++g)
        EXPECT_NEAR(c.group_mean_distances[g], pm_bd_centroid[20 + g], 1e-12);
    EXPECT_NEAR(c.f, pm_bd_centroid[24], 1e-10);
    EXPECT_NEAR(c.permutation_p_value, pm_bd_centroid[25], 1e-15);
    EXPECT_NEAR(c.parametric_p_value, pm_bd_centroid[26], 1e-10);
    ASSERT_EQ(c.eigenvalues.size(), pm_bd_eig.size());
    for (std::size_t k = 0; k < pm_bd_eig.size(); ++k)
        EXPECT_NEAR(c.eigenvalues[k], pm_bd_eig[k], 1e-12);
    const auto b = dispersion_test(d, pm_management, {.center = DispersionCenter::centroid, .bias_adjust = true,
                                                      .permutations = 0});
    for (std::size_t i = 0; i < 20; ++i)
        EXPECT_NEAR(b.distances[i], pm_bd_bias[i], 1e-12);
    EXPECT_TRUE(std::isnan(b.permutation_p_value));
    // Spatial medians: exact Weiszfeld reference; vegan's optim() agrees to about 1e-4.
    const auto m = dispersion_test(d, pm_management, {.permutation_matrix = perms});
    for (std::size_t i = 0; i < 20; ++i) {
        EXPECT_NEAR(m.distances[i], pm_bd_median_exact[i], 1e-10) << i;
        EXPECT_NEAR(m.distances[i], pm_bd_median[i], 2e-3) << i;
    }
    EXPECT_NEAR(m.permutation_p_value, pm_bd_median[25], 0.01);
    EXPECT_THROW(dispersion_test(d, std::vector<std::string>(20, "a")), std::invalid_argument);
}
