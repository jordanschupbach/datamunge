#include <datamunge/stats/compositional.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_compositional_data.inc"

Matrix as_matrix(const std::vector<double>& v, std::size_t cols) { return Matrix(v.size() / cols, cols, v); }

void expect_matrix(const Matrix& m, const std::vector<double>& ref, double tol) {
    ASSERT_EQ(m.rows() * m.cols(), ref.size());
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j)
            EXPECT_NEAR(m(i, j), ref[i * m.cols() + j], tol * std::max(1.0, std::abs(ref[i * m.cols() + j])))
                << i << "," << j;
}
} // namespace

TEST(Compositional, TransformsMatchCompositions) {
    const auto x = as_matrix(cd_house, 4);
    expect_matrix(clr(x), cd_clr, 1e-14);
    expect_matrix(alr(x), cd_alr, 1e-14);
    expect_matrix(ilr(x), cd_ilr, 1e-14);
    expect_matrix(ilr_basis(5), cd_ilrbase5, 1e-15);
    expect_matrix(balance_basis(as_matrix(cd_sbp, 3)), cd_sbpbase, 1e-15);
    // Inverses recover the closed data, for every basis and reference.
    const auto closed = closure(x);
    expect_matrix(clr_inverse(clr(x)), closed.data(), 1e-14);
    for (std::size_t r = 0; r < 4; ++r)
        expect_matrix(alr_inverse(alr(x, r), r), closed.data(),
                      1e-14);
    for (const auto b : {ilr_basis(4, IlrBasis::helmert), ilr_basis(4, IlrBasis::pivot),
                         balance_basis(as_matrix(cd_sbp, 3))}) {
        expect_matrix(ilr_inverse(ilr(x, b), b), closed.data(),
                      1e-14);
        // Orthonormal columns orthogonal to the ones vector.
        for (std::size_t j = 0; j < 3; ++j) {
            double one = 0;
            for (std::size_t i = 0; i < 4; ++i)
                one += b(i, j);
            EXPECT_NEAR(one, 0, 1e-15);
            for (std::size_t k = 0; k < 3; ++k) {
                double dot = 0;
                for (std::size_t i = 0; i < 4; ++i)
                    dot += b(i, j) * b(i, k);
                EXPECT_NEAR(dot, j == k, 1e-15);
            }
        }
    }
    // Pivot coordinates of the reversed parts are the Helmert coordinates in reverse order.
    Matrix rev(x.rows(), 4, 0.0);
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < 4; ++j)
            rev(i, j) = x(i, 3 - j);
    const auto p = ilr(rev, ilr_basis(4, IlrBasis::pivot));
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(p(i, j), cd_ilr[i * 3 + (2 - j)], 1e-13);
}

TEST(Compositional, SummariesAndDistances) {
    const auto x = as_matrix(cd_house, 4);
    expect_matrix(variation_matrix(x), cd_variation, 1e-13);
    EXPECT_NEAR(total_variance(x), cd_totvar[0], 1e-13);
    const auto g = compositional_center(x);
    for (std::size_t j = 0; j < 4; ++j)
        EXPECT_NEAR(g[j], cd_center[j], 1e-15);
    const auto d = aitchison_distances(x);
    for (std::size_t k = 0; k < 5; ++k) {
        EXPECT_NEAR(d(0, k + 1), cd_dist[k], 1e-14);
        EXPECT_NEAR(aitchison_distance(x.row(0), x.row(k + 1)), cd_dist[k], 1e-14);
    }
    // Scale invariance and perturbation invariance.
    EXPECT_NEAR(aitchison_distance({1, 2, 3}, {3, 2, 1}), aitchison_distance({2, 4, 6}, {30, 20, 10}), 1e-15);
    EXPECT_NEAR(aitchison_distance({1, 2, 3}, {3, 2, 1}), aitchison_distance({5, 2, 3}, {15, 2, 1}), 1e-14);
    EXPECT_THROW(clr(Matrix(1, 2, {0, 1})), std::invalid_argument);
    EXPECT_THROW(alr(x, 4), std::invalid_argument);
    EXPECT_THROW(balance_basis(Matrix(3, 2, {1, 1, 1, -1, 0, 0})), std::invalid_argument);
}

TEST(Compositional, ZeroReplacementMatchesZCompositions) {
    const auto lp = as_matrix(cd_lp, 15);
    expect_matrix(multiplicative_replacement(lp, cd_lp_dl), cd_multrepl, 1e-13);
    // Closed data keep their total.
    auto closed = closure(lp, 100);
    std::vector<double> dl(cd_lp_dl);
    for (auto& v : dl)
        v /= 1e4;
    const auto rc = multiplicative_replacement(closed, dl);
    expect_matrix(rc, cd_multrepl_closed, 1e-12);
    const auto pigs = as_matrix(cd_pigs, 6);
    expect_matrix(count_zero_replacement(pigs), cd_pigs_gbm, 1e-14);
    expect_matrix(count_zero_replacement(pigs, {.method = CountZeroMethod::square_root}), cd_pigs_sq, 1e-14);
    expect_matrix(count_zero_replacement(pigs, {.method = CountZeroMethod::bayes_laplace}), cd_pigs_bl, 1e-14);
    expect_matrix(count_zero_replacement(pigs, {.method = CountZeroMethod::count_zero_multiplicative}), cd_pigs_czm,
                  1e-14);
    expect_matrix(count_zero_replacement(pigs, {.method = CountZeroMethod::bayes_laplace, .adjust = false}),
                  cd_pigs_bl_noadj, 1e-14);
    EXPECT_THROW(multiplicative_replacement(lp, std::vector<double>(15, 0.0)), std::invalid_argument);
    EXPECT_THROW(count_zero_replacement(Matrix(2, 2, {0, 1, 0, 2})), std::invalid_argument);
}
