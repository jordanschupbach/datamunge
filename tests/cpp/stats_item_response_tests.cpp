#include <datamunge/stats/item_response.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_irt_data.inc"

Matrix matrix(const std::vector<double>& x, std::size_t n, std::size_t p) {
    Matrix m(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            m(i, j) = x[i * p + j];
    return m;
}

void check(const Matrix& x, std::size_t p, const std::vector<double>& r1_est, const std::vector<double>& r1_se,
           const std::vector<double>& r1_ll, const std::vector<double>& r_est, const std::vector<double>& r_se,
           const std::vector<double>& r_ll, const std::vector<double>& m_est, const std::vector<double>& m_se,
           const std::vector<double>& m_ll, const std::vector<double>& map, const std::vector<double>& eap) {
    const auto r1 = fit_irt(x, {.model = IrtModel::rasch, .rasch_discrimination = 1});
    const auto r = fit_irt(x, {.model = IrtModel::rasch});
    const auto m = fit_irt(x);
    ASSERT_TRUE(r1.converged && r.converged && m.converged);
    EXPECT_NEAR(r1.log_likelihood, r1_ll[0], 1e-6);
    EXPECT_NEAR(r.log_likelihood, r_ll[0], 1e-6);
    EXPECT_NEAR(m.log_likelihood, m_ll[0], 1e-6);
    EXPECT_EQ(r1.parameters, p);
    EXPECT_EQ(r.parameters, p + 1);
    EXPECT_EQ(m.parameters, 2 * p);
    for (std::size_t j = 0; j < p; ++j) {
        EXPECT_NEAR(r1.items[j].difficulty, r1_est[j], 2e-4) << j;
        EXPECT_NEAR(r1.items[j].difficulty_se, r1_se[j], 2e-4) << j;
        EXPECT_TRUE(std::isnan(r1.items[j].discrimination_se));
        EXPECT_NEAR(r.items[j].difficulty, r_est[j], 2e-4) << j;
        EXPECT_NEAR(r.items[j].difficulty_se, r_se[j], 2e-4) << j;
        EXPECT_NEAR(r.items[j].discrimination, r_est[p], 2e-4) << j;
        EXPECT_NEAR(r.items[j].discrimination_se, r_se[p], 2e-4) << j;
        EXPECT_NEAR(m.items[j].difficulty, m_est[j], 2e-3 * std::max(1.0, std::abs(m_est[j]))) << j;
        EXPECT_NEAR(m.items[j].difficulty_se, m_se[j], 2e-3 * std::max(1.0, m_se[j])) << j;
        EXPECT_NEAR(m.items[j].discrimination, m_est[p + j], 1e-3) << j;
        EXPECT_NEAR(m.items[j].discrimination_se, m_se[p + j], 1e-3) << j;
    }
    Matrix first(5, p, 0.0);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < p; ++j)
            first(i, j) = x(i, j);
    const auto s = irt_scores(m, first);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(s[i].map, map[i], 1e-3) << i;
        EXPECT_NEAR(s[i].map_se, map[5 + i], 1e-3) << i;
        EXPECT_NEAR(s[i].eap, eap[i], 1e-3) << i;
        EXPECT_NEAR(s[i].eap_sd, eap[5 + i], 1e-3) << i;
    }
}
} // namespace

TEST(ItemResponse, LsatMatchesLtm) {
    check(matrix(lsat_x, lsat_n, lsat_p), lsat_p, lsat_r1_est, lsat_r1_se, lsat_r1_ll, lsat_r_est, lsat_r_se,
          lsat_r_ll, lsat_m_est, lsat_m_se, lsat_m_ll, lsat_map, lsat_eap);
}

TEST(ItemResponse, AbortionMatchesLtm) {
    check(matrix(abort_x, abort_n, abort_p), abort_p, abort_r1_est, abort_r1_se, abort_r1_ll, abort_r_est,
          abort_r_se, abort_r_ll, abort_m_est, abort_m_se, abort_m_ll, abort_map, abort_eap);
}

TEST(ItemResponse, SimulatedMatchesLtm) {
    check(matrix(sim_x, sim_n, sim_p), sim_p, sim_r1_est, sim_r1_se, sim_r1_ll, sim_r_est, sim_r_se, sim_r_ll,
          sim_m_est, sim_m_se, sim_m_ll, sim_map, sim_eap);
}

TEST(ItemResponse, InformationAndErrors) {
    IrtFit f;
    f.items = {{.difficulty = 0, .discrimination = 2}, {.difficulty = 1, .discrimination = 1}};
    const double p2 = 1 / (1 + std::exp(1.0));
    EXPECT_NEAR(irt_test_information(f, 0), 1 + p2 * (1 - p2), 1e-15);
    EXPECT_NEAR(irt_probability(f.items[1], 1), 0.5, 1e-15);
    EXPECT_THROW(fit_irt(Matrix(3, 2, {1, 0, 1, 1, 1, 0})), std::invalid_argument); // item 1 all correct
    EXPECT_THROW(fit_irt(Matrix(2, 2, {2, 0, 1, 1})), std::invalid_argument);
    EXPECT_THROW(fit_irt(Matrix(2, 2, {1, 0, 0, 1}), {.model = IrtModel::rasch, .rasch_discrimination = -1}),
                 std::invalid_argument);
}

namespace {
#include "stats_grm_data.inc"

void check_grm(const Matrix& x, const std::vector<double>& coef, const std::vector<double>& se,
               const std::vector<double>& ll, const std::vector<double>& eap, const std::vector<double>& map,
               const std::vector<double>& c_ll, const std::vector<double>& c_coef) {
    const auto f = fit_graded_response(x);
    const auto c = fit_graded_response(x, {.common_discrimination = true});
    ASSERT_TRUE(f.converged && c.converged);
    // ltm's quasi-Newton can stop about 1e-6 short of the maximum; ours must be at least as high.
    EXPECT_NEAR(f.log_likelihood, ll[0], 5e-6);
    EXPECT_GE(f.log_likelihood, ll[0] - 1e-9);
    EXPECT_NEAR(c.log_likelihood, c_ll[0], 5e-6);
    EXPECT_GE(c.log_likelihood, c_ll[0] - 1e-9);
    std::size_t k = 0;
    for (std::size_t j = 0; j < f.items.size(); ++j) {
        for (std::size_t t = 0; t < f.items[j].thresholds.size(); ++t, ++k) {
            EXPECT_NEAR(f.items[j].thresholds[t], coef[k], 1e-3 * std::max(1.0, std::abs(coef[k]))) << j;
            EXPECT_NEAR(f.items[j].threshold_se[t], se[k], 2e-3 * std::max(1.0, se[k])) << j;
            EXPECT_NEAR(c.items[j].thresholds[t], c_coef[k], 1e-3 * std::max(1.0, std::abs(c_coef[k]))) << j;
        }
        EXPECT_NEAR(f.items[j].discrimination, coef[k], 1e-3) << j;
        EXPECT_NEAR(f.items[j].discrimination_se, se[k], 1e-3) << j;
        EXPECT_NEAR(c.items[j].discrimination, c_coef[k], 1e-3) << j;
        ++k;
    }
    Matrix first(5, x.cols(), 0.0);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            first(i, j) = x(i, j);
    const auto s = graded_scores(f, first);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(s[i].eap, eap[i], 1e-3) << i;
        EXPECT_NEAR(s[i].eap_sd, eap[5 + i], 1e-3) << i;
        EXPECT_NEAR(s[i].map, map[i], 1e-3) << i;
        EXPECT_NEAR(s[i].map_se, map[5 + i], 1e-3) << i;
    }
}
} // namespace

TEST(GradedResponse, ScienceMatchesLtm) {
    check_grm(matrix(sci_x, sci_n, sci_p), sci_coef, sci_se, sci_ll, sci_eap, sci_map, sci_c_ll, sci_c_coef);
}

TEST(GradedResponse, EnvironmentMatchesLtm) {
    check_grm(matrix(env_x, env_n, env_p), env_coef, env_se, env_ll, env_eap, env_map, env_c_ll, env_c_coef);
}

TEST(GradedResponse, ReducesTo2plAndInformation) {
    const Matrix x = matrix(lsat_x, lsat_n, lsat_p);
    const auto g = fit_graded_response(x);
    const auto m = fit_irt(x);
    EXPECT_NEAR(g.log_likelihood, m.log_likelihood, 1e-8);
    for (std::size_t j = 0; j < 5; ++j) {
        EXPECT_NEAR(g.items[j].thresholds[0], m.items[j].difficulty, 1e-5);
        EXPECT_NEAR(g.items[j].discrimination, m.items[j].discrimination, 1e-5);
        EXPECT_NEAR(graded_item_information(g.items[j], 0.3),
                    irt_test_information(IrtFit{.items = {m.items[j]}}, 0.3), 1e-5);
    }
    // Category probabilities sum to one; information equals the numerical Fisher information.
    const GradedItem it{.thresholds = {-1.2, 0.1, 1.5}, .discrimination = 1.7};
    const auto p = graded_category_probabilities(it, 0.4);
    EXPECT_NEAR(p[0] + p[1] + p[2] + p[3], 1.0, 1e-15);
    double fisher = 0;
    for (std::size_t c = 0; c < 4; ++c) {
        const double h = 1e-6;
        const double d = (std::log(graded_category_probabilities(it, 0.4 + h)[c]) -
                          std::log(graded_category_probabilities(it, 0.4 - h)[c])) / (2 * h);
        fisher += p[c] * d * d;
    }
    EXPECT_NEAR(graded_item_information(it, 0.4), fisher, 1e-6);
    EXPECT_THROW(fit_graded_response(Matrix(3, 2, {0, 0, 2, 1, 0, 1})), std::invalid_argument); // category 1 unused
}
