#include <datamunge/stats/point_pattern.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_point_pattern_data.inc"
#include "stats_poisson_process_data.inc"

const RectangleWindow window{.xmin = 0, .xmax = 2, .ymin = 0, .ymax = 1};

std::vector<double> grid() {
    std::vector<double> r;
    for (int k = 0; k <= 50; ++k)
        r.push_back(0.005 * k);
    return r;
}
Matrix pattern(const std::vector<double>& xy) { return Matrix(xy.size() / 2, 2, xy); }

void check_k(const std::vector<double>& xy, const std::vector<double>& ref, const std::string& what) {
    const auto r = grid();
    const auto p = pattern(xy);
    const EdgeCorrection c[4]{EdgeCorrection::none, EdgeCorrection::border, EdgeCorrection::translation,
                              EdgeCorrection::isotropic};
    for (int m = 0; m < 4; ++m) {
        const auto k = ripley_k(p, window, r, c[m]);
        for (std::size_t i = 0; i < r.size(); ++i)
            EXPECT_NEAR(k[i], ref[4 * i + m], 1e-12) << what << " correction " << m << " r " << r[i];
    }
}

void check_g(const std::vector<double>& xy, const std::vector<double>& ref, const std::string& what) {
    const auto r = grid();
    const auto p = pattern(xy);
    const std::vector<double> g[5]{
        pair_correlation(p, window, r, {.correction = EdgeCorrection::translation}),
        pair_correlation(p, window, r, {.correction = EdgeCorrection::isotropic}),
        pair_correlation(p, window, r, {.correction = EdgeCorrection::translation, .divide_by_distance = true}),
        pair_correlation(p, window, r, {.correction = EdgeCorrection::isotropic, .divide_by_distance = true}),
        pair_correlation(p, window, r, {.correction = EdgeCorrection::translation, .bandwidth = 0.01})};
    for (int m = 0; m < 5; ++m)
        for (std::size_t i = 1; i < r.size(); ++i)
            EXPECT_NEAR(g[m][i], ref[5 * i + m], 1e-9 * (1 + std::abs(ref[5 * i + m])))
                << what << " variant " << m << " r " << r[i];
}
} // namespace

TEST(PointPattern, RipleyKMatchesSpatstat) {
    check_k(csr_xy, csr_k, "csr");
    check_k(thomas_xy, thomas_k, "thomas");
    check_k(ssi_xy, ssi_k, "ssi");
}

TEST(PointPattern, PairCorrelationMatchesSpatstat) {
    check_g(csr_xy, csr_g, "csr");
    check_g(thomas_xy, thomas_g, "thomas");
    check_g(ssi_xy, ssi_g, "ssi");
}

TEST(PointPattern, LAndValidation) {
    const auto p = pattern(csr_xy);
    const auto k = ripley_k(p, window, {0.1});
    EXPECT_NEAR(besag_l(p, window, {0.1})[0], std::sqrt(k[0] / std::numbers::pi), 1e-15);
    EXPECT_TRUE(std::isnan(pair_correlation(p, window, {0.0})[0]));
    EXPECT_TRUE(std::isnan(ripley_k(p, window, {0.6}, EdgeCorrection::border)[0]));
    EXPECT_THROW(ripley_k(p, RectangleWindow{.xmax = 1}, {0.1}), std::invalid_argument);
    EXPECT_THROW(ripley_k(p, window, {-0.1}), std::invalid_argument);
    EXPECT_THROW(pair_correlation(p, window, {0.1}, {.correction = EdgeCorrection::border}), std::invalid_argument);
}

namespace {
void expect_ppm(const PoissonProcessFit& f, const std::vector<double>& r, const std::string& what) {
    const std::size_t p = f.coefficients.size();
    ASSERT_EQ(r.size(), 2 * p + 2) << what;
    EXPECT_TRUE(f.converged) << what;
    for (std::size_t k = 0; k < p; ++k) {
        EXPECT_NEAR(f.coefficients[k], r[k], 1e-6) << what << " beta " << k;
        EXPECT_NEAR(f.standard_errors[k], r[p + k], 1e-6 * r[p + k]) << what << " se " << k;
    }
    EXPECT_NEAR(f.log_likelihood, r[2 * p], 1e-7) << what;
    EXPECT_NEAR(f.aic, r[2 * p + 1], 1e-6) << what;
}
} // namespace

TEST(PoissonProcess, QuadratureMatchesSpatstat) {
    const auto q = berman_turner_quadrature(pattern(small_xy), window, 4, 3);
    ASSERT_EQ(q.points.rows() * 3, small_quad.size());
    EXPECT_EQ(q.data_points, 5u);
    // spatstat orders dummies differently; compare as sets keyed by location.
    for (std::size_t r = 0; r < q.points.rows(); ++r) {
        bool found = false;
        for (std::size_t s = 0; s < small_quad.size(); s += 3)
            if (std::abs(q.points(r, 0) - small_quad[s]) < 1e-12 && std::abs(q.points(r, 1) - small_quad[s + 1]) < 1e-12) {
                EXPECT_NEAR(q.weights[r], small_quad[s + 2], 1e-14) << r;
                found = true;
            }
        EXPECT_TRUE(found) << r;
    }
    double total = 0;
    for (const double w : q.weights)
        total += w;
    EXPECT_NEAR(total, window.area(), 1e-12);
}

TEST(PoissonProcess, RegressionMatchesSpatstatPpm) {
    const auto p = pattern(pp_xy);
    expect_ppm(poisson_process_regression(p, window, [](double x, double y) { return std::vector<double>{1, x, y}; }),
               fit_default, "default");
    expect_ppm(poisson_process_regression(
                   p, window, [](double x, double y) { return std::vector<double>{1, x, y, x * x, x * y}; },
                   {.grid_x = 30, .grid_y = 15}),
               fit_grid_quadratic, "quadratic, 30 x 15");
    const auto f = poisson_process_regression(p, window, [](double x, double y) {
        return std::vector<double>{1, std::sin(3 * x) + y * y};
    });
    expect_ppm(f, fit_function, "function covariate");
    EXPECT_NEAR(f.expected_count, static_cast<double>(p.rows()), 1e-6);
}

TEST(PoissonProcess, InterceptOnlyAndValidation) {
    const auto p = pattern(pp_xy);
    const auto z = [](double, double) { return std::vector<double>{1.0}; };
    const auto f = poisson_process_regression(p, window, z);
    EXPECT_NEAR(std::exp(f.coefficients[0]), static_cast<double>(p.rows()) / window.area(), 1e-9);
    EXPECT_NEAR(poisson_process_intensity(f, z, 0.5, 0.5), std::exp(f.coefficients[0]), 1e-12);
    EXPECT_THROW(poisson_process_regression(p, window, [](double x, double) {
                     return x < 1 ? std::vector<double>{1.0} : std::vector<double>{1.0, 2.0};
                 }),
                 std::invalid_argument);
    EXPECT_THROW(poisson_process_regression(p, window, [](double, double) { return std::vector<double>{1, 1}; }),
                 std::runtime_error);
    EXPECT_THROW(berman_turner_quadrature(p, window, 4, 0), std::invalid_argument);
}
