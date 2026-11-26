#include <datamunge/stats/geostatistics.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_geostatistics_data.inc"
#include "stats_variogram_fit_data.inc"
#include "stats_kriging_data.inc"

const Matrix xy(155, 2, meuse_xy);

void expect_bins(const std::vector<VariogramBin>& v, const std::vector<double>& r, const std::string& what) {
    ASSERT_EQ(v.size() * 4, r.size()) << what;
    for (std::size_t k = 0; k < v.size(); ++k) {
        EXPECT_EQ(v[k].pairs, static_cast<std::size_t>(r[4 * k])) << what << " " << k;
        EXPECT_NEAR(v[k].distance, r[4 * k + 1], 1e-9) << what << " " << k;
        EXPECT_NEAR(v[k].gamma, r[4 * k + 2], 1e-12) << what << " " << k;
        if (!std::isnan(v[k].direction))
            EXPECT_DOUBLE_EQ(v[k].direction, r[4 * k + 3]) << what << " " << k;
    }
}
} // namespace

TEST(EmpiricalVariogram, MatchesGstat) {
    expect_bins(empirical_variogram(xy, meuse_logzinc, {.cutoff = 1500, .width = 100}), v_iso, "isotropic");
    expect_bins(empirical_variogram(xy, meuse_logzinc, {.boundaries = {0, 50, 150, 300, 600, 1000, 1600}}),
                v_bounds, "boundaries");
    expect_bins(empirical_variogram(xy, meuse_logzinc, {.cutoff = 1500, .width = 150, .directions = {0, 45, 90, 135}}),
                v_dir, "four directions");
    expect_bins(empirical_variogram(xy, meuse_logzinc,
                                    {.cutoff = 1200, .width = 200, .directions = {30, 120}, .angular_tolerance = 15}),
                v_dir_tol, "narrow tolerance");
    expect_bins(empirical_variogram(xy, meuse_logzinc, {.cutoff = 1500, .width = 100, .cressie = true}), v_cressie,
                "cressie");
}

TEST(EmpiricalVariogram, DefaultsAndBookkeeping) {
    const auto v = empirical_variogram(xy, meuse_logzinc);
    ASSERT_FALSE(v.empty());
    EXPECT_LE(v.size(), 15u);
    double diag = std::hypot(181390.0 - 178605.0, 333611.0 - 329714.0);
    EXPECT_NEAR(v.back().upper, diag / 3, 1e-6);
    // With four directions of tolerance 22.5 every pair falls in exactly one direction.
    const auto all = empirical_variogram(xy, meuse_logzinc, {.cutoff = 1500, .width = 150});
    const auto dir = empirical_variogram(xy, meuse_logzinc, {.cutoff = 1500, .width = 150, .directions = {0, 45, 90, 135}});
    std::size_t a = 0, b = 0;
    for (const auto& x : all)
        a += x.pairs;
    for (const auto& x : dir)
        b += x.pairs;
    EXPECT_GE(b, a); // pairs exactly on a sector edge count in both neighbours
    EXPECT_LE(b, a + 20);
}

TEST(EmpiricalVariogram, Validates) {
    EXPECT_THROW(empirical_variogram(Matrix(1, 2, 0.0), {1.0}), std::invalid_argument);
    EXPECT_THROW(empirical_variogram(xy, std::vector<double>(3, 1.0)), std::invalid_argument);
    EXPECT_THROW(empirical_variogram(xy, meuse_logzinc, {.boundaries = {0, 10, 5}}), std::invalid_argument);
    EXPECT_THROW(empirical_variogram(Matrix(3, 1, std::vector<double>{0, 1, 2}), {1, 2, 3}, {.directions = {0}}),
                 std::invalid_argument);
}

namespace {
std::vector<VariogramBin> meuse_bins() {
    std::vector<VariogramBin> out;
    for (std::size_t k = 0; k < bins.size(); k += 3)
        out.push_back({.distance = bins[k + 1], .gamma = bins[k + 2], .pairs = static_cast<std::size_t>(bins[k])});
    return out;
}
void expect_fit(const VariogramFit& f, const std::vector<double>& r, const std::string& what) {
    EXPECT_NEAR(f.model.nugget, r[0], 1e-6) << what;
    EXPECT_NEAR(f.model.partial_sill, r[1], 1e-6) << what;
    EXPECT_NEAR(f.model.range, r[2], 1e-5 * r[2]) << what;
    EXPECT_NEAR(f.weighted_sse, r[3], 1e-9 * r[3]) << what;
}
} // namespace

TEST(VariogramModel, ValuesMatchGstat) {
    using K = VariogramModelKind;
    const auto check = [](VariogramModel m, const std::vector<double>& r) {
        for (std::size_t k = 0; k < model_h.size(); ++k)
            EXPECT_NEAR(variogram_model_value(m, model_h[k]), r[k], 1e-10) << k;
    };
    check({.kind = K::spherical, .nugget = 0.2, .partial_sill = 1, .range = 800}, model_sph);
    check({.kind = K::exponential, .nugget = 0.2, .partial_sill = 1, .range = 300}, model_exp);
    check({.kind = K::gaussian, .nugget = 0.2, .partial_sill = 1, .range = 400}, model_gau);
    check({.kind = K::matern, .nugget = 0.2, .partial_sill = 1, .range = 200, .kappa = 2.5}, model_mat25);
    const VariogramModel m{.kind = K::exponential, .nugget = 0.2, .partial_sill = 1, .range = 300};
    EXPECT_DOUBLE_EQ(variogram_model_covariance(m, 0), 1.2);
    EXPECT_NEAR(variogram_model_covariance(m, 300) + variogram_model_value(m, 300), 1.2, 1e-14);
    EXPECT_THROW(variogram_model_value({.range = -1}, 1), std::invalid_argument);
}

TEST(VariogramFit, MatchesExactWeightedLeastSquaresAndGstat) {
    using K = VariogramModelKind;
    using W = VariogramWeighting;
    const auto b = meuse_bins();
    const auto fit = [&](K k, W w, double kappa = 0.5) {
        return fit_variogram(b, {.kind = k, .kappa = kappa}, {.weighting = w});
    };
    expect_fit(fit(K::spherical, W::pairs_over_distance_squared), exact_sph_7, "sph 7");
    expect_fit(fit(K::spherical, W::pairs), exact_sph_1, "sph 1");
    expect_fit(fit(K::spherical, W::uniform), exact_sph_6, "sph 6");
    expect_fit(fit(K::exponential, W::pairs_over_distance_squared), exact_exp_7, "exp 7");
    expect_fit(fit(K::exponential, W::pairs), exact_exp_1, "exp 1");
    expect_fit(fit(K::exponential, W::uniform), exact_exp_6, "exp 6");
    expect_fit(fit(K::gaussian, W::pairs_over_distance_squared), exact_gau_7, "gau 7");
    expect_fit(fit(K::gaussian, W::pairs), exact_gau_1, "gau 1");
    expect_fit(fit(K::gaussian, W::uniform), exact_gau_6, "gau 6");
    const auto mat = fit(K::matern, W::pairs_over_distance_squared, 1.5);
    expect_fit(mat, exact_mat15_7, "mat 1.5");
    // gstat converges to the same optimum (to its iteration tolerance) where it converges...
    const auto sph = fit(K::spherical, W::pairs_over_distance_squared);
    EXPECT_NEAR(sph.model.range, gstat_sph_7[2], 1e-4 * gstat_sph_7[2]);
    EXPECT_NEAR(sph.model.nugget, gstat_sph_7[0], 1e-4);
    EXPECT_NEAR(mat.model.range, gstat_mat15_7[2], 1e-5 * gstat_mat15_7[2]);
    // ...and stops at a worse point for the Gaussian model, where it warns about convergence.
    EXPECT_LT(fit(K::gaussian, W::pairs_over_distance_squared).weighted_sse, gstat_gau_7[3]);
}

TEST(VariogramFit, CressieWeightsAndFixedParameters) {
    using K = VariogramModelKind;
    const auto b = meuse_bins();
    const auto c = fit_variogram(b, {.kind = K::spherical, .nugget = 0.05, .partial_sill = 0.6, .range = 800},
                                 {.weighting = VariogramWeighting::cressie});
    EXPECT_TRUE(c.converged);
    EXPECT_NEAR(c.model.nugget, gstat_sph_2[0], 2e-3);
    EXPECT_NEAR(c.model.partial_sill, gstat_sph_2[1], 2e-3);
    EXPECT_NEAR(c.model.range, gstat_sph_2[2], 2e-3 * gstat_sph_2[2]);
    const auto fixed = fit_variogram(b, {.kind = K::spherical, .nugget = 0.1, .range = 1000},
                                     {.fit_nugget = false, .fit_range = false});
    EXPECT_DOUBLE_EQ(fixed.model.nugget, 0.1);
    EXPECT_DOUBLE_EQ(fixed.model.range, 1000);
    EXPECT_GT(fixed.model.partial_sill, 0);
    EXPECT_THROW(fit_variogram({b[0]}, {}), std::invalid_argument);
}

namespace {
const VariogramModel sph{.kind = VariogramModelKind::spherical, .nugget = 0.05, .partial_sill = 0.59, .range = 897};
const Matrix grid(grid_xy.size() / 2, 2, grid_xy);

Matrix intercept_and(const std::vector<double>& v) {
    Matrix x(v.size(), 2);
    for (std::size_t i = 0; i < v.size(); ++i)
        x(i, 0) = 1, x(i, 1) = v[i];
    return x;
}
void expect_krige(const KrigingResult& k, const std::vector<double>& r, const std::string& what) {
    ASSERT_EQ(k.prediction.size() * 2, r.size()) << what;
    for (std::size_t t = 0; t < k.prediction.size(); ++t) {
        EXPECT_NEAR(k.prediction[t], r[2 * t], 1e-9) << what << " " << t;
        EXPECT_NEAR(k.variance[t], r[2 * t + 1], 1e-9) << what << " " << t;
    }
}
} // namespace

TEST(Kriging, MatchesGstat) {
    expect_krige(ordinary_kriging(xy, meuse_logzinc, grid, sph), ok, "ordinary");
    expect_krige(simple_kriging(xy, meuse_logzinc, grid, sph, 5.9), sk, "simple");
    expect_krige(universal_kriging(xy, meuse_logzinc, intercept_and(sqrt_dist), grid, intercept_and(grid_sqrt_dist), sph),
                 uk, "universal");
    expect_krige(ordinary_kriging(xy, meuse_logzinc, grid, sph, {.max_neighbours = 10}), ok_nmax10, "ordinary nmax 10");
    expect_krige(universal_kriging(xy, meuse_logzinc, intercept_and(sqrt_dist), grid, intercept_and(grid_sqrt_dist), sph,
                                   {.max_neighbours = 20}),
                 uk_nmax20, "universal nmax 20");
    const VariogramModel ex{.kind = VariogramModelKind::exponential, .nugget = 0.02, .partial_sill = 0.7, .range = 450};
    expect_krige(ordinary_kriging(xy, meuse_logzinc, grid, ex), ok_exp, "ordinary exponential");
}

TEST(Kriging, ExactInterpolationAndTrend) {
    const Matrix first(2, 2, std::vector<double>{meuse_xy[0], meuse_xy[1], meuse_xy[2], meuse_xy[3]});
    const auto k = ordinary_kriging(xy, meuse_logzinc, first, sph);
    EXPECT_NEAR(k.prediction[0], meuse_logzinc[0], 1e-10);
    EXPECT_NEAR(k.prediction[1], meuse_logzinc[1], 1e-10);
    EXPECT_NEAR(k.variance[0], 0, 1e-10);
    ASSERT_EQ(k.trend_coefficients.size(), 1u);
    // Far beyond the range, ordinary kriging returns the GLS mean.
    const Matrix far(1, 2, std::vector<double>{0, 0});
    const auto f = ordinary_kriging(xy, meuse_logzinc, far, sph);
    EXPECT_NEAR(f.prediction[0], f.trend_coefficients[0], 1e-10);
    // Nothing within max_distance gives NaN.
    EXPECT_TRUE(std::isnan(ordinary_kriging(xy, meuse_logzinc, far, sph, {.max_distance = 100}).prediction[0]));
    EXPECT_THROW(ordinary_kriging(xy, std::vector<double>(3, 1.0), grid, sph), std::invalid_argument);
}

TEST(Kriging, CrossValidationMatchesGstat) {
    const auto check = [](const KrigingCrossValidation& cv, const std::vector<double>& r, const std::string& what) {
        ASSERT_EQ(cv.prediction.size() * 4, r.size()) << what;
        for (std::size_t i = 0; i < cv.prediction.size(); ++i) {
            EXPECT_NEAR(cv.prediction[i], r[4 * i], 1e-9) << what << " " << i;
            EXPECT_NEAR(cv.variance[i], r[4 * i + 1], 1e-9) << what << " " << i;
            EXPECT_NEAR(cv.residual[i], r[4 * i + 2], 1e-9) << what << " " << i;
            EXPECT_NEAR(cv.zscore[i], r[4 * i + 3], 1e-8) << what << " " << i;
        }
    };
    check(kriging_cross_validation(xy, meuse_logzinc, sph), cv_ok, "ordinary");
    check(kriging_cross_validation(xy, meuse_logzinc, sph, intercept_and(sqrt_dist)), cv_uk, "universal");
}
