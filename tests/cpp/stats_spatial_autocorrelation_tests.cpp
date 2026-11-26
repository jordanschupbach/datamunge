#include <datamunge/stats/spatial_autocorrelation.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "stats_spatial_autocorrelation_data.inc"
#include "stats_spatial_local_data.inc"

const Matrix xy(60, 2, coords);

void expect_test(const SpatialAutocorrelationTest& t, const std::vector<double>& r, const std::string& what) {
    EXPECT_NEAR(t.statistic, r[0], 1e-12) << what;
    EXPECT_NEAR(t.expectation, r[1], 1e-14) << what;
    EXPECT_NEAR(t.variance, r[2], 1e-12) << what;
    EXPECT_NEAR(t.z, r[3], 1e-9) << what;
    EXPECT_NEAR(t.p_value, r[4], 1e-12 + 1e-9 * r[4]) << what;
}
} // namespace

TEST(SpatialAutocorrelation, WeightsMatchSpdep) {
    const auto knn = knn_weights(xy, 5);
    const auto band = distance_band_weights(xy, 0, 2.2);
    for (std::size_t k = 0; k < knn_dense.size(); ++k) {
        EXPECT_EQ(knn.data()[k], knn_dense[k]) << k;
        EXPECT_EQ(band.data()[k], band_dense[k]) << k;
    }
    const auto w = row_standardize(knn);
    for (std::size_t i = 0; i < 60; ++i) {
        double s = 0;
        for (std::size_t j = 0; j < 60; ++j)
            s += w(i, j);
        EXPECT_NEAR(s, 1, 1e-15);
    }
}

TEST(SpatialAutocorrelation, MoranAndGearyMatchSpdep) {
    const auto b = knn_weights(xy, 5), w = row_standardize(b);
    const struct {
        const Matrix* weights;
        bool rand;
        SpatialAlternative alt;
        const std::vector<double>* moran;
        const std::vector<double>* geary;
    } cases[] = {
        {&w, true, SpatialAlternative::positive, &moran_W_rand_greater, &geary_W_rand_greater},
        {&w, true, SpatialAlternative::negative, &moran_W_rand_less, &geary_W_rand_less},
        {&w, true, SpatialAlternative::two_sided, &moran_W_rand_twosided, &geary_W_rand_twosided},
        {&w, false, SpatialAlternative::positive, &moran_W_norm_greater, &geary_W_norm_greater},
        {&w, false, SpatialAlternative::negative, &moran_W_norm_less, &geary_W_norm_less},
        {&w, false, SpatialAlternative::two_sided, &moran_W_norm_twosided, &geary_W_norm_twosided},
        {&b, true, SpatialAlternative::positive, &moran_B_rand_greater, &geary_B_rand_greater},
        {&b, true, SpatialAlternative::two_sided, &moran_B_rand_twosided, &geary_B_rand_twosided},
        {&b, false, SpatialAlternative::positive, &moran_B_norm_greater, &geary_B_norm_greater},
        {&b, false, SpatialAlternative::negative, &moran_B_norm_less, &geary_B_norm_less},
    };
    int k = 0;
    for (const auto& c : cases) {
        const SpatialAutocorrelationOptions o{.randomisation = c.rand, .alternative = c.alt};
        expect_test(morans_i(values, *c.weights, o), *c.moran, "moran case " + std::to_string(k));
        expect_test(gearys_c(values, *c.weights, o), *c.geary, "geary case " + std::to_string(k));
        ++k;
    }
    const auto band = distance_band_weights(xy, 0, 2.2);
    expect_test(morans_i(values, band), moran_band, "moran band");
    expect_test(gearys_c(values, band), geary_band, "geary band");
}

TEST(SpatialAutocorrelation, PermutationDistributionMatchesMoments) {
    const auto w = row_standardize(knn_weights(xy, 5));
    const SpatialAutocorrelationOptions o{.permutations = 4000, .seed = 9};
    const auto mi = morans_i(values, w, o);
    EXPECT_NEAR(mi.permutation_mean, mi.expectation, 0.004);
    EXPECT_NEAR(mi.permutation_sd, std::sqrt(mi.variance), 0.05 * std::sqrt(mi.variance));
    EXPECT_NEAR(mi.permutation_p_value, 1.0 / 4001, 1e-12); // observed I exceeds every permutation
    const auto gc = gearys_c(values, w, o);
    EXPECT_NEAR(gc.permutation_mean, 1, 0.005);
    EXPECT_NEAR(gc.permutation_sd, std::sqrt(gc.variance), 0.05 * std::sqrt(gc.variance));
    // Pure noise: the permutation p-value is not small.
    std::vector<double> noise(60);
    for (std::size_t i = 0; i < 60; ++i) {
        const double h = std::sin(12.9898 * static_cast<double>(i)) * 43758.5453;
        noise[i] = h - std::floor(h);
    }
    EXPECT_GT(morans_i(noise, w, {.permutations = 999}).permutation_p_value, 0.01);
}

TEST(SpatialAutocorrelation, Validates) {
    const auto w = knn_weights(xy, 5);
    EXPECT_THROW(morans_i({1, 2, 3}, Matrix(3, 3, 0.0)), std::invalid_argument);
    EXPECT_THROW(morans_i(std::vector<double>(60, 1.0), w), std::invalid_argument);
    EXPECT_THROW(gearys_c(values, Matrix(60, 60, 0.0)), std::invalid_argument);
    EXPECT_THROW(knn_weights(xy, 60), std::invalid_argument);
    EXPECT_THROW(distance_band_weights(xy, 1, 1), std::invalid_argument);
    Matrix diag = w;
    diag(0, 0) = 1;
    EXPECT_THROW(morans_i(values, diag), std::invalid_argument);
}

namespace {
void expect_local(const std::vector<LocalStatistic>& s, const std::vector<double>& r, const std::string& what) {
    ASSERT_EQ(s.size() * 5, r.size()) << what;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const double* q = &r[5 * i];
        EXPECT_NEAR(s[i].statistic, q[0], 1e-12 * (1 + std::abs(q[0]))) << what << " " << i;
        EXPECT_NEAR(s[i].expectation, q[1], 1e-12 * (1 + std::abs(q[1]))) << what << " " << i;
        EXPECT_NEAR(s[i].variance, q[2], 1e-12 * (1 + q[2])) << what << " " << i;
        EXPECT_NEAR(s[i].z, q[3], 1e-9) << what << " " << i;
        EXPECT_NEAR(s[i].p_value, q[4], 1e-12 + 1e-9 * q[4]) << what << " " << i;
    }
}
} // namespace

TEST(LocalSpatial, LocalMoranMatchesSpdep) {
    const auto b = knn_weights(xy, 5), w = row_standardize(b);
    const auto cond = local_morans_i(values, w);
    expect_local(cond, lm_cond, "conditional");
    const LisaQuadrant codes[]{LisaQuadrant::low_low, LisaQuadrant::high_low, LisaQuadrant::low_high,
                               LisaQuadrant::high_high};
    for (std::size_t i = 0; i < cond.size(); ++i)
        EXPECT_EQ(cond[i].quadrant, codes[static_cast<int>(lm_cond_quad[i]) - 1]) << i;
    expect_local(local_morans_i(values, w, {.conditional = false, .alternative = SpatialAlternative::positive}),
                 lm_total_greater, "total, greater");
    expect_local(local_morans_i(values, b, {.alternative = SpatialAlternative::negative}), lm_cond_binary_less,
                 "binary, less");
}

TEST(LocalSpatial, GetisOrdMatchesSpdep) {
    const auto b = knn_weights(xy, 5), w = row_standardize(b);
    expect_local(local_getis_ord(xpos, w), g_w, "G, W");
    expect_local(local_getis_ord(xpos, b), g_b, "G, B");
    Matrix bs = b;
    for (std::size_t i = 0; i < 60; ++i)
        bs(i, i) = 1;
    expect_local(local_getis_ord(xpos, row_standardize(bs), true), gstar_w, "G*, W");
    expect_local(local_getis_ord(xpos, bs, true, {.alternative = SpatialAlternative::positive}), gstar_b_greater,
                 "G*, B");
    EXPECT_THROW(local_getis_ord(xpos, bs), std::invalid_argument);
}

TEST(LocalSpatial, ConditionalPermutationMatchesConditionalMoments) {
    const auto w = row_standardize(knn_weights(xy, 5));
    const auto s = local_morans_i(values, w, {.permutations = 20000, .seed = 3});
    for (std::size_t i = 0; i < s.size(); i += 7) {
        EXPECT_NEAR(s[i].permutation_mean, s[i].expectation, 0.03 * std::sqrt(s[i].variance) + 1e-12) << i;
        EXPECT_NEAR(s[i].permutation_sd, std::sqrt(s[i].variance), 0.03 * std::sqrt(s[i].variance)) << i;
        EXPECT_GE(s[i].permutation_p_value, 1.0 / 20001);
        EXPECT_LE(s[i].permutation_p_value, 1.0);
    }
    Matrix bs = knn_weights(xy, 5);
    for (std::size_t i = 0; i < 60; ++i)
        bs(i, i) = 1;
    const auto g = local_getis_ord(xpos, bs, true, {.permutations = 20000, .seed = 4});
    double total = 0;
    for (const double v : xpos)
        total += v;
    for (std::size_t i = 0; i < g.size(); i += 7) {
        // Conditional on x_i staying at i: (x_i + 5 * mean of the other values) / total.
        const double cond_mean = (xpos[i] + 5 * (total - xpos[i]) / 59) / total;
        EXPECT_NEAR(g[i].permutation_mean, cond_mean, 0.03 * g[i].permutation_sd) << i;
        // Conditioning on x_i removes its own variance from G*, so the permutation SD is smaller.
        EXPECT_LT(g[i].permutation_sd, 1.05 * std::sqrt(g[i].variance)) << i;
    }
}
