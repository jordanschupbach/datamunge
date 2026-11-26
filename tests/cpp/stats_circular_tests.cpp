#include <datamunge/stats/circular.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_circular_data.inc"
#include "stats_circular_tests_data.inc"

void expect_summary(const std::vector<double>& x, const std::vector<double>& r, const std::string& what) {
    const auto s = circular_summary(x);
    EXPECT_NEAR(s.mean_direction, r[0], 1e-12) << what;
    EXPECT_NEAR(s.resultant_length, r[1], 1e-12) << what;
    EXPECT_NEAR(s.circular_variance, r[2], 1e-12) << what;
    EXPECT_NEAR(s.circular_sd, r[3], 1e-12) << what;
    EXPECT_NEAR(s.angular_deviation, r[4], 1e-12) << what;
    EXPECT_NEAR(s.second_central_cosine, r[5], 1e-12) << what;
    EXPECT_NEAR(s.circular_dispersion, r[6], 1e-10 * r[6]) << what;
    EXPECT_NEAR(s.standard_error, r[7], 1e-12) << what;
    if (std::isnan(r[8])) {
        EXPECT_TRUE(std::isnan(s.lower) && std::isnan(s.upper)) << what;
    } else {
        EXPECT_NEAR(s.lower, r[8], 1e-9) << what;
        EXPECT_NEAR(s.upper, r[9], 1e-9) << what;
    }
    const auto m2 = trigonometric_moment(x, 2);
    EXPECT_NEAR(m2.direction, r[10], 1e-12) << what;
    EXPECT_NEAR(m2.resultant_length, r[11], 1e-12) << what;
}
} // namespace

TEST(Circular, SummaryMatchesCircularPackage) {
    expect_summary(sample_a, summary_a, "von Mises");
    expect_summary(sample_b, summary_b, "wrapped Cauchy");
    expect_summary(sample_c3, summary_c3, "bimodal");
}

TEST(Circular, RotationEquivarianceAndBootstrap) {
    std::vector<double> shifted;
    for (const double v : sample_a)
        shifted.push_back(v + 10 * std::numbers::pi + 0.3);
    const auto s0 = circular_summary(sample_a), s1 = circular_summary(shifted);
    EXPECT_NEAR(std::remainder(s1.mean_direction - s0.mean_direction - 0.3, 2 * std::numbers::pi), 0, 1e-10);
    EXPECT_NEAR(s1.resultant_length, s0.resultant_length, 1e-12);
    const auto ci = circular_mean_bootstrap_interval(sample_a, 0.95, 4000, 9);
    EXPECT_LT(ci.lower, s0.mean_direction);
    EXPECT_GT(ci.upper, s0.mean_direction);
    // For a well-concentrated sample of 30 the bootstrap and large-sample intervals are similar.
    EXPECT_NEAR(ci.upper - ci.lower, s0.upper - s0.lower, 0.25 * (s0.upper - s0.lower));
}

TEST(Circular, VonMisesSamplerMoments) {
    for (const double kappa : {0.5, 2.0, 10.0}) {
        const auto x = von_mises_sample(200000, 2.5, kappa, 17);
        const auto s = circular_summary(x);
        // E cos(theta - mu) = I1(kappa) / I0(kappa).
        const double a = std::cyl_bessel_i(1.0, kappa) / std::cyl_bessel_i(0.0, kappa);
        EXPECT_NEAR(s.resultant_length, a, 4 * std::sqrt((1 - a * a) / 2 / 200000.0) + 1e-3) << kappa;
        EXPECT_NEAR(std::remainder(s.mean_direction - 2.5, 2 * std::numbers::pi), 0, 0.02) << kappa;
    }
    const auto u = von_mises_sample(100000, 0, 0, 3);
    EXPECT_LT(circular_summary(u).resultant_length, 0.01);
}

TEST(Circular, Validates) {
    EXPECT_THROW(circular_summary({}), std::invalid_argument);
    EXPECT_THROW(circular_summary({std::numeric_limits<double>::infinity()}), std::invalid_argument);
    EXPECT_THROW(trigonometric_moment({1.0}, 0), std::invalid_argument);
    EXPECT_THROW(circular_mean_bootstrap_interval({0.0, std::numbers::pi}), std::invalid_argument);
    EXPECT_THROW(von_mises_sample(3, 0, -1, 1), std::invalid_argument);
    const auto balanced = circular_summary({0.0, std::numbers::pi / 2, std::numbers::pi, -std::numbers::pi / 2});
    EXPECT_TRUE(std::isnan(balanced.mean_direction));
    EXPECT_TRUE(std::isnan(balanced.lower));
}

TEST(CircularTests, RayleighAndKuiperMatchCircularPackage) {
    const std::vector<double>* samples[4]{&t_a, &t_b, &t_u, &t_ax};
    const std::vector<double>* refs[4]{&r_a, &r_b, &r_u, &r_ax};
    for (int s = 0; s < 4; ++s) {
        const auto& x = *samples[s];
        const auto& r = *refs[s];
        const auto ray = rayleigh_test(x);
        EXPECT_NEAR(ray.statistic, r[0], 1e-12) << s;
        EXPECT_NEAR(ray.p_value, r[1], 1e-12 + 1e-10 * r[1]) << s;
        EXPECT_TRUE(std::isnan(ray.monte_carlo_p_value));
        const auto v = rayleigh_test_direction(x, 0.5);
        EXPECT_NEAR(v.statistic, r[2], 1e-12) << s;
        EXPECT_NEAR(v.p_value, r[3], 1e-12 + 1e-10 * r[3]) << s;
        const auto k = kuiper_test(x);
        EXPECT_NEAR(k.statistic, r[4], 1e-12) << s;
        EXPECT_NEAR(k.p_value, r[5], 1e-12 + 1e-9 * r[5]) << s;
    }
}

TEST(CircularTests, KuiperAgainstVonMisesNullAndMonteCarlo) {
    // von Mises(5.5, 2) CDF on [0, 2 pi) by composite Simpson integration of the density.
    const auto cdf = [](double q) {
        const int m = 20000;
        const double h = q / m, c = 1 / (2 * std::numbers::pi * std::cyl_bessel_i(0.0, 2.0));
        const auto f = [&](double x) { return c * std::exp(2 * std::cos(x - 5.5)); };
        double s = f(0) + f(q);
        for (int i = 1; i < m; ++i)
            s += (i % 2 ? 4 : 2) * f(i * h);
        return std::min(1.0, s * h / 3);
    };
    const auto k = kuiper_test(t_a, cdf);
    EXPECT_NEAR(k.statistic, kuiper_vm_null[0], 1e-8);
    EXPECT_NEAR(k.p_value, kuiper_vm_null[1], 1e-7);
    // Monte Carlo calibration agrees with the analytic p-values for moderate n.
    // Stephens' approximation is tuned for the upper tail: close near p = 0.06, looser near p = 0.8.
    const auto ku = kuiper_test(t_u, {}, {.simulations = 20000, .seed = 4});
    EXPECT_NEAR(ku.monte_carlo_p_value, ku.p_value, 0.04);
    const auto kax = kuiper_test(t_ax, {}, {.simulations = 20000, .seed = 7});
    EXPECT_NEAR(kax.monte_carlo_p_value, kax.p_value, 0.008);
    const auto ru = rayleigh_test(t_u, {.simulations = 20000, .seed = 5});
    EXPECT_NEAR(ru.monte_carlo_p_value, ru.p_value, 0.02);
    const auto vu = rayleigh_test_direction(t_u, 0.5, {.simulations = 20000, .seed = 6});
    EXPECT_NEAR(vu.monte_carlo_p_value, vu.p_value, 0.02);
    EXPECT_THROW(kuiper_test(t_a, [](double) { return 2.0; }), std::invalid_argument);
}

namespace {
#include "stats_von_mises_data.inc"
} // namespace

TEST(VonMisesFit, MatchesCircularPackage) {
    const std::vector<double>* samples[3]{&vm_a, &vm_b, &vm_c3};
    const std::vector<double>* refs[3]{&fit_a, &fit_b, &fit_c3};
    for (int s = 0; s < 3; ++s) {
        const auto& x = *samples[s];
        const auto& r = *refs[s];
        const auto approx = von_mises_fit(x, {.exact_inverse = false});
        EXPECT_NEAR(approx.mean_direction, r[0], 1e-12) << s;
        EXPECT_NEAR(approx.mean_se, r[1], 1e-10) << s;
        EXPECT_NEAR(approx.concentration, r[2], 1e-10) << s;
        EXPECT_NEAR(approx.concentration_se, r[3], 1e-10) << s;
        const auto biased = von_mises_fit(x, {.exact_inverse = false, .bias_correction = true});
        EXPECT_NEAR(biased.concentration, r[4], 1e-10) << s;
        const auto exact = von_mises_fit(x);
        EXPECT_NEAR(exact.concentration, r[5], 1e-9 * r[5]) << s;
        EXPECT_NEAR(exact.log_likelihood, r[6], 1e-9) << s;
        EXPECT_NEAR(exact.concentration_lower, r[7], 1e-8) << s;
        EXPECT_NEAR(exact.concentration_upper, r[8], 1e-8 * r[8]) << s;
        const auto known = von_mises_fit(x, {.mean_direction = 0.5, .exact_inverse = false});
        EXPECT_NEAR(known.concentration, r[9], 1e-10) << s;
        EXPECT_NEAR(known.concentration_se, r[10], 1e-10) << s;
        EXPECT_EQ(known.mean_se, 0.0);
        EXPECT_NEAR(von_mises_a1(r[5]), r[11], 1e-13) << s;
        EXPECT_NEAR(exact.log_likelihood, von_mises_log_likelihood(x, exact.mean_direction, exact.concentration),
                    1e-12);
        EXPECT_FALSE(exact.boundary);
    }
}

TEST(VonMisesFit, BesselRatiosInverseAndCdf) {
    for (std::size_t i = 0; i < a1_k.size(); ++i) {
        EXPECT_NEAR(von_mises_a1(a1_k[i]), a1_v[i], 1e-14) << a1_k[i];
        EXPECT_NEAR(von_mises_a1_inverse(a1_v[i]), a1_k[i], 1e-8 * a1_k[i]) << a1_k[i];
    }
    EXPECT_EQ(von_mises_a1_inverse(0.0), 0.0);
    EXPECT_TRUE(std::isinf(von_mises_a1_inverse(1.0)));
    for (std::size_t i = 0; i < cdf_q.size(); ++i) {
        EXPECT_NEAR(von_mises_cdf(cdf_q[i], 1.0, 2.0), cdf_k2[i], 1e-10) << i;
        EXPECT_NEAR(von_mises_cdf(cdf_q[i], 5.0, 50.0), cdf_k50[i], 1e-10) << i;
    }
    EXPECT_NEAR(von_mises_cdf(1.0, 0.3, 0.0), 1.0 / (2 * std::numbers::pi), 1e-15);
    // Kuiper with the fitted von Mises CDF accepts von Mises data.
    const auto fit = von_mises_fit(vm_a);
    const auto k = kuiper_test(vm_a, [&](double q) { return von_mises_cdf(q, fit.mean_direction, fit.concentration); });
    EXPECT_GT(k.p_value, 0.05);
}

TEST(VonMisesFit, VanishingResultantDiagnostics) {
    const auto balanced = von_mises_fit({0.0, std::numbers::pi / 2, std::numbers::pi, -std::numbers::pi / 2});
    EXPECT_TRUE(balanced.boundary);
    EXPECT_TRUE(balanced.vanishing_resultant);
    EXPECT_TRUE(std::isnan(balanced.mean_direction));
    EXPECT_TRUE(std::isnan(balanced.mean_lower));
    EXPECT_EQ(balanced.concentration, 0.0);
    EXPECT_EQ(balanced.concentration_lower, 0.0);
    EXPECT_GT(balanced.concentration_upper, 0.0);
    const auto diffuse = von_mises_fit(von_mises_sample(20, 1.0, 0.0, 8));
    EXPECT_TRUE(diffuse.vanishing_resultant);
    EXPECT_EQ(diffuse.concentration_lower, 0.0);
    const auto same = von_mises_fit({0.7, 0.7, 0.7});
    EXPECT_TRUE(std::isinf(same.concentration));
    EXPECT_NEAR(same.mean_direction, 0.7, 1e-15);
    EXPECT_THROW(von_mises_fit({1.0}), std::invalid_argument);
    EXPECT_THROW(von_mises_a1_inverse(1.5), std::invalid_argument);
    EXPECT_THROW(von_mises_cdf(0.0, 0.0, -1.0), std::invalid_argument);
}
