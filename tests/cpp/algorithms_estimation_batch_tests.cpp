#include <gtest/gtest.h>

#include <datamunge/algorithms/linde_buzo_gray.hpp>
#include <datamunge/algorithms/online_statistics.hpp>
#include <datamunge/algorithms/ransac.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using datamunge::algorithms::linde_buzo_gray;
using datamunge::algorithms::OnlineCovariance;
using datamunge::algorithms::OnlineVariance;
using datamunge::algorithms::ransac_line;
using datamunge::algorithms::ransac_required_iterations;

TEST(Ransac, RecoversLineDespiteOutliers) {
    // True line y = 2x + 1. 80 inliers with small noise, 40 gross outliers.
    std::mt19937_64                  rng(1);
    std::normal_distribution<double> noise(0.0, 0.05);
    std::uniform_real_distribution<double> ux(0.0, 10.0), uy(-20.0, 40.0);
    std::vector<double>              xs, ys;
    for (int i = 0; i < 80; ++i) {
        double x = ux(rng);
        xs.push_back(x);
        ys.push_back(2.0 * x + 1.0 + noise(rng));
    }
    for (int i = 0; i < 40; ++i) {
        xs.push_back(ux(rng));
        ys.push_back(uy(rng));  // outliers anywhere
    }

    auto line = ransac_line(xs, ys, 0.2, 500, 7);
    // Recover slope/intercept from a x + b y + c = 0  ->  y = -(a/b) x - c/b.
    ASSERT_GT(std::abs(line.b), 1e-9);
    double slope     = -line.a / line.b;
    double intercept = -line.c / line.b;
    EXPECT_NEAR(slope, 2.0, 0.1);
    EXPECT_NEAR(intercept, 1.0, 0.3);
    EXPECT_GE(line.inlier_count, 70u);  // found most of the 80 inliers
    EXPECT_LT(line.inlier_count, 100u); // did not absorb the outliers
}

TEST(Ransac, RequiredIterationsFormula) {
    // w=0.5, s=2, p=0.99: N = ceil(log(0.01)/log(1-0.25)) = ceil(16.0) = 16 or 17.
    std::size_t n = ransac_required_iterations(0.5, 2, 0.99);
    EXPECT_GE(n, 16u);
    EXPECT_LE(n, 17u);
    // More outliers -> more iterations needed.
    EXPECT_GT(ransac_required_iterations(0.3, 2, 0.99), n);
}

TEST(OnlineVariance, MatchesTwoPassAndIsStableWithLargeOffset) {
    std::mt19937_64                  rng(2);
    std::normal_distribution<double> g(1e9, 4.0);  // huge mean, small spread
    std::vector<double>              data;
    OnlineVariance                   ov;
    for (int i = 0; i < 5000; ++i) {
        double x = g(rng);
        data.push_back(x);
        ov.add(x);
    }
    // Reference two-pass variance.
    double mean = 0.0;
    for (double x : data) mean += x;
    mean /= data.size();
    double ss = 0.0;
    for (double x : data) ss += (x - mean) * (x - mean);
    double ref_var = ss / (data.size() - 1);

    EXPECT_NEAR(ov.mean(), mean, 1e-3);
    EXPECT_NEAR(ov.sample_variance(), ref_var, ref_var * 1e-6);
    EXPECT_GT(ov.sample_variance(), 0.0);  // naive sum-of-squares would collapse here
}

TEST(OnlineVariance, MergeMatchesSingleStream) {
    std::mt19937_64                  rng(3);
    std::normal_distribution<double> g(5.0, 2.0);
    OnlineVariance                   all, a, b;
    for (int i = 0; i < 1000; ++i) {
        double x = g(rng);
        all.add(x);
        (i % 2 ? a : b).add(x);
    }
    OnlineVariance merged = a;
    merged.merge(b);
    EXPECT_NEAR(merged.mean(), all.mean(), 1e-9);
    EXPECT_NEAR(merged.sample_variance(), all.sample_variance(), 1e-6);
}

TEST(OnlineCovariance, RecoversCorrelation) {
    std::mt19937_64                  rng(4);
    std::normal_distribution<double> g(0.0, 1.0);
    OnlineCovariance                 oc;
    for (int i = 0; i < 5000; ++i) {
        double x = g(rng);
        double y = 3.0 * x + g(rng) * 0.5;  // strong positive linear relation
        oc.add(x, y);
    }
    EXPECT_GT(oc.correlation(), 0.95);
    EXPECT_NEAR(oc.sample_covariance(), 3.0, 0.2);  // Cov(x, 3x+noise) ~ 3 Var(x) = 3
}

TEST(LindeBuzoGray, DistortionDecreasesWithCodebookSize) {
    // Four blobs; distortion should drop sharply as the codebook grows to 4.
    std::mt19937_64                  rng(5);
    std::normal_distribution<double> j(0.0, 0.2);
    std::vector<std::vector<double>> data;
    const std::vector<std::vector<double>> c = {{0, 0}, {10, 0}, {0, 10}, {10, 10}};
    for (const auto& ctr : c)
        for (int i = 0; i < 50; ++i) data.push_back({ctr[0] + j(rng), ctr[1] + j(rng)});

    auto vq = linde_buzo_gray(data, 4, 0.01, 100, 1e-8);
    ASSERT_EQ(vq.codebook.size(), 4u);
    // Distortion history is 1,2,4 codewords: strictly decreasing.
    ASSERT_GE(vq.distortion_history.size(), 2u);
    for (std::size_t i = 1; i < vq.distortion_history.size(); ++i)
        EXPECT_LT(vq.distortion_history[i], vq.distortion_history[i - 1]);
    // With 4 codewords for 4 tight blobs, final distortion is tiny.
    EXPECT_LT(vq.distortion, 0.2);
}
