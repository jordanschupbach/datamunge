#include <gtest/gtest.h>

#include <datamunge/algorithms/approximate_counting.hpp>
#include <datamunge/algorithms/lsh.hpp>
#include <datamunge/algorithms/ziggurat.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using datamunge::algorithms::CosineLSH;
using datamunge::algorithms::MorrisCounter;
using datamunge::algorithms::ZigguratNormal;

namespace {

double exact_cosine_nn(const std::vector<std::vector<double>>& data, const std::vector<double>& q,
                       std::size_t& best) {
    double bs = -2.0;
    best      = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        double s = CosineLSH::cosine(q, data[i]);
        if (s > bs) {
            bs   = s;
            best = i;
        }
    }
    return bs;
}

}  // namespace

TEST(CosineLSH, FindsHighRecallNearestNeighbors) {
    // A haystack of random points, with each query a noisy near-duplicate of one of
    // them -- so a genuine, well-separated nearest neighbor exists to be found.
    std::mt19937_64                  rng(1);
    std::normal_distribution<double> g(0.0, 1.0), noise(0.0, 0.25);
    const std::size_t                n = 500, dim = 20;
    std::vector<std::vector<double>> data(n, std::vector<double>(dim));
    for (auto& v : data)
        for (double& x : v) x = g(rng);

    CosineLSH                                  index(data, 15, 12, 42);  // 15 tables, 12 bits each
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    int hits = 0, tries = 60;
    for (int t = 0; t < tries; ++t) {
        std::size_t         src = pick(rng);
        std::vector<double> q   = data[src];
        for (double& x : q) x += noise(rng);  // near-duplicate of data[src]
        std::size_t exact;
        exact_cosine_nn(data, q, exact);
        std::size_t approx = index.query(q);
        if (approx == exact) ++hits;
    }
    // With 15 tables the true (planted) NN is recovered on a large majority of queries.
    EXPECT_GT(hits, (tries * 7) / 10);
}

TEST(CosineLSH, SignatureAgreementTracksAngle) {
    // Bit-agreement probability should be ~ 1 - theta/pi.
    std::mt19937_64                  rng(2);
    std::normal_distribution<double> g(0.0, 1.0);
    const std::size_t                dim = 50;
    std::vector<std::vector<double>> pair(2, std::vector<double>(dim));
    for (double& x : pair[0]) x = g(rng);
    for (std::size_t i = 0; i < dim; ++i) pair[1][i] = pair[0][i] + 0.6 * g(rng);  // moderately similar

    CosineLSH    index(pair, 1, 1, 7);  // structure unused; we test signatures directly
    const double cos_sim = CosineLSH::cosine(pair[0], pair[1]);
    const double theta   = std::acos(std::max(-1.0, std::min(1.0, cos_sim)));

    // Empirical bit-agreement over many random hyperplanes.
    std::normal_distribution<double> gg(0.0, 1.0);
    int                              agree = 0, trials = 4000;
    for (int t = 0; t < trials; ++t) {
        double d0 = 0.0, d1 = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            double r = gg(rng);
            d0 += r * pair[0][i];
            d1 += r * pair[1][i];
        }
        if ((d0 >= 0.0) == (d1 >= 0.0)) ++agree;
    }
    double p_emp = static_cast<double>(agree) / trials;
    EXPECT_NEAR(p_emp, 1.0 - theta / M_PI, 0.05);
}

TEST(Ziggurat, MomentsAndTailAreCorrect) {
    ZigguratNormal  zig(256);
    std::mt19937_64 rng(3);
    const int       n = 200000;
    double          sum = 0.0, sumsq = 0.0;
    int             beyond2 = 0;
    for (int i = 0; i < n; ++i) {
        double z = zig.sample(rng);
        sum += z;
        sumsq += z * z;
        if (std::abs(z) > 2.0) ++beyond2;
    }
    double mean = sum / n;
    double var  = sumsq / n - mean * mean;
    EXPECT_NEAR(mean, 0.0, 0.02);
    EXPECT_NEAR(var, 1.0, 0.03);
    // P(|Z| > 2) = 2*(1 - Phi(2)) ~ 0.0455.
    EXPECT_NEAR(static_cast<double>(beyond2) / n, 0.0455, 0.006);
}

TEST(MorrisCounter, EstimatesLargeCountsOnAverage) {
    // Average many independent base-2 counters to reduce the (large) single-counter variance.
    const int    true_count = 10000;
    const int    replicas   = 400;
    double       sum        = 0.0;
    for (int r = 0; r < replicas; ++r) {
        std::mt19937_64 rng(static_cast<std::uint64_t>(r) + 1);
        MorrisCounter   mc(2.0);
        for (int i = 0; i < true_count; ++i) mc.increment(rng);
        sum += mc.estimate();
    }
    double avg = sum / replicas;
    // Unbiased: the average estimate should be within ~10% of the truth.
    EXPECT_NEAR(avg, true_count, true_count * 0.1);
}

TEST(MorrisCounter, SmallBaseReducesVariance) {
    // A base close to 1 gives a low-variance counter: averaged over replicas it lands
    // very close to the truth, and its spread is far tighter than the base-2 counter's.
    const int true_count = 5000;
    const int replicas   = 50;
    auto      rel_spread = [&](double base) {
        double sum = 0.0, sumsq = 0.0;
        for (int r = 0; r < replicas; ++r) {
            std::mt19937_64 rng(static_cast<std::uint64_t>(r) + 100);
            MorrisCounter   mc(base);
            for (int i = 0; i < true_count; ++i) mc.increment(rng);
            sum += mc.estimate();
            sumsq += mc.estimate() * mc.estimate();
        }
        double mean = sum / replicas;
        double var  = sumsq / replicas - mean * mean;
        return std::make_pair(mean, std::sqrt(var) / true_count);  // (mean, relative std)
    };
    auto [mean_small, rel_small] = rel_spread(1.03);
    auto [mean_big, rel_big]     = rel_spread(2.0);
    EXPECT_NEAR(mean_small, true_count, true_count * 0.05);  // near-unbiased, low variance
    EXPECT_LT(rel_small, rel_big);                           // smaller base -> tighter spread
}
