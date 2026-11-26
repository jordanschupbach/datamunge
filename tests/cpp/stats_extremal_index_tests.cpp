#include <datamunge/random/random.hpp>
#include <datamunge/stats/extremal_index.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_extremal_index_data.inc"
} // namespace

TEST(ExtremalIndex, IntervalsMatchExtRemes) {
    const double us[]{0.395, 0.6, 1.0, 1.5};
    for (std::size_t i = 0; i < 4; ++i) {
        const auto e = extremal_index(ei_fort, us[i]);
        EXPECT_NEAR(e.theta, ei_intervals[3 * i], 1e-14);
        EXPECT_EQ(e.clusters, static_cast<std::size_t>(ei_intervals[3 * i + 1]));
        EXPECT_EQ(e.run_length, static_cast<std::size_t>(ei_intervals[3 * i + 2]));
        EXPECT_FALSE(e.all_gaps_short);
        EXPECT_TRUE(std::isnan(e.lower));
    }
    // All gaps <= 2: the theta-hat form, capped at 1; every exceedance is its own cluster.
    const auto s = extremal_index(ei_short, 0.5);
    EXPECT_TRUE(s.all_gaps_short);
    EXPECT_EQ(s.theta, ei_short_theta[0]);
    EXPECT_EQ(s.clusters, s.exceedances);
    EXPECT_EQ(s.run_length, 0u);
}

TEST(ExtremalIndex, RunsAndDeclustering) {
    const auto r9 = extremal_index(ei_fort, 0.395, {.method = ExtremalIndexMethod::runs, .run_length = 9});
    EXPECT_NEAR(r9.theta, ei_runs[0], 1e-15);
    EXPECT_EQ(r9.clusters, static_cast<std::size_t>(ei_runs[1]));
    const auto r1 = extremal_index(ei_fort, 0.395, {.method = ExtremalIndexMethod::runs});
    EXPECT_NEAR(r1.theta, ei_runs[2], 1e-15);
    EXPECT_EQ(r1.clusters, static_cast<std::size_t>(ei_runs[3]));
    const auto d = decluster(ei_fort, 0.395, 9);
    ASSERT_EQ(d.cluster_maxima.size(), static_cast<std::size_t>(ei_decluster[1]));
    double sum = 0;
    for (const double v : d.cluster_maxima)
        sum += v;
    EXPECT_NEAR(sum, ei_decluster[2], 1e-10);
    EXPECT_EQ(*std::ranges::max_element(d.cluster_maxima), ei_decluster[3]);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ(d.cluster_maximum_index[i] + 1, static_cast<std::size_t>(ei_decluster[4 + i]));
    EXPECT_EQ(d.cluster.size(), d.exceedance_index.size());
    EXPECT_EQ(d.cluster.back() + 1, d.cluster_maxima.size());
    EXPECT_EQ(decluster(ei_fort, 0.395, 0).cluster_maxima.size(), d.exceedance_index.size());
    // The intervals estimate's run length reproduces its cluster count.
    const auto e = extremal_index(ei_fort, 0.395);
    EXPECT_EQ(decluster(ei_fort, 0.395, e.run_length).cluster_maxima.size(), e.clusters);
}

TEST(ExtremalIndex, BootstrapAndKnownTheta) {
    const auto e = extremal_index(ei_fort, 0.395, {.bootstrap = 4000, .seed = 5});
    EXPECT_EQ(e.bootstrap, 4000u);
    EXPECT_NEAR(e.lower, ei_boot[0], 0.01); // extRemes ci(), different random streams
    EXPECT_NEAR(e.upper, ei_boot[1], 0.01);
    EXPECT_EQ(extremal_index(ei_fort, 0.395, {.bootstrap = 200, .seed = 5}).lower,
              extremal_index(ei_fort, 0.395, {.bootstrap = 200, .seed = 5}).lower);
    const auto r = extremal_index(ei_fort, 0.395,
                                  {.method = ExtremalIndexMethod::runs, .run_length = 9, .bootstrap = 4000, .seed = 6});
    EXPECT_NEAR(r.lower, ei_boot_runs[0], 0.01);
    EXPECT_NEAR(r.upper, ei_boot_runs[1], 0.01);
    // Max-autoregressive process X_t = max(a X_{t-1}, (1 - a) Z_t), Z unit Frechet: theta = 1 - a.
    datamunge::random::SplitMix64 rng(11);
    std::vector<double> x(400000);
    double prev = 1;
    for (auto& v : x) {
        const double z = -1 / std::log(rng.uniform01() * 0.999999 + 5e-7);
        v = prev = std::max(0.6 * prev, 0.4 * z);
    }
    auto sorted = x;
    std::ranges::nth_element(sorted, sorted.begin() + 398000);
    const double u = sorted[398000];
    EXPECT_NEAR(extremal_index(x, u).theta, 0.4, 0.04);
    EXPECT_THROW(extremal_index(ei_fort, 100), std::invalid_argument);
    EXPECT_THROW(extremal_index(ei_fort, 0.395, {.method = ExtremalIndexMethod::runs, .run_length = 0}),
                 std::invalid_argument);
}
