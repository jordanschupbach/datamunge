#include <datamunge/stats/tail_index.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_tail_index_data.inc"
} // namespace

TEST(TailIndex, MatchesReIns) {
    const std::size_t hk[]{1, 10, 50, 200, 499}, mk[]{2, 10, 50, 200, 499}, pk[]{1, 10, 50, 125};
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(tail_index(ti_x, hk[i], TailIndexMethod::hill).estimate, ti_hill[i], 1e-13);
        EXPECT_NEAR(tail_index(ti_x, mk[i], TailIndexMethod::moment).estimate, ti_moment[i], 1e-12);
    }
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(tail_index(ti_x, pk[i], TailIndexMethod::pickands).estimate, ti_pickands[i], 1e-13);
    const std::size_t wk[]{10, 50, 200};
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(weissman_quantile(ti_x, wk[i], 1e-4) / ti_weissman[i], 1.0, 1e-13);
}

TEST(TailIndex, ExactParetoAndVariances) {
    // Exact Pareto quantiles with alpha = 2: log-spacings make the Hill estimate exactly the mean of
    // harmonic increments, and the standard error is xi / sqrt(k).
    std::vector<double> x;
    const std::size_t n = 1000;
    for (std::size_t i = 1; i <= n; ++i)
        x.push_back(std::pow(1 - (i - 0.5) / n, -0.5));
    const auto h = tail_index(x, 100, TailIndexMethod::hill);
    EXPECT_NEAR(h.estimate, 0.5, 0.01);
    EXPECT_NEAR(h.standard_error, h.estimate / 10, 1e-15);
    EXPECT_NEAR(h.upper - h.estimate, 1.959963984540054 * h.standard_error, 1e-12);
    const auto p = tail_index(x, 100, TailIndexMethod::pickands);
    EXPECT_NEAR(p.estimate, 0.5, 0.02);
    const double xi = p.estimate, d = 2 * (std::exp2(xi) - 1) * std::numbers::ln2;
    EXPECT_NEAR(p.standard_error, std::sqrt(xi * xi * (std::exp2(2 * xi + 1) + 1) / (d * d) / 100), 1e-14);
    const auto m = tail_index(x, 100, TailIndexMethod::moment);
    EXPECT_NEAR(m.estimate, 0.5, 0.03); // finite-k bias on a quantile grid
    EXPECT_NEAR(m.standard_error, std::sqrt((1 + m.estimate * m.estimate) / 100), 1e-14);
    // Location-scale invariance of Pickands; shift changes Hill.
    std::vector<double> y;
    for (const double v : x)
        y.push_back(3 * v + 7);
    EXPECT_NEAR(tail_index(y, 100, TailIndexMethod::pickands).estimate, p.estimate, 1e-12);
    // Short-tailed uniform data: moment and Pickands near -1.
    std::vector<double> u;
    for (std::size_t i = 1; i <= n; ++i)
        u.push_back((i - 0.5) / n);
    EXPECT_NEAR(tail_index(u, 100, TailIndexMethod::pickands).estimate, -1.0, 1e-10);
    EXPECT_NEAR(tail_index(u, 100, TailIndexMethod::moment).estimate, -1.0, 0.1);
}

TEST(TailIndex, PathsAndValidation) {
    const auto path = tail_index_path(ti_x, TailIndexMethod::pickands, 1, 1000);
    EXPECT_EQ(path.size(), 125u);
    EXPECT_NEAR(path[9].estimate, ti_pickands[1], 1e-13);
    EXPECT_EQ(tail_index_path(ti_x, TailIndexMethod::hill, 5, 20).size(), 16u);
    EXPECT_THROW(tail_index(ti_x, 126, TailIndexMethod::pickands), std::invalid_argument);
    EXPECT_THROW(tail_index(ti_x, 1, TailIndexMethod::moment), std::invalid_argument);
    EXPECT_THROW(tail_index(ti_x, 500, TailIndexMethod::hill), std::invalid_argument);
    EXPECT_THROW(tail_index({-3, -2, -1, 1}, 3, TailIndexMethod::hill), std::invalid_argument);
    EXPECT_THROW(tail_index_path(ti_x, TailIndexMethod::moment, 1, 5), std::invalid_argument);
    EXPECT_THROW(weissman_quantile(ti_x, 10, 0), std::invalid_argument);
}
