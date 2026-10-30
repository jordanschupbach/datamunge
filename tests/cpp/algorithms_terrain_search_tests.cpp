#include <gtest/gtest.h>

#include <datamunge/algorithms/bellman_ford.hpp>
#include <datamunge/algorithms/dijkstra.hpp>
#include <datamunge/algorithms/terrain_search.hpp>
#include <datamunge/algorithms/tour_search.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/plot.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

using namespace datamunge::algorithms;

namespace {

TerrainGrid random_hills(std::size_t rows, std::size_t cols, unsigned seed, double climb) {
    TerrainGrid g;
    g.rows = rows;
    g.cols = cols;
    g.cell_size = 2.0;
    g.climb_cost = climb;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double a = u(rng) * 0.5, b = u(rng) * 0.5;
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            g.elevation.push_back(10 * std::sin(a * i) * std::cos(b * j) + 3 * u(rng));
    return g;
}

}  // namespace

TEST(TerrainSearch, AllAdmissibleHeuristicsFindTheDijkstraOptimum) {
    for (unsigned seed = 1; seed <= 5; ++seed) {
        const auto g = random_hills(25, 30, seed, 3.0);
        const std::size_t s = 3, t = g.elevation.size() - 5;
        const auto dijkstra = terrain_a_star(g, s, t, TerrainHeuristic::Zero);
        ASSERT_TRUE(dijkstra.found);
        // The generic graph A* on the same edges agrees.
        const auto generic = a_star_search(g.elevation.size(), terrain_edges(g), s, t,
                                           std::vector<double>(g.elevation.size(), 0.0));
        EXPECT_NEAR(generic.cost, dijkstra.cost, 1e-9);
        const auto to_go = terrain_cost_to_go(g, t);
        EXPECT_NEAR(to_go[s], dijkstra.cost, 1e-9);
        EXPECT_NEAR(terrain_cost_from(g, s)[t], dijkstra.cost, 1e-9);
        std::size_t previous = dijkstra.expansion_order.size();
        for (auto kind : {TerrainHeuristic::Euclidean, TerrainHeuristic::Octile, TerrainHeuristic::OctileClimb}) {
            const auto r = terrain_a_star(g, s, t, kind);
            EXPECT_NEAR(r.cost, dijkstra.cost, 1e-9);
            EXPECT_LE(r.expansion_order.size(), previous);  // tighter bounds expand fewer cells
            previous = r.expansion_order.size();
            // Admissible and consistent everywhere.
            for (std::size_t c = 0; c < g.elevation.size(); ++c)
                EXPECT_LE(terrain_heuristic(g, c, t, kind), to_go[c] + 1e-9);
        }
    }
}

TEST(TerrainSearch, HeuristicsAreConsistent) {
    const auto g = random_hills(12, 12, 9, 2.5);
    const std::size_t t = 77;
    for (auto kind : {TerrainHeuristic::Euclidean, TerrainHeuristic::Octile, TerrainHeuristic::OctileClimb})
        for (const auto& [u, v, w] : terrain_edges(g))
            EXPECT_LE(terrain_heuristic(g, u, t, kind), w + terrain_heuristic(g, v, t, kind) + 1e-12);
}

TEST(TerrainSearch, WeightedAStarIsBoundedSuboptimal) {
    for (unsigned seed = 1; seed <= 5; ++seed) {
        const auto g = random_hills(30, 30, seed, 4.0);
        const std::size_t s = 0, t = g.elevation.size() - 1;
        const auto opt = terrain_a_star(g, s, t, TerrainHeuristic::OctileClimb);
        for (double w : {1.5, 2.0, 5.0}) {
            for (bool reopen : {true, false}) {
                const auto r = terrain_a_star(g, s, t, TerrainHeuristic::OctileClimb, w, reopen);
                ASSERT_TRUE(r.found);
                EXPECT_GE(r.cost, opt.cost - 1e-9);
                EXPECT_LE(r.cost, w * opt.cost + 1e-9);
                if (!reopen) EXPECT_LE(r.expansion_order.size(), g.elevation.size());
            }
        }
    }
    EXPECT_THROW(terrain_a_star(random_hills(3, 3, 1, 0), 0, 8, TerrainHeuristic::Zero, 0.5), std::invalid_argument);
}

TEST(TerrainSearch, PathBookkeeping) {
    TerrainGrid g{1, 4, {0, 5, 2, 2}, 10.0, 2.0};
    const auto r = terrain_a_star(g, 0, 3, TerrainHeuristic::Octile);
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.path, (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_DOUBLE_EQ(r.length, 30.0);
    EXPECT_DOUBLE_EQ(r.ascent, 5.0);
    EXPECT_DOUBLE_EQ(r.descent, 3.0);
    EXPECT_DOUBLE_EQ(r.cost, 30.0 + 2.0 * 5.0);
    EXPECT_DOUBLE_EQ(terrain_step_cost(g, 1, 0), 10.0);  // descending is free
    EXPECT_THROW(terrain_step_cost(g, 0, 2), std::invalid_argument);
}

TEST(TerrainSearch, FromVolcanoFrame) {
    const auto g = terrain_from_frame(datamunge::datasets::volcano(), "x", "y", "elevation", 5.0);
    EXPECT_EQ(g.rows, 87u);
    EXPECT_EQ(g.cols, 61u);
    EXPECT_DOUBLE_EQ(g.cell_size, 10.0);
    EXPECT_DOUBLE_EQ(g.elevation[19 * 61 + 30], 195.0);  // R: volcano[20, 31]
    EXPECT_DOUBLE_EQ(g.elevation[0], 100.0);
}

TEST(RPlotImage, CellsAndTerrainPalette) {
    using namespace datamunge::graphics;
    auto p = RPlot::create().image({0.0, 1.0}, {0.0, 1.0, 2.0}, {{1, 2, 3}, {4, 5, std::nan("")}});
    EXPECT_EQ(p.series().size(), 5u);
    EXPECT_THROW(RPlot::create().image({0.0, 1.0}, {0.0, 1.0}, {{1, 2}}), std::invalid_argument);
    // R: col2rgb(terrain.colors(7))
    const auto t = terrain_colors(7);
    ASSERT_EQ(t.size(), 7u);
    EXPECT_EQ(t[0].r, 0);
    EXPECT_EQ(t[0].g, 166);
    EXPECT_EQ(t[1].r, 99);
    EXPECT_EQ(t[1].g, 198);
    EXPECT_EQ(t[3].r, 233);
    EXPECT_EQ(t[3].g, 189);
    EXPECT_EQ(t[3].b, 58);
    EXPECT_EQ(t[5].b, 179);
    EXPECT_EQ(t[6].r, 242);
}

TEST(RPlotImage, DivergingPalette) {
    using namespace datamunge::graphics;
    const auto d = diverging_colors(7);  // exactly the RdBu anchors, blue to red
    ASSERT_EQ(d.size(), 7u);
    EXPECT_EQ(d[0].r, 33);
    EXPECT_EQ(d[0].b, 172);
    EXPECT_EQ(d[3].r, 247);
    EXPECT_EQ(d[3].g, 247);
    EXPECT_EQ(d[6].r, 178);
    EXPECT_EQ(d[6].b, 43);
    const auto one = diverging_colors(1);
    EXPECT_EQ(one[0].g, 247);
    EXPECT_EQ(diverging_colors(13)[6].b, 247);  // odd n: white in the middle
    EXPECT_THROW(diverging_colors(0), std::invalid_argument);
}

TEST(TerrainSearch, ConsistentHeuristicNeverReopensCells) {
    // Equal-cost grid routes differ in the last bit; that must not count as an improvement.
    const auto g = terrain_from_frame(datamunge::datasets::volcano(), "x", "y", "elevation", 5000.0 / 600.0);
    const std::size_t s = 80 * g.cols + 5, t = 5 * g.cols + 55;
    const auto gs = terrain_cost_from(g, s);
    for (auto kind : {TerrainHeuristic::Zero, TerrainHeuristic::Octile, TerrainHeuristic::OctileClimb}) {
        const auto r = terrain_a_star(g, s, t, kind);
        std::vector<std::size_t> sorted = r.expansion_order;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end());
        for (std::size_t c : r.expansion_order) EXPECT_LE(gs[c] + terrain_heuristic(g, c, t, kind), gs[t] + 1e-9);
    }
}

TEST(TerrainBeamSearch, UnlimitedWidthIsOptimal) {
    for (unsigned seed = 1; seed <= 5; ++seed) {
        const auto g = random_hills(20, 24, seed, 3.0);
        const std::size_t s = 2, t = g.elevation.size() - 3;
        const double optimum = terrain_cost_from(g, s)[t];
        for (auto kind : {TerrainHeuristic::Zero, TerrainHeuristic::OctileClimb}) {
            const auto r = terrain_beam_search(g, s, t, kind, g.elevation.size());
            ASSERT_TRUE(r.best.found);
            EXPECT_NEAR(r.best.cost, optimum, 1e-9);
            EXPECT_EQ(r.best.path.front(), s);
            EXPECT_EQ(r.best.path.back(), t);
            EXPECT_EQ(r.layer_start.size(), r.layers);
            // Incumbents strictly improve and end at the returned cost.
            for (std::size_t k = 1; k < r.incumbent_costs.size(); ++k)
                EXPECT_LT(r.incumbent_costs[k], r.incumbent_costs[k - 1]);
            EXPECT_NEAR(r.incumbent_costs.back(), r.best.cost, 1e-9);
        }
    }
}

TEST(TerrainBeamSearch, NarrowBeamsAreBoundedByWidthAndNeverBeatTheOptimum) {
    const auto g = random_hills(30, 30, 7, 3.0);
    const std::size_t s = 0, t = g.elevation.size() - 1;
    const double optimum = terrain_cost_from(g, s)[t];
    for (std::size_t width : {1u, 2u, 5u, 20u}) {
        for (auto ranking : {BeamRanking::CostPlusHeuristic, BeamRanking::Heuristic}) {
            const auto r = terrain_beam_search(g, s, t, TerrainHeuristic::Octile, width, ranking);
            EXPECT_LE(r.max_beam, width);
            if (r.best.found) {
                EXPECT_GE(r.best.cost, optimum - 1e-9);
                EXPECT_LE(r.best.cost, r.first_cost + 1e-9);
            }
            const auto first = terrain_beam_search(g, s, t, TerrainHeuristic::Octile, width, ranking, true);
            ASSERT_EQ(first.best.found, r.best.found);
            if (first.best.found) {
                EXPECT_NEAR(first.first_cost, r.first_cost, 1e-9);
                EXPECT_LE(first.best.cost, r.first_cost + 1e-9);  // the traced path can only be cheaper
                EXPECT_EQ(first.layers, r.first_layer);
            }
        }
    }
    // Width 1 with the heuristic alone walks straight down an empty grid.
    TerrainGrid flat{5, 5, std::vector<double>(25, 0.0), 1.0, 0.0};
    const auto r = terrain_beam_search(flat, 0, 24, TerrainHeuristic::Octile, 1, BeamRanking::Heuristic, true);
    ASSERT_TRUE(r.best.found);
    EXPECT_EQ(r.best.path.size(), 5u);
    EXPECT_NEAR(r.best.cost, 4 * std::sqrt(2.0), 1e-12);
    EXPECT_THROW(terrain_beam_search(flat, 0, 24, TerrainHeuristic::Octile, 0), std::invalid_argument);
}

TEST(TerrainBeamStackSearch, OptimalForEveryWidth) {
    for (unsigned seed = 1; seed <= 4; ++seed) {
        const auto g = random_hills(12, 14, seed, 3.0);
        const std::size_t s = 1, t = g.elevation.size() - 2;
        const double optimum = terrain_cost_from(g, s)[t];
        for (std::size_t width : {1u, 2u, 3u, 8u, 1000u}) {
            const auto r = terrain_beam_stack_search(g, s, t, TerrainHeuristic::OctileClimb, width);
            ASSERT_TRUE(r.best.found) << "seed " << seed << " width " << width;
            EXPECT_NEAR(r.best.cost, optimum, 1e-9) << "seed " << seed << " width " << width;
            EXPECT_EQ(r.best.path.front(), s);
            EXPECT_EQ(r.best.path.back(), t);
            EXPECT_LE(r.max_stored, width * (r.max_depth + 1) + 1);
            for (std::size_t k = 1; k < r.incumbent_costs.size(); ++k)
                EXPECT_LT(r.incumbent_costs[k], r.incumbent_costs[k - 1]);
            if (width == 1000u) EXPECT_EQ(r.backtracks, 0u);
            EXPECT_TRUE(r.exhausted);
            const auto table = terrain_beam_stack_search(g, s, t, TerrainHeuristic::OctileClimb, width,
                                                         std::numeric_limits<double>::infinity(), true);
            ASSERT_TRUE(table.best.found);
            EXPECT_NEAR(table.best.cost, optimum, 1e-9) << "table, seed " << seed << " width " << width;
            EXPECT_LE(table.best.expansion_order.size(), r.best.expansion_order.size());
        }
    }
}

TEST(TerrainBeamStackSearch, UpperBoundAndUnreachableBound) {
    const auto g = random_hills(10, 10, 3, 2.0);
    const std::size_t s = 0, t = 99;
    const double optimum = terrain_cost_from(g, s)[t];
    const auto seeded = terrain_beam_stack_search(g, s, t, TerrainHeuristic::Octile, 2, optimum * 1.01);
    ASSERT_TRUE(seeded.best.found);
    EXPECT_NEAR(seeded.best.cost, optimum, 1e-9);
    // No path is cheaper than the optimum itself.
    const auto none = terrain_beam_stack_search(g, s, t, TerrainHeuristic::Octile, 2, optimum - 1e-6);
    EXPECT_FALSE(none.best.found);
    EXPECT_THROW(terrain_beam_stack_search(g, s, t, TerrainHeuristic::Octile, 0), std::invalid_argument);
    // A tiny budget stops the search early without a proof of optimality.
    const auto capped = terrain_beam_stack_search(g, s, t, TerrainHeuristic::Octile, 1,
                                                  std::numeric_limits<double>::infinity(), false, 5);
    EXPECT_FALSE(capped.exhausted);
    EXPECT_LE(capped.best.expansion_order.size(), 5u + 1u);
}

TEST(TerrainEnergyEdges, NegativeEdgesButNoNegativeCycleUnlessCreditExceedsClimbCost) {
    const auto g = random_hills(15, 18, 4, 3.0);
    const std::size_t s = 7, n = g.elevation.size();
    const double kappa = 0.8 * g.climb_cost;
    const auto edges = terrain_energy_edges(g, kappa);
    EXPECT_TRUE(std::ranges::any_of(edges, [](const auto& e) { return std::get<2>(e) < 0; }));
    const auto bf = bellman_ford(n, edges, s);
    ASSERT_FALSE(bf.has_negative_cycle);
    // Johnson reweighting with the potential kappa * z makes every edge non-negative.
    auto reduced = edges;
    for (auto& [u, v, w] : reduced) {
        w += kappa * (g.elevation[u] - g.elevation[v]);
        EXPECT_GE(w, -1e-9);
        w = std::max(w, 0.0);
    }
    const auto dj = dijkstra(n, reduced, s);
    for (std::size_t v = 0; v < n; ++v)
        EXPECT_NEAR(bf.distance[v], dj.distance[v] + kappa * (g.elevation[v] - g.elevation[s]), 1e-9);
    // A credit above the climb cost makes steep up-and-down loops gain energy.
    const auto perpetual = bellman_ford_queue(n, terrain_energy_edges(g, 1.5 * g.climb_cost), s);
    EXPECT_TRUE(perpetual.has_negative_cycle);
    EXPECT_GE(perpetual.negative_cycle.size(), 2u);
}

TEST(TerrainBestFirst, AStarModeIsOptimalAndGreedyWithPerfectHeuristicFollowsOnePath) {
    for (unsigned seed = 1; seed <= 4; ++seed) {
        const auto g = random_hills(20, 24, seed, 3.0);
        const std::size_t s = 2, t = g.elevation.size() - 3;
        const auto optimal = terrain_a_star(g, s, t, TerrainHeuristic::Zero);
        const auto hstar = terrain_cost_to_go(g, t);
        std::vector<double> octile(g.elevation.size());
        for (std::size_t c = 0; c < octile.size(); ++c) octile[c] = terrain_heuristic(g, c, t, TerrainHeuristic::OctileClimb);
        // alpha = 1 with an admissible, consistent heuristic: A*, optimal.
        EXPECT_NEAR(terrain_best_first(g, s, t, octile, 1.0).cost, optimal.cost, 1e-9);
        const auto perfect = terrain_best_first(g, s, t, hstar, 1.0);
        EXPECT_NEAR(perfect.cost, optimal.cost, 1e-9);
        // Greedy with h*: h* strictly decreases, so every expansion is the next path cell.
        const auto greedy = terrain_best_first(g, s, t, hstar, 0.0);
        ASSERT_TRUE(greedy.found);
        EXPECT_EQ(greedy.expansion_order, greedy.path);
        EXPECT_GE(greedy.cost, optimal.cost - 1e-9);
        // Greedy with a weak heuristic still finds a (possibly worse) path.
        const auto weak = terrain_best_first(g, s, t, octile, 0.0);
        EXPECT_TRUE(weak.found);
        EXPECT_GE(weak.cost, optimal.cost - 1e-9);
    }
    const auto g = random_hills(5, 5, 1, 1.0);
    EXPECT_THROW((void)terrain_best_first(g, 0, 3, std::vector<double>(4, 0.0)), std::invalid_argument);
    EXPECT_THROW((void)terrain_best_first(g, 0, 3, std::vector<double>(25, 0.0), -1.0), std::invalid_argument);
}

