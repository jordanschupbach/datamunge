#include <datamunge/algorithms/ziggurat.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace alg = datamunge::algorithms;
TEST(Ziggurat, RejectsUnsupportedLayersBeforeAllocation) {
    for (auto layers : {0u, 1u, 64u, 129u, 255u, 257u})
        EXPECT_THROW((void)alg::ZigguratNormal(layers), std::invalid_argument);
    EXPECT_THROW((void)alg::ZigguratNormal(std::numeric_limits<std::size_t>::max()), std::invalid_argument);
}
TEST(Ziggurat, TablesHaveEqualAreaAndValidGeometry) {
    for (auto layers : {128u, 256u}) {
        const alg::ZigguratNormal sampler(layers);
        const auto table = sampler.table();
        EXPECT_EQ(table.nrows(), layers);
        const double r = sampler.tail_boundary();
        const double v = r * std::exp(-r*r/2) + std::sqrt(std::acos(-1.0)/2) * std::erfc(r/std::sqrt(2.0));
        for (std::size_t i = 0; i < layers; ++i) {
            EXPECT_GT(table.double_at("outer", i), table.double_at("inner", i));
            EXPECT_NEAR(table.double_at("rectangle_area", i), v, 5e-10);
            EXPECT_NEAR(table.double_at("fast_fraction", i), table.double_at("inner", i)/table.double_at("outer", i), 1e-15);
        }
        EXPECT_EQ(table.double_at("inner", layers-1), 0);
        EXPECT_THROW(sampler.plot_envelope(0), std::invalid_argument);
        EXPECT_FALSE(sampler.plot_envelope(16).to_svg().empty());
    }
}
TEST(Ziggurat, InstrumentationConservesReturnedDrawsAndProposals) {
    for (auto layers : {128u, 256u}) {
        const alg::ZigguratNormal sampler(layers);
        const auto d = sampler.diagnostics(1000000, 27);
        auto get = [&](const char* key) { return d.double_at(key, 0); };
        EXPECT_EQ(get("fast") + get("wedge") + get("tail"), get("draws"));
        EXPECT_EQ(get("draws") + get("rejected"), get("proposals"));
        EXPECT_GE(get("tail_trials"), get("tail"));
        EXPECT_GT(get("tail"), 100);
        EXPECT_NEAR(get("fast_per_proposal"), get("expected_fast_per_proposal"), .001);
        EXPECT_NEAR(get("draws") / get("proposals"), get("expected_acceptance"), .001);
    }
}
TEST(Ziggurat, BatchMatchesScalarAndNormalTailsForBothTables) {
    for (auto layers : {128u, 256u}) {
        const alg::ZigguratNormal sampler(layers);
        const auto sample = sampler.draw(500000, 19);
        std::mt19937_64 rng(19);
        for (std::size_t i = 0; i < 100; ++i) EXPECT_DOUBLE_EQ(sample[i], sampler.sample(rng));
        double sum = 0, sumsq = 0;
        int tails = 0;
        for (double z : sample) { ASSERT_TRUE(std::isfinite(z)); sum += z; sumsq += z*z; tails += std::abs(z)>3.5; }
        EXPECT_NEAR(sum/sample.size(), 0, .008);
        EXPECT_NEAR(sumsq/sample.size(), 1, .012);
        const double p = std::erfc(3.5/std::sqrt(2.0));
        EXPECT_NEAR(tails, sample.size()*p, 7*std::sqrt(sample.size()*p*(1-p)));
    }
    const alg::ZigguratNormal sampler;
    EXPECT_TRUE(sampler.draw(0).empty());
    EXPECT_EQ(sampler.draw(3, 0, 7, 0), (std::vector<double>{7,7,7}));
    EXPECT_THROW(sampler.draw(3, 0, 0, -1), std::invalid_argument);
    EXPECT_THROW(sampler.draw(3, 0, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(sampler.diagnostics(0), std::invalid_argument);
}
TEST(Ziggurat, MeasurementSimulationChecksOracleAndDoesNotMutateInput) {
    const alg::ZigguratNormal sampler;
    datamunge::dstruct::DataFrame frame;
    frame.add_column("mu", std::vector<double>{2, 0, -1});
    frame.add_column("se", std::vector<double>{1, 0, 0});
    const auto out = sampler.simulate_measurements(frame, "mu", "se", 100000, 17);
    EXPECT_EQ(frame.ncols(), 2);
    EXPECT_EQ(out.ncols(), 7);
    EXPECT_NEAR(out.double_at("mc_mean", 0), 2, .02);
    EXPECT_NEAR(out.double_at("mc_sd", 0), 1, .02);
    EXPECT_NEAR(out.double_at("normal_positive", 0), .9772498680518208, 1e-15);
    EXPECT_NEAR(out.double_at("mc_positive", 0), out.double_at("normal_positive", 0), .004);
    EXPECT_EQ(out.double_at("mc_sd", 1), 0);
    EXPECT_EQ(out.double_at("mc_positive", 1), 0); // Strictly greater than zero.
    EXPECT_EQ(out.double_at("normal_positive", 2), 0);
    EXPECT_THROW(sampler.simulate_measurements(frame, "missing", "se"), std::out_of_range);
    EXPECT_THROW(sampler.simulate_measurements(frame, "mu", "se", 1), std::invalid_argument);
    EXPECT_THROW(sampler.simulate_measurements(out, "mu", "se"), std::invalid_argument);
    frame.set_null("mu", 0);
    EXPECT_THROW(sampler.simulate_measurements(frame, "mu", "se"), std::invalid_argument);
}
