// std::format / std::print support for datamunge::bayes types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <string_view>

namespace by = dm::bayes;

static_assert(std::formattable<by::INLAIntegrationStrategy, char>);
static_assert(std::formattable<by::HMCOptions, char>);
static_assert(std::formattable<by::HMCResult, char>);
static_assert(std::formattable<by::NUTSOptions, char>);
static_assert(std::formattable<by::NUTSResult, char>);
static_assert(std::formattable<by::RWMOptions, char>);
static_assert(std::formattable<by::RWMResult, char>);
static_assert(std::formattable<by::GibbsOptions, char>);
static_assert(std::formattable<by::GibbsResult, char>);
static_assert(std::formattable<by::ImportanceSamplingOptions, char>);
static_assert(std::formattable<by::ImportanceSamplingResult, char>);
static_assert(std::formattable<by::INLALikelihoodPoint, char>);
static_assert(std::formattable<by::INLAOptions, char>);
static_assert(std::formattable<by::INLAResult, char>);
static_assert(std::formattable<by::MAPOptions, char>);
static_assert(std::formattable<by::HMC, char>);
static_assert(std::formattable<by::NUTS, char>);
static_assert(std::formattable<by::RandomWalkMetropolis, char>);
static_assert(std::formattable<by::GibbsSampler, char>);
static_assert(std::formattable<by::ImportanceSampling, char>);
static_assert(std::formattable<by::INLA, char>);
static_assert(std::formattable<by::MAP, char>);

namespace {
const std::vector<std::vector<double>> kDraws{{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0}};
}

TEST(BayesFormat, Enum) {
    EXPECT_EQ(std::format("{}", by::INLAIntegrationStrategy::EmpiricalBayes), "EmpiricalBayes");
    EXPECT_EQ(std::format("{}", static_cast<by::INLAIntegrationStrategy>(5)), "INLAIntegrationStrategy(5)");
}

TEST(BayesFormat, Options) {
    EXPECT_EQ(std::format("{}", by::HMCOptions{}),
              "HMCOptions(num_samples=1000, num_warmup=1000, num_leapfrog_steps=10, initial_step_size=0.1, "
              "target_accept_rate=0.8, seed=42)");
    EXPECT_EQ(std::format("{}", by::NUTSOptions{}),
              "NUTSOptions(num_samples=1000, num_warmup=1000, max_tree_depth=10, initial_step_size=0.1, "
              "target_accept_rate=0.8, max_delta_error=1000, seed=42)");
    EXPECT_EQ(std::format("{}", by::RWMOptions{}),
              "RWMOptions(num_samples=1000, num_warmup=1000, initial_step_size=1, target_accept_rate=0.234, seed=42)");
    EXPECT_EQ(std::format("{}", by::GibbsOptions{}),
              "GibbsOptions(num_samples=1000, num_warmup=1000, initial_step_sizes=[1], target_accept_rate=0.44, seed=42)");
    EXPECT_EQ(std::format("{}", by::ImportanceSamplingOptions{}), "ImportanceSamplingOptions(num_samples=1000, seed=42)");
    EXPECT_EQ(std::format("{}", by::MAPOptions{}), "MAPOptions(max_iterations=500, tolerance=1e-08, history_size=10)");
    EXPECT_EQ(std::format("{}", by::INLAOptions{}),
              "INLAOptions(strategy=Grid, newton_max_iterations=100, newton_tolerance=1e-08, grid_points_per_dim=7, "
              "grid_span=4, mode_population_size=40, mode_max_generations=200, seed=42)");
}

TEST(BayesFormat, SamplerResults) {
    by::HMCResult h{kDraws, 0.75, 0.5};
    EXPECT_EQ(std::format("{}", h),
              "HMCResult(draws=3, dim=2, mean=[2, 20], sd=[1, 10], accept_rate=0.75, final_step_size=0.5)");
    EXPECT_EQ(std::format("{}", by::HMCResult{}), "HMCResult(draws=0, accept_rate=0, final_step_size=0)");
    EXPECT_EQ(std::format("{}", by::NUTSResult{kDraws, 0.9, 0.25, 2}),
              "NUTSResult(draws=3, dim=2, mean=[2, 20], sd=[1, 10], accept_rate=0.9, final_step_size=0.25, divergences=2)");
    EXPECT_EQ(std::format("{}", by::RWMResult{{{4.0}}, 0.2, 1.5}),
              "RWMResult(draws=1, dim=1, mean=[4], accept_rate=0.2, final_step_size=1.5)");
    EXPECT_EQ(std::format("{}", by::GibbsResult{kDraws, {0.4, 0.5}, {1, 2}}),
              "GibbsResult(draws=3, dim=2, mean=[2, 20], sd=[1, 10], accept_rates=[0.4, 0.5], final_step_sizes=[1, 2])");
    EXPECT_EQ(std::format("{}", by::ImportanceSamplingResult{kDraws, {0.5, 0.25, 0.25}, 2.5, -1.25}),
              "ImportanceSamplingResult(draws=3, dim=2, weighted_mean=[1.75, 17.5], effective_sample_size=2.5, "
              "log_evidence=-1.25)");
}

TEST(BayesFormat, InlaAndSamplers) {
    EXPECT_EQ(std::format("{}", by::INLALikelihoodPoint{-1.5, 0.5, 2}),
              "INLALikelihoodPoint(log_density=-1.5, score=0.5, curvature=2)");
    by::INLAResult r;
    r.latent_mean = {1, 2};
    r.latent_sd = {0.5, 0.25};
    r.theta_mean = {3};
    r.theta_sd = {0.125};
    r.theta_dim = 1;
    r.log_marginal_likelihood = -10.5;
    r.theta_grid_weights = {0.25, 0.5, 0.25};
    EXPECT_EQ(std::format("{}", r),
              "INLAResult(theta_dim=1, log_marginal_likelihood=-10.5, grid_points=3)\n"
              "  latent_mean: [1, 2]\n  latent_sd:   [0.5, 0.25]\n  theta_mean:  [3]\n  theta_sd:    [0.125]");
}

/// @brief Expected sampler text: its options' text with the options type name replaced.
template <typename Sampler>
static std::string expected_sampler_text(std::string_view name, const Sampler& sampler) {
    std::string options = std::format("{}", sampler.options());
    return std::string(name) + options.substr(options.find('('));
}

TEST(BayesFormat, SamplersPrintTheirOptions) {
    EXPECT_EQ(std::format("{}", by::MAP{by::MAPOptions{.max_iterations = 5, .tolerance = 0.5, .history_size = 3}}),
              "MAP(max_iterations=5, tolerance=0.5, history_size=3)");
    EXPECT_EQ(std::format("{}", by::HMC{}).rfind("HMC(num_samples=1000, num_warmup=1000, ", 0), 0u);
    EXPECT_EQ(std::format("{}", by::HMC{}), expected_sampler_text("HMC", by::HMC{}));
    EXPECT_EQ(std::format("{}", by::NUTS{}), expected_sampler_text("NUTS", by::NUTS{}));
    EXPECT_EQ(std::format("{}", by::RandomWalkMetropolis{}),
              expected_sampler_text("RandomWalkMetropolis", by::RandomWalkMetropolis{}));
    EXPECT_EQ(std::format("{}", by::GibbsSampler{}), expected_sampler_text("GibbsSampler", by::GibbsSampler{}));
    EXPECT_EQ(std::format("{}", by::ImportanceSampling{}),
              expected_sampler_text("ImportanceSampling", by::ImportanceSampling{}));
    EXPECT_EQ(std::format("{}", by::INLA{}), expected_sampler_text("INLA", by::INLA{}));
}
