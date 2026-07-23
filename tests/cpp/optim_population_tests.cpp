#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::BayesianOptimization;
using datamunge::optim::BayesianOptimizationOptions;
using datamunge::optim::BayesianSurrogate;
using datamunge::optim::RBFGaussianProcessSurrogate;
using datamunge::optim::CMAES;
using datamunge::optim::CMAESOptions;
using datamunge::optim::DEOptions;
using datamunge::optim::DifferentialEvolution;
using datamunge::optim::GAOptions;
using datamunge::optim::GeneticAlgorithm;
using datamunge::optim::PSO;
using datamunge::optim::PSOOptions;

namespace {

class SphereFunction : public ArbitraryFunction {
public:
    explicit SphereFunction(std::vector<double> target) : target_(std::move(target)) {}
    double evaluate(const std::vector<double>& x) override {
        double total = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double d = x[i] - target_[i];
            total += d * d;
        }
        return total;
    }

private:
    std::vector<double> target_;
};

// Classic Rastrigin function: highly multimodal, global minimum f=0 at x=0.
class RastriginFunction : public ArbitraryFunction {
public:
    double evaluate(const std::vector<double>& x) override {
        double total = 10.0 * static_cast<double>(x.size());
        for (const double xi : x) total += xi * xi - 10.0 * std::cos(2.0 * M_PI * xi);
        return total;
    }
};

} // namespace

TEST(BayesianOptimization, UsesSurrogateWithinBounds) {
    class MockSurrogate : public BayesianSurrogate { public:
        int fits{0}; void fit(const std::vector<std::vector<double>>&, const std::vector<double>&) override { ++fits; }
        double acquisition(const std::vector<double>& x, double) override { return -std::abs(x[0] - 1.0); }
    } surrogate;
    SphereFunction f({1.0}); std::vector<double> x{0.0}; BayesianOptimizationOptions o; o.initial_samples=3; o.max_iterations=10; o.seed=7;
    const double v=BayesianOptimization(o).optimize(f,x,{-2.0},{2.0},surrogate);
    EXPECT_GE(x[0],-2.0); EXPECT_LE(x[0],2.0); EXPECT_LT(v,0.1); EXPECT_EQ(surrogate.fits,10);
}

TEST(BayesianOptimization, RunsWithRBFGaussianProcessSurrogate) {
    SphereFunction f({0.5});
    RBFGaussianProcessSurrogate surrogate(0.5, 1e-6);
    std::vector<double> x{0.0}; BayesianOptimizationOptions o; o.initial_samples=4; o.max_iterations=12; o.seed=7;
    const double value=BayesianOptimization(o).optimize(f,x,{-1.0},{1.0},surrogate);
    EXPECT_GE(x[0],-1.0); EXPECT_LE(x[0],1.0); EXPECT_LT(value,.05);
}

TEST(RBFGaussianProcessSurrogate, ProducesFiniteAcquisition) {
    RBFGaussianProcessSurrogate surrogate(1.0);
    surrogate.fit({{0.0}, {1.0}}, {1.0, 0.0});
    EXPECT_TRUE(std::isfinite(surrogate.acquisition({0.75}, 0.0)));
}

TEST(RBFGaussianProcessSurrogate, ExpectedImprovementIsFiniteAwayFromObservations) {
    RBFGaussianProcessSurrogate surrogate(0.5, 1e-6);
    surrogate.fit({{0.0}, {1.0}}, {1.0, 0.0});
    EXPECT_TRUE(std::isfinite(surrogate.acquisition({0.5}, 0.0)));
}

TEST(RBFGaussianProcessSurrogate, RejectsInvalidTrainingShape) {
    RBFGaussianProcessSurrogate surrogate;
    EXPECT_THROW(surrogate.fit({{0.0}, {1.0, 2.0}}, {1.0, 0.0}), std::invalid_argument);
}

TEST(RBFGaussianProcessSurrogate, RejectsInvalidHyperparametersAndQueryDimension) {
    EXPECT_THROW(RBFGaussianProcessSurrogate(0.0), std::invalid_argument);
    RBFGaussianProcessSurrogate surrogate;
    surrogate.fit({{0.0}}, {1.0});
    EXPECT_THROW(surrogate.acquisition({0.0, 1.0}, 1.0), std::invalid_argument);
}

TEST(CMAES, ConvergesOnShiftedSphere) {
    SphereFunction f({3.0, -2.0, 1.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    CMAESOptions options;
    options.population_size = 16;
    options.initial_step_size = 2.0;
    options.max_generations = 300;
    options.seed = 7;
    const double value = CMAES(options).optimize(f, x);
    EXPECT_LT(value, 1e-8);
    EXPECT_NEAR(x[0], 3.0, 1e-3);
    EXPECT_NEAR(x[1], -2.0, 1e-3);
    EXPECT_NEAR(x[2], 1.0, 1e-3);
}

TEST(CMAES, RejectsPopulationSmallerThanTwo) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    CMAESOptions options;
    options.population_size = 1;
    EXPECT_THROW(CMAES(options).optimize(f, x), std::invalid_argument);
}

// ---- PSO ----

TEST(PSO, GlobalTopologyConvergesOnSphere) {
    SphereFunction f({3.0, -2.0});
    std::vector<double> x{0.0, 0.0};
    PSOOptions options;
    options.population_size = 30;
    options.max_iterations = 300;
    const PSO optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0}, {10.0, 10.0});
    EXPECT_LT(value, 1e-4);
    EXPECT_NEAR(x[0], 3.0, 0.05);
    EXPECT_NEAR(x[1], -2.0, 0.05);
}

TEST(PSO, RingTopologyWithLinearDecayGetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    PSOOptions options;
    options.population_size = 60;
    options.max_iterations = 800;
    options.topology = "ring";
    options.inertia_strategy = "linear_decay";
    const PSO optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 1.0);
}

TEST(PSO, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const PSO optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(PSO, RejectsUnknownTopology) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    PSOOptions options;
    options.topology = "bogus";
    const PSO optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0, -1.0}, {1.0, 1.0}), std::invalid_argument);
}

// ---- Differential Evolution ----

class DifferentialEvolutionMutationStrategy : public ::testing::TestWithParam<std::string> {};

TEST_P(DifferentialEvolutionMutationStrategy, ConvergesOnSphere) {
    SphereFunction f({1.5, 4.0, -3.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    DEOptions options;
    options.population_size = 40;
    options.max_generations = 400;
    options.mutation_strategy = GetParam();
    const DifferentialEvolution optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_LT(value, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(AllMutationStrategies, DifferentialEvolutionMutationStrategy,
                         ::testing::Values("rand1", "best1", "current_to_best1", "rand2"));

TEST(DifferentialEvolution, ExponentialCrossoverConvergesOnSphere) {
    SphereFunction f({2.0, 2.0});
    std::vector<double> x{0.0, 0.0};
    DEOptions options;
    options.crossover_strategy = "exponential";
    options.max_generations = 400;
    const DifferentialEvolution optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0}, {10.0, 10.0});
    EXPECT_LT(value, 1e-3);
}

TEST(DifferentialEvolution, RejectsPopulationTooSmallForMutationStrategy) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    DEOptions options;
    options.mutation_strategy = "rand2";
    options.population_size = 4; // rand2 needs 5 distinct helpers + the target = 6
    const DifferentialEvolution optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(DifferentialEvolution, RejectsUnknownMutationStrategy) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    DEOptions options;
    options.mutation_strategy = "bogus";
    const DifferentialEvolution optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

// ---- Genetic Algorithm ----

class GeneticAlgorithmSelectionStrategy : public ::testing::TestWithParam<std::string> {};

TEST_P(GeneticAlgorithmSelectionStrategy, ConvergesOnSphere) {
    SphereFunction f({-1.0, 2.5});
    std::vector<double> x{0.0, 0.0};
    GAOptions options;
    options.population_size = 60;
    options.max_generations = 300;
    options.selection_strategy = GetParam();
    const GeneticAlgorithm optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0}, {10.0, 10.0});
    EXPECT_LT(value, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(AllSelectionStrategies, GeneticAlgorithmSelectionStrategy,
                         ::testing::Values("tournament", "roulette", "rank"));

class GeneticAlgorithmCrossoverStrategy : public ::testing::TestWithParam<std::string> {};

TEST_P(GeneticAlgorithmCrossoverStrategy, ConvergesOnSphere) {
    SphereFunction f({3.0, -3.0});
    std::vector<double> x{0.0, 0.0};
    GAOptions options;
    options.population_size = 60;
    options.max_generations = 300;
    options.crossover_strategy = GetParam();
    const GeneticAlgorithm optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0}, {10.0, 10.0});
    EXPECT_LT(value, 1e-2);
}

INSTANTIATE_TEST_SUITE_P(AllCrossoverStrategies, GeneticAlgorithmCrossoverStrategy,
                         ::testing::Values("single_point", "uniform", "blend"));

TEST(GeneticAlgorithm, WorksWithElitismDisabled) {
    SphereFunction f({1.0, 1.0});
    std::vector<double> x{0.0, 0.0};
    GAOptions options;
    options.elitism = false;
    options.max_generations = 500;
    const GeneticAlgorithm optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0}, {10.0, 10.0});
    EXPECT_LT(value, 0.5);
}

TEST(GeneticAlgorithm, RejectsUnknownSelectionStrategy) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    GAOptions options;
    options.selection_strategy = "bogus";
    const GeneticAlgorithm optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(GeneticAlgorithm, RejectsUnknownCrossoverStrategy) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    GAOptions options;
    options.crossover_strategy = "bogus";
    const GeneticAlgorithm optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}
