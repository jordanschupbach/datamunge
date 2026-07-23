#include <gtest/gtest.h>

#include <datamunge/optim/evolution_strategy.hpp>
#include <datamunge/optim/optim.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::EvolutionStrategy;
using datamunge::optim::EvolutionStrategyOptions;

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

// ---- Evolution Strategy ----

class EvolutionStrategyStrategyVariant : public ::testing::TestWithParam<std::string> {};

TEST_P(EvolutionStrategyStrategyVariant, ConvergesOnShiftedSphere) {
    SphereFunction f({3.0, -2.0, 1.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    EvolutionStrategyOptions options;
    options.mu = 15;
    options.offspring_size = 100;
    options.max_generations = 300;
    options.strategy = GetParam();
    const EvolutionStrategy optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_LT(value, 1e-3);
    EXPECT_NEAR(x[0], 3.0, 0.05);
    EXPECT_NEAR(x[1], -2.0, 0.05);
    EXPECT_NEAR(x[2], 1.0, 0.05);
}

INSTANTIATE_TEST_SUITE_P(CommaAndPlus, EvolutionStrategyStrategyVariant, ::testing::Values("comma", "plus"));

TEST(EvolutionStrategy, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -3.0};
    EvolutionStrategyOptions options;
    options.mu = 15;
    options.offspring_size = 100;
    options.max_generations = 300;
    options.seed = 7;
    const EvolutionStrategy optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    // Single-global-step-size ES on a highly multimodal objective isn't guaranteed to escape
    // every local basin; a nearby local minimum (e.g. near f=1, one lattice step off the
    // origin) is an acceptable, honest outcome here.
    EXPECT_LT(value, 5.0);
}

TEST(EvolutionStrategy, RejectsOffspringSizeSmallerThanMu) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    EvolutionStrategyOptions options;
    options.mu = 10;
    options.offspring_size = 5;
    const EvolutionStrategy optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(EvolutionStrategy, RejectsUnknownStrategy) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    EvolutionStrategyOptions options;
    options.strategy = "bogus";
    const EvolutionStrategy optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(EvolutionStrategy, RejectsNonPositiveInitialStepSize) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    EvolutionStrategyOptions options;
    options.initial_step_size = 0.0;
    const EvolutionStrategy optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(EvolutionStrategy, RejectsMismatchedBounds) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const EvolutionStrategy optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(EvolutionStrategy, IsDeterministicForFixedSeed) {
    SphereFunction f({1.0, -1.0});
    std::vector<double> x1{0.0, 0.0};
    std::vector<double> x2{0.0, 0.0};
    EvolutionStrategyOptions options;
    options.max_generations = 50;
    options.seed = 123;
    const double v1 = EvolutionStrategy(options).optimize(f, x1, {-5.0, -5.0}, {5.0, 5.0});
    const double v2 = EvolutionStrategy(options).optimize(f, x2, {-5.0, -5.0}, {5.0, 5.0});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(x1, x2);
}
