#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>
#include <datamunge/optim/whale_optimization.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::WhaleOptimization;
using datamunge::optim::WhaleOptimizationOptions;

namespace {

// Simple quadratic bowl, minimized at target_.
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

TEST(WhaleOptimization, ConvergesOnShiftedSphere) {
    SphereFunction f({3.0, -2.0, 1.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    WhaleOptimizationOptions options;
    options.population_size = 80;
    options.max_iterations = 500;
    options.seed = 7;
    const WhaleOptimization optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_LT(value, 1e-3);
    EXPECT_NEAR(x[0], 3.0, 0.05);
    EXPECT_NEAR(x[1], -2.0, 0.05);
    EXPECT_NEAR(x[2], 1.0, 0.05);
}

TEST(WhaleOptimization, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    WhaleOptimizationOptions options;
    options.population_size = 60;
    options.max_iterations = 800;
    options.seed = 42;
    const WhaleOptimization optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 1e-2);
}

TEST(WhaleOptimization, IsDeterministicGivenSameSeed) {
    SphereFunction f({3.0, -2.0});
    WhaleOptimizationOptions options;
    options.population_size = 20;
    options.max_iterations = 100;
    options.seed = 5;
    std::vector<double> x1{0.0, 0.0};
    std::vector<double> x2{0.0, 0.0};
    const double v1 = WhaleOptimization(options).optimize(f, x1, {-10.0, -10.0}, {10.0, 10.0});
    const double v2 = WhaleOptimization(options).optimize(f, x2, {-10.0, -10.0}, {10.0, 10.0});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(x1, x2);
}

TEST(WhaleOptimization, RejectsPopulationSmallerThanTwo) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    WhaleOptimizationOptions options;
    options.population_size = 1;
    const WhaleOptimization optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(WhaleOptimization, RejectsNonPositiveSpiralConstant) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    WhaleOptimizationOptions options;
    options.spiral_constant = 0.0;
    const WhaleOptimization optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(WhaleOptimization, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const WhaleOptimization optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}
