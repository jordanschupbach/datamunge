#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::FireflyAlgorithm;
using datamunge::optim::FireflyAlgorithmOptions;

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

TEST(FireflyAlgorithm, ConvergesOnShiftedSphere) {
    SphereFunction f({1.5, -1.0, 0.5});
    std::vector<double> x{0.0, 0.0, 0.0};
    FireflyAlgorithmOptions options;
    options.population_size = 40;
    options.max_iterations = 500;
    // light_absorption (gamma) must be scaled to the domain: with the default gamma=1.0,
    // attraction between fireflies more than a few units apart is essentially zero (beta
    // decays as exp(-gamma*r^2)), a well-documented sensitivity of vanilla Firefly Algorithm.
    options.light_absorption = 0.1;
    options.seed = 7;
    const FireflyAlgorithm optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.0, -5.0, -5.0}, {5.0, 5.0, 5.0});
    EXPECT_LT(value, 1e-2);
    EXPECT_NEAR(x[0], 1.5, 0.05);
    EXPECT_NEAR(x[1], -1.0, 0.05);
    EXPECT_NEAR(x[2], 0.5, 0.05);
}

TEST(FireflyAlgorithm, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    FireflyAlgorithmOptions options;
    options.population_size = 60;
    options.max_iterations = 800;
    options.light_absorption = 0.1;
    options.seed = 7;
    const FireflyAlgorithm optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 1.0);
}

TEST(FireflyAlgorithm, RejectsPopulationSmallerThanTwo) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    FireflyAlgorithmOptions options;
    options.population_size = 1;
    EXPECT_THROW(FireflyAlgorithm(options).optimize(f, x, {-1.0, -1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(FireflyAlgorithm, RejectsNonPositiveAttractivenessAtZero) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    FireflyAlgorithmOptions options;
    options.attractiveness_at_zero = 0.0;
    EXPECT_THROW(FireflyAlgorithm(options).optimize(f, x, {-1.0, -1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(FireflyAlgorithm, RejectsNegativeLightAbsorption) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    FireflyAlgorithmOptions options;
    options.light_absorption = -0.1;
    EXPECT_THROW(FireflyAlgorithm(options).optimize(f, x, {-1.0, -1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(FireflyAlgorithm, RejectsNegativeRandomizationStep) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    FireflyAlgorithmOptions options;
    options.randomization_step = -0.1;
    EXPECT_THROW(FireflyAlgorithm(options).optimize(f, x, {-1.0, -1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(FireflyAlgorithm, RejectsRandomizationDecayOutOfRange) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    FireflyAlgorithmOptions zero_decay;
    zero_decay.randomization_decay = 0.0;
    EXPECT_THROW(FireflyAlgorithm(zero_decay).optimize(f, x, {-1.0, -1.0}, {1.0, 1.0}), std::invalid_argument);

    FireflyAlgorithmOptions too_big;
    too_big.randomization_decay = 1.5;
    EXPECT_THROW(FireflyAlgorithm(too_big).optimize(f, x, {-1.0, -1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(FireflyAlgorithm, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const FireflyAlgorithm optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}
