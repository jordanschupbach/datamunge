#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::HarmonySearch;
using datamunge::optim::HarmonySearchOptions;

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

TEST(HarmonySearch, ConvergesOnShiftedSphere) {
    SphereFunction f({3.0, -2.0, 1.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    HarmonySearchOptions options;
    options.population_size = 30;
    options.max_iterations = 10000;
    options.seed = 7;
    const HarmonySearch optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_LT(value, 1e-3);
    EXPECT_NEAR(x[0], 3.0, 0.1);
    EXPECT_NEAR(x[1], -2.0, 0.1);
    EXPECT_NEAR(x[2], 1.0, 0.1);
}

TEST(HarmonySearch, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    HarmonySearchOptions options;
    options.population_size = 30;
    options.max_iterations = 20000;
    options.seed = 7;
    const HarmonySearch optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 0.5);
}

TEST(HarmonySearch, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const HarmonySearch optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(HarmonySearch, RejectsMemoryConsiderationRateOutOfRange) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    HarmonySearchOptions options;
    options.memory_consideration_rate = 1.0;
    const HarmonySearch optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(HarmonySearch, RejectsPitchAdjustmentRateOutOfRange) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    HarmonySearchOptions options;
    options.pitch_adjustment_rate = -0.1;
    const HarmonySearch optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(HarmonySearch, RejectsNonPositiveBandwidthFraction) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    HarmonySearchOptions options;
    options.bandwidth_fraction = 0.0;
    const HarmonySearch optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(HarmonySearch, RejectsPopulationSmallerThanTwo) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    HarmonySearchOptions options;
    options.population_size = 1;
    const HarmonySearch optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}
