#include <gtest/gtest.h>
#include <datamunge/optim/optim.hpp>
#include <datamunge/optim/artificial_bee_colony.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::ArtificialBeeColony;
using datamunge::optim::ArtificialBeeColonyOptions;

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

TEST(ArtificialBeeColony, ConvergesOnShiftedSphere) {
    SphereFunction f({1.5, 4.0, -3.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    ArtificialBeeColonyOptions options;
    options.population_size = 40;
    options.max_generations = 500;
    options.seed = 42;
    const ArtificialBeeColony optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_LT(value, 1e-3);
    EXPECT_NEAR(x[0], 1.5, 0.05);
    EXPECT_NEAR(x[1], 4.0, 0.05);
    EXPECT_NEAR(x[2], -3.0, 0.05);
}

TEST(ArtificialBeeColony, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{0.5, -0.5};
    ArtificialBeeColonyOptions options;
    options.population_size = 40;
    options.max_generations = 500;
    options.seed = 42;
    const ArtificialBeeColony optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 1.0);
}

TEST(ArtificialBeeColony, IsDeterministicGivenSameSeed) {
    SphereFunction f1({1.5, 4.0, -3.0});
    SphereFunction f2({1.5, 4.0, -3.0});
    std::vector<double> x1{0.0, 0.0, 0.0};
    std::vector<double> x2{0.0, 0.0, 0.0};
    ArtificialBeeColonyOptions options;
    options.population_size = 20;
    options.max_generations = 100;
    options.seed = 123;
    const double v1 = ArtificialBeeColony(options).optimize(f1, x1, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    const double v2 = ArtificialBeeColony(options).optimize(f2, x2, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(x1, x2);
}

TEST(ArtificialBeeColony, SmallAbandonmentLimitConvergesSlowerThanLarge) {
    // A very small abandonment_limit causes food sources to be scouted away (replaced by fresh
    // random points) almost immediately, discarding partially-optimized progress; a very large
    // limit effectively disables the scout phase. With few generations, the large-limit run
    // should reach a noticeably better (lower) objective value.
    SphereFunction f_small({1.5, 4.0, -3.0});
    SphereFunction f_large({1.5, 4.0, -3.0});
    std::vector<double> x_small{0.0, 0.0, 0.0};
    std::vector<double> x_large{0.0, 0.0, 0.0};

    ArtificialBeeColonyOptions small_opts;
    small_opts.population_size = 20;
    small_opts.max_generations = 60;
    small_opts.abandonment_limit = 1;
    small_opts.seed = 7;
    const double value_small =
        ArtificialBeeColony(small_opts).optimize(f_small, x_small, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});

    ArtificialBeeColonyOptions large_opts;
    large_opts.population_size = 20;
    large_opts.max_generations = 60;
    large_opts.abandonment_limit = 1000;
    large_opts.seed = 7;
    const double value_large =
        ArtificialBeeColony(large_opts).optimize(f_large, x_large, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});

    EXPECT_LT(value_large, value_small);
}

TEST(ArtificialBeeColony, RejectsPopulationSmallerThanTwo) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    ArtificialBeeColonyOptions options;
    options.population_size = 1;
    const ArtificialBeeColony optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(ArtificialBeeColony, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const ArtificialBeeColony optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}
