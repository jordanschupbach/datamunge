#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>
#include <datamunge/optim/cross_entropy_method.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::CrossEntropyMethod;
using datamunge::optim::CrossEntropyMethodOptions;

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

TEST(CrossEntropyMethod, ConvergesOnShiftedSphere) {
    SphereFunction f({3.0, -2.0, 1.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    CrossEntropyMethodOptions options;
    options.population_size = 60;
    options.max_iterations = 300;
    // Distribution needs to cover the initial-to-optimum distance; mirrors CMAES's own sphere
    // test overriding its default initial_step_size for the same reason.
    options.initial_std_dev = 2.0;
    options.seed = 7;
    const double value = CrossEntropyMethod(options).optimize(f, x, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_LT(value, 1e-8);
    EXPECT_NEAR(x[0], 3.0, 1e-3);
    EXPECT_NEAR(x[1], -2.0, 1e-3);
    EXPECT_NEAR(x[2], 1.0, 1e-3);
}

TEST(CrossEntropyMethod, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    CrossEntropyMethodOptions options;
    options.population_size = 100;
    options.max_iterations = 500;
    options.initial_std_dev = 3.0;
    const CrossEntropyMethod optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 1.0);
}

TEST(CrossEntropyMethod, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const CrossEntropyMethod optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(CrossEntropyMethod, RejectsPopulationSizeBelowFour) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    CrossEntropyMethodOptions options;
    options.population_size = 3;
    const CrossEntropyMethod optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(CrossEntropyMethod, RejectsEliteRatioOutOfRange) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    {
        CrossEntropyMethodOptions options;
        options.elite_ratio = 0.0;
        const CrossEntropyMethod optimizer(options);
        EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
    }
    {
        CrossEntropyMethodOptions options;
        options.elite_ratio = 1.5;
        const CrossEntropyMethod optimizer(options);
        EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
    }
}

TEST(CrossEntropyMethod, RejectsSmoothingOutOfRange) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    {
        CrossEntropyMethodOptions options;
        options.smoothing = 0.0;
        const CrossEntropyMethod optimizer(options);
        EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
    }
    {
        CrossEntropyMethodOptions options;
        options.smoothing = 1.5;
        const CrossEntropyMethod optimizer(options);
        EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
    }
}

TEST(CrossEntropyMethod, RejectsNonPositiveInitialStdDev) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    CrossEntropyMethodOptions options;
    options.initial_std_dev = 0.0;
    const CrossEntropyMethod optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}
