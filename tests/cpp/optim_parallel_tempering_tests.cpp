#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::ParallelTempering;
using datamunge::optim::ParallelTemperingOptions;

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

TEST(ParallelTempering, ConvergesOnShiftedSphere) {
    SphereFunction f({3.0, -2.0});
    std::vector<double> x{0.0, 0.0};
    ParallelTemperingOptions options;
    options.num_replicas = 10;
    options.initial_temperature = 10.0;
    options.final_temperature = 0.1;
    options.step_std_dev = 1.0;
    options.swap_interval = 10;
    options.max_sweeps = 1000;
    options.seed = 7;
    const ParallelTempering optimizer(options);
    const double value = optimizer.optimize(f, x);
    EXPECT_LT(value, 1.0);
}

TEST(ParallelTempering, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    ParallelTemperingOptions options;
    options.num_replicas = 10;
    options.initial_temperature = 10.0;
    options.final_temperature = 0.05;
    options.step_std_dev = 1.0;
    options.swap_interval = 10;
    options.max_sweeps = 2000;
    options.seed = 7;
    const ParallelTempering optimizer(options);
    const double value = optimizer.optimize(f, x);
    EXPECT_LT(value, 1.0);
}

TEST(ParallelTempering, DeterministicForFixedSeed) {
    SphereFunction f({1.0, 1.0});
    std::vector<double> x1{0.0, 0.0};
    std::vector<double> x2{0.0, 0.0};
    ParallelTemperingOptions options;
    options.max_sweeps = 200;
    options.seed = 123;
    const double v1 = ParallelTempering(options).optimize(f, x1);
    const double v2 = ParallelTempering(options).optimize(f, x2);
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(x1, x2);
}

TEST(ParallelTempering, RejectsTooFewReplicas) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    ParallelTemperingOptions options;
    options.num_replicas = 1;
    EXPECT_THROW(ParallelTempering(options).optimize(f, x), std::invalid_argument);
}

TEST(ParallelTempering, RejectsInitialTemperatureNotAboveFinal) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    ParallelTemperingOptions options;
    options.initial_temperature = 0.1;
    options.final_temperature = 0.1;
    EXPECT_THROW(ParallelTempering(options).optimize(f, x), std::invalid_argument);
}

TEST(ParallelTempering, RejectsNonPositiveStepStdDev) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    ParallelTemperingOptions options;
    options.step_std_dev = 0.0;
    EXPECT_THROW(ParallelTempering(options).optimize(f, x), std::invalid_argument);
}

TEST(ParallelTempering, RejectsZeroSwapInterval) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    ParallelTemperingOptions options;
    options.swap_interval = 0;
    EXPECT_THROW(ParallelTempering(options).optimize(f, x), std::invalid_argument);
}
