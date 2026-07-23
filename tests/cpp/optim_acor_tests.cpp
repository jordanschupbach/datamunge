#include <gtest/gtest.h>

#include <datamunge/optim/acor.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ACOR;
using datamunge::optim::ACOROptions;
using datamunge::optim::ArbitraryFunction;

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

TEST(ACOR, ConvergesOnShiftedSphere) {
    SphereFunction f({3.0, -2.0});
    std::vector<double> x{0.0, 0.0};
    ACOROptions options;
    options.archive_size = 40;
    options.samples_per_iteration = 30;
    options.max_iterations = 300;
    options.seed = 7;
    const ACOR optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0}, {10.0, 10.0});
    EXPECT_LT(value, 1e-4);
    EXPECT_NEAR(x[0], 3.0, 0.05);
    EXPECT_NEAR(x[1], -2.0, 0.05);
}

TEST(ACOR, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    ACOROptions options;
    options.archive_size = 40;
    options.samples_per_iteration = 30;
    options.max_iterations = 500;
    options.seed = 7;
    const ACOR optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 1.0);
}

TEST(ACOR, IsDeterministicForFixedSeed) {
    SphereFunction f1({1.0, 1.0});
    SphereFunction f2({1.0, 1.0});
    std::vector<double> x1{0.0, 0.0};
    std::vector<double> x2{0.0, 0.0};
    ACOROptions options;
    options.archive_size = 20;
    options.samples_per_iteration = 15;
    options.max_iterations = 50;
    options.seed = 123;
    const double v1 = ACOR(options).optimize(f1, x1, {-5.0, -5.0}, {5.0, 5.0});
    const double v2 = ACOR(options).optimize(f2, x2, {-5.0, -5.0}, {5.0, 5.0});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(x1, x2);
}

TEST(ACOR, RejectsArchiveSizeTooSmall) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    ACOROptions options;
    options.archive_size = 1;
    const ACOR optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(ACOR, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const ACOR optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(ACOR, RejectsNonPositiveLocality) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    ACOROptions options;
    options.locality = 0.0;
    const ACOR optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(ACOR, RejectsNonPositiveConvergenceSpeed) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    ACOROptions options;
    options.convergence_speed = -0.1;
    const ACOR optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}

TEST(ACOR, RejectsZeroSamplesPerIteration) {
    SphereFunction f({0.0});
    std::vector<double> x{0.0};
    ACOROptions options;
    options.samples_per_iteration = 0;
    const ACOR optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0}), std::invalid_argument);
}
