#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::GreyWolfOptimizer;
using datamunge::optim::GreyWolfOptimizerOptions;

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

// Note: GWO unconditionally replaces every wolf's position each iteration (no greedy
// accept/reject, faithful to Mirjalili et al. 2014), so the best-ever value can plateau for
// stretches even while later iterations still find real improvement. The generic "20
// consecutive non-improving iterations" stagnation stop (shared with every other population
// method here) can therefore trigger well before the run has actually converged when
// options.tolerance is very tight, unlike the greedy methods (PSO/DE/GA) where the same
// stopping rule reliably fires only once truly converged. Setting tolerance to 0 (never
// satisfied, since the comparison is a strict '<') disables the early stop for this test so
// the run completes its full iteration budget and reaches deep convergence.

TEST(GreyWolfOptimizer, ConvergesOnShiftedSphere) {
    SphereFunction f({3.0, -2.0, 1.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    GreyWolfOptimizerOptions options;
    options.population_size = 30;
    options.max_iterations = 300;
    options.tolerance = 0.0;
    options.seed = 7;
    const GreyWolfOptimizer optimizer(options);
    const double value = optimizer.optimize(f, x, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_LT(value, 1e-3);
    EXPECT_NEAR(x[0], 3.0, 0.05);
    EXPECT_NEAR(x[1], -2.0, 0.05);
    EXPECT_NEAR(x[2], 1.0, 0.05);
}

TEST(GreyWolfOptimizer, GetsCloseOnRastrigin) {
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    GreyWolfOptimizerOptions options;
    options.population_size = 60;
    options.max_iterations = 800;
    options.tolerance = 0.0;
    options.seed = 7;
    const GreyWolfOptimizer optimizer(options);
    const double value = optimizer.optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 1.0);
}

TEST(GreyWolfOptimizer, RejectsPopulationSmallerThanFour) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    GreyWolfOptimizerOptions options;
    options.population_size = 3;
    const GreyWolfOptimizer optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0, -1.0}, {1.0, 1.0}), std::invalid_argument);
}

TEST(GreyWolfOptimizer, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{0.0, 0.0};
    const GreyWolfOptimizer optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}
