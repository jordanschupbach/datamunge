#include <gtest/gtest.h>

#include <datamunge/optim/optim.hpp>
#include <datamunge/optim/estimation_of_distribution.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::optim::ArbitraryFunction;
using datamunge::optim::EstimationOfDistribution;
using datamunge::optim::EstimationOfDistributionOptions;

namespace {

// Shifted sphere: global minimum f=0 at x=target.
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

TEST(EstimationOfDistribution, ConvergesOnShiftedSphere) {
    // initial_std_dev is set comparably to the initial distance-to-optimum (~3.6): EMNA_global
    // re-estimates its covariance from scratch each generation under truncation selection with
    // no evolution-path-based re-expansion (unlike CMA-ES), so the mean's total achievable
    // displacement is bounded roughly by the initial standard deviation -- a well-known
    // characteristic of this exact algorithm, not a bug. See CMAES's own sphere test, which
    // similarly overrides its default (small) step size for the same reason.
    SphereFunction f({3.0, -2.0, 1.0});
    std::vector<double> x{0.0, 0.0, 0.0};
    EstimationOfDistributionOptions options;
    options.population_size = 40;
    options.initial_std_dev = 2.0;
    options.max_generations = 300;
    options.seed = 7;
    const double value = EstimationOfDistribution(options).optimize(f, x, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_LT(value, 1e-8);
    EXPECT_NEAR(x[0], 3.0, 1e-3);
    EXPECT_NEAR(x[1], -2.0, 1e-3);
    EXPECT_NEAR(x[2], 1.0, 1e-3);
}

TEST(EstimationOfDistribution, DeterministicAcrossRepeatedRuns) {
    SphereFunction f({3.0, -2.0, 1.0});
    std::vector<double> x1{0.0, 0.0, 0.0};
    std::vector<double> x2{0.0, 0.0, 0.0};
    EstimationOfDistributionOptions options;
    options.seed = 55;
    const double v1 = EstimationOfDistribution(options).optimize(f, x1, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    const double v2 = EstimationOfDistribution(options).optimize(f, x2, {-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(x1, x2);
}

TEST(EstimationOfDistribution, GetsCloseOnRastrigin) {
    // A larger population and an initial_std_dev spanning the search box makes premature
    // convergence onto a wrong local basin much less likely (though, being a genuinely
    // multimodal landscape, EDA methods can still occasionally do so on an unlucky seed --
    // this is an honestly-reportable characteristic of the algorithm, not asserted away here).
    RastriginFunction f;
    std::vector<double> x{2.0, -2.0};
    EstimationOfDistributionOptions options;
    options.population_size = 100;
    options.initial_std_dev = 3.0;
    options.max_generations = 500;
    options.seed = 1;
    const double value = EstimationOfDistribution(options).optimize(f, x, {-5.12, -5.12}, {5.12, 5.12});
    EXPECT_LT(value, 1e-6);
}

TEST(EstimationOfDistribution, RejectsPopulationSmallerThanFour) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    EstimationOfDistributionOptions options;
    options.population_size = 3;
    EstimationOfDistribution optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-10.0}, {10.0}), std::invalid_argument);
}

TEST(EstimationOfDistribution, RejectsSelectionRatioOutOfRange) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    {
        EstimationOfDistributionOptions options;
        options.selection_ratio = 0.0;
        EstimationOfDistribution optimizer(options);
        EXPECT_THROW(optimizer.optimize(f, x, {-10.0}, {10.0}), std::invalid_argument);
    }
    {
        EstimationOfDistributionOptions options;
        options.selection_ratio = 1.5;
        EstimationOfDistribution optimizer(options);
        EXPECT_THROW(optimizer.optimize(f, x, {-10.0}, {10.0}), std::invalid_argument);
    }
}

TEST(EstimationOfDistribution, RejectsNonPositiveInitialStdDev) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    EstimationOfDistributionOptions options;
    options.initial_std_dev = 0.0;
    EstimationOfDistribution optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-10.0}, {10.0}), std::invalid_argument);
}

TEST(EstimationOfDistribution, RejectsNegativeCovarianceRegularization) {
    SphereFunction f({0.0});
    std::vector<double> x{1.0};
    EstimationOfDistributionOptions options;
    options.covariance_regularization = -1e-6;
    EstimationOfDistribution optimizer(options);
    EXPECT_THROW(optimizer.optimize(f, x, {-10.0}, {10.0}), std::invalid_argument);
}

TEST(EstimationOfDistribution, RejectsMismatchedBoundSize) {
    SphereFunction f({0.0, 0.0});
    std::vector<double> x{1.0, 1.0};
    EstimationOfDistribution optimizer;
    EXPECT_THROW(optimizer.optimize(f, x, {-1.0}, {1.0, 1.0}), std::invalid_argument);
}
