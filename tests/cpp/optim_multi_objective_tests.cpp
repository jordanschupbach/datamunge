#include <gtest/gtest.h>

#include <datamunge/optim/moead.hpp>
#include <datamunge/optim/multi_objective.hpp>
#include <datamunge/optim/nsga2.hpp>
#include <datamunge/optim/spea2.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

using datamunge::optim::dominates;
using datamunge::optim::MOEAD;
using datamunge::optim::MOEADOptions;
using datamunge::optim::MultiObjectiveFunction;
using datamunge::optim::non_dominated_front;
using datamunge::optim::NSGA2;
using datamunge::optim::NSGA2Options;
using datamunge::optim::ParetoPoint;
using datamunge::optim::SPEA2;
using datamunge::optim::SPEA2Options;

namespace {

// Schaffer N.1: one variable, f1 = x^2, f2 = (x-2)^2. The Pareto set is exactly x in [0, 2].
class Schaffer : public MultiObjectiveFunction {
public:
    std::vector<double> evaluate(const std::vector<double>& x) override {
        const double v = x[0];
        return {v * v, (v - 2.0) * (v - 2.0)};
    }
};

// ZDT1: f1 = x1; g = 1 + 9*sum(x_2..x_n)/(n-1); f2 = g*(1 - sqrt(f1/g)). The (convex) Pareto
// front is g = 1, i.e. f2 = 1 - sqrt(f1) for f1 in [0, 1].
class ZDT1 : public MultiObjectiveFunction {
public:
    explicit ZDT1(std::size_t n) : n_(n) {}
    std::vector<double> evaluate(const std::vector<double>& x) override {
        const double f1 = x[0];
        double s = 0.0;
        for (std::size_t i = 1; i < n_; ++i) s += x[i];
        const double g = 1.0 + 9.0 * s / static_cast<double>(n_ - 1);
        const double f2 = g * (1.0 - std::sqrt(f1 / g));
        return {f1, f2};
    }

private:
    std::size_t n_;
};

using Solver = std::function<std::vector<ParetoPoint>(MultiObjectiveFunction&, const std::vector<double>&,
                                                      const std::vector<double>&)>;

// A returned front must be internally non-dominated (no point dominates another).
void expect_internally_non_dominated(const std::vector<ParetoPoint>& front) {
    for (std::size_t i = 0; i < front.size(); ++i)
        for (std::size_t j = 0; j < front.size(); ++j)
            if (i != j) EXPECT_FALSE(dominates(front[i].objectives, front[j].objectives));
}

} // namespace

// ---- Dominance primitives ----

TEST(MultiObjectiveDominance, MinimizationSemantics) {
    EXPECT_TRUE(dominates({1.0, 1.0}, {2.0, 2.0}));   // better in both
    EXPECT_TRUE(dominates({1.0, 2.0}, {1.0, 3.0}));   // equal in one, better in the other
    EXPECT_FALSE(dominates({1.0, 3.0}, {2.0, 2.0}));  // a trade-off: neither dominates
    EXPECT_FALSE(dominates({2.0, 2.0}, {2.0, 2.0}));  // identical: no strict improvement
}

TEST(MultiObjectiveDominance, FrontExtraction) {
    std::vector<ParetoPoint> pts = {
        {{0.0}, {1.0, 4.0}}, {{0.0}, {2.0, 2.0}}, {{0.0}, {4.0, 1.0}},
        {{0.0}, {3.0, 3.0}},  // dominated by {2,2}
        {{0.0}, {2.5, 5.0}},  // dominated by {2,2}
    };
    const auto front = non_dominated_front(pts);
    EXPECT_EQ(front.size(), 3u);
    expect_internally_non_dominated(front);
}

// ---- The three algorithms on shared benchmarks ----

class MultiObjectiveAlgorithm : public ::testing::TestWithParam<Solver> {};

TEST_P(MultiObjectiveAlgorithm, SchafferRecoversTheKnownParetoSet) {
    Schaffer f;
    const auto front = GetParam()(f, {-5.0}, {5.0});
    ASSERT_GE(front.size(), 5u);
    expect_internally_non_dominated(front);

    double min_f1 = 1e300, max_f1 = -1e300;
    for (const auto& p : front) {
        // Pareto-optimal decision variables lie in [0, 2] (allow a little slack).
        EXPECT_GE(p.coordinates[0], -0.2);
        EXPECT_LE(p.coordinates[0], 2.2);
        // On the front, x = sqrt(f1) and f2 = (x - 2)^2 -- check the point sits near it.
        const double f2_true = (std::sqrt(p.objectives[0]) - 2.0) * (std::sqrt(p.objectives[0]) - 2.0);
        EXPECT_NEAR(p.objectives[1], f2_true, 0.2);
        min_f1 = std::min(min_f1, p.objectives[0]);
        max_f1 = std::max(max_f1, p.objectives[0]);
    }
    EXPECT_GT(max_f1 - min_f1, 1.5); // the front is spread out, not collapsed to one point
}

TEST_P(MultiObjectiveAlgorithm, ZDT1ConvergesToConvexFront) {
    ZDT1 f(10);
    const std::vector<double> lo(10, 0.0), hi(10, 1.0);
    const auto front = GetParam()(f, lo, hi);
    ASSERT_GE(front.size(), 5u);
    expect_internally_non_dominated(front);

    // Distance "above" the true front f2 = 1 - sqrt(f1): should be small (g driven toward 1).
    std::vector<double> gap;
    double min_f1 = 1e300, max_f1 = -1e300;
    for (const auto& p : front) {
        const double f2_true = 1.0 - std::sqrt(std::max(0.0, std::min(1.0, p.objectives[0])));
        gap.push_back(p.objectives[1] - f2_true);
        min_f1 = std::min(min_f1, p.objectives[0]);
        max_f1 = std::max(max_f1, p.objectives[0]);
    }
    std::sort(gap.begin(), gap.end());
    const double median_gap = gap[gap.size() / 2];
    EXPECT_LT(median_gap, 0.15);   // converged close to the front
    EXPECT_LT(min_f1, 0.2);        // covers the low-f1 end
    EXPECT_GT(max_f1, 0.8);        // covers the high-f1 end
}

INSTANTIATE_TEST_SUITE_P(
    Algorithms, MultiObjectiveAlgorithm,
    ::testing::Values(
        Solver{[](MultiObjectiveFunction& f, const std::vector<double>& lo, const std::vector<double>& hi) {
            return NSGA2(NSGA2Options{60, 200, 0.9, 15.0, -1.0, 20.0, 42}).optimize(f, lo, hi);
        }},
        Solver{[](MultiObjectiveFunction& f, const std::vector<double>& lo, const std::vector<double>& hi) {
            return SPEA2(SPEA2Options{60, 60, 200, 0.9, 15.0, -1.0, 20.0, 42}).optimize(f, lo, hi);
        }},
        Solver{[](MultiObjectiveFunction& f, const std::vector<double>& lo, const std::vector<double>& hi) {
            return MOEAD(MOEADOptions{60, 200, 20, "tchebycheff", 0.9, 20.0, -1.0, 20.0, 42}).optimize(f, lo, hi);
        }}));

// ---- Input validation ----

TEST(MultiObjectiveValidation, RejectsBadBounds) {
    Schaffer f;
    EXPECT_THROW(NSGA2().optimize(f, {}, {}), std::invalid_argument);
    EXPECT_THROW(NSGA2().optimize(f, {0.0, 0.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(SPEA2().optimize(f, {1.0}, {1.0}), std::invalid_argument);          // lower !< upper
    EXPECT_THROW(MOEAD(MOEADOptions{}).optimize(f, {2.0}, {1.0}), std::invalid_argument);
    MOEADOptions bad;
    bad.decomposition = "nonsense";
    EXPECT_THROW(MOEAD(bad).optimize(f, {0.0}, {1.0}), std::invalid_argument);
}
