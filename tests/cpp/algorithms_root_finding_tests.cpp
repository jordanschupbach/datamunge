#include <gtest/gtest.h>

#include <datamunge/algorithms/root_finding.hpp>

#include <cmath>
#include <functional>
#include <vector>

using datamunge::algorithms::bisection;
using datamunge::algorithms::newton_raphson;
using datamunge::algorithms::ridders;
using datamunge::algorithms::RootResult;
using datamunge::algorithms::secant;

namespace {

struct Problem {
    std::function<double(double)> f;
    std::function<double(double)> df;
    double                        a, b;   // bracket
    double                        x0, x1; // starting points for open methods
    double                        root;   // known root
    const char*                   name;
};

std::vector<Problem> problems() {
    return {
        {[](double x) { return x * x - 2.0; }, [](double x) { return 2.0 * x; }, 1.0, 2.0, 2.0, 1.0,
         1.4142135623730951, "x^2 - 2"},
        {[](double x) { return std::cos(x) - x; }, [](double x) { return -std::sin(x) - 1.0; }, 0.0, 1.0, 0.0, 1.0,
         0.7390851332151607, "cos(x) - x"},
        {[](double x) { return x * x * x - x - 2.0; }, [](double x) { return 3.0 * x * x - 1.0; }, 1.0, 2.0, 2.0, 1.0,
         1.5213797068045676, "x^3 - x - 2"},
        {[](double x) { return x - std::exp(-x); }, [](double x) { return 1.0 + std::exp(-x); }, 0.0, 1.0, 0.0, 1.0,
         0.5671432904097838, "x - e^{-x}"},
    };
}

} // namespace

TEST(RootFinding, AllMethodsConvergeToKnownRoots) {
    for (const Problem& p : problems()) {
        const RootResult bi = bisection(p.f, p.a, p.b);
        const RootResult nr = newton_raphson(p.f, p.df, p.x0);
        const RootResult se = secant(p.f, p.x0, p.x1);
        const RootResult ri = ridders(p.f, p.a, p.b);
        for (const auto& [r, method] : {std::pair{bi, "bisection"}, std::pair{nr, "newton"},
                                        std::pair{se, "secant"}, std::pair{ri, "ridders"}}) {
            EXPECT_TRUE(r.converged) << p.name << " / " << method;
            EXPECT_NEAR(r.root, p.root, 1e-6) << p.name << " / " << method;
            EXPECT_LT(r.residual, 1e-8) << p.name << " / " << method;
        }
    }
}

TEST(RootFinding, BisectionRejectsNoSignChange) {
    const RootResult r = bisection([](double x) { return x * x + 1.0; }, 0.0, 1.0); // always positive
    EXPECT_FALSE(r.converged);
    const RootResult r2 = ridders([](double x) { return x * x + 1.0; }, 0.0, 1.0);
    EXPECT_FALSE(r2.converged);
}

TEST(RootFinding, EndpointRootsDetected) {
    EXPECT_TRUE(bisection([](double x) { return x; }, 0.0, 1.0).converged);
    EXPECT_DOUBLE_EQ(bisection([](double x) { return x - 1.0; }, 0.0, 1.0).root, 1.0);
}

TEST(RootFinding, NewtonBeatsBisectionInIterations) {
    // Quadratic convergence: Newton reaches the same tolerance in far fewer iterations.
    const RootResult bi = bisection([](double x) { return x * x - 2.0; }, 1.0, 2.0, 1e-12);
    const RootResult nr = newton_raphson([](double x) { return x * x - 2.0; }, [](double x) { return 2.0 * x; }, 2.0, 1e-12);
    EXPECT_TRUE(bi.converged);
    EXPECT_TRUE(nr.converged);
    EXPECT_LT(nr.iterations, bi.iterations);
}

TEST(RootFinding, SecantConvergesWithoutDerivative) {
    const RootResult se = secant([](double x) { return std::exp(x) - 2.0; }, 0.0, 1.0); // root ln 2
    EXPECT_TRUE(se.converged);
    EXPECT_NEAR(se.root, std::log(2.0), 1e-9);
}
