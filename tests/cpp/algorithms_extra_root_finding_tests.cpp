#include <gtest/gtest.h>

#include <datamunge/algorithms/root_finding.hpp>

#include <cmath>
#include <functional>
#include <vector>

using datamunge::algorithms::false_position;
using datamunge::algorithms::halley;
using datamunge::algorithms::itp;
using datamunge::algorithms::muller;
using datamunge::algorithms::RootResult;

namespace {

struct Problem {
    std::function<double(double)> f, df, d2f;
    double                        a, b, x0, root;
    const char*                   name;
};

std::vector<Problem> problems() {
    return {
        {[](double x) { return x * x - 2.0; }, [](double x) { return 2.0 * x; }, [](double) { return 2.0; },
         1.0, 2.0, 2.0, 1.4142135623730951, "x^2 - 2"},
        {[](double x) { return std::cos(x) - x; }, [](double x) { return -std::sin(x) - 1.0; }, [](double x) { return -std::cos(x); },
         0.0, 1.0, 0.5, 0.7390851332151607, "cos(x) - x"},
        {[](double x) { return x * x * x - x - 2.0; }, [](double x) { return 3.0 * x * x - 1.0; }, [](double x) { return 6.0 * x; },
         1.0, 2.0, 1.5, 1.5213797068045676, "x^3 - x - 2"},
        {[](double x) { return x - std::exp(-x); }, [](double x) { return 1.0 + std::exp(-x); }, [](double x) { return -std::exp(-x); },
         0.0, 1.0, 0.5, 0.5671432904097838, "x - e^{-x}"},
    };
}

} // namespace

TEST(ExtraRootFinding, AllMethodsConvergeToKnownRoots) {
    for (const Problem& p : problems()) {
        const RootResult fp = false_position(p.f, p.a, p.b);
        const RootResult ha = halley(p.f, p.df, p.d2f, p.x0);
        const RootResult mu = muller(p.f, p.a, 0.5 * (p.a + p.b), p.b);
        const RootResult it = itp(p.f, p.a, p.b);
        for (const auto& [r, method] : {std::pair{fp, "false_position"}, std::pair{ha, "halley"},
                                        std::pair{mu, "muller"}, std::pair{it, "itp"}}) {
            EXPECT_TRUE(r.converged) << p.name << " / " << method;
            EXPECT_NEAR(r.root, p.root, 1e-6) << p.name << " / " << method;
            EXPECT_LT(r.residual, 1e-7) << p.name << " / " << method;
        }
    }
}

TEST(ExtraRootFinding, BracketingMethodsRejectNoSignChange) {
    auto pos = [](double x) { return x * x + 1.0; };
    EXPECT_FALSE(false_position(pos, 0.0, 1.0).converged);
    EXPECT_FALSE(itp(pos, 0.0, 1.0).converged);
}

TEST(ExtraRootFinding, HalleyBeatsAndMullerNeedsNoDerivative) {
    // Halley is cubic: reaches tight tolerance in very few iterations.
    const RootResult ha = halley([](double x) { return x * x - 2.0; }, [](double x) { return 2.0 * x; },
                                 [](double) { return 2.0; }, 2.0, 1e-12);
    EXPECT_TRUE(ha.converged);
    EXPECT_LE(ha.iterations, 5);

    // Muller finds ln 2 as a root of e^x - 2 from three real points, no derivative used.
    const RootResult mu = muller([](double x) { return std::exp(x) - 2.0; }, 0.0, 0.5, 1.0);
    EXPECT_TRUE(mu.converged);
    EXPECT_NEAR(mu.root, std::log(2.0), 1e-9);
}
