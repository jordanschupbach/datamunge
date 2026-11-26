#include <datamunge/stats/acceptance_sampling.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_acceptance_data.inc"
} // namespace

TEST(AcceptanceSampling, OperatingCharacteristicMatchesR) {
    const std::vector<AttributePlan> plans_oc{{50, 1}, {125, 3}, {13, 0}, {200, 7}};
    const std::vector<double> ps{0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3};
    const SamplingModel models[3] = {SamplingModel::binomial, SamplingModel::poisson,
                                     SamplingModel::hypergeometric};
    std::size_t k = 0;
    for (const auto& plan : plans_oc)
        for (const auto m : models)
            for (const double p : ps)
                EXPECT_NEAR(acceptance_probability(plan, p, {.model = m, .lot_size = 1000}), oc[k++], 1e-12)
                    << plan.n << " " << plan.c << " " << p;
}

TEST(AcceptanceSampling, PlansMatchFindPlan) {
    for (std::size_t i = 0; i + 10 <= plans.size(); i += 10) {
        const double aql = plans[i], rql = plans[i + 1], a = plans[i + 2], b = plans[i + 3];
        const auto bin = find_attribute_plan(aql, a, rql, b);
        EXPECT_EQ(bin.n, plans[i + 4]);
        EXPECT_EQ(bin.c, plans[i + 5]);
        const auto poi = find_attribute_plan(aql, a, rql, b, {.model = SamplingModel::poisson});
        EXPECT_EQ(poi.n, plans[i + 6]);
        EXPECT_EQ(poi.c, plans[i + 7]);
        const auto hyp = find_attribute_plan(aql, a, rql, b, {.model = SamplingModel::hypergeometric, .lot_size = 1000});
        EXPECT_EQ(hyp.n, plans[i + 8]);
        EXPECT_EQ(hyp.c, plans[i + 9]);
        // The plan meets both risk points and no smaller n does with any c.
        EXPECT_LE(acceptance_probability(bin, rql), b);
        EXPECT_GE(acceptance_probability(bin, aql), 1 - a);
        for (std::size_t c = 0; c < bin.n - 1; ++c) {
            const AttributePlan smaller{bin.n - 1, c};
            EXPECT_FALSE(acceptance_probability(smaller, rql) <= b && acceptance_probability(smaller, aql) >= 1 - a);
        }
    }
    EXPECT_THROW(find_attribute_plan(0.05, 0.05, 0.01, 0.1), std::invalid_argument);
    EXPECT_THROW(find_attribute_plan(0.01, 0.05, 0.05, 0.1, {.model = SamplingModel::hypergeometric}),
                 std::invalid_argument);
}

TEST(AcceptanceSampling, RectifyingInspection) {
    const std::vector<AttributePlan> pl{{50, 1}, {125, 3}};
    std::size_t k = 0;
    for (const auto& plan : pl) {
        for (const double p : {0.01, 0.02, 0.05}) {
            const auto r = rectifying_performance(plan, p, 1000);
            EXPECT_NEAR(r.acceptance, rect[k], 1e-12);
            EXPECT_NEAR(r.aoq, rect[k + 1], 1e-14);
            EXPECT_NEAR(r.ati, rect[k + 2], 1e-9);
            k += 3;
        }
        const auto a = average_outgoing_quality_limit(plan, 1000);
        EXPECT_NEAR(a.aoql, rect[k], 1e-10);
        EXPECT_NEAR(a.quality, rect[k + 1], 1e-5);
        k += 2;
    }
    EXPECT_THROW(rectifying_performance({50, 1}, 0.01, 20), std::invalid_argument);
}
