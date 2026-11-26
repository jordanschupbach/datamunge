#include <datamunge/stats/process_capability.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_capability_data.inc"

std::vector<std::vector<double>> groups(const std::vector<double>& x, const std::vector<double>& sizes) {
    std::vector<std::vector<double>> g;
    std::size_t at = 0;
    for (const double s : sizes) {
        g.emplace_back(x.begin() + static_cast<std::ptrdiff_t>(at), x.begin() + static_cast<std::ptrdiff_t>(at + s));
        at += static_cast<std::size_t>(s);
    }
    return g;
}

void expect_index(const CapabilityIndex& c, double v, double lo, double hi) {
    EXPECT_NEAR(c.value, v, 1e-10);
    EXPECT_NEAR(c.lower, lo, 1e-10);
    EXPECT_NEAR(c.upper, hi, 1e-10);
}

// ref = (sigma, Cp, lo, hi, Cpl, lo, hi, Cpu, lo, hi, Cpk, lo, hi, Cpm, lo, hi)
void expect_family(const CapabilityIndex& p, const CapabilityIndex& l, const CapabilityIndex& u,
                   const CapabilityIndex& k, const std::vector<double>& r) {
    expect_index(p, r[1], r[2], r[3]);
    expect_index(l, r[4], r[5], r[6]);
    expect_index(u, r[7], r[8], r[9]);
    expect_index(k, r[10], r[11], r[12]);
}
} // namespace

TEST(ProcessCapability, Constants) {
    for (std::size_t n = 2; n <= 25; ++n) {
        EXPECT_NEAR(range_constant_d2(n), d2_exact[n - 2], 1e-11) << n;
        EXPECT_NEAR(range_constant_d2(n), d2[n - 2], 6e-4) << n; // qcc's 3-decimal table
        EXPECT_NEAR(sd_constant_c4(n), c4[n - 2], 1e-14) << n;
    }
    EXPECT_THROW(range_constant_d2(1), std::invalid_argument);
}

TEST(ProcessCapability, PistonRingsMatchQcc) {
    const auto g = groups(pr_x, std::vector<double>(25, 5));
    const CapabilityOptions base{.lsl = 73.95, .usl = 74.05, .target = 74};
    auto o = base;
    const auto pooled = process_capability(g, o);
    EXPECT_NEAR(pooled.sigma_within, pr_pooled[0], 1e-15);
    expect_family(pooled.cp, pooled.cpl, pooled.cpu, pooled.cpk, pr_pooled);
    EXPECT_NEAR(pooled.sigma_overall, pr_overall[0], 1e-15);
    expect_family(pooled.pp, pooled.ppl, pooled.ppu, pooled.ppk, pr_overall);
    expect_index(pooled.cpm, pr_overall[13], pr_overall[14], pr_overall[15]);
    EXPECT_NEAR(pooled.expected_ppm_overall, pr_ppm[0], 1e-9);
    EXPECT_EQ(pooled.observed_ppm, pr_ppm[1]);
    o.within = WithinSigma::average_sd;
    const auto sd = process_capability(g, o);
    EXPECT_NEAR(sd.sigma_within, pr_sd[0], 1e-15);
    expect_family(sd.cp, sd.cpl, sd.cpu, sd.cpk, pr_sd);
    o.within = WithinSigma::average_range;
    const auto r = process_capability(g, o);
    EXPECT_NEAR(r.sigma_within, range_exact[0], 1e-14);
    EXPECT_NEAR(r.cp.value * r.sigma_within, pr_range[1] * pr_range[0], 1e-12); // same Cp up to d2
    o = base;
    o.unbiased_overall = true;
    const auto c4o = process_capability(g, o);
    EXPECT_NEAR(c4o.sigma_overall, pr_overall_c4[0], 1e-15);
    expect_family(c4o.pp, c4o.ppl, c4o.ppu, c4o.ppk, pr_overall_c4);
}

TEST(ProcessCapability, UnequalSubgroupsAndIndividuals) {
    const auto g = groups(un_x, un_sizes);
    CapabilityOptions o{.lsl = 73.95, .usl = 74.05};
    o.within = WithinSigma::average_range;
    EXPECT_NEAR(process_capability(g, o).sigma_within, range_exact[1], 1e-14);
    o.within = WithinSigma::average_sd;
    EXPECT_NEAR(process_capability(g, o).sigma_within, un_sd[1], 1e-15);
    o.within = WithinSigma::pooled_sd;
    EXPECT_NEAR(process_capability(g, o).sigma_within, un_sd[2], 1e-15);

    const auto i2 = process_capability(ind_x, {.usl = 74.03});
    EXPECT_NEAR(i2.sigma_within, range_exact[2], 1e-14);
    EXPECT_NEAR(i2.cpu.value * i2.sigma_within, ind_sd[2] * ind_sd[0], 1e-12);
    EXPECT_EQ(i2.cpk.value, i2.cpu.value); // one-sided: Cpk is the single index
    EXPECT_TRUE(std::isnan(i2.cp.value) && std::isnan(i2.cpl.value) && std::isnan(i2.cpm.value));
    const auto i3 = process_capability(ind_x, {.usl = 74.03, .moving_range_span = 3});
    EXPECT_NEAR(i3.sigma_within, range_exact[3], 1e-14);

    EXPECT_THROW(process_capability(ind_x, {}), std::invalid_argument);
    EXPECT_THROW(process_capability(ind_x, {.lsl = 2, .usl = 1}), std::invalid_argument);
    EXPECT_THROW(process_capability(groups(ind_x, std::vector<double>(45, 1)), {.lsl = 1}), std::invalid_argument);
}
