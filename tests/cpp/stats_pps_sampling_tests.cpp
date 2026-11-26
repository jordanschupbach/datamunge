#include <datamunge/stats/pps_sampling.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using datamunge::random::SplitMix64;

#include "stats_pps_sampling_data.inc"

// Reference: sampling::inclusionprobabilities with two certainty units after capping.
TEST(PpsSampling, InclusionProbabilitiesMatchSampling) {
    const auto pi = inclusion_probabilities(k_sizes,7);
    ASSERT_EQ(pi.size(),ref_pi.size());
    for (std::size_t k = 0; k < pi.size(); ++k)
        EXPECT_NEAR(pi[k],ref_pi[k],1e-13) << k;
    EXPECT_THROW((void)inclusion_probabilities({1,0,2},3),std::invalid_argument);
    EXPECT_THROW((void)inclusion_probabilities({1,-1},1),std::invalid_argument);
}

// References: sampling::UPsystematicpi2 and sampling::UPsampfordpi2.
TEST(PpsSampling, JointInclusionMatchesSampling) {
    const auto sys = systematic_joint_inclusion(ref_pi);
    const std::size_t N = ref_pi.size();
    for (std::size_t k = 0; k < N; ++k)
        for (std::size_t l = 0; l < N; ++l)
            EXPECT_NEAR(sys(k,l),ref_systematic_joint[k+l*N],1e-12) << k << "," << l;
    const auto sam = sampford_joint_inclusion(k_pi_open);
    const std::size_t M = k_pi_open.size();
    for (std::size_t k = 0; k < M; ++k)
        for (std::size_t l = 0; l < M; ++l)
            EXPECT_NEAR(sam(k,l),ref_sampford_joint[k+l*M],1e-12) << k << "," << l;
    // Fixed size: sum over l != k of pi_kl = (n - 1) pi_k, including with certainty units.
    const auto with_certain = sampford_joint_inclusion(ref_pi);
    for (std::size_t k = 0; k < N; ++k) {
        double s = 0;
        for (std::size_t l = 0; l < N; ++l)
            if (l != k)
                s += with_certain(k,l);
        EXPECT_NEAR(s,6*ref_pi[k],1e-10) << k;
    }
}

// Monte Carlo check of first-order and (for systematic and Sampford) joint probabilities.
TEST(PpsSampling, SamplersReproduceDeclaredProbabilities) {
    const std::size_t N = ref_pi.size(), reps = 40000;
    for (const auto method : {PpsMethod::systematic,PpsMethod::randomized_systematic,PpsMethod::sampford,
                              PpsMethod::brewer,PpsMethod::pivotal}) {
        SplitMix64 rng(17);
        std::vector<double> count(N,0.0);
        std::vector<double> pair(N*N,0.0);
        for (std::size_t r = 0; r < reps; ++r) {
            const auto s = pps_sample(ref_pi,method,rng);
            ASSERT_EQ(s.size(),7u);
            for (std::size_t a = 0; a < s.size(); ++a) {
                if (a > 0)
                    ASSERT_LT(s[a-1],s[a]);
                count[s[a]] += 1;
                for (std::size_t b = 0; b < s.size(); ++b)
                    pair[s[a]*N+s[b]] += 1;
            }
        }
        for (std::size_t k = 0; k < N; ++k) {
            const double se = std::sqrt(ref_pi[k]*(1-ref_pi[k])/reps);
            EXPECT_NEAR(count[k]/reps,ref_pi[k],4.5*se+1e-12) << static_cast<int>(method) << " unit " << k;
        }
        if (method == PpsMethod::systematic || method == PpsMethod::sampford) {
            const auto exact = method == PpsMethod::systematic ? systematic_joint_inclusion(ref_pi)
                                                               : sampford_joint_inclusion(ref_pi);
            for (std::size_t k = 0; k < N; ++k)
                for (std::size_t l = 0; l < k; ++l) {
                    const double p = exact(k,l), se = std::sqrt(p*(1-p)/reps);
                    EXPECT_NEAR(pair[k*N+l]/reps,p,4.5*se+1e-12) << static_cast<int>(method) << " " << k << "," << l;
                }
        }
    }
}

TEST(PpsSampling, Validates) {
    SplitMix64 rng(1);
    EXPECT_THROW((void)pps_sample({0.5,0.4},PpsMethod::sampford,rng),std::invalid_argument);
    EXPECT_THROW((void)pps_sample({0.5,1.5},PpsMethod::brewer,rng),std::invalid_argument);
    EXPECT_EQ(pps_sample({1,0,1},PpsMethod::pivotal,rng),(std::vector<std::size_t>{0,2}));
    EXPECT_THROW((void)systematic_joint_inclusion({}),std::invalid_argument);
}

} // namespace
