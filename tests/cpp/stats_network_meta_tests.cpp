#include <datamunge/stats/network_meta_analysis.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_network_meta_data.inc"

std::vector<NetworkStudy> smoking() {
    std::vector<NetworkStudy> out;
    std::size_t i = 0;
    while (i < nma_study.size()) {
        std::vector<std::size_t> t;
        std::vector<double> e, v;
        const double id = nma_study[i];
        for (; i < nma_study.size() && nma_study[i] == id; ++i)
            t.push_back(static_cast<std::size_t>(nma_treat[i])), e.push_back(nma_est[i]), v.push_back(nma_var[i]);
        out.push_back(network_study_from_arms(t, e, v));
    }
    return out;
}

void expect_matrix(const datamunge::linalg::DenseMatrix<double>& m, const std::vector<double>& ref, double tol,
                   const char* what) {
    for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t i = 0; i < 4; ++i)
            EXPECT_NEAR(m(i, j), ref[i + 4 * j], tol) << what << " " << i << "," << j;
}
} // namespace

TEST(NetworkMeta, MatchesNetmetaSmokingCessation) {
    const auto studies = smoking();
    ASSERT_EQ(studies.size(), 24u);
    const auto n = network_meta_analysis(studies);
    EXPECT_EQ(n.treatments, 4u);
    EXPECT_EQ(n.studies, static_cast<std::size_t>(nma_q[14]));
    EXPECT_EQ(n.designs, static_cast<std::size_t>(nma_q[16]));
    expect_matrix(n.common.estimate, nma_te_common, 1e-10, "TE common");
    expect_matrix(n.common.standard_error, nma_se_common, 1e-10, "se common");
    expect_matrix(n.random.estimate, nma_te_random, 1e-9, "TE random");
    expect_matrix(n.random.standard_error, nma_se_random, 1e-9, "se random");
    expect_matrix(n.random.lower, nma_lower_random, 1e-9, "lower random");
    for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t i = 0; i < 4; ++i)
            if (i != j)
                EXPECT_NEAR(n.random.p_value(i, j), nma_p_random[i + 4 * j], 1e-9);
    EXPECT_NEAR(n.q, nma_q[0], 1e-8);
    EXPECT_EQ(n.q_df, nma_q[1]);
    EXPECT_NEAR(n.q_p_value, nma_q[2], 1e-40 + 1e-8 * nma_q[2]);
    EXPECT_NEAR(n.tau2, nma_q[3], 1e-10);
    EXPECT_NEAR(n.i2, nma_q[4], 1e-8);
    EXPECT_NEAR(n.q_within, nma_q[6], 1e-8);
    EXPECT_NEAR(n.q_between, nma_q[7], 1e-8);
    EXPECT_EQ(n.q_within_df, nma_q[9]);
    EXPECT_EQ(n.q_between_df, nma_q[10]);
    EXPECT_NEAR(n.q_between_p_value, nma_q[13], 1e-9);
    EXPECT_NEAR(n.q_within_p_value, nma_q[12], 1e-40 + 1e-8 * nma_q[12]);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(n.common.p_scores[i], nma_pscore_common[i], 1e-10) << i;
        EXPECT_NEAR(n.random.p_scores[i], nma_pscore_random[i], 1e-10) << i;
    }
    const auto small = network_meta_analysis(studies, {.larger_is_better = false});
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(small.random.p_scores[i], nma_pscore_random_small[i], 1e-10) << i;
}

TEST(NetworkMeta, RemlAndReferenceInvariance) {
    const auto studies = smoking();
    const auto r = network_meta_analysis(studies, {.estimator = NetworkTauEstimator::reml});
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.tau2, nma_reml[0], 1e-10);
    std::vector<double> te(nma_reml.begin() + 1, nma_reml.begin() + 17), se(nma_reml.begin() + 17, nma_reml.end());
    expect_matrix(r.random.estimate, te, 1e-9, "TE REML");
    expect_matrix(r.random.standard_error, se, 1e-9, "se REML");
    // The reference only changes the parameterisation.
    const auto c = network_meta_analysis(studies, {.reference = 2});
    const auto a = network_meta_analysis(studies);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(c.random.estimate(i, j), a.random.estimate(i, j), 1e-10);
}

TEST(NetworkMeta, IndirectEvidenceAndValidation) {
    // A-B and B-C trials only: the A-C effect is purely indirect, with variance summed.
    std::vector<NetworkStudy> s{{{0, 1}, {0.5}, datamunge::linalg::DenseMatrix<double>(1, 1, 0.04)},
                                {{1, 2}, {0.3}, datamunge::linalg::DenseMatrix<double>(1, 1, 0.09)}};
    const auto n = network_meta_analysis(s);
    EXPECT_NEAR(n.common.estimate(2, 0), 0.8, 1e-12);
    EXPECT_NEAR(n.common.standard_error(2, 0), std::sqrt(0.13), 1e-12);
    EXPECT_EQ(n.q_df, 0.0);
    // Disconnected network.
    std::vector<NetworkStudy> d{{{0, 1}, {0.5}, datamunge::linalg::DenseMatrix<double>(1, 1, 0.04)},
                                {{2, 3}, {0.3}, datamunge::linalg::DenseMatrix<double>(1, 1, 0.09)}};
    EXPECT_THROW(network_meta_analysis(d), std::invalid_argument);
    EXPECT_THROW(network_study_from_arms({0, 0}, {1, 2}, {1, 1}), std::invalid_argument);
    EXPECT_THROW(network_meta_analysis(s, {.reference = 5}), std::invalid_argument);
}
