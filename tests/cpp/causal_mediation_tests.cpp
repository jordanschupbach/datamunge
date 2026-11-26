#include <datamunge/causal/mediation.hpp>
#include <datamunge/stats/gauss_hermite.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using datamunge::causal::mediation_analysis;
using datamunge::causal::MediationModel;
using datamunge::causal::MediationOptions;
using datamunge::causal::MediationResult;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "causal_mediation_data.inc"

Matrix covariates() {
    Matrix c(jobs_treat.size(), 3);
    for (std::size_t i = 0; i < jobs_treat.size(); ++i) {
        c(i, 0) = jobs_econ_hard[i];
        c(i, 1) = jobs_sex[i];
        c(i, 2) = jobs_age[i];
    }
    return c;
}

// Layout: delta(0), delta(1), zeta(0), zeta(1), total.
void expect_effects(const MediationResult& r, const std::vector<double>& ref, double tol, const std::string& what) {
    EXPECT_NEAR(r.indirect_control.estimate, ref[0], tol) << what;
    EXPECT_NEAR(r.indirect_treated.estimate, ref[1], tol) << what;
    EXPECT_NEAR(r.direct_control.estimate, ref[2], tol) << what;
    EXPECT_NEAR(r.direct_treated.estimate, ref[3], tol) << what;
    EXPECT_NEAR(r.total.estimate, ref[4], tol) << what;
    EXPECT_NEAR(r.proportion_mediated.estimate, ref[1] / ref[4], 1e3 * tol) << what;
}
} // namespace

TEST(Mediation, LinearModelsMatchReferences) {
    const auto c = covariates();
    expect_effects(mediation_analysis(jobs_depress2, jobs_treat, jobs_job_seek, c, {.interaction = false}),
                   ref_linear, 1e-12, "linear");
    expect_effects(mediation_analysis(jobs_depress2, jobs_treat, jobs_job_seek, c), ref_linear_inter, 1e-12,
                   "linear with interaction");
}

TEST(Mediation, NonlinearModelsMatchExactGFormula) {
    const auto c = covariates();
    expect_effects(mediation_analysis(jobs_work1, jobs_treat, jobs_job_seek, c,
                                      {.outcome_model = MediationModel::logistic}),
                   ref_logit_outcome, 1e-9, "logistic outcome");
    expect_effects(mediation_analysis(jobs_depress2, jobs_treat, jobs_jb, c,
                                      {.mediator_model = MediationModel::logistic}),
                   ref_binary_mediator, 1e-9, "binary mediator");
    expect_effects(mediation_analysis(jobs_work1, jobs_treat, jobs_jb, c,
                                      {.mediator_model = MediationModel::logistic,
                                       .outcome_model = MediationModel::logistic}),
                   ref_binary_both, 1e-9, "binary mediator and outcome");
}

TEST(Mediation, BootstrapIsReproducible) {
    const auto c = covariates();
    const MediationOptions o{.bootstrap = 200, .seed = 5};
    const auto a = mediation_analysis(jobs_depress2, jobs_treat, jobs_job_seek, c, o);
    const auto b = mediation_analysis(jobs_depress2, jobs_treat, jobs_job_seek, c, o);
    EXPECT_EQ(a.indirect_treated.standard_error, b.indirect_treated.standard_error);
    EXPECT_GT(a.indirect_treated.standard_error, 0);
    EXPECT_LT(a.indirect_treated.lower, a.indirect_treated.estimate);
    EXPECT_GT(a.indirect_treated.upper, a.indirect_treated.estimate);
    EXPECT_EQ(a.bootstrap_failures, 0u);
    const auto none = mediation_analysis(jobs_depress2, jobs_treat, jobs_job_seek, c);
    EXPECT_TRUE(std::isnan(none.total.standard_error));
}

TEST(Mediation, ValidatesInputs) {
    const auto c = covariates();
    EXPECT_THROW(mediation_analysis({1, 2}, {0, 1, 1}, {1, 2}), std::invalid_argument);
    EXPECT_THROW(mediation_analysis(jobs_depress2, jobs_treat, jobs_job_seek, c,
                                    {.outcome_model = MediationModel::logistic}),
                 std::invalid_argument);
    EXPECT_THROW(mediation_analysis(jobs_depress2, jobs_treat, jobs_job_seek, c,
                                    {.control_value = 1, .treated_value = 1}),
                 std::invalid_argument);
}

TEST(GaussHermite, IntegratesNormalMoments) {
    for (const std::size_t n : {1u, 2u, 5u, 20u, 60u}) {
        const auto r = datamunge::stats::normal_gauss_hermite(n);
        ASSERT_EQ(r.nodes.size(), n);
        double s0 = 0, s2 = 0, s1 = 0;
        for (std::size_t j = 0; j < n; ++j) {
            s0 += r.weights[j];
            s1 += r.weights[j] * r.nodes[j];
            s2 += r.weights[j] * r.nodes[j] * r.nodes[j];
        }
        EXPECT_NEAR(s0, 1, 1e-12);
        EXPECT_NEAR(s1, 0, 1e-12);
        if (n >= 2)
            EXPECT_NEAR(s2, 1, 1e-10);
    }
    // E[exp(Z)] = exp(1/2).
    const auto r = datamunge::stats::normal_gauss_hermite(30);
    double e = 0;
    for (std::size_t j = 0; j < 30; ++j)
        e += r.weights[j] * std::exp(r.nodes[j]);
    EXPECT_NEAR(e, std::exp(0.5), 1e-12);
    EXPECT_THROW(datamunge::stats::normal_gauss_hermite(0), std::invalid_argument);
}
