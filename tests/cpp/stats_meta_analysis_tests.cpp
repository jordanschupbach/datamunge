#include <datamunge/stats/meta_analysis.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_meta_analysis_data.inc"
#include "stats_random_effects_data.inc"
#include "stats_hartung_knapp_data.inc"

void expect_fe(const MetaAnalysis& m, const std::vector<double>& r) {
    EXPECT_NEAR(m.estimate, r[0], 1e-12);
    EXPECT_NEAR(m.standard_error, r[1], 1e-12);
    EXPECT_NEAR(m.statistic, r[2], 1e-10);
    EXPECT_NEAR(m.p_value, r[3], 1e-12 + 1e-6 * r[3]);
    EXPECT_NEAR(m.lower, r[4], 1e-10);
    EXPECT_NEAR(m.upper, r[5], 1e-10);
    EXPECT_NEAR(m.q, r[6], 1e-9);
    EXPECT_NEAR(m.q_p_value, r[7], 1e-300 + 1e-8 * r[7]);
    EXPECT_NEAR(m.i2, r[8], 1e-9);
    EXPECT_NEAR(m.h2, r[9], 1e-10);
}
} // namespace

TEST(CommonEffectMeta, MatchesMetafor) {
    const auto m = common_effect_meta(bcg_yi, bcg_vi);
    expect_fe(m, bcg_fe);
    ASSERT_EQ(m.weights.size(), bcg_fe_w.size());
    for (std::size_t i = 0; i < m.weights.size(); ++i)
        EXPECT_NEAR(m.weights[i], bcg_fe_w[i], 1e-12);
    EXPECT_NEAR(std::accumulate(m.weights.begin(), m.weights.end(), 0.0), 1.0, 1e-14);
    EXPECT_EQ(m.k, 13u);
    EXPECT_EQ(m.q_df, 12.0);
    const auto m90 = common_effect_meta(bcg_yi, bcg_vi, 0.90);
    EXPECT_NEAR(m90.lower, bcg_fe90[0], 1e-10);
    EXPECT_NEAR(m90.upper, bcg_fe90[1], 1e-10);
    expect_fe(common_effect_meta(small_y, small_v), small_fe);
}

TEST(CommonEffectMeta, SingleStudyAndValidation) {
    const auto one = common_effect_meta({0.4}, {0.09});
    EXPECT_DOUBLE_EQ(one.estimate, 0.4);
    EXPECT_NEAR(one.standard_error, 0.3, 1e-15);
    EXPECT_TRUE(std::isnan(one.q_p_value) && std::isnan(one.i2));
    EXPECT_THROW(common_effect_meta({}, {}), std::invalid_argument);
    EXPECT_THROW(common_effect_meta({1.0, 2.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(common_effect_meta({1.0}, {0.0}), std::invalid_argument);
    EXPECT_THROW(common_effect_meta({1.0}, {1.0}, 1.0), std::invalid_argument);
}

namespace {
void expect_re(const std::vector<double>& y, const std::vector<double>& v, TauEstimator e,
               const std::vector<double>& r, const char* what) {
    const auto m = random_effects_meta(y, v, {.estimator = e});
    const double tol = 1e-9;
    EXPECT_NEAR(m.estimate, r[0], tol) << what;
    EXPECT_NEAR(m.standard_error, r[1], tol) << what;
    EXPECT_NEAR(m.statistic, r[2], 1e-8) << what;
    EXPECT_NEAR(m.p_value, r[3], 1e-9 + 1e-7 * r[3]) << what;
    EXPECT_NEAR(m.lower, r[4], tol) << what;
    EXPECT_NEAR(m.upper, r[5], tol) << what;
    EXPECT_NEAR(m.tau2, r[6], tol) << what;
    EXPECT_NEAR(m.i2, r[7], 1e-7) << what;
    EXPECT_NEAR(m.h2, r[8], 1e-8) << what;
    EXPECT_NEAR(m.prediction_lower, r[9], tol) << what;
    EXPECT_NEAR(m.prediction_upper, r[10], tol) << what;
    EXPECT_NEAR(m.tau2_lower, r[11], 1e-8) << what;
    EXPECT_NEAR(m.tau2_upper, r[12], 1e-8 * std::max(1.0, r[12])) << what;
    EXPECT_NEAR(m.weights[0], r[15], 1e-10) << what;
    EXPECT_TRUE(m.converged) << what;
    const auto t = random_effects_meta(y, v, {.estimator = e, .t_prediction = true});
    EXPECT_NEAR(t.prediction_lower, r[13], tol) << what;
    EXPECT_NEAR(t.prediction_upper, r[14], tol) << what;
}
} // namespace

TEST(RandomEffectsMeta, MatchesMetaforForAllEstimators) {
    const std::pair<TauEstimator, const char*> est[4]{{TauEstimator::dersimonian_laird, "DL"},
                                                      {TauEstimator::paule_mandel, "PM"},
                                                      {TauEstimator::reml, "REML"},
                                                      {TauEstimator::ml, "ML"}};
    const std::vector<double>* bcg[4]{&bcg_dl, &bcg_pm, &bcg_reml, &bcg_ml};
    const std::vector<double>* norm[4]{&norm_dl, &norm_pm, &norm_reml, &norm_ml};
    const std::vector<double>* hom[4]{&hom_dl, &hom_pm, &hom_reml, &hom_ml};
    for (int i = 0; i < 4; ++i) {
        expect_re(bcg_yi, bcg_vi, est[i].first, *bcg[i], est[i].second);
        expect_re(norm_y, norm_v, est[i].first, *norm[i], est[i].second);
        expect_re(hom_y, hom_v, est[i].first, *hom[i], est[i].second);
    }
}

TEST(RandomEffectsMeta, ReducesToCommonEffectWithoutHeterogeneity) {
    const auto re = random_effects_meta(hom_y, hom_v);
    const auto ce = common_effect_meta(hom_y, hom_v);
    EXPECT_EQ(re.tau2, 0.0);
    EXPECT_NEAR(re.estimate, ce.estimate, 1e-14);
    EXPECT_NEAR(re.prediction_upper, ce.prediction_upper, 1e-14);
    EXPECT_NEAR(re.q, ce.q, 1e-12);
    EXPECT_EQ(ce.tau2, 0.0);
    EXPECT_TRUE(std::isnan(ce.tau2_lower));
    EXPECT_THROW(random_effects_meta({1.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(random_effects_meta({1.0, 2.0}, {1.0, 1.0}, {.t_prediction = true}), std::invalid_argument);
}

TEST(HartungKnapp, MatchesMetaforKnhaAndAdhoc) {
    struct Case {
        const std::vector<double>*y, *v;
        TauEstimator e;
        MetaInference inf;
        const std::vector<double>* ref;
    };
    const Case cases[]{
        {&bcg_yi, &bcg_vi, TauEstimator::dersimonian_laird, MetaInference::hartung_knapp, &bcg_dl_knha},
        {&bcg_yi, &bcg_vi, TauEstimator::dersimonian_laird, MetaInference::hartung_knapp_truncated, &bcg_dl_adhoc},
        {&bcg_yi, &bcg_vi, TauEstimator::reml, MetaInference::hartung_knapp, &bcg_reml_knha},
        {&bcg_yi, &bcg_vi, TauEstimator::reml, MetaInference::hartung_knapp_truncated, &bcg_reml_adhoc},
        {&norm_y, &norm_v, TauEstimator::dersimonian_laird, MetaInference::hartung_knapp, &norm_dl_knha},
        {&norm_y, &norm_v, TauEstimator::dersimonian_laird, MetaInference::hartung_knapp_truncated, &norm_dl_adhoc},
        {&norm_y, &norm_v, TauEstimator::reml, MetaInference::hartung_knapp, &norm_reml_knha},
        {&norm_y, &norm_v, TauEstimator::reml, MetaInference::hartung_knapp_truncated, &norm_reml_adhoc},
        {&hom_y, &hom_v, TauEstimator::dersimonian_laird, MetaInference::hartung_knapp, &hom_dl_knha},
        {&hom_y, &hom_v, TauEstimator::dersimonian_laird, MetaInference::hartung_knapp_truncated, &hom_dl_adhoc},
        {&hom_y, &hom_v, TauEstimator::reml, MetaInference::hartung_knapp, &hom_reml_knha},
        {&hom_y, &hom_v, TauEstimator::reml, MetaInference::hartung_knapp_truncated, &hom_reml_adhoc}};
    int c = 0;
    for (const auto& cs : cases) {
        const auto m = random_effects_meta(*cs.y, *cs.v, {.estimator = cs.e, .inference = cs.inf});
        const auto& r = *cs.ref;
        EXPECT_NEAR(m.estimate, r[0], 1e-9) << c;
        EXPECT_NEAR(m.standard_error, r[1], 1e-9) << c;
        EXPECT_NEAR(m.statistic, r[2], 1e-8) << c;
        EXPECT_NEAR(m.p_value, r[3], 1e-9) << c;
        EXPECT_NEAR(m.lower, r[4], 1e-9) << c;
        EXPECT_NEAR(m.upper, r[5], 1e-9) << c;
        EXPECT_NEAR(m.prediction_lower, r[6], 1e-9) << c;
        EXPECT_NEAR(m.prediction_upper, r[7], 1e-9) << c;
        EXPECT_EQ(m.df, r[8]) << c;
        ++c;
    }
    // The truncated variant is never narrower than Wald; plain HK can be (homogeneous data).
    const auto wald = random_effects_meta(hom_y, hom_v);
    const auto hk = random_effects_meta(hom_y, hom_v, {.inference = MetaInference::hartung_knapp});
    EXPECT_LT(hk.variance_scale, 1.0);
    EXPECT_LT(hk.standard_error, wald.standard_error);
    EXPECT_TRUE(std::isnan(wald.df));
    EXPECT_EQ(wald.variance_scale, 1.0);
}

namespace {
#include "stats_meta_regression_data.inc"

datamunge::linalg::DenseMatrix<double> columns(const std::vector<std::vector<double>>& cols) {
    datamunge::linalg::DenseMatrix<double> x(cols[0].size(), cols.size());
    for (std::size_t j = 0; j < cols.size(); ++j)
        for (std::size_t i = 0; i < cols[j].size(); ++i)
            x(i, j) = cols[j][i];
    return x;
}

void expect_mr(const MetaRegression& r, const std::vector<double>& ref, const std::vector<double>& tci,
               const char* what) {
    const std::size_t p = r.p;
    for (std::size_t j = 0; j < p; ++j) {
        EXPECT_NEAR(r.coefficients[j], ref[j], 1e-9) << what << j;
        EXPECT_NEAR(r.standard_errors[j], ref[p + j], 1e-9) << what << j;
        EXPECT_NEAR(r.statistics[j], ref[2 * p + j], 1e-7) << what << j;
        EXPECT_NEAR(r.p_values[j], ref[3 * p + j], 1e-9) << what << j;
        EXPECT_NEAR(r.lower[j], ref[4 * p + j], 1e-9) << what << j;
        EXPECT_NEAR(r.upper[j], ref[5 * p + j], 1e-9) << what << j;
    }
    const std::size_t o = 6 * p;
    EXPECT_NEAR(r.tau2, ref[o], 1e-10) << what;
    EXPECT_NEAR(r.qe, ref[o + 1], 1e-8) << what;
    EXPECT_NEAR(r.qe_p_value, ref[o + 2], 1e-10) << what;
    EXPECT_NEAR(r.qm, ref[o + 3], 1e-7) << what;
    EXPECT_NEAR(r.qm_p_value, ref[o + 4], 1e-10) << what;
    EXPECT_NEAR(r.i2, ref[o + 5], 1e-6) << what;
    EXPECT_NEAR(r.h2, ref[o + 6], 1e-8) << what;
    EXPECT_NEAR(r.r2, ref[o + 7], 1e-6) << what;
    EXPECT_NEAR(r.tau2_lower, tci[0], 1e-8) << what;
    EXPECT_NEAR(r.tau2_upper, tci[1], 1e-8) << what;
}
} // namespace

TEST(MetaRegression, MatchesMetafor) {
    expect_mr(meta_regression(bcg_yi, bcg_vi, columns({bcg_ablat, bcg_year})), mr_reml2, mr_reml2_tau2ci,
              "REML ablat + year");
    expect_mr(meta_regression(bcg_yi, bcg_vi, columns({bcg_ablat}),
                              {.model = {.estimator = TauEstimator::dersimonian_laird,
                                         .inference = MetaInference::hartung_knapp}}),
              mr_dl_knha, mr_dl_knha_tau2ci, "DL knha ablat");
    const auto r = meta_regression(bcg_yi, bcg_vi, columns({bcg_ablat}), {.influence = true});
    expect_mr(r, mr_reml1, mr_reml1_tau2ci, "REML ablat");
    const auto& inf = r.influence;
    for (std::size_t i = 0; i < 13; ++i) {
        EXPECT_NEAR(r.fitted[i], mr_fitted[i], 1e-10) << i;
        EXPECT_NEAR(inf.standardized_residuals[i], inf_rstandard[i], 1e-8) << i;
        EXPECT_NEAR(inf.studentized_residuals[i], inf_rstudent[i], 1e-7) << i;
        EXPECT_NEAR(inf.dffits[i], inf_dffits[i], 1e-7) << i;
        EXPECT_NEAR(inf.cooks_distance[i], inf_cook[i], 1e-7) << i;
        EXPECT_NEAR(inf.covariance_ratio[i], inf_covr[i], 1e-7) << i;
        EXPECT_NEAR(inf.tau2_deleted[i], inf_tau2del[i], 1e-9) << i;
        EXPECT_NEAR(inf.qe_deleted[i], inf_qedel[i], 1e-7) << i;
        EXPECT_NEAR(inf.hat[i], inf_hat[i], 1e-10) << i;
        EXPECT_NEAR(inf.weight[i], inf_weight[i], 1e-8) << i;
        EXPECT_NEAR(inf.dfbetas[i][0], inf_dfbs_int[i], 1e-7) << i;
        EXPECT_NEAR(inf.dfbetas[i][1], inf_dfbs_ablat[i], 1e-7) << i;
        EXPECT_EQ(inf.influential[i], inf_flag[i] != 0) << i;
    }
    auto y = bcg_yi, v = bcg_vi, lat = bcg_ablat;
    y.push_back(1.5), v.push_back(0.01), lat.push_back(50);
    const auto o = meta_regression(y, v, columns({lat}), {.influence = true});
    for (std::size_t i = 0; i < 14; ++i) {
        EXPECT_NEAR(o.influence.cooks_distance[i], out_cook[i], 1e-6) << i;
        EXPECT_NEAR(o.influence.studentized_residuals[i], out_rstudent[i], 1e-6) << i;
        EXPECT_EQ(o.influence.influential[i], out_flag[i] != 0) << i;
    }
    EXPECT_TRUE(o.influence.influential[13]);
    EXPECT_THROW(meta_regression({1.0, 2.0}, {1.0, 1.0}, columns({{1.0, 2.0}})), std::invalid_argument);
    EXPECT_THROW(meta_regression(bcg_yi, bcg_vi, columns({{1.0, 2.0}})), std::invalid_argument);
}

namespace {
#include "stats_egger_data.inc"

void expect_egger(const EggerTest& e, const std::vector<double>& r, const char* what) {
    EXPECT_NEAR(e.slope, r[0], 1e-9 * std::max(1.0, std::abs(r[0]))) << what;
    EXPECT_NEAR(e.slope_se, r[1], 1e-9 * std::max(1.0, r[1])) << what;
    EXPECT_NEAR(e.statistic, r[2], 1e-8) << what;
    if (std::isnan(r[3]))
        EXPECT_TRUE(std::isnan(e.df)) << what;
    else
        EXPECT_EQ(e.df, r[3]) << what;
    EXPECT_NEAR(e.p_value, r[4], 1e-9) << what;
    if (r.size() > 6) {
        EXPECT_NEAR(e.limit_estimate, r[5], 1e-9) << what;
        EXPECT_NEAR(e.limit_lower, r[6], 1e-9) << what;
        EXPECT_NEAR(e.limit_upper, r[7], 1e-9) << what;
        EXPECT_NEAR(e.limit_se, r[8], 1e-9) << what;
    } else {
        EXPECT_NEAR(e.limit_se, r[5], 1e-9) << what;
    }
}
} // namespace

TEST(EggerTest, MatchesMetaforRegtest) {
    using P = SmallStudyPredictor;
    expect_egger(egger_test(bcg_yi, bcg_vi), eg_lm_sei, "lm sei");
    expect_egger(egger_test(bcg_yi, bcg_vi, {.predictor = P::variance}), eg_lm_vi, "lm vi");
    expect_egger(egger_test(bcg_yi, bcg_vi, {.predictor = P::inverse_sample_size, .sample_sizes = bcg_n}),
                 eg_lm_ninv, "lm ninv");
    expect_egger(egger_test(bcg_yi, bcg_vi, {.predictor = P::inverse_sqrt_sample_size, .sample_sizes = bcg_n}),
                 eg_lm_sqrtninv, "lm sqrtninv");
    expect_egger(egger_test(bcg_yi, bcg_vi, {.predictor = P::sample_size, .sample_sizes = bcg_n}), eg_lm_ni,
                 "lm ni");
    expect_egger(egger_test(bcg_yi, bcg_vi, {.model = SmallStudyModel::random_effects}), eg_rma_sei, "rma sei");
    expect_egger(egger_test(bcg_yi, bcg_vi,
                            {.predictor = P::inverse_sample_size,
                             .model = SmallStudyModel::random_effects,
                             .sample_sizes = bcg_n,
                             .random_effects = {.inference = MetaInference::hartung_knapp}}),
                 eg_rma_ninv_knha, "rma ninv knha");
    expect_egger(egger_test(bcg_yi, bcg_vi, {.inverse_variance_weights = false}), eg_lm_unweighted, "unweighted");
    EXPECT_THROW(egger_test(bcg_yi, bcg_vi, {.predictor = P::inverse_sample_size}), std::invalid_argument);
    EXPECT_THROW(egger_test({0.1, 0.2}, {0.01, 0.02}), std::invalid_argument);
    EXPECT_THROW(egger_test({0.1, 0.2, 0.3}, {0.01, 0.01, 0.01}), std::invalid_argument);
}
