#include <datamunge/stats/survey_glm.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_survey_glm_data.inc"

std::vector<int> ids(const std::vector<double>& x) {
    return {x.begin(),x.end()};
}

Matrix columns(const std::vector<std::vector<double>>& cols) {
    Matrix x(cols[0].size(),cols.size()+1);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        x(i,0) = 1;
        for (std::size_t a = 0; a < cols.size(); ++a)
            x(i,a+1) = cols[a][i];
    }
    return x;
}

// R's glm evaluates the IRLS weights at the means from the start of its last iteration, so for
// non-Gaussian families its sandwich differs from the converged one at about 1e-6.
void expect_fit(const SurveyGlmFit& f, const std::vector<double>& coef, const std::vector<double>& se,
                const std::vector<double>& p, const std::vector<double>& df, double tol = 1e-8) {
    EXPECT_TRUE(f.converged);
    ASSERT_EQ(f.coefficients.size(),coef.size());
    EXPECT_DOUBLE_EQ(f.df_residual,df[0]);
    for (std::size_t a = 0; a < coef.size(); ++a) {
        EXPECT_NEAR(f.coefficients[a],coef[a],1e-8*std::max(1.0,std::abs(coef[a]))) << a;
        EXPECT_NEAR(f.standard_errors[a],se[a],tol*se[a]) << a;
        EXPECT_NEAR(f.p_values[a],p[a],tol) << a;
    }
}

SurveyDesign stratified() {
    return {.weights = s_w, .strata = ids(s_strata), .population_psus = s_fpc};
}

// References: svyglm gaussian on apistrat, glm with rescaled weights, regTermTest Wald and a domain fit.
TEST(SurveyGlm, GaussianMatchesSvyglm) {
    const auto f = fit_survey_glm(stratified(),columns({s_ell,s_meals,s_mobility}),s_api00);
    expect_fit(f,ref_gauss_coef,ref_gauss_se,ref_gauss_p,ref_gauss_df);
    for (std::size_t a = 0; a < 4; ++a)
        EXPECT_NEAR(f.model_standard_errors[a],ref_gauss_model_se[a],1e-8*ref_gauss_model_se[a]) << a;
    const auto w = survey_glm_wald_test(f,{1,2});
    EXPECT_NEAR(w.statistic,ref_wald[0],1e-8*ref_wald[0]);
    EXPECT_DOUBLE_EQ(w.df1,ref_wald[1]);
    EXPECT_DOUBLE_EQ(w.df2,ref_wald[2]);
    EXPECT_NEAR(w.p_value,ref_wald[3],1e-10);
    std::vector<bool> elementary;
    for (const double e : s_elem)
        elementary.push_back(e == 1);
    expect_fit(fit_survey_glm(stratified(),columns({s_ell,s_meals}),s_api00,{.domain = elementary}),ref_domain_coef,
               ref_domain_se,ref_domain_p,ref_domain_df);
}

// References: svyglm quasibinomial and quasipoisson on apiclus1.
TEST(SurveyGlm, LogisticAndPoissonMatchSvyglm) {
    const SurveyDesign d{.weights = c_w, .clusters = ids(c_psu), .population_psus = c_fpc};
    expect_fit(fit_survey_glm(d,columns({c_ell,c_meals,c_mobility}),c_schwide,{.family = SurveyGlmFamily::binomial}),
               ref_logit_coef,ref_logit_se,ref_logit_p,ref_logit_df,1e-5);
    std::vector<double> high, middle;
    for (const double s : c_stype) {
        high.push_back(s == 2);
        middle.push_back(s == 3);
    }
    expect_fit(fit_survey_glm(d,columns({high,middle,c_meals}),c_enroll,{.family = SurveyGlmFamily::poisson}),
               ref_pois_coef,ref_pois_se,ref_pois_p,ref_pois_df,1e-5);
}

TEST(SurveyGlm, Validates) {
    const SurveyDesign d{.weights = {1,1,1}};
    EXPECT_THROW((void)fit_survey_glm(d,Matrix(3,1,1.0),{0,2,1},{.family = SurveyGlmFamily::binomial}),
                 std::invalid_argument);
    EXPECT_THROW((void)fit_survey_glm(d,Matrix(3,1,1.0),{0,-1,1},{.family = SurveyGlmFamily::poisson}),
                 std::invalid_argument);
    EXPECT_THROW((void)fit_survey_glm(d,Matrix(3,2,1.0),{0,1,2}),std::runtime_error); // collinear
    SurveyGlmFit f;
    f.coefficients = {1};
    f.covariance = Matrix(1,1,1.0);
    f.df_residual = 5;
    EXPECT_THROW((void)survey_glm_wald_test(f,{0,0}),std::invalid_argument);
}

} // namespace
