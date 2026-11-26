#include <datamunge/stats/survey.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_survey_data.inc"

void expect_estimate(const SurveyEstimate& e, const std::vector<double>& ref) {
    EXPECT_NEAR(e.estimate,ref[0],1e-9*std::abs(ref[0]));
    EXPECT_NEAR(e.standard_error,ref[1],1e-9*ref[1]);
    if (ref.size() == 4) {
        EXPECT_NEAR(e.lower,ref[2],1e-9*std::abs(ref[2]));
        EXPECT_NEAR(e.upper,ref[3],1e-9*std::abs(ref[3]));
    }
}

// References: survey svytotal/svymean with svydesign(ids = ~1, probs = ~p) on apistrat.
TEST(HorvitzThompson, MatchesSurveyPackage) {
    const auto enroll = horvitz_thompson(s_enroll,s_prob);
    expect_estimate(enroll.total,ref_enroll_total);
    EXPECT_NEAR(enroll.estimated_population_size,ref_nhat[0],1e-9);
    const auto api = horvitz_thompson(s_api00,s_prob,{.population_size=6194});
    expect_estimate(api.hajek_mean,ref_api_mean);
    expect_estimate(api.total,ref_api_total);
    ASSERT_TRUE(api.ht_mean.has_value());
    EXPECT_NEAR(api.ht_mean->estimate,ref_api_total[0]/6194,1e-9);
    EXPECT_NEAR(api.ht_mean->standard_error,ref_api_total[1]/6194,1e-12);
    const auto ninety = horvitz_thompson(s_api00,s_prob,{.confidence=.9});
    EXPECT_NEAR(ninety.hajek_mean.lower,ref_api_mean90[0],1e-9*ref_api_mean90[0]);
    EXPECT_NEAR(ninety.hajek_mean.upper,ref_api_mean90[1],1e-9*ref_api_mean90[1]);
    EXPECT_FALSE(ninety.ht_mean.has_value());
}

TEST(HorvitzThompson, EqualProbabilitiesGiveSampleMean) {
    const auto r = horvitz_thompson({1,2,3,6},{.1,.1,.1,.1});
    EXPECT_DOUBLE_EQ(r.hajek_mean.estimate,3);
    EXPECT_DOUBLE_EQ(r.total.estimate,120);
    EXPECT_DOUBLE_EQ(r.estimated_population_size,40);
}

TEST(HorvitzThompson, Validation) {
    EXPECT_THROW((void)horvitz_thompson({1},{.5}),std::invalid_argument);
    EXPECT_THROW((void)horvitz_thompson({1,2},{.5}),std::invalid_argument);
    EXPECT_THROW((void)horvitz_thompson({1,2},{.5,0}),std::invalid_argument);
    EXPECT_THROW((void)horvitz_thompson({1,2},{.5,1.5}),std::invalid_argument);
    EXPECT_THROW((void)horvitz_thompson({1,2},{.5,.5},{.population_size=0}),std::invalid_argument);
    EXPECT_THROW((void)horvitz_thompson({1,2},{.5,.5},{.confidence=1}),std::invalid_argument);
}

} // namespace

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_survey_joint_data.inc"

Matrix joint() {
    const std::size_t n = e_bush.size();
    Matrix j(n,n);
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b)
            j(a,b) = e_joint[b*n+a];
    return j;
}

// References: survey svytotal/svymean with svydesign(pps = ppsmat(election_jointprob), variance = "YG"/"HT").
TEST(HorvitzThompsonJoint, MatchesSurveyPpsmat) {
    const auto j = joint();
    for (const auto method : {SurveyVariance::sen_yates_grundy,SurveyVariance::horvitz_thompson}) {
        const auto& total = method == SurveyVariance::sen_yates_grundy ? ref_YG_total : ref_HT_total;
        const auto& mean = method == SurveyVariance::sen_yates_grundy ? ref_YG_mean : ref_HT_mean;
        const auto bush = horvitz_thompson_joint(e_bush,j,{.method=method});
        const auto kerry = horvitz_thompson_joint(e_kerry,j,{.method=method});
        EXPECT_NEAR(bush.total.estimate,total[0],1e-6);
        EXPECT_NEAR(kerry.total.estimate,total[1],1e-6);
        EXPECT_NEAR(bush.total.standard_error,total[2],1e-8*total[2]);
        EXPECT_NEAR(kerry.total.standard_error,total[3],1e-8*total[3]);
        EXPECT_NEAR(bush.hajek_mean.estimate,mean[0],1e-9*mean[0]);
        EXPECT_NEAR(bush.hajek_mean.standard_error,mean[1],1e-8*mean[1]);
        EXPECT_FALSE(bush.negative_variance);
    }
}

TEST(HorvitzThompsonJoint, SimpleRandomSamplingVariance) {
    // SRS of 3 from 6: pi = 1/2, pi_ij = n(n-1)/(N(N-1)) = 1/5; SYG equals N^2 (1-f) s^2 / n.
    const Matrix j(3,3,{.5,.2,.2,.2,.5,.2,.2,.2,.5});
    const std::vector<double> y{2,5,11};
    const auto r = horvitz_thompson_joint(y,j,{.population_size=6});
    const double s2 = ((2-6.)*(2-6.)+(5-6.)*(5-6.)+(11-6.)*(11-6.))/2;
    EXPECT_NEAR(r.total.standard_error,std::sqrt(36*.5*s2/3),1e-12);
    EXPECT_NEAR(r.ht_mean->standard_error,std::sqrt(.5*s2/3),1e-12);
    const auto h = horvitz_thompson_joint(y,j,{.method=SurveyVariance::horvitz_thompson,.population_size=6});
    EXPECT_NEAR(h.total.standard_error,r.total.standard_error,1e-12); // forms agree under SRS
}

TEST(HorvitzThompsonJoint, Validation) {
    EXPECT_THROW((void)horvitz_thompson_joint({1,2},Matrix(3,3,.5)),std::invalid_argument);
    EXPECT_THROW((void)horvitz_thompson_joint({1,2},Matrix(2,2,{.5,.2,.3,.5})),std::invalid_argument); // asymmetric
    EXPECT_THROW((void)horvitz_thompson_joint({1,2},Matrix(2,2,{.5,0,0,.5})),std::invalid_argument);  // pi_ij = 0
    EXPECT_THROW((void)horvitz_thompson_joint({1,2},Matrix(2,2,{.5,.6,.6,.5})),std::invalid_argument); // pi_ij > pi_i
}

} // namespace

namespace {
using namespace datamunge::stats;

#include "stats_survey_linear_data.inc"

std::vector<int> ints(const std::vector<double>& v) { return {v.begin(),v.end()}; }

void expect_se(const SurveyEstimate& e, const std::vector<double>& ref, std::size_t k = 0, std::size_t m = 1) {
    EXPECT_NEAR(e.estimate,ref[k],1e-9*std::abs(ref[k]));
    EXPECT_NEAR(e.standard_error,ref[m+k],1e-8*ref[m+k]);
}

// References: survey svytotal/svymean/svyratio on apistrat with strata and fpc.
TEST(SurveyLinearization, StratifiedWithFpc) {
    const SurveyDesign d{.weights=st_pw,.strata=ints(st_stype),.population_psus=st_fpc};
    expect_se(survey_total(d,st_enroll),ref_st_total);
    expect_se(survey_mean(d,st_api00),ref_st_mean);
    expect_se(survey_ratio(d,st_apistu,st_enroll),ref_st_ratio);
}

// References: apiclus1 one-stage cluster sample with fpc, including svyby domains.
TEST(SurveyLinearization, ClusterAndDomains) {
    const SurveyDesign d{.weights=c1_pw,.clusters=ints(c1_dnum),.population_psus=c1_fpc};
    expect_se(survey_mean(d,c1_api00),ref_c1_mean);
    expect_se(survey_total(d,c1_enroll),ref_c1_total);
    for (std::size_t k = 0; k < 3; ++k) {
        std::vector<bool> domain;
        for (const double s : c1_stype)
            domain.push_back(static_cast<std::size_t>(s) == k+1);
        expect_se(survey_mean(d,c1_api00,domain),ref_c1_dmean,k,3);
        expect_se(survey_total(d,c1_enroll,domain),ref_c1_dtotal,k,3);
    }
}

// References: two-stage apiclus2 (ultimate cluster), NHANES stratified clusters with a missing-data
// domain, and a lonely PSU treated as certainty.
TEST(SurveyLinearization, MultistageNhanesAndLonelyPsu) {
    expect_se(survey_mean({.weights=c2_pw,.clusters=ints(c2_dnum)},c2_api00),ref_c2_mean);
    std::vector<bool> observed;
    for (const double v : nh_chol)
        observed.push_back(v >= 0);
    const SurveyDesign nh{.weights=nh_wt,.strata=ints(nh_strata),.clusters=ints(nh_psu)};
    expect_se(survey_mean(nh,nh_chol,observed),ref_nh_mean);
    SurveyDesign lonely{.weights=lo_pw,.strata=ints(lo_stype)};
    EXPECT_THROW((void)survey_mean(lonely,lo_api00),std::invalid_argument);
    lonely.lonely_psu = LonelyPsu::certainty;
    expect_se(survey_mean(lonely,lo_api00),ref_lo_mean);
}

TEST(SurveyLinearization, Validation) {
    const SurveyDesign d{.weights={1,2,3},.strata={1,1,2}};
    EXPECT_THROW((void)survey_total({.weights={1,0}},{1,2}),std::invalid_argument);
    EXPECT_THROW((void)survey_total({.weights={1,2}},{1,2,3}),std::invalid_argument);
    EXPECT_THROW((void)survey_total({.weights={1,2},.population_psus={1,1}},{1,2}),std::invalid_argument);
    EXPECT_THROW((void)survey_mean({.weights={1,2}},{1,2},{false,false}),std::invalid_argument);
    EXPECT_THROW((void)survey_ratio({.weights={1,2}},{1,2},{2,-1}),std::invalid_argument);
    EXPECT_THROW((void)survey_total(d,{1,2,3}),std::invalid_argument); // lonely PSU in stratum 2
}

} // namespace

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_survey_replicate_data.inc"

std::vector<int> as_ints(const std::vector<double>& v) { return {v.begin(),v.end()}; }

void expect_rep(const SurveyEstimate& e, const std::vector<double>& ref, std::size_t k = 0, std::size_t m = 1) {
    EXPECT_NEAR(e.estimate,ref[k],1e-9*std::abs(ref[k]));
    EXPECT_NEAR(e.standard_error,ref[m+k],1e-8*ref[m+k]);
}

// Rebuilds survey's replicate weights from its per-PSU factors (rows: PSUs, then replicates).
ReplicateDesign from_factors(const std::vector<double>& strata, const std::vector<double>& psu,
                             const std::vector<double>& factor, double scale) {
    const std::size_t n = r_wt.size(), psus = strata.size(), R = factor.size()/psus;
    ReplicateDesign d{.weights=r_wt,.replicate_weights=Matrix(n,R),.scale=scale};
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t c = 0;
        while (strata[c] != r_strata[i] || psu[c] != r_psu[i])
            ++c;
        for (std::size_t r = 0; r < R; ++r)
            d.replicate_weights(i,r) = r_wt[i]*factor[c*R+r];
    }
    return d;
}

// References: survey as.svrepdesign BRR and Fay (rho 0.3) on NHANES with two PSUs per stratum.
TEST(SurveyReplicate, BrrAndFayMatchSurvey) {
    const SurveyDesign design{.weights=r_wt,.strata=as_ints(r_strata),.clusters=as_ints(r_psu)};
    const auto brr = make_replicate_design(design,{.method=ReplicateMethod::brr});
    const auto fay = make_replicate_design(design,{.method=ReplicateMethod::fay,.fay_rho=.3});
    EXPECT_EQ(brr.replicate_weights.cols(),16u);
    EXPECT_NEAR(brr.scale,brr_scale[0],1e-15);
    EXPECT_NEAR(fay.scale,fay_scale[0],1e-12);
    std::vector<double> gender; // RIAGENDR coded 1 and 2, as in the reference
    for (const double f : r_female)
        gender.push_back(f+1);
    expect_rep(replicate_total(brr,gender),ref_brr_total);
    expect_rep(replicate_total(fay,gender),ref_fay_total);
    expect_rep(survey_total(design,gender),ref_brr_total); // balance makes BRR exact for totals
    std::vector<bool> observed;
    for (const double v : r_chol)
        observed.push_back(v >= 0);
    // Nonlinear statistics depend on the Hadamard matrix: use survey's own replicates.
    expect_rep(replicate_mean(from_factors(brr_factor_strata,brr_factor_psu,brr_factor,brr_scale[0]),r_chol,observed),
               ref_brr_mean);
    expect_rep(replicate_mean(from_factors(fay_factor_strata,fay_factor_psu,fay_factor,fay_scale[0]),r_chol,observed),
               ref_fay_mean);
    auto mse = from_factors(brr_factor_strata,brr_factor_psu,brr_factor,brr_scale[0]);
    mse.mse = true;
    expect_rep(replicate_mean(mse,r_chol,observed),ref_brr_mse_mean);
}

// References: JK1 on apiclus1 and JKn on apistrat, both with finite-population corrections.
TEST(SurveyReplicate, JackknivesMatchSurvey) {
    const auto jk1 = make_replicate_design({.weights=j_pw,.clusters=as_ints(j_dnum),.population_psus=j_fpc},
                                           {.method=ReplicateMethod::jk1});
    EXPECT_EQ(jk1.replicate_weights.cols(),15u);
    expect_rep(replicate_mean(jk1,j_api00),ref_jk1_mean);
    expect_rep(replicate_ratio(jk1,j_apistu,j_enroll),ref_jk1_ratio);
    for (std::size_t k = 0; k < 3; ++k) {
        std::vector<bool> domain;
        for (const double s : j_stype)
            domain.push_back(static_cast<std::size_t>(s) == k+1);
        expect_rep(replicate_mean(jk1,j_api00,domain),ref_jk1_dmean,k,3);
    }
    const auto jkn = make_replicate_design({.weights=n_pw,.strata=as_ints(n_stype),.population_psus=n_fpc});
    EXPECT_EQ(jkn.replicate_weights.cols(),200u);
    expect_rep(replicate_mean(jkn,n_api00),ref_jkn_mean);
    expect_rep(replicate_total(jkn,n_enroll),ref_jkn_total);
}

// Full orthogonal balance: BRR totals reproduce the linearized variance for any number of strata.
TEST(SurveyReplicate, BrrIsBalancedForEveryStratumCount) {
    for (std::size_t strata = 1; strata <= 40; ++strata) {
        SurveyDesign d;
        std::vector<double> y;
        for (std::size_t s = 0; s < strata; ++s)
            for (int c = 0; c < 2; ++c)
                for (int k = 0; k < 2; ++k) {
                    d.weights.push_back(1+static_cast<double>((s*7+c*3+k)%5));
                    d.strata.push_back(static_cast<int>(s));
                    d.clusters.push_back(c);
                    y.push_back(std::sin(static_cast<double>(s*13+c*5+k)));
                }
        const auto brr = make_replicate_design(d,{.method=ReplicateMethod::brr});
        EXPECT_GT(brr.replicate_weights.cols(),strata);
        EXPECT_NEAR(replicate_total(brr,y).standard_error,survey_total(d,y).standard_error,1e-9) << strata;
    }
}

TEST(SurveyReplicate, Validation) {
    const SurveyDesign three{.weights={1,1,1},.strata={1,1,1}};
    EXPECT_THROW((void)make_replicate_design(three,{.method=ReplicateMethod::brr}),std::invalid_argument);
    EXPECT_THROW((void)make_replicate_design({.weights={1,1,1,1},.strata={1,1,2,2}},{.method=ReplicateMethod::jk1}),
                 std::invalid_argument);
    EXPECT_THROW((void)make_replicate_design({.weights={1,1,1},.strata={1,1,2}}),std::invalid_argument);
    EXPECT_THROW((void)make_replicate_design({.weights={1,1}},{.method=ReplicateMethod::fay,.fay_rho=1}),
                 std::invalid_argument);
    const ReplicateDesign bad{.weights={1,1},.replicate_weights=Matrix(3,2)};
    EXPECT_THROW((void)replicate_total(bad,{1,2}),std::invalid_argument);
}

} // namespace
