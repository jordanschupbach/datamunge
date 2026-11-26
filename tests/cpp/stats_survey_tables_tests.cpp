#include <datamunge/stats/survey_tables.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_survey_tables_data.inc"

std::vector<std::size_t> labels(const std::vector<double>& x) {
    return {x.begin(),x.end()};
}

std::vector<int> ids(const std::vector<double>& x) {
    return {x.begin(),x.end()};
}

// ref layout per statistic (Chisq, F, Wald, adjWald): statistic, df1, df2, p.
void expect_matches(const SurveyTableTests& t, const std::vector<double>& ref) {
    const SurveyTableTest* tests[4]{&t.first_order,&t.second_order,&t.wald,&t.adjusted_wald};
    for (int s = 0; s < 4; ++s) {
        const auto& x = *tests[s];
        const double* r = &ref[4*s];
        // svychisq "Chisq" reports the unadjusted X^2 and applies the correction only in its p-value.
        const double stat = s == 0 ? t.pearson : x.statistic;
        EXPECT_NEAR(stat,r[0],1e-8*std::max(1.0,r[0])) << s;
        EXPECT_NEAR(x.df1,r[1],1e-8*r[1]) << s;
        if (std::isinf(r[2]))
            EXPECT_TRUE(std::isinf(x.df2)) << s;
        else
            EXPECT_NEAR(x.df2,r[2],1e-8*r[2]) << s;
        EXPECT_NEAR(x.p_value,r[3],1e-9) << s;
    }
}

// Reference: survey svychisq on apiclus1 (one-stage cluster sample with fpc), 2 x 3 table.
TEST(SurveyTables, MatchesSvychisqClusterDesign) {
    SurveyDesign d{.weights = c1_w, .clusters = ids(c1_psu), .population_psus = c1_fpc};
    const auto t = survey_table_test(d,labels(c1_row),labels(c1_col));
    EXPECT_EQ(t.rows,2u);
    EXPECT_EQ(t.cols,3u);
    EXPECT_DOUBLE_EQ(t.design_df,14);
    expect_matches(t,ref_c1);
    double mean = 0;
    for (const double e : t.design_effects)
        mean += e/t.design_effects.size();
    EXPECT_NEAR(t.first_order.statistic,t.pearson/mean,1e-10*t.pearson);
}

// Reference: svychisq on apistrat (stratified with fpc), two 3 x 3 tables.
TEST(SurveyTables, MatchesSvychisqStratifiedDesign) {
    SurveyDesign d{.weights = s_w, .strata = ids(s_strata), .population_psus = s_fpc};
    expect_matches(survey_table_test(d,labels(s_row),labels(s_col)),ref_s1);
    expect_matches(survey_table_test(d,labels(s_row2),labels(s_col)),ref_s2);
}

TEST(SurveyTables, Validates) {
    SurveyDesign d{.weights = {1,1,1,1}};
    EXPECT_THROW((void)survey_table_test(d,{0,0,0,0},{0,1,0,1}),std::invalid_argument);
    EXPECT_THROW((void)survey_table_test(d,{0,1},{0,1}),std::invalid_argument);
}

} // namespace
