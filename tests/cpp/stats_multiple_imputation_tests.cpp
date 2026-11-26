#include <datamunge/stats/multiple_imputation.hpp>

#include <datamunge/random/random.hpp>
#include <datamunge/random/variates.hpp>

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
using datamunge::random::SplitMix64;

#include "stats_multiple_imputation_data.inc"

const double na = std::numeric_limits<double>::quiet_NaN();

RubinPooling pooled(double dfc) {
    const std::vector<std::vector<double>> est{est_1,est_2,est_3,est_4,est_5};
    std::vector<Matrix> cov;
    for (const auto* v : {&vcov_1,&vcov_2,&vcov_3,&vcov_4,&vcov_5}) {
        Matrix c(3,3);
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t b = 0; b < 3; ++b)
                c(a,b) = (*v)[a+3*b];
        cov.push_back(c);
    }
    return pool_rubin(est,cov,dfc);
}

// References: mice pool() and summary(pool()) of lm(chl ~ age + bmi) on 5 imputations of nhanes.
TEST(MultipleImputation, PoolingMatchesMice) {
    const auto r = pooled(dfcom[0]);
    for (std::size_t a = 0; a < 3; ++a) {
        const auto& e = r.parameters[a];
        EXPECT_NEAR(e.estimate,ref_estimate[a],1e-10*std::abs(ref_estimate[a])) << a;
        EXPECT_NEAR(e.within,ref_ubar[a],1e-10*ref_ubar[a]) << a;
        EXPECT_NEAR(e.between,ref_b[a],1e-10*ref_b[a]) << a;
        EXPECT_NEAR(e.total,ref_t[a],1e-10*ref_t[a]) << a;
        EXPECT_NEAR(e.df,ref_df[a],1e-9*ref_df[a]) << a;
        EXPECT_NEAR(e.riv,ref_riv[a],1e-10*ref_riv[a]) << a;
        EXPECT_NEAR(e.lambda,ref_lambda[a],1e-10) << a;
        EXPECT_NEAR(e.fmi,ref_fmi[a],1e-10) << a;
        EXPECT_NEAR(e.p_value,ref_p[a],1e-9) << a;
        EXPECT_NEAR(e.lower,ref_lower[a],1e-8*std::abs(ref_lower[a])) << a;
        EXPECT_NEAR(e.upper,ref_upper[a],1e-8*std::abs(ref_upper[a])) << a;
    }
    const auto inf = pooled(std::numeric_limits<double>::infinity());
    for (std::size_t a = 0; a < 3; ++a)
        EXPECT_NEAR(inf.parameters[a].df,ref_df_inf[a],1e-9*ref_df_inf[a]) << a;
}

// References: mice D1() (Reiter df with dfcom) and mitml testModels D1 (large sample).
TEST(MultipleImputation, D1MatchesMiceAndMitml) {
    for (const bool small : {true,false}) {
        const auto& ref = small ? ref_d1 : ref_d1_inf;
        const auto t = pooled_wald_test(pooled(small ? dfcom[0] : std::numeric_limits<double>::infinity()),{1,2});
        EXPECT_NEAR(t.statistic,ref[0],1e-9*ref[0]);
        EXPECT_DOUBLE_EQ(t.df1,ref[1]);
        EXPECT_NEAR(t.df2,ref[2],1e-9*ref[2]);
        EXPECT_NEAR(t.p_value,ref[3],1e-9);
        EXPECT_NEAR(t.riv,ref[4],1e-10*ref[4]);
    }
}

TEST(MultipleImputation, PmmImputesObservedValuesAndKeepsObservedData) {
    SplitMix64 rng(5);
    Matrix x(200,3);
    std::set<double> observed;
    for (std::size_t i = 0; i < 200; ++i) {
        x(i,0) = rng.normal();
        x(i,1) = std::round(10*(x(i,0)+rng.normal()))/10;
        x(i,2) = rng.uniform01() < 0.3 ? na : x(i,0)+x(i,1);
        if (rng.uniform01() < 0.2)
            x(i,1) = na;
        if (!std::isnan(x(i,1)))
            observed.insert(x(i,1));
    }
    const auto r = mice_impute(x,rng,{.imputations = 3,.iterations = 5});
    ASSERT_EQ(r.completed.size(),3u);
    EXPECT_EQ(r.incomplete_columns,(std::vector<std::size_t>{1,2}));
    for (const auto& c : r.completed)
        for (std::size_t i = 0; i < 200; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                ASSERT_FALSE(std::isnan(c(i,j)));
                if (!std::isnan(x(i,j)))
                    EXPECT_EQ(c(i,j),x(i,j));
                else if (j == 1)
                    EXPECT_TRUE(observed.contains(c(i,j)));
            }
    SplitMix64 a(9), b(9);
    EXPECT_EQ(mice_impute(x,a,{.imputations = 2}).completed[1](0,2),mice_impute(x,b,{.imputations = 2}).completed[1](0,2));
}

// Bivariate normal, y missing at random given x: pooled MI mean of y is (nearly) unbiased with near-nominal coverage,
// while the complete-case mean is biased. A shift moves the imputed values by delta.
TEST(MultipleImputation, ProperImputationUnderMar) {
    SplitMix64 rng(2024);
    double mi_bias[2]{}, cc_bias = 0, cover = 0, shifted = 0;
    const int reps = 150;
    for (int r = 0; r < reps; ++r) {
        Matrix x(300,2);
        for (std::size_t i = 0; i < 300; ++i) {
            x(i,0) = rng.normal();
            const double y = 0.7*x(i,0)+std::sqrt(1-0.49)*rng.normal();
            x(i,1) = rng.uniform01() < 1/(1+std::exp(-(-0.5+1.5*x(i,0)))) ? na : y;
        }
        for (const int k : {0,1}) {
            const auto method = k == 0 ? ImputationMethod::pmm : ImputationMethod::norm;
            const auto imp = mice_impute(x,rng,{.imputations = 10,.iterations = 5,
                                                .methods = {method,method}});
            std::vector<std::vector<double>> est;
            std::vector<Matrix> var;
            for (const auto& c : imp.completed) {
                double s = 0, s2 = 0;
                for (std::size_t i = 0; i < 300; ++i)
                    s += c(i,1);
                const double m = s/300;
                for (std::size_t i = 0; i < 300; ++i)
                    s2 += (c(i,1)-m)*(c(i,1)-m);
                est.push_back({m});
                var.push_back(Matrix(1,1,s2/299/300));
            }
            const auto p = pool_rubin(est,var,299);
            mi_bias[k] += p.parameters[0].estimate/reps;
            cover += (p.parameters[0].lower <= 0 && 0 <= p.parameters[0].upper)/(2.0*reps);
        }
        double s = 0, c = 0;
        for (std::size_t i = 0; i < 300; ++i)
            if (!std::isnan(x(i,1))) {
                s += x(i,1);
                c += 1;
            }
        cc_bias += s/c/reps;
        if (r < 20) {
            const auto a = mice_impute(x,rng,{.imputations = 1,.methods = {ImputationMethod::norm,ImputationMethod::norm}});
            const auto b = mice_impute(x,rng,{.imputations = 1,.methods = {ImputationMethod::norm,ImputationMethod::norm},
                                              .shift = {0,1}});
            double da = 0, db = 0, count = 0;
            for (std::size_t i = 0; i < 300; ++i)
                if (std::isnan(x(i,1))) {
                    da += a.completed[0](i,1);
                    db += b.completed[0](i,1);
                    count += 1;
                }
            shifted += (db-da)/count/20;
        }
    }
    // Bayesian normal imputation is unbiased; PMM is slightly biased because missingness thins the
    // donors at high x (about -0.024 in a 400-replicate study), far less than complete cases.
    EXPECT_LT(std::abs(mi_bias[1]),0.02);
    EXPECT_LT(std::abs(mi_bias[0]),0.05);
    EXPECT_LT(cc_bias,-0.2);
    EXPECT_GT(cover,0.9);
    EXPECT_NEAR(shifted,1,0.1);
}

// With one incomplete outcome and common random numbers, the shift moves the pooled mean exactly by
// delta times the share of shifted missing values; the tipping and null deltas interpolate the grid.
TEST(MultipleImputation, TippingPointShiftsExactlyAndInterpolates) {
    SplitMix64 rng(31);
    const std::size_t n = 120;
    Matrix x(n,3);
    std::vector<bool> arm(n);
    double shifted_missing = 0;
    for (std::size_t i = 0; i < n; ++i) {
        arm[i] = i % 2 == 0;
        x(i,0) = arm[i];
        x(i,1) = rng.normal();
        const double y = 0.8+0.5*x(i,1)+0.5*rng.normal()+0.6*x(i,0);
        x(i,2) = rng.uniform01() < 0.3 ? na : y;
        shifted_missing += arm[i] && std::isnan(x(i,2));
    }
    // Analysis: mean difference between arms, with its variance.
    const ImputedAnalysis diff = [](const Matrix& c) {
        double s[2]{}, ss[2]{}, k[2]{};
        for (std::size_t i = 0; i < c.rows(); ++i) {
            const int g = c(i,0) > 0.5;
            s[g] += c(i,2);
            ss[g] += c(i,2)*c(i,2);
            k[g] += 1;
        }
        double v = 0;
        for (int g = 0; g < 2; ++g) {
            const double m = s[g]/k[g];
            v += (ss[g]/k[g]-m*m)*k[g]/(k[g]-1)/k[g];
        }
        return std::pair{std::vector<double>{s[1]/k[1]-s[0]/k[0]},Matrix(1,1,v)};
    };
    std::vector<double> deltas;
    for (double d = 0; d >= -4.01; d -= 0.5)
        deltas.push_back(d);
    for (const auto method : {ImputationMethod::pmm,ImputationMethod::norm}) {
        const auto r = tipping_point_analysis(x,diff,{.column = 2,.shifted_rows = arm,.deltas = deltas,
                                                      .mice = {.imputations = 10,.methods = {method,method,method}},
                                                      .seed = 7});
        ASSERT_EQ(r.rows.size(),deltas.size());
        const double treated = static_cast<double>(n/2);
        for (std::size_t k = 1; k < deltas.size(); ++k) {
            const double expected = r.rows[0].result.estimate+deltas[k]*shifted_missing/treated;
            EXPECT_NEAR(r.rows[k].result.estimate,expected,1e-10) << k;
        }
        EXPECT_LT(r.rows[0].result.p_value,0.05);
        ASSERT_TRUE(r.tipping_delta.has_value());
        ASSERT_TRUE(r.null_delta.has_value());
        EXPECT_LT(*r.null_delta,*r.tipping_delta);
        // Interpolation brackets: the tipping delta lies between grid points with p below and above alpha.
        for (std::size_t k = 1; k < deltas.size(); ++k)
            if (r.rows[k-1].result.p_value < 0.05 && r.rows[k].result.p_value >= 0.05) {
                EXPECT_LE(*r.tipping_delta,deltas[k-1]);
                EXPECT_GE(*r.tipping_delta,deltas[k]);
            }
        // Linear estimate: the null delta is exact.
        EXPECT_NEAR(*r.null_delta,-r.rows[0].result.estimate*treated/shifted_missing,1e-9);
    }
    EXPECT_THROW((void)tipping_point_analysis(x,diff,{.column = 3,.deltas = {0}}),std::invalid_argument);
}

TEST(MultipleImputation, VariatesAndValidation) {
    SplitMix64 rng(3);
    double s = 0;
    for (int i = 0; i < 20000; ++i)
        s += datamunge::random::chi_squared_variate(rng,3.5)/20000;
    EXPECT_NEAR(s,3.5,0.1);
    EXPECT_THROW((void)datamunge::random::gamma_variate(rng,0),std::invalid_argument);
    EXPECT_THROW((void)mice_impute(Matrix(2,1,na),rng),std::invalid_argument);
    EXPECT_THROW((void)pool_rubin({{1}},{Matrix(1,1,1.0)}),std::invalid_argument);
    EXPECT_THROW((void)pooled_wald_test(pooled(10),{3}),std::invalid_argument);
}

} // namespace
