#include <datamunge/stats/markov_switching.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_markov_switching_data.inc"

// Reference parameters (statsmodels order) unpacked to transition, coefficients and variances.
struct Reference {
    Matrix p, beta;
    std::vector<double> var;
};

Reference unpack(const std::vector<double>& v, std::size_t k, std::size_t cols, bool shared) {
    Reference r{Matrix(k,k),Matrix(k,cols),std::vector<double>(k)};
    std::size_t idx = 0;
    for (std::size_t j = 0; j+1 < k; ++j)
        for (std::size_t i = 0; i < k; ++i)
            r.p(i,j) = v[idx++];
    for (std::size_t i = 0; i < k; ++i) {
        double s = 0;
        for (std::size_t j = 0; j+1 < k; ++j)
            s += r.p(i,j);
        r.p(i,k-1) = 1-s;
    }
    for (std::size_t c = 0; c < cols; ++c)
        for (std::size_t s = 0; s < k; ++s)
            r.beta(s,c) = v[idx++];
    for (std::size_t s = 0; s < k; ++s)
        r.var[s] = shared ? v[idx] : v[idx+s];
    return r;
}

// Checks a fit against statsmodels allowing a relabelling of regimes (matched on the constant).
void expect_matches(const MarkovSwitchingFit& f, const std::vector<double>& params, const std::vector<double>& bse,
                    const std::vector<double>& llf, const std::vector<double>& smoothed,
                    const std::vector<double>& filtered, const std::vector<double>& durations, std::size_t k,
                    bool shared) {
    const std::size_t cols = f.coefficients.cols(), n = f.smoothed_probabilities.rows();
    const auto ref = unpack(params,k,cols,shared);
    std::vector<std::size_t> perm(k);
    for (std::size_t s = 0; s < k; ++s) {
        std::size_t best = 0;
        for (std::size_t r = 1; r < k; ++r)
            if (std::abs(ref.beta(r,0)-f.coefficients(s,0)) < std::abs(ref.beta(best,0)-f.coefficients(s,0)))
                best = r;
        perm[s] = best;
    }
    const bool identity = std::ranges::equal(perm,std::views::iota(std::size_t{0},k));
    EXPECT_GE(f.log_likelihood,llf[0]-1e-7);
    EXPECT_NEAR(f.log_likelihood,llf[0],1e-6);
    EXPECT_NEAR(f.aic,llf[1],2e-6);
    EXPECT_NEAR(f.bic,llf[2],2e-6);
    for (std::size_t s = 0; s < k; ++s) {
        for (std::size_t r = 0; r < k; ++r)
            EXPECT_NEAR(f.transition(s,r),ref.p(perm[s],perm[r]),2e-6) << s << "->" << r;
        for (std::size_t c = 0; c < cols; ++c)
            EXPECT_NEAR(f.coefficients(s,c),ref.beta(perm[s],c),2e-6) << s << "," << c;
        EXPECT_NEAR(f.variances[s],ref.var[perm[s]],2e-6*ref.var[perm[s]]);
        EXPECT_NEAR(f.expected_durations[s],durations[perm[s]],1e-3*durations[perm[s]]);
        for (std::size_t t = 0; t < n; ++t) {
            EXPECT_NEAR(f.smoothed_probabilities(t,s),smoothed[t*k+perm[s]],2e-6) << t;
            EXPECT_NEAR(f.filtered_probabilities(t,s),filtered[t*k+perm[s]],2e-6) << t;
        }
    }
    // Standard errors are compared when the labels agree (they are listed in statsmodels' order).
    if (identity) {
        ASSERT_EQ(f.standard_errors.size(),bse.size());
        for (std::size_t i = 0; i < bse.size(); ++i)
            EXPECT_NEAR(f.standard_errors[i],bse[i],5e-4*bse[i]) << i;
    } else {
        ADD_FAILURE() << "regime labels differ from statsmodels";
    }
}

// References: statsmodels 0.14.6 MarkovRegression, refitted to a gradient below 2e-6.
TEST(MarkovSwitching, RegressionWithSwitchingVarianceMatchesStatsmodels) {
    Matrix x(a_x.size(),1);
    for (std::size_t t = 0; t < a_x.size(); ++t)
        x(t,0) = a_x[t];
    const auto f = fit_markov_switching(a_y,x);
    EXPECT_TRUE(f.converged);
    expect_matches(f,a_params,a_bse,a_llf,a_smoothed,a_filtered,a_durations,2,false);
}

TEST(MarkovSwitching, SharedVarianceMatchesStatsmodels) {
    const auto f = fit_markov_switching(b_y,Matrix(b_y.size(),0),{.switching_variance=false});
    expect_matches(f,b_params,b_bse,b_llf,b_smoothed,b_filtered,b_durations,2,true);
    EXPECT_DOUBLE_EQ(f.variances[0],f.variances[1]);
}

TEST(MarkovSwitching, ThreeRegimesMatchStatsmodels) {
    const auto f = fit_markov_switching(c_y,Matrix(c_y.size(),0),{.regimes=3});
    expect_matches(f,c_params,c_bse,c_llf,c_smoothed,c_filtered,c_durations,3,false);
    double total = 0;
    for (const double v : f.steady_state)
        total += v;
    EXPECT_NEAR(total,1,1e-12);
}

TEST(MarkovSwitching, Validation) {
    EXPECT_THROW((void)fit_markov_switching(b_y,Matrix(b_y.size(),0),{.regimes=1}),std::invalid_argument);
    EXPECT_THROW((void)fit_markov_switching(b_y,Matrix(5,0)),std::invalid_argument);
    EXPECT_THROW((void)fit_markov_switching(b_y,Matrix(b_y.size(),0),{.intercept=false}),std::invalid_argument);
    EXPECT_THROW((void)fit_markov_switching({1,2,3,4,5},Matrix(5,0)),std::invalid_argument);
    auto bad = b_y;
    bad[3] = NAN;
    EXPECT_THROW((void)fit_markov_switching(bad,Matrix(bad.size(),0)),std::invalid_argument);
}

} // namespace
