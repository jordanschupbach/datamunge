#include <datamunge/stats/cointegration.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
using Rows = std::vector<std::vector<double>>;

struct JohansenReference {
    JohansenDeterministic deterministic;
    std::size_t lags;
    std::vector<double> eigenvalues;
    Rows eigenvectors;
    std::vector<double> trace, max_eigenvalue;
    Rows trace_critical, max_critical;
};

struct VecmReference {
    VecmTerm constant, trend;
    std::size_t rank;
    Rows alpha, beta, gamma, det, det_coint, sigma;
    double llf;
    Rows se_alpha, se_beta, se_gamma, se_det, se_det_coint, forecast;
};

#include "stats_cointegration_data.inc"

Matrix macro() {
    Matrix y(macro_rows,3);
    for (std::size_t i = 0; i < macro_rows; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            y(i,j) = macro_levels[3*i+j];
    return y;
}

void expect_matrix(const Matrix& got, const Rows& want, double tol, bool transposed = false) {
    const std::size_t rows = transposed ? (want.empty() ? 0 : want[0].size()) : want.size();
    ASSERT_EQ(got.rows()*got.cols() == 0,rows == 0 || want.empty() || want[0].empty());
    for (std::size_t i = 0; i < want.size(); ++i)
        for (std::size_t j = 0; j < want[i].size(); ++j)
            EXPECT_NEAR(transposed ? got(j,i) : got(i,j),want[i][j],tol*std::max(1.0,std::abs(want[i][j])))
                << i << "," << j;
}

TEST(Johansen, MatchesStatsmodels) {
    for (const auto& ref : johansen_refs) {
        const auto j = johansen_test(macro(),{.deterministic=ref.deterministic,.lagged_differences=ref.lags});
        for (std::size_t i = 0; i < 3; ++i) {
            // Uncentered log levels (deterministic none) are ill-conditioned: agreement is about 1e-10.
            EXPECT_NEAR(j.eigenvalues[i],ref.eigenvalues[i],1e-9);
            EXPECT_NEAR(j.trace[i],ref.trace[i],1e-7*std::max(1.0,ref.trace[i]));
            EXPECT_NEAR(j.max_eigenvalue[i],ref.max_eigenvalue[i],1e-7*std::max(1.0,ref.max_eigenvalue[i]));
            for (std::size_t q = 0; q < 3; ++q) {
                EXPECT_EQ(j.trace_critical[i][q],ref.trace_critical[i][q]);
                EXPECT_EQ(j.max_eigenvalue_critical[i][q],ref.max_critical[i][q]);
            }
            // Columns agree up to sign (statsmodels fixes only one global sign).
            const double s = (j.eigenvectors(0,i) > 0) == (ref.eigenvectors[0][i] > 0) ? 1.0 : -1.0;
            for (std::size_t a = 0; a < 3; ++a)
                EXPECT_NEAR(s*j.eigenvectors(a,i),ref.eigenvectors[a][i],1e-7*std::max(1.0,std::abs(ref.eigenvectors[a][i])));
        }
    }
    const auto j = johansen_test(macro());
    EXPECT_EQ(j.trace_rank,0U); // 28.87 is below the 5% value 29.80 (it exceeds the 10% value 27.07)
    EXPECT_EQ(j.observations,201U);
}

TEST(Vecm, MatchesStatsmodels) {
    for (const auto& ref : vecm_refs) {
        SCOPED_TRACE(testing::Message() << "constant " << int(ref.constant) << " trend " << int(ref.trend) << " rank " << ref.rank);
        const auto m = fit_vecm(macro(),{.rank=ref.rank,.lagged_differences=2,.constant=ref.constant,.trend=ref.trend});
        expect_matrix(m.alpha,ref.alpha,1e-7);
        expect_matrix(m.beta,ref.beta,1e-7);
        expect_matrix(m.gamma,ref.gamma,1e-7);
        expect_matrix(m.deterministic,ref.det,1e-7);
        expect_matrix(m.beta_deterministic,ref.det_coint,1e-7);
        expect_matrix(m.sigma,ref.sigma,1e-9);
        EXPECT_NEAR(m.log_likelihood,ref.llf,1e-7);
        expect_matrix(m.alpha_standard_errors,ref.se_alpha,1e-7);
        expect_matrix(m.gamma_standard_errors,ref.se_gamma,1e-7);
        expect_matrix(m.deterministic_standard_errors,ref.se_det,1e-7);
        // statsmodels reshapes vec(beta_2') column-major, which permutes the entries when more than one
        // free row and more than one relation are present; compare the sorted values in that case.
        std::vector<double> ours, theirs;
        for (std::size_t i = ref.rank; i < 3; ++i)
            for (std::size_t q = 0; q < ref.rank; ++q) {
                ours.push_back(m.beta_standard_errors(i,q));
                theirs.push_back(ref.se_beta[i][q]);
            }
        for (std::size_t i = 0; i < m.beta_deterministic.rows(); ++i)
            for (std::size_t q = 0; q < ref.rank; ++q) {
                ours.push_back(m.beta_deterministic_standard_errors(i,q));
                theirs.push_back(ref.se_det_coint[i][q]);
            }
        if (ref.rank > 1 && ours.size() > ref.rank) {
            std::ranges::sort(ours);
            std::ranges::sort(theirs);
        }
        for (std::size_t i = 0; i < ours.size(); ++i)
            EXPECT_NEAR(ours[i],theirs[i],1e-6*std::max(1.0,theirs[i]));
        expect_matrix(forecast_vecm(m,4),ref.forecast,1e-9);
    }
}

TEST(Vecm, ValidationAndRepresentation) {
    const auto m = fit_vecm(macro(),{.rank=1,.lagged_differences=2});
    ASSERT_EQ(m.var_coefficients.size(),3U);
    EXPECT_EQ(m.beta(0,0),1.0);
    EXPECT_EQ(m.observations,200U);
    EXPECT_THROW((void)fit_vecm(macro(),{.rank=0}),std::invalid_argument);
    EXPECT_THROW((void)fit_vecm(macro(),{.rank=3}),std::invalid_argument);
    EXPECT_THROW((void)johansen_test(Matrix(10,1)),std::invalid_argument);
    EXPECT_THROW((void)forecast_vecm(m,0),std::invalid_argument);
}

} // namespace
