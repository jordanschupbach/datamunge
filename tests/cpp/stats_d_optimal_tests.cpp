#include <datamunge/stats/d_optimal.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

// Full quadratic model on a levels^k grid with the first factor fastest (as gen.factorial).
Matrix quadratic_candidates(std::size_t k, const std::vector<double>& levels) {
    std::size_t n = 1;
    for (std::size_t j = 0; j < k; ++j)
        n *= levels.size();
    const std::size_t p = (k+1)*(k+2)/2;
    Matrix f(n,p);
    for (std::size_t r = 0; r < n; ++r) {
        std::vector<double> x(k);
        std::size_t rest = r;
        for (std::size_t j = 0; j < k; ++j) {
            x[j] = levels[rest%levels.size()];
            rest /= levels.size();
        }
        std::size_t c = 0;
        f(r,c++) = 1;
        for (std::size_t j = 0; j < k; ++j)
            f(r,c++) = x[j];
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = i+1; j < k; ++j)
                f(r,c++) = x[i]*x[j];
        for (std::size_t j = 0; j < k; ++j)
            f(r,c++) = x[j]*x[j];
    }
    return f;
}

// Reference values: AlgDesign 1.2 optFederov(~quad(A, B), gen.factorial(3, 2), nTrials = 6), best D over
// 20 calls with nRepeats = 20, in R 4.5.3; D, A and Ge recomputed from the selected rows.
TEST(DOptimal, SixRunQuadraticMatchesAlgDesign) {
    const auto f = quadratic_candidates(2,{-1,0,1});
    const auto d = d_optimal_design(f,6);
    EXPECT_NEAR(d.criteria.d_criterion,.419973683298291,1e-12);
    EXPECT_NEAR(std::exp(d.criteria.log_det),256,1e-9);
    EXPECT_NEAR(d.criteria.a_criterion,6.5,1e-10);
    EXPECT_NEAR(d.criteria.g_efficiency,.363636363636363,1e-12);
    EXPECT_TRUE(d.criteria.estimable);
    EXPECT_EQ(d.rows.size(),6U);
    // All nine points: AlgDesign reports D = 0.46224084956709, A = 3.20833333333333.
    const auto all = design_criteria(f,f);
    EXPECT_NEAR(all.d_criterion,.46224084956709,1e-12);
    EXPECT_NEAR(all.a_criterion,3.20833333333333,1e-12);
}

TEST(DOptimal, LargerCandidateSetsMatchAlgDesignOptimum) {
    // 3^3 grid, 14 runs: AlgDesign best D = 0.463044741912218, log|M| = 18.6912573485012.
    const auto f3 = quadratic_candidates(3,{-1,0,1});
    const auto d3 = d_optimal_design(f3,14,{.starts=30});
    EXPECT_NEAR(d3.criteria.log_det,18.6912573485012,1e-9);
    EXPECT_NEAR(d3.criteria.d_criterion,.463044741912218,1e-12);
    // 5^2 grid on -2..2, 8 runs: AlgDesign best D = 2.88449914061482. Several 8-run designs share this
    // determinant with different A values, so only D is compared.
    const auto f5 = quadratic_candidates(2,{-2,-1,0,1,2});
    const auto d5 = d_optimal_design(f5,8);
    EXPECT_NEAR(d5.criteria.d_criterion,2.88449914061482,1e-12);
}

TEST(DOptimal, QuadraticInOneFactorAndRepeats) {
    std::vector<double> values;
    for (int i = 0; i <= 20; ++i) {
        const double x = -1+.1*i;
        values.insert(values.end(),{1,x,x*x});
    }
    const Matrix f(21,3,values);
    const auto d = d_optimal_design(f,3);
    EXPECT_EQ(d.rows,(std::vector<std::size_t>{0,10,20}));
    EXPECT_NEAR(d.criteria.g_efficiency,1.0,1e-12); // Kiefer–Wolfowitz: D-optimal is G-optimal here.
    // Six runs with repeats: two at each of -1, 0, 1.
    const auto r = d_optimal_design(f,6,{.allow_repeats=true});
    EXPECT_EQ(r.rows,(std::vector<std::size_t>{0,0,10,10,20,20}));
}

TEST(DOptimal, ReportsNonestimableDesignsAndValidates) {
    const auto f = quadratic_candidates(2,{-1,0,1});
    Matrix corners(4,6);
    for (std::size_t i = 0, r = 0; r < 9; ++r)
        if (r == 0 || r == 2 || r == 6 || r == 8) {
            for (std::size_t c = 0; c < 6; ++c)
                corners(i,c) = f(r,c);
            ++i;
        }
    const auto c = design_criteria(corners,f);
    EXPECT_FALSE(c.estimable);
    EXPECT_EQ(c.rank,4U);
    EXPECT_TRUE(std::isinf(c.a_criterion));
    EXPECT_EQ(c.d_criterion,0.0);
    EXPECT_THROW((void)d_optimal_design(f,5),std::invalid_argument);
    EXPECT_THROW((void)d_optimal_design(f,10),std::invalid_argument);
    EXPECT_THROW((void)d_optimal_design(f,6,{.starts=0}),std::invalid_argument);
    EXPECT_THROW((void)d_optimal_design(corners,6,{.allow_repeats=true}),std::invalid_argument);
    EXPECT_THROW((void)design_criteria(corners,Matrix(2,3)),std::invalid_argument);
}

} // namespace
