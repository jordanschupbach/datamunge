#include <datamunge/causal/rosenbaum.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::causal;
using Matrix = datamunge::linalg::DenseMatrix<double>;
const double NaN = std::numeric_limits<double>::quiet_NaN();
#include "causal_rosenbaum_data.inc"

const std::vector<double> gammas{1, 1.5, 2, 3, 4.5};

Matrix rows_of(const std::vector<double>& flat, std::size_t cols) {
    return Matrix(flat.size() / cols, cols, flat);
}

void expect_senmv(const Matrix& y, const MTestOptions& o, const std::vector<double>& ref, const std::string& what) {
    for (std::size_t g = 0; g < gammas.size(); ++g) {
        const auto b = rosenbaum_m_test(y, gammas[g], o);
        const double* r = &ref[4 * g];
        EXPECT_NEAR(b.statistic, r[0], 1e-10 * (1 + std::abs(r[0]))) << what << " gamma " << gammas[g];
        EXPECT_NEAR(b.expectation, r[1], 1e-10 * (1 + std::abs(r[1]))) << what << " gamma " << gammas[g];
        EXPECT_NEAR(b.variance, r[2], 1e-10 * (1 + r[2])) << what << " gamma " << gammas[g];
        EXPECT_NEAR(b.p_upper, r[3], 1e-10) << what << " gamma " << gammas[g];
        EXPECT_LE(b.p_lower, b.p_upper + 1e-15);
    }
}
} // namespace

TEST(Rosenbaum, MTestMatchesSensitivitymv) {
    const Matrix mercury = rows_of(mercury_flat, 3);
    expect_senmv(mercury, {}, ref_mercury_h, "method h");
    expect_senmv(mercury, {.inner = 0.5}, ref_mercury_i, "method i");
    expect_senmv(mercury, {.inner = 0.5, .trim = 3, .scale_quantile = 0.8, .shift = 0.2}, ref_mercury_custom,
                 "custom");
    expect_senmv(mercury, {.trim = std::numeric_limits<double>::infinity()}, ref_mercury_untrimmed, "untrimmed");
    expect_senmv(rows_of(tb_flat, 3), {}, ref_tb, "variable set sizes");
    Matrix pairs(erpcp_welder.size(), 2);
    for (std::size_t i = 0; i < erpcp_welder.size(); ++i) {
        pairs(i, 0) = (erpcp_welder[i] - erpcp_control[i]) / 2;
        pairs(i, 1) = -(erpcp_welder[i] - erpcp_control[i]) / 2;
    }
    expect_senmv(pairs, {}, ref_erpcp_m, "pairs");
}

TEST(Rosenbaum, MeanOfDifferencesIsMethodT) {
    // senmv method "t": untrimmed, unscaled, divided by (set size - 1) and the number of sets.
    const Matrix mercury = rows_of(mercury_flat, 3);
    const double scale = 3.0 / 2.0 / static_cast<double>(mercury.rows());
    for (std::size_t g = 0; g < gammas.size(); ++g) {
        const auto b = rosenbaum_m_test(mercury, gammas[g], {.trim = std::numeric_limits<double>::infinity()});
        EXPECT_NEAR(b.statistic * scale, ref_mercury_t[4 * g], 1e-12);
        EXPECT_NEAR(b.p_upper, ref_mercury_t[4 * g + 3], 1e-10);
    }
}

TEST(Rosenbaum, WilcoxonMatchesPsens) {
    std::vector<double> d(erpcp_welder.size());
    for (std::size_t i = 0; i < d.size(); ++i)
        d[i] = erpcp_welder[i] - erpcp_control[i];
    for (std::size_t g = 0; g < ref_psens_gamma.size(); ++g) {
        const auto b = rosenbaum_wilcoxon(d, ref_psens_gamma[g]);
        EXPECT_NEAR(b.p_upper, ref_psens_upper4[g], 5.0001e-5);
        EXPECT_NEAR(b.p_lower, ref_psens_lower4[g], 5.0001e-5);
    }
    for (std::size_t g = 0; g < gammas.size(); ++g) {
        const auto b = rosenbaum_wilcoxon(wilcoxon_ties_d, gammas[g]);
        const double* r = &ref_wilcoxon_ties[5 * g];
        EXPECT_DOUBLE_EQ(b.statistic, r[0]);
        EXPECT_NEAR(b.expectation, r[1], 1e-10);
        EXPECT_NEAR(b.variance, r[2], 1e-9);
        EXPECT_NEAR(b.p_upper, r[3], 1e-12);
        EXPECT_NEAR(b.p_lower, r[4], 1e-12);
    }
}

TEST(Rosenbaum, SignTestIsExactBinomial) {
    for (std::size_t g = 0; g < gammas.size(); ++g) {
        const auto b = rosenbaum_sign_test(33, 12, gammas[g]);
        EXPECT_NEAR(b.p_upper, ref_sign[2 * g], 1e-13 + 1e-10 * ref_sign[2 * g]);
        EXPECT_NEAR(b.p_lower, ref_sign[2 * g + 1], 1e-13 + 1e-10 * ref_sign[2 * g + 1]);
    }
    EXPECT_DOUBLE_EQ(rosenbaum_sign_test(0, 5, 2).p_upper, 1.0);
}

TEST(Rosenbaum, SensitivityValueAndAmplification) {
    const Matrix mercury = rows_of(mercury_flat, 3);
    const auto f = [&](double g) { return rosenbaum_m_test(mercury, g).p_upper; };
    const double gv = rosenbaum_sensitivity_value(f, 0.05);
    EXPECT_NEAR(f(gv), 0.05, 1e-8);
    EXPECT_GT(gv, 1);
    EXPECT_DOUBLE_EQ(rosenbaum_sensitivity_value([](double) { return 0.3; }, 0.05), 1.0);
    const std::vector<double> lambda{3, 4, 6, 10};
    for (std::size_t k = 0; k < lambda.size(); ++k)
        EXPECT_NEAR(rosenbaum_amplify(2.5, lambda[k]), ref_amplify[k], 1e-12);
}

TEST(Rosenbaum, ValidatesInputs) {
    EXPECT_THROW(rosenbaum_wilcoxon({1, 2}, 0.5), std::invalid_argument);
    EXPECT_THROW(rosenbaum_wilcoxon({0, 0}, 1), std::invalid_argument);
    EXPECT_THROW(rosenbaum_sign_test(0, 0, 1), std::invalid_argument);
    Matrix y(2, 2, 1.0);
    y(0, 0) = NaN;
    EXPECT_THROW(rosenbaum_m_test(y, 1), std::invalid_argument);
    EXPECT_THROW(rosenbaum_m_test(Matrix(3, 2, 1.0), 1), std::invalid_argument);
    EXPECT_THROW(rosenbaum_amplify(2, 1.5), std::invalid_argument);
}
