#include <datamunge/causal/matching.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using datamunge::causal::match_on_propensity;
using datamunge::causal::matched_effect;
using datamunge::causal::propensity_score_match;
using datamunge::causal::PropensityMatch;
using datamunge::causal::PropensityMatchOptions;
using Matrix = datamunge::linalg::DenseMatrix<double>;
const double na = std::numeric_limits<double>::quiet_NaN();
#include "causal_matching_data.inc"

struct Lalonde {
    Matrix x;
    std::vector<int> a;
    std::vector<double> y;
};

Lalonde lalonde() {
    const std::size_t n = k_treat.size(), p = k_x.size() / n;
    Lalonde d{Matrix(n, p), {}, k_re78};
    for (std::size_t i = 0; i < n; ++i) {
        d.a.push_back(static_cast<int>(k_treat[i]));
        for (std::size_t j = 0; j < p; ++j)
            d.x(i, j) = k_x[i * p + j];
    }
    return d;
}

void expect_close(const std::vector<double>& got, const std::vector<double>& want, double tol, const char* what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (std::isnan(want[i]))
            EXPECT_TRUE(std::isnan(got[i])) << what << " " << i;
        else
            EXPECT_NEAR(got[i], want[i], tol * (1 + std::abs(want[i]))) << what << " " << i;
    }
}

void expect_matchit(const Lalonde& d, const PropensityMatch& m, const std::vector<double>& weights,
                    const std::vector<double>& matches, const std::vector<double>& dim, const std::vector<double>& att,
                    const std::vector<double>& smd, const std::vector<double>& vr, const std::vector<double>& ecdf) {
    expect_close(m.weights, weights, 1e-10, "weights");
    const auto cols = static_cast<std::size_t>(dim[1]);
    std::size_t row = 0;
    for (std::size_t i = 0; i < d.a.size(); ++i) {
        if (d.a[i] != 1)
            continue;
        // Equidistant controls may be listed in either order (their scores tie to the last bit), so
        // compare each treated unit's matches as a set; weights and effects pin down the selection.
        std::vector<std::size_t> got = m.matches[i];
        std::ranges::sort(got);
        std::vector<double> want_row(matches.begin() + static_cast<std::ptrdiff_t>(row * cols),
                                     matches.begin() + static_cast<std::ptrdiff_t>((row + 1) * cols));
        std::ranges::sort(want_row, [](double l, double r) { return std::isnan(r) ? !std::isnan(l) : l < r; });
        for (std::size_t k = 0; k < cols; ++k) {
            const double want = want_row[k];
            if (std::isnan(want))
                EXPECT_LE(got.size(), k) << "treated row " << i;
            else {
                ASSERT_LT(k, got.size()) << "treated row " << i;
                EXPECT_EQ(got[k] + 1, static_cast<std::size_t>(want)) << "treated row " << i << " match " << k;
            }
        }
        ++row;
    }
    const auto e = matched_effect(d.y, d.a, m);
    EXPECT_NEAR(e.estimate, att[0], 1e-8 * std::abs(att[0]));
    EXPECT_NEAR(e.standard_error, att[1], 1e-8 * att[1]);
    expect_close(m.after.standardized_difference, smd, 1e-8, "smd");
    expect_close(m.after.variance_ratio, vr, 1e-8, "variance ratio");
    expect_close(m.after.ecdf_max, ecdf, 1e-8, "ecdf");
}
} // namespace

TEST(CausalMatching, MatchesMatchItNearestNeighbor) {
    const auto d = lalonde();
    const auto m = propensity_score_match(d.a, d.x);
    expect_close(m.propensity, ref_distance, 1e-7, "propensity");
    expect_close(m.before.standardized_difference, ref_smd_before, 1e-8, "smd before");
    expect_close(m.before.variance_ratio, ref_vr_before, 1e-8, "vr before");
    expect_close(m.before.ecdf_max, ref_ecdf_before, 1e-8, "ecdf before");
    expect_matchit(d, m, ref_nn_weights, ref_nn_matches, ref_nn_dim, ref_nn_att, ref_nn_smd, ref_nn_vr, ref_nn_ecdf);
}

TEST(CausalMatching, MatchesMatchItCaliperRatioAndReplacement) {
    const auto d = lalonde();
    const auto cal = propensity_score_match(d.a, d.x, {.caliper = 0.1});
    EXPECT_EQ(cal.matched_treated, 111u);
    expect_matchit(d, cal, ref_cal_weights, ref_cal_matches, ref_cal_dim, ref_cal_att, ref_cal_smd, ref_cal_vr,
                   ref_cal_ecdf);
    const auto r2 = propensity_score_match(d.a, d.x, {.ratio = 2});
    expect_matchit(d, r2, ref_r2_weights, ref_r2_matches, ref_r2_dim, ref_r2_att, ref_r2_smd, ref_r2_vr, ref_r2_ecdf);
    const auto rep = propensity_score_match(d.a, d.x, {.ratio = 2, .replace = true});
    expect_matchit(d, rep, ref_rep_weights, ref_rep_matches, ref_rep_dim, ref_rep_att, ref_rep_smd, ref_rep_vr,
                   ref_rep_ecdf);
}

TEST(CausalMatching, SmallCasesAndValidation) {
    const Matrix none(4, 0);
    // Largest-first order lets the high-propensity treated unit claim the close control.
    const auto m = match_on_propensity({1, 0, 0, 1}, {0.6, 0.55, 0.2, 0.3}, none);
    EXPECT_EQ(m.matches[0], std::vector<std::size_t>{1});
    EXPECT_EQ(m.matches[3], std::vector<std::size_t>{2});
    EXPECT_EQ(m.subclass[1], 0u);
    // Data order with replacement: both treated units take the closest control.
    const auto rep = match_on_propensity({1, 0, 0, 1}, {0.6, 0.55, 0.2, 0.5}, none, {.replace = true});
    EXPECT_EQ(rep.matches[3], std::vector<std::size_t>{1});
    EXPECT_DOUBLE_EQ(rep.weights[1], 1.0);
    EXPECT_DOUBLE_EQ(rep.weights[2], 0.0);
    // A tight caliper leaves the distant treated unit unmatched.
    const auto cal = match_on_propensity({1, 0, 0, 1}, {0.6, 0.55, 0.2, 0.9}, none, {.caliper = 0.3});
    EXPECT_TRUE(cal.matches[3].empty());
    EXPECT_EQ(cal.matched_treated, 1u);
    const auto e = matched_effect({5, 2, 1, 4}, {1, 0, 0, 1}, m);
    EXPECT_DOUBLE_EQ(e.estimate, 3.0);
    EXPECT_THROW((void)match_on_propensity({1, 0}, {0.5, 1.0}, Matrix(2, 0)), std::invalid_argument);
    EXPECT_THROW((void)match_on_propensity({1, 0}, {0.5, 0.4}, Matrix(2, 0), {.ratio = 0}), std::invalid_argument);
    EXPECT_THROW((void)match_on_propensity({1, 1}, {0.5, 0.4}, Matrix(2, 0)), std::invalid_argument);
}
