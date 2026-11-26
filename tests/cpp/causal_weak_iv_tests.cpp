#include <datamunge/causal/weak_iv.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using datamunge::causal::ConfidenceSetShape;
using datamunge::causal::weak_iv_inference;
using datamunge::causal::WeakIvInference;
using datamunge::causal::WeakIvOptions;
using Matrix = datamunge::linalg::DenseMatrix<double>;
const double Inf = std::numeric_limits<double>::infinity();
#include "causal_weak_iv_data.inc"

void near(double got, double want, const std::string& what) {
    if (std::isinf(want))
        EXPECT_EQ(got, want) << what;
    else
        EXPECT_NEAR(got, want, 1e-8 * (1 + std::abs(want))) << what;
}

// Layout: F, p, df1, df2, shape, set (lo, hi, 0, 0 or lo1, lo2, hi1, hi2), 2SLS est, se, lower,
// upper, LIML est, se, lower, upper, k.
void expect_iv(const WeakIvInference& r, const std::vector<double>& ref) {
    near(r.ar_statistic, ref[0], "AR F");
    near(r.ar_p_value, ref[1], "AR p");
    EXPECT_EQ(r.df1, static_cast<std::size_t>(ref[2]));
    EXPECT_EQ(r.df2, static_cast<std::size_t>(ref[3]));
    EXPECT_EQ(static_cast<int>(r.ar_set.shape), static_cast<int>(ref[4]));
    if (r.ar_set.shape == ConfidenceSetShape::interval) {
        near(r.ar_set.pieces.at(0).first, ref[5], "lower");
        near(r.ar_set.pieces.at(0).second, ref[6], "upper");
    } else if (r.ar_set.shape == ConfidenceSetShape::two_rays) {
        ASSERT_EQ(r.ar_set.pieces.size(), 2u);
        near(r.ar_set.pieces[0].first, ref[5], "ray 1 start");
        near(r.ar_set.pieces[1].first, ref[6], "ray 2 start");
        near(r.ar_set.pieces[0].second, ref[7], "ray 1 end");
        near(r.ar_set.pieces[1].second, ref[8], "ray 2 end");
    }
    const auto check = [&](const auto& e, std::size_t at, const std::string& name) {
        near(e.estimate, ref[at], name + " estimate");
        near(e.standard_error, ref[at + 1], name + " se");
        near(e.lower, ref[at + 2], name + " lower");
        near(e.upper, ref[at + 3], name + " upper");
    };
    check(r.tsls, 9, "2SLS");
    check(r.liml, 13, "LIML");
    near(r.liml.k, ref[17], "LIML k");
}

Matrix column(const std::vector<double>& v) { return Matrix(v.size(), 1, v); }
} // namespace

TEST(WeakIv, CardMatchesIvmodel) {
    const Matrix x(card_n, card_p, card_x);
    expect_iv(weak_iv_inference(card_y, card_d, column(card_nearc4), x, {.null_value = 0.1}), ref_card_just);
    Matrix z(card_n, 2);
    for (std::size_t i = 0; i < card_n; ++i) {
        z(i, 0) = card_nearc4[i];
        z(i, 1) = card_nearc2[i];
    }
    expect_iv(weak_iv_inference(card_y, card_d, z, x, {.alpha = 0.1}), ref_card_over);
}

TEST(WeakIv, UnboundedSetsMatchIvmodel) {
    const auto weak = weak_iv_inference(weak_y, weak_d, Matrix(200, 3, weak_z));
    expect_iv(weak, ref_weak);
    EXPECT_TRUE(weak.ar_set.contains(-1e6));
    EXPECT_FALSE(weak.ar_set.contains(1.5));
    std::vector<double> z1;
    for (std::size_t i = 0; i < 200; ++i)
        z1.push_back(weak_z[3 * i]);
    expect_iv(weak_iv_inference(weak_y, weak_d, column(z1), {}, {.null_value = 0.5, .intercept = false}),
              ref_weak_noint);
    expect_iv(weak_iv_inference(rays_y, rays_d, column(rays_z)), ref_rays);
}

TEST(WeakIv, EmptySetWhenOveridentificationFails) {
    // Two instruments with contradictory effects on y: no single beta fits.
    std::vector<double> y, d;
    Matrix z(400, 2);
    for (int i = 0; i < 400; ++i) {
        z(i, 0) = std::sin(1.7 * i);
        z(i, 1) = std::cos(0.9 * i);
        d.push_back(z(i, 0) + z(i, 1) + 0.1 * std::sin(5.3 * i));
        y.push_back(3 * z(i, 0) - 3 * z(i, 1) + 0.1 * std::cos(4.1 * i));
    }
    const auto r = weak_iv_inference(y, d, z);
    EXPECT_EQ(r.ar_set.shape, ConfidenceSetShape::empty);
    EXPECT_TRUE(r.ar_set.pieces.empty());
}

TEST(WeakIv, ValidatesInputs) {
    EXPECT_THROW(weak_iv_inference({1, 2}, {1}, column({1, 2})), std::invalid_argument);
    EXPECT_THROW(weak_iv_inference({1, 2, 3}, {1, 2, 3}, Matrix(3, 0)), std::invalid_argument);
    EXPECT_THROW(weak_iv_inference({1, 2, 3, 4}, {1, 2, 3, 5}, column({1, 0, 1, 0}), {}, {.alpha = 0}),
                 std::invalid_argument);
    EXPECT_THROW(weak_iv_inference({1, 2, 3}, {1, 2, 3}, column({1, 0, 1})), std::invalid_argument);
    EXPECT_THROW(weak_iv_inference({1, 2, 3, 4, 5}, {1, 2, 3, 5, 4}, column({1, 1, 1, 1, 1})),
                 std::runtime_error);
}
