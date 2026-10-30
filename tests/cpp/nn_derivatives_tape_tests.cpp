#include <gtest/gtest.h>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/derivatives.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/tape.hpp"

#include <cmath>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;

namespace {

// f(x) = [x0^2 x1, sin(x0) + x1, x0 x1^3], hand Jacobian below.
nn::Tensor f3(const nn::Tensor& x) {
    auto a = nn::slice(x, {0}, {1});
    auto b = nn::slice(x, {1}, {1});
    return nn::concat({nn::mul(nn::mul(a, a), b), nn::add(nn::sin(a), b), nn::mul(a, nn::pow(b, 3.0))}, 0);
}

std::vector<double> J3(double x0, double x1) {
    return {2 * x0 * x1, x0 * x0, std::cos(x0), 1.0, x1 * x1 * x1, 3 * x0 * x1 * x1};
}

}  // namespace

TEST(NnDerivatives, JacobianMatchesHandDerivative) {
    const auto x = nn::tensor({0.7, -1.3}, {2}, DType::Float64);
    const auto J = nn::jacobian(f3, x);
    ASSERT_EQ(J.shape(), (nn::Shape{3, 2}));
    const auto got = J.to_vector<double>(), want = J3(0.7, -1.3);
    for (std::size_t i = 0; i < want.size(); ++i) EXPECT_NEAR(got[i], want[i], 1e-12);
    EXPECT_FALSE(x.requires_grad());
}

TEST(NnDerivatives, VjpAndJvpAreTransposeProducts) {
    const auto x = nn::tensor({0.7, -1.3}, {2}, DType::Float64);
    const auto v = nn::tensor({1.0, -2.0, 0.5}, {3}, DType::Float64);
    const auto u = nn::tensor({0.3, 2.0}, {2}, DType::Float64);
    const auto J = J3(0.7, -1.3);
    const auto vj = nn::vjp(f3, x, v).to_vector<double>();
    const auto ju = nn::jvp(f3, x, u).to_vector<double>();
    const double vv[3] = {1.0, -2.0, 0.5}, uu[2] = {0.3, 2.0};
    for (int j = 0; j < 2; ++j) {
        double s = 0;
        for (int i = 0; i < 3; ++i) s += vv[i] * J[i * 2 + j];
        EXPECT_NEAR(vj[j], s, 1e-12);
    }
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(ju[i], J[i * 2] * uu[0] + J[i * 2 + 1] * uu[1], 1e-12);
    EXPECT_THROW((void)nn::vjp(f3, x, u), std::invalid_argument);
    EXPECT_THROW((void)nn::jvp(f3, x, v), std::invalid_argument);
}

TEST(NnDerivatives, HessianAndHvpOfQuadraticPlusCubic) {
    // f = x0^2 x1 + x1^3: H = [[2 x1, 2 x0], [2 x0, 6 x1]].
    auto f = [](const nn::Tensor& x) {
        auto a = nn::slice(x, {0}, {1});
        auto b = nn::slice(x, {1}, {1});
        return nn::sum(nn::add(nn::mul(nn::mul(a, a), b), nn::pow(b, 3.0)));
    };
    const auto x = nn::tensor({1.5, -0.5}, {2}, DType::Float64);
    const auto H = nn::hessian(f, x).to_vector<double>();
    const std::vector<double> want = {-1.0, 3.0, 3.0, -3.0};
    for (std::size_t i = 0; i < 4; ++i) EXPECT_NEAR(H[i], want[i], 1e-12);
    const auto hv = nn::hvp(f, x, nn::tensor({2.0, 1.0}, {2}, DType::Float64)).to_vector<double>();
    EXPECT_NEAR(hv[0], 1.0, 1e-12);
    EXPECT_NEAR(hv[1], 3.0, 1e-12);
    EXPECT_THROW((void)nn::hessian(f3, x), std::invalid_argument);
}

TEST(NnDerivatives, UnusedInputGivesZeros) {
    const auto x = nn::tensor({1.0, 2.0}, {2}, DType::Float64);
    auto constant = [](const nn::Tensor&) { return nn::tensor({5.0}, {1}, DType::Float64); };
    const auto J = nn::jacobian(constant, x).to_vector<double>();
    EXPECT_EQ(J, (std::vector<double>{0.0, 0.0}));
    EXPECT_EQ(nn::vjp(constant, x, nn::tensor({1.0}, {1}, DType::Float64)).to_vector<double>(),
              (std::vector<double>{0.0, 0.0}));
}

TEST(NnTape, CubicPlusLinearTape) {
    auto x = nn::scalar(2.0).requires_grad_();
    auto y = nn::add(nn::pow(x, 3.0), nn::mul(nn::scalar(2.0), x));
    const auto t = nn::tape({y}, {{x, "x"}});
    ASSERT_EQ(t.nodes.size(), 4u);
    EXPECT_EQ(t.op_count(), 3u);
    EXPECT_EQ(t.leaf_count(), 1u);
    EXPECT_EQ(t.nodes[0].op, "Add");
    EXPECT_EQ(t.nodes[0].depth, 0u);
    EXPECT_EQ(t.nodes.back().label, "x");
    EXPECT_EQ(t.nodes.back().depth, 2u);
    const auto text = t.to_text();
    EXPECT_NE(text.find("<- const x"), std::string::npos) << text;
    EXPECT_NE(t.to_dot().find("digraph"), std::string::npos);
    const auto svg = nn::plot_tape(t).to_svg();
    EXPECT_NE(svg.find("<svg"), std::string::npos);
}

TEST(NnTape, NoHistoryIsEmpty) {
    const auto t = nn::tape({nn::scalar(1.0)});
    EXPECT_TRUE(t.nodes.empty());
    EXPECT_THROW((void)nn::plot_tape(t), std::invalid_argument);
}
