#include <gtest/gtest.h>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/gradient_tricks.hpp"
#include "datamunge/nn/ops.hpp"

#include <vector>

namespace nn = datamunge::nn;
using nn::DType;

namespace {

std::vector<double> grad_of(const nn::Tensor& out, const nn::Tensor& x) {
    return nn::grad(nn::sum(out), {x})[0].to_vector<double>();
}

}  // namespace

TEST(NnGradientTricks, StraightThroughForwardIsValueBackwardIsIdentity) {
    auto x = nn::tensor({0.2, 1.7, -0.4}, {3}, DType::Float64).requires_grad_();
    const auto y = nn::straight_through(x, nn::ceil(x));
    EXPECT_EQ(y.to_vector<double>(), (std::vector<double>{1, 2, 0}));
    EXPECT_EQ(grad_of(nn::mul(y, y), x), (std::vector<double>{2, 4, 0}));
    EXPECT_THROW((void)nn::straight_through(x, nn::tensor({1.0}, {1}, DType::Float64)), std::invalid_argument);
}

TEST(NnGradientTricks, RoundSteRoundsHalfToEven) {
    auto x = nn::tensor({0.5, 1.5, -2.5, 2.4}, {4}, DType::Float64).requires_grad_();
    const auto y = nn::round_ste(x);
    EXPECT_EQ(y.to_vector<double>(), (std::vector<double>{0, 2, -2, 2}));
    EXPECT_EQ(grad_of(y, x), (std::vector<double>{1, 1, 1, 1}));
}

TEST(NnGradientTricks, FakeQuantizeClipsValueAndGradient) {
    // Grid 0.25 * {-4..4} = [-1, 1].
    auto x = nn::tensor({-1.3, -1.0, 0.1, 0.38, 1.0, 2.0}, {6}, DType::Float64).requires_grad_();
    const auto q = nn::fake_quantize(x, 0.25, -4, 4);
    const auto v = q.to_vector<double>();
    const std::vector<double> want = {-1.0, -1.0, 0.0, 0.5, 1.0, 1.0};
    for (std::size_t i = 0; i < want.size(); ++i) EXPECT_DOUBLE_EQ(v[i], want[i]);
    EXPECT_EQ(grad_of(q, x), (std::vector<double>{0, 1, 1, 1, 1, 0}));
    EXPECT_THROW((void)nn::fake_quantize(x, 0.0, -1, 1), std::invalid_argument);
    EXPECT_THROW((void)nn::fake_quantize(x, 1.0, 2, 1), std::invalid_argument);
}

TEST(NnGradientTricks, ReversalScaleAndClipOnlyChangeTheGradient) {
    auto x = nn::tensor({1.0, -2.0, 3.0}, {3}, DType::Float64).requires_grad_();
    const auto sq = [](const nn::Tensor& t) { return nn::mul(t, t); };  // gradient 2x
    EXPECT_EQ(nn::gradient_reversal(x, 0.5).to_vector<double>(), x.to_vector<double>());
    EXPECT_EQ(grad_of(sq(nn::gradient_reversal(x, 0.5)), x), (std::vector<double>{-1, 2, -3}));
    EXPECT_EQ(grad_of(sq(nn::scale_gradient(x, 3.0)), x), (std::vector<double>{6, -12, 18}));
    EXPECT_EQ(grad_of(sq(nn::clip_gradient(x, -1.0, 5.0)), x), (std::vector<double>{2, -1, 5}));
    EXPECT_THROW((void)nn::clip_gradient(x, 1.0, 0.0), std::invalid_argument);
}

TEST(NnGradientTricks, NoHistoryWithoutRequiresGrad) {
    const auto x = nn::tensor({0.3}, {1}, DType::Float64);
    EXPECT_FALSE(nn::round_ste(x).requires_grad());
    EXPECT_FALSE(nn::fake_quantize(x, 0.1, -10, 10).requires_grad());
}

TEST(NnGradientTricks, StochasticRoundIsUnbiasedWithIdentityGradient) {
    nn::Generator gen(7);
    auto x = nn::full({20000}, 1.3, DType::Float64).requires_grad_();
    const auto r = nn::stochastic_round_ste(x, gen);
    const auto v = r.to_vector<double>();
    double mean = 0;
    for (double e : v) {
        EXPECT_TRUE(e == 1.0 || e == 2.0);
        mean += e / static_cast<double>(v.size());
    }
    EXPECT_NEAR(mean, 1.3, 0.01);  // sd of the mean = sqrt(0.21 / 20000) = 0.0032
    const auto g = grad_of(r, x);
    EXPECT_EQ(g.front(), 1.0);
    EXPECT_EQ(g.back(), 1.0);
}
