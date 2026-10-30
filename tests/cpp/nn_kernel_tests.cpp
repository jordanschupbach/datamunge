#include <gtest/gtest.h>

#include "datamunge/nn/kernel.hpp"
#include "datamunge/nn/ops.hpp"

#include <cmath>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;

TEST(NnKernel, DispatchFloatingPicksTheElementType) {
    EXPECT_EQ(nn::dispatch_floating(DType::Float32, "t", [](auto tag) { return sizeof(tag); }), sizeof(float));
    EXPECT_EQ(nn::dispatch_floating(DType::Float64, "t", [](auto tag) { return sizeof(tag); }), sizeof(double));
    EXPECT_THROW(nn::dispatch_floating(DType::Int32, "t", [](auto) { return 0; }), std::invalid_argument);
}

TEST(NnKernel, MapElementsMatchesOpsAndPacksViews) {
    const auto x = nn::tensor({{1, 2, 3}, {4, 5, 6}}, DType::Float64);
    const auto y = nn::map_elements(x, [](auto v) { return 2 * v + 0.5; });
    EXPECT_EQ(y.dtype(), DType::Float64);
    EXPECT_EQ(y.to_vector<double>(), (std::vector<double>{2.5, 4.5, 6.5, 8.5, 10.5, 12.5}));
    const auto t = nn::map_elements(nn::transpose(x), [](auto v) { return v * v; });  // a strided view
    EXPECT_EQ(t.shape(), (nn::Shape{3, 2}));
    EXPECT_EQ(t.to_vector<double>(), (std::vector<double>{1, 16, 4, 25, 9, 36}));
    EXPECT_FALSE(nn::map_elements(x.clone().requires_grad_(), [](auto v) { return v; }).requires_grad());
    EXPECT_THROW((void)nn::map_elements(nn::tensor({1}, {1}, DType::Int64), [](auto v) { return v; }), std::invalid_argument);
}

TEST(NnKernel, MapElementsIsThreadCountIndependent) {
    std::vector<float> v(200000);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.001f * static_cast<float>(i));
    const auto x = nn::from_vector(v, {});
    nn::set_num_threads(1);
    const auto a = nn::map_elements(x, [](auto e) { return std::exp(e) * e; }, 1000).to_vector<float>();
    nn::set_num_threads(4);
    const auto b = nn::map_elements(x, [](auto e) { return std::exp(e) * e; }, 1000).to_vector<float>();
    nn::set_num_threads(0);
    EXPECT_EQ(a, b);
}

TEST(NnKernel, SummationMethodsOnAHardCase) {
    // 1 followed by 2^25 copies of 2^-25: the exact sum is 2, but naive float32 stalls at 1.
    const std::int64_t n = (1 << 25) + 1;
    std::vector<float> v(static_cast<std::size_t>(n), std::ldexp(1.0f, -25));
    v[0] = 1.0f;
    EXPECT_EQ(nn::sum_with(v.data(), n, nn::Summation::Naive), 1.0f);
    // Pairwise loses only the 7 tiny terms in the first naive block of 8, next to the 1.
    const float pairwise = nn::sum_with(v.data(), n, nn::Summation::Pairwise);
    EXPECT_LT(pairwise, 2.0f);
    EXPECT_NEAR(pairwise, 2.0f, 3e-7);
    EXPECT_EQ(nn::sum_with(v.data(), n, nn::Summation::Kahan), 2.0f);
    EXPECT_EQ(nn::sum_with(v.data(), n, nn::Summation::WideAccumulator), 2.0f);
    EXPECT_EQ(nn::sum_with(v.data(), 0, nn::Summation::Pairwise), 0.0f);
}

TEST(NnKernel, SummationBounds) {
    const double u = std::ldexp(1.0, -24);
    EXPECT_NEAR(nn::summation_error_bound(nn::Summation::Naive, 1001, DType::Float32), 1000 * u / (1 - 1000 * u), 1e-18);
    EXPECT_NEAR(nn::summation_error_bound(nn::Summation::Pairwise, 1024, DType::Float32), 17 * u / (1 - 17 * u), 1e-18);
    EXPECT_NEAR(nn::summation_error_bound(nn::Summation::Kahan, 10, DType::Float32), 2 * u + 20 * u * u, 1e-20);
    EXPECT_GT(nn::summation_error_bound(nn::Summation::WideAccumulator, 10, DType::Float32), u);
    EXPECT_TRUE(std::isinf(nn::summation_error_bound(nn::Summation::Naive, 1 << 25, DType::Float32)));
    EXPECT_THROW((void)nn::summation_error_bound(nn::Summation::Naive, 0, DType::Float32), std::invalid_argument);
    EXPECT_EQ(nn::summation_name(nn::Summation::Kahan), "kahan");
}
