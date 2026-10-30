/// @file nn_attention_variants_tests.cpp
/// @brief Tests for kernel attention, positive random features, local masks and attention entropy.

#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/nn.hpp>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

/// Direct O(Tq Tk) kernel attention with bottom-right causal alignment.
Tensor direct_kernel_attention(const Tensor& pq, const Tensor& pk, const Tensor& v, bool causal) {
  Tensor w = nn::matmul(pq, nn::swapaxes(pk, -2, -1));
  if (causal) w = nn::where(nn::causal_keep_mask(pq.size(-2), pk.size(-2)), w, 0.0);
  return nn::matmul(nn::div(w, nn::add(nn::sum(w, {-1}, true), 1e-6)), v);
}

}  // namespace

TEST(NnAttentionVariants, KernelAttentionMatchesDirectSum) {
  nn::Generator g(1);
  const auto F64 = nn::DType::Float64;
  const Tensor q = nn::normal({2, 3, 5, 4}, 0, 1, g, F64), k = nn::normal({2, 3, 7, 4}, 0, 1, g, F64), v = nn::normal({2, 3, 7, 2}, 0, 1, g, F64);
  const Tensor pq = nn::elu_feature_map(q), pk = nn::elu_feature_map(k);
  for (bool causal : {false, true}) {
    const Tensor got = nn::kernel_attention(pq, pk, v, causal);
    EXPECT_EQ(got.shape(), (nn::Shape{2, 3, 5, 2}));
    EXPECT_LT(nn::max(nn::abs(got - direct_kernel_attention(pq, pk, v, causal))).item(), 1e-12) << causal;
  }
  EXPECT_THROW((void)nn::kernel_attention(pk, pq, nn::normal({2, 3, 5, 2}, 0, 1, g, F64), true), std::invalid_argument);
  // Differentiable through the prefix sums.
  const Tensor a = nn::normal({4, 3}, 0, 1, g, F64).requires_grad_(true), b = nn::normal({6, 3}, 0, 1, g, F64).requires_grad_(true),
               c = nn::normal({6, 2}, 0, 1, g, F64).requires_grad_(true);
  const auto r = nn::gradcheck([](const std::vector<Tensor>& in) { return nn::kernel_attention(nn::elu_feature_map(in[0]), nn::elu_feature_map(in[1]), in[2], true); }, {a, b, c});
  EXPECT_TRUE(r.ok) << r.message;
}

TEST(NnAttentionVariants, PositiveRandomFeaturesAreUnbiased) {
  nn::Generator g(2);
  const auto F64 = nn::DType::Float64;
  const Tensor q = nn::tensor({0.3, -0.5, 0.8, 0.1}, {1, 4}, F64), k = nn::tensor({0.6, 0.2, -0.4, 0.9}, {1, 4}, F64);
  const double s = 0.5, exact = std::exp(s * (0.18 - 0.1 - 0.32 + 0.09));
  for (bool orth : {false, true}) {
    const Tensor w = orth ? nn::orthogonal_gaussian_projection(40000, 4, g, F64) : nn::normal({40000, 4}, 0, 1, g, F64);
    const double est = nn::sum(nn::positive_random_features(q, w, s) * nn::positive_random_features(k, w, s)).item();
    EXPECT_NEAR(est / exact, 1.0, 0.02) << orth;
    EXPECT_GT(nn::min(nn::positive_random_features(q, w, s)).item(), 0.0);
  }
  EXPECT_THROW((void)nn::positive_random_features(q, nn::normal({8, 3}, 0, 1, g, F64)), std::invalid_argument);
}

TEST(NnAttentionVariants, OrthogonalProjectionBlocks) {
  nn::Generator g(3);
  const Tensor w = nn::orthogonal_gaussian_projection(10, 4, g, nn::DType::Float64);
  ASSERT_EQ(w.shape(), (nn::Shape{10, 4}));
  const Tensor gram = nn::matmul(w, nn::swapaxes(w, 0, 1));
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      if (i != j && i / 4 == j / 4) EXPECT_NEAR(gram.at({i, j}), 0.0, 1e-12) << i << "," << j;
  EXPECT_THROW((void)nn::orthogonal_gaussian_projection(0, 4, g), std::invalid_argument);
}

TEST(NnAttentionVariants, LocalMasks) {
  const auto m = nn::local_keep_mask(4, 4, 2);
  const bool want[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 1, 1}};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_EQ(m.at({i, j}) != 0, want[i][j]) << i << "," << j;
  const auto two = nn::local_keep_mask(3, 3, 2, false);
  EXPECT_EQ(nn::sum(nn::cast(two, nn::DType::Int64)).item(), 7.0);  // tridiagonal 3x3
  // A window as long as the keys is the causal mask, also with Tq < Tk.
  EXPECT_TRUE(nn::all(nn::equal(nn::local_keep_mask(3, 6, 6), nn::causal_keep_mask(3, 6))).item<bool>());
  EXPECT_THROW((void)nn::local_keep_mask(3, 3, 0), std::invalid_argument);
}

TEST(NnAttentionVariants, EntropyOfWeights) {
  const Tensor w = nn::tensor({0.25, 0.25, 0.25, 0.25, 1, 0, 0, 0, 0, 0, 0, 0}, {3, 4}, nn::DType::Float64);
  const Tensor h = nn::attention_entropy(w);
  EXPECT_NEAR(h.at({0}), std::log(4.0), 1e-12);
  EXPECT_EQ(h.at({1}), 0.0);
  EXPECT_EQ(h.at({2}), 0.0);
}
