#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace {
namespace n = dm::nn;
namespace m = n::models;
TEST(AttentionComponents, MaskEveryEntryAndFullyMaskedDerivative) {
  auto kl   = n::from_vector(std::vector<int>{4, 2, 0}),
       ql   = n::from_vector(std::vector<int>{2, 1, 2});
  auto mask = m::AttentionAllowed(kl, ql, 2, 5, true, 3);
  auto bits = n::cast(mask, n::DType::Int32).to_vector<int>();
  for (int b = 0; b < 3; ++b)
    for (int q = 0; q < 2; ++q)
      for (int k = 0; k < 5; ++k)
        EXPECT_EQ(
            bits[(b * 2 + q) * 5 + k],
            q < (b == 1 ? 1 : 2) &&
                k < (b == 0   ? 4
                     : b == 1 ? 2
                              : 0) &&
                k <= q + 3);
  auto q = n::ones({3, 2, 4}).requires_grad_(), k = n::ones({3, 5, 4}).requires_grad_(),
       v = n::ones({3, 5, 2}).requires_grad_();
  n::AttentionOptions o;
  o.mask   = mask;
  o.impl   = n::AttentionImpl::Plain;
  auto out = n::scaled_dot_product_attention(q, k, v, o);
  auto g   = n::grad(n::sum(out.output), {q, k, v});
  for (auto& value : g)
    EXPECT_EQ(n::sum(n::abs(n::select(value, 0, 2))).item(), 0);
  EXPECT_THROW(
      m::AttentionAllowed(
          n::from_vector(std::vector<int>{6}), n::from_vector(std::vector<int>{1}), 1, 5),
      std::invalid_argument);
}
TEST(AttentionComponents, AlibiAuthorsOrderingAndExtremeBuckets) {
  auto s = m::AlibiSlopes(3);
  ASSERT_EQ(s.size(), 3);
  EXPECT_FLOAT_EQ(s[0], 1.f / 16);
  EXPECT_FLOAT_EQ(s[1], 1.f / 256);
  EXPECT_FLOAT_EQ(s[2], 1.f / 4);
  EXPECT_EQ(m::RelativePositionBucket(std::numeric_limits<long long>::min()), 15);
  EXPECT_EQ(m::RelativePositionBucket(std::numeric_limits<long long>::min(), false), 31);
  for (int d = -256; d <= 256; ++d) {
    int a = m::RelativePositionBucket(d, true, 16, 32);
    EXPECT_GE(a, 0);
    EXPECT_LT(a, 16);
  }
  auto               table = n::zeros({16, 3}).requires_grad_();
  auto               bias  = m::RelativePositionBias(table, 2, 5, true, 16, 32, 3);
  auto               g     = n::grad(n::sum(bias), {table})[0].to_vector<float>();
  std::vector<float> expected(48);
  auto indices = m::RelativePositionBuckets(2, 5, true, 16, 32, 3).to_vector<std::int64_t>();
  for (auto i : indices)
    for (int h = 0; h < 3; ++h)
      ++expected[i * 3 + h];
  EXPECT_EQ(g, expected);
}
TEST(AttentionComponents, RotarySharedOffsetAndCachedSuffix) {
  auto x = n::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 4});
  auto a = m::ApplyRotary(x), b = m::ApplyRotary(x, 7);
  EXPECT_NEAR(
      n::sum(n::select(a, 1, 0) * n::select(a, 1, 1)).item(),
      n::sum(n::select(b, 1, 0) * n::select(b, 1, 1)).item(),
      1e-5);
  auto suffix = m::ApplyRotary(n::slice_axis(x, 1, 1, 2), 1);
  EXPECT_EQ(suffix.to_vector<float>(), n::slice_axis(a, 1, 1, 2).to_vector<float>());
  EXPECT_THROW(m::ApplyRotary(x, -1), std::invalid_argument);
}
} // namespace
