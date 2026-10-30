#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

namespace nn = datamunge::nn;

TEST(NnViews, ViewStridesFollowsTheNoCopyRule) {
  // Contiguous: always a view with row-major strides.
  EXPECT_EQ(*nn::view_strides({2, 3, 4}, {12, 4, 1}, {6, 4}), (nn::Shape{4, 1}));
  EXPECT_EQ(*nn::view_strides({2, 3, 4}, {12, 4, 1}, {24}), (nn::Shape{1}));
  // A transpose cannot merge its axes...
  EXPECT_FALSE(nn::view_strides({3, 2}, {1, 3}, {6}).has_value());
  // ...but can gain or lose size-1 axes and split an axis.
  EXPECT_EQ(*nn::view_strides({3, 2}, {1, 3}, {3, 1, 2}), (nn::Shape{1, 6, 3}));  // a size-1 stride is arbitrary
  EXPECT_EQ(*nn::view_strides({4, 2}, {1, 4}, {2, 2, 2}), (nn::Shape{2, 1, 4}));
  // Rows of a matrix taken every other row: merging rows and columns needs a copy, splitting rows does not.
  EXPECT_FALSE(nn::view_strides({2, 3}, {6, 1}, {6}).has_value());
  EXPECT_EQ(*nn::view_strides({4, 3}, {6, 1}, {2, 2, 3}), (nn::Shape{12, 6, 1}));
  // Broadcast axes (stride 0) cannot merge with real ones.
  EXPECT_FALSE(nn::view_strides({2, 3}, {0, 1}, {6}).has_value());
  EXPECT_EQ(*nn::view_strides({0, 3}, {3, 1}, {3, 0}), nn::contiguous_strides({3, 0}));
  EXPECT_THROW((void)nn::view_strides({2, 3}, {3, 1}, {5}), std::invalid_argument);
}

TEST(NnViews, ReshapeReturnsViewsWhenPossible) {
  const auto x = nn::reshape(nn::arange(0, 24), {2, 3, 4});
  const auto t = nn::transpose(x, {0, 2, 1});  // [2, 4, 3], not contiguous
  const auto v = nn::reshape(t, {2, 4, 1, 3});  // inserts an axis: a view now
  EXPECT_TRUE(nn::may_share_memory(v, x));
  EXPECT_TRUE(nn::can_view(t, {2, 4, 1, 3}));
  const auto c = nn::reshape(t, {2, 12});  // merges transposed axes: a copy
  EXPECT_FALSE(nn::may_share_memory(c, x));
  EXPECT_FALSE(nn::can_view(t, {2, 12}));
  EXPECT_EQ(c.to_vector<float>(), t.contiguous().to_vector<float>());
  EXPECT_EQ(nn::squeeze(v).to_vector<float>(), t.to_vector<float>());
  // Gradients flow through a view reshape as before.
  auto leaf = nn::reshape(nn::arange(0, 6, 1, nn::DType::Float64), {2, 3}).requires_grad_(true);
  auto y    = nn::sum(nn::square(nn::reshape(nn::transpose(leaf), {3, 1, 2})));
  auto g    = nn::grad({y}, {leaf})[0];
  EXPECT_EQ(g.to_vector<double>(), (std::vector<double>{0, 2, 4, 6, 8, 10}));
}

TEST(NnViews, SlidingWindowIsAZeroCopyView) {
  const auto x = nn::arange(0, 10, 1, nn::DType::Float64);
  const auto w = nn::sliding_window(x, 0, 4, 3);  // windows starting at 0, 3, 6
  EXPECT_EQ(w.shape(), (nn::Shape{3, 4}));
  EXPECT_EQ(w.to_vector<double>(), (std::vector<double>{0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9}));
  EXPECT_TRUE(nn::may_share_memory(w, x));
  EXPECT_EQ(nn::storage_bytes(w), 10 * sizeof(double));
  const auto m = nn::reshape(nn::arange(0, 12), {3, 4});
  const auto r = nn::sliding_window(m, 1, 2);  // [3, 3, 2]
  EXPECT_EQ(r.shape(), (nn::Shape{3, 3, 2}));
  EXPECT_EQ(nn::select(nn::select(r, 0, 2), 0, 1).to_vector<float>(), (std::vector<float>{9, 10}));
  EXPECT_THROW((void)nn::sliding_window(x, 0, 11), std::invalid_argument);
  EXPECT_THROW((void)nn::sliding_window(x, 0, 2, 0), std::invalid_argument);
}

TEST(NnViews, SlidingWindowGradientCountsCoverage) {
  auto x = nn::arange(0, 7, 1, nn::DType::Float64).requires_grad_(true);
  auto y = nn::sum(nn::sliding_window(x, 0, 3, 2));  // windows [0,3), [2,5), [4,7)
  auto g = nn::grad({y}, {x}, {.create_graph = true})[0];
  EXPECT_EQ(g.to_vector<double>(), (std::vector<double>{1, 1, 2, 1, 2, 1, 1}));
  // A weighted sum checks positions, and gradcheck the whole map.
  auto f = [](const std::vector<nn::Tensor>& in) {
    return nn::sliding_window(in[0], 1, 3, 1) * nn::tensor({1.0, 2.0, 3.0}, {3}, nn::DType::Float64);
  };
  nn::Generator gen(1);
  auto z = nn::normal({2, 5}, 0, 1, gen, nn::DType::Float64).requires_grad_(true);
  EXPECT_TRUE(nn::gradcheck(f, {z}).ok);
  EXPECT_TRUE(nn::gradgradcheck(f, {z}).ok);
}

TEST(NnViews, MayShareMemory) {
  const auto a = nn::arange(0, 10);
  EXPECT_TRUE(nn::may_share_memory(a, a));
  EXPECT_FALSE(nn::may_share_memory(nn::slice(a, {0}, {5}), nn::slice(a, {5}, {5})));
  EXPECT_TRUE(nn::may_share_memory(nn::slice(a, {0}, {6}), nn::slice(a, {5}, {5})));
  EXPECT_FALSE(nn::may_share_memory(a, a.clone()));
  EXPECT_FALSE(nn::may_share_memory(a, nn::Tensor()));
  // Negative strides reach backwards from the offset.
  const auto rev = nn::slice_axis(a, 0, 9, -11, -1);
  EXPECT_TRUE(nn::may_share_memory(rev, nn::slice(a, {0}, {1})));
}
