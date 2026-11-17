#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace {
namespace n = dm::nn;
namespace m = n::models;
TEST(DncComponents, AllocationStableTiesAndAllPartials) {
  auto   u        = n::from_vector(std::vector<float>{.8f, .1f, .6f, .3f}, {1, 4}).requires_grad_();
  auto   path     = m::DncAllocation(u);
  auto   a        = path.to_vector<float>();
  double values[] = {.8, .1, .6, .3};
  int    order[]  = {1, 3, 2, 0};
  double product  = 1;
  for (int slot : order) {
    double v = 1e-6 + (1 - 1e-6) * values[slot];
    EXPECT_NEAR(a[slot], (1 - v) * product, 2e-8);
    product *= v;
  }
  auto probe = n::from_vector(std::vector<float>{.3f, -.2f, .8f, 1.f}, {1, 4});
  auto g     = n::grad(n::sum(path * probe), {u})[0].to_vector<float>();
  auto raw   = u.to_vector<float>();
  for (int j = 0; j < 4; ++j) {
    auto plus = raw, minus = raw;
    plus[j] += .001;
    minus[j] -= .001;
    double fp = n::sum(m::DncAllocation(n::from_vector(plus, {1, 4})) * probe).item(),
           fm = n::sum(m::DncAllocation(n::from_vector(minus, {1, 4})) * probe).item();
    EXPECT_NEAR(g[j], (fp - fm) / (plus[j] - minus[j]), 3e-5);
  }
  auto empty = m::DncAllocation(n::zeros({1, 4})).to_vector<float>();
  EXPECT_GT(empty[0], .99999);
  EXPECT_LT(empty[1], .000002);
  EXPECT_THROW(m::DncAllocation(n::full({1, 4}, 1.1)), std::invalid_argument);
}
TEST(DncComponents, TemporalChainHasCorrectDirectionAndZeroDiagonal) {
  auto L = n::zeros({1, 4, 4}), p = n::zeros({1, 4});
  for (int i = 0; i < 4; ++i) {
    std::vector<float> w(4);
    w[i]   = 1;
    auto y = m::DncTemporal(L, p, n::from_vector(w, {1, 4}));
    L      = y.links;
    p      = y.precedence;
  }
  auto v = L.to_vector<float>();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_EQ(v[i * 4 + j], float(i == j + 1));
}
TEST(DncComponents, SequenceCensusPaddingAndStreaming) {
  n::ParameterStore p(45203);
  auto              input   = n::zeros({2, 9, 4});
  auto              lengths = n::from_vector(std::vector<int>{1, 3});
  auto              path    = m::DncSequence(p, input, lengths);
  EXPECT_EQ(p.count(), 799);
  EXPECT_EQ(p.size(), 7);
  auto changed                     = input.clone();
  changed.data_ptr<float>()[8 * 4] = .93;
  auto altered                     = m::DncSequence(p, changed, lengths);
  for (auto& [name, value] : path)
    EXPECT_EQ(value.to_vector<float>(), altered.at(name).to_vector<float>());
  for (int t = 0; t < 9; ++t) {
    std::vector<n::Tensor> v;
    for (auto& name : m::DncNames())
      v.push_back(n::select(path.at(name), 1, t));
    auto step = m::DncCell(p, n::select(input, 1, t), m::DncState::From(v));
    for (int b = 0; b < 2; ++b)
      if (t <= 2 * (b == 0 ? 1 : 3)) {
        EXPECT_LT(
            n::max(
                n::abs(
                    n::select(step.logits, 0, b) -
                    n::select(n::select(path.at("logits"), 0, b), 0, t)))
                .item(),
            3e-6);
      }
  }
  EXPECT_THROW(m::DncState::From({}), std::invalid_argument);
}
} // namespace
