#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace n = dm::nn;
namespace m = n::models;
namespace {
void Near(const n::Tensor& a, const n::Tensor& b, double tolerance = 1e-6) {
  ASSERT_EQ(a.shape(), b.shape());
  EXPECT_LE(n::max(n::abs(a - b)).item(), tolerance);
}
void Zero(n::ParameterStore& p) {
  for (const auto& name : p.names())
    p.set(name, n::zeros(p.at(name).shape(), p.dtype()));
}
} // namespace
TEST(NnRecurrentExtensions, TruncationPreservesEveryForwardValueButCutsEarlyInputPaths) {
  n::ParameterStore p(39);
  auto              x = n::ones({2, 12, 2});
  x.requires_grad_(true);
  auto h       = n::zeros({2, 12});
  auto lengths = n::from_vector(std::vector<std::int32_t>{12, 5});
  auto full    = m::MaskedTruncatedSequence(p, x, lengths, {h}, 0);
  auto cut     = m::MaskedTruncatedSequence(p, x, lengths, {h}, 3);
  Near(full.states, cut.states, 0);
  Near(full.logits, cut.logits, 0);
  auto gf = n::grad(n::sum(full.logits), {x})[0], gc = n::grad(n::sum(cut.logits), {x})[0];
  EXPECT_GT(n::max(n::abs(n::slice_axis(gf, 1, 0, 2))).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(gc, 1, 0, 2))).item(), 0);
  EXPECT_GT(n::max(n::abs(n::slice_axis(gc, 1, 9, 12))).item(), 0);
}
TEST(NnRecurrentExtensions, DropoutOneIsVanillaAndExplicitZeroMaskBreaksHiddenRecurrence) {
  n::ParameterStore p(30);
  auto              x = n::ones({1, 12, 2}), h = n::ones({1, 12});
  auto              l = n::from_vector(std::vector<std::int32_t>{12});
  auto              a = m::MaskedTanhSequence(p, x, l, {h});
  auto              y = m::MaskedDropoutSequence(p, x, l, {h}, n::ones({1, 12, 12}));
  Near(a.states, y.states, 0);
  auto other = m::MaskedDropoutSequence(p, x, l, {h}, n::zeros({1, 12, 12}));
  EXPECT_GT(n::max(n::abs(a.states - other.states)).item(), 0);
}
TEST(NnRecurrentExtensions, DropoutDrawBankAndTimeSharingAreExact) {
  auto a = m::MakeRecurrentDropoutMask(2, 7, true), b = m::MakeRecurrentDropoutMask(2, 7, false);
  Near(a.uniforms, b.uniforms, 0);
  std::mt19937 rng(7);
  auto         u = a.uniforms.to_vector<double>();
  for (auto v : u)
    EXPECT_DOUBLE_EQ(v, (double(rng()) + .5) / 4294967296.);
  auto mask = a.mask.to_vector<float>();
  for (int i = 0; i < 288; ++i)
    EXPECT_EQ(mask[i], u[(i / 144) * 144 + i % 12] >= .2 ? 1.25f : 0.f);
}
TEST(NnRecurrentExtensions, StackedUpperConsumesCurrentLowerState) {
  n::ParameterStore p(29, n::DType::Float64);
  auto x = n::ones({1, 12, 2}, n::DType::Float64), h = n::zeros({1, 12}, n::DType::Float64);
  auto l = n::from_vector(std::vector<std::int32_t>{1});
  m::MaskedStackedSequence(p, x, l, {h, h});
  Zero(p);
  p.set("lower/b", n::full({12}, .2, n::DType::Float64));
  p.set("upper/Wx", n::eye(12, n::DType::Float64));
  auto y = m::MaskedStackedSequence(p, x, l, {h, h});
  Near(y.final_lower, n::full({1, 12}, std::tanh(.2), n::DType::Float64), 1e-14);
  Near(y.final_state, n::full({1, 12}, std::tanh(std::tanh(.2)), n::DType::Float64), 1e-14);
  EXPECT_EQ(p.count(), 506);
}
TEST(NnRecurrentExtensions, InputOnlyPoolingHasAnalyticZeroKernelMemory) {
  for (bool sru : {false, true}) {
    n::ParameterStore p(1, n::DType::Float64);
    auto x = n::zeros({1, 12, 2}, n::DType::Float64), h = n::zeros({1, 12}, n::DType::Float64),
         c   = n::full({1, 12}, .4, n::DType::Float64);
    auto l   = n::from_vector(std::vector<std::int32_t>{1});
    auto run = [&]() {
      return sru ? m::MaskedSruSequence(p, x, l, {h, c}) : m::MaskedQrnnSequence(p, x, l, {h, c});
    };
    run();
    Zero(p);
    auto y = run();
    Near(y.final_cell, n::full({1, 12}, .2, n::DType::Float64), 1e-14);
    Near(y.final_state, n::full({1, 12}, sru ? .5 * std::tanh(.2) : .1, n::DType::Float64), 1e-14);
    EXPECT_EQ(p.count(), sru ? 146 : 206);
  }
}
TEST(NnRecurrentExtensions, PhasedClosedGateHoldsStateAndTimestampsRemainDifferentiable) {
  n::ParameterStore p(36);
  auto              x = n::ones({1, 12, 2}), h = n::ones({1, 12}), c = n::full({1, 12}, .4);
  auto              l     = n::from_vector(std::vector<std::int32_t>{1});
  auto              times = n::zeros({1, 12});
  m::MaskedPhasedSequence(p, x, l, {h, c}, times, n::scalar(0.f));
  p.set("time/log_period", n::zeros({12}));
  p.set("time/shift", n::full({12}, -.75));
  auto closed = m::MaskedPhasedSequence(p, x, l, {h, c}, times, n::scalar(0.f));
  Near(closed.final_state, h, 0);
  Near(closed.final_cell, c, 0);
  p.set("time/shift", n::full({12}, -.125));
  times.requires_grad_(true);
  auto open = m::MaskedPhasedSequence(p, x, l, {h, c}, times, n::scalar(0.f));
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(open.logits), {times})[0])).item(), 0);
  EXPECT_EQ(p.count(), 770);
}
TEST(NnRecurrentExtensions, AllExtensionPaddingHasZeroInputDerivativesAndHeldStates) {
  auto data = m::MakeMarkedSequences(2);
  for (int kind = 0; kind < 6; ++kind) {
    n::ParameterStore p(1);
    auto              x = data.inputs.clone();
    x.requires_grad_(true);
    auto h = n::zeros({2, 12});
    auto l = n::zeros({2}, n::DType::Int32);
    auto y = kind == 0   ? m::MaskedStackedSequence(p, x, l, {h, h})
             : kind == 1 ? m::MaskedTruncatedSequence(p, x, l, {h}, 3)
             : kind == 2 ? m::MaskedDropoutSequence(p, x, l, {h}, n::ones({2, 12, 12}))
             : kind == 3 ? m::MaskedPhasedSequence(
                               p, x, l, {h, h}, m::RecurrentObservationTimes(2), n::scalar(0.f))
             : kind == 4 ? m::MaskedQrnnSequence(p, x, l, {h, h})
                         : m::MaskedSruSequence(p, x, l, {h, h});
    EXPECT_EQ(n::max(n::abs(y.states)).item(), 0);
    EXPECT_EQ(n::max(n::abs(n::grad(n::sum(y.logits), {x})[0])).item(), 0);
  }
}
TEST(NnRecurrentExtensions, AddressUniformCaseIsAnalyticAndRetainsGradients) {
  auto memory = n::zeros({1, 4, 4});
  memory.requires_grad_(true);
  auto y = m::NtmAddressing(memory, n::zeros({1, 10}), n::full({1, 4}, .25));
  Near(y.content, n::full({1, 4}, .25), 1e-7);
  Near(y.weights, n::full({1, 4}, .25), 1e-7);
  auto g = n::grad(n::sum(y.weights), {memory})[0];
  EXPECT_TRUE(g.defined());
  EXPECT_TRUE(std::isfinite(n::sum(g).item()));
}
TEST(NnRecurrentExtensions, MemoryQaEmptyFactsReadZeroAndHaveZeroFactGradients) {
  for (bool kv : {false, true}) {
    n::ParameterStore p(1);
    auto              facts = n::ones({2, 6, 2, 13});
    facts.requires_grad_(true);
    auto q = n::one_hot(n::from_vector(std::vector<std::int32_t>{1, 2}), 13),
         l = n::zeros({2}, n::DType::Int32);
    auto y =
        kv ? m::KeyValueMemoryNetwork(p, facts, q, l) : m::AdjacentMemoryNetwork(p, facts, q, l);
    EXPECT_EQ(p.count(), kv ? 264 : 312);
    EXPECT_EQ(n::max(n::abs(y.reads)).item(), 0);
    EXPECT_EQ(n::max(n::abs(y.attention)).item(), 0);
    EXPECT_EQ(n::max(n::abs(n::grad(n::sum(y.logits), {facts})[0])).item(), 0);
  }
}
TEST(NnRecurrentExtensions, MemoryQaFactPermutationPreservesAnswer) {
  for (bool kv : {false, true}) {
    n::ParameterStore p(1);
    auto              ids   = n::reshape(n::arange(0, 12, 1, n::DType::Int32), {1, 6, 2});
    auto              facts = n::one_hot(ids, 13),
         q                  = n::one_hot(n::from_vector(std::vector<std::int32_t>{1}), 13),
         l                  = n::from_vector(std::vector<std::int32_t>{6});
    auto reverse            = n::concat(
        {n::slice_axis(facts, 1, 5, 6),
                    n::slice_axis(facts, 1, 4, 5),
                    n::slice_axis(facts, 1, 3, 4),
                    n::slice_axis(facts, 1, 2, 3),
                    n::slice_axis(facts, 1, 1, 2),
                    n::slice_axis(facts, 1, 0, 1)},
        1);
    auto a =
        kv ? m::KeyValueMemoryNetwork(p, facts, q, l) : m::AdjacentMemoryNetwork(p, facts, q, l);
    auto b = kv ? m::KeyValueMemoryNetwork(p, reverse, q, l)
                : m::AdjacentMemoryNetwork(p, reverse, q, l);
    Near(a.logits, b.logits, 2e-6);
  }
}
TEST(NnRecurrentExtensions, NtmSingleStepReproducesUnrollAndMasksInvalidSteps) {
  n::ParameterStore p(45103);
  auto              x = n::zeros({1, 9, 4}), l = n::from_vector(std::vector<std::int32_t>{1});
  auto              y = m::NtmSequence(p, x, l);
  EXPECT_EQ(p.count(), 851);
  m::NtmState initial{
      n::select(y.h, 1, 0),
      n::select(y.c, 1, 0),
      n::select(y.memory, 1, 0),
      n::select(y.read, 1, 0),
      n::select(y.read_weights, 1, 0),
      n::select(y.write_weights, 1, 0)};
  auto step = m::NtmCell(p, n::select(x, 1, 0), initial);
  Near(step.memory, n::select(y.memory, 1, 1), 0);
  Near(step.logits, n::select(y.logits, 1, 0), 0);
  Near(n::select(y.memory, 1, 3), n::select(y.memory, 1, 9), 0);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(y.logits, 1, 3, 9))).item(), 0);
}
TEST(NnRecurrentExtensions, InvalidExtensionOptionsRejectBeforeCreatingParameters) {
  n::ParameterStore p(1);
  auto x = n::zeros({1, 12, 2}), h = n::zeros({1, 12}), l = n::zeros({1}, n::DType::Int32);
  EXPECT_THROW(m::MaskedTruncatedSequence(p, x, l, {h}, -1), std::invalid_argument);
  EXPECT_THROW(m::MaskedDropoutSequence(p, x, l, {h}, n::ones({1, 1, 12})), std::invalid_argument);
  EXPECT_THROW(
      m::MaskedPhasedSequence(p, x, l, {h, h}, n::zeros({1, 12}), n::scalar(2.f)),
      std::invalid_argument);
  EXPECT_EQ(p.count(), 0);
  EXPECT_THROW(m::MakeRecurrentDropoutMask(0, 1, true), std::invalid_argument);
  EXPECT_THROW(m::RecurrentObservationTimes(0), std::invalid_argument);
}
