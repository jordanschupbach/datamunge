#include <gtest/gtest.h>

#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/training.hpp>
#include <datamunge/nn/recurrent_data.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
n::Tensor Lengths(std::initializer_list<std::int32_t> rows) {
  return n::from_vector(std::vector<std::int32_t>(rows));
}
void Zero(n::ParameterStore& p, const std::string& name) {
  auto old = p.at(name);
  p.set(name, n::zeros(old.shape(), old.dtype()));
}
} // namespace
TEST(NnRecurrentComponents, MarkedFixturePreservesEveryDrawCueLabelAndPadding) {
  auto data     = m::MakeMarkedSequences(8, 26001, 10000);
  auto uniforms = data.uniforms.to_vector<double>();
  auto values   = data.inputs.to_vector<float>();
  auto lengths  = data.lengths.to_vector<std::int32_t>(),
       labels   = data.labels.to_vector<std::int32_t>(),
       ids      = data.source_ids.to_vector<std::int32_t>();
  std::mt19937 rng(26001);
  for (int row = 0; row < 8; ++row) {
    for (int j = 0; j < 14; ++j)
      EXPECT_EQ(uniforms[row * 14 + j], (double(rng()) + .5) / 4294967296.);
    int length = 4 + int(9 * uniforms[row * 14]), cue = int((length - 1) * uniforms[row * 14 + 1]);
    EXPECT_EQ(lengths[row], length);
    EXPECT_EQ(labels[row], row % 2);
    EXPECT_EQ(ids[row], 10000 + row);
    for (int t = 0; t < 12; ++t) {
      EXPECT_EQ(values[row * 24 + 2 * t + 1], t == cue ? 1.f : 0.f);
      if (t >= length)
        EXPECT_EQ(values[row * 24 + 2 * t], 0);
      else if (t == cue)
        EXPECT_EQ(values[row * 24 + 2 * t], row % 2 ? 1.f : -1.f);
      else
        EXPECT_EQ(values[row * 24 + 2 * t], uniforms[row * 14 + t + 2] < .5 ? -1.f : 1.f);
    }
  }
  EXPECT_TRUE(n::equal_bits(data.inputs, m::MakeMarkedSequences(8, 26001, 10000).inputs));
  EXPECT_FALSE(data.inputs.requires_grad());
}
TEST(NnRecurrentComponents, MovingDotFixtureTargetsTheNextTorusPosition) {
  auto      data = m::MakeMovingDotSequences(8, 37001, 20000);
  auto      u    = data.uniforms.to_vector<double>();
  auto      x = data.inputs.to_vector<float>(), target = data.labels.to_vector<float>();
  auto      lengths = data.lengths.to_vector<std::int32_t>();
  const int dy[] = {0, 0, 1, -1}, dx[] = {1, -1, 0, 0};
  for (int b = 0; b < 8; ++b) {
    int length = 2 + int(3 * u[b * 4]), y = int(4 * u[b * 4 + 1]), column = int(4 * u[b * 4 + 2]),
        direction = int(4 * u[b * 4 + 3]);
    EXPECT_EQ(lengths[b], length);
    for (int t = 0; t < 4; ++t)
      for (int at = 0; at < 16; ++at) {
        int position =
            ((y + dy[direction] * t + 16) % 4) * 4 + (column + dx[direction] * t + 16) % 4;
        EXPECT_EQ(x[b * 64 + t * 16 + at], t < length && at == position ? 1.f : 0.f);
      }
    int next =
        ((y + dy[direction] * length + 16) % 4) * 4 + (column + dx[direction] * length + 16) % 4;
    for (int at = 0; at < 16; ++at)
      EXPECT_EQ(target[b * 16 + at], at == next ? 1.f : 0.f);
  }
}
TEST(NnRecurrentComponents, GatedCensusAndInitializationOrderMatchExistingCanonicalCells) {
  for (auto kind :
       {m::MaskedGatedCell::Lstm,
        m::MaskedGatedCell::Gru,
        m::MaskedGatedCell::Peephole,
        m::MaskedGatedCell::LayerNorm}) {
    n::ParameterStore p(27003);
    auto              h = n::zeros({1, 12});
    auto         out = m::MaskedGatedSequence(p, n::zeros({1, 12, 2}), Lengths({0}), {h, h}, kind);
    std::int64_t count = kind == m::MaskedGatedCell::Lstm       ? 746
                         : kind == m::MaskedGatedCell::Gru      ? 566
                         : kind == m::MaskedGatedCell::Peephole ? 782
                                                                : 818;
    EXPECT_EQ(p.count(), count);
    EXPECT_EQ(out.states.shape(), (n::Shape{1, 13, 12}));
    EXPECT_EQ(out.cell_states.shape(), out.states.shape());
    if (kind == m::MaskedGatedCell::Lstm || kind == m::MaskedGatedCell::Gru) {
      n::ParameterStore q(27003);
      if (kind == m::MaskedGatedCell::Lstm)
        n::lstm_step(q, "cell", n::zeros({1, 2}), {h, h});
      else
        n::gru_step(q, "cell", n::zeros({1, 2}), h);
      q.get("head/W", {12, 2}, n::init::normal(.1));
      q.get("head/b", {2}, n::init::zeros());
      EXPECT_EQ(p.names(), q.names());
      for (auto name : p.names())
        EXPECT_TRUE(n::equal_bits(p.at(name), q.at(name))) << name;
      auto before = p.flatten();
      m::MaskedGatedSequence(p, n::zeros({1, 12, 2}), Lengths({0}), {h, h}, kind);
      EXPECT_EQ(before, p.flatten());
      EXPECT_TRUE(
          n::equal_bits(
              p.get("probe", {5}, n::init::normal(.1)), q.get("probe", {5}, n::init::normal(.1))));
    }
  }
}
TEST(NnRecurrentComponents, LstmZeroKernelsHaveHandComputedMemoryAndReadout) {
  n::ParameterStore p(1, n::DType::Float64);
  auto              x = n::zeros({1, 2}, n::DType::Float64), h = n::ones({1, 2}, n::DType::Float64),
       c = n::full({1, 2}, .4, n::DType::Float64).requires_grad_();
  m::LstmGateStep(p, "cell", x, 2, 2, {h, c});
  Zero(p, "cell/Wx");
  Zero(p, "cell/Wh");
  auto   out = m::LstmGateStep(p, "cell", x, 2, 2, {h, c});
  double f   = 1. / (1. + std::exp(-1.));
  EXPECT_NEAR(out.cell.to_vector<double>()[0], .4 * f, 1e-15);
  EXPECT_NEAR(out.hidden.to_vector<double>()[0], .5 * std::tanh(.4 * f), 1e-15);
  auto g = n::grad(n::sum(out.cell), {c})[0];
  for (double value : g.to_vector<double>())
    EXPECT_NEAR(value, f, 1e-15);
}
TEST(NnRecurrentComponents, GruResetsBeforeTheNonDiagonalRecurrentCandidateMatrix) {
  n::ParameterStore p(1, n::DType::Float64);
  auto              x = n::zeros({1, 2}, n::DType::Float64),
       h              = n::from_vector(std::vector<double>{1, 2}, {1, 2});
  m::GruGateStep(p, "cell", x, 2, 2, h);
  for (const auto& name : p.names())
    Zero(p, name);
  p.set("cell/b_zr", n::from_vector(std::vector<double>{0, 0, 0, -std::log(3.)}, {4}));
  p.set("cell/Wh_n", n::from_vector(std::vector<double>{1, 2, 3, 4}, {2, 2}));
  auto out    = m::GruGateStep(p, "cell", x, 2, 2, h);
  auto values = out.hidden.to_vector<double>();
  EXPECT_NEAR(values[0], .5 + .5 * std::tanh(2.), 1e-14);
  EXPECT_NEAR(values[1], 1. + .5 * std::tanh(3.), 1e-14);
  EXPECT_FALSE(out.cell.defined());
}
TEST(NnRecurrentComponents, PeepholeOutputUsesTheNewCellValue) {
  n::ParameterStore p(1, n::DType::Float64);
  auto x = n::zeros({1, 1}, n::DType::Float64), h = x, c = n::full({1, 1}, .4, n::DType::Float64);
  m::PeepholeGateStep(p, "cell", x, 1, 1, {h, c});
  Zero(p, "cell/Wx");
  Zero(p, "cell/Wh");
  p.set("cell/b", n::from_vector(std::vector<double>{0, 1, .2, 0}, {4}));
  p.set("cell/peephole_i", n::full({1}, .3, n::DType::Float64));
  p.set("cell/peephole_f", n::full({1}, .4, n::DType::Float64));
  p.set("cell/peephole_o", n::full({1}, .7, n::DType::Float64));
  auto   out     = m::PeepholeGateStep(p, "cell", x, 1, 1, {h, c});
  auto   sigmoid = [](double v) { return 1. / (1. + std::exp(-v)); };
  double cell    = .4 * sigmoid(1. + .4 * .4) + sigmoid(.4 * .3) * std::tanh(.2);
  EXPECT_NEAR(out.cell.item(), cell, 1e-14);
  EXPECT_NEAR(out.hidden.item(), sigmoid(.7 * cell) * std::tanh(cell), 1e-14);
  EXPECT_GT(std::abs(out.hidden.item() - sigmoid(.7 * .4) * std::tanh(cell)), 1e-4);
}
TEST(NnRecurrentComponents, LayerNormRetainsRawMemoryAndHasNoBatchStatistics) {
  n::ParameterStore p(7, n::DType::Float64);
  auto x = n::zeros({2, 2}, n::DType::Float64), h = n::zeros({2, 3}, n::DType::Float64),
       c = n::ones({2, 3}, n::DType::Float64);
  m::LayerNormGateStep(p, "cell", x, 2, 3, {h, c});
  Zero(p, "cell/Wx");
  Zero(p, "cell/Wh");
  auto first = m::LayerNormGateStep(p, "cell", x, 2, 3, {h, c});
  EXPECT_GT(first.cell.to_vector<double>()[0], .7);
  EXPECT_EQ(n::max(n::abs(first.normalized_cell)).item(), 0);
  EXPECT_EQ(n::max(n::abs(first.hidden)).item(), 0);
  auto second = m::LayerNormGateStep(p, "cell", x, 2, 3, {first.hidden, first.cell});
  EXPECT_GT(second.cell.to_vector<double>()[0], .5);
  n::ParameterStore q(27003, n::DType::Float64);
  auto              input = n::from_vector(std::vector<double>{.1, .2, .3, .4}, {2, 2}),
       altered            = n::from_vector(std::vector<double>{.1, .2, 123, -456}, {2, 2});
  auto a                  = m::LayerNormGateStep(q, "cell", input, 2, 3, {h, c}),
       b                  = m::LayerNormGateStep(q, "cell", altered, 2, 3, {h, c});
  EXPECT_TRUE(n::equal_bits(n::take_rows(a.hidden, {0}), n::take_rows(b.hidden, {0})));
  EXPECT_TRUE(
      n::equal_bits(
          n::take_rows(a.normalized_gate_affine, {0}),
          n::take_rows(b.normalized_gate_affine, {0})));
}
TEST(NnRecurrentComponents, EveryVectorFamilyHoldsPaddingAndHasZeroInvalidInputGradients) {
  using Path = std::function<m::MaskedRecurrentOutput(
      n::ParameterStore&, const n::Tensor&, const n::Tensor&, const std::vector<n::Tensor>&)>;
  struct Case {
    Path path;
    int  states;
  };
  std::vector<Case> cases{
      {m::MaskedTanhSequence, 1},
      {m::MaskedBidirectionalSequence, 2},
      {m::MaskedClockworkSequence, 1},
      {m::MaskedIndRnnSequence, 1},
      {m::MaskedMguSequence, 1}};
  for (auto kind :
       {m::MaskedGatedCell::Lstm,
        m::MaskedGatedCell::Gru,
        m::MaskedGatedCell::Peephole,
        m::MaskedGatedCell::LayerNorm})
    cases.push_back(
        {[kind](auto& p, const auto& x, const auto& l, const auto& h) {
           return m::MaskedGatedSequence(p, x, l, h, kind);
         },
         2});
  for (std::size_t index = 0; index < cases.size(); ++index) {
    auto              test = cases[index];
    n::ParameterStore p(27003, n::DType::Float64);
    auto x       = n::cast(m::MakeMarkedSequences(2).inputs, n::DType::Float64).requires_grad_();
    auto lengths = Lengths({0, 3});
    std::vector<n::Tensor> initial(test.states, n::full({2, 12}, .2, n::DType::Float64));
    auto                   out = test.path(p, x, lengths, initial);
    auto                   g   = n::grad(n::sum(out.logits), {x})[0].to_vector<double>();
    for (int t = 0; t < 12; ++t)
      for (int c = 0; c < 2; ++c) {
        EXPECT_EQ(g[t * 2 + c], 0) << index;
        if (t >= 3)
          EXPECT_EQ(g[24 + t * 2 + c], 0) << index;
      }
    auto padded = x.detach().to_vector<double>();
    for (int t = 0; t < 12; ++t)
      for (int c = 0; c < 2; ++c) {
        padded[2 * t + c] = 17;
        if (t >= 3)
          padded[24 + 2 * t + c] = 17;
      }
    auto altered = test.path(p, n::from_vector(padded, {2, 12, 2}), lengths, initial);
    EXPECT_TRUE(n::equal_bits(out.logits, altered.logits)) << index;
    auto states = out.states.to_vector<double>();
    for (int t = 0; t < 13; ++t)
      for (int unit = 0; unit < 12; ++unit)
        EXPECT_EQ(states[t * 12 + unit], .2) << index;
  }
}
TEST(NnRecurrentComponents, BidirectionalContextDistinguishesFutureValidTokensFromPadding) {
  n::ParameterStore p(28003, n::DType::Float64);
  auto              x = n::full({1, 12, 2}, .2, n::DType::Float64).requires_grad_(),
       h              = n::zeros({1, 12}, n::DType::Float64);
  auto out            = m::MaskedBidirectionalSequence(p, x, Lengths({4}), {h, h});
  auto forward =
           n::grad(n::sum(n::select(out.states, 1, 1)), {x}, n::GradOptions{true, false, false})[0],
       backward = n::grad(n::sum(n::select(out.backward_aligned, 1, 0)), {x})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(forward, 1, 1, 12))).item(), 0);
  EXPECT_GT(n::max(n::abs(n::slice_axis(backward, 1, 1, 4))).item(), 1e-8);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(backward, 1, 4, 12))).item(), 0);
  EXPECT_EQ(p.count(), 410);
}
TEST(NnRecurrentComponents, ClockworkMasksAndActivityMatchTheDiscreteSourceSchedule) {
  auto connections = m::ClockworkRecurrentConnectivity(),
       schedule    = m::ClockworkRecurrentSchedule();
  EXPECT_EQ(n::sum(connections).item(), 90);
  EXPECT_EQ(n::sum(schedule).item(), 69);
  auto table = connections.to_vector<float>();
  EXPECT_EQ(table[3], 0);
  EXPECT_EQ(table[3 * 12], 1);
  n::ParameterStore   p(32003, n::DType::Float64);
  std::vector<double> values(24, 0);
  for (int t = 0; t < 12; ++t)
    values[t * 2] = t * .1;
  auto x = n::from_vector(values, {1, 12, 2}), h = n::zeros({1, 12}, n::DType::Float64);
  m::MaskedClockworkSequence(p, x, Lengths({4}), {h});
  std::vector<double> weight(24, 0);
  for (int j = 0; j < 12; ++j)
    weight[j] = 1;
  p.set("cell/Wx", n::from_vector(weight, {2, 12}));
  Zero(p, "cell/Wh");
  Zero(p, "cell/b");
  auto out   = m::MaskedClockworkSequence(p, x, Lengths({4}), {h});
  auto final = out.final_state.to_vector<double>();
  EXPECT_NEAR(final[0], std::tanh(.3), 1e-15);
  EXPECT_NEAR(final[3], std::tanh(.2), 1e-15);
  EXPECT_EQ(final[6], 0);
  EXPECT_EQ(final[9], 0);
  auto g = n::grad(n::sum(out.logits), {p.at("cell/Wh")})[0].to_vector<double>();
  for (int at = 0; at < 144; ++at)
    if (!table[at])
      EXPECT_EQ(g[at], 0);
}
TEST(
    NnRecurrentComponents,
    IndependentRecurrenceHasClosedFormInitialStateDerivativeAndExplicitProjection) {
  n::ParameterStore p(35003, n::DType::Float64);
  auto              x = n::zeros({1, 12, 2}, n::DType::Float64),
       h              = n::ones({1, 12}, n::DType::Float64).requires_grad_();
  auto   out          = m::MaskedIndRnnSequence(p, x, Lengths({12}), {h});
  double u = p.at("cell/u").to_vector<double>()[0], b = p.at("cell/b").to_vector<double>()[0],
         value = 1;
  for (int t = 0; t < 12; ++t)
    value = u * value + b;
  EXPECT_NEAR(out.final_state.to_vector<double>()[0], value, 1e-15);
  auto g = n::grad(n::sum(n::select(out.final_state, 1, 0)), {h})[0].to_vector<double>();
  EXPECT_NEAR(g[0], std::pow(u, 12), 1e-14);
  for (int j = 1; j < 12; ++j)
    EXPECT_EQ(g[j], 0);
  p.set("cell/u", n::full({12}, 2, n::DType::Float64));
  float upper = std::pow(2.f, 1.f / 12);
  p.set("cell/u", n::clip(p.at("cell/u").detach(), 0., double(upper)));
  EXPECT_EQ(n::max(p.at("cell/u")).item(), double(upper));
  EXPECT_EQ(p.count(), 74);
}
TEST(NnRecurrentComponents, SpatialConvLstmKeepsItsLocalStencilAndLengthMasks) {
  n::ParameterStore p(37003, n::DType::Float64);
  auto              x = n::full({1, 4, 4, 4, 1}, .2, n::DType::Float64).requires_grad_(),
       h              = n::zeros({1, 4, 4, 2}, n::DType::Float64);
  auto out            = m::MaskedConvLstmSequence(p, x, Lengths({1}), {h, h});
  EXPECT_EQ(p.count(), 227);
  auto   first  = n::slice(out.states, {0, 1, 1, 1, 0}, {1, 1, 1, 1, 1});
  auto   g      = n::grad(n::sum(first), {x})[0].to_vector<double>();
  double inside = 0;
  for (int t = 0; t < 4; ++t)
    for (int y = 0; y < 4; ++y)
      for (int col = 0; col < 4; ++col) {
        double value = g[t * 16 + y * 4 + col];
        if (t || y > 2 || col > 2)
          EXPECT_EQ(value, 0);
        else
          inside = std::max(inside, std::abs(value));
      }
  EXPECT_GT(inside, 1e-8);
  auto empty = m::MaskedConvLstmSequence(p, x, Lengths({0}), {h + .2, h + .4});
  EXPECT_EQ(n::max(n::abs(empty.final_state - (h + .2))).item(), 0);
  EXPECT_EQ(n::max(n::abs(empty.final_cell - (h + .4))).item(), 0);
  EXPECT_EQ(n::max(n::abs(empty.gates)).item(), 0);
}
TEST(NnRecurrentComponents, GradientPoliciesExposeSourceScaleAndDirection) {
  auto raw      = n::from_vector(std::vector<float>{3, 4});
  auto original = raw.clone();
  auto none = m::RecurrentGradientControl({raw}, 0), global = m::RecurrentGradientControl({raw}, 1),
       value = m::RecurrentGradientControl({raw}, 2);
  EXPECT_EQ(global.raw_norm.item(), 5);
  EXPECT_NEAR(global.scale.item(), .01, 1e-9);
  EXPECT_NEAR(global.applied_norm.item(), .05, 1e-8);
  EXPECT_EQ(none.values[0].to_vector<float>(), (std::vector<float>{3, 4}));
  EXPECT_EQ(global.values[0].to_vector<float>(), (std::vector<float>{.03f, .04f}));
  EXPECT_EQ(value.values[0].to_vector<float>(), (std::vector<float>{.01f, .01f}));
  EXPECT_TRUE(n::equal_bits(raw, original));
  EXPECT_EQ(m::RecurrentGradientControl({n::zeros({3})}, 1).scale.item(), 1);
}
TEST(NnRecurrentComponents, LiveEndpointsOutliveStoreAndKeepHigherDerivatives) {
  auto                     x = n::full({1, 12, 2}, .2, n::DType::Float64).requires_grad_();
  m::MaskedRecurrentOutput out;
  {
    n::ParameterStore p(38003, n::DType::Float64);
    out = m::MaskedMguSequence(p, x, Lengths({4}), {n::zeros({1, 12}, n::DType::Float64)});
  }
  for (auto value : {out.logits, out.states, out.gates}) {
    auto g = n::grad(n::sum(n::square(value)), {x}, n::GradOptions{true, true, false})[0];
    EXPECT_GT(n::max(n::abs(g)).item(), 1e-9);
    auto higher = n::grad(n::sum(g), {x}, n::GradOptions{true, false, false})[0];
    EXPECT_TRUE(higher.defined());
  }
}
TEST(NnRecurrentComponents, CallerOwnedAdamUpdatesAVisibleMaskedTensorObjective) {
  n::ParameterStore p(27003);
  auto              data = m::MakeMarkedSequences(8);
  auto              h    = n::zeros({8, 12});
  m::MaskedGatedSequence(p, data.inputs, data.lengths, {h, h}, m::MaskedGatedCell::Gru);
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer optimizer(p, .01, config);
  auto         before = p.flatten();
  auto out  = m::MaskedGatedSequence(p, data.inputs, data.lengths, {h, h}, m::MaskedGatedCell::Gru);
  auto loss = -n::mean(n::sum(n::one_hot(data.labels, 2) * n::log_softmax(out.logits, -1), {1}));
  optimizer.step(n::grad(loss, p.parameters()));
  EXPECT_NE(before, p.flatten());
  EXPECT_EQ(optimizer.iterations(), 1);
}
TEST(NnRecurrentComponents, InvalidInputsRejectBeforeLeafCreationOrUnsafeDisplayConversion) {
  n::ParameterStore p(1);
  auto              x = n::zeros({1, 12, 2}), h = n::zeros({1, 12});
  EXPECT_THROW(m::MaskedTanhSequence(p, n::scalar(1), Lengths({1}), {h}), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  EXPECT_THROW(m::MaskedTanhSequence(p, x, Lengths({13}), {h}), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  EXPECT_THROW(
      m::MaskedGatedSequence(p, x, Lengths({1}), {h, h}, static_cast<m::MaskedGatedCell>(99)),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  EXPECT_THROW(
      m::LstmGateStep(p, "cell", n::zeros({1, 2}), 2, 12, {h, n::Tensor{}}), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  EXPECT_THROW(m::MakeMarkedSequences(0), std::invalid_argument);
  EXPECT_THROW(
      m::MakeMovingDotSequences(2, 1, std::numeric_limits<std::int32_t>::max()),
      std::invalid_argument);
  EXPECT_THROW(m::RecurrentGradientControl({}, 0), std::invalid_argument);
  EXPECT_THROW(m::RecurrentGradientControl({n::ones({1})}, 3), std::invalid_argument);
  EXPECT_THROW(
      m::MarkedSequenceHeatmap(n::full({1, 12, 2}, 1e300, n::DType::Float64), true),
      std::invalid_argument);
}
TEST(NnRecurrentComponents, NativeRastersPreserveSourceCueAndPanelPixelGeometry) {
  auto marked = m::MakeMarkedSequences(24);
  auto heat   = m::MarkedSequenceHeatmap(marked.inputs);
  EXPECT_EQ(heat.shape(), (n::Shape{384, 360, 3}));
  EXPECT_FALSE(heat.requires_grad());
  auto values = heat.to_vector<float>();
  auto x      = marked.inputs.to_vector<float>();
  int  cue    = -1;
  for (int t = 0; t < 12; ++t)
    if (x[t * 2 + 1] > 0)
      cue = t;
  ASSERT_GE(cue, 0);
  EXPECT_EQ(values[(cue * 30) * 3], 0);
  EXPECT_EQ(values[(15 * 360 + 29) * 3], 220);
  auto dot    = m::MakeMovingDotSequences(2);
  auto panels = m::MovingDotPanels(dot.inputs, dot.labels, n::zeros({2, 4, 4, 1}));
  EXPECT_EQ(panels.shape(), (n::Shape{50, 150, 3}));
  EXPECT_FALSE(panels.requires_grad());
  auto pixels = panels.to_vector<float>();
  EXPECT_EQ(pixels[(24 * 150 + 149) * 3], 180);
  EXPECT_EQ(pixels[(125) * 3], 0);
}
