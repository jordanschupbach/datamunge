
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnReversibleModels, NontrivialBasicAndBottleneckInversesInBothBnModes) {
  const auto type = n::DType::Float64;
  for (bool bottle : {false, true})
    for (bool training : {false, true}) {
      n::ParameterStore p(1, type), state(1, type);
      const n::models::RevContext c{p, state, training, .9, bottle, false};
      const auto value = n::sin(.37 * n::arange(0, 800).to(type) + .2).reshape({4, 5, 5, 8});
      const auto halves = n::split(value, 2, 3);
      const n::models::RevPair x{halves[0], halves[1]};
      n::NoGradGuard guard;
      const auto y = n::models::RevBlockForward(c, "u", x, 8, 8, 1, false);
      const auto recovered = n::models::RevBlockInverse(c, "u", y, 8);
      EXPECT_GT(n::max(n::abs(y.first - x.first)).item(), .01);
      EXPECT_LT(
          n::max(n::abs(n::concat({recovered.first, recovered.second}, 3) - value)).item(),
          1e-11);
    }
}
TEST(NnReversibleModels, StoredAndReconstructedFullModelGradientsMatchFloat64Autograd) {
  n::set_num_threads(2);
  const auto type = n::DType::Float64;
  using Mode = n::models::RevNetMode;
  for (bool bottle : {false, true}) {
    n::models::RevNetConfig cfg;
    cfg.units = {2, 2, 2};
    cfg.filters = bottle ? std::vector<int>{8, 2, 4, 6} : std::vector<int>{8, 8, 16, 24};
    cfg.bottleneck = bottle;
    cfg.classes = 3;
    n::ParameterStore p(17, type), state(17, type);
    n::Generator gen(29);
    const auto x = n::normal({2, 8, 8, 3}, 0, .2, gen, type);
    const auto ordinary = n::models::RevNet(p, state, x, cfg, true, Mode::Ordinary, false);
    const auto expected = n::grad(n::mean(n::square(ordinary.logits)), p.parameters());
    for (Mode mode : {Mode::StoredUnits, Mode::Reversible}) {
      const auto out = n::models::RevNet(p, state, x, cfg, true, mode, false);
      EXPECT_LT(n::max(n::abs(out.logits - ordinary.logits)).item(), 1e-12);
      const auto actual = n::grad(n::mean(n::square(out.logits)), p.parameters());
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_LT(n::max(n::abs(actual[i] - expected[i])).item(), 1e-8) << p.names()[i];
    }
  }
}
TEST(NnReversibleModels, MovingMomentsUpdateOnceAndNeverDuringRecomputation) {
  n::set_num_threads(2);
  using Mode = n::models::RevNetMode;
  n::models::RevNetConfig cfg;
  cfg.units = {2, 2, 2};
  cfg.filters = {8, 8, 16, 24};
  cfg.classes = 3;
  n::ParameterStore p(17), state(17);
  n::Generator gen(29);
  const auto x = n::normal({2, 8, 8, 3}, 0, .2, gen);
  {
    n::NoGradGuard guard;
    n::models::RevNet(p, state, x, cfg);
  }
  std::vector<double> reference;
  for (Mode mode : {Mode::Ordinary, Mode::StoredUnits, Mode::Reversible}) {
    auto q = p.clone(), moments = state.clone();
    const auto out = n::models::RevNet(q, moments, x, cfg, true, mode);
    const auto once = moments.flatten();
    n::grad(n::mean(n::square(out.logits)), q.parameters());
    EXPECT_EQ(moments.flatten(), once);
    if (mode == Mode::Ordinary)
      reference = once;
    else
      EXPECT_EQ(once, reference);
  }
}
TEST(NnReversibleModels, ReleasedCifarCountAndConservativeStoredEntries) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  const auto cfg = n::models::RevNetConfig::Cifar(38);
  n::NoGradGuard guard;
  const auto out = n::models::RevNet(p, state, n::zeros({1, 8, 8, 3}), cfg);
  EXPECT_EQ(p.count(), 464890);
  EXPECT_EQ(state.count(), 2672);
  ASSERT_EQ(out.units.size(), 9);
  int stored = 0;
  for (const auto &u : out.units) {
    stored += u.stored;
    EXPECT_TRUE(u.input.first.defined());
    EXPECT_EQ(u.input.first.size(3), u.in / 2);
    EXPECT_EQ(u.output.first.size(3), u.out / 2);
  }
  EXPECT_EQ(stored, 3);
  EXPECT_TRUE(out.units[0].no_activation);
  EXPECT_FALSE(out.units[1].no_activation);
  EXPECT_EQ(n::models::RevNetConfig::ImageNet(56).Widths(),
            std::vector<int>({128, 512, 1024, 2048, 3328}));
}
TEST(NnReversibleModels, RejectsInvalidLayoutsModesAndPairGeometry) {
  n::models::RevNetConfig cfg;
  cfg.units.clear();
  EXPECT_THROW(cfg.Widths(), std::invalid_argument);
  cfg = {};
  cfg.filters[1] = 34;
  EXPECT_THROW(cfg.Widths(), std::invalid_argument);
  EXPECT_THROW(n::models::RevNetConfig::Cifar(50), std::invalid_argument);
  EXPECT_THROW(n::models::RevNetConfig::ImageNet(38), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  const n::models::RevContext c{p, state, false, .9, false, false};
  const n::models::RevPair pair{n::ones({1, 5, 5, 4}), n::ones({1, 5, 5, 4})};
  EXPECT_THROW(n::models::RevBlockForward(c, "u", pair, 8, 8, 2, false),
               std::invalid_argument);
  EXPECT_THROW(n::models::RevBlockInverse(c, "u", pair, 6), std::invalid_argument);
  EXPECT_THROW(n::models::RevNet(p, state, n::ones({1, 8, 8, 3}), {}, false,
                                 static_cast<n::models::RevNetMode>(5)),
               std::invalid_argument);
}
