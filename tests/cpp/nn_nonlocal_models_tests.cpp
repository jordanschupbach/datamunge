
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
namespace {
n::models::NonLocalOutputs independent(const n::ParameterStore &p,
                                       const n::ParameterStore &state, const n::Tensor &x,
                                       n::models::Pairwise mode, bool pooled) {
  const int64_t batch = x.size(0), channels = x.size(3), inner = channels / 2;
  auto flat = [&](const auto &t) {
    return t.reshape({batch, -1, t.size(3)});
  };
  const auto keys = pooled ? n::max_pool2d(x, 2, 2, n::Padding::Valid) : x;
  auto linear = [&](const std::string &name, const n::Tensor &t, int64_t width) {
    return n::matmul(flat(t), p.at("nl/" + name + "/W").reshape({t.size(3), width})) +
           p.at("nl/" + name + "/b");
  };
  const auto g = linear("g", keys, inner);
  n::Tensor affinity;
  if (mode == n::models::Pairwise::kGaussian)
    affinity = n::matmul(flat(x), flat(keys), false, true);
  else {
    const auto theta = linear("theta", x, inner), phi = linear("phi", keys, inner);
    if (mode == n::models::Pairwise::kConcatenation)
      affinity = n::relu(n::matmul(theta, p.at("nl/concat/W1")) +
                         n::transpose(n::matmul(phi, p.at("nl/concat/W2")), {0, 2, 1}));
    else {
      affinity = n::matmul(theta, phi, false, true);
      if (mode == n::models::Pairwise::kEmbeddedGaussian)
        affinity = affinity / std::sqrt(double(inner));
    }
  }
  const auto weights = (mode == n::models::Pairwise::kGaussian ||
                        mode == n::models::Pairwise::kEmbeddedGaussian)
                           ? n::softmax(affinity, -1)
                           : affinity / double(g.size(1));
  const auto response =
      n::matmul(n::matmul(weights, g), p.at("nl/W/W").reshape({inner, channels})) +
      p.at("nl/W/b");
  const auto normalized = (response - state.at("nl/bn/moving_mean")) *
                              n::rsqrt(state.at("nl/bn/moving_var") + 1e-5) *
                              p.at("nl/bn/gamma") +
                          p.at("nl/bn/beta");
  return {weights, x + normalized.reshape(x.shape())};
}
} // namespace
TEST(NnNonLocalModels, AllFourAffinitiesWithBothPoolModesMatchDenseValuesAndGradients) {
  const auto type = n::DType::Float64;
  for (auto mode : {n::models::Pairwise::kGaussian, n::models::Pairwise::kEmbeddedGaussian,
                    n::models::Pairwise::kDotProduct, n::models::Pairwise::kConcatenation})
    for (bool pool : {false, true}) {
      n::ParameterStore p(17, type), state(17, type);
      const auto x = n::sin(.37 * n::arange(0, 2 * 4 * 4 * 4).to(type) + .2)
                         .reshape({2, 4, 4, 4})
                         .detach()
                         .requires_grad_();
      {
        n::NoGradGuard guard;
        n::models::NonLocalBlock(p, state, "nl", x, mode, pool, false);
        p.set("nl/bn/gamma", n::full({4}, .7, type));
        p.set("nl/bn/beta", n::full({4}, .2, type));
        state.set("nl/bn/moving_mean", n::full({4}, -.1, type));
        state.set("nl/bn/moving_var", n::full({4}, 1.3, type));
      }
      const auto actual = n::models::NonLocalBlock(p, state, "nl", x, mode, pool, false);
      auto vars = p.parameters();
      vars.insert(vars.begin(), x);
      const auto a = n::grad(n::mean(n::square(actual.output)), vars);
      const auto expected = independent(p, state, x, mode, pool);
      EXPECT_LT(n::max(n::abs(actual.output - expected.output)).item(), 1e-12);
      EXPECT_LT(n::max(n::abs(actual.attention - expected.attention)).item(), 1e-12);
      const auto b = n::grad(n::mean(n::square(expected.output)), vars);
      for (size_t i = 0; i < a.size(); ++i)
        EXPECT_LT(n::max(n::abs(a[i] - b[i])).item(), 1e-11);
      EXPECT_EQ(actual.attention.shape(), n::Shape({2, 16, pool ? 4 : 16}));
      if (mode == n::models::Pairwise::kGaussian ||
          mode == n::models::Pairwise::kEmbeddedGaussian)
        EXPECT_LT(n::max(n::abs(n::sum(actual.attention, {2}) - 1)).item(), 1e-12);
    }
}
TEST(NnNonLocalModels, ZeroGammaIsExactIdentityThenActiveSmoothResponsesReachAllSites) {
  const auto type = n::DType::Float64;
  for (auto mode : {n::models::Pairwise::kGaussian, n::models::Pairwise::kEmbeddedGaussian,
                    n::models::Pairwise::kDotProduct}) {
    n::ParameterStore p(17, type), state(17, type);
    const auto x = n::sin(.37 * n::arange(0, 5 * 5 * 4).to(type) + .2)
                       .reshape({1, 5, 5, 4})
                       .detach()
                       .requires_grad_();
    const auto fresh = n::models::NonLocalBlock(p, state, "nl", x, mode, false, false);
    EXPECT_EQ(n::max(n::abs(fresh.output - x)).item(), 0);
    EXPECT_EQ(n::max(n::abs(n::grad(n::sum(fresh.output), {x})[0] - 1)).item(), 0);
    p.set("nl/bn/gamma", n::ones({4}, type));
    const auto active = n::models::NonLocalBlock(p, state, "nl", x, mode, false, false);
    const auto picked = n::slice(active.output, {0, 2, 2, 0}, {1, 1, 1, 4});
    const auto derivative = n::grad(n::sum(picked), {x})[0];
    EXPECT_EQ(n::count_nonzero(n::greater(n::sum(n::abs(derivative), {0, 3}), 1e-14)).item(),
              25);
  }
}
TEST(NnNonLocalModels, FiveHookPlacementsCountsAndFrozenCompleteBackward) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::models::AttentionResNetConfig cfg;
  cfg.classes = 10;
  cfg.width_divisor = 4;
  cfg.small_stem = true;
  cfg.stride_on_3x3 = false;
  cfg.update_stats = false;
  cfg.after_block = n::models::NonLocalResNetHook(.9, false);
  const auto x = n::sin(.37 * n::arange(0, 2 * 32 * 32 * 3).to(n::DType::Float32))
                     .reshape({2, 32, 32, 3});
  {
    n::NoGradGuard guard;
    n::models::AttentionResNet(p, state, x, cfg);
    int modules = 0;
    for (const auto &name : p.names())
      if (name.ends_with("non_local/bn/gamma")) {
        ++modules;
        p.set(name, n::full(p.at(name).shape(), .1));
      }
    EXPECT_EQ(modules, 5);
  }
  EXPECT_EQ(p.count(), 1947546);
  EXPECT_EQ(state.count(), 15328);
  const auto before = state.flatten();
  const auto out = n::models::AttentionResNet(p, state, x, cfg, true);
  for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  EXPECT_EQ(before, state.flatten());
  n::NoGradGuard guard;
  n::ParameterStore full(19), moving(19);
  n::models::AttentionResNetConfig large;
  large.after_block = n::models::NonLocalResNetHook(.9, false);
  n::models::AttentionResNet(full, moving, n::zeros({1, 32, 32, 3}), large);
  EXPECT_EQ(full.count(), 32915496);
}
TEST(NnNonLocalModels, InvalidAxesModesAndPoolGeometryAreRejected) {
  n::ParameterStore p(17), state(17);
  EXPECT_THROW(n::models::NonLocalBlock(p, state, "nl", n::ones({4}),
                                        n::models::Pairwise::kGaussian, false, false),
               std::invalid_argument);
  EXPECT_THROW(n::models::NonLocalBlock(p, state, "nl", n::ones({1, 1, 4, 4}),
                                        n::models::Pairwise::kGaussian, true, false),
               std::invalid_argument);
  EXPECT_THROW(n::models::NonLocalBlock(p, state, "nl", n::ones({1, 4, 4, 4}),
                                        static_cast<n::models::Pairwise>(9), false, false),
               std::invalid_argument);
  EXPECT_THROW(n::models::NonLocalResNetHook(1, false), std::invalid_argument);
}
