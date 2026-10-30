
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
namespace {
void nontrivial(n::ParameterStore &p, n::ParameterStore &state) {
  n::NoGradGuard guard;
  for (const auto &name : p.names())
    if (name.find("/bn/") != std::string::npos) {
      const auto pattern = .2 * n::sin(.37 * n::arange(0, p.at(name).numel()).to(p.dtype()) +
                                       double(name.size()));
      p.set(name, pattern + (name.ends_with("gamma") ? 1. : .3));
    }
  for (const auto &name : state.names()) {
    const auto pattern =
        .2 * n::sin(.37 * n::arange(0, state.at(name).numel()).to(state.dtype()) +
                    double(name.size()));
    state.set(name, pattern + (name.ends_with("moving_var") ? 1. : 0.));
  }
}
} // namespace
TEST(NnRepVGGModels, TensorFoldingPreservesFloat64ValuesAndEverySourceDerivative) {
  for (const auto &b :
       std::vector<n::models::RepVGGBlockSpec>{{"identity", 8, 8, 1, 1, true},
                                               {"stride", 8, 12, 2, 1, false},
                                               {"grouped", 8, 8, 1, 2, true},
                                               {"group_stride", 8, 12, 2, 2, false}}) {
    const auto type = n::DType::Float64;
    n::ParameterStore p(17, type), state(17, type);
    const auto x = n::sin(.17 * n::arange(0, 2 * 7 * 6 * 8).to(type) + .3)
                       .reshape({2, 7, 6, 8})
                       .detach()
                       .requires_grad_();
    {
      n::NoGradGuard guard;
      n::models::RepVGGBlock(p, state, b, x, false, false);
    }
    nontrivial(p, state);
    const auto out = n::models::RepVGGBlock(p, state, b, x, false, false);
    auto vars = p.parameters();
    vars.insert(vars.begin(), x);
    const auto expected = n::grad(n::mean(n::square(out)), vars);
    const auto fused = n::models::FuseRepVGGBlock(b, p, state);
    EXPECT_EQ(fused.bias.dtype(), type);
    for (const auto &kernel : fused.kernels)
      EXPECT_EQ(kernel.dtype(), type);
    auto options = n::conv_options(b.stride, n::Padding::Explicit);
    options.explicit_paddings = {1, 1, 1, 1};
    options.groups = b.groups;
    const auto packed =
        n::relu(n::bias_add(n::conv2d(x, n::concat(fused.kernels, 3), options), fused.bias));
    EXPECT_LT(n::max(n::abs(out - packed)).item(), 1e-11);
    const auto actual = n::grad(n::mean(n::square(packed)), vars);
    for (size_t i = 0; i < vars.size(); ++i)
      EXPECT_LT(n::max(n::abs(actual[i] - expected[i])).item(), 1e-10);
  }
}
TEST(NnRepVGGModels, CompleteA0DeploymentCopiesWeightsWithoutMovingState) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::models::RepVGGConfig cfg;
  cfg.classes = 10;
  cfg.small_stem = true;
  const auto x = n::sin(.37 * n::arange(0, 1 * 32 * 32 * 3).to(n::DType::Float32))
                     .reshape({1, 32, 32, 3});
  n::NoGradGuard guard;
  n::models::RepVGG(p, state, x, cfg);
  nontrivial(p, state);
  const auto original = n::models::RepVGG(p, state, x, cfg);
  EXPECT_EQ(p.count(), 7840778);
  EXPECT_EQ(original.blocks.size(), 22);
  n::ParameterStore deployed(19), empty(19);
  n::models::DeployRepVGG(deployed, original.blocks, p, state);
  cfg.deploy = true;
  const auto converted = n::models::RepVGG(deployed, empty, x, cfg);
  EXPECT_EQ(deployed.count(), 7041194);
  EXPECT_EQ(empty.count(), 0);
  const double error = n::max(n::abs(original.logits - converted.logits)).item() /
                       std::max(1., n::max(n::abs(original.logits)).item());
  EXPECT_LT(error, 1e-4);
  const auto saved = deployed.at("output/W").clone();
  p.set("output/W", n::zeros_like(p.at("output/W")));
  EXPECT_EQ(n::max(n::abs(saved - deployed.at("output/W"))).item(), 0);
}
TEST(NnRepVGGModels, CompleteTrainingGradientAndFrozenBnState) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::models::RepVGGConfig cfg;
  cfg.classes = 10;
  cfg.small_stem = true;
  cfg.update_stats = false;
  const auto x = n::sin(.37 * n::arange(0, 2 * 32 * 32 * 3).to(n::DType::Float32))
                     .reshape({2, 32, 32, 3});
  {
    n::NoGradGuard guard;
    n::models::RepVGG(p, state, x, cfg);
  }
  const auto before = state.flatten();
  const auto out = n::models::RepVGG(p, state, x, cfg, true);
  for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  EXPECT_EQ(state.flatten(), before);
}
TEST(NnRepVGGModels, InvalidGroupsModesAndMissingFusionStateFailExplicitly) {
  EXPECT_THROW((n::models::RepVGGBlockSpec{"b", 8, 8, 1, 0, true}.Validate()),
               std::invalid_argument);
  EXPECT_THROW((n::models::RepVGGBlockSpec{"b", 8, 12, 1, 2, true}.Validate()),
               std::invalid_argument);
  n::ParameterStore p(17), state(17);
  const n::models::RepVGGBlockSpec b{"b", 8, 8, 1, 2, true};
  EXPECT_THROW(n::models::RepVGGBlock(p, state, b, n::ones({1, 5, 5, 8}), true, true),
               std::invalid_argument);
  EXPECT_THROW(n::models::FuseRepVGGBlock(b, p, state), std::runtime_error);
  n::ParameterStore different(17, n::DType::Float64);
  EXPECT_THROW(n::models::DeployRepVGG(different, {b}, p, state), std::invalid_argument);
  EXPECT_THROW(n::models::RepVGGLayout("unknown"), std::invalid_argument);
}
