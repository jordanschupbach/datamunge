#include <print>
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnGroupedWideModels, GroupedAndAggregatedValuesAndAllDerivatives) {
  namespace nn = datamunge::nn;
  nn::set_num_threads(2);
  for (int stride : {1, 2}) {
    nn::ParameterStore p(17), state(17);
    nn::models::ResNetNormConfig cfg;
    cfg.scale = true;
    cfg.epsilon = 1e-5;
    const auto input_values =
        .2 + .3 * nn::sin(.17 * nn::arange(0, 400).to(nn::DType::Float32));
    nn::Tensor input = input_values.reshape({2, 5, 5, 8});
    {
      nn::NoGradGuard guard; // create the parameters, then set nontrivial normalization state
      nn::models::ResNeXtBlock(p, state, "block", input, 2, 2, 64, cfg, stride);
      for (const auto &name : state.names()) {
        const auto count = state.at(name).numel();
        const auto i = nn::arange(0, count).to(nn::DType::Float32);
        state.set(name,
                  name.find("moving_var") != std::string::npos ? .7 + .03 * i : .03 * (i - 2));
      }
      for (const auto &name : p.names())
        if (name.find("/bn/") != std::string::npos) {
          const auto count = p.at(name).numel();
          const auto i = nn::arange(0, count).to(nn::DType::Float32);
          p.set(name, name.find("gamma") != std::string::npos ? .8 + .03 * i : .1 + .01 * i);
        }
    }
    input.requires_grad_(true);
    const auto grouped =
        nn::models::ResNeXtBlock(p, state, "block", input, 2, 2, 64, cfg, stride);
    auto norm = [&](const nn::Tensor &x, const std::string &layer, std::int64_t start,
                    std::int64_t count) {
      const auto name = "block/" + layer + "/bn/";
      auto slice = [&](const nn::ParameterStore &store, const std::string &part) {
        return nn::slice_axis(store.at(name + part), 0, start, start + count);
      };
      return (x - slice(state, "moving_mean")) *
                 nn::rsqrt(slice(state, "moving_var") + cfg.epsilon) * slice(p, "gamma") +
             slice(p, "beta");
    };
    std::vector<nn::Tensor> branches;
    for (int group = 0; group < 2; ++group) {
      const nn::Tensor first =
          nn::slice_axis(p.at("block/conv1/W"), 3, 2 * group, 2 * group + 2);
      nn::Tensor h =
          nn::relu(norm(nn::conv2d(input, first, nn::conv_options(1, nn::Padding::Valid)),
                        "conv1", 2 * group, 2));
      const nn::Tensor middle =
          nn::slice_axis(p.at("block/conv2/W"), 3, 2 * group, 2 * group + 2);
      h = nn::conv2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), middle,
                     nn::conv_options(stride, nn::Padding::Valid));
      h = nn::relu(norm(h, "conv2", 2 * group, 2));
      const nn::Tensor last =
          nn::slice_axis(p.at("block/conv3/W"), 2, 2 * group, 2 * group + 2);
      branches.push_back(nn::conv2d(h, last, nn::conv_options(1, nn::Padding::Valid)));
    }
    nn::Tensor shortcut = input;
    if (stride == 2)
      shortcut = norm(
          nn::conv2d(input, p.at("block/shortcut/W"), nn::conv_options(2, nn::Padding::Valid)),
          "shortcut", 0, 8);
    const nn::Tensor aggregated =
        nn::relu(shortcut + norm(branches[0] + branches[1], "conv3", 0, 8));
    auto variables = p.parameters();
    variables.insert(variables.begin(), input);
    const auto ga = nn::grad(nn::mean(nn::square(grouped.output)), variables);
    const auto gb = nn::grad(nn::mean(nn::square(aggregated)), variables);
    std::vector<std::pair<nn::Tensor, nn::Tensor>> pairs = {{grouped.output, aggregated}};
    for (std::size_t i = 0; i < ga.size(); ++i)
      pairs.emplace_back(ga[i], gb[i]);
    double error = 0;
    for (const auto &[x, y] : pairs) {
      if (x.shape() != y.shape())
        throw std::runtime_error("aggregation value or derivative mismatch");
      const auto a = x.to_vector<float>(), b = y.to_vector<float>();
      for (std::size_t i = 0; i < a.size(); ++i)
        error = std::max(error,
                         std::abs(double(a[i]) - b[i]) /
                             std::max({1., std::abs(double(a[i])), std::abs(double(b[i]))}));
    }
    if (error > 3e-5)
      throw std::runtime_error("aggregation value or derivative mismatch");
    std::print("Stride {}: grouped versus split-transform-sum outputs and all input/parameter "
               "derivatives agree; relative error {:.9f}.\n",
               stride, error);
  }
}
TEST(NnGroupedWideModels, DropoutValuesGradientsFreshMasksAndSignedIdentity) {
  namespace n = datamunge::nn;
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::Generator gen(17);
  const auto x = n::sin(.17 * n::arange(0, 2048).to(n::DType::Float32))
                     .reshape({8, 8, 8, 4})
                     .detach()
                     .requires_grad_();
  const n::models::PreactivationNormConfig train{true, false, .9, 1e-5},
      infer{false, false, .9, 1e-5};
  auto block = [&](const auto &norm, double rate) {
    return n::models::WideBasicBlock(p, state, "b", x, 4, 1, norm, rate, gen);
  };
  const auto first = block(train, .3);
  const auto before = first.before_dropout.to_vector<float>(),
             after = first.after_dropout.to_vector<float>();
  const auto dx =
      n::grad(n::sum(first.after_dropout), {first.before_dropout})[0].to_vector<float>();
  int positive = 0, removed = 0;
  for (size_t i = 0; i < before.size(); ++i) {
    const bool dropped = after[i] == 0;
    if (before[i] > 1e-6) {
      ++positive;
      removed += dropped;
      if (std::abs(after[i] - (dropped ? 0 : before[i] / .7)) > 2e-6 ||
          std::abs(dx[i] - (dropped ? 0 : 1 / .7)) > 2e-6)
        throw std::runtime_error("dropout or shortcut invariant failed");
    }
  }
  const auto repeated = block(train, .3);
  if (n::sum(n::abs(repeated.after_dropout - first.after_dropout)).item() < 1)
    throw std::runtime_error("dropout or shortcut invariant failed");
  const auto prediction = block(infer, .3).output;
  if (n::max(n::abs(prediction - block(infer, 0).output)).item() != 0 ||
      n::max(n::abs(prediction - block(infer, .3).output)).item() != 0)
    throw std::runtime_error("dropout or shortcut invariant failed");
  if (positive < 500 || std::abs(double(removed) / positive - .3) > .06)
    throw std::runtime_error("dropout or shortcut invariant failed");
  p.set("b/conv2/W", n::zeros_like(p.at("b/conv2/W")));
  const auto identity = block(train, .3);
  if (n::max(n::abs(identity.output - x)).item() != 0 ||
      n::max(n::abs(n::grad(n::sum(identity.output), {x})[0] - 1)).item() != 0)
    throw std::runtime_error("dropout or shortcut invariant failed");
}

TEST(NnGroupedWideModels, CompactResNeXtCountsStagesGradientsAndState) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::Generator rng(29);
  n::models::ResNeXtConfig cfg;
  cfg.cifar_layout = true;
  cfg.depth = 29;
  cfg.cardinality = 8;
  cfg.base_width = 16;
  cfg.stage_width = 16;
  cfg.classes = 10;
  cfg.update_stats = false;
  const auto x = n::normal({2, 16, 16, 3}, 0, .2, rng);
  const auto out = n::models::ResNeXt(p, state, x, cfg, true);
  EXPECT_EQ(p.count(), 359898);
  EXPECT_EQ(state.count(), 6304);
  EXPECT_EQ(out.blocks.size(), 9);
  ASSERT_EQ(out.stages.size(), 3);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(out.stages[i].shape(), n::Shape({2, 16 / (1 << i), 16 / (1 << i), 64 << i}));
  for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto saved = state.flatten();
  n::NoGradGuard guard;
  n::models::ResNeXt(p, state, x, cfg);
  EXPECT_EQ(state.flatten(), saved);
  cfg.update_stats = true;
  n::models::ResNeXt(p, state, x, cfg, true);
  EXPECT_NE(state.flatten(), saved);
}
TEST(NnGroupedWideModels, WideCountsStagesGradientsAndState) {
  n::set_num_threads(2);
  for (int k : {1, 2}) {
    n::ParameterStore p(17), state(17);
    n::Generator gen(17), rng(29);
    n::models::WideResNetConfig cfg;
    cfg.depth = 16;
    cfg.widen_factor = k;
    cfg.update_stats = false;
    const auto x = n::normal({2, 16, 16, 3}, 0, .2, rng);
    const auto out = n::models::WideResNet(p, state, x, cfg, true, gen);
    EXPECT_EQ(p.count(), k == 1 ? 175066 : 691674);
    EXPECT_EQ(state.count(), k == 1 ? 928 : 1824);
    EXPECT_EQ(out.blocks.size(), 6);
    for (int i = 0; i < 3; ++i)
      EXPECT_EQ(out.stages[i].shape(),
                n::Shape({2, 16 / (1 << i), 16 / (1 << i), (16 << i) * k}));
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    const auto saved = state.flatten();
    n::NoGradGuard guard;
    n::models::WideResNet(p, state, x, cfg, false, gen);
    EXPECT_EQ(state.flatten(), saved);
    cfg.update_stats = true;
    n::models::WideResNet(p, state, x, cfg, true, gen);
    EXPECT_NE(state.flatten(), saved);
  }
}
TEST(NnGroupedWideModels, ResNeXtRejectsInvalidLayoutsAndShapes) {
  n::models::ResNeXtConfig cfg;
  cfg.depth = 18;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg = {};
  cfg.cardinality = 0;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg = {};
  cfg.cifar_layout = true;
  cfg.depth = 29;
  cfg.stage_width = 1;
  cfg.base_width = 1;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  n::models::ResNetNormConfig norm;
  EXPECT_THROW(n::models::ResNeXtBlock(p, state, "b", n::ones({2, 4}), 4, 2, 16, norm),
               std::invalid_argument);
  EXPECT_THROW(n::models::ResNeXtBlock(p, state, "b", n::ones({1, 4, 4, 8}), 4, 2, 1, norm),
               std::invalid_argument);
}
TEST(NnGroupedWideModels, WideRejectsInvalidLayoutsAndShapes) {
  n::models::WideResNetConfig cfg;
  cfg.depth = 12;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg = {};
  cfg.dropout = 1;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg = {};
  cfg.momentum = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  n::Generator gen(17);
  n::models::PreactivationNormConfig norm;
  EXPECT_THROW(n::models::WideBasicBlock(p, state, "b", n::ones({1, 4}), 4, 1, norm, 0, gen),
               std::invalid_argument);
  EXPECT_THROW(
      n::models::WideBasicBlock(p, state, "b", n::ones({1, 4, 4, 4}), 4, 3, norm, 0, gen),
      std::invalid_argument);
}
