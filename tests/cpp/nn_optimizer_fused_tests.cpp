#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <datamunge/nn/nn.hpp>
#include <vector>
namespace nn = datamunge::nn;
namespace {
void near(const nn::Tensor& a, const nn::Tensor& b, double tol) {
  ASSERT_EQ(a.shape(), b.shape());
  const auto x = a.to_vector<double>(), y = b.to_vector<double>();
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_NEAR(x[i], y[i], tol) << i;
}
} // namespace
TEST(NnOptimFused, PackedUpdatesMatchGeneralStridedPathAndCheckpointContinuation) {
  for (auto dt : {nn::DType::Float32, nn::DType::Float64})
    for (auto kind :
         {nn::OptimizerKind::Sgd,
          nn::OptimizerKind::Momentum,
          nn::OptimizerKind::Nesterov,
          nn::OptimizerKind::Adagrad,
          nn::OptimizerKind::RMSProp,
          nn::OptimizerKind::Adam,
          nn::OptimizerKind::AdamW,
          nn::OptimizerKind::Nadam,
          nn::OptimizerKind::AMSGrad,
          nn::OptimizerKind::Adamax,
          nn::OptimizerKind::Lamb,
          nn::OptimizerKind::Lars,
          nn::OptimizerKind::Lion,
          nn::OptimizerKind::Adafactor,
          nn::OptimizerKind::Adadelta}) {
      auto packed  = nn::tensor({1, -2, 0.3, 0.7, 2, -0.1}, {2, 3}, dt).requires_grad_();
      auto strided = nn::transpose(nn::transpose(packed).contiguous()).detach().requires_grad_();
      ASSERT_FALSE(strided.is_contiguous());
      nn::OptimizerConfig c;
      c.kind    = kind;
      c.beta1   = 0.8;
      c.beta2   = 0.97;
      c.epsilon = 1e-7;
      if (kind == nn::OptimizerKind::AdamW || kind == nn::OptimizerKind::Lamb ||
          kind == nn::OptimizerKind::Lars || kind == nn::OptimizerKind::Lion)
        c.weight_decay = 0.1;
      nn::Optimizer fast({packed}, 0.03, c), general({strided}, 0.03, c);
      const double  tol = dt == nn::DType::Float32 ? 2e-6 : 1e-13;
      for (int step = 0; step < 7; ++step) {
        auto gradient = nn::tensor({0.2 * (step + 1), -0.13, 0, 1e-8, 0.8, -0.3}, {2, 3}, dt);
        fast.step({gradient});
        general.step({gradient});
        near(packed, strided, tol);
        const auto a = fast.state(), b = general.state();
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
          EXPECT_EQ(a[i].first, b[i].first);
          near(a[i].second, b[i].second, tol);
        }
      }
      auto          resumed_weight = packed.detach().clone().requires_grad_();
      nn::Optimizer resumed({resumed_weight}, 0.03, c);
      resumed.load_state(fast.state());
      auto gradient = nn::ones({2, 3}, dt);
      fast.step({gradient});
      resumed.step({gradient});
      near(packed, resumed_weight, 0);
    }
}
TEST(NnOptimFused, DuplicateParametersAndOverlappingGradientViewsStaySynchronous) {
  auto          w = nn::full({4}, 2, nn::DType::Float64).requires_grad_();
  nn::Optimizer duplicated({w, w}, 0.1);
  duplicated.step({w, w});
  near(w, nn::full({4}, 1.6, nn::DType::Float64), 1e-14);
  auto          base  = nn::tensor({1, 2, 3, 4, 5}, {5}, nn::DType::Float64);
  auto          left  = nn::slice_axis(base, 0, 0, 4).detach().requires_grad_();
  auto          right = nn::slice_axis(base, 0, 1, 5).detach().requires_grad_();
  nn::Optimizer overlapping({left, right}, 0.1);
  overlapping.step({right, left});
  near(base, nn::tensor({0.8, 1.6, 2.4, 3.2, 4.6}, {5}, nn::DType::Float64), 1e-14);
}
TEST(NnOptimFused, UpdatesInvalidateSavedGraphsAndSkipUndefinedGradients) {
  auto          w    = nn::ones({4}, nn::DType::Float64).requires_grad_();
  auto          loss = nn::sum(nn::square(w));
  nn::Optimizer opt({w}, 0.1);
  auto          version = w.impl()->storage->version;
  opt.step({nn::ones({4}, nn::DType::Float64)});
  EXPECT_EQ(w.impl()->storage->version, version + 1);
  EXPECT_THROW(nn::grad(loss, {w}), std::logic_error);
  auto before = w.clone();
  version     = w.impl()->storage->version;
  opt.step({nn::Tensor()});
  near(w, before, 0);
  EXPECT_EQ(w.impl()->storage->version, version);
  EXPECT_EQ(opt.iterations(), 2);
}

TEST(NnOptimFused, RestoredSlotShapeStillReceivesGeneralValidation) {
  for (auto kind : {nn::OptimizerKind::Momentum, nn::OptimizerKind::Adam}) {
    auto                w = nn::ones({2, 3}, nn::DType::Float64).requires_grad_();
    nn::OptimizerConfig c;
    c.kind = kind;
    nn::Optimizer     opt({w}, 0.1, c);
    const std::string name = kind == nn::OptimizerKind::Momentum ? "velocity/0" : "m/0";
    opt.load_state({{name, nn::ones({1, 2, 3}, nn::DType::Float64)}});
    EXPECT_THROW(opt.step({nn::ones({2, 3}, nn::DType::Float64)}), std::invalid_argument);
    near(w, nn::ones({2, 3}, nn::DType::Float64), 0);
  }
}

TEST(NnOptimFused, GlobalNormPreservesDoublePromotionAndStridedOrder) {
  auto   x       = nn::tensor({1e20, -3, 4, 1e-20}, {4}, nn::DType::Float32);
  auto   values  = x.to_vector<double>();
  double squared = 0;
  for (double value : values) {
    const double product = value * value;
    squared += product;
  }
  EXPECT_DOUBLE_EQ(nn::global_norm({x, nn::Tensor()}), std::sqrt(squared));
  auto strided = nn::transpose(nn::reshape(nn::arange(0, 12, 1, nn::DType::Float64), {3, 4}));
  EXPECT_DOUBLE_EQ(nn::global_norm({strided}), std::sqrt(506.0));
  EXPECT_DOUBLE_EQ(nn::global_norm({nn::empty({0}), nn::Tensor()}), 0.0);
  auto integers = nn::from_vector(std::vector<std::int64_t>{3, -4}, {2});
  EXPECT_DOUBLE_EQ(nn::global_norm({integers}), 5.0);
}
TEST(NnOptimFused, GradientAccumulationKeepsSnapshotsAndInputIndependent) {
  auto                    g = nn::ones({4}, nn::DType::Float64);
  nn::GradientAccumulator acc;
  acc.add({g});
  auto snapshot = acc.sum();
  acc.add({g});
  near(snapshot[0], g, 0);
  near(acc.sum()[0], nn::full({4}, 2, nn::DType::Float64), 0);
  g.fill_(7);
  near(acc.sum()[0], nn::full({4}, 2, nn::DType::Float64), 0);
  acc.add({g}, 2);
  near(acc.sum()[0], nn::full({4}, 16, nn::DType::Float64), 0);
}

TEST(NnOptimFused, AdafactorVectorAndHigherRankStateMatchStridedFallback) {
  for (auto dt : {nn::DType::Float32, nn::DType::Float64})
    for (const auto& shape : {nn::Shape{12}, nn::Shape{2, 2, 3}}) {
      auto       initial = nn::reshape(nn::add(nn::arange(0, 12, 1, dt), 1.0), shape);
      auto       packed  = initial.clone().requires_grad_();
      nn::Tensor strided;
      if (shape.size() == 1) {
        auto repeated = nn::reshape(nn::stack({initial, initial}, 1), {24});
        strided       = nn::slice_axis(repeated, 0, 0, 24, 2).detach().requires_grad_();
      } else
        strided = nn::transpose(nn::transpose(initial).contiguous()).detach().requires_grad_();
      ASSERT_FALSE(strided.is_contiguous());
      nn::OptimizerConfig c;
      c.kind        = nn::OptimizerKind::Adafactor;
      c.update_clip = 0.6;
      nn::Optimizer fast({packed}, 0.03, c), general({strided}, 0.03, c);
      for (int i = 0; i < 5; ++i) {
        auto g = nn::mul(initial, 0.1 * (i + 1));
        fast.step({g});
        general.step({g});
        near(packed, strided, dt == nn::DType::Float32 ? 2e-6 : 1e-12);
        auto a = fast.state(), b = general.state();
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t k = 0; k < a.size(); ++k) {
          EXPECT_EQ(a[k].first, b[k].first);
          near(a[k].second, b[k].second, dt == nn::DType::Float32 ? 2e-6 : 1e-12);
        }
      }
    }
}

TEST(NnOptimFused, RestoredDifferentiableSlotsLoseHistoryWhenUpdated) {
  nn::EnableGradGuard enabled;
  for (auto kind : {nn::OptimizerKind::Momentum, nn::OptimizerKind::Nesterov,
                   nn::OptimizerKind::Adagrad, nn::OptimizerKind::RMSProp,
                   nn::OptimizerKind::Adam, nn::OptimizerKind::AdamW,
                   nn::OptimizerKind::Nadam, nn::OptimizerKind::AMSGrad,
                   nn::OptimizerKind::Adamax, nn::OptimizerKind::Lamb,
                   nn::OptimizerKind::Lars, nn::OptimizerKind::Lion,
                   nn::OptimizerKind::Adafactor, nn::OptimizerKind::Adadelta}) {
    auto parameter = nn::ones({2, 3}).requires_grad_();
    auto gradient = nn::full({2, 3}, 0.1);
    nn::OptimizerConfig config; config.kind = kind;
    nn::Optimizer optimizer({parameter}, 0.01, config);
    optimizer.step({gradient});
    auto checkpoint = optimizer.state();
    for (auto& entry : checkpoint)
      if (entry.first.find('/') != std::string::npos) entry.second.requires_grad_(true);
    optimizer.load_state(checkpoint);
    optimizer.step({gradient});
    for (const auto& entry : optimizer.state())
      if (entry.first.find('/') != std::string::npos)
        EXPECT_FALSE(entry.second.requires_grad()) << static_cast<int>(kind) << " " << entry.first;
  }
}

TEST(NnOptimFused, ManyParametersGrowSlotsAndRestoreWithoutDanglingReferences) {
  std::vector<nn::Tensor> params, gradients;
  for (int i = 0; i < 257; ++i) {
    params.push_back(nn::full({1}, 1 + i * 0.01, nn::DType::Float64).requires_grad_());
    gradients.push_back(nn::full({1}, 0.25 + i * 0.001, nn::DType::Float64));
  }
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::AMSGrad;
  nn::Optimizer optimizer(params, 0.01, config);
  optimizer.step(gradients);
  const auto checkpoint = optimizer.state();
  ASSERT_EQ(checkpoint.size(), 2 + 3 * params.size());
  std::vector<nn::Tensor> copies;
  for (const auto& p : params) copies.push_back(p.clone().detach().requires_grad_());
  nn::Optimizer restored(copies, 0.01, config);
  restored.load_state(checkpoint);
  optimizer.step(gradients);
  restored.step(gradients);
  for (std::size_t i = 0; i < params.size(); ++i) {
    EXPECT_NEAR(params[i].item(), 1 + i * 0.01 - 0.02, 1e-8);
    EXPECT_EQ(params[i].item(), copies[i].item());
  }
  const auto expected = optimizer.state(), actual = restored.state();
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].first, expected[i].first);
    near(actual[i].second, expected[i].second, 1e-14);
  }
}

TEST(NnOptimFused, RestoreKeepsUnknownNamesAndUsesLastExactDuplicate) {
  auto p = nn::ones({1}, nn::DType::Float64).requires_grad_();
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer({p}, 0.01, config);
  optimizer.load_state({
      {"m/0", nn::full({1}, 0.1, p.dtype())},
      {"m/00", nn::full({1}, 999, p.dtype())},
      {"custom/0", nn::full({1}, 42, p.dtype())},
      {"v/0", nn::full({1}, 0.4, p.dtype())},
      {"m/0", nn::full({1}, 0.2, p.dtype())}});
  optimizer.step({nn::full({1}, 0.3, p.dtype())});
  const auto state = optimizer.state();
  ASSERT_EQ(state.size(), 7u);
  EXPECT_EQ(state[2].first, "m/0");
  EXPECT_DOUBLE_EQ(state[2].second.item(), 0.1);
  EXPECT_EQ(state[3].first, "m/00");
  EXPECT_DOUBLE_EQ(state[3].second.item(), 999);
  EXPECT_EQ(state[4].first, "custom/0");
  EXPECT_DOUBLE_EQ(state[4].second.item(), 42);
  EXPECT_NEAR(state[5].second.item(), 0.999 * 0.4 + 0.001 * 0.09, 1e-15);
  EXPECT_NEAR(state[6].second.item(), 0.9 * 0.2 + 0.1 * 0.3, 1e-15);
}
