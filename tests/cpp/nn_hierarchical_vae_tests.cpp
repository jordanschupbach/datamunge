#include <gtest/gtest.h>

#include <chrono>
#include <datamunge/nn/hierarchical_variational_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnHierarchy, ResidualGaussianAlgebraAllFourKlFirstAndSecondDerivatives) {
  const auto t  = n::DType::Float64;
  auto       pm = n::tensor({1., -2., .5}, {1, 3}, t).requires_grad_(),
       pl       = n::tensor({.2, -.4, .6}, {1, 3}, t).requires_grad_();
  auto hm = n::full({1, 3}, .2, t).requires_grad_(), hl = n::full({1, 3}, .1, t).requires_grad_();
  const auto same = m::HierarchyPosterior(pm, pl, n::zeros({1, 3}, t), n::zeros({1, 3}, t));
  EXPECT_EQ(same.mu.to_vector<double>(), pm.to_vector<double>());
  EXPECT_EQ(same.log_variance.to_vector<double>(), pl.to_vector<double>());
  EXPECT_EQ(m::ConditionalGaussianKl(same.mu, same.log_variance, pm, pl).item(), 0);
  const auto q        = m::HierarchyPosterior(pm, pl, hm, hl);
  double     expected = 0;
  for (double value : pl.to_vector<double>())
    expected += .5 * (-.1 + std::exp(.1) + .04 * std::exp(-value) - 1);
  EXPECT_NEAR(m::ConditionalGaussianKl(q.mu, q.log_variance, pm, pl).item(), expected, 1e-14);
  const auto direct = m::HierarchyPosterior(pm, pl, hm, hl, false);
  EXPECT_EQ(direct.mu.to_vector<double>(), hm.to_vector<double>());
  n::TensorFunction f = [](const auto& a) {
    return n::sum(m::ConditionalGaussianKl(a[0], a[1], a[2], a[3]));
  };
  EXPECT_TRUE(n::gradcheck(f, {hm, hl, pm, pl}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {hm, hl, pm, pl}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnHierarchy, CellCensusExplicitMomentUpdatesAndStateExcludedFromWeightOptimization) {
  n::ParameterStore  p(2609), state(0);
  std::vector<float> v(2 * 2 * 2 * 4);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = float(.1 + .01 * i);
  const auto x = n::from_vector(v, {2, 2, 2, 4});
  m::HierarchyNvaeCell(p, state, x);
  EXPECT_EQ(p.size(), 13);
  EXPECT_EQ(p.count(), 198);
  EXPECT_EQ(state.size(), 6);
  EXPECT_EQ(state.count(), 40);
  const auto weights = p.flatten(), before = state.flatten();
  const auto probe = m::HierarchyNvaeCell(p, state, x, true, false);
  EXPECT_EQ(state.flatten(), before);
  const auto update = m::HierarchyNvaeCell(p, state, x, true, true);
  EXPECT_EQ(p.flatten(), weights);
  EXPECT_NE(state.flatten(), before);
  EXPECT_EQ(probe.to_vector<float>(), update.to_vector<float>());
  n::ConvLayerOptions no_bias;
  no_bias.bias        = false;
  const auto expanded = n::conv2d(p, "cell/expand", x, 8, 1, no_bias);
  const auto moment   = n::batch_norm_train(
      expanded, p.at("cell/expand_bn/gamma"), p.at("cell/expand_bn/beta"), .001);
  EXPECT_EQ(
      state.at("cell/expand_bn/moving_mean").to_vector<float>(),
      (moment.mean * .1).to_vector<float>());
  EXPECT_EQ(
      state.at("cell/expand_bn/moving_var").to_vector<float>(),
      (n::ones({8}) * .9 + moment.variance * .1).to_vector<float>());
  const auto         state_before = state.flatten();
  const auto         out          = m::HierarchyNvaeCell(p, state, x, true, false);
  n::OptimizerConfig c;
  c.kind = n::OptimizerKind::Adam;
  n::Optimizer optimizer(p, .003, c);
  optimizer.step(n::grad(n::mean(n::square(out)), p.parameters()));
  EXPECT_EQ(state.flatten(), state_before);
  EXPECT_EQ(optimizer.iterations(), 1);
  EXPECT_NE(p.flatten(), weights);
}
TEST(NnHierarchy, CellInputFirstAndSecondDerivativesWithTrainingMomentsAndFrozenState) {
  const auto          t = n::DType::Float64;
  n::ParameterStore   p(2609, t), state(0, t);
  std::vector<double> v(16);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = .1 + .03 * i;
  auto x = n::from_vector(v, {2, 2, 2, 2}).requires_grad_();
  m::HierarchyNvaeCell(p, state, x, true, false);
  const auto        before = state.flatten();
  n::TensorFunction f      = [&](const auto& a) {
    return n::mean(n::square(m::HierarchyNvaeCell(p, state, a[0], true, false)));
  };
  EXPECT_TRUE(n::gradcheck(f, {x}, {.atol = 1e-7, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x}, {.atol = 1e-7, .rtol = 1e-5}).ok);
  EXPECT_EQ(state.flatten(), before);
}
TEST(NnHierarchy, EverySeventyOrNinetySixTinyModelLeafDerivativeAndLiveImageNoisePaths) {
  const auto          t = n::DType::Float64;
  std::vector<double> pixels(32);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = .2 + .01 * i;
  const auto x = n::from_vector(pixels, {2, 16}), e = n::tensor({.3, -.6, -.2, .8}, {2, 2}, t);
  for (bool nvae : {false, true}) {
    n::ParameterStore p(2609, t), state(0, t);
    auto              value = [&]() {
      return m::HierarchicalVariationalAutoencoder(p, state, x, e, nvae, true, false, 4, 4, 1, 1)
          .train_loss;
    };
    value();
    for (const auto& name : p.names()) {
      std::vector<double> a(p.at(name).numel());
      for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = name.ends_with("/b") || name.ends_with("/beta") ? .05 + .003 * i
                                                               : .025 + .002 * (i % 11);
      p.set(name, n::from_vector(a, p.at(name).shape()));
    }
    const auto          gs = n::grad(value(), p.parameters());
    std::vector<double> analytic;
    for (const auto& g : gs) {
      const auto a = g.to_vector<double>();
      analytic.insert(analytic.end(), a.begin(), a.end());
    }
    EXPECT_EQ(p.count(), nvae ? 96 : 70);
    EXPECT_EQ(p.size(), nvae ? 37 : 26);
    EXPECT_EQ(analytic.size(), p.count());
    const auto original = p.flatten(), state_before = state.flatten();
    for (std::size_t i = 0; i < original.size(); ++i) {
      auto a = original;
      a[i] += 1e-6;
      p.unflatten(a);
      const double plus = value().item();
      a[i] -= 2e-6;
      p.unflatten(a);
      const double minus = value().item();
      EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-7);
    }
    p.unflatten(original);
    auto              image = x.clone().requires_grad_(), epsilon = e.clone().requires_grad_();
    n::TensorFunction f = [&](const auto& a) {
      return m::HierarchicalVariationalAutoencoder(
                 p, state, a[0], a[1], nvae, true, false, 4, 4, 1, 1)
          .train_loss;
    };
    EXPECT_TRUE(n::gradcheck(f, {image, epsilon}, {.atol = 1e-7, .rtol = 1e-5}).ok);
    EXPECT_TRUE(n::gradgradcheck(f, {image, epsilon}, {.atol = 1e-7, .rtol = 1e-5}).ok);
    EXPECT_EQ(state.flatten(), state_before);
  }
}
TEST(NnHierarchy, DefaultCensusBothLatentDependenciesAndOwningAncestralEndpoints) {
  for (bool nvae : {false, true}) {
    n::ParameterStore p(2609), state(0);
    const auto        x = n::full({2, 784}, .3), e = n::zeros({2, 51});
    const auto        base = m::HierarchicalVariationalAutoencoder(p, state, x, e, nvae);
    EXPECT_EQ(p.size(), nvae ? 37 : 26);
    EXPECT_EQ(p.count(), nvae ? 2027 : 1977);
    EXPECT_EQ(state.size(), nvae ? 6 : 0);
    EXPECT_EQ(base.lower_z.to_vector<float>(), base.lower_mu.to_vector<float>());
    EXPECT_EQ(base.top_z.to_vector<float>(), base.top_mu.to_vector<float>());
    auto lower_noise = n::concat({n::full({2, 49}, .5), n::zeros({2, 2})}, 1),
         top_noise   = n::concat({n::zeros({2, 49}), n::full({2, 2}, .5)}, 1);
    const auto low   = m::HierarchicalVariationalAutoencoder(p, state, x, lower_noise, nvae),
               top   = m::HierarchicalVariationalAutoencoder(p, state, x, top_noise, nvae);
    EXPECT_EQ(low.lower_mu.to_vector<float>(), base.lower_mu.to_vector<float>());
    EXPECT_EQ(low.prior_mu.to_vector<float>(), base.prior_mu.to_vector<float>());
    EXPECT_NE(low.lower_z.to_vector<float>(), base.lower_z.to_vector<float>());
    EXPECT_GT(n::max(n::abs(top.prior_mu - base.prior_mu)).item(), 1e-6);
    EXPECT_GT(n::max(n::abs(top.lower_mu - base.lower_mu)).item(), 1e-6);
    const auto weights = p.flatten(), stats = state.flatten();
    EXPECT_EQ(
        m::HierarchyDecode(p, state, base.lower_z, base.top_z, nvae).to_vector<float>(),
        base.probability.to_vector<float>());
    n::Tensor z, probability;
    {
      const auto prior = m::HierarchyPrior(p, state, n::ones({2, 2}), n::zeros({2, 49}), nvae);
      z                = prior.lower_z;
      probability      = prior.probability;
      EXPECT_EQ(z.to_vector<float>(), prior.prior_mu.to_vector<float>());
    }
    EXPECT_EQ(
        probability.to_vector<float>(),
        m::HierarchyDecode(p, state, z, n::ones({2, 2}), nvae).to_vector<float>());
    EXPECT_EQ(p.flatten(), weights);
    EXPECT_EQ(state.flatten(), stats);
  }
}
TEST(NnHierarchy, TwoSmoothedRankOnePriorPenaltiesAndIsolatedGradients) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2609, t), state(0, t);
  const auto        x = n::full({2, 16}, .3, t), e = n::ones({2, 2}, t);
  auto              value = [&]() {
    return m::HierarchicalVariationalAutoencoder(p, state, x, e, true, true, false, 4, 4, 1, 1);
  };
  value();
  p.set("prior_mu/W", n::full({1, 1, 1, 1}, .3, t));
  p.set("prior_lv/W", n::full({1, 1, 1, 1}, .4, t));
  const auto o = value();
  EXPECT_NEAR(
      o.regularizer.item(), 1e-4 * (std::sqrt(.09 + 1e-12) + std::sqrt(.16 + 1e-12)), 1e-18);
  EXPECT_NEAR(o.train_loss.item(), o.loss.item() + o.regularizer.item(), 1e-15);
  const auto heads       = std::vector<n::Tensor>{p.at("prior_mu/W"), p.at("prior_lv/W")};
  const auto regularized = n::grad(o.train_loss, heads), r = n::grad(value().regularizer, heads),
             plain = n::grad(value().loss, heads);
  for (int j = 0; j < 2; ++j) {
    const double w = j == 0 ? .3 : .4;
    EXPECT_NEAR(r[j].item(), 1e-4 * w / std::sqrt(w * w + 1e-12), 1e-18);
    EXPECT_NEAR(regularized[j].item(), plain[j].item() + r[j].item(), 1e-12);
  }
}
TEST(NnHierarchy, RestoringBothWeightsAndMovingStatisticsReproducesTrainedInference) {
  n::ParameterStore p(2609), state(0);
  const auto        x = n::full({2, 784}, .3), e = n::ones({2, 51});
  for (int k = 0; k < 3; ++k)
    m::HierarchicalVariationalAutoencoder(p, state, x, e, true, true, true);
  const auto expected =
      m::HierarchicalVariationalAutoencoder(p, state, x, e, true).probability.to_vector<float>();
  const auto path = std::filesystem::temp_directory_path() /
                    ("datamunge-hierarchy-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  p.save(path.string() + "-weights.npz");
  state.save(path.string() + "-state.npz");
  n::ParameterStore restored(7), stats(9);
  m::HierarchicalVariationalAutoencoder(restored, stats, x, e, true);
  restored.load(path.string() + "-weights.npz");
  EXPECT_NE(
      m::HierarchicalVariationalAutoencoder(restored, stats, x, e, true)
          .probability.to_vector<float>(),
      expected);
  stats.load(path.string() + "-state.npz");
  EXPECT_EQ(
      m::HierarchicalVariationalAutoencoder(restored, stats, x, e, true)
          .probability.to_vector<float>(),
      expected);
  std::filesystem::remove(path.string() + "-weights.npz");
  std::filesystem::remove(path.string() + "-state.npz");
}
TEST(NnHierarchy, InvalidGeometryNoiseModesAndAliasedStoresBeforeLeafCreation) {
  n::ParameterStore p(1), state(0), double_state(0, n::DType::Float64);
  const auto        x = n::zeros({2, 16}), e = n::zeros({2, 2});
  EXPECT_THROW(
      m::HierarchicalVariationalAutoencoder(p, p, x, e, true, false, false, 4, 4, 1, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::HierarchicalVariationalAutoencoder(p, double_state, x, e, true, false, false, 4, 4, 1, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::HierarchicalVariationalAutoencoder(p, state, x, e, true, false, true, 4, 4, 1, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::HierarchicalVariationalAutoencoder(p, state, x, e, false, false, false, 5, 4, 1, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::HierarchicalVariationalAutoencoder(
          p, state, x, n::zeros({1, 2}), false, false, false, 4, 4, 1, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::HierarchicalVariationalAutoencoder(
          p, state, n::full({2, 16}, 1.1), e, false, false, false, 4, 4, 1, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::HierarchyDecode(p, state, n::zeros({2, 2}), n::zeros({2, 1}), false, 4, 4, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::HierarchyPrior(p, state, n::zeros({2, 1}), n::zeros({1, 1}), false, 4, 4, 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::HierarchyNvaeCell(p, state, n::zeros({2, 2, 2, 1}), false, true), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  EXPECT_EQ(state.size(), 0);
  const auto strided = n::transpose(n::transpose(n::full({2, 16}, .3)).contiguous());
  EXPECT_FALSE(strided.is_contiguous());
  const auto a = m::HierarchicalVariationalAutoencoder(
                 p, state, strided, e, false, false, false, 4, 4, 1, 1),
             b = m::HierarchicalVariationalAutoencoder(
                 p, state, strided.contiguous(), e, false, false, false, 4, 4, 1, 1);
  EXPECT_EQ(a.probability.to_vector<float>(), b.probability.to_vector<float>());
}
