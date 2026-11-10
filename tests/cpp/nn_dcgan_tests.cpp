#include <gtest/gtest.h>

#include <datamunge/nn/dcgan.hpp>
#include <datamunge/nn/gan_priors.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Snapshot(const n::ParameterStore& p) {
  std::vector<n::Tensor> a;
  for (auto& v : p.parameters())
    a.push_back(v.detach().clone());
  return a;
}
void Same(const n::ParameterStore& p, const std::vector<n::Tensor>& a) {
  auto values = p.parameters();
  ASSERT_EQ(values.size(), a.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    EXPECT_TRUE(n::equal_bits(a[i], values[i]));
}
void Commit(
    n::ParameterStore& s,
    const std::string& name,
    const n::Tensor&   mean,
    const n::Tensor&   variance) {
  auto next = n::batch_norm_moving_values(
      s.at(name + "/moving_mean"), s.at(name + "/moving_var"), mean, variance, .9);
  s.set(name + "/moving_mean", next.mean);
  s.set(name + "/moving_var", next.variance);
}
} // namespace
TEST(NnDcgan, FusedAffineCapturePreservesExactFusedArithmeticAndDetachedSampleMoments) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    n::ParameterStore p(0, type), s(0, type);
    auto              x     = n::tensor({1., 3.}, {2, 1}, type).requires_grad_();
    auto              out   = n::capture_affine_batch_norm(p, s, "bn", x, true);
    auto              fused = n::batch_norm_train(x, p.at("bn/gamma"), p.at("bn/beta"), .001);
    EXPECT_TRUE(n::equal_bits(out.value, fused.y));
    EXPECT_EQ(out.mean.item(), 2);
    EXPECT_EQ(out.unbiased_variance.item(), 2);
    EXPECT_FALSE(out.mean.requires_grad());
    EXPECT_FALSE(out.unbiased_variance.requires_grad());
    EXPECT_TRUE(out.value.requires_grad());
    EXPECT_EQ(s.at("bn/moving_var").item(), 1);
    auto next = n::batch_norm_moving_values(
        s.at("bn/moving_mean"), s.at("bn/moving_var"), out.mean, out.unbiased_variance, .9);
    EXPECT_NEAR(next.mean.item(), .2, 1e-8);
    EXPECT_NEAR(next.variance.item(), 1.1, 1e-7);
    auto saved = n::capture_affine_batch_norm(p, s, "bn", x);
    EXPECT_FALSE(saved.mean.defined());
    EXPECT_FALSE(saved.unbiased_variance.defined());
    auto one = n::capture_affine_batch_norm(p, s, "bn", n::ones({1, 1}, type), true);
    EXPECT_EQ(one.unbiased_variance.item(), 0);
  }
}
TEST(NnDcgan, FusedAffineHasFirstAndHigherInputAffinePathsButSavedMomentsStop) {
  auto              t = n::DType::Float64;
  n::ParameterStore p(0, t), s(0, t);
  auto              x = n::tensor({1., 2., 4., 8.}, {2, 2}, t).requires_grad_();
  n::capture_affine_batch_norm(p, s, "bn", x, true);
  n::TensorFunction fn = [&](const auto& a) {
    return n::sum(n::square(n::capture_affine_batch_norm(p, s, "bn", a[0], true).value));
  };
  EXPECT_TRUE(n::gradcheck(fn, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  auto saved   = n::capture_affine_batch_norm(p, s, "bn", x);
  auto stopped = n::grad(n::sum(saved.value), s.parameters(), {.allow_unused = true});
  for (auto& v : stopped)
    EXPECT_FALSE(v.defined());
}
TEST(NnDcgan, FusedInvalidInputChecksPrecedeLeavesAndStoredFailuresAreVisible) {
  n::ParameterStore p(0), s(0), other(0, n::DType::Float64);
  auto              x = n::ones({2, 1});
  EXPECT_THROW(n::capture_affine_batch_norm(p, p, "bn", x), std::invalid_argument);
  EXPECT_THROW(n::capture_affine_batch_norm(p, other, "bn", x), std::invalid_argument);
  EXPECT_THROW(n::capture_affine_batch_norm(p, s, "", x), std::invalid_argument);
  EXPECT_THROW(n::capture_affine_batch_norm(p, s, "bn", x, false, 0), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  n::capture_affine_batch_norm(p, s, "bn", x);
  s.set("bn/moving_var", n::full({1}, -1));
  EXPECT_THROW(n::capture_affine_batch_norm(p, s, "bn", x), std::invalid_argument);
}
TEST(NnDcgan, InitializationCensusNamesAndSavedInferenceAreImmutable) {
  n::ParameterStore g(5003), d(5011), gs(0), ds(0);
  auto              x = n::full({2, 28, 28, 1}, .3), z = m::GaussianRejectionPrior(2, 16, 4862).z;
  auto              out = m::Dcgan(g, d, gs, ds, x, z);
  EXPECT_EQ(g.count(), 6873);
  EXPECT_EQ(g.size(), 8);
  EXPECT_EQ(d.count(), 989);
  EXPECT_EQ(d.size(), 7);
  EXPECT_EQ(gs.count(), 24);
  EXPECT_EQ(ds.count(), 16);
  EXPECT_EQ(gs.size() + ds.size(), 6);
  EXPECT_EQ(g.names().front(), "projection/W");
  EXPECT_EQ(g.names().back(), "output/b");
  EXPECT_EQ(d.names().front(), "down14/W");
  EXPECT_EQ(d.names().back(), "output/b");
  auto a = Snapshot(g), b = Snapshot(d), c = Snapshot(gs), e = Snapshot(ds);
  auto repeat = m::Dcgan(g, d, gs, ds, x, z);
  EXPECT_TRUE(n::equal_bits(out.fake, repeat.fake));
  Same(g, a);
  Same(d, b);
  Same(gs, c);
  Same(ds, e);
  EXPECT_EQ(out.fake.shape(), n::Shape({2, 784}));
  auto generated = m::DcganGenerate(g, gs, z);
  EXPECT_FALSE(generated.stem_mean.defined());
  EXPECT_TRUE(n::equal_bits(n::reshape((generated.image + 1.) * .5, {-1, 784}), out.fake));
}
TEST(NnDcgan, DiscriminatorPhaseUsesJointRealFakeMomentsAndStopsGeneratorPaths) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(5003, t), d(5011, t), gs(0, t), ds(0, t);
  auto              x = n::full({2, 28, 28, 1}, .3, t).requires_grad_(),
       z              = m::GaussianRejectionPrior(2, 16, 4862, 2, t).z.requires_grad_();
  auto out            = m::DcganDiscriminatorPhase(g, d, gs, ds, x, z);
  EXPECT_FALSE(out.fake.requires_grad());
  EXPECT_FALSE(out.down7_mean.requires_grad());
  EXPECT_TRUE(out.real_logit.requires_grad());
  auto direct = m::DcganDiscriminate(d, ds, n::concat({x * 2. - 1., out.fake}, 0), true);
  EXPECT_TRUE(n::equal_bits(out.down7_mean, direct.mean));
  EXPECT_TRUE(n::equal_bits(out.down7_variance, direct.variance));
  EXPECT_TRUE(n::equal_bits(out.fake_logit, n::slice_axis(direct.logit, 0, 2, 4)));
  auto stopped = n::grad(out.loss, {z}, {.allow_unused = true});
  EXPECT_FALSE(stopped[0].defined());
  auto live = n::grad(out.loss, {x});
  EXPECT_TRUE(live[0].defined());
}
TEST(NnDcgan, GeneratorPhaseUsesSavedCurrentDAndRetainsOnlySelectedWriteResponsibility) {
  n::ParameterStore g(5003), d(5011), gs(0), ds(0);
  auto              x = n::full({2, 28, 28, 1}, .3), z = m::GaussianRejectionPrior(2, 16, 4862).z;
  auto              out = m::DcganGeneratorPhase(g, d, gs, ds, x, z);
  EXPECT_TRUE(out.fake.requires_grad());
  EXPECT_FALSE(out.real_logit.defined());
  EXPECT_FALSE(out.down7_mean.defined());
  EXPECT_FALSE(out.stem_mean.requires_grad());
  auto direct = m::DcganDiscriminate(d, ds, out.fake);
  EXPECT_TRUE(n::equal_bits(out.fake_logit, direct.logit));
  EXPECT_TRUE(n::equal_bits(out.loss, n::mean(n::softplus(-direct.logit))));
  auto partials = n::grad(out.loss, d.parameters());
  EXPECT_EQ(partials.size(), d.size());
}
TEST(NnDcgan, FourExplicitPhaseOwnersCommitCapturesBeforeTheirSelectedAdamWrites) {
  n::ParameterStore g(5003), d(5011), gs(0), ds(0);
  auto              x = n::full({2, 28, 28, 1}, .3), z = m::GaussianRejectionPrior(2, 16, 4862).z;
  m::Dcgan(g, d, gs, ds, x, z);
  n::OptimizerConfig cfg;
  cfg.kind = n::OptimizerKind::Adam;
  cfg.beta1 = .5;
  n::Optimizer go(g, .0002, cfg), disc(d, .0002, cfg);
  auto         a        = m::DcganDiscriminatorPhase(g, d, gs, ds, x, z);
  auto         before_g = Snapshot(g), before_d = Snapshot(d), before_gs = Snapshot(gs);
  Commit(ds, "down7_bn", a.down7_mean, a.down7_variance);
  Same(g, before_g);
  Same(d, before_d);
  Same(gs, before_gs);
  auto saved_ds = Snapshot(ds);
  disc.step(n::grad(a.loss, d.parameters()));
  Same(g, before_g);
  Same(gs, before_gs);
  Same(ds, saved_ds);
  auto b   = m::DcganGeneratorPhase(g, d, gs, ds, x, z);
  before_d = Snapshot(d);
  Commit(gs, "stem_bn", b.stem_mean, b.stem_variance);
  Commit(gs, "up14_bn", b.up14_mean, b.up14_variance);
  Same(g, before_g);
  Same(d, before_d);
  Same(ds, saved_ds);
  auto saved_gs = Snapshot(gs);
  go.step(n::grad(b.loss, g.parameters()));
  Same(d, before_d);
  Same(gs, saved_gs);
  Same(ds, saved_ds);
  EXPECT_EQ(go.iterations(), 1);
  EXPECT_EQ(disc.iterations(), 1);
}
TEST(NnDcgan, GeneratorNoisePathsHaveFirstAndSecondDerivativesThroughTransposedMaps) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(5003, t), d(5011, t), gs(0, t), ds(0, t);
  auto              x  = n::full({1, 28, 28, 1}, .3, t);
  auto              z  = n::tensor({.37, -.19}, {1, 2}, t).requires_grad_();
  n::TensorFunction fn = [&](const auto& a) {
    auto code = n::concat({a[0], n::full({1, 14}, .2, t)}, 1);
    auto out  = m::DcganGeneratorPhase(g, d, gs, ds, x, code);
    return out.loss + n::mean(n::square(out.fake));
  };
  EXPECT_TRUE(n::gradcheck(fn, {z}, {.atol = 2e-7, .rtol = 3e-4}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {z}, {.atol = 2e-7, .rtol = 3e-4}).ok);
}
TEST(NnDcgan, FixedGeometryAliasDtypeAndNonfiniteInputsFailBeforeInitialization) {
  n::ParameterStore g(5003), d(5011), gs(0), ds(0), other(0, n::DType::Float64);
  auto              x = n::zeros({2, 28, 28, 1}), z = n::ones({2, 16});
  EXPECT_THROW(m::Dcgan(g, g, gs, ds, x, z), std::invalid_argument);
  EXPECT_THROW(m::Dcgan(g, d, gs, other, x, z), std::invalid_argument);
  EXPECT_THROW(m::Dcgan(g, d, gs, ds, x, n::ones({1, 16})), std::invalid_argument);
  EXPECT_THROW(m::DcganGenerate(g, gs, n::ones({2, 12})), std::invalid_argument);
  EXPECT_THROW(m::DcganDiscriminate(d, ds, n::ones({2, 8, 8, 1})), std::invalid_argument);
  EXPECT_THROW(
      m::DcganGenerate(g, gs, n::full({2, 16}, std::numeric_limits<double>::infinity())),
      std::invalid_argument);
  EXPECT_EQ(g.size() + d.size() + gs.size() + ds.size(), 0);
}
TEST(NnDcgan, SavedGeneratorRowsAreIndependentWhileBatchBNUsesOtherRows) {
  n::ParameterStore g(5003), gs(0);
  auto              z    = m::GaussianRejectionPrior(2, 16, 4862).z;
  auto              full = m::DcganGenerate(g, gs, z),
       single            = m::DcganGenerate(g, gs, n::slice_axis(z, 0, 0, 1));
  auto delta             = n::max(n::abs(n::slice_axis(full.image, 0, 0, 1) - single.image));
  EXPECT_LT(delta.item(), 2e-6);
  auto train = m::DcganGenerate(g, gs, z, true);
  EXPECT_GT(n::max(n::abs(train.image - full.image)).item(), 1e-5);
}
