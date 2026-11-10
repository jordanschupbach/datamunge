#include <gtest/gtest.h>

#include <datamunge/nn/big_gan.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Snapshot(const n::ParameterStore& p) {
  std::vector<n::Tensor> a;
  for (const auto& v : p.parameters())
    a.push_back(v.detach().clone());
  return a;
}
void Same(const n::ParameterStore& p, const std::vector<n::Tensor>& a) {
  ASSERT_EQ(p.size(), a.size());
  auto values = p.parameters();
  for (std::size_t i = 0; i < a.size(); ++i)
    EXPECT_TRUE(n::equal_bits(values[i], a[i]));
}
void Initialize(
    n::ParameterStore& g,
    n::ParameterStore& d,
    n::ParameterStore& gs,
    n::ParameterStore& ds,
    n::ParameterStore& moving) {
  n::NoGradGuard guard;
  m::BigGan(
      g,
      d,
      gs,
      ds,
      moving,
      n::zeros({2, 8, 8, 1}, g.dtype()),
      n::zeros({2, 12}, g.dtype()),
      n::zeros({2}, n::DType::Int32));
  n::advance_cached_spectral_state(g, gs, 16);
  n::advance_cached_spectral_state(d, ds, 16);
}
} // namespace
TEST(NnBigGan, CapturedBatchNormMatchesHandMomentsWithoutWritingMovingState) {
  for (auto t : {n::DType::Float32, n::DType::Float64}) {
    n::ParameterStore stats(0, t);
    auto              x   = n::tensor({1., 3.}, {2, 1}, t).requires_grad_();
    auto              out = n::capture_batch_norm(stats, "bn", x, true);
    EXPECT_EQ(out.mean.at({0}), 2);
    EXPECT_EQ(out.unbiased_variance.at({0}), 2);
    EXPECT_EQ(stats.at("bn/moving_mean").at({0}), 0);
    EXPECT_EQ(stats.at("bn/moving_var").at({0}), 1);
    EXPECT_NEAR(
        out.value.at({0, 0}),
        -1 / std::sqrt(1 + double(1e-5f)),
        t == n::DType::Float32 ? 1e-7 : 1e-14);
    EXPECT_TRUE(out.mean.requires_grad());
    EXPECT_TRUE(out.unbiased_variance.requires_grad());
    auto before = Snapshot(stats);
    auto saved  = n::capture_batch_norm(stats, "bn", n::ones({1, 1}, t), false);
    Same(stats, before);
    EXPECT_FALSE(saved.mean.defined());
    EXPECT_FALSE(saved.unbiased_variance.defined());
  }
}
TEST(NnBigGan, CapturedBatchNormHasFirstHigherAndCapturePathsWhileSavedMomentsStop) {
  auto              t = n::DType::Float64;
  n::ParameterStore stats(0, t);
  auto              x  = n::tensor({1., 2., 4., 8.}, {2, 2}, t).requires_grad_();
  n::TensorFunction fn = [&](const auto& a) {
    auto out = n::capture_batch_norm(stats, "bn", a[0], true);
    return n::sum(n::square(out.value)) + n::sum(n::square(out.mean)) +
           .3 * n::sum(out.unbiased_variance);
  };
  EXPECT_TRUE(n::gradcheck(fn, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  auto inference = n::capture_batch_norm(stats, "bn", x, false);
  auto grad      = n::grad(n::sum(inference.value), stats.parameters(), {.allow_unused = true});
  for (const auto& v : grad)
    EXPECT_FALSE(v.defined());
  auto strided = n::transpose(x);
  auto a       = n::capture_batch_norm(stats, "bn", strided, true),
       b       = n::capture_batch_norm(stats, "bn", strided.contiguous(), true);
  EXPECT_TRUE(n::equal_bits(a.value, b.value));
}
TEST(NnBigGan, CapturedBatchNormRejectsBadInputsGuardsAndSingleTrainingObservationEarly) {
  n::ParameterStore stats(0);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 2}),
        n::ones({2}),
        n::ones({2, 2}, n::DType::Float64),
        n::full({2, 2}, NAN)})
    EXPECT_THROW(n::capture_batch_norm(stats, "bn", bad, true), std::invalid_argument);
  EXPECT_THROW(n::capture_batch_norm(stats, "bn", n::ones({1, 2}), true), std::invalid_argument);
  EXPECT_THROW(n::capture_batch_norm(stats, "", n::ones({2, 2}), true), std::invalid_argument);
  for (double eps : {0., -1., double(NAN), double(INFINITY), 1e-300, 1e300})
    EXPECT_THROW(
        n::capture_batch_norm(stats, "bn", n::ones({2, 2}), true, eps), std::invalid_argument);
  EXPECT_EQ(stats.size(), 0);
  n::capture_batch_norm(stats, "bn", n::ones({2, 2}), false);
  stats.set("bn/moving_var", n::full({2}, -1));
  EXPECT_THROW(n::capture_batch_norm(stats, "bn", n::ones({2, 2}), false), std::invalid_argument);
}
TEST(NnBigGan, MovingCandidatesUseCapturedUnbiasedVarianceAndStopEveryPath) {
  auto t        = n::DType::Float64;
  auto old      = n::tensor({0., 1.}, {2}, t).requires_grad_(),
       oldvar   = n::tensor({1., 2.}, {2}, t).requires_grad_();
  auto mean     = n::tensor({4., 5.}, {2}, t).requires_grad_(),
       variance = n::tensor({3., 4.}, {2}, t).requires_grad_();
  auto next     = n::batch_norm_moving_values(old, oldvar, mean, variance, .5);
  EXPECT_EQ(next.mean.to_vector<double>(), (std::vector<double>{2., 3.}));
  EXPECT_EQ(next.variance.to_vector<double>(), (std::vector<double>{2., 3.}));
  EXPECT_FALSE(next.mean.requires_grad());
  EXPECT_FALSE(next.variance.requires_grad());
  EXPECT_TRUE(
      n::equal_bits(n::batch_norm_moving_values(old, oldvar, mean, variance, 1.).mean, old));
  EXPECT_TRUE(
      n::equal_bits(n::batch_norm_moving_values(old, oldvar, mean, variance, 0.).mean, mean));
  for (double momentum : {-.1, 1.1, double(NAN), double(INFINITY)})
    EXPECT_THROW(
        n::batch_norm_moving_values(old, oldvar, mean, variance, momentum), std::invalid_argument);
  EXPECT_THROW(
      n::batch_norm_moving_values(old, oldvar, mean, n::full({2}, -1, t)), std::invalid_argument);
  EXPECT_THROW(
      n::batch_norm_moving_values(old, oldvar, n::ones({3}, t), variance), std::invalid_argument);
}
TEST(NnBigGan, RejectionPriorGoldenStreamFinalCastingOddCountsAndOwnedFields) {
  auto draw = m::GaussianRejectionPrior(1, 3, 6907, .5);
  EXPECT_EQ(draw.z.shape(), n::Shape({1, 3}));
  EXPECT_EQ(draw.z.dtype(), n::DType::Float32);
  EXPECT_EQ(draw.proposals.dtype(), n::DType::Float64);
  EXPECT_EQ(
      draw.z.to_vector<double>(),
      (std::vector<double>{-0.4454278349876404, -0.08479340374469757, 0.38326919078826904}));
  EXPECT_EQ(
      draw.proposals.to_vector<double>(),
      (std::vector<double>{
          -0.9388149930815035,
          -0.44542782166946016,
          -1.2078747239081584,
          -0.08479340404496108,
          0.3832691843572766}));
  auto wide = m::GaussianRejectionPrior(1, 3, 6907, .5, n::DType::Float64);
  EXPECT_TRUE(n::equal_bits(draw.z, n::cast(wide.z, n::DType::Float32)));
  EXPECT_TRUE(n::equal_bits(draw.proposals, wide.proposals));
  EXPECT_FALSE(draw.z.requires_grad());
  EXPECT_FALSE(draw.proposals.requires_grad());
  auto full = m::GaussianRejectionPrior(1, 3, 6907);
  EXPECT_EQ(full.proposals.size(0), 3);
  EXPECT_TRUE(
      n::equal_bits(
          n::cast(full.z, n::DType::Float64),
          n::cast(n::cast(full.proposals.reshape({1, 3}), n::DType::Float32), n::DType::Float64)));
  auto repeat = m::GaussianRejectionPrior(1, 3, 6907, .5);
  EXPECT_TRUE(n::equal_bits(draw.z, repeat.z));
  EXPECT_FALSE(n::equal_bits(draw.z, m::GaussianRejectionPrior(1, 3, 6908, .5).z));
}
TEST(NnBigGan, RejectionPriorRejectsBadGeometryLimitDtypeAndExhaustedBudget) {
  for (double limit : {0., -1., double(NAN), -double(INFINITY)})
    EXPECT_THROW(m::GaussianRejectionPrior(1, 3, 1, limit), std::invalid_argument);
  EXPECT_THROW(m::GaussianRejectionPrior(0, 3, 1), std::invalid_argument);
  EXPECT_THROW(m::GaussianRejectionPrior(1, 0, 1), std::invalid_argument);
  EXPECT_THROW(
      m::GaussianRejectionPrior(std::numeric_limits<std::int64_t>::max(), 12, 1),
      std::invalid_argument);
  EXPECT_THROW(m::GaussianRejectionPrior(1, 3, 1, 1., n::DType::Int32), std::invalid_argument);
  EXPECT_THROW(m::GaussianRejectionPrior(1, 1, 1, 1e-300), std::runtime_error);
}
TEST(NnBigGan, ResizeUsesHalfPixelDoubleInterpolationWithoutClippingAndRetainsInputGradient) {
  auto x   = n::arange(0, 784, 1, n::DType::Float64).reshape({1, 28, 28, 1}).requires_grad_();
  auto out = m::BigGanResize8(x);
  EXPECT_EQ(out.shape(), n::Shape({1, 8, 8, 1}));
  EXPECT_EQ(out.at({0, 0, 0, 0}), 36.25);
  EXPECT_EQ(out.at({0, 0, 1, 0}), 39.75);
  EXPECT_EQ(n::sum(n::grad(n::sum(out), {x})[0]).item(), 64);
  auto transposed = n::transpose(x, {0, 2, 1, 3});
  EXPECT_TRUE(
      n::equal_bits(m::BigGanResize8(transposed), m::BigGanResize8(transposed.contiguous())));
  EXPECT_THROW(m::BigGanResize8(n::zeros({1, 8, 8, 1})), std::invalid_argument);
  EXPECT_THROW(m::BigGanResize8(n::full({1, 28, 28, 1}, NAN)), std::invalid_argument);
}
TEST(NnBigGan, DefaultCensusCreationOrderAndInferenceCapturesMatchTheSource) {
  n::ParameterStore g(6803), d(6823), gs(0), ds(0), moving(0);
  auto x = n::zeros({2, 8, 8, 1}), z = n::zeros({2, 12}), c = n::zeros({2}, n::DType::Int32);
  auto out = m::BigGan(g, d, gs, ds, moving, x, z, c);
  EXPECT_EQ(g.count(), 789);
  EXPECT_EQ(d.count(), 241);
  EXPECT_EQ(gs.count(), 241);
  EXPECT_EQ(ds.count(), 77);
  EXPECT_EQ(moving.count(), 24);
  EXPECT_EQ(g.size(), 17);
  EXPECT_EQ(d.size(), 9);
  EXPECT_EQ(gs.size(), 18);
  EXPECT_EQ(ds.size(), 10);
  EXPECT_EQ(moving.size(), 6);
  EXPECT_EQ(g.names().front(), "class_embedding");
  EXPECT_EQ(g.names().back(), "rgb/b");
  EXPECT_EQ(d.names().back(), "projection/W");
  auto inference = m::BigGanGenerate(g, gs, moving, z, c, false);
  EXPECT_FALSE(inference.moments.bn1_mean.defined());
  EXPECT_TRUE(out.moments.bn1_mean.defined());
  EXPECT_TRUE(n::equal_bits(out.fake, out.fake_inference));
  EXPECT_TRUE(n::equal_bits(out.fake_score, out.fake_score_inference));
}
TEST(NnBigGan, DisplaySelectionNeverChangesTheBatchStatisticsLossesOrMovingState) {
  n::ParameterStore g(6803), d(6823), gs(0), ds(0), moving(0);
  Initialize(g, d, gs, ds, moving);
  auto z = m::GaussianRejectionPrior(2, 12, 6872).z, c = n::tensor({0, 1}, {2}, n::DType::Int32),
       x      = n::full({2, 8, 8, 1}, .3);
  auto before = Snapshot(moving);
  auto frozen = m::BigGan(g, d, gs, ds, moving, x, z, c, false),
       batch  = m::BigGan(g, d, gs, ds, moving, x, z, c, true);
  Same(moving, before);
  EXPECT_TRUE(n::equal_bits(frozen.d_loss, batch.d_loss));
  EXPECT_TRUE(n::equal_bits(frozen.g_loss, batch.g_loss));
  EXPECT_FALSE(n::equal_bits(frozen.fake, batch.fake));
  EXPECT_TRUE(n::equal_bits(batch.fake, batch.fake_training));
  EXPECT_TRUE(n::equal_bits(frozen.fake, frozen.fake_inference));
  auto loss = m::HingeGanObjectives(frozen.real_score, frozen.fake_score_training);
  EXPECT_TRUE(n::equal_bits(loss.discriminator, frozen.d_loss));
  EXPECT_TRUE(n::equal_bits(loss.generator, frozen.g_source));
}
TEST(NnBigGan, ModifiedOrthogonalExcludesDiagonalRawEmbeddingAndBiasWithLiveKernelHessian) {
  auto              t = n::DType::Float64;
  n::ParameterStore p(1, t);
  auto              w = p.get("layer/W", {2, 2}, n::init::ones());
  p.get("bias", {2}, n::init::ones());
  p.get("class_embedding", {2, 2}, n::init::ones());
  auto penalty = m::ModifiedOrthogonalPenalty(p, .5);
  EXPECT_EQ(penalty.item(), 4);
  auto gradients = n::grad(penalty, p.parameters(), {.create_graph = true});
  EXPECT_EQ(gradients[0].to_vector<double>(), (std::vector<double>{4, 4, 4, 4}));
  EXPECT_EQ(n::sum(n::abs(gradients[1])).item(), 0);
  EXPECT_EQ(n::sum(n::abs(gradients[2])).item(), 0);
  auto h = n::grad(n::sum(gradients[0]), {w});
  EXPECT_EQ(h[0].to_vector<double>(), (std::vector<double>{12, 12, 12, 12}));
  p.set("layer/W", n::eye(2, t));
  EXPECT_EQ(m::ModifiedOrthogonalPenalty(p, .5).item(), 0);
  EXPECT_THROW(m::ModifiedOrthogonalPenalty(p, -1), std::invalid_argument);
  n::ParameterStore empty(0);
  EXPECT_THROW(m::ModifiedOrthogonalPenalty(empty), std::invalid_argument);
}
TEST(NnBigGan, CapturedMomentCommitValidatesEveryLayerBeforeAnyWrite) {
  n::ParameterStore g(6803), d(6823), gs(0), ds(0), moving(0);
  Initialize(g, d, gs, ds, moving);
  auto z = m::GaussianRejectionPrior(2, 12, 6872).z, c = n::tensor({0, 1}, {2}, n::DType::Int32);
  auto generated               = m::BigGanGenerate(g, gs, moving, z, c, true);
  auto original                = Snapshot(moving);
  auto bad                     = generated.moments;
  bad.output_unbiased_variance = n::full({4}, NAN);
  EXPECT_THROW(m::BigGanCommitMoments(moving, bad), std::invalid_argument);
  Same(moving, original);
  n::ParameterStore partial(0);
  for (const std::string name : {"bn1", "bn2"}) {
    partial.get(name + "/moving_mean", {4}, n::init::zeros());
    partial.get(name + "/moving_var", {4}, n::init::ones());
  }
  auto before = Snapshot(partial);
  EXPECT_THROW(m::BigGanCommitMoments(partial, generated.moments), std::out_of_range);
  Same(partial, before);
  m::BigGanCommitMoments(moving, generated.moments);
  EXPECT_FALSE(n::equal_bits(original[1], moving.at("bn1/moving_var")));
  EXPECT_THROW(
      m::BigGanCommitMoments(moving, m::BigGanGenerate(g, gs, moving, z, c, false).moments),
      std::invalid_argument);
}
TEST(NnBigGan, ExplicitDSpectralDGSpectralGAndCapturedBNPhasesOwnTheirStores) {
  n::ParameterStore g(6803), d(6823), gs(0), ds(0), moving(0);
  Initialize(g, d, gs, ds, moving);
  auto z = m::GaussianRejectionPrior(2, 12, 6872).z, c = n::tensor({0, 1}, {2}, n::DType::Int32),
       x = n::full({2, 8, 8, 1}, .3);
  n::OptimizerConfig config;
  config.kind  = n::OptimizerKind::Adam;
  config.beta1 = 0;
  config.beta2 = .999;
  n::Optimizer go(g, .0002, config), disc(d, .0002, config);
  auto         gb = Snapshot(g), db = Snapshot(d), gsb = Snapshot(gs), dsb = Snapshot(ds),
       mb = Snapshot(moving);
  n::advance_cached_spectral_state(d, ds);
  Same(g, gb);
  Same(d, db);
  Same(gs, gsb);
  Same(moving, mb);
  dsb     = Snapshot(ds);
  auto df = m::BigGan(g, d, gs, ds, moving, x, z, c, true);
  disc.step(n::grad(df.d_loss, d.parameters()));
  Same(g, gb);
  Same(gs, gsb);
  Same(ds, dsb);
  Same(moving, mb);
  db = Snapshot(d);
  n::advance_cached_spectral_state(g, gs);
  Same(g, gb);
  Same(d, db);
  Same(ds, dsb);
  Same(moving, mb);
  gsb          = Snapshot(gs);
  auto gf      = m::BigGan(g, d, gs, ds, moving, x, z, c, true);
  auto capture = gf.moments;
  go.step(n::grad(gf.g_loss, g.parameters()));
  Same(d, db);
  Same(gs, gsb);
  Same(ds, dsb);
  Same(moving, mb);
  auto recomputed = m::BigGanGenerate(g, gs, moving, z, c, true).moments;
  EXPECT_FALSE(n::equal_bits(capture.output_mean, recomputed.output_mean));
  auto expected = n::batch_norm_moving_values(
      moving.at("output_bn/moving_mean"),
      moving.at("output_bn/moving_var"),
      capture.output_mean,
      capture.output_unbiased_variance);
  m::BigGanCommitMoments(moving, capture);
  Same(d, db);
  Same(gs, gsb);
  Same(ds, dsb);
  EXPECT_TRUE(n::equal_bits(moving.at("output_bn/moving_mean"), expected.mean));
  EXPECT_EQ(go.iterations(), 1);
  EXPECT_EQ(disc.iterations(), 1);
}
TEST(NnBigGan, ProjectionClassAndAllHierarchicalChunkPathsAreActiveWithFrozenStates) {
  n::ParameterStore g(6803), d(6823), gs(0), ds(0), moving(0);
  Initialize(g, d, gs, ds, moving);
  auto z = m::GaussianRejectionPrior(2, 12, 6872).z, c = n::tensor({0, 1}, {2}, n::DType::Int32),
       other     = n::tensor({1, 2}, {2}, n::DType::Int64);
  auto generated = m::BigGanGenerate(g, gs, moving, z, c, false).image;
  EXPECT_GT(
      n::max(n::abs(generated - m::BigGanGenerate(g, gs, moving, z, other, false).image)).item(),
      1e-5);
  auto image = n::full({2, 8, 8, 1}, .3);
  auto a = m::BigGanDiscriminate(d, ds, image, c), b = m::BigGanDiscriminate(d, ds, image, other);
  EXPECT_TRUE(n::equal_bits(a.features, b.features));
  EXPECT_TRUE(n::equal_bits(a.source, b.source));
  EXPECT_GT(n::max(n::abs(a.projection - b.projection)).item(), 1e-5);
  EXPECT_TRUE(n::equal_bits(a.score, a.source + a.projection));
  for (int chunk = 0; chunk < 3; ++chunk) {
    auto values = z.to_vector<float>();
    for (int row = 0; row < 2; ++row)
      for (int j = 0; j < 4; ++j)
        values[row * 12 + chunk * 4 + j] = 0;
    auto changed =
        m::BigGanGenerate(g, gs, moving, n::from_vector(values, {2, 12}), c, false).image;
    EXPECT_GT(n::max(n::abs(changed - generated)).item(), 1e-5);
  }
}
TEST(NnBigGan, GeneratorNoiseAndCapturedNormalizationSupportFirstAndHigherPaths) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(6803, t), d(6823, t), gs(0, t), ds(0, t), moving(0, t);
  Initialize(g, d, gs, ds, moving);
  auto z               = m::GaussianRejectionPrior(2, 12, 6872, 2., t).z.requires_grad_(),
       c               = n::tensor({0, 1}, {2}, n::DType::Int64);
  n::TensorFunction fn = [&](const auto& a) {
    auto result = m::BigGanGenerate(g, gs, moving, a[0], c, true);
    return n::sum(n::square(result.image));
  };
  EXPECT_TRUE(n::gradcheck(fn, {z}, {.atol = 2e-7, .rtol = 2e-4}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {z}, {.atol = 2e-7, .rtol = 2e-4}).ok);
  auto image = n::full({2, 8, 8, 1}, .3, t).requires_grad_();
  auto score = m::BigGanDiscriminate(d, ds, image, c).score;
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(score), {image})[0])).item(), 1e-6);
  auto output   = m::BigGan(g, d, gs, ds, moving, image, z, c, true);
  auto detached = n::grad(output.g_loss, gs.parameters(), {.allow_unused = true});
  for (const auto& partial : detached)
    EXPECT_FALSE(partial.defined());
}
TEST(NnBigGan, InvalidInputsLabelsStoreAliasesAndMismatchedRowsPrecedeLeaves) {
  n::ParameterStore g(1), d(2), gs(0), ds(0), moving(0), wide(0, n::DType::Float64);
  auto x = n::zeros({2, 8, 8, 1}), z = n::zeros({2, 12}), c = n::zeros({2}, n::DType::Int32);
  EXPECT_THROW(m::BigGan(g, g, gs, ds, moving, x, z, c), std::invalid_argument);
  EXPECT_THROW(m::BigGan(g, d, gs, gs, moving, x, z, c), std::invalid_argument);
  EXPECT_THROW(m::BigGan(g, d, gs, ds, g, x, z, c), std::invalid_argument);
  EXPECT_THROW(m::BigGan(g, d, gs, ds, wide, x, z, c), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 8, 8, 1}),
        n::zeros({2, 8, 8}),
        n::full({2, 8, 8, 1}, NAN),
        n::zeros({2, 8, 8, 1}, n::DType::Float64)})
    EXPECT_THROW(m::BigGan(g, d, gs, ds, moving, bad, z, c), std::invalid_argument);
  EXPECT_THROW(m::BigGan(g, d, gs, ds, moving, x, n::zeros({3, 12}), c), std::invalid_argument);
  EXPECT_THROW(m::BigGan(g, d, gs, ds, moving, x, n::zeros({2, 11}), c), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::zeros({1}, n::DType::Int32),
        n::tensor({0, 10}, {2}, n::DType::Int32),
        n::tensor({-1, 1}, {2}, n::DType::Int64),
        n::zeros({2})})
    EXPECT_THROW(m::BigGan(g, d, gs, ds, moving, x, z, bad), std::invalid_argument);
  EXPECT_EQ(g.size() + d.size() + gs.size() + ds.size() + moving.size(), 0);
}
