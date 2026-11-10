#include <gtest/gtest.h>

#include <datamunge/nn/gan_priors.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/progressive_gan.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
struct Model {
  std::vector<std::unique_ptr<n::ParameterStore>> p;
  n::Tensor                                       x, z;
  explicit Model(n::DType dtype = n::DType::Float64) {
    for (auto seed : {6401, 6407, 6413, 6421, 6427, 6449, 6451, 6469})
      p.push_back(std::make_unique<n::ParameterStore>(seed, dtype));
    x = n::full({2, 8, 8, 1}, .3, dtype);
    z = m::GaussianRejectionPrior(2, 16, 6504, 2, dtype).z;
  }
  m::ProgressiveGanOutput Forward(const n::Tensor& a) {
    return m::ProgressiveGan(*p[0], *p[1], *p[2], *p[3], *p[4], *p[5], *p[6], *p[7], x, z, a);
  }
  m::ProgressiveGanGenerated Generate(const n::Tensor& a) {
    return m::ProgressiveGanGenerate(*p[0], *p[1], *p[2], *p[3], z, a);
  }
  n::Tensor Alpha(double a) { return n::scalar_like(a, z); }
};
std::vector<n::Tensor> Snapshot(const n::ParameterStore& p) {
  std::vector<n::Tensor> a;
  for (auto& v : p.parameters())
    a.push_back(v.detach().clone());
  return a;
}
void Same(const n::ParameterStore& p, const std::vector<n::Tensor>& a) {
  auto values = p.parameters();
  ASSERT_EQ(values.size(), a.size());
  for (std::size_t k = 0; k < a.size(); ++k)
    EXPECT_TRUE(n::equal_bits(values[k], a[k]));
}
} // namespace
TEST(NnProgressiveGan, PixelRmsIsPerPositionWithoutCenteringAndPreservesSourceGrouping) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    auto x        = n::tensor({3., 4., 0., 0.}, {2, 2}, type);
    auto out      = n::pixel_rms_normalize(x);
    auto expected = x / n::sqrt(n::mean(n::square(x), {1}, true) + double(1e-8f));
    EXPECT_TRUE(n::equal_bits(out, expected));
    EXPECT_NEAR(out.at({0, 0}), .8485281371, 1e-7);
    EXPECT_NEAR(out.at({0, 1}), 1.1313708499, 1e-7);
    EXPECT_EQ(out.at({1, 0}), 0);
    EXPECT_EQ(out.shape(), x.shape());
  }
}
TEST(NnProgressiveGan, PixelRmsHasFirstAndHigherPathsAndRejectsInvalidGuardsOrInputs) {
  auto              t  = n::DType::Float64;
  auto              x  = n::tensor({.2, -.5, .7, 1.1}, {2, 2}, t).requires_grad_();
  n::TensorFunction fn = [](const auto& a) {
    return n::sum(n::square(n::pixel_rms_normalize(a[0]))) + n::sum(n::pixel_rms_normalize(a[0]));
  };
  EXPECT_TRUE(n::gradcheck(fn, {x}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {x}).ok);
  EXPECT_THROW(n::pixel_rms_normalize(n::scalar_like(1., x)), std::invalid_argument);
  EXPECT_THROW(n::pixel_rms_normalize(x, 0), std::invalid_argument);
  EXPECT_THROW(
      n::pixel_rms_normalize(n::full({2, 2}, std::numeric_limits<double>::infinity())),
      std::invalid_argument);
  EXPECT_THROW(n::pixel_rms_normalize(n::full({2, 2}, 1e30)), std::invalid_argument);
}
TEST(NnProgressiveGan, StageActivityIncludesOldRgbAtFinalFadeAndNeverStepsInactiveGroups) {
  EXPECT_EQ(m::ProgressiveActiveGroups(0), std::vector<int>({0, 1}));
  EXPECT_EQ(m::ProgressiveActiveGroups(1), std::vector<int>({0, 1, 2, 3}));
  EXPECT_EQ(m::ProgressiveActiveGroups(2), std::vector<int>({0, 2, 3}));
  EXPECT_EQ(m::ProgressiveActiveGroups(1, false), std::vector<int>({4, 5, 6, 7}));
  EXPECT_THROW(m::ProgressiveActiveGroups(-1), std::invalid_argument);
  EXPECT_THROW(m::ProgressiveActiveGroups(3), std::invalid_argument);
}
TEST(NnProgressiveGan, AllGroupsInitializeAtEndpointsWithSourceCensusAndImmutableReads) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    Model            model(type);
    auto             out = model.Forward(model.Alpha(0));
    std::vector<int> expected{1236, 5, 296, 5, 213, 8, 296, 8};
    std::int64_t     count = 0, leaves = 0;
    for (int k = 0; k < 8; ++k) {
      EXPECT_EQ(model.p[k]->count(), expected[k]);
      count += model.p[k]->count();
      leaves += model.p[k]->size();
    }
    EXPECT_EQ(count, 2067);
    EXPECT_EQ(leaves, 24);
    std::vector<std::vector<n::Tensor>> before;
    for (auto& p : model.p)
      before.push_back(Snapshot(*p));
    auto repeat = model.Forward(n::full({1}, 0, type));
    EXPECT_TRUE(n::equal_bits(out.fake, repeat.fake));
    for (int k = 0; k < 8; ++k)
      Same(*model.p[k], before[k]);
    EXPECT_EQ(out.old4.shape(), n::Shape({2, 4, 4, 1}));
    EXPECT_EQ(out.new8.shape(), n::Shape({2, 8, 8, 1}));
  }
}
TEST(NnProgressiveGan, GeneratorAndDiscriminatorFadeActualImagesAndFeaturesAtBothEndpoints) {
  Model model;
  auto  old = model.Generate(model.Alpha(0)), high = model.Generate(model.Alpha(1)),
       half = model.Generate(model.Alpha(.5));
  EXPECT_TRUE(n::equal_bits(old.image, n::upsample_nearest(old.old4, 2)));
  EXPECT_TRUE(n::equal_bits(high.image, high.new8));
  EXPECT_TRUE(n::equal_bits(half.image, .5 * half.new8 + .5 * n::upsample_nearest(half.old4, 2)));
  auto d = [&](double a) {
    return m::ProgressiveGanDiscriminate(
        *model.p[4], *model.p[5], *model.p[6], *model.p[7], model.x, model.Alpha(a));
  };
  auto low = d(0), hi = d(1), mid = d(.5);
  EXPECT_TRUE(n::equal_bits(low.mixed, low.old_path));
  EXPECT_TRUE(n::equal_bits(hi.mixed, hi.new_path));
  EXPECT_TRUE(n::equal_bits(mid.mixed, .5 * mid.new_path + .5 * mid.old_path));
  EXPECT_EQ(mid.base.shape(), n::Shape({2, 4, 4, 4}));
}
TEST(NnProgressiveGan, GeneratorLeastSquaresLossKeepsSourceNoHalfFactor) {
  Model model;
  model.Forward(model.Alpha(.5));
  for (int k = 4; k < 8; ++k)
    for (auto& name : model.p[k]->names())
      model.p[k]->set(name, n::zeros(model.p[k]->at(name).shape(), model.z.dtype()));
  auto out = model.Forward(model.Alpha(.5));
  EXPECT_EQ(out.d_loss.item(), .5);
  EXPECT_EQ(out.g_loss.item(), 1.);
}
TEST(NnProgressiveGan, NoiseAndLiveFadeHaveCheckedFirstAndSecondPaths) {
  Model             model;
  auto              a    = model.Alpha(.37).requires_grad_();
  n::TensorFunction fade = [&](const auto& in) {
    auto out = model.Generate(in[0]);
    return n::sum(n::square(out.image));
  };
  EXPECT_TRUE(n::gradcheck(fade, {a}).ok);
  EXPECT_TRUE(n::gradgradcheck(fade, {a}).ok);
  auto              input = n::tensor({.37, -.19}, {1, 2}, n::DType::Float64).requires_grad_();
  n::TensorFunction noise = [&](const auto& in) {
    auto z   = n::concat({in[0], n::full({1, 14}, .2, n::DType::Float64)}, 1);
    auto out = m::ProgressiveGanGenerate(
        *model.p[0], *model.p[1], *model.p[2], *model.p[3], z, model.Alpha(.5));
    return n::mean(n::square(out.image));
  };
  EXPECT_TRUE(n::gradcheck(noise, {input}, {.atol = 2e-7, .rtol = 3e-4}).ok);
  EXPECT_TRUE(n::gradgradcheck(noise, {input}, {.atol = 2e-7, .rtol = 3e-4}).ok);
}
TEST(NnProgressiveGan, AllActiveGradientsAreCapturedBeforeWritesAndFreshGUsesUpdatedD) {
  Model model;
  model.Forward(model.Alpha(.5));
  n::OptimizerConfig cfg;
  cfg.kind  = n::OptimizerKind::Adam;
  cfg.beta1 = .5;
  std::vector<std::unique_ptr<n::Optimizer>> optim;
  for (auto& p : model.p)
    optim.push_back(std::make_unique<n::Optimizer>(*p, .0002, cfg));
  std::vector<std::vector<n::Tensor>> initial;
  for (auto& p : model.p)
    initial.push_back(Snapshot(*p));
  for (bool generator : {false, true}) {
    auto           groups = m::ProgressiveActiveGroups(1, generator);
    auto           out    = model.Forward(model.Alpha(.5));
    n::GradOptions options;
    options.retain_graph = true;
    options.allow_unused = true;
    std::vector<std::vector<n::Tensor>> captured;
    for (int k : groups)
      captured.push_back(
          n::grad(generator ? out.g_loss : out.d_loss, model.p[k]->parameters(), options));
    std::vector<std::vector<n::Tensor>> before;
    for (auto& p : model.p)
      before.push_back(Snapshot(*p));
    for (std::size_t i = 0; i < groups.size(); ++i)
      optim[groups[i]]->step(captured[i]);
    for (int k = 0; k < 8; ++k)
      if (std::find(groups.begin(), groups.end(), k) == groups.end())
        Same(*model.p[k], before[k]);
  }
  for (auto& o : optim)
    EXPECT_EQ(o->iterations(), 1);
}
TEST(NnProgressiveGan, FrozenOldRgbWeightsDoNotFreezeTheSharedCoreOrOldEndpoint) {
  Model model;
  auto  out    = model.Generate(model.Alpha(0));
  auto  before = out.old4.detach().clone();
  auto  rgb    = Snapshot(*model.p[1]);
  model.p[0]->set("stem/b", model.p[0]->at("stem/b") + .25);
  auto changed = model.Generate(model.Alpha(0));
  Same(*model.p[1], rgb);
  EXPECT_GT(n::max(n::abs(changed.old4 - before)).item(), 1e-6);
}
TEST(NnProgressiveGan, InvalidAliasesGeometryAlphaAndRowsFailBeforeAnyLeaves) {
  Model model;
  auto  a = model.Alpha(.5);
  EXPECT_THROW(model.Forward(n::full({2}, .5, n::DType::Float64)), std::invalid_argument);
  EXPECT_THROW(model.Forward(model.Alpha(1.1)), std::invalid_argument);
  model.x = n::zeros({1, 8, 8, 1}, n::DType::Float64);
  EXPECT_THROW(model.Forward(a), std::invalid_argument);
  model.x = n::zeros({2, 7, 8, 1}, n::DType::Float64);
  EXPECT_THROW(model.Forward(a), std::invalid_argument);
  EXPECT_THROW(
      m::ProgressiveGanGenerate(*model.p[0], *model.p[0], *model.p[2], *model.p[3], model.z, a),
      std::invalid_argument);
  for (auto& p : model.p)
    EXPECT_EQ(p->size(), 0);
}
