
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnSpatialTransformerModels, AlignedIdentityTranslationAndRotationMatchPixelOracle) {
  const auto type = n::DType::Float64;
  const auto x = n::sin(.37 * n::arange(0, 9 * 9 * 2).to(type) + .2).reshape({1, 9, 9, 2});
  for (int transform = 0; transform < 3; ++transform) {
    const auto theta =
        n::from_vector(transform == 0   ? std::vector<double>{1, 0, 0, 0, 1, 0}
                       : transform == 1 ? std::vector<double>{1, 0, .25, 0, 1, 0}
                                        : std::vector<double>{0, -1, 0, 1, 0, 0},
                       {1, 6});
    const auto y = n::models::AffineSample(x, theta, 9, 9);
    for (int row = 0; row < 9; ++row)
      for (int col = 0; col < 9; ++col)
        for (int ch = 0; ch < 2; ++ch) {
          const int r = transform == 2 ? col : row,
                    c = transform == 2 ? 8 - row : col + (transform == 1);
          const double expected =
              (r >= 0 && r < 9 && c >= 0 && c < 9) ? x.at({0, r, c, ch}) : 0;
          EXPECT_NEAR(y.at({0, row, col, ch}), expected, 1e-12);
        }
  }
}
TEST(NnSpatialTransformerModels, EveryAffineDerivativeMatchesFractionalCentralDifferences) {
  const auto type = n::DType::Float64;
  const auto x = n::sin(.37 * n::arange(0, 9 * 9 * 2).to(type) + .2).reshape({1, 9, 9, 2}),
             weights = n::sin(.31 * n::arange(0, 9 * 9 * 2).to(type) + .4).reshape(x.shape());
  const auto original =
      n::from_vector(std::vector<double>{.93, .11, .047, -.083, 1.06, -.031}, {1, 6});
  const auto theta = original.detach().requires_grad_();
  const auto derivative =
      n::grad(n::sum(n::models::AffineSample(x, theta, 9, 9) * weights), {theta})[0];
  n::NoGradGuard guard;
  const double h = 1e-6;
  for (int i = 0; i < 6; ++i) {
    const auto direction =
        n::one_hot(n::from_vector(std::vector<int64_t>{i}).to(n::DType::Int64), 6, 1, 0, type).reshape({1, 6});
    const double plus = n::sum(n::models::AffineSample(x, original + h * direction, 9, 9) *
                               weights)
                            .item(),
                 minus = n::sum(n::models::AffineSample(x, original - h * direction, 9, 9) *
                                weights)
                             .item();
    EXPECT_NEAR(derivative.at({0, i}), (plus - minus) / (2 * h), 1e-7);
  }
}
TEST(NnSpatialTransformerModels, IdentityHeadHasZeroInteriorGradientUntilActivated) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x =
      (.5 + .3 * n::sin(.37 * n::arange(0, 2 * 42 * 42).to(type))).reshape({2, 42, 42, 1});
  const auto initial = n::models::SpatialTransformer(p, "stn", x);
  const auto identity = n::from_vector(std::vector<double>{1, 0, 0, 0, 1, 0}, {1, 6});
  EXPECT_EQ(n::max(n::abs(initial.theta - identity)).item(), 0);
  EXPECT_LT(n::max(n::abs(initial.output - x)).item(), 1e-12);
  EXPECT_EQ(p.count(), 131896);
  const auto gradients = n::grad(n::mean(n::square(initial.output)), p.parameters());
  const auto names = p.names();
  double head = 0;
  for (size_t i = 0; i < gradients.size(); ++i) {
    const double norm = n::sum(n::abs(gradients[i])).item();
    if (names[i].find("/theta/") == std::string::npos)
      EXPECT_EQ(norm, 0);
    else
      head += norm;
  }
  EXPECT_GT(head, 0);
  p.set("stn/theta/W", (.0001 * n::sin(.37 * n::arange(0, 300).to(type))).reshape({50, 6}));
  const auto active = n::models::SpatialTransformer(p, "stn", x);
  double interior = 0;
  const auto live = n::grad(n::mean(n::square(active.output)), p.parameters());
  for (size_t i = 0; i < live.size(); ++i) {
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(live[i])).item()));
    if (names[i].find("/theta/") == std::string::npos)
      interior += n::sum(n::abs(live[i])).item();
  }
  EXPECT_GT(interior, 0);
}
TEST(NnSpatialTransformerModels, TensorDistortionIsSeededQuantizedAndGeometricallyBounded) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    const auto image =
        n::linspace(0, 1, 3 * 28 * 28, type).reshape({3, 28, 28, 1}).detach().requires_grad_();
    n::Generator a(101), b(101), c(202);
    const auto first = n::models::DistortMnist(image, a),
               same = n::models::DistortMnist(image, b),
               other = n::models::DistortMnist(image, c);
    EXPECT_EQ(first.images.shape(), n::Shape({3, 42, 42, 1}));
    EXPECT_EQ(first.images.dtype(), type);
    EXPECT_FALSE(first.images.requires_grad());
    EXPECT_EQ(n::max(n::abs(first.images - same.images)).item(), 0);
    EXPECT_EQ(n::max(n::abs(first.theta - same.theta)).item(), 0);
    EXPECT_GT(n::max(n::abs(first.theta - other.theta)).item(), 0);
    EXPECT_GE(n::min(first.images).item(), 0);
    EXPECT_LE(n::max(first.images).item(), 1);
    EXPECT_LT(n::max(n::abs(first.images * 255 - n::round(first.images * 255))).item(), 2e-5);
    for (int row = 0; row < 3; ++row) {
      const double xx = first.theta.at({row, 0}), xy = first.theta.at({row, 1}),
                   yx = first.theta.at({row, 3}), yy = first.theta.at({row, 4});
      EXPECT_NEAR(xx, yy, 1e-7);
      EXPECT_NEAR(xy, -yx, 1e-7);
      const double scale = std::hypot(xx, yx);
      EXPECT_GE(scale, 20.5 / (13.5 * 1.2) - 1e-6);
      EXPECT_LE(scale, 20.5 / (13.5 * .7) + 1e-6);
      EXPECT_GT(xx * yy - xy * yx, 0);
    }
  }
}
TEST(NnSpatialTransformerModels, CompleteBaselineAndTransformedClassifierCountsAndBackward) {
  n::set_num_threads(2);
  for (bool transformer : {false, true}) {
    n::ParameterStore p(17);
    const auto x =
        n::sin(.37 * n::arange(0, 2 * 42 * 42).to(n::DType::Float32)).reshape({2, 42, 42, 1});
    const auto out = n::models::StnMnistClassifier(p, x, transformer);
    EXPECT_EQ(p.count(), transformer ? 1176642 : 1044746);
    EXPECT_EQ(out.logits.shape(), n::Shape({2, 10}));
    EXPECT_EQ(out.theta.defined(), transformer);
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  }
}
TEST(NnSpatialTransformerModels, InvalidPixelShapesRangesAndTransformBatchesAreRejected) {
  n::Generator gen(17);
  EXPECT_THROW(n::models::DistortMnist(n::full({1, 28, 28, 1}, 1.1), gen),
               std::invalid_argument);
  EXPECT_THROW(n::models::DistortMnist(n::ones({1, 27, 28, 1}), gen), std::invalid_argument);
  n::ParameterStore p(17);
  EXPECT_THROW(n::models::StnMnistClassifier(p, n::ones({1, 28, 28, 1})),
               std::invalid_argument);
  EXPECT_THROW(n::models::AffineSample(n::ones({2, 5, 5, 1}), n::ones({1, 6}), 5, 5),
               std::invalid_argument);
}
