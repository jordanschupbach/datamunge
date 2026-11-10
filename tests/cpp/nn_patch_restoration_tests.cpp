#include <gtest/gtest.h>

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/masked_autoencoder.hpp>
#include <datamunge/nn/patch_restoration.hpp>
namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnPatchRestoration, ExactSelectedBitsSignedZerosAndIndependentOwnership) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    auto x =
        n::cast(n::tensor({-0., .2, .3, .4, -0., .6, .7, .8}, {2, 2, 2}, n::DType::Float64), type);
    auto p             = n::full({2, 2, 2}, .9, type),
         mask          = n::cast(n::tensor({0., 1., 1., 0.}, {2, 2}, n::DType::Float64), type);
    const auto before  = x.to_vector<double>();
    auto       visible = m::VisiblePatchValues(x, mask, -0.);
    auto       values  = visible.to_vector<double>();
    EXPECT_TRUE(std::signbit(values[0]));
    EXPECT_TRUE(std::signbit(values[2]));
    EXPECT_TRUE(std::signbit(values[4]));
    EXPECT_EQ(values[1], before[1]);
    EXPECT_EQ(values[7], before[7]);
    auto out = m::CompositePatchValues(x, p, mask);
    values   = out.to_vector<double>();
    EXPECT_TRUE(std::signbit(values[0]));
    EXPECT_EQ(values[2], p.at({0, 1, 0}));
    EXPECT_EQ(values[7], before[7]);
    out.fill_(42);
    visible.fill_(13);
    EXPECT_EQ(x.to_vector<double>(), before);
    EXPECT_TRUE(std::signbit(x.at({0, 0, 0})));
    EXPECT_EQ(p.at({0, 0, 0}), type == n::DType::Float32 ? double(float(.9)) : .9);
  }
}
TEST(NnPatchRestoration, AllVisibleAllHiddenArbitraryRangesAndStrides) {
  const auto t = n::DType::Float64;
  auto       x = n::tensor({-3., 2., -1., 4., 5., -6., 7., 8., 9., 10., 11., 12.}, {2, 3, 2}, t),
       p       = x + 20;
  EXPECT_TRUE(n::equal_bits(m::CompositePatchValues(x, p, n::zeros({2, 3}, t)), x));
  EXPECT_TRUE(n::equal_bits(m::CompositePatchValues(x, p, n::ones({2, 3}, t)), p));
  EXPECT_TRUE(
      n::equal_bits(m::VisiblePatchValues(x, n::ones({2, 3}, t), -7), n::full({2, 3, 2}, -7, t)));
  const auto sx = n::transpose(n::transpose(x, {2, 1, 0}).contiguous(), {2, 1, 0});
  const auto sm = n::transpose(n::ones({3, 2}, t));
  EXPECT_FALSE(sx.is_contiguous());
  EXPECT_FALSE(sm.is_contiguous());
  EXPECT_TRUE(
      n::equal_bits(
          m::VisiblePatchValues(sx, sm, 2), m::VisiblePatchValues(x, sm.contiguous(), 2)));
  EXPECT_TRUE(n::equal_bits(m::CompositePatchValues(sx, p, sm), p));
}
TEST(NnPatchRestoration, ExactContinuousBranchDerivativesAndStoppedHiddenInputs) {
  const auto t    = n::DType::Float64;
  auto       x    = n::full({2, 2, 3}, .2, t).requires_grad_(),
       p          = n::full({2, 2, 3}, .7, t).requires_grad_();
  const auto mask = n::tensor({0., 1., 1., 0.}, {2, 2}, t);
  const auto g    = n::grad(n::sum(m::CompositePatchValues(x, p, mask)), {x, p});
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>{1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1}));
  EXPECT_EQ(g[1].to_vector<double>(), (std::vector<double>{0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}));
  EXPECT_EQ(
      n::grad(n::sum(m::VisiblePatchValues(x, mask, 9)), {x})[0].to_vector<double>(),
      g[0].to_vector<double>());
}
TEST(NnPatchRestoration, BothBranchFirstSecondDerivativesAndFloat32FillNarrowing) {
  const auto t           = n::DType::Float64;
  auto       x           = n::full({2, 2, 2}, .2, t).requires_grad_(),
       p                 = n::full({2, 2, 2}, .7, t).requires_grad_();
  const auto        mask = n::tensor({0., 1., 1., 0.}, {2, 2}, t);
  n::TensorFunction f    = [&](const auto& a) {
    return n::mean(n::square(m::CompositePatchValues(a[0], a[1], mask)));
  };
  EXPECT_TRUE(n::gradcheck(f, {x, p}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, p}).ok);
  n::TensorFunction v = [&](const auto& a) {
    return n::mean(n::square(m::VisiblePatchValues(a[0], mask, -.4)));
  };
  EXPECT_TRUE(n::gradcheck(v, {x}).ok);
  EXPECT_TRUE(n::gradgradcheck(v, {x}).ok);
  const double fill = .123456789123;
  const auto   out  = m::VisiblePatchValues(n::zeros({1, 1, 2}), n::ones({1, 1}), fill);
  EXPECT_EQ(out.at({0, 0, 0}), double(float(fill)));
  EXPECT_EQ(m::VisiblePatchValues(n::ones({1, 1, 1}), n::ones({1, 1}), 1e-100).item(), 0);
}
TEST(NnPatchRestoration, RejectsMalformedPatchesIncludingUnselectedNonfiniteValues) {
  const auto x = n::zeros({2, 3, 4}), mask = n::zeros({2, 3});
  for (const auto& bad : std::vector<n::Tensor>{
           n::Tensor{},
           n::zeros({2, 3}),
           n::zeros({0, 3, 4}),
           n::zeros({2, 3, 4}, n::DType::Int32),
           n::full({2, 3, 4}, NAN),
           n::full({2, 3, 4}, INFINITY)}) {
    EXPECT_THROW(m::VisiblePatchValues(bad, mask), std::invalid_argument);
    EXPECT_THROW(m::CompositePatchValues(x, bad, mask), std::invalid_argument);
  }
  EXPECT_THROW(m::CompositePatchValues(x, n::zeros({2, 3, 5}), mask), std::invalid_argument);
  EXPECT_THROW(
      m::CompositePatchValues(x, n::zeros({2, 3, 4}, n::DType::Float64), mask),
      std::invalid_argument);
}
TEST(NnPatchRestoration, RejectsMalformedLiveMasksAndUnrepresentableFill) {
  const auto x = n::zeros({2, 3, 4});
  for (const auto& mask : std::vector<n::Tensor>{
           n::Tensor{},
           n::zeros({2, 3, 1}),
           n::zeros({1, 3}),
           n::zeros({2, 3}, n::DType::Float64),
           n::zeros({2, 3}, n::DType::Bool),
           n::full({2, 3}, .5),
           n::full({2, 3}, -1),
           n::full({2, 3}, NAN),
           n::ones({2, 3}).requires_grad_()}) {
    EXPECT_THROW(m::VisiblePatchValues(x, mask), std::invalid_argument);
    EXPECT_THROW(m::CompositePatchValues(x, x, mask), std::invalid_argument);
  }
  for (double fill : {double(NAN), double(INFINITY), 1e100, -1e100})
    EXPECT_THROW(m::VisiblePatchValues(x, n::zeros({2, 3}), fill), std::invalid_argument);
}
TEST(NnPatchRestoration, NativeMaeInferenceIsTargetFreeAndCompositesCopyAllVisiblePixels) {
  n::NoGradGuard    guard;
  n::ParameterStore p(1877);
  const auto        image = n::full({2, 28, 28, 1}, .3);
  const auto selector     = m::MaeSelection(n::from_vector(std::vector<int>{23, 91}, {2}), 0, 4243),
             hidden       = 1. - n::sum(selector, {1});
  const auto patches      = m::MaePatches(image),
             observed = m::MaeImage(m::VisiblePatchValues(patches, hidden)).reshape({2, 28, 28, 1});
  const auto changed =
      m::MaeImage(m::VisiblePatchValues(patches, hidden, 1)).reshape({2, 28, 28, 1});
  const auto encoded = m::MaeEncode(p, observed, selector),
             other   = m::MaeEncode(p, changed, selector);
  EXPECT_TRUE(n::equal_bits(encoded.code, other.code));
  const auto prediction = m::MaeDecode(p, encoded.code, selector).prediction;
  EXPECT_EQ(p.count(), 13649);
  const auto composite = m::CompositePatchValues(m::MaePatches(observed), prediction, hidden);
  const auto bits = composite.to_vector<float>(), original = patches.to_vector<float>(),
             h = hidden.to_vector<float>();
  for (int b = 0; b < 2; ++b)
    for (int patch = 0; patch < 16; ++patch)
      if (h[b * 16 + patch] == 0)
        for (int j = 0; j < 49; ++j)
          EXPECT_EQ(bits[(b * 16 + patch) * 49 + j], original[(b * 16 + patch) * 49 + j]);
  EXPECT_TRUE(
      n::equal_bits(m::MaePatches(m::MaeImage(composite).reshape({2, 28, 28, 1})), composite));
}
