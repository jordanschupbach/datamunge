#include <gtest/gtest.h>

#include <datamunge/nn/autoencoder_models.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/latent_arithmetic.hpp>
namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnLatentArithmetic, OriginalGroupingEndpointsExtrapolationAndSignedCodes) {
  for (auto t : {n::DType::Float32, n::DType::Float64}) {
    const auto a      = n::tensor({1., 2., 3., 4.}, {2, 2}, t),
               b      = n::tensor({5., 6., 7., 8.}, {2, 2}, t);
    const auto levels = n::tensor({0., .5, 1., 2.}, {4}, t);
    auto       path   = m::LatentLerp(a, b, levels);
    EXPECT_EQ(path.shape(), (n::Shape{2, 4, 2}));
    EXPECT_TRUE(n::equal_bits(n::select(path, 1, 0), a));
    EXPECT_TRUE(n::equal_bits(n::select(path, 1, 2), b));
    EXPECT_EQ(path.at({0, 3, 0}), 9);
    auto edit = m::LatentDirection(a, b, n::tensor({-1., 0., 2.}, {3}, t));
    EXPECT_EQ(edit.at({0, 0, 0}), -4);
    EXPECT_TRUE(n::equal_bits(n::select(edit, 1, 1), a));
    const auto strided = n::transpose(n::transpose(a).contiguous());
    EXPECT_FALSE(strided.is_contiguous());
    EXPECT_TRUE(n::equal_bits(m::LatentLerp(strided, b, levels), path));
    path.fill_(77);
    EXPECT_EQ(a.at({0, 0}), 1);
  }
}
TEST(NnLatentArithmetic, AllAffineInputsHaveFirstAndMixedSecondDerivatives) {
  const auto t = n::DType::Float64;
  auto a = n::full({2, 3}, .4, t).requires_grad_(), b = n::full({2, 3}, -.7, t).requires_grad_();
  auto levels            = n::tensor({-.3, 0., .6, 1.4}, {4}, t).requires_grad_();
  n::TensorFunction lerp = [](const auto& v) {
    return n::mean(n::square(m::LatentLerp(v[0], v[1], v[2])));
  };
  n::TensorFunction edit = [](const auto& v) {
    return n::mean(n::square(m::LatentDirection(v[0], v[1], v[2])));
  };
  EXPECT_TRUE(n::gradcheck(lerp, {a, b, levels}).ok);
  EXPECT_TRUE(n::gradgradcheck(lerp, {a, b, levels}).ok);
  EXPECT_TRUE(n::gradcheck(edit, {a, b, levels}).ok);
  EXPECT_TRUE(n::gradgradcheck(edit, {a, b, levels}).ok);
  const auto g = n::grad(n::sum(m::LatentLerp(a, b, levels)), {a, b, levels});
  for (auto value : g[0].to_vector<double>())
    EXPECT_NEAR(value, 2.3, 1e-14);
  for (auto value : g[1].to_vector<double>())
    EXPECT_NEAR(value, 1.7, 1e-14);
  for (auto value : g[2].to_vector<double>())
    EXPECT_NEAR(value, -6.6, 1e-14);
}
TEST(NnLatentArithmetic, ClassMeansAreRowOrderedDetachedAndSupportStridesAndUnequalCounts) {
  for (auto t : {n::DType::Float32, n::DType::Float64}) {
    auto       code   = n::tensor({1., -3., 2., 4., 4., 9.}, {3, 2}, t).requires_grad_();
    const auto labels = n::from_vector(std::vector<std::int64_t>{0, 1, 0});
    auto       means  = m::LatentClassMeans(code, labels, 2);
    EXPECT_EQ(means.to_vector<double>(), (std::vector<double>{2.5, 3, 2, 4}));
    EXPECT_FALSE(means.requires_grad());
    EXPECT_EQ(means.dtype(), t);
    EXPECT_TRUE(
        n::equal_bits(
            m::LatentClassMeans(n::transpose(n::transpose(code).contiguous()), labels, 2), means));
    means.fill_(20);
    EXPECT_EQ(code.at({0, 0}), 1);
    const auto rounding =
        m::LatentClassMeans(n::tensor({1., 2., 4.}, {3, 1}, t), n::zeros({3}, n::DType::Int32), 1);
    EXPECT_EQ(rounding.item(), t == n::DType::Float32 ? double(float(7. / 3)) : 7. / 3);
  }
}
TEST(NnLatentArithmetic, FlattenOwnsRowsPreservesPathOrderAndGradient) {
  const auto t = n::DType::Float64;
  auto       code =
      n::transpose(
          n::tensor({0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.}, {2, 3, 2}, t), {1, 0, 2})
          .requires_grad_();
  EXPECT_FALSE(code.is_contiguous());
  auto rows = m::LatentFlatten(code);
  EXPECT_EQ(rows.shape(), (n::Shape{6, 2}));
  EXPECT_TRUE(rows.is_contiguous());
  EXPECT_EQ(rows.to_vector<double>(), code.contiguous().to_vector<double>());
  const auto g = n::grad(n::sum(n::square(rows)), {code})[0];
  EXPECT_EQ(g.to_vector<double>(), (2 * code).to_vector<double>());
  {
    n::NoGradGuard guard;
    rows.fill_(77);
  }
  EXPECT_EQ(code.at({0, 0, 0}), 0);
  auto contiguous  = n::ones({1, 2, 3});
  auto independent = m::LatentFlatten(contiguous);
  independent.fill_(9);
  EXPECT_EQ(contiguous.at({0, 0, 0}), 1);
}
TEST(NnLatentArithmetic, RejectsMalformedPathsLabelsAndMissingClasses) {
  const auto a = n::ones({2, 3}), levels = n::ones({4});
  for (const auto& bad : std::vector<n::Tensor>{
           n::Tensor{},
           n::zeros({0, 3}),
           n::ones({2, 3}, n::DType::Int32),
           n::full({2, 3}, NAN),
           n::ones({2, 4})}) {
    EXPECT_THROW(m::LatentLerp(a, bad, levels), std::invalid_argument);
    EXPECT_THROW(m::LatentDirection(bad, a, levels), std::invalid_argument);
  }
  for (const auto& bad : std::vector<n::Tensor>{
           n::Tensor{},
           n::zeros({0}),
           n::ones({4}, n::DType::Float64),
           n::full({4}, INFINITY),
           n::ones({1, 4})})
    EXPECT_THROW(m::LatentLerp(a, a, bad), std::invalid_argument);
  EXPECT_THROW(m::LatentFlatten(n::Tensor{}), std::invalid_argument);
  EXPECT_THROW(m::LatentFlatten(n::ones({2, 3})), std::invalid_argument);
  for (const auto& bad : std::vector<n::Tensor>{
           n::Tensor{},
           n::zeros({2}),
           n::zeros({1}, n::DType::Int32),
           n::full({2}, -1, n::DType::Int32),
           n::full({2}, 2, n::DType::Int64),
           n::zeros({2}, n::DType::Int32)})
    EXPECT_THROW(m::LatentClassMeans(a, bad, 2), std::invalid_argument);
  EXPECT_THROW(
      m::LatentClassMeans(n::Tensor{}, n::zeros({2}, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(m::LatentClassMeans(a, n::zeros({2}, n::DType::Int32), 0), std::invalid_argument);
  EXPECT_THROW(
      m::LatentClassMeans(
          n::full({2, 1}, 1e308, n::DType::Float64), n::zeros({2}, n::DType::Int32), 1),
      std::invalid_argument);
  EXPECT_THROW(
      m::latent_path_detail::Product(std::numeric_limits<std::int64_t>::max(), 2, 1),
      std::invalid_argument);
}
TEST(NnLatentArithmetic, CodeOnlyDecoderMatchesFullNetworkAndCreatesOnlyDecoderLeaves) {
  for (auto t : {n::DType::Float32, n::DType::Float64}) {
    n::ParameterStore p(1471, t);
    const auto        full   = m::UndercompleteAutoencoder(p, n::full({3, 4}, .3, t), 4, 2);
    const auto        before = p.flatten();
    EXPECT_TRUE(n::equal_bits(m::UndercompleteDecoder(p, full.code, 4, 2), full.reconstruction));
    EXPECT_EQ(p.flatten(), before);
    const auto strided = n::transpose(n::full({2, 3}, -.4, t));
    EXPECT_TRUE(
        n::equal_bits(
            m::UndercompleteDecoder(p, strided, 4, 2),
            m::UndercompleteDecoder(p, strided.contiguous(), 4, 2)));
  }
  n::ParameterStore decoder(1471);
  m::UndercompleteDecoder(decoder, n::full({2, 32}, -1));
  EXPECT_EQ(decoder.size(), 2u);
  EXPECT_EQ(decoder.count(), 25872);
  EXPECT_EQ(decoder.names(), (std::vector<std::string>{"decoder/W", "decoder/b"}));
}
TEST(NnLatentArithmetic, CodeOnlyDecoderHasAllParameterDerivativesAndCodeSecondPaths) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(1471, t);
  auto              code = n::tensor({-.4, .3, .2, .8}, {2, 2}, t).requires_grad_();
  auto       loss = [&]() { return n::mean(n::square(m::UndercompleteDecoder(p, code, 3, 2))); };
  const auto objective = loss();
  const auto gradients = n::grad(objective, p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gradients) {
    const auto v = g.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  const auto initial = p.flatten();
  for (std::size_t j = 0; j < initial.size(); ++j) {
    auto v = initial;
    v[j] += 1e-6;
    p.unflatten(v);
    const auto plus = loss().item();
    v[j] -= 2e-6;
    p.unflatten(v);
    const auto minus = loss().item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[j], 1e-9);
  }
  p.unflatten(initial);
  n::TensorFunction f = [&](const auto& args) {
    return n::mean(n::square(m::UndercompleteDecoder(p, args[0], 3, 2)));
  };
  EXPECT_TRUE(n::gradcheck(f, {code}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {code}).ok);
}
TEST(NnLatentArithmetic, DecoderRejectsBadInputsBeforeCreatingLeaves) {
  n::ParameterStore p(1471);
  for (const auto& bad : std::vector<n::Tensor>{
           n::Tensor{},
           n::zeros({0, 32}),
           n::ones({2, 31}),
           n::ones({2, 32}, n::DType::Float64),
           n::full({2, 32}, INFINITY)})
    EXPECT_THROW(m::UndercompleteDecoder(p, bad), std::invalid_argument);
  EXPECT_THROW(m::UndercompleteDecoder(p, n::ones({2, 32}), 32, 32), std::invalid_argument);
  EXPECT_EQ(p.size(), 0u);
}
