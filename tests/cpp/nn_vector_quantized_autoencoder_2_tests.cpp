#include <gtest/gtest.h>

#include <datamunge/nn/autoencoder_models.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/vector_quantized_autoencoder_2.hpp>
namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnVq2, DefaultCensusSourceEndpointsAndExplicitAdam) {
  n::ParameterStore p(2801);
  const auto        image = n::full({2, 28, 28, 1}, .3);
  const auto        o     = m::VectorQuantizedAutoencoder2(p, image);
  EXPECT_EQ(p.size(), 14);
  EXPECT_EQ(p.count(), 1157);
  EXPECT_EQ(o.encoded_top.shape(), (n::Shape{98, 4}));
  EXPECT_EQ(o.encoded_bottom.shape(), (n::Shape{392, 4}));
  EXPECT_EQ(o.top_feature.shape(), (n::Shape{2, 14, 14, 4}));
  EXPECT_EQ(o.probability.shape(), (n::Shape{2, 784}));
  EXPECT_TRUE(n::equal_bits(o.top.quantized, o.top.straight_through));
  EXPECT_TRUE(n::equal_bits(o.bottom.quantized, o.bottom.straight_through));
  const auto top    = m::VqLookup(o.top.indices.reshape({2, 49}), o.top_book),
             bottom = m::VqLookup(o.bottom.indices.reshape({2, 196}), o.bottom_book);
  EXPECT_TRUE(
      n::equal_bits(o.probability, n::sigmoid(m::Vq2Decoder(p, bottom, m::Vq2TopDecoder(p, top)))));
  EXPECT_FLOAT_EQ(
      o.loss.item<float>(),
      o.reconstruction_loss.item<float>() +
          ((o.top.codebook_loss.item<float>() + o.bottom.codebook_loss.item<float>()) +
           .25f * (o.top.commitment_loss.item<float>() + o.bottom.commitment_loss.item<float>())));
  const auto gradients = n::grad(o.loss, p.parameters());
  ASSERT_EQ(gradients.size(), 14);
  for (const auto& g : gradients)
    for (double x : g.to_vector<double>())
      EXPECT_TRUE(std::isfinite(x));
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer adam(p, .003, config);
  const auto   before = p.flatten();
  adam.step(gradients);
  EXPECT_NE(before, p.flatten());
}

TEST(NnVq2, FineEncodingDependsOnTopCodesAndFineLossReachesCoarseEncoder) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(2801, type);
  auto              image = n::full({2, 8, 12, 1}, .3, type).requires_grad_();
  m::VectorQuantizedAutoencoder2(p, image, n::Tensor{}, 3, 2, 3);
  p.set("top_codebook", n::tensor({.1, .2, .5, .6, -.2, .3}, {3, 2}, type));
  p.set("top_decoder/W", n::full(p.at("top_decoder/W").shape(), .1, type));
  p.set("top_decoder/b", n::full({2}, .1, type));
  p.set("bottom_projection/W", n::full(p.at("bottom_projection/W").shape(), .2, type));
  const auto o = m::VectorQuantizedAutoencoder2(p, image, n::Tensor{}, 3, 2, 3);
  const auto coupling =
      n::grad(n::mean(n::square(o.encoded_bottom)), {o.encoded_top}, {.retain_graph = true});
  EXPECT_GT(n::sum(n::abs(coupling[0])).item(), 1e-10);
  const auto books =
      n::grad(o.reconstruction_loss, {o.top_book, o.bottom_book}, {.allow_unused = true});
  EXPECT_FALSE(books[0].defined());
  EXPECT_FALSE(books[1].defined());
  const auto zero =
      m::VectorQuantizedAutoencoder2(p, image, n::zeros({2, 6}, n::DType::Int32), 3, 2, 3);
  const auto one =
      m::VectorQuantizedAutoencoder2(p, image, n::ones({2, 6}, n::DType::Int32), 3, 2, 3);
  EXPECT_GT(n::max(n::abs(zero.encoded_bottom - one.encoded_bottom)).item(), 1e-5);
  EXPECT_EQ(zero.top.indices.to_vector<int>(), (std::vector<int>(12, 0)));
  EXPECT_EQ(one.top.indices.to_vector<int>(), (std::vector<int>(12, 1)));
  EXPECT_EQ(
      n::grad(n::sum(one.top.straight_through), {one.encoded_top})[0].to_vector<double>(),
      (std::vector<double>(24, 1)));
}

TEST(NnVq2, OwningNestedFieldsFloat64StridesAndCodeOnlyRestore) {
  const auto          type = n::DType::Float64;
  n::Tensor           probability, top_ids, bottom_ids;
  std::vector<double> expected;
  n::NamedTensors     arrays;
  {
    n::ParameterStore p(2801, type);
    const auto        strided = n::slice_axis(n::full({2, 8, 24, 1}, .3, type), 2, 0, 24, 2);
    ASSERT_FALSE(strided.is_contiguous());
    const auto o = m::VectorQuantizedAutoencoder2(p, strided, n::Tensor{}, 3, 2, 3);
    probability  = o.probability;
    expected     = probability.to_vector<double>();
    top_ids      = o.top.indices.reshape({2, 6});
    bottom_ids   = o.bottom.indices.reshape({2, 24});
    EXPECT_TRUE(
        n::equal_bits(
            probability,
            m::VectorQuantizedAutoencoder2(p, strided.contiguous(), n::Tensor{}, 3, 2, 3)
                .probability));
    for (const auto& name : p.names())
      arrays.emplace_back(name, p.at(name).detach().clone());
  }
  EXPECT_EQ(probability.to_vector<double>(), expected);
  auto find = [&](const std::string& name) {
    for (const auto& [key, value] : arrays)
      if (key == name)
        return value;
    throw std::runtime_error(name);
  };
  n::ParameterStore decoder(2801, type);
  auto              top_feature = m::Vq2TopDecoder(decoder, n::zeros({6, 2}, type), 8, 12, 2);
  m::Vq2Decoder(decoder, n::zeros({24, 2}, type), top_feature, 3);
  ASSERT_EQ(decoder.size(), 6);
  for (const auto& name : decoder.names())
    decoder.set(name, find(name));
  top_feature = m::Vq2TopDecoder(decoder, m::VqLookup(top_ids, find("top_codebook"), 3), 8, 12, 2);
  const auto decoded = n::sigmoid(
      m::Vq2Decoder(decoder, m::VqLookup(bottom_ids, find("bottom_codebook"), 3), top_feature, 3));
  EXPECT_TRUE(n::equal_bits(probability, decoded));
}

TEST(NnVq2, ContinuousDecoderFirstSecondAndEverySmallLeafDerivative) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(2801, type);
  auto              top = n::full({2, 2}, .3, type).requires_grad_(),
       bottom           = n::full({8, 2}, .2, type).requires_grad_();
  auto f                = [&](const std::vector<n::Tensor>& a) {
    const auto feature = m::Vq2TopDecoder(p, a[0], 4, 8, 2);
    return .5 * n::sum(n::square(m::Vq2Decoder(p, a[1], feature, 2)));
  };
  f({top, bottom});
  p.unflatten(std::vector<double>(p.count(), .05));
  EXPECT_TRUE(n::gradcheck(f, {top, bottom}, {.atol = 1e-9, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {top, bottom}, {.atol = 1e-9, .rtol = 1e-5}).ok);
  std::vector<double> analytic;
  for (const auto& g : n::grad(f({top, bottom}), p.parameters())) {
    auto a = g.to_vector<double>();
    analytic.insert(analytic.end(), a.begin(), a.end());
  }
  const auto initial = p.flatten();
  ASSERT_EQ(initial.size(), 131);
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto a = initial;
    a[i] += 1e-6;
    p.unflatten(a);
    const double plus = f({top, bottom}).item();
    a[i] -= 2e-6;
    p.unflatten(a);
    const double minus = f({top, bottom}).item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(initial);
}

TEST(NnVq2, BadImagesGeometryOptionsAndOverridesPrecedeLeafCreation) {
  n::ParameterStore p(2801);
  const auto        image = n::zeros({1, 28, 28, 1});
  for (const auto& ids :
       {n::zeros({1, 49}),
        n::zeros({1, 48}, n::DType::Int32),
        n::full({1, 49}, 16, n::DType::Int32)})
    EXPECT_THROW(m::VectorQuantizedAutoencoder2(p, image, ids), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantizedAutoencoder2(p, image, n::Tensor{}, 0), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantizedAutoencoder2(p, image, n::Tensor{}, 16, 0), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantizedAutoencoder2(p, image, n::Tensor{}, 16, 4, 0), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantizedAutoencoder2(p, image, n::Tensor{}, 16, 4, 8, -.1), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantizedAutoencoder2(p, n::full({1, 28, 28, 1}, 1.1)), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantizedAutoencoder2(p, n::zeros({1, 27, 28, 1})), std::invalid_argument);
  EXPECT_THROW(m::Vq2TopDecoder(p, n::zeros({48, 4})), std::invalid_argument);
  EXPECT_THROW(
      m::Vq2Decoder(p, n::zeros({196, 4}), n::zeros({1, 13, 14, 4})), std::invalid_argument);
  EXPECT_THROW(
      m::Vq2Decoder(p, n::zeros({195, 4}), n::zeros({1, 14, 14, 4})), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
}

TEST(NnVq2, ConditionalPriorParentAlignmentBOSAndIndependentExports) {
  const auto top = n::from_vector(std::vector<int>{0, 1}, {1, 2});
  // Fine 2x4 rows: parent0 positions 0,1,4,5; parent1 positions 2,3,6,7.
  const auto             bottom = n::from_vector(std::vector<int>{0, 0, 1, 1, 0, 0, 1, 1}, {1, 8});
  m::Vq2ConditionalPrior prior(top, bottom, 1, 2, 2);
  EXPECT_EQ(prior.counts().shape(), (n::Shape{2, 3, 2}));
  EXPECT_EQ(
      prior.counts().to_vector<int>(), (std::vector<int>{2, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 0}));
  const auto before   = prior.probabilities().to_vector<double>();
  auto       exported = prior.probabilities();
  exported.fill_(0);
  EXPECT_EQ(prior.probabilities().to_vector<double>(), before);
  EXPECT_NEAR(
      prior.Nll(top, bottom), -std::log(.75) - 6 * std::log(2.5 / 3.) - std::log(.75), 1e-14);
  const auto ids = prior.Sample(top, n::full({1, 8}, .2, n::DType::Float64));
  EXPECT_EQ(ids.to_vector<int>(), bottom.to_vector<int>());
  // Different supplied parents alter fine samples; no image or network is consulted.
  const auto other = prior.Sample(
      n::from_vector(std::vector<int>{1, 0}, {1, 2}), n::full({1, 8}, .2, n::DType::Float64));
  EXPECT_NE(other.to_vector<int>(), ids.to_vector<int>());
}

TEST(NnVq2, ConditionalPriorExactCdfBoundariesStridesAndCopiedState) {
  const auto top = n::zeros({1, 1}, n::DType::Int32), bottom = n::zeros({1, 4}, n::DType::Int32);
  m::Vq2ConditionalPrior prior(top, bottom, 1, 1, 2);
  EXPECT_EQ(
      prior.Sample(top, n::full({1, 4}, .75, n::DType::Float64)).to_vector<int>(),
      (std::vector<int>{1, 1, 1, 1}));
  EXPECT_NEAR(prior.Nll(top, bottom), -std::log(.75) - 3 * std::log(.875), 1e-14);
  const auto noncontiguous = n::slice_axis(n::full({1, 8}, .2, n::DType::Float64), 1, 0, 8, 2);
  EXPECT_FALSE(noncontiguous.is_contiguous());
  EXPECT_TRUE(n::equal_bits(prior.Sample(top, noncontiguous), bottom));
  const auto bank = m::MidpointUniformNoise({1, 4}, 2833);
  EXPECT_TRUE(n::equal_bits(prior.Sample(top, bank), prior.Sample(top, bank)));
  auto counts = prior.counts();
  counts.fill_(99);
  EXPECT_NE(prior.counts().to_vector<int>(), counts.to_vector<int>());
}

TEST(NnVq2, ConditionalPriorRejectsMalformedGridsSymbolsAndUniforms) {
  const auto top = n::zeros({1, 1}, n::DType::Int32), bottom = n::zeros({1, 4}, n::DType::Int32);
  m::Vq2ConditionalPrior prior(top, bottom, 1, 1, 2);
  EXPECT_THROW(m::Vq2ConditionalPrior(top, bottom, 0, 1, 2), std::invalid_argument);
  EXPECT_THROW(m::Vq2ConditionalPrior(top, bottom, 1, 1, 2, 0), std::invalid_argument);
  EXPECT_THROW(m::Vq2ConditionalPrior(top, bottom, 1, 1, 0), std::invalid_argument);
  EXPECT_THROW(
      m::Vq2ConditionalPrior(n::zeros({0, 1}, n::DType::Int32), bottom, 1, 1, 2),
      std::invalid_argument);
  EXPECT_THROW(prior.Sample(top, n::zeros({1, 4})), std::invalid_argument);
  EXPECT_THROW(prior.Sample(top, n::full({1, 4}, 1, n::DType::Float64)), std::invalid_argument);
  EXPECT_THROW(prior.Sample(top, n::zeros({2, 4}, n::DType::Float64)), std::invalid_argument);
  EXPECT_THROW(prior.Nll(top, n::full({1, 4}, 2, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(prior.Nll(n::ones({1, 2}, n::DType::Int32), bottom), std::invalid_argument);
}
