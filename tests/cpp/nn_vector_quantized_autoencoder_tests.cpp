#include <gtest/gtest.h>

#include <datamunge/nn/autoencoder_models.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/vector_quantized_autoencoder.hpp>
namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnVectorQuantized, LowestTieExactForwardAndThreeIndependentGradientRoutes) {
  const auto t = n::DType::Float64;
  auto       e = n::tensor({-.5, 0., .9, .2}, {2, 2}, t).requires_grad_();
  auto       b = n::tensor({-1., 0., 0., 0., 1., 0.}, {3, 2}, t).requires_grad_();
  const auto q = m::VectorQuantize(e, b, 3, 2);
  EXPECT_EQ(q.indices.to_vector<int>(), (std::vector<int>{0, 2}));
  EXPECT_TRUE(n::equal_bits(q.quantized, q.straight_through));
  EXPECT_EQ(q.assignment.dtype(), t);
  const auto reconstruction = n::grad(
      .5 * n::sum(n::square(q.straight_through)),
      {e, b},
      {.create_graph = true, .allow_unused = true});
  EXPECT_TRUE(n::equal_bits(reconstruction[0], q.quantized));
  EXPECT_FALSE(reconstruction[1].defined());
  const auto second = n::grad(n::sum(reconstruction[0]), {e});
  EXPECT_EQ(second[0].to_vector<double>(), (std::vector<double>(4, 1)));
  const auto dictionary = n::grad(q.codebook_loss, {e, b}, {.allow_unused = true});
  EXPECT_FALSE(dictionary[0].defined());
  const auto commitment = n::grad(.25 * q.commitment_loss, {e, b}, {.allow_unused = true});
  EXPECT_FALSE(commitment[1].defined());
  const auto db = dictionary[1].to_vector<double>(), de = commitment[0].to_vector<double>();
  const std::vector<double> expected_book{-.25, 0., 0., 0., .05, -.1},
      expected_encoded{.0625, 0., -.0125, .025};
  for (int i = 0; i < 6; ++i)
    EXPECT_NEAR(db[i], expected_book[i], 1e-14);
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(de[i], expected_encoded[i], 1e-14);
  n::TensorFunction book_loss = [&](const auto& a) {
    return m::VectorQuantize(e, a[0], 3, 2).codebook_loss;
  };
  // Dictionary finite differences are valid away from assignment boundaries.
  auto interior = n::tensor({-.6, .1, .9, .2}, {2, 2}, t);
  book_loss = [&](const auto& a) { return m::VectorQuantize(interior, a[0], 3, 2).codebook_loss; };
  EXPECT_TRUE(n::gradcheck(book_loss, {b}, {.atol = 1e-9, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(book_loss, {b}, {.atol = 1e-9, .rtol = 1e-5}).ok);
}

TEST(NnVectorQuantized, OverrideStridesDuplicateCodesAndCancellationControl) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    auto       encoded = n::full({2, 2}, 1e20, type).requires_grad_();
    const auto book    = n::tensor({1., 2., 1., 2.}, {2, 2}, type);
    const auto ids     = n::from_vector(std::vector<int>{1, 0}, {1, 2});
    const auto o       = m::VectorQuantize(encoded, book, 2, 2, ids);
    EXPECT_TRUE(n::equal_bits(o.quantized, o.straight_through));
    EXPECT_EQ(o.indices.to_vector<int>(), (std::vector<int>{1, 0}));
    const auto tiny = m::VectorQuantize(n::zeros({2, 2}, type), book, 2, 2);
    EXPECT_EQ(tiny.indices.to_vector<int>(), (std::vector<int>{0, 0}));
    const auto strided = n::transpose(n::transpose(book).contiguous());
    EXPECT_FALSE(strided.is_contiguous());
    EXPECT_TRUE(n::equal_bits(m::VqLookup(ids, strided, 2), m::VqLookup(ids, book, 2)));
    EXPECT_EQ(
        n::grad(n::sum(o.straight_through), {encoded})[0].to_vector<double>(),
        (std::vector<double>(4, 1)));
  }
}

TEST(NnVectorQuantized, RejectsInvalidQuantizerAndLookupBeforeUse) {
  const auto e = n::zeros({2, 2}), b = n::zeros({3, 2});
  EXPECT_THROW(m::VectorQuantize(n::Tensor{}, b, 3, 2), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantize(e, b, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantize(e, b, 3, 0), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantize(e, n::zeros({3, 2}, n::DType::Float64), 3, 2), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantize(e, b, 3, 2, n::zeros({2})), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantize(e, b, 3, 2, n::full({2}, 3, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantize(e, b, 3, 2, n::zeros({1}, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantize(n::full({2, 2}, std::numeric_limits<double>::infinity()), b, 3, 2),
      std::invalid_argument);
  EXPECT_THROW(m::VectorQuantize(n::full({2, 2}, 1e30), b, 3, 2), std::invalid_argument);
  EXPECT_THROW(m::VqLookup(n::full({1, 2}, -1, n::DType::Int32), b, 3), std::invalid_argument);
  EXPECT_THROW(m::VqLookup(n::zeros({0, 2}, n::DType::Int32), b, 3), std::invalid_argument);
}

TEST(NnVectorQuantized, SpatialDefaultsCallerAdamAndImageFreeDecoder) {
  n::ParameterStore p(2707);
  const auto        x   = n::full({2, 28, 28, 1}, .3);
  const auto        out = m::VectorQuantizedAutoencoder(p, x);
  ASSERT_EQ(p.size(), 9);
  EXPECT_EQ(p.count(), 805);
  EXPECT_EQ(out.encoded.shape(), (n::Shape{98, 4}));
  EXPECT_EQ(out.probability.shape(), (n::Shape{2, 784}));
  const auto ids = out.vq.indices.reshape({2, 49});
  EXPECT_TRUE(
      n::equal_bits(
          out.probability, n::sigmoid(m::VqDecoder(p, m::VqLookup(ids, p.at("codebook"))))));
  EXPECT_FLOAT_EQ(
      out.loss.item<float>(),
      out.reconstruction_loss.item<float>() +
          (out.vq.codebook_loss.item<float>() + .25f * out.vq.commitment_loss.item<float>()));
  const auto         before = p.flatten();
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer adam(p, .003, config);
  adam.step(n::grad(out.loss, p.parameters()));
  EXPECT_NE(before, p.flatten());
  n::ParameterStore decoder(2707);
  m::VqDecoder(decoder, n::zeros({49, 4}));
  for (const auto& name : decoder.names())
    decoder.set(name, p.at(name));
  EXPECT_EQ(decoder.size(), 4);
  EXPECT_EQ(decoder.count(), 369);
  EXPECT_TRUE(
      n::equal_bits(
          m::VqDecoder(decoder, m::VqLookup(ids, p.at("codebook"))),
          m::VqDecoder(p, m::VqLookup(ids, p.at("codebook")))));
}

TEST(NnVectorQuantized, GeneralFloat64SpatialGeometryOwningFieldsAndLiveNoiseFreePaths) {
  const auto          type = n::DType::Float64;
  n::Tensor           retained;
  std::vector<double> values;
  {
    n::ParameterStore p(2707, type);
    auto              image = n::full({2, 8, 12, 1}, .3, type).requires_grad_();
    const auto        o     = m::VectorQuantizedAutoencoder(p, image, 3, 2, 3, .5);
    EXPECT_EQ(o.encoded.shape(), (n::Shape{12, 2}));
    EXPECT_EQ(o.vq.assignment.shape(), (n::Shape{12, 3}));
    const auto gradients = n::grad(o.loss, {image, p.at("codebook"), p.at("output/W")});
    for (const auto& g : gradients)
      for (double v : g.to_vector<double>())
        EXPECT_TRUE(std::isfinite(v));
    const auto book_grad = n::grad(o.reconstruction_loss, {o.book}, {.allow_unused = true});
    EXPECT_FALSE(book_grad[0].defined());
    retained           = o.probability;
    values             = retained.to_vector<double>();
    const auto strided = n::slice_axis(n::full({2, 8, 24, 1}, .3, type), 2, 0, 24, 2);
    EXPECT_FALSE(strided.is_contiguous());
    EXPECT_TRUE(
        n::equal_bits(
            retained, m::VectorQuantizedAutoencoder(p, strided, 3, 2, 3, .5).probability));
  }
  EXPECT_EQ(retained.to_vector<double>(), values);
}

TEST(NnVectorQuantized, InvalidArchitectureAndPixelsLeaveStoresEmpty) {
  n::ParameterStore p(2707);
  const auto        x = n::zeros({1, 28, 28, 1});
  EXPECT_THROW(m::VectorQuantizedAutoencoder(p, x, 0), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantizedAutoencoder(p, x, 16, 0), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantizedAutoencoder(p, x, 16, 4, 0), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantizedAutoencoder(p, x, 16, 4, 8, -.1), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantizedAutoencoder(p, n::full({1, 28, 28, 1}, 1.1)), std::invalid_argument);
  EXPECT_THROW(m::VectorQuantizedAutoencoder(p, n::zeros({1, 27, 28, 1})), std::invalid_argument);
  EXPECT_THROW(
      m::VectorQuantizedAutoencoder(p, n::zeros({1, 28, 28, 1}, n::DType::Float64)),
      std::invalid_argument);
  EXPECT_THROW(m::VqDecoder(p, n::zeros({48, 4})), std::invalid_argument);
  EXPECT_THROW(
      m::VqDecoder(p, n::zeros({49, 4}), 28, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
}

TEST(NnVectorQuantized, RasterPriorCountsBoundariesOwnershipAndNoImageDependence) {
  const auto             fit = n::from_vector(std::vector<int>{0, 1, 1, 0}, {2, 2});
  const m::VqMarkovPrior prior(fit, 2);
  EXPECT_EQ(prior.counts().to_vector<int>(), (std::vector<int>{0, 1, 1, 0, 1, 1}));
  EXPECT_EQ(
      prior.probabilities().to_vector<double>(), (std::vector<double>{.25, .75, .75, .25, .5, .5}));
  auto exported = prior.counts();
  exported.fill_(99);
  EXPECT_EQ(prior.counts().to_vector<int>(), (std::vector<int>{0, 1, 1, 0, 1, 1}));
  EXPECT_NEAR(prior.Nll(fit), -std::log(.5) - std::log(.75), 1e-14);
  const auto uniforms = n::tensor({.5, .75, .25, .5, .1, .75}, {2, 3}, n::DType::Float64);
  EXPECT_EQ(prior.Sample(uniforms).to_vector<int>(), (std::vector<int>{1, 1, 0, 1, 0, 1}));
  m::VqMarkovPrior singleton(n::zeros({1, 2}, n::DType::Int32), 1);
  EXPECT_EQ(singleton.Nll(n::zeros({2, 4}, n::DType::Int32)), 0);
  EXPECT_EQ(
      singleton.Sample(n::full({2, 4}, .99, n::DType::Float64)).to_vector<int>(),
      (std::vector<int>(8, 0)));
}

TEST(NnVectorQuantized, RasterPriorExplicitMidpointUniformsAndInvalidInputs) {
  m::VqMarkovPrior prior(n::zeros({2, 3}, n::DType::Int32), 2);
  const auto       uniforms = m::MidpointUniformNoise({20, 49}, 2729);
  EXPECT_TRUE(n::equal_bits(prior.Sample(uniforms), prior.Sample(uniforms)));
  EXPECT_THROW(m::VqMarkovPrior(n::zeros({0, 3}, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(m::VqMarkovPrior(n::zeros({2, 3})), std::invalid_argument);
  EXPECT_THROW(m::VqMarkovPrior(n::zeros({2, 3}, n::DType::Int32), 2, 0), std::invalid_argument);
  EXPECT_THROW(
      m::VqMarkovPrior(
          n::zeros({2, 3}, n::DType::Int32), 2, std::numeric_limits<double>::infinity()),
      std::invalid_argument);
  EXPECT_THROW(prior.Nll(n::full({1, 2}, 2, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(prior.Sample(n::zeros({1, 2})), std::invalid_argument);
  EXPECT_THROW(prior.Sample(n::full({1, 2}, 1, n::DType::Float64)), std::invalid_argument);
  EXPECT_THROW(prior.Sample(n::full({1, 2}, -1, n::DType::Float64)), std::invalid_argument);
}

TEST(NnVectorQuantized, UsageIncludesDeadCodesAndRejectsInvalidInputs) {
  auto u = m::VqUsage(n::from_vector(std::vector<int>{0, 1, 0, 1}, {2, 2}), 4);
  EXPECT_EQ(u.counts.to_vector<int>(), (std::vector<int>{2, 2, 0, 0}));
  EXPECT_DOUBLE_EQ(u.perplexity, 2);
  EXPECT_EQ(m::VqUsage(n::zeros({4}, n::DType::Int32)).perplexity, 1);
  EXPECT_THROW(m::VqUsage(n::zeros({0}, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(m::VqUsage(n::full({1}, 16, n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(m::VqUsage(n::zeros({1})), std::invalid_argument);
}

TEST(NnVectorQuantized, Float64DecoderEveryLeafDerivativeAndCodeHigherPaths) {
  const auto        type = n::DType::Float64;
  n::ParameterStore decoder(2707, type);
  auto              code = n::full({2, 2}, .3, type).requires_grad_();
  m::VqDecoder(decoder, code, 4, 8, 2, 2);
  decoder.unflatten(std::vector<double>(decoder.count(), .05));
  n::TensorFunction f = [&](const auto& values) {
    return .5 * n::sum(n::square(m::VqDecoder(decoder, values[0], 4, 8, 2, 2)));
  };
  EXPECT_TRUE(n::gradcheck(f, {code}, {.atol = 1e-9, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {code}, {.atol = 1e-9, .rtol = 1e-5}).ok);
  std::vector<double> analytic;
  for (const auto& g : n::grad(f({code}), decoder.parameters())) {
    const auto values = g.to_vector<double>();
    analytic.insert(analytic.end(), values.begin(), values.end());
  }
  const auto initial = decoder.flatten();
  ASSERT_EQ(initial.size(), 57);
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto values = initial;
    values[i] += 1e-6;
    decoder.unflatten(values);
    const auto plus = f({code}).item();
    values[i] -= 2e-6;
    decoder.unflatten(values);
    const auto minus = f({code}).item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-9);
  }
  decoder.unflatten(initial);
}
