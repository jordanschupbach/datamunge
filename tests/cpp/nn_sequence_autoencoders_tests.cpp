#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>

namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnSequenceAutoencoders, ImageRowsPreserveOrderStridesAndEveryGradient) {
  const auto type     = n::DType::Float64;
  const auto images   = n::arange(0, 24, 1, type).reshape({2, 3, 4, 1}).requires_grad_();
  const auto sequence = m::ImageRowSequence(images, 3, 4);
  EXPECT_EQ(sequence.shape(), (n::Shape{2, 3, 4}));
  EXPECT_EQ(sequence.to_vector<double>(), images.to_vector<double>());
  EXPECT_EQ(
      m::ImageRowSequence(images.reshape({2, 12}), 3, 4).to_vector<double>(),
      images.to_vector<double>());
  const auto strided = n::transpose(n::transpose(sequence, {0, 2, 1}).contiguous(), {0, 2, 1});
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_EQ(m::ImageRowSequence(strided, 3, 4).to_vector<double>(), sequence.to_vector<double>());
  EXPECT_EQ(
      n::grad(n::sum(sequence), {images})[0].to_vector<double>(), (std::vector<double>(24, 1)));
  EXPECT_THROW(m::ImageRowSequence(images, 2, 6), std::invalid_argument);
  EXPECT_THROW(m::ImageRowSequence(n::zeros({0, 3, 4}), 3, 4), std::invalid_argument);
}

TEST(NnSequenceAutoencoders, ScalarThreeStepStatesAndRepeatedCodeMatchHandRecurrences) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(5, type);
  const auto        x = n::tensor({.2, .4, .8}, {1, 3, 1}, type);
  m::RecurrentAutoencoder(p, x, 3, 1, 1);
  p.unflatten({.5, .25, 0., .7, .3, -.1, 1.2, -.2});
  const auto out = m::RecurrentAutoencoder(p, x, 3, 1, 1);
  EXPECT_EQ(p.count(), 8);
  double              h = 0;
  std::vector<double> enc, dec, pred;
  for (double input : {.2, .4, .8}) {
    h = std::tanh(input * .5 + h * .25);
    enc.push_back(h);
  }
  EXPECT_NEAR(out.code.item(), h, 1e-15);
  double d = 0;
  for (int t = 0; t < 3; ++t) {
    d = std::tanh(h * .7 + d * .3 - .1);
    dec.push_back(d);
    pred.push_back(1 / (1 + std::exp(-(d * 1.2 - .2))));
  }
  for (int t = 0; t < 3; ++t) {
    EXPECT_NEAR(out.encoder_states.at({0, t, 0}), enc[t], 1e-15);
    EXPECT_NEAR(out.decoder_states.at({0, t, 0}), dec[t], 1e-15);
    EXPECT_NEAR(out.reconstruction.at({0, t, 0}), pred[t], 1e-15);
  }
  const auto decoded = m::RecurrentDecode(p, out.code, 3, 1);
  EXPECT_EQ(decoded.reconstruction.to_vector<double>(), out.reconstruction.to_vector<double>());
  EXPECT_FALSE(decoded.encoder_states.defined());
  EXPECT_FALSE(m::RecurrentEncode(p, x, 3, 1, 1).decoder_states.defined());
}

TEST(NnSequenceAutoencoders, AllParameterAndInputDerivativesAcrossBothUnrolledHalves) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto        x      = n::tensor({.1, .6, .7, .2, .4, .8}, {1, 3, 2}, type).requires_grad_();
  const auto        result = m::RecurrentAutoencoder(p, x, 3, 2, 2);
  EXPECT_EQ(p.count(), 26);
  const auto          gradients = n::grad(n::mse(result.reconstruction, x), p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gradients) {
    const auto v = g.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  const auto initial   = p.flatten();
  const auto objective = [&]() {
    return n::mse(m::RecurrentAutoencoder(p, x, 3, 2, 2).reconstruction, x).item();
  };
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto altered = initial;
    altered[i] += 1e-6;
    p.unflatten(altered);
    const double plus = objective();
    altered[i] -= 2e-6;
    p.unflatten(altered);
    const double minus = objective();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(initial);
  const n::TensorFunction loss = [&](const auto& args) {
    return n::mse(m::RecurrentAutoencoder(p, args[0], 3, 2, 2).reconstruction, args[0]);
  };
  EXPECT_TRUE(n::gradcheck(loss, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(loss, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  const auto encoder      = m::RecurrentEncode(p, x, 3, 2, 2);
  const auto through_time = n::grad(n::sum(encoder.code), {x})[0];
  for (int t = 0; t < 3; ++t)
    EXPECT_GT(n::sum(n::square(n::select(through_time, 1, t))).item(), 0);
}

TEST(NnSequenceAutoencoders, OriginalCensusHistoriesStandaloneDecodeAndImmutableBatchInference) {
  n::ParameterStore  p(1871);
  std::vector<float> values(2 * 28 * 28);
  for (int i = 0; i < int(values.size()); ++i)
    values[i] = float((i * 13) % 31) / 30;
  const auto x   = n::from_vector(values, {2, 28, 28});
  const auto net = m::RecurrentAutoencoder(p, x);
  EXPECT_EQ(p.size(), 8u);
  EXPECT_EQ(p.count(), 4956);
  EXPECT_EQ(net.code.shape(), (n::Shape{2, 32}));
  EXPECT_EQ(net.encoder_states.shape(), (n::Shape{2, 28, 32}));
  EXPECT_EQ(net.decoder_states.shape(), (n::Shape{2, 28, 32}));
  EXPECT_EQ(n::select(net.encoder_states, 1, 27).to_vector<float>(), net.code.to_vector<float>());
  const auto gradients = n::grad(n::mse(net.reconstruction, x), p.parameters());
  for (const auto& g : gradients) {
    double norm = 0;
    for (double v : g.to_vector<double>()) {
      EXPECT_TRUE(std::isfinite(v));
      norm += v * v;
    }
    EXPECT_GT(norm, 0);
  }
  const auto     before = p.flatten();
  n::NoGradGuard guard;
  EXPECT_EQ(
      m::RecurrentDecode(p, net.code).reconstruction.to_vector<float>(),
      net.reconstruction.to_vector<float>());
  EXPECT_EQ(
      m::RecurrentAutoencoder(p, x).reconstruction.to_vector<float>(),
      net.reconstruction.to_vector<float>());
  EXPECT_LT(
      n::max(
          n::abs(
              m::RecurrentAutoencoder(p, n::take_rows(x, {0})).reconstruction -
              n::take_rows(net.reconstruction, {0})))
          .item(),
      1e-6);
  EXPECT_EQ(before, p.flatten());
}

TEST(NnSequenceAutoencoders, InvalidInputsFailBeforeCreatingAnyParameters) {
  n::ParameterStore p(1);
  for (const auto& x : std::vector<n::Tensor>{
           n::Tensor{},
           n::zeros({1, 3, 2}, n::DType::Int32),
           n::zeros({1, 3, 2}, n::DType::Float64),
           n::zeros({1, 6}),
           n::zeros({0, 3, 2}),
           n::full({1, 3, 2}, std::numeric_limits<double>::quiet_NaN())}) {
    EXPECT_THROW(m::RecurrentAutoencoder(p, x, 3, 2, 2), std::invalid_argument);
    EXPECT_EQ(p.size(), 0u);
  }
  EXPECT_THROW(m::RecurrentAutoencoder(p, n::zeros({1, 3, 2}), 3, 2, 6), std::invalid_argument);
  EXPECT_THROW(
      m::RecurrentAutoencoder(
          p, n::zeros({1, 3, 2}), 3, 2, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_THROW(m::RecurrentDecode(p, n::zeros({1, 2}), 0, 2), std::invalid_argument);
  EXPECT_THROW(
      m::RecurrentDecode(p, n::zeros({1, 2}, n::DType::Float64), 3, 2), std::invalid_argument);
  EXPECT_EQ(p.size(), 0u);
  EXPECT_EQ(
      m::RecurrentDecode(p, n::zeros({1, 2}), 1, 1).reconstruction.shape(), (n::Shape{1, 1, 1}));
}
