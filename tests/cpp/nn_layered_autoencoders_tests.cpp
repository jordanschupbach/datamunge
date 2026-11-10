#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <tuple>

namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnLayeredAutoencoders, SourceToyAllTwentyFourActivitiesAndFortyEightDerivatives) {
  n::ParameterStore p(1543);
  const n::Shape    widths{4, 3, 2, 3, 4};
  m::SigmoidAutoencoder(p, n::zeros({1, 4}), widths);
  std::vector<std::vector<float>> weights(4), biases(4);
  for (int layer = 0; layer < 4; ++layer) {
    for (int i = 0; i < widths[layer] * widths[layer + 1]; ++i)
      weights[layer].push_back(float(.1 * std::sin(i + layer + .3)));
    for (int i = 0; i < widths[layer + 1]; ++i)
      biases[layer].push_back(float(.01 * (i - 1)));
    p.set(
        "layer" + std::to_string(layer) + "/W",
        n::from_vector(weights[layer], {widths[layer], widths[layer + 1]}));
    p.set(
        "layer" + std::to_string(layer) + "/b", n::from_vector(biases[layer], {widths[layer + 1]}));
  }
  std::vector<float> values;
  for (int i = 0; i < 8; ++i)
    values.push_back(.1f * float(i + 1));
  const auto                       input    = n::from_vector(values, {2, 4});
  const auto                       actual   = m::SigmoidAutoencoder(p, input, widths);
  const auto                       loss     = n::mean(n::square(actual.back() - input));
  const auto                       gradient = n::grad(loss, p.parameters());
  std::vector<std::vector<double>> expected(5);
  expected[0].assign(values.begin(), values.end());
  for (int layer = 0; layer < 4; ++layer) {
    expected[layer + 1].resize(2 * widths[layer + 1]);
    for (int b = 0; b < 2; ++b)
      for (int j = 0; j < widths[layer + 1]; ++j) {
        double value = biases[layer][j];
        for (int i = 0; i < widths[layer]; ++i)
          value +=
              expected[layer][b * widths[layer] + i] * weights[layer][i * widths[layer + 1] + j];
        const double h                                 = 1 / (1 + std::exp(-value));
        expected[layer + 1][b * widths[layer + 1] + j] = h;
        EXPECT_NEAR(actual[layer].at({b, j}), h, 2e-7);
      }
  }
  std::vector<double> delta(8);
  double              objective = 0;
  for (int i = 0; i < 8; ++i) {
    const double h = expected[4][i], error = h - expected[0][i];
    objective += error * error / 8;
    delta[i] = 2 * error * h * (1 - h) / 8;
  }
  EXPECT_NEAR(loss.item(), objective, 2e-8);
  int checked = 0;
  for (int layer = 3; layer >= 0; --layer) {
    std::vector<double> previous(2 * widths[layer]);
    for (int j = 0; j < widths[layer + 1]; ++j) {
      double db = 0;
      for (int b = 0; b < 2; ++b)
        db += delta[b * widths[layer + 1] + j];
      EXPECT_NEAR(gradient[2 * layer + 1].at({j}), db, 2e-8);
      ++checked;
      for (int i = 0; i < widths[layer]; ++i) {
        double dw = 0;
        for (int b = 0; b < 2; ++b) {
          dw += expected[layer][b * widths[layer] + i] * delta[b * widths[layer + 1] + j];
          previous[b * widths[layer] + i] +=
              delta[b * widths[layer + 1] + j] * weights[layer][i * widths[layer + 1] + j];
        }
        EXPECT_NEAR(gradient[2 * layer].at({i, j}), dw, 2e-8);
        ++checked;
      }
    }
    for (std::size_t i = 0; i < previous.size(); ++i)
      previous[i] *= expected[layer][i] * (1 - expected[layer][i]);
    delta = previous;
  }
  EXPECT_EQ(checked, 48);
  EXPECT_EQ(p.count(), 48);
  EXPECT_EQ(p.size(), 8u);
}

TEST(NnLayeredAutoencoders, AllParameterAndInputFirstAndSecondFiniteDifferences) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(1543, type);
  const n::Shape    widths{4, 3, 2, 3, 4};
  const auto        x = n::tensor({.1, .2, .3, .4, .5, .6, .7, .8}, {2, 4}, type).requires_grad_();
  const auto        layers    = m::SigmoidAutoencoder(p, x, widths);
  const auto        gradients = n::grad(n::mean(n::square(layers.back() - x)), p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gradients) {
    const auto values = g.to_vector<double>();
    analytic.insert(analytic.end(), values.begin(), values.end());
  }
  const auto initial = p.flatten();
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto changed = initial;
    changed[i] += 1e-6;
    p.unflatten(changed);
    const double plus = n::mean(n::square(m::SigmoidAutoencoder(p, x, widths).back() - x)).item();
    changed[i] -= 2e-6;
    p.unflatten(changed);
    const double minus = n::mean(n::square(m::SigmoidAutoencoder(p, x, widths).back() - x)).item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(initial);
  const n::TensorFunction objective = [&](const auto& args) {
    return n::mean(n::square(m::SigmoidAutoencoder(p, args[0], widths).back() - args[0]));
  };
  EXPECT_TRUE(n::gradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  const auto strided = n::transpose(n::transpose(x, {1, 0}).contiguous(), {1, 0});
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_LT(
      n::max(n::abs(m::SigmoidAutoencoder(p, strided, widths).back() - layers.back())).item(),
      1e-14);
}

TEST(NnLayeredAutoencoders, StandaloneDecoderUsesOriginalNamesAndRetainsCodeGradients) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const n::Shape    widths{4, 3, 2, 3, 4};
  const auto        x       = n::full({2, 4}, .2, type);
  const auto        layers  = m::SigmoidAutoencoder(p, x, widths);
  const auto        code    = layers[1].detach().clone().requires_grad_();
  const auto        decoded = m::SigmoidAutoencoderDecode(p, code, widths);
  EXPECT_EQ(decoded.to_vector<double>(), layers.back().to_vector<double>());
  const auto derivative = n::grad(n::mean(n::square(decoded - x)), {code})[0];
  EXPECT_GT(n::sum(n::square(derivative)).item(), 0);
  const n::TensorFunction objective = [&](const auto& args) {
    return n::mean(n::square(m::SigmoidAutoencoderDecode(p, args[0], widths) - x));
  };
  EXPECT_TRUE(n::gradcheck(objective, {code}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  n::ParameterStore only_decoder(17);
  const auto        prediction =
      m::SigmoidAutoencoderDecode(only_decoder, n::zeros({2, 16}), {784, 64, 16, 64, 784});
  EXPECT_EQ(prediction.shape(), (n::Shape{2, 784}));
  EXPECT_EQ(only_decoder.count(), 52048);
  EXPECT_EQ(only_decoder.size(), 4u);
  EXPECT_FALSE(only_decoder.contains("layer0/W"));
  EXPECT_TRUE(only_decoder.contains("layer2/W"));
}

TEST(NnLayeredAutoencoders, FrozenCachedTargetsAndBothDecoderTransfersStayIndependent) {
  const auto        type = n::DType::Float64;
  n::ParameterStore first(7, type), second(11, type), joint(13, type);
  const auto        x = n::tensor({.1, .2, .3, .4, .5, .6, .7, .8}, {2, 4}, type);
  n::Tensor         cached;
  {
    n::NoGradGuard guard;
    cached = m::SigmoidAutoencoder(first, x, {4, 3, 4})[0];
    m::SigmoidAutoencoder(second, cached, {3, 2, 3});
    m::SigmoidAutoencoder(joint, x, {4, 3, 2, 3, 4});
  }
  EXPECT_FALSE(cached.requires_grad());
  const auto frozen1 = first.flatten(), frozen2 = second.flatten(),
             cached_values = cached.to_vector<double>();
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer optimizer(joint, .003, config);
  const auto   held_leaf = joint.at("layer0/W");
  for (const auto& [target, source_name, store] :
       std::vector<std::tuple<std::string, std::string, n::ParameterStore*>>{
           {"layer0", "layer0", &first},
           {"layer3", "layer1", &first},
           {"layer1", "layer0", &second},
           {"layer2", "layer1", &second}})
    for (const auto& suffix : {"/W", "/b"})
      joint.set(target + suffix, store->at(source_name + suffix));
  EXPECT_EQ(held_leaf.to_vector<double>(), first.at("layer0/W").to_vector<double>());
  const auto direct = m::SigmoidAutoencoder(joint, x, {4, 3, 2, 3, 4});
  const auto inner  = m::SigmoidAutoencoder(second, cached, {3, 2, 3});
  const auto nested = m::SigmoidAutoencoderDecode(first, inner.back(), {4, 3, 4});
  EXPECT_EQ(direct.back().to_vector<double>(), nested.to_vector<double>());
  const auto before = joint.flatten();
  optimizer.minimize(n::mean(n::square(direct.back() - x)));
  EXPECT_NE(joint.flatten(), before);
  EXPECT_EQ(first.flatten(), frozen1);
  EXPECT_EQ(second.flatten(), frozen2);
  EXPECT_EQ(cached.to_vector<double>(), cached_values);
}

TEST(NnLayeredAutoencoders, InvalidGeometryAndInputsFailBeforeAnyLeafAndOriginalCensus) {
  n::ParameterStore p(1);
  const auto        input = n::zeros({1, 4});
  for (const auto& widths : std::vector<n::Shape>{
           {}, {4, 2}, {4, 3, 2, 4}, {4, 4, 4}, {4, 0, 4}, {4, 2, 5}, {4, 2, 3, 2, 4}}) {
    EXPECT_THROW(m::SigmoidAutoencoder(p, input, widths), std::invalid_argument);
    EXPECT_EQ(p.size(), 0u);
  }
  EXPECT_THROW(m::SigmoidAutoencoder(p, n::zeros({1, 3}), {4, 2, 4}), std::invalid_argument);
  EXPECT_THROW(
      m::SigmoidAutoencoder(p, n::zeros({1, 4}, n::DType::Float64), {4, 2, 4}),
      std::invalid_argument);
  EXPECT_THROW(
      m::SigmoidAutoencoder(
          p, n::full({1, 4}, std::numeric_limits<double>::quiet_NaN()), {4, 2, 4}),
      std::invalid_argument);
  EXPECT_THROW(m::SigmoidAutoencoderDecode(p, n::zeros({1, 3}), {4, 2, 4}), std::invalid_argument);
  EXPECT_EQ(p.size(), 0u);
  n::ParameterStore first(1549), second(1553), joint(1553);
  m::SigmoidAutoencoder(first, n::zeros({1, 784}), {784, 64, 784});
  m::SigmoidAutoencoder(second, n::zeros({1, 64}), {64, 16, 64});
  m::SigmoidAutoencoder(joint, n::zeros({1, 784}), {784, 64, 16, 64, 784});
  EXPECT_EQ(first.count(), 101200);
  EXPECT_EQ(second.count(), 2128);
  EXPECT_EQ(joint.count(), 103328);
}
