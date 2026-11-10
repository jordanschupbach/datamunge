#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>
namespace nn = datamunge::nn;
namespace m  = nn::models;
namespace {
void Require(bool condition, const char* message) {
  EXPECT_TRUE(condition) << message;
}
void Near(double a, double b, double tolerance, const char* message) {
  EXPECT_NEAR(a, b, tolerance) << message;
}
} // namespace
TEST(NnTiedAutoencoder, SourceToyBothSharedPathsAndAllElevenDerivatives) {
  nn::ParameterStore p(1861);
  {
    nn::NoGradGuard guard;
    m::TiedAutoencoder(p, nn::zeros({1, 3}), 3, 2, true);
  }
  const std::vector<float> w{.2f, .1f, -.3f, .4f, .5f, -.2f}, eb{.1f, .2f}, db{-.1f, .05f, .2f},
      in{.1f, .2f, .7f, .8f, .3f, .4f};
  p.set("encoder/W", nn::from_vector(w, {3, 2}));
  p.set("encoder/b", nn::from_vector(eb, {2}));
  p.set("decoder/b", nn::from_vector(db, {3}));
  auto       input    = nn::from_vector(in, {2, 3});
  auto       net      = m::TiedAutoencoder(p, input, 3, 2, true);
  auto       loss     = nn::mse(net.reconstruction, input);
  auto       gradient = nn::grad(loss, p.parameters());
  const auto code = net.code.to_vector<float>(), recon = net.reconstruction.to_vector<float>();
  const auto gw = gradient[0].to_vector<float>(), geb = gradient[1].to_vector<float>(),
             gdb       = gradient[2].to_vector<float>();
  double encoder[3][2] = {}, decoder[3][2] = {}, ge[2] = {}, gd[3] = {}, mse = 0;
  for (int b = 0; b < 2; ++b) {
    double c[2] = {eb[0], eb[1]}, delta[3];
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 3; ++i)
        c[j] += in[b * 3 + i] * double(w[i * 2 + j]);
      Require(c[j] > 0, "toy avoids ReLU kink");
      Near(code[b * 2 + j], c[j], 5e-8, "all4toy codes");
    }
    for (int i = 0; i < 3; ++i) {
      double value = db[i];
      for (int j = 0; j < 2; ++j)
        value += c[j] * w[i * 2 + j];
      const double y = 1 / (1 + std::exp(-value)), error = y - in[b * 3 + i];
      Near(recon[b * 3 + i], y, 8e-8, "all6toy sigmoid predictions");
      mse += error * error / 6;
      delta[i] = error / 3 * y * (1 - y);
      gd[i] += delta[i];
      for (int j = 0; j < 2; ++j)
        decoder[i][j] += delta[i] * c[j];
    }
    for (int j = 0; j < 2; ++j) {
      double dz = 0;
      for (int i = 0; i < 3; ++i)
        dz += delta[i] * w[i * 2 + j];
      ge[j] += dz;
      for (int i = 0; i < 3; ++i)
        encoder[i][j] += in[b * 3 + i] * dz;
    }
  }
  Require(
      p.size() == 3 && p.count() == 11,
      "one sharedmatrix plus two biases,not a copied decoder parameter");
  Near(loss.item<float>(), mse, 2e-8, "known tied MSE");
  double encoder_norm = 0, decoder_norm = 0;
  for (int i = 0; i < 3; ++i) {
    Near(gdb[i], gd[i], 3e-8, "all3decoder bias derivatives");
    for (int j = 0; j < 2; ++j) {
      Near(
          gw[i * 2 + j],
          encoder[i][j] + decoder[i][j],
          3e-8,
          "ALL6shared derivatives equal encoder-path PLUS decoder-path");
      encoder_norm += std::abs(encoder[i][j]);
      decoder_norm += std::abs(decoder[i][j]);
    }
  }
  for (int j = 0; j < 2; ++j)
    Near(geb[j], ge[j], 3e-8, "all2encoder bias derivatives");
  Require(
      encoder_norm > 1e-3 && decoder_norm > 1e-3,
      "neither shared gradient contribution is silently missing");
  std::cout << "known tied toy4codes/6predictions/MSE/ALL11derivatives/BOTHsharedmatrix "
               "contributions passed\n";
}

TEST(NnTiedAutoencoder, AllTiedAndUntiedParameterAndInputFirstSecondDerivatives) {
  for (bool tied : {true, false}) {
    nn::ParameterStore p(1861, nn::DType::Float64);
    auto x = nn::tensor({.1, .2, .7, .8, .3, .4}, {2, 3}, nn::DType::Float64).requires_grad_();
    m::TiedAutoencoder(p, x, 3, 2, tied);
    p.set("encoder/W", nn::tensor({.2, .1, -.3, .4, .5, -.2}, {3, 2}, nn::DType::Float64));
    p.set("encoder/b", nn::tensor({.1, .2}, {2}, nn::DType::Float64));
    p.set("decoder/b", nn::tensor({-.1, .05, .2}, {3}, nn::DType::Float64));
    if (!tied)
      p.set("decoder/W", nn::transpose(p.at("encoder/W"), {1, 0}).contiguous());
    const auto          loss = nn::mse(m::TiedAutoencoder(p, x, 3, 2, tied).reconstruction, x);
    auto                gs   = nn::grad(loss, p.parameters());
    std::vector<double> analytic;
    for (const auto& g : gs) {
      auto v = g.to_vector<double>();
      analytic.insert(analytic.end(), v.begin(), v.end());
    }
    const auto initial = p.flatten();
    for (std::size_t i = 0; i < initial.size(); ++i) {
      auto changed = initial;
      changed[i] += 1e-6;
      p.unflatten(changed);
      double plus = nn::mse(m::TiedAutoencoder(p, x, 3, 2, tied).reconstruction, x).item();
      changed[i] -= 2e-6;
      p.unflatten(changed);
      double minus = nn::mse(m::TiedAutoencoder(p, x, 3, 2, tied).reconstruction, x).item();
      EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-9);
    }
    p.unflatten(initial);
    nn::TensorFunction objective = [&](const auto& args) {
      return nn::mse(m::TiedAutoencoder(p, args[0], 3, 2, tied).reconstruction, x.detach());
    };
    EXPECT_TRUE(nn::gradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
    EXPECT_TRUE(nn::gradgradcheck(objective, {x}, {.atol = 1e-8, .rtol = 1e-5}).ok);
    auto strided = nn::transpose(nn::transpose(x, {1, 0}).contiguous(), {1, 0});
    EXPECT_FALSE(strided.is_contiguous());
    EXPECT_EQ(
        m::TiedAutoencoder(p, x, 3, 2, tied).reconstruction.to_vector<double>(),
        m::TiedAutoencoder(p, strided, 3, 2, tied).reconstruction.to_vector<double>());
  }
}
TEST(NnTiedAutoencoder, SameInitialFunctionThenLiveTieAfterEveryAdamUpdate) {
  nn::ParameterStore tied(1853), untied(1853);
  auto               x = nn::full({2, 784}, .2);
  auto               t = m::TiedAutoencoder(tied, x);
  m::TiedAutoencoder(untied, x, 784, 32, false);
  for (const auto& name : tied.names())
    untied.set(name, tied.at(name).detach().clone());
  untied.set("decoder/W", nn::transpose(tied.at("encoder/W"), {1, 0}).contiguous());
  auto u = m::TiedAutoencoder(untied, x, 784, 32, false);
  EXPECT_EQ(t.reconstruction.to_vector<float>(), u.reconstruction.to_vector<float>());
  EXPECT_EQ(tied.count(), 25904);
  EXPECT_EQ(untied.count(), 50992);
  const auto          initial = tied.at("encoder/W").detach().clone();
  nn::OptimizerConfig options;
  options.kind    = nn::OptimizerKind::Adam;
  options.beta1   = .9;
  options.beta2   = .999;
  options.epsilon = 1e-8;
  nn::Optimizer optimizer(tied, .003, options), control(untied, .003, options);
  for (int step = 0; step < 3; ++step) {
    optimizer.minimize(nn::mse(m::TiedAutoencoder(tied, x).reconstruction, x));
    control.minimize(nn::mse(m::TiedAutoencoder(untied, x, 784, 32, false).reconstruction, x));
    auto updated = m::TiedAutoencoder(tied, x);
    EXPECT_FALSE(tied.contains("decoder/W"));
    EXPECT_EQ(tied.size(), 3u);
    EXPECT_EQ(
        updated.decoder_weight.contiguous().to_vector<float>(),
        nn::transpose(tied.at("encoder/W"), {1, 0}).contiguous().to_vector<float>());
    EXPECT_GT(nn::max(nn::abs(tied.at("encoder/W") - initial)).item(), 0);
  }
  EXPECT_FALSE(
      nn::equal_bits(untied.at("decoder/W"), nn::transpose(untied.at("encoder/W"), {1, 0})));
}
TEST(NnTiedAutoencoder, DecoderOnlyOwningHandlesAndAllDecoderGradients) {
  for (bool tied : {true, false}) {
    nn::ParameterStore p(1853, nn::DType::Float64);
    auto code    = nn::tensor({-.2, .3, .5, 0.}, {2, 2}, nn::DType::Float64).requires_grad_();
    auto decoded = m::TiedAutoencoderDecode(p, code, 3, 2, tied);
    EXPECT_EQ(p.size(), 2u);
    EXPECT_EQ(p.count(), 9);
    EXPECT_FALSE(p.contains("encoder/b"));
    EXPECT_EQ(p.contains("encoder/W"), tied);
    EXPECT_EQ(p.contains("decoder/W"), !tied);
    auto gs = nn::grad(nn::sum(decoded), {code, p.parameters()[0], p.parameters()[1]});
    for (const auto& g : gs) {
      EXPECT_TRUE(g.defined());
      EXPECT_GT(nn::sum(nn::abs(g)).item(), 0);
    }
    nn::TensorFunction objective = [&](const auto& args) {
      return nn::sum(m::TiedAutoencoderDecode(p, args[0], 3, 2, tied));
    };
    EXPECT_TRUE(nn::gradcheck(objective, {code}).ok);
    EXPECT_TRUE(nn::gradgradcheck(objective, {code}).ok);
  }
  nn::ParameterStore p(1853);
  nn::Tensor         code, recon;
  {
    const auto net = m::TiedAutoencoder(p, nn::full({2, 3}, .2), 3, 2);
    code           = net.code;
    recon          = net.reconstruction;
  }
  EXPECT_EQ(m::TiedAutoencoderDecode(p, code, 3, 2).to_vector<float>(), recon.to_vector<float>());
}
TEST(NnTiedAutoencoder, ValidationBeforeLeafCreationAndReluZeroConvention) {
  nn::ParameterStore p(1);
  EXPECT_THROW(m::TiedAutoencoder(p, nn::zeros({1, 3}), 3, 3), std::invalid_argument);
  EXPECT_THROW(m::TiedAutoencoder(p, nn::zeros({0, 3}), 3, 2), std::invalid_argument);
  EXPECT_THROW(m::TiedAutoencoder(p, nn::zeros({1, 4}), 3, 2), std::invalid_argument);
  EXPECT_THROW(
      m::TiedAutoencoder(p, nn::full({1, 3}, std::numeric_limits<double>::quiet_NaN()), 3, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::TiedAutoencoder(p, nn::zeros({1, 3}, nn::DType::Float64), 3, 2), std::invalid_argument);
  EXPECT_THROW(m::TiedAutoencoderDecode(p, nn::zeros({1, 3}), 3, 2), std::invalid_argument);
  EXPECT_EQ(p.size(), 0u);
  p.get("decoder/W", {2, 3}, nn::init::zeros());
  EXPECT_THROW(m::TiedAutoencoder(p, nn::zeros({1, 3}), 3, 2), std::invalid_argument);
  EXPECT_EQ(p.size(), 1u);
  nn::ParameterStore q(1);
  auto               x   = nn::zeros({1, 3}).requires_grad_();
  auto               net = m::TiedAutoencoder(q, x, 3, 2);
  auto               gs  = nn::grad(nn::sum(net.code), {x, q.at("encoder/b")});
  for (const auto& g : gs)
    EXPECT_EQ(nn::sum(nn::abs(g)).item(), 0);
}
