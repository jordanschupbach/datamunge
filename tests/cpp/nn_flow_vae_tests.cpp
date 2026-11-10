#include <gtest/gtest.h>

#include <datamunge/nn/flow_variational_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnFlowVae, HandAffineScaleShiftInverseAndCoordinateOrdering) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(3049, t);
  const auto        z = n::tensor({.4, -.2}, {1, 2}, t), c = n::ones({1, 1}, t);
  m::ConditionalAffinePosteriorFlow(p, "f", z, c, false, false, 2, 2);
  p.set("f/heads/W", n::zeros({2, 2}, t));
  p.set("f/heads/b", n::tensor({std::atanh(.4), .3}, {2}, t));
  for (bool swap : {false, true}) {
    auto f = m::ConditionalAffinePosteriorFlow(p, "f", z, c, swap, false, 2, 2);
    auto b = m::ConditionalAffinePosteriorFlow(p, "f", f.value, c, swap, true, 2, 2);
    EXPECT_NEAR(f.log_determinant.item(), .2, 1e-15);
    EXPECT_NEAR(b.log_determinant.item(), -.2, 1e-15);
    EXPECT_NEAR(f.value.at({0, swap ? 1 : 0}), z.at({0, swap ? 1 : 0}), 0);
    EXPECT_NEAR(f.value.at({0, swap ? 0 : 1}), z.at({0, swap ? 0 : 1}) * std::exp(.2) + .3, 1e-15);
    EXPECT_NEAR(n::max(n::abs(b.value - z)).item(), 0, 1e-15);
  }
}
TEST(NnFlowVae, FullFourCoordinateJacobianIncludesConditionersAndMatchesAnalyticLogdet) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(3049, t);
  auto              z = n::tensor({.2, -.4, .6, -.8}, {1, 4}, t).requires_grad_();
  auto              c = n::tensor({.1, .2}, {1, 2}, t).requires_grad_();
  auto              a = m::ConditionalAffinePosteriorFlow(p, "a", z, c),
       b              = m::ConditionalAffinePosteriorFlow(p, "b", a.value, c, true);
  std::vector<double> jac;
  for (int j = 0; j < 4; ++j) {
    auto row =
        n::grad(n::sum(n::select(b.value, 1, j)), {z}, {true, false, false})[0].to_vector<double>();
    jac.insert(jac.end(), row.begin(), row.end());
  }
  const double determinant = n::det(n::from_vector(jac, {4, 4})).item();
  EXPECT_GT(determinant, 0);
  EXPECT_NEAR(std::log(determinant), (a.log_determinant + b.log_determinant).item(), 1e-13);
  auto back1 = m::ConditionalAffinePosteriorFlow(p, "b", b.value, c, true, true),
       back0 = m::ConditionalAffinePosteriorFlow(p, "a", back1.value, c, false, true);
  EXPECT_NEAR(n::max(n::abs(back0.value - z)).item(), 0, 1e-14);
  EXPECT_NEAR(
      (a.log_determinant + b.log_determinant + back1.log_determinant + back0.log_determinant)
          .item(),
      0,
      1e-14);
  EXPECT_GT(n::sum(n::square(n::grad(n::sum(b.value), {c})[0])).item(), 0);
}
TEST(NnFlowVae, CouplingFirstAndSecondDerivativesOfCodeAndContextInBothDirections) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(1, t);
  auto              z = n::tensor({.2, -.4, .6, -.8}, {2, 2}, t).requires_grad_(),
       c              = n::full({2, 1}, .3, t).requires_grad_();
  m::ConditionalAffinePosteriorFlow(p, "f", z, c, false, false, 2, 2);
  for (const auto& name : p.names())
    p.set(name, n::full(p.at(name).shape(), name.ends_with("/b") ? .2 : .03, t));
  for (bool swap : {false, true})
    for (bool inverse : {false, true}) {
      n::TensorFunction f = [&](const auto& inputs) {
        auto o =
            m::ConditionalAffinePosteriorFlow(p, "f", inputs[0], inputs[1], swap, inverse, 2, 2);
        return n::mean(n::square(o.value)) + .3 * n::mean(o.log_determinant);
      };
      EXPECT_TRUE(n::gradcheck(f, {z, c}, {.atol = 1e-8, .rtol = 1e-5}).ok);
      EXPECT_TRUE(n::gradgradcheck(f, {z, c}, {.atol = 1e-8, .rtol = 1e-5}).ok);
    }
}
TEST(NnFlowVae, AllSixtyThreeTinyModelParameterDerivativesIncludeLiveContextAndDensityTerms) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(3001, t);
  auto              x = n::tensor({.1, .4, .8, .2, .6, .3}, {2, 3}, t),
       e              = n::tensor({.3, -.6, -.2, .8}, {2, 2}, t);
  auto value          = [&]() { return m::FlowVariationalAutoencoder(p, x, e, 3, 2, 2, 2).loss; };
  value();
  for (const auto& name : p.names())
    p.set(name, n::full(p.at(name).shape(), name.ends_with("/b") ? .1 : .03, t));
  const auto          loss = value();
  auto                gs   = n::grad(loss, p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gs) {
    auto v = g.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  EXPECT_EQ(p.size(), 18);
  EXPECT_EQ(p.count(), 63);
  EXPECT_EQ(analytic.size(), 63);
  const auto original = p.flatten();
  for (std::size_t j = 0; j < original.size(); ++j) {
    auto v = original;
    v[j] += 1e-6;
    p.unflatten(v);
    const double plus = value().item();
    v[j] -= 2e-6;
    p.unflatten(v);
    const double minus = value().item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[j], 1e-8);
  }
  p.unflatten(original);
  auto              live = x.clone().requires_grad_(), noise = e.clone().requires_grad_();
  n::TensorFunction f = [&](const auto& a) {
    return m::FlowVariationalAutoencoder(p, a[0], a[1], 3, 2, 2, 2).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {live, noise}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {live, noise}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnFlowVae, DefaultCensusOwningFieldsTargetFreeDecodeAndZeroModel) {
  n::ParameterStore p(3001);
  n::NamedTensors   values;
  {
    const auto o = m::FlowVariationalAutoencoder(p, n::full({2, 784}, .3), n::ones({2, 4}));
    values       = o.Named();
  }
  EXPECT_EQ(p.count(), 52672);
  EXPECT_EQ(p.size(), 18);
  EXPECT_EQ(values.size(), 18);
  EXPECT_EQ(values.front().first, "context");
  EXPECT_EQ(values.back().first, "inverse_logdet");
  EXPECT_EQ(
      n::sigmoid(m::GaussianVaeDecode(p, values[5].second)).to_vector<float>(),
      values[13].second.to_vector<float>());
  n::ParameterStore decoder(1);
  m::GaussianVaeDecode(decoder, n::zeros({2, 4}));
  EXPECT_EQ(decoder.count(), 26032);
  EXPECT_EQ(decoder.size(), 4);
  for (const auto& name : p.names())
    p.set(name, n::zeros(p.at(name).shape()));
  const auto o = m::FlowVariationalAutoencoder(p, n::zeros({2, 784}), n::ones({2, 4}));
  EXPECT_EQ(o.logratio.to_vector<float>(), (std::vector<float>{0, 0}));
  EXPECT_EQ(n::max(n::abs(o.logdet)).item(), 0);
  EXPECT_NEAR(o.loss.item(), 784 * std::log(2.), 1e-3);
  EXPECT_EQ(n::min(o.probability).item(), .5);
}
TEST(NnFlowVae, InvalidInputsAndWidthsFailBeforeLeafCreationAndStridesWork) {
  n::ParameterStore p(1);
  auto              x = n::zeros({2, 3}), e = n::zeros({2, 2});
  EXPECT_THROW(m::FlowVariationalAutoencoder(p, x, e, 3, 3, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::FlowVariationalAutoencoder(p, x, e, 3, 2, 2, 0), std::invalid_argument);
  EXPECT_THROW(
      m::FlowVariationalAutoencoder(p, n::full({2, 3}, 1.1), e, 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::FlowVariationalAutoencoder(p, x, n::zeros({1, 2}), 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::ConditionalAffinePosteriorFlow(p, "", e, x, false, false, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::ConditionalAffinePosteriorFlow(p, "f", e, n::zeros({1, 1}), false, false, 2, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::ConditionalAffinePosteriorFlow(
          p, "f", e, n::zeros({2, 1}, n::DType::Float64), false, false, 2, 2),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  const auto strided = n::transpose(n::transpose(n::full({2, 3}, .3)).contiguous());
  EXPECT_FALSE(strided.is_contiguous());
  const auto a = m::FlowVariationalAutoencoder(p, strided, e, 3, 2, 2, 2),
             b = m::FlowVariationalAutoencoder(p, strided.contiguous(), e, 3, 2, 2, 2);
  EXPECT_EQ(a.probability.to_vector<float>(), b.probability.to_vector<float>());
}
