#include <gtest/gtest.h>

#include <cmath>
#include <dm/dm.hpp>
namespace {
namespace n = dm::nn;
namespace m = n::models;
double sigmoid(double x) {
  return 1 / (1 + std::exp(-x));
}
TEST(EnergyComponents, BinaryEnumerationAndBarBank) {
  auto states = m::BinaryStates(8).to_vector<float>();
  for (int b = 0; b < 256; ++b)
    for (int j = 0; j < 8; ++j)
      EXPECT_EQ(states[b * 8 + j], (b >> j) & 1);
  auto         d = m::BinaryBars(512, 21001);
  std::mt19937 rng(21001);
  auto         points     = d.points.to_vector<float>();
  auto         bank       = d.uniforms.to_vector<double>();
  auto         components = d.components.to_vector<int>();
  for (int b = 0; b < 512; ++b) {
    double u = (double(rng()) + .5) / 4294967296.;
    EXPECT_EQ(u, bank[b * 9]);
    int c = int(6 * u);
    EXPECT_EQ(c, components[b]);
    for (int j = 0; j < 8; ++j) {
      u = (double(rng()) + .5) / 4294967296.;
      EXPECT_EQ(u, bank[b * 9 + j + 1]);
      EXPECT_EQ(points[b * 8 + j], float((c < 4 ? j % 4 == c : j / 4 == c - 4) != (u < .05)));
    }
  }
  EXPECT_THROW(m::BinaryStates(13), std::invalid_argument);
  EXPECT_THROW(m::BinaryBars(0, 1), std::invalid_argument);
}
TEST(EnergyComponents, RbmExactJointAndEveryParameterDerivative) {
  n::ParameterStore p(21003);
  m::RbmParameters(p, 3, 2);
  auto x = n::from_vector(std::vector<float>{1, 0, 1, 0, 1, 0}, {2, 3});
  auto W = p.at("W").to_vector<float>(), vb = p.at("visible_bias").to_vector<float>(),
       hb               = p.at("hidden_bias").to_vector<float>();
  double              z = 0;
  std::vector<double> mass(8);
  for (int v = 0; v < 8; ++v)
    for (int h = 0; h < 4; ++h) {
      double score = 0;
      for (int i = 0; i < 3; ++i) {
        score += ((v >> i) & 1) * vb[i];
        for (int j = 0; j < 2; ++j)
          score += ((v >> i) & 1) * ((h >> j) & 1) * W[i * 2 + j];
      }
      for (int j = 0; j < 2; ++j)
        score += ((h >> j) & 1) * hb[j];
      double q = std::exp(score);
      z += q;
      mass[v] += q;
    }
  EXPECT_NEAR(m::RbmLogPartition(p, 3).item(), std::log(z), 2e-7);
  auto loss = [&]() { return n::mean(m::RbmFreeEnergy(p, x)) + m::RbmLogPartition(p, 3); };
  EXPECT_NEAR(loss().item(), -.5 * (std::log(mass[5] / z) + std::log(mass[2] / z)), 4e-7);
  auto grads = n::grad(loss(), p.parameters());
  auto names = p.names();
  for (size_t k = 0; k < names.size(); ++k) {
    auto original = p.at(names[k]).detach().clone();
    auto values   = original.to_vector<float>();
    auto g        = grads[k].to_vector<float>();
    for (size_t i = 0; i < values.size(); ++i) {
      auto plus = values, minus = values;
      plus[i] += .001f;
      minus[i] -= .001f;
      p.set(names[k], n::from_vector(plus, original.shape()));
      double a = loss().item();
      p.set(names[k], n::from_vector(minus, original.shape()));
      double b = loss().item();
      EXPECT_NEAR(g[i], (a - b) / (double(plus[i]) - minus[i]), 3e-4);
    }
    p.set(names[k], original);
  }
}
TEST(EnergyComponents, DbmExactAllJointStates) {
  n::ParameterStore p(23003);
  m::DeepBoltzmannParameters(p);
  auto                W1 = p.at("W1").to_vector<float>(), W2 = p.at("W2").to_vector<float>();
  std::vector<double> mass(256);
  double              total = 0;
  for (int v = 0; v < 256; ++v)
    for (int a = 0; a < 16; ++a)
      for (int b = 0; b < 8; ++b) {
        double score = 0;
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 8; ++i)
            score += ((v >> i) & 1) * ((a >> j) & 1) * W1[i * 4 + j];
          for (int k = 0; k < 3; ++k)
            score += ((a >> j) & 1) * ((b >> k) & 1) * W2[j * 3 + k];
        }
        double q = std::exp(score);
        mass[v] += q;
        total += q;
      }
  auto path   = m::DbmUnnormalized(p, m::BinaryStates(8));
  auto actual = path.log_weight.to_vector<float>();
  for (int v = 0; v < 256; ++v)
    EXPECT_NEAR(actual[v], std::log(mass[v]), 7e-7);
  EXPECT_NEAR(m::LogSumExp(path.log_weight, 0).item(), std::log(total), 1e-6);
  auto mf    = m::DbmMeanField(p, m::BinaryStates(8));
  auto bound = mf.free_energy.to_vector<float>();
  for (int v = 0; v < 256; ++v)
    EXPECT_GE(bound[v] + std::log(mass[v]), -2e-6);
  EXPECT_THROW(m::DbmMeanField(p, n::zeros({1, 8}), 0), std::invalid_argument);
}
TEST(EnergyComponents, DeepBeliefNormalizationAndRecognitionIsolation) {
  n::ParameterStore lower(21003), top(22003);
  m::RbmParameters(lower, 8, 4);
  m::RbmParameters(top, 4, 3);
  auto x    = m::BinaryStates(8);
  auto path = m::DeepBelief(lower, top, x);
  EXPECT_NEAR(n::sum(n::exp(path.log_probability)).item(), 1, 5e-7);
  n::GradOptions options;
  options.allow_unused = true;
  EXPECT_FALSE(n::grad(path.nll, {lower.at("hidden_bias")}, options)[0].defined());
  auto before = path.log_probability.to_vector<float>();
  lower.set("hidden_bias", n::full({4}, 100));
  auto after = m::DeepBelief(lower, top, x).log_probability.to_vector<float>();
  EXPECT_EQ(before, after);
}
TEST(EnergyComponents, ConfiningForceAndHessianAgainstClosedForm) {
  n::ParameterStore p(25003);
  m::ConfiningEnergyParameters(p);
  auto x     = n::from_vector(std::vector<float>{.2f, -.4f, 1.f, 2.f}, {2, 2}).requires_grad_();
  auto path  = m::ConfiningEnergy(&p, x);
  auto force = n::grad(n::sum(path.energy), {x})[0].to_vector<float>();
  auto W = p.at("hidden/W").to_vector<float>(), v = p.at("output/W").to_vector<float>(),
       values = x.to_vector<float>();
  auto E      = path.energy.to_vector<float>();
  for (int b = 0; b < 2; ++b) {
    double e = .5 * (values[b * 2] * values[b * 2] + values[b * 2 + 1] * values[b * 2 + 1]);
    for (int j = 0; j < 16; ++j)
      e += .5 * v[j] * std::tanh(values[b * 2] * W[j] + values[b * 2 + 1] * W[16 + j]);
    EXPECT_NEAR(e, E[b], 3e-7);
    for (int a = 0; a < 2; ++a) {
      double g = values[b * 2 + a];
      for (int j = 0; j < 16; ++j) {
        double h = std::tanh(values[b * 2] * W[j] + values[b * 2 + 1] * W[16 + j]);
        g += .5 * v[j] * (1 - h * h) * W[a * 16 + j];
      }
      EXPECT_NEAR(g, force[b * 2 + a], 3e-7);
      auto row = path.hessian_rows[a].to_vector<float>();
      for (int c = 0; c < 2; ++c) {
        double hess = a == c ? 1 : 0;
        for (int j = 0; j < 16; ++j) {
          double h = std::tanh(values[b * 2] * W[j] + values[b * 2 + 1] * W[16 + j]);
          hess -= v[j] * h * (1 - h * h) * W[a * 16 + j] * W[c * 16 + j];
        }
        EXPECT_NEAR(hess, row[b * 2 + c], 3e-7);
      }
    }
  }
}
TEST(EnergyComponents, HopfieldEnergyDerivativeAndDescent) {
  auto memory = n::from_vector(std::vector<float>{1, 0, 0, 1, -1, 0}, {3, 2});
  auto x      = n::from_vector(std::vector<float>{.3f, .8f}, {1, 2}).requires_grad_();
  auto path   = m::ModernHopfield(x, memory, 8);
  auto g      = n::grad(n::sum(path.energy), {x})[0];
  EXPECT_LT(n::max(n::abs(g - (x - path.next))).item(), 3e-7);
  double previous = path.energy.item();
  for (int i = 0; i < 8; ++i) {
    x    = path.next.detach();
    path = m::ModernHopfield(x, memory, 8);
    EXPECT_LE(path.energy.item(), previous + 2e-7);
    previous = path.energy.item();
  }
  EXPECT_THROW(m::ModernHopfield(x, memory, 0), std::invalid_argument);
}
} // namespace
