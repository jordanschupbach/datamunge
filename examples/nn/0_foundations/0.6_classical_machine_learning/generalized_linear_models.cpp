// CMD: just run-nn generalized_linear_models
// Poisson, gamma and negative-binomial log-link generalized linear models.
//
// Port of tflow's generalized_linear_models (likelihoods in common/glm.hpp).
// Fit eta=Xw, mu=exp(eta) to separately generated count or positive
// continuous responses. Gamma shape=3 and negative-binomial size=2 are fixed
// and known; dispersion estimation is outside this example. Check likelihood
// values against scalar probability formulas and autograd derivatives against
// the analytic score. Train all coefficients (float32, nn::Optimizer Adam,
// rate .025, 500 full-batch steps), test first-order stationarity, and compare
// held-out likelihood with a training-mean intercept-only baseline. Data come
// from the same std::mt19937 streams as the TF version. No files are written.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/glm.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using Family = ml::GlmFamily;

namespace {

double Nll(double eta, double y, Family family, double shape) {
  const double mu = std::exp(eta);
  if (family == Family::kPoisson) return mu - y * eta + std::lgamma(y + 1);
  if (family == Family::kGamma)
    return shape * (eta + y / mu) - (shape - 1) * std::log(y) + std::lgamma(shape) - shape * std::log(shape);
  return std::lgamma(shape) + std::lgamma(y + 1) - std::lgamma(y + shape) + (y + shape) * std::log(shape + mu) -
         shape * std::log(shape) - y * eta;
}

void Fit(Family family, const char* name, float shape) {
  // Analytic probes of the likelihood and its derivative in eta.
  const std::vector<float> eta_values{-1, 0, .5f, 1}, y_values{family == Family::kGamma ? .2f : 0.f, 1, 3, 9};
  Tensor       probe_eta  = nn::from_vector(eta_values).requires_grad_();
  const Tensor probe_y    = nn::from_vector(y_values);
  const Tensor probe_loss = ml::GlmNll(probe_eta, probe_y, family, shape);
  const Tensor derivative = nn::grad(nn::sum(probe_loss, {0}), {probe_eta})[0];
  for (int i = 0; i < 4; ++i) {
    const double mean = std::exp(eta_values[i]), response = y_values[i];
    const double score = family == Family::kPoisson ? mean - response
                         : family == Family::kGamma ? shape * (1 - response / mean)
                                                    : shape * (mean - response) / (shape + mean);
    check::Near(probe_loss.at({i}), Nll(eta_values[i], response, family, shape), 3e-6, "GLM scalar likelihood");
    check::Near(derivative.at({i}), score, 2e-6, "GLM analytic derivative");
  }

  std::mt19937                          rng(107 + int(family));
  std::uniform_real_distribution<float> uniform(-1, 1);
  auto sample = [&](int n) {
    std::vector<float> features(n * 3), response(n);
    for (int i = 0; i < n; ++i) {
      const double a = uniform(rng), b = uniform(rng), mean = std::exp(.3 + .55 * a - .4 * b);
      features[i * 3] = 1, features[i * 3 + 1] = a, features[i * 3 + 2] = b;
      if (family == Family::kPoisson) response[i] = std::poisson_distribution<int>(mean)(rng);
      else if (family == Family::kGamma) response[i] = std::gamma_distribution<double>(shape, mean / shape)(rng);
      else {
        const double rate = std::gamma_distribution<double>(shape, mean / shape)(rng);
        response[i]       = std::poisson_distribution<int>(rate)(rng);
      }
    }
    return std::pair<Tensor, Tensor>{nn::from_vector(features, {n, 3}), nn::from_vector(response, {n, 1})};
  };
  const auto [training, response]  = sample(2400);
  const auto [test, test_response] = sample(1000);

  Tensor w         = nn::zeros({3, 1}).requires_grad_();
  auto   objective = [&](const Tensor& x, const Tensor& y) {
    return nn::mean(ml::GlmNll(nn::matmul(x, w), y, family, shape), {0, 1});
  };
  nn::Optimizer optimizer({w}, .025, {nn::OptimizerKind::Adam});
  for (int step = 0; step < 500; ++step) optimizer.step(nn::grad(objective(training, response), {w}));
  const Tensor gradient = nn::grad(objective(training, response), {w})[0];
  const double truth[]{.3, .55, -.4};
  for (int j = 0; j < 3; ++j) {
    check::Near(gradient.at({j, 0}), 0, 2e-5, "GLM stationarity");
    check::Near(w.at({j, 0}), truth[j], .12, "recover generating GLM coefficients");
  }
  std::cout << "  " << name << " coefficients:";
  for (int j = 0; j < 3; ++j) std::cout << ' ' << w.at({j, 0});

  nn::NoGradGuard guard;
  const auto      train_y    = response.to_vector<float>();
  double          train_mean = 0;
  for (int i = 0; i < 2400; ++i) train_mean += train_y[i] / 2400.;
  const double nll = objective(test, test_response).item();
  const auto   mu  = nn::exp(nn::matmul(test, w)).to_vector<float>();
  const auto   ty = test_response.to_vector<float>(), tx = test.to_vector<float>();
  double       null_nll = 0, relative_error = 0;
  for (int i = 0; i < 1000; ++i) {
    null_nll += Nll(std::log(train_mean), ty[i], family, shape) / 1000;
    const double mean = std::exp(.3 + .55 * tx[i * 3 + 1] - .4 * tx[i * 3 + 2]);
    relative_error += std::abs(mu[i] / mean - 1) / 1000;
  }
  check::Require(nll < null_nll - .03 && relative_error < .1, "GLM held-out likelihood and mean prediction");
  std::cout << "; test NLL=" << nll << " (null=" << null_nll << "); relative mean error=" << relative_error << '\n';
}

}  // namespace

int main() {
  try {
    std::cout << "== Generalized linear models\n";
    Fit(Family::kPoisson, "Poisson", 1);
    Fit(Family::kGamma, "gamma(shape=3)", 3);
    Fit(Family::kNegativeBinomial, "negative binomial(size=2)", 2);
    std::cout << "All GLM checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
