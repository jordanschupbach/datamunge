#pragma once
#include "nn_interpretation_check_support.hpp"
namespace {
void EstimatorChecks() {
  const Tensor c        = sal::FromF({1.f, -2.f, 0.f}, {3});
  auto         callback = [](std::function<Tensor(const Tensor&)> f) {
    return [f](const Tensor& points, const Tensor&) {
      Tensor x = points.detach().clone();
      x.requires_grad_(true);
      return nn::grad(f(x), {x})[0].detach().contiguous();
    };
  };
  auto linear    = [&](const Tensor& x) { return nn::sum(nn::mul(x, c)); };
  auto quadratic = [&](const Tensor& x) { return nn::sum(nn::mul(nn::mul(x, x), nn::mul(c, .5))); };
  const Tensor input   = sal::FromF({.1f, .3f, .5f, -.2f, .7f, 1.2f}, {2, 3}),
               targets = sal::FromI({0, 0});
  const auto noise     = nn::models::SmoothGradNoise(input.shape(), 32, 1171),
             same      = nn::models::SmoothGradNoise(input.shape(), 32, 1171);
  check::Require(
      sal::Bytes(noise) == sal::Bytes(same) &&
          noise.to_vector<double>() == same.to_vector<double>(),
      "exactseeded noise replay");
  const auto constant =
                 nn::models::SmoothGrad(input, targets, noise, .5f, false, 7, callback(linear)),
             q = nn::models::SmoothGrad(input, targets, noise, .5f, false, 1, callback(quadratic)),
             qbatch =
                 nn::models::SmoothGrad(input, targets, noise, .5f, false, 11, callback(quadratic)),
             zero = nn::models::SmoothGrad(input, targets, noise, 0, false, 5, callback(quadratic));
  const auto cm = sal::F(constant.mean), cv = sal::F(constant.sample_variance), qm = sal::F(q.mean),
             qn = sal::F(q.noisy_input), qb = sal::F(qbatch.mean), zm = sal::F(zero.mean),
             qa = sal::F(q.mean_absolute), x = sal::F(input);
  double difference = 0;
  for (int i = 0; i < 6; ++i) {
    const float coefficient = std::array<float, 3>{1, -2, 0}[i % 3];
    check::Near(cm[i], coefficient, 0, "linear SmoothGrad invariant to noise");
    check::Near(cv[i], 0, 0, "constant-gradient variancezero");
    double expected = 0;
    for (int k = 0; k < 32; ++k)
      expected += coefficient * qn[k * 6 + i] / 32.;
    check::Near(
        qm[i], expected, 2e-7, "quadratic exact FINITE-noise-bank mean,not assuming noise mean0");
    check::Near(qb[i], qm[i], 0, "noise-point batching invariance");
    check::Near(zm[i], coefficient * x[i], 0, "sigma0recovers ordinary derivative");
    difference += qa[i] - std::abs(qm[i]);
  }
  check::Require(difference > 1e-3, "meanabs differs from absmean;no silent magnitude variant");
  std::cout << "SmoothGrad linear/quadratic finitebank/sigma0/seed/batching/dummy/variance tests "
               "passed;meanabs-vsabsmean difference="
            << difference << "\n";
}

} // namespace
