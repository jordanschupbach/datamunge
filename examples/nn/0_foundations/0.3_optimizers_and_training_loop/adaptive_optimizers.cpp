// CMD: just run-nn adaptive_optimizers
// Adagrad, RMSProp, Adam, AdamW, Nadam, AMSGrad and Adamax.
//
// Port of tflow's adaptive_optimizers. Compare Adagrad, RMSProp and Adam
// against TensorFlow's fused-kernel formulas (ApplyAdagradV2, ApplyRMSProp,
// ApplyAdam) on a changing gradient stream. Adam uses epsilon after bias
// correction; the fused kernel's epsilon is rescaled to represent that same
// convention. Check first-update bias correction, AdamW's geometric decay
// with zero gradients, AMSGrad's nondecreasing variance envelope, Adamax's
// infinity accumulator and Nadam's single shared momentum-product update.
// Finally train every method on the same anisotropic quadratic with autograd
// gradients and a step-decaying learning rate.
//
// References: Kingma & Ba (https://arxiv.org/abs/1412.6980), Loshchilov &
// Hutter (https://arxiv.org/abs/1711.05101), Reddi et al. (OpenReview ryQu7f-RZ),
// and the scheduled-momentum Nadam update in keras/src/optimizers/nadam.py.
//
// Differences from TF: no sessions or fused kernels, so the reference is each
// kernel's update written out in double arithmetic; slots are read from
// Optimizer::state() and "re-running init" is load_state() of the initial
// state plus restoring the parameters. Learning rates are set with
// set_learning_rate() instead of being fed as scalar tensors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using Kind = nn::OptimizerKind;

namespace {

const std::vector<Kind> kinds{Kind::Adagrad, Kind::RMSProp, Kind::Adam, Kind::AdamW,
                              Kind::Nadam,   Kind::AMSGrad, Kind::Adamax};

long long SlotFloats(const nn::Optimizer& opt) {
  long long n = 0;
  for (const auto& [name, value] : opt.state())
    if (name != "step" && (name != "nadam_product" || opt.config().kind == Kind::Nadam)) n += value.numel();
  return n;
}

Tensor Slot(const nn::Optimizer& opt, const std::string& name) {
  for (const auto& [key, value] : opt.state())
    if (key == name) return value;
  throw std::runtime_error("missing optimizer slot " + name);
}

void FusedKernel(Kind kind) {
  const std::vector<float> initial{1, -2, 0.5f};
  Tensor                   w = nn::from_vector(initial, {3}).requires_grad_();
  nn::OptimizerConfig      cfg;
  cfg.kind    = kind;
  cfg.epsilon = 1e-4;  // large enough to detect incorrect epsilon placement
  nn::Optimizer      opt({w}, 0.03, cfg);
  // The kernels are evaluated in double: a float transcription of ApplyAdam
  // loses ~1e-6 to cancellation in 1 - beta2^t, more than the error under test.
  std::vector<double> ref(initial.begin(), initial.end()), first(3, 0.0), second(3, kind == Kind::Adagrad ? 0.1 : 0.0);
  const double        eps = cfg.epsilon, rho = cfg.rho, b1 = cfg.beta1, b2 = cfg.beta2;
  double              max_error = 0;
  for (int t = 1; t <= 40; ++t) {
    const std::vector<float> g{t % 3 == 0 ? 2.f : -0.2f, t % 4 == 0 ? 0.f : 0.005f, 0.f};
    const float              lr = t < 20 ? 0.03f : 0.01f;
    opt.set_learning_rate(lr);
    opt.step({nn::from_vector(g, {3})});
    const double b1p = std::pow(b1, t), b2p = std::pow(b2, t);
    for (int i = 0; i < 3; ++i) {
      const double gi = g[i];
      if (kind == Kind::Adagrad) {  // ApplyAdagradV2
        second[i] += gi * gi;
        ref[i] -= lr * gi / (std::sqrt(second[i]) + eps);
      } else if (kind == Kind::RMSProp) {  // ApplyRMSProp, momentum 0
        second[i] += (gi * gi - second[i]) * (1 - rho);
        first[i] = lr * gi / std::sqrt(second[i] + eps);
        ref[i] -= first[i];
      } else {  // ApplyAdam with epsilon * sqrt(1 - beta2^t)
        const double lr_t = lr * std::sqrt(1 - b2p) / (1 - b1p), eps_hat = eps * std::sqrt(1 - b2p);
        first[i] += (gi - first[i]) * (1 - b1);
        second[i] += (gi * gi - second[i]) * (1 - b2);
        ref[i] -= lr_t * first[i] / (std::sqrt(second[i]) + eps_hat);
      }
    }
    const auto got = w.to_vector<double>();
    for (int i = 0; i < 3; ++i) {
      max_error = std::max(max_error, std::abs(got[i] - ref[i]));
      check::Near(got[i], ref[i], 3e-6, "adaptive fused kernel agreement");
    }
  }
  std::printf("  %-8s max error over 40 updates: %.3g\n", nn::optimizer_name(kind).c_str(), max_error);
}

void Invariants(Kind kind) {
  Tensor w = nn::tensor({1, -2}).requires_grad_();
  // A separate parameter must not advance shared bias correction a second time.
  Tensor              twin = nn::tensor({1, -2}).requires_grad_();
  nn::OptimizerConfig cfg;
  cfg.kind = kind;
  if (kind == Kind::AdamW) cfg.weight_decay = 0.1;
  nn::Optimizer opt({w, twin}, 0.1, cfg);
  const auto    fresh = opt.state();
  auto          feed  = [&](std::vector<float> g) {
    const Tensor gt = nn::from_vector(g, {2});
    opt.step({gt, gt});
  };
  feed({3, -4});
  const auto wv = w.to_vector<double>(), tv = twin.to_vector<double>();
  for (int i = 0; i < 2; ++i) {
    const double gradient = i == 0 ? 3 : -4, initial = i == 0 ? 1 : -2;
    double       direction;
    if (kind == Kind::Adagrad) {
      direction = gradient / (std::sqrt(0.1 + gradient * gradient) + cfg.epsilon);
    } else if (kind == Kind::RMSProp) {
      direction = gradient / std::sqrt((1 - cfg.rho) * gradient * gradient + cfg.epsilon);
    } else {
      double scale = 1;
      if (kind == Kind::Nadam) {
        const double mu      = cfg.beta1 * (1 - 0.5 * 0.96);
        const double mu_next = cfg.beta1 * (1 - 0.5 * 0.96 * 0.96);
        scale += mu_next * (1 - cfg.beta1) / (1 - mu * mu_next);
      }
      direction = scale * gradient / (std::abs(gradient) + cfg.epsilon) + cfg.weight_decay * initial;
    }
    check::Near(wv[i], initial - 0.1 * direction, 2e-6, "first update closed form");
    check::Near(wv[i], tv[i], 1e-7, "identical parameter groups");
  }
  if (kind == Kind::AMSGrad || kind == Kind::Adamax) {
    const std::string slot     = kind == Kind::AMSGrad ? "vmax/0" : "u/0";
    float             previous = -1;
    for (int t = 0; t < 8; ++t) {
      const float value = Slot(opt, slot).to_vector<float>()[0];
      if (kind == Kind::AMSGrad)
        check::Require(value >= previous, "AMSGrad envelope must never decrease");
      else
        check::Near(value, 3 * std::pow(cfg.beta2, t), 2e-6, "Adamax maximum decays after a pulse");
      previous = value;
      feed({0, 0});
    }
  }
  // Reset before zero gradients.
  {
    nn::NoGradGuard guard;
    w.copy_(nn::tensor({1, -2}));
    twin.copy_(nn::tensor({1, -2}));
  }
  opt.load_state(fresh);
  for (int t = 0; t < 5; ++t) feed({0, 0});
  check::Near(w.to_vector<double>()[0], std::pow(1 - 0.1 * cfg.weight_decay, 5), 2e-6, "decoupled weight decay");
  check::Require(opt.iterations() == 5, "shared optimizer counter");
  std::printf("  %-8s first update, independent slots, reset and zero-gradient checks passed\n",
              nn::optimizer_name(kind).c_str());
}

void Fit(Kind kind) {
  Tensor              w      = nn::tensor({4, -4}).requires_grad_();
  const Tensor        target = nn::tensor({1.5, -0.75}), weight = nn::tensor({1, 30});
  auto                loss   = [&] { return nn::sum(weight * nn::square(w - target)); };
  nn::OptimizerConfig cfg;
  cfg.kind = kind;
  if (kind == Kind::AdamW) cfg.weight_decay = 0.01;
  nn::Optimizer opt({w}, 0.1, cfg);
  for (int t = 0; t < 4000; ++t) {
    opt.set_learning_rate(t < 3200 ? 0.1f : t < 3600 ? 0.01f : 0.001f);
    opt.minimize(loss());
  }
  nn::NoGradGuard guard;
  const auto      v = w.to_vector<float>();
  check::Near(v[0], 1.5, 0.04, "fit first coordinate");
  check::Near(v[1], -0.75, 0.04, "fit second coordinate");
  std::printf("  %-8s w=(%9.5f, %9.5f), loss %.6g, %lld slot floats + one int64 step\n",
              nn::optimizer_name(kind).c_str(), v[0], v[1], loss().item<float>(), SlotFloats(opt));
}

}  // namespace

int main() {
  try {
    std::cout << "== Agreement with TensorFlow fused-kernel formulas\n";
    for (auto kind : {Kind::Adagrad, Kind::RMSProp, Kind::Adam}) FusedKernel(kind);
    std::cout << "\n== Bias correction and state invariants\n";
    for (auto kind : kinds) Invariants(kind);
    std::cout << "\n== Anisotropic quadratic, true w=(1.5,-.75), 4000 updates\n"
                 "   lr=.1 for 3200 steps, .01 for 400, .001 for 400; AdamW decay=.01\n";
    for (auto kind : kinds) Fit(kind);
    std::cout << "All adaptive optimizer checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
