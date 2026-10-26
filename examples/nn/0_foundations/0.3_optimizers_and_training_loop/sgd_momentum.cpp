// CMD: just run-nn sgd_momentum
// SGD, momentum and Nesterov updates.
//
// Port of tflow's sgd_momentum. Compare nn::Optimizer with an independent
// scalar transcription of TensorFlow's fused ApplyGradientDescent and
// ApplyMomentum kernels on a prescribed stream of changing gradients and
// learning rates. The momentum convention is v <- mu*v + g; SGD subtracts
// lr*g, momentum subtracts lr*v, and Nesterov subtracts lr*(g + mu*v). This
// is TensorFlow's reparameterized Nesterov convention; the gradient is
// evaluated at the stored parameter, which represents the look-ahead point.
//
// Then minimize an ill-conditioned quadratic with autograd gradients and fit
// a line from shuffled minibatches. A coupled two-variable check catches
// updating one parameter before the other parameter's gradient is evaluated.
// Optimizer slots are separate from model parameters and can be reset or
// checkpointed through Optimizer::state() / load_state().
//
// Differences from TF: there are no sessions or fused kernels here, so the
// "fused" reference is the kernel formula written out in float arithmetic,
// and "re-running the init op" becomes restoring the parameters and loading
// the optimizer's initial state(). The minibatch order uses the same
// std::mt19937(7) shuffle as the TF version.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using Kind = nn::OptimizerKind;

namespace {

/// Floats held in optimizer slots (excluding the integer step counter).
long long SlotFloats(const nn::Optimizer& opt) {
  long long n = 0;
  for (const auto& [name, value] : opt.state())
    if (name != "step" && (name != "nadam_product" || opt.config().kind == Kind::Nadam)) n += value.numel();
  return n;
}

void KernelAgreement(Kind kind) {
  const std::vector<float> initial{1, -2, 0.5f};
  Tensor                   w = nn::from_vector(initial, {3}).requires_grad_();
  nn::OptimizerConfig      cfg;
  cfg.kind = kind;
  nn::Optimizer      opt({w}, 0.1, cfg);
  const auto         fresh = opt.state();
  std::vector<float> ref = initial, velocity(3, 0.f);
  const float        mu  = static_cast<float>(cfg.momentum);
  float              worst = 0;
  for (int t = 0; t < 20; ++t) {
    const std::vector<float> gradient{t % 3 == 0 ? -0.7f : 0.3f, 0.1f * static_cast<float>(t - 8), 0.f};
    const float              lr = t < 10 ? 0.1f : 0.03f;
    opt.set_learning_rate(lr);
    opt.step({nn::from_vector(gradient, {3})});
    // TF fused kernels: ApplyGradientDescent / ApplyMomentum(use_nesterov).
    for (int j = 0; j < 3; ++j) {
      if (kind == Kind::Sgd) {
        ref[j] -= lr * gradient[j];
      } else {
        velocity[j] = velocity[j] * mu + gradient[j];
        ref[j] -= kind == Kind::Nesterov ? gradient[j] * lr + velocity[j] * mu * lr : lr * velocity[j];
      }
    }
    const auto got = w.to_vector<float>();
    for (int j = 0; j < 3; ++j) {
      worst = std::max(worst, std::abs(got[j] - ref[j]));
      check::Near(got[j], ref[j], 2e-6, "fused kernel agreement");
    }
    check::Require(opt.iterations() == t + 1, "one step increment per update");
  }
  std::printf("  %-10s 20 changing-gradient updates: max fused-kernel error %.2g, slots %lld floats\n",
              nn::optimizer_name(kind).c_str(), worst, SlotFloats(opt));
  // Reset all state: parameters back to their initial values, optimizer to its fresh state.
  {
    nn::NoGradGuard guard;
    w.copy_(nn::from_vector(initial, {3}));
  }
  opt.load_state(fresh);
  check::Near(w.to_vector<float>()[0], 1, 0, "parameter reset");
  check::Require(opt.iterations() == 0 && SlotFloats(opt) == 0, "counter reset");
}

void CoupledGradients() {
  Tensor        x = nn::scalar(1).requires_grad_(), y = nn::scalar(2).requires_grad_();
  nn::Optimizer opt({x, y}, 0.1);
  double        expected_x = 1, expected_y = 2;
  for (int t = 0; t < 10; ++t) {
    const double next_x = expected_x - 0.1 * expected_y;
    expected_y -= 0.1 * expected_x;
    expected_x = next_x;
    opt.minimize(x * y);
    check::Near(x.item(), expected_x, 2e-6, "synchronous x gradient");
    check::Near(y.item(), expected_y, 2e-6, "synchronous y gradient");
  }
  std::cout << "  coupled gradients: all updates use the same old parameter state\n";
}

void Quadratic(Kind kind, int steps) {
  Tensor              w     = nn::tensor({4, 4}).requires_grad_();
  const Tensor        scale = nn::tensor({0.5, 50});
  auto                loss  = [&] { return nn::sum(scale * nn::square(w)); };
  nn::OptimizerConfig cfg;
  cfg.kind = kind;
  nn::Optimizer opt({w}, 0.01, cfg);
  std::printf("  %-10s", nn::optimizer_name(kind).c_str());
  for (int t = 0; t <= steps; ++t) {
    if (t == 0 || t == 10 || t == 50 || t == steps) {
      nn::NoGradGuard guard;
      const float     value = loss().item<float>();
      std::printf(" %11.5g", value);
      check::Require(std::isfinite(value), "quadratic loss must stay finite");
      if (t == steps && steps >= 200) check::Require(value < 0.15f, "quadratic must converge");
    }
    if (t < steps) opt.minimize(loss());
  }
  std::printf("\n");
}

void Minibatches(Kind kind, int epochs) {
  Tensor              w = nn::scalar(0).requires_grad_(), b = nn::scalar(0).requires_grad_();
  nn::OptimizerConfig cfg;
  cfg.kind = kind;
  nn::Optimizer    opt({w, b}, 0.03, cfg);
  std::vector<int> order(64);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(7);
  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(order.begin(), order.end(), rng);
    for (int start = 0; start < 64; start += 8) {
      std::vector<float> xs, ys;
      for (int j = start; j < start + 8; ++j) {
        const float v = -1.f + 2.f * static_cast<float>(order[j]) / 63;
        xs.push_back(v), ys.push_back(3 * v - 0.5f);
      }
      const Tensor x = nn::from_vector(xs), y = nn::from_vector(ys);
      opt.minimize(nn::mean(nn::square(x * w + b - y)));
    }
  }
  check::Require(std::isfinite(w.item()) && std::isfinite(b.item()), "finite learned parameters");
  if (epochs >= 40) {
    check::Near(w.item(), 3, 0.005, "learned slope");
    check::Near(b.item(), -0.5, 0.005, "learned intercept");
  }
  check::Require(opt.iterations() == 8 * epochs, "one update per minibatch");
  std::printf("  %-10s slope %.5f, intercept %.5f (true: 3, -0.5)\n", nn::optimizer_name(kind).c_str(), w.item(),
              b.item());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int quadratic_steps = 200, epochs = 40;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) epochs = quadratic_steps = std::stoi(argument.substr(7));
      if (argument.rfind("steps=", 0) == 0) epochs = quadratic_steps = std::stoi(argument.substr(6));
    }
    check::Require(epochs > 0, "positive training budget required");
    std::cout << "== Update equations vs TensorFlow fused-kernel formulas\n";
    for (auto kind : {Kind::Sgd, Kind::Momentum, Kind::Nesterov}) KernelAgreement(kind);
    CoupledGradients();
    std::cout << "\n== Ill-conditioned quadratic: 0.5*x^2 + 50*y^2, lr=.01\n"
                 "  method           step 0     step 10     step 50    step 200\n";
    for (auto kind : {Kind::Sgd, Kind::Momentum, Kind::Nesterov}) Quadratic(kind, quadratic_steps);
    std::cout << "\n== Shuffled minibatch regression: " << epochs << " epochs, batch 8, seed 7\n";
    for (auto kind : {Kind::Sgd, Kind::Momentum, Kind::Nesterov}) Minibatches(kind, epochs);
    std::cout << "All SGD/momentum/Nesterov checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
