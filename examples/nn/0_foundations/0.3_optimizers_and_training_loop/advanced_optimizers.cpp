// CMD: just run-nn advanced_optimizers
// LAMB, LARS, Lion and Adafactor.
//
// Port of tflow's advanced_optimizers. Check layerwise relative update norms
// for LAMB and LARS, Lion's sign update using the OLD momentum, and
// Adafactor's factored second moments on a rank-one squared gradient.
// Adafactor stores row/column means for matrices, flattens leading axes for
// higher-rank tensors, and keeps full moments for vectors/scalars. Its update
// clipping, relative step cap and parameter RMS scaling are configurable; the
// default follows the paper's no-momentum rule.
//
// The zero-gradient cases exercise scalar, vector and tensor fallbacks.
// A matrix-fitting example then trains every method with autograd gradients
// and prints the persistent slot memory. This demonstrates update mechanics,
// not the large-batch performance of the original language/vision models.
//
// References: LAMB https://arxiv.org/abs/1904.00962;
// LARS https://arxiv.org/abs/1708.03888;
// Lion https://arxiv.org/abs/2302.06675;
// Adafactor https://arxiv.org/abs/1804.04235 (Algorithms 4-6).
//
// Differences from TF: no sessions; gradients are fed straight to
// Optimizer::step and learning rates set with set_learning_rate(). nn creates
// optimizer slots lazily on the first update (TF at graph construction), so
// the slot-memory counts are read after one update.

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

const std::vector<Kind> kinds{Kind::Lamb, Kind::Lars, Kind::Lion, Kind::Adafactor};

/// Persistent slot floats (tflow's opt.slots.NumParams(); excludes the step counter).
long long SlotFloats(const nn::Optimizer& opt) {
  long long n = 0;
  for (const auto& [name, value] : opt.state())
    if (name != "step" && name != "nadam_product") n += value.numel();
  return n;
}

Tensor Slot(const nn::Optimizer& opt, const std::string& name) {
  for (const auto& [key, value] : opt.state())
    if (key == name) return value;
  throw std::runtime_error("missing optimizer slot " + name);
}

void TrustRatio(Kind kind) {
  Tensor              w = nn::tensor({3, 4}).requires_grad_();
  nn::OptimizerConfig cfg;
  cfg.kind              = kind;
  cfg.momentum          = 0;
  cfg.trust_coefficient = 0.1;
  nn::Optimizer opt({w}, 0.1, cfg);
  opt.step({nn::tensor({1, -2})});
  const auto   out      = w.to_vector<float>();
  const double relative = std::hypot(out[0] - 3, out[1] - 4) / 5;
  check::Near(relative, kind == Kind::Lamb ? 0.1 : 0.01, 1e-7, "layer-relative update norm");
  std::printf("  %-9s ||delta w||/||w|| = %.6f (lr=.1, LARS coefficient=.1)\n", nn::optimizer_name(kind).c_str(),
              relative);
}

void LionSign() {
  Tensor              w = nn::scalar(1).requires_grad_();
  nn::OptimizerConfig cfg;
  cfg.kind  = Kind::Lion;
  cfg.beta2 = 0.99;
  nn::Optimizer opt({w}, 0.1, cfg);
  for (float gradient : {10.f, -0.85f}) opt.step({nn::scalar(gradient)});
  check::Near(w.item(), 0.8, 1e-6, "Lion must use old momentum to choose sign");
  check::Near(Slot(opt, "m/0").item(), 0.0905, 1e-6, "Lion uses beta2 for stored momentum");
  std::cout << "  Lion: gradients 10 then -.85 still take two negative sign steps; w=.8\n";
}

void Factorization() {
  Tensor              w = nn::ones({2, 3}).requires_grad_();
  nn::OptimizerConfig cfg;
  cfg.kind            = Kind::Adafactor;
  cfg.scale_parameter = cfg.relative_step = false;
  cfg.update_clip     = 0.5;
  nn::Optimizer opt({w}, 0.1, cfg);
  opt.step({nn::tensor({1, 2, 3, 2, 4, 6}, {2, 3})});
  const auto wv = w.to_vector<double>(), row = Slot(opt, "row/0").to_vector<double>(),
             col = Slot(opt, "col/0").to_vector<double>();
  for (int i = 0; i < 6; ++i) check::Near(wv[i], 0.95, 1e-6, "exact rank-one preconditioner and update clipping");
  check::Near(row[0], 14.0 / 3, 1e-6, "row moment");
  check::Near(row[1], 56.0 / 3, 2e-6, "row moment");
  for (int j = 0; j < 3; ++j) check::Near(col[j], 2.5 * (j + 1) * (j + 1), 1e-6, "column moment");
  check::Require(SlotFloats(opt) == 5, "matrix slots must be rows + columns");
  std::cout << "  Adafactor: exact rank-one reconstruction, clipped RMS update .5, 5 slots for 2x3 matrix\n";
}

void ZerosAndMemory() {
  for (auto kind : kinds) {
    for (const nn::Shape& shape : {nn::Shape{}, nn::Shape{3}, nn::Shape{2, 3, 4}}) {
      Tensor              w = nn::zeros(shape).requires_grad_();
      nn::OptimizerConfig cfg;
      cfg.kind = kind;
      nn::Optimizer opt({w}, 0.01, cfg);
      for (int t = 0; t < 3; ++t) opt.step({nn::zeros_like(w)});
      for (double v : w.to_vector<double>()) check::Near(v, 0, 0, "zero gradients must stay finite");
    }
  }
  Tensor              w = nn::zeros({256, 128}).requires_grad_();
  nn::OptimizerConfig cfg;
  cfg.kind = Kind::Adafactor;
  nn::Optimizer opt({w}, 0.01, cfg);
  opt.step({nn::ones_like(w)});  // nn allocates slots on the first update
  check::Require(SlotFloats(opt) == 256 + 128, "Adafactor sublinear matrix storage");
  std::cout << "  zero-gradient scalar/vector/tensor checks passed for all four methods\n"
               "  256x128 weights: Adafactor 384 moment floats; Adam would need 65536\n";
}

void Fit(Kind kind, int steps) {
  Tensor             w = nn::ones({3, 4}).requires_grad_();
  std::vector<float> tv(12);
  for (int i = 0; i < 12; ++i) tv[i] = 0.3f + 0.5f * std::sin(static_cast<float>(i));
  const Tensor        target  = nn::from_vector(tv, {3, 4});
  auto                loss_of = [&] { return nn::mean(nn::square(nn::sub(w, target))); };
  nn::OptimizerConfig cfg;
  cfg.kind              = kind;
  cfg.trust_coefficient = 0.1;
  if (kind == Kind::Lion) cfg.beta2 = 0.99;
  const float   initial_lr = kind == Kind::Lion ? 0.01f : kind == Kind::Lars ? 0.1f : 0.03f;
  nn::Optimizer opt({w}, initial_lr, cfg);
  const double initial_loss = loss_of().item();
  for (int t = 0; t < steps; ++t) {
    opt.set_learning_rate(initial_lr * (1 - t / 1500.f));
    opt.minimize(loss_of());
  }
  const float final_loss = static_cast<float>(loss_of().item());
  check::Require(std::isfinite(final_loss) && final_loss < initial_loss, "matrix fit must reduce loss");
  if (steps >= 1500) check::Require(final_loss < 1e-5, "matrix fit must converge");
  std::printf("  %-9s initial lr %.3f, final MSE %.5g, persistent slots %lld floats\n",
              nn::optimizer_name(kind).c_str(), initial_lr, final_loss, SlotFloats(opt));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int steps = 1500;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) steps = std::stoi(argument.substr(7));
      if (argument.rfind("steps=", 0) == 0) steps = std::stoi(argument.substr(6));
    }
    check::Require(steps > 0 && steps <= 1500, "training budget must be in [1,1500]");
    nn::set_num_threads(2);
    std::cout << "== Trust ratios, signs and factorization\n";
    TrustRatio(Kind::Lamb);
    TrustRatio(Kind::Lars);
    LionSign();
    Factorization();
    ZerosAndMemory();
    std::cout << "\n== Fit a 3x4 matrix, " << steps << " updates with linear learning-rate decay\n";
    for (auto kind : kinds) Fit(kind, steps);
    std::cout << "All advanced optimizer checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
