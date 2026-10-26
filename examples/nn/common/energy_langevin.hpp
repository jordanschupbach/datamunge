// Port of tflow's common/nn/energy_langevin.h and common/ml/energy_langevin.h: a
// confining energy E(x) = .5 (||x||^2 + v . tanh(x W + b)) (bounded residual,
// finite partition), stopped-negative contrastive training (Adam .01) and the
// unadjusted Langevin step x' = x - .5 alpha dE/dx + sqrt(alpha) z whose force is
// the actual autograd input gradient. The analytic Hessian rows are diagnostics.
//
// tflow -> datamunge: AddSymbolicGradients(sum E, state) -> nn::grad w.r.t. a leaf
// input; StopGradient(negative) -> nn::stop_gradient. Parameter names hidden/W,
// hidden/b, output/W as tflow (TF's .npz arrays load directly).
#pragma once

#include "common/binary_energy_data.hpp"

namespace ebm {

struct ConfiningEnergyPath {
  Tensor              energy, hidden;
  std::vector<Tensor> hessian_rows;
};

inline void ConfiningEnergyParameters(nn::ParameterStore& p) {
  p.get("hidden/W", {2, 16}, nn::init::normal(.3));
  p.get("hidden/b", {16}, nn::init::zeros());
  p.get("output/W", {16, 1}, nn::init::normal(.1));
}

/// tflow's nn::ConfiningEnergy; `oracle`: the known Gaussian energy with mean (.3, -.2), variance (.49, .25).
inline ConfiningEnergyPath ConfiningEnergy(const nn::ParameterStore* p, const Tensor& x, bool oracle = false) {
  if (oracle) {
    const Tensor delta  = nn::sub(x, nn::from_vector(std::vector<float>{.3f, -.2f}, {1, 2}));
    const Tensor energy = nn::mul(nn::sum(nn::div(nn::square(delta), nn::from_vector(std::vector<float>{.49f, .25f}, {1, 2})), {1}), .5);
    const Tensor zeros  = nn::mul(x, 0.0);
    return {energy, x,
            {nn::add(zeros, nn::from_vector(std::vector<float>{1 / .49f, 0.f}, {1, 2})),
             nn::add(zeros, nn::from_vector(std::vector<float>{0.f, 4.f}, {1, 2}))}};
  }
  const Tensor hidden   = nn::tanh(nn::bias_add(nn::matmul(x, p->at("hidden/W")), p->at("hidden/b")));
  const Tensor residual = nn::reshape(nn::matmul(hidden, p->at("output/W")), {-1});
  const Tensor energy   = nn::mul(nn::add(nn::sum(nn::square(x), {1}), residual), .5);
  const Tensor v        = nn::reshape(p->at("output/W"), {1, 16});
  const Tensor curvature = nn::neg(nn::mul(nn::mul(v, hidden), nn::sub(1.0, nn::square(hidden))));
  const Tensor w         = p->at("hidden/W");
  std::vector<Tensor> rows;
  for (int a = 0; a < 2; ++a) {
    std::vector<Tensor> columns;
    for (int b = 0; b < 2; ++b) {
      const Tensor wa = nn::slice_axis(w, 0, a, a + 1), wb = nn::slice_axis(w, 0, b, b + 1);
      columns.push_back(nn::add(nn::sum(nn::mul(curvature, nn::mul(wa, wb)), {1}, true), a == b ? 1.0 : 0.0));
    }
    rows.push_back(nn::concat(columns, 1));
  }
  return {energy, hidden, rows};
}

/// tflow's ml::EnergyLangevinModel (seed 25003; 64 parameters / 3 tensors, no output bias).
class EnergyLangevinModel {
 public:
  nn::ParameterStore             store{25003};
  bool                           oracle;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit EnergyLangevinModel(bool training = true, bool oracle_energy = false) : oracle(oracle_energy) {
    if (!oracle) {
      ConfiningEnergyParameters(store);
      if (training) optimizer = std::make_unique<nn::Optimizer>(store, .01, Adam());
    }
  }
  ConfiningEnergyPath Path(const Tensor& x) const { return ConfiningEnergy(&store, x, oracle); }
  /// mean E(positive) - mean E(stop(negative)).
  Tensor Loss(const Tensor& positive, const Tensor& negative, Map* fields = nullptr) const {
    const Tensor pe = Path(positive).energy, ne = Path(nn::stop_gradient(negative)).energy;
    const Tensor loss = nn::sub(nn::mean(pe, {0}), nn::mean(ne, {0}));
    if (fields) *fields = Detached({{"positive_energy", pe}, {"negative_energy", ne}, {"contrastive_loss", loss}});
    return loss;
  }
  /// Actual autograd force dE/dx (sum over the batch) and the path at x.
  std::pair<Tensor, ConfiningEnergyPath> Force(const Tensor& x) const {
    const Tensor leaf = x.detach().clone().requires_grad_(true);
    const auto   path = Path(leaf);
    const Tensor f    = nn::grad(nn::sum(path.energy), {leaf})[0].detach();
    return {f, path};
  }
  Map Evaluate(const Tensor& x) const {
    auto [force, path] = Force(x);
    return {{"energy", path.energy.detach()}, {"force", force}, {"hessian_row0", path.hessian_rows[0].detach()},
            {"hessian_row1", path.hessian_rows[1].detach()}};
  }
  /// One unadjusted Langevin step with explicit innovation z.
  Map Step(const Tensor& x, const Tensor& z, float alpha) const {
    check::Require(alpha > 0, "positive ULA step size");
    auto [force, path] = Force(x);
    nn::NoGradGuard guard;
    const float     half = .5f * alpha, root = std::sqrt(alpha);
    const Tensor    next = nn::add(nn::sub(x, nn::mul(force, double(half))), nn::mul(z, double(root)));
    Map             out{{"energy", path.energy.detach()}, {"force", force}, {"next", next}};
    for (int row = 0; row < 2; ++row) {
      const Tensor identity = nn::add(nn::mul(x, 0.0), nn::from_vector(row == 0 ? std::vector<float>{1.f, 0.f} : std::vector<float>{0.f, 1.f}, {1, 2}));
      out["analytic_jacobian_row" + std::to_string(row)] = nn::sub(identity, nn::mul(path.hessian_rows[row].detach(), double(half)));
    }
    return out;
  }
  Map Contrastive(const Tensor& x, const Tensor& negative) const {
    nn::NoGradGuard guard;
    Map             fields;
    Loss(x, negative, &fields);
    return fields;
  }
  Map Gradient(const Tensor& x, const Tensor& negative) const { return ParameterGradients(store, Loss(x, negative)); }
  /// True when the training loss has no gradient path to the negative chains.
  bool NegativeStopped(const Tensor& x, const Tensor& negative) const {
    const Tensor    leaf = negative.detach().clone().requires_grad_(true);
    nn::GradOptions o;
    o.allow_unused = true;
    return !nn::grad(Loss(x, leaf), {leaf}, o)[0].defined();
  }
  void                     Update(const Tensor& x, const Tensor& negative) { optimizer->step(nn::grad(Loss(x, negative), store.parameters())); }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const { return ebm::Snapshot(Arrays()); }
};

}  // namespace ebm
