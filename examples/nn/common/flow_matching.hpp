// Port of tflow's common/nn/flow_matching.h and common/ml/flow_matching.h:
// independent straight endpoint coupling (conditional flow matching / first
// rectified flow) on 2-D points. Velocity MLP [x, t] 3 -> 32 SiLU -> 32 SiLU -> 2
// (1250 parameters, tflow names hidden1/W ...), or the analytic velocity of the
// 8-Gaussian ring mixture (radius 2, std .15) as an "oracle" control. Euler or
// RK4 deterministic flow steps with the full per-point 2x2 step Jacobian.
//
// tflow -> datamunge: graph/session -> eager; AddSymbolicGradients -> nn::grad;
// Adam train_op -> nn::Optimizer (Adam .003); SaveV2 "final" -> final.npz.
#pragma once

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/ddpm.hpp"

namespace flowm {

namespace nn = datamunge::nn;
using nn::Tensor;
using Map = replay::Map;

/// tflow's FlowMatchingVelocity: learned MLP or the analytic mixture velocity.
inline Tensor Velocity(const nn::ParameterStore* p, const Tensor& x, const Tensor& time) {
  const Tensor t = nn::reshape(time, {-1, 1});
  if (!p) {
    const Tensor one_minus = 1.0 - t;
    const Tensor variance  = nn::add(nn::square(one_minus), nn::mul(nn::square(t), .0225));
    const Tensor gain      = nn::div(nn::sub(nn::mul(t, .0225), one_minus), variance);
    std::vector<Tensor> logits, vs;
    for (int k = 0; k < 8; ++k) {
      const Tensor mu    = nn::from_vector(std::vector<float>{float(2 * std::cos(2 * M_PI * k / 8)), float(2 * std::sin(2 * M_PI * k / 8))}, {1, 2});
      const Tensor delta = nn::sub(x, nn::mul(t, mu));
      logits.push_back(nn::div(nn::mul(nn::sum(nn::square(delta), {1}, true), -.5), variance));
      vs.push_back(nn::expand_dims(nn::add(mu, nn::mul(gain, delta)), 1));
    }
    const Tensor weights = nn::expand_dims(nn::softmax(nn::concat(logits, 1), 1), 2);
    return nn::sum(nn::mul(weights, nn::concat(vs, 1)), {1});
  }
  const Tensor input = nn::concat({x, t}, 1);
  const Tensor h1 = diffusion::Dense(*p, "hidden1", input, true), h2 = diffusion::Dense(*p, "hidden2", h1, true);
  return diffusion::Dense(*p, "output", h2, false);
}

struct StepPath {
  Tensor              value;
  std::vector<Tensor> states, velocities;
};

/// tflow's FlowMatchingStep: x + h v (Euler) or classical RK4.
inline StepPath Step(const nn::ParameterStore* p, const Tensor& x, const Tensor& time, const Tensor& h, bool rk4) {
  const Tensor a = Velocity(p, x, time);
  if (!rk4) return {nn::add(x, nn::mul(h, a)), {x}, {a}};
  const Tensor half = nn::mul(h, .5);
  const Tensor xb = nn::add(x, nn::mul(half, a));
  const Tensor b  = Velocity(p, xb, nn::add(time, half));
  const Tensor xc = nn::add(x, nn::mul(half, b));
  const Tensor c  = Velocity(p, xc, nn::add(time, half));
  const Tensor xd = nn::add(x, nn::mul(h, c));
  const Tensor d  = Velocity(p, xd, nn::add(time, h));
  const Tensor value = nn::add(x, nn::mul(nn::div(h, 6.0), nn::add(nn::add(a, d), nn::mul(nn::add(b, c), 2.0))));
  return {value, {x, xb, xc, xd}, {a, b, c, d}};
}

/// tflow's ml::FlowMatchingModel (seed 20003; `oracle`: no weights, analytic velocity).
class FlowMatchingModel {
 public:
  nn::ParameterStore             store{20003};
  std::unique_ptr<nn::Optimizer> optimizer;
  bool                           oracle;

  explicit FlowMatchingModel(bool training = true, bool oracle_ = false) : oracle(oracle_) {
    if (oracle) return;
    diffusion::CreateMlp(store, 3, 2);
    if (training) optimizer = std::make_unique<nn::Optimizer>(store, .003, diffusion::Adam());
  }
  const nn::ParameterStore* params() const { return oracle ? nullptr : &store; }

  /// x: endpoint (data), z: base (noise), t: [n] time in [0, 1].
  Tensor Loss(const Tensor& x, const Tensor& z, const Tensor& t, Map* fields = nullptr) const {
    const Tensor tt           = nn::reshape(t, {-1, 1});
    const Tensor interpolated = nn::add(nn::mul(1.0 - tt, z), nn::mul(tt, x));
    const Tensor target       = nn::sub(x, z);
    const Tensor velocity     = Velocity(params(), interpolated, t);
    const Tensor loss         = nn::mean(nn::square(nn::sub(velocity, target)));
    if (fields)
      *fields = diffusion::Detached({{"interpolated", interpolated}, {"target", target}, {"velocity", velocity}, {"loss", loss}});
    return loss;
  }
  Map Evaluate(const Tensor& x, const Tensor& z, const Tensor& t) const {
    nn::NoGradGuard guard;
    Map             fields;
    Loss(x, z, t, &fields);
    return fields;
  }
  Map  Gradient(const Tensor& x, const Tensor& z, const Tensor& t) { return diffusion::ParameterGradients(store, Loss(x, z, t)); }
  void Update(const Tensor& x, const Tensor& z, const Tensor& t) { optimizer->step(nn::grad(Loss(x, z, t), store.parameters())); }

  /// One deterministic flow step from state/time only, with stages and the step Jacobian rows.
  Map StepFields(const Tensor& x, const Tensor& t, float dt, bool rk4) const {
    Tensor s = x.detach().clone();
    s.requires_grad_();
    const Tensor h    = nn::full({}, dt);
    const auto   path = Step(params(), s, t, h, rk4);
    Map          out;
    for (int row = 0; row < 2; ++row) {
      nn::GradOptions keep;
      keep.retain_graph = row == 0;
      out["jacobian_row" + std::to_string(row)] = nn::grad(nn::sum(nn::slice_axis(path.value, 1, row, row + 1)), {s}, keep)[0];
    }
    out["next"] = path.value.detach();
    for (std::size_t k = 0; k < path.states.size(); ++k) {
      out["state" + std::to_string(k)]    = path.states[k].detach();
      out["velocity" + std::to_string(k)] = path.velocities[k].detach();
    }
    return out;
  }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : Arrays()) out.push_back(bpix::Bits(v));
    return out;
  }
};

}  // namespace flowm
