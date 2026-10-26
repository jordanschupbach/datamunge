// Port of tflow's common/nn/score_sde.h and common/ml/score_sde.h: the VE reverse
// SDE (Euler-Maruyama) and RK4 probability-flow ODE driven by a frozen NCSN
// epsilon MLP (examples/nn/common/noise_conditional_score.hpp) with continuous
// conditioning, sigma(t) = .05 20^t, g^2 = 2 log 20 sigma^2, network level
// 15 (1 - t); or the Gaussian oracle score (.3 - x) / (.49 + sigma^2).
//
// tflow -> datamunge: graph/session -> eager; the three first-pixel input
// gradients (score, EM next, ODE next) -> nn::grad; SaveV2 restore -> npz.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "common/noise_conditional_score.hpp"

namespace scoresde {

namespace nn = datamunge::nn;
using nn::Tensor;
using Map = replay::Map;

struct FieldPath {
  Tensor score, epsilon, sigma, g_squared, reverse_drift, probability_drift;
};

/// tflow's ScoreSdeField; `p` null = Gaussian oracle.
inline FieldPath Field(const nn::ParameterStore* p, const Tensor& state, const Tensor& physical_time) {
  const Tensor time  = nn::reshape(physical_time, {-1, 1, 1, 1});
  const Tensor sigma = nn::mul(nn::exp(nn::mul(time, double(float(std::log(20.))))), .05);
  const Tensor g2    = nn::mul(nn::square(sigma), double(float(2 * std::log(20.))));
  Tensor       score, epsilon;
  if (!p) {
    score   = nn::div(nn::sub(nn::scalar_like(.3, state), state), nn::add(nn::square(sigma), .49));
    epsilon = nn::neg(nn::mul(sigma, score));
  } else {
    const Tensor level = nn::mul(1.0 - physical_time, 15.0);
    epsilon            = ncsn::Predictor(*p, state, level, 15.f).epsilon;
    score              = nn::neg(nn::div(epsilon, sigma));
  }
  const Tensor reverse = nn::mul(g2, score);
  return {score, epsilon, sigma, g2, reverse, nn::mul(reverse, .5)};
}

struct StepPath {
  FieldPath           field;
  Tensor              em_mean, em_value, ode_value;
  std::vector<Tensor> ode_stage_states, ode_stage_drifts;
};

/// tflow's ScoreSdeStep: one reverse EM step and one RK4 probability-flow step (time decreasing).
inline StepPath Step(const nn::ParameterStore* p, const Tensor& state, const Tensor& time, const Tensor& h, const Tensor& innovation) {
  const auto   a    = Field(p, state, time);
  const Tensor mean = nn::add(state, nn::mul(h, a.reverse_drift));
  const Tensor em   = nn::add(mean, nn::mul(nn::sqrt(nn::mul(h, a.g_squared)), innovation));
  const Tensor half = nn::mul(h, .5);
  const Tensor xb   = nn::add(state, nn::mul(half, a.probability_drift));
  const auto   b    = Field(p, xb, nn::sub(time, half));
  const Tensor xc   = nn::add(state, nn::mul(half, b.probability_drift));
  const auto   c    = Field(p, xc, nn::sub(time, half));
  const Tensor xd   = nn::add(state, nn::mul(h, c.probability_drift));
  const auto   d    = Field(p, xd, nn::sub(time, h));
  const Tensor ode  = nn::add(state, nn::mul(nn::div(h, 6.0), nn::add(nn::add(a.probability_drift, d.probability_drift),
                                                                         nn::mul(nn::add(b.probability_drift, c.probability_drift), 2.0))));
  return {a, mean, em, ode, {state, xb, xc, xd}, {a.probability_drift, b.probability_drift, c.probability_drift, d.probability_drift}};
}

/// tflow's ml::ScoreSdeModel: frozen NCSN weights (no training) or the oracle.
class ScoreSdeModel {
 public:
  nn::ParameterStore store{12003};
  bool               oracle;

  explicit ScoreSdeModel(bool oracle_ = false) : oracle(oracle_) {
    if (!oracle) diffusion::CreateMlp(store, 72, 64);
  }
  const nn::ParameterStore* params() const { return oracle ? nullptr : &store; }

  /// mode 0: EM fields only, 1: ODE fields only, 2: all.
  Map Evaluate(const Tensor& x, float physical_time, float h, const Tensor& z, int mode = 2) const {
    check::Require(physical_time >= 0 && physical_time <= 1 && h >= 0 && h <= physical_time + 1e-7 && x.shape() == z.shape(),
                   "valid reverse score SDE time, step and noise");
    const Tensor t  = nn::full({x.size(0)}, physical_time);
    const Tensor dt = nn::full({}, h);
    Tensor       s  = x.detach().clone();
    s.requires_grad_();
    const auto path = Step(params(), s, t, dt, z);
    Map        out;
    const std::vector<std::pair<std::string, Tensor>> outputs{
        {"score_input_gradient", path.field.score}, {"em_input_gradient", path.em_value}, {"ode_input_gradient", path.ode_value}};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if ((mode == 0 && i == 2) || (mode == 1 && i == 1)) continue;
      nn::GradOptions keep;
      keep.retain_graph = true;
      const Tensor first = nn::slice_axis(nn::reshape(outputs[i].second, {x.size(0), 64}), 1, 0, 1);
      out[outputs[i].first] = nn::grad(nn::sum(first), {s}, keep)[0];
    }
    out["score"]             = path.field.score.detach();
    out["epsilon"]           = path.field.epsilon.detach();
    out["sigma"]             = path.field.sigma.detach();
    out["g_squared"]         = path.field.g_squared.detach();
    out["reverse_drift"]     = path.field.reverse_drift.detach();
    out["probability_drift"] = path.field.probability_drift.detach();
    if (mode != 1) {
      out["em_mean"] = path.em_mean.detach();
      out["em_next"] = path.em_value.detach();
    }
    if (mode != 0) {
      out["ode_next"] = path.ode_value.detach();
      for (int i = 0; i < 4; ++i) {
        out["ode_stage_state" + std::to_string(i)] = path.ode_stage_states[i].detach();
        out["ode_stage_drift" + std::to_string(i)] = path.ode_stage_drifts[i].detach();
      }
    }
    return out;
  }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : Arrays()) out.push_back(diffusion::Bits(v));
    return out;
  }
};

}  // namespace scoresde
