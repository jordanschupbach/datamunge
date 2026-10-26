// Continuous normalizing flow: the augmented ODE dx/dt = f(x, t), d logdet / dt =
// tr(df/dx) integrated by fixed-step RK4 with FFJORD-style paired Hutchinson
// traces. Port of tflow's common/nn/cnf_density.h and common/ml/cnf_density.h
// (harness in flow_density.hpp).
//
// Field: f(x, t) = .3 (tanh([x, t] W1 + b1) W2 + b2), 3 -> 8 -> 2. Its divergence is
// analytic: with a, b the x rows of W1 and c, d the columns of W2, tr = .3 sum
// (1 - h^2)(a c + b d). The paired probes (1, 1) and (1, -1) give (a + b)(c + d) and
// (a - b)(c - d); their mean (trace_mode 3, used for training) equals the exact 2-D
// trace. Eight RK4 steps over t in [0, 1] (reverse integration for the inverse),
// then a global diagonal scale. Training differentiates through the unrolled RK4
// steps with ordinary reverse mode (as tflow: no constant-memory adjoint).
// Store seed 10003.
#pragma once

#include "common/real_nvp_density.hpp"

namespace flowd {

struct CnfFieldPath {
  Tensor velocity, divergence, exact_trace, hutchinson_plus, hutchinson_minus;
};

inline CnfFieldPath CnfField(nn::ParameterStore& p, const Tensor& x, float time, int trace_mode = 3) {
  const Tensor t       = Col(x, 0) * 0.0 + double(time);
  const Tensor encoded = nn::concat({x, t}, 1);
  const Tensor hidden  = Dense(p, "field/hidden", encoded, 3, 8, .3, .05, true);
  const Tensor velocity = 0.3 * Dense(p, "field/output", hidden, 8, 2, .1, 0);
  const Tensor wi = p.at("field/hidden/W"), wo = p.at("field/output/W");
  const Tensor a = nn::slice(wi, {0, 0}, {1, 8}), b = nn::slice(wi, {1, 0}, {1, 8});
  const Tensor c = nn::transpose(nn::slice(wo, {0, 0}, {8, 1}), {1, 0}), d = nn::transpose(nn::slice(wo, {0, 1}, {8, 1}), {1, 0});
  const Tensor slope  = 1.0 - nn::square(hidden);
  const auto   reduce = [&](const Tensor& product) { return 0.3 * nn::sum(slope * product, {1}); };
  const Tensor exact = reduce(a * c + b * d), plus = reduce((a + b) * (c + d)), minus = reduce((a - b) * (c - d));
  const Tensor trace = trace_mode == 0 ? exact : trace_mode == 1 ? plus : trace_mode == 2 ? minus : 0.5 * (plus + minus);
  return {velocity, trace, exact, plus, minus};
}

/// Fixed-step RK4 of the augmented ODE; records the state, cumulative logdet, endpoint
/// divergence and velocity after each quarter of the trajectory (tflow's stage fields).
inline FlowPath CnfIntegrate(nn::ParameterStore& p, const Tensor& input, bool inverse = false, int steps = 8, int trace_mode = 3) {
  FlowPath    out;
  Tensor      h = input, ld = ZeroLogdet(input);
  const float dt = (inverse ? -1.f : 1.f) / steps, half = .5f * dt, sixth = dt / 6.f;
  int         stage = 0;
  for (int i = 0; i < steps; ++i) {
    const float t = inverse ? 1.f + dt * i : dt * i;
    const auto  a = CnfField(p, h, t, trace_mode);
    const auto  b = CnfField(p, h + double(half) * a.velocity, t + half, trace_mode);
    const auto  c = CnfField(p, h + double(half) * b.velocity, t + half, trace_mode);
    const auto  d = CnfField(p, h + double(dt) * c.velocity, t + dt, trace_mode);
    h  = h + double(sixth) * ((a.velocity + d.velocity) + 2.0 * (b.velocity + c.velocity));
    ld = ld + double(sixth) * ((a.divergence + d.divergence) + 2.0 * (b.divergence + c.divergence));
    if ((i + 1) % (steps / 4) == 0) {
      const auto endpoint = CnfField(p, h, t + dt, trace_mode);
      CouplingFields(out.fields, stage++, h, ld, endpoint.divergence, endpoint.velocity);
    }
  }
  out.latent = out.unscaled = h;
  out.logdet                = ld;
  return out;
}

class CnfDensity : public FlowModel {
 public:
  explicit CnfDensity(bool training = true) : FlowModel(10003) {
    store.get("field/hidden/W", {3, 8}, nn::init::normal(.3));  // CnfField reads the weights directly
    store.get("field/output/W", {8, 2}, nn::init::normal(.1));
    Build(training);
  }
  FlowPath Forward(const Tensor& x) override {
    auto         out  = CnfIntegrate(store, x);
    const Tensor logs = store.get("log_scale", {2}, nn::init::zeros());
    out.latent        = out.latent * nn::exp(logs);
    out.logdet        = out.logdet + nn::sum(logs, {0});
    return out;
  }
  FlowInverse Inverse(const Tensor& z) override {
    const Tensor logs = store.get("log_scale", {2}, nn::init::zeros());
    const auto   path = CnfIntegrate(store, z * nn::exp(-logs), true);
    return {path.latent, path.logdet - nn::sum(logs, {0})};
  }
  /// tflow's field diagnostics at t = .5 on the input: velocity, the exact and paired
  /// Hutchinson traces and the actual field Jacobian rows.
  void Extra(const Tensor& x, const FlowPath&, Map& out) override {
    const auto field          = CnfField(store, x, .5f);
    out["field_velocity"]       = field.velocity;
    out["field_exact_trace"]    = field.exact_trace;
    out["field_hutchinson_plus"] = field.hutchinson_plus;
    out["field_hutchinson_minus"] = field.hutchinson_minus;
    for (int k = 0; k < 2; ++k)
      out["field_jacobian_row" + std::to_string(k)] =
          nn::grad(nn::sum(Col(field.velocity, k)), {x}, nn::GradOptions{true, false, false})[0];
  }
};

}  // namespace flowd
