// Port of tflow's common/nn/consistency_model.h and common/ml/consistency_model.h:
// boundary-preserving consistency map f(x, t) = cskip(t) x + cout(t) F(x, t) on
// 2-D points, the exact Gaussian VE-score Euler teacher, and the online Adam /
// separate post-Adam EMA target distillation model (target branch stopped).
#pragma once

#include "common/ddpm.hpp"

namespace diffusion {

struct ConsistencyPath {
  Tensor value, raw, skip, out, time_embedding;
};

/// tflow's ConsistencyMap: log-time sinusoidal features, input x / sqrt(.25 + t^2),
/// 10 -> 32 SiLU -> 32 SiLU -> 2; cskip = .25 / ((t - .01)^2 + .25), cout = .5 (t - .01) / sqrt(.25 + t^2).
inline ConsistencyPath ConsistencyMap(const nn::ParameterStore& p, const Tensor& x, const Tensor& time) {
  const Tensor t     = nn::reshape(time, {-1, 1});
  const Tensor phase = nn::mul(nn::div(nn::log(nn::div(t, double(.01f))), double(float(std::log(200.)))),
                               nn::from_vector(std::vector<float>{float(M_PI), float(2 * M_PI), float(4 * M_PI), float(8 * M_PI)}, {1, 4}));
  const Tensor embedding = nn::concat({nn::sin(phase), nn::cos(phase)}, 1);
  const Tensor scale     = nn::sqrt(nn::add(nn::square(t), double(.25f)));
  const Tensor input     = nn::concat({nn::div(x, scale), embedding}, 1);
  const Tensor h1 = Dense(p, "hidden1", input, true), h2 = Dense(p, "hidden2", h1, true);
  const Tensor raw   = Dense(p, "output", h2, false);
  const Tensor shift = nn::sub(t, double(.01f));
  const Tensor skip  = nn::div(.25, nn::add(nn::square(shift), .25));
  const Tensor out   = nn::div(nn::mul(shift, .5), scale);
  return {nn::add(nn::mul(skip, x), nn::mul(out, raw)), raw, skip, out, embedding};
}

/// t (x - mu) / (variance + t^2) with mu (.3, -.2), variance (.49, .25).
inline Tensor ConsistencyGaussianDrift(const Tensor& x, const Tensor& time) {
  const Tensor t = nn::reshape(time, {-1, 1});
  return nn::div(nn::mul(t, nn::sub(x, nn::tensor({{.3f, -.2f}}))), nn::add(nn::tensor({{.49f, .25f}}), nn::square(t)));
}

inline Tensor ConsistencyEulerTeacher(const Tensor& x, const Tensor& time, const Tensor& previous) {
  return nn::add(x, nn::mul(nn::reshape(nn::sub(previous, time), {-1, 1}), ConsistencyGaussianDrift(x, time)));
}

/// tflow's ml::ConsistencyModel: online (seed 19003) and target (19007) 1474-parameter
/// maps; the target starts as a hard copy of the online network.
class ConsistencyModel {
 public:
  nn::ParameterStore             online{19003}, target{19007};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit ConsistencyModel(bool training = true) {
    CreateMlp(online, 10, 2);
    CreateMlp(target, 10, 2);
    if (training) optimizer = std::make_unique<nn::Optimizer>(online, .003, Adam());
    HardSync();
  }
  void HardSync() {
    for (const auto& name : online.names()) target.set(name, online.at(name).detach().clone());
  }
  Tensor Loss(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& p, Map* fields = nullptr) {
    const Tensor noised  = nn::add(x, nn::mul(nn::reshape(t, {-1, 1}), e));
    const Tensor teacher = ConsistencyEulerTeacher(noised, t, p);
    const auto   source  = ConsistencyMap(online, noised, t);
    const auto   stopped = ConsistencyMap(target, teacher, p);
    const Tensor loss    = nn::mean(nn::square(nn::sub(source.value, nn::stop_gradient(stopped.value))));
    if (fields)
      *fields = Detached({{"noised", noised}, {"teacher", teacher}, {"online", source.value}, {"target", stopped.value},
                          {"skip", source.skip}, {"out", source.out}, {"loss", loss}});
    return loss;
  }
  Map Evaluate(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& p) {
    nn::NoGradGuard guard;
    Map             fields;
    Loss(x, e, t, p, &fields);
    return fields;
  }
  Map Gradient(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& p) { return ParameterGradients(online, Loss(x, e, t, p)); }
  /// The stop-gradient leaves every target parameter without a gradient.
  bool TargetGradientsAbsent(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& p) {
    nn::GradOptions options;
    options.allow_unused = true;
    for (const auto& g : nn::grad(Loss(x, e, t, p), target.parameters(), options))
      if (g.defined()) return false;
    return true;
  }
  void Update(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& p) {
    optimizer->step(nn::grad(Loss(x, e, t, p), online.parameters()));
  }
  /// Separate post-Adam EMA: target = .1 online + (1 - .1f) target, in float32 as tflow.
  void Ema() {
    nn::NoGradGuard guard;
    for (const auto& name : online.names())
      target.set(name, nn::add(nn::mul(online.at(name), double(.1f)), nn::mul(target.at(name), double(1.f - .1f))).detach());
  }
  /// State/time-only EMA consistency map with its full input Jacobian.
  Map Predict(const Tensor& x, const Tensor& t) {
    Tensor s = x.detach().clone();
    s.requires_grad_();
    const auto path = ConsistencyMap(target, s, t);
    Map        out;
    for (int row = 0; row < 2; ++row) {
      nn::GradOptions options;
      options.retain_graph = true;
      out["jacobian_row" + std::to_string(row)] = nn::grad(nn::sum(nn::slice_axis(path.value, 1, row, row + 1)), {s}, options)[0];
    }
    nn::NoGradGuard guard;
    out["value"]  = path.value.detach();
    out["online"] = ConsistencyMap(online, x, t).value;
    out["raw"]    = path.raw.detach();
    out["skip"]   = path.skip.detach();
    out["out"]    = path.out.detach();
    return out;
  }
  Map                      Arrays(bool use_target = false) const { return replay::Arrays(use_target ? target : online); }
  std::vector<std::string> Snapshot(bool use_target = false) const {
    std::vector<std::string> out;
    for (const auto& [k, v] : Arrays(use_target)) out.push_back(Bits(v));
    return out;
  }
};

}  // namespace diffusion
