// Port of tflow's common/nn/latent_diffusion.h + common/ml/latent_diffusion.h:
// a DDPM on standardized 2-D VAE posterior-mean codes. The latent x0 estimate
// is deliberately UNCLIPPED. Shares the cosine 32-level schedule, time
// embedding and dense helpers of common/ddpm.hpp (namespace diffusion).
//
// tflow -> datamunge: session graph -> eager calls, AddSymbolicGradients ->
// nn::grad, Adam train_op -> nn::Optimizer (Adam .003), SaveV2 "final" ->
// final.npz. Parameter names match tflow's ("hidden1/W", ...), so TF's
// initial_arrays.npz / model_arrays.npz load directly.
#pragma once

#include "common/ddpm.hpp"

namespace diffusion {

/// tflow's LatentExtract: float(schedule[key][time]) as float32 [n, 1].
inline Tensor LatentExtract(const std::string& key, const Tensor& time) { return nn::reshape(Extract(key, time), {-1, 1}); }

/// tflow's LatentDenoiser: [state (2), 8 sin/cos time features] -> 32 SiLU -> 32 SiLU -> 2.
inline PredictorPath LatentDenoiser(const nn::ParameterStore& p, const Tensor& state, const Tensor& time) {
  const Tensor embedding = TimeEmbedding(time);
  const Tensor encoded   = nn::concat({state, embedding}, 1);
  const Tensor h1 = Dense(p, "hidden1", encoded, true), h2 = Dense(p, "hidden2", h1, true);
  return {Dense(p, "output", h2, false), embedding, h1, h2};
}

struct LatentReversePath {
  Tensor value, mean, variance, x0;
};

/// tflow's LatentReverse: UNCLIPPED x0 from epsilon, posterior mean c1 x0 + c2 x_t,
/// fixed posterior variance, value = mean + sqrt(variance) z.
inline LatentReversePath LatentReverse(const Tensor& state, const Tensor& time, const Tensor& epsilon, const Tensor& innovation) {
  const Tensor signal = LatentExtract("sqrt_alpha_bar", time), sigma = LatentExtract("sqrt_one_minus_alpha_bar", time);
  const Tensor x0       = nn::div(nn::sub(state, nn::mul(sigma, epsilon)), signal);
  const Tensor mean     = nn::add(nn::mul(LatentExtract("posterior_coefficient1", time), x0),
                                  nn::mul(LatentExtract("posterior_coefficient2", time), state));
  const Tensor variance = LatentExtract("posterior_variance", time);
  return {nn::add(mean, nn::mul(nn::sqrt(variance), innovation)), mean, variance, x0};
}

/// d sum(f(state)[:, column]) / d state.
template <class F>
inline Tensor ColumnInputGradient(const Tensor& x, std::int64_t column, F&& f) {
  Tensor s = x.detach().clone();
  s.requires_grad_();
  const Tensor y = f(s);
  return nn::grad(nn::sum(nn::slice_axis(y, 1, column, column + 1)), {s})[0];
}

/// tflow's ml::LatentDiffusionModel (seed 17003, 1474 parameters / 6 tensors).
class LatentDiffusionModel {
 public:
  nn::ParameterStore             store{17003};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit LatentDiffusionModel(bool training = true) {
    CreateMlp(store, 10, 2);
    if (training) optimizer = std::make_unique<nn::Optimizer>(store, .003, Adam());
  }
  Tensor Loss(const Tensor& x, const Tensor& e, const Tensor& t, Map* fields = nullptr) {
    const Tensor noised = nn::add(nn::mul(LatentExtract("sqrt_alpha_bar", t), x), nn::mul(LatentExtract("sqrt_one_minus_alpha_bar", t), e));
    const auto   path   = LatentDenoiser(store, noised, t);
    const Tensor loss   = nn::mean(nn::square(nn::sub(path.epsilon, e)));
    if (fields)
      *fields = Detached({{"noised", noised}, {"epsilon", path.epsilon}, {"time_embedding", path.time_embedding},
                          {"hidden1", path.hidden1}, {"hidden2", path.hidden2}, {"loss", loss}});
    return loss;
  }
  Map Evaluate(const Tensor& x, const Tensor& e, const Tensor& t) {
    nn::NoGradGuard guard;
    Map             fields;
    Loss(x, e, t, &fields);
    return fields;
  }
  Map  Gradient(const Tensor& x, const Tensor& e, const Tensor& t) { return ParameterGradients(store, Loss(x, e, t)); }
  void Update(const Tensor& x, const Tensor& e, const Tensor& t) { optimizer->step(nn::grad(Loss(x, e, t), store.parameters())); }
  /// State-only unclipped reverse transition plus the full 2x2 epsilon / transition Jacobians.
  Map Step(const Tensor& x, const Tensor& t, const Tensor& z) {
    Map out;
    {
      nn::NoGradGuard guard;
      const auto      path = LatentDenoiser(store, x, t);
      const auto      r    = LatentReverse(x, t, path.epsilon, z);
      out = {{"epsilon", path.epsilon}, {"next", r.value}, {"mean", r.mean}, {"variance", r.variance}, {"x0_raw", r.x0}};
    }
    for (int c = 0; c < 2; ++c) {
      out["epsilon_jacobian_row" + std::to_string(c)] =
          ColumnInputGradient(x, c, [&](const Tensor& s) { return LatentDenoiser(store, s, t).epsilon; });
      out["next_jacobian_row" + std::to_string(c)] = ColumnInputGradient(x, c, [&](const Tensor& s) {
        return LatentReverse(s, t, LatentDenoiser(store, s, t).epsilon, z).value;
      });
    }
    return out;
  }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : Arrays()) out.push_back(Bits(v));
    return out;
  }
};

}  // namespace diffusion
