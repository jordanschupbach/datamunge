// Port of tflow's common/nn/ddim.h and common/ml/ddim.h: DDIM respaced
// transition with clipping-consistent epsilon (eta = 1 on the full 32-level
// schedule equals the DDPM posterior), on frozen DDPM epsilon weights.
#pragma once

#include "common/ddpm.hpp"

namespace diffusion {

struct DdimPath {
  Tensor value, mean, sigma, direction, x0_raw, x0_clipped, epsilon_consistent;
};

/// tflow's DdimStep. `previous` = -1 means the clean level (alpha_bar 1); `eta` is a float32 scalar.
inline DdimPath DdimStep(const Tensor& state, const Tensor& time, const Tensor& previous, const Tensor& epsilon,
                         const Tensor& innovation, const Tensor& eta) {
  const Tensor bar = Extract("alpha_bar", time), signal = Extract("sqrt_alpha_bar", time),
               noise_scale = Extract("sqrt_one_minus_alpha_bar", time);
  const Tensor raw     = nn::div(nn::sub(state, nn::mul(noise_scale, epsilon)), signal);
  const Tensor clipped = nn::minimum(nn::maximum(raw, nn::scalar_like(-1, raw)), nn::scalar_like(1, raw));
  const auto   prev    = nn::cast(previous, nn::DType::Int64).to_vector<std::int64_t>();
  std::vector<std::int32_t> clamped(prev.size());
  for (std::size_t i = 0; i < prev.size(); ++i) clamped[i] = std::int32_t(std::max<std::int64_t>(prev[i], 0));
  const Tensor condition    = nn::reshape(nn::greater_equal(nn::cast(previous, nn::DType::Int32), nn::scalar(0, nn::DType::Int32)), {-1, 1, 1, 1});
  const Tensor previous_bar = nn::where(condition, Extract("alpha_bar", Ints(clamped)), nn::add(nn::mul(bar, 0.0), 1.0));
  const Tensor variance     = nn::mul(nn::square(eta), nn::mul(nn::div(nn::sub(1.0, previous_bar), nn::sub(1.0, bar)),
                                                               nn::sub(1.0, nn::div(bar, previous_bar))));
  const Tensor sigma        = nn::sqrt(nn::maximum(variance, nn::scalar_like(0, variance)));
  const Tensor dir_arg      = nn::sub(nn::sub(1.0, previous_bar), variance);
  const Tensor direction    = nn::sqrt(nn::maximum(dir_arg, nn::scalar_like(0, dir_arg)));
  const Tensor consistent   = nn::div(nn::sub(state, nn::mul(signal, clipped)), noise_scale);
  const Tensor mean         = nn::add(nn::mul(nn::sqrt(previous_bar), clipped), nn::mul(direction, consistent));
  return {nn::add(mean, nn::mul(sigma, innovation)), mean, sigma, direction, raw, clipped, consistent};
}

/// tflow's ml::DdimModel: the frozen 5504-parameter DDPM epsilon MLP, DDIM-only sampling.
class DdimModel {
 public:
  nn::ParameterStore store{11003};

  DdimModel() { CreateMlp(store, 72, 64); }

  Map Reverse(const Tensor& x, int t, int prev, float amount, const Tensor& z) {
    check::Require(t >= 0 && t < 32 && prev >= -1 && prev < t && amount >= 0 && amount <= 1 && x.shape() == z.shape(),
                   "valid DDIM ordered levels-eta-state-noise");
    const auto   n = x.size(0);
    const Tensor ti = FullInt(n, t), pi = FullInt(n, prev), eta = nn::scalar(amount, nn::DType::Float32);
    Map          out;
    {
      nn::NoGradGuard guard;
      const auto      p    = DdpmPredictor(store, x, ti);
      const auto      path = DdimStep(x, ti, pi, p.epsilon, z, eta);
      out = {{"epsilon", p.epsilon},          {"next", path.value},          {"mean", path.mean},
             {"sigma", path.sigma},           {"direction", path.direction}, {"x0_raw", path.x0_raw},
             {"x0_clipped", path.x0_clipped}, {"epsilon_consistent", path.epsilon_consistent}};
    }
    out["input_gradient"] = FirstPixelInputGradient(
        x, [&](const Tensor& s) { return DdimStep(s, ti, pi, DdpmPredictor(store, s, ti).epsilon, z, eta).value; });
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
