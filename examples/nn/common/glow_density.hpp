// Two-channel Glow steps: data-initialised ActNorm, a learned invertible 1x1
// mixing (2x2 matrix) and an affine coupling. Port of tflow's
// common/nn/glow_density.h and common/ml/glow_density.h (harness in
// flow_density.hpp; the coupling is RealNVP's).
//
// mix<i>/W starts as the rotation by .35 + .2 i; its log|det| uses the explicit
// 2x2 determinant and the inverse its adjugate, as tflow (nn::slogdet /
// nn::inverse give the same values; the closed form keeps float parity).
// ActNorm bias = -mean, log_scale = -.5 log(population variance + 1e-6) of the
// FIRST 32 fitting points, set once, sequentially per step. Store seed 9503.
#pragma once

#include "common/real_nvp_density.hpp"

namespace flowd {

struct GlowMixPath {
  Tensor value, logdet;  // logdet is one scalar (the matrix does not depend on x)
};

inline GlowMixPath GlowMix(nn::ParameterStore& p, const Tensor& x, int block, bool inverse = false) {
  const double theta = .35 + .2 * block;
  const Tensor w     = p.get("mix" + std::to_string(block) + "/W", {2, 2}, [theta](const nn::Shape&, nn::DType, nn::Generator&) {
    return nn::from_vector(std::vector<float>{float(std::cos(theta)), float(-std::sin(theta)), float(std::sin(theta)), float(std::cos(theta))},
                           {2, 2});
  });
  const auto   e     = [&](int i, int j) { return nn::reshape(nn::slice(w, {i, j}, {1, 1}), {}); };
  const Tensor a = e(0, 0), b = e(0, 1), c = e(1, 0), d = e(1, 1);
  const Tensor det    = a * d - b * c;
  const Tensor ld     = 0.5 * nn::log(nn::square(det));
  Tensor       matrix = w;
  if (inverse) matrix = nn::stack({nn::stack({d, -b}), nn::stack({-c, a})}) / det;
  return {nn::matmul(x, matrix), inverse ? -ld : ld};
}

class GlowDensity : public FlowModel {
 public:
  explicit GlowDensity(bool training = true) : FlowModel(9503) { Build(training); }

  FlowPath Forward(const Tensor& x) override {
    FlowPath out;
    Tensor   h = x, ld = ZeroLogdet(x);
    for (int i = 0; i < 4; ++i) {
      const auto   k     = std::to_string(i);
      const Tensor bias  = store.get("actnorm" + k + "/bias", {2}, nn::init::zeros());
      const Tensor scale = store.get("actnorm" + k + "/log_scale", {2}, nn::init::zeros());
      out.fields.emplace_back("actnorm_input" + k, h);
      h = (h + bias) * nn::exp(scale);
      out.fields.emplace_back("actnorm_output" + k, h);
      const auto mix = GlowMix(store, h, i);
      h              = mix.value;
      out.fields.emplace_back("mixed" + k, h);
      const auto coupling = RealNVPCoupling(store, h, i);
      h                   = coupling.value;
      ld                  = ld + (nn::sum(scale, {0}) + (mix.logdet + coupling.logdet));
      CouplingFields(out.fields, i, h, coupling.logdet, coupling.scale, coupling.shift);
    }
    out.latent = out.unscaled = h;
    out.logdet                = ld;
    return out;
  }
  FlowInverse Inverse(const Tensor& z) override {
    Tensor h = z, ld = ZeroLogdet(z);
    for (int i = 3; i >= 0; --i) {
      const auto   k        = std::to_string(i);
      const auto   coupling = RealNVPCoupling(store, h, i, true);
      const auto   mix      = GlowMix(store, coupling.value, i, true);
      const Tensor scale    = store.get("actnorm" + k + "/log_scale", {2}, nn::init::zeros());
      h  = mix.value * nn::exp(-scale) - store.get("actnorm" + k + "/bias", {2}, nn::init::zeros());
      ld = ld + ((coupling.logdet + mix.logdet) - nn::sum(scale, {0}));
    }
    return {h, ld};
  }

  /// tflow's one-time ActNorm data-dependent initialisation from the first 32 fit points.
  void Prepare(const Tensor& fit, const fs::path& root) override {
    std::vector<std::int64_t> rows(32);
    for (int i = 0; i < 32; ++i) rows[i] = i;
    const Tensor calibration = nn::take_rows(fit, rows);
    nn::save_npz((root / "calibration.npz").string(), {{"points", calibration}});
    Calibrate(calibration);
  }
  void Calibrate(const Tensor& calibration) {
    nn::NoGradGuard none;
    for (int i = 0; i < 4; ++i) {
      const auto path = Forward(calibration);
      Tensor     input;
      for (const auto& [k, v] : path.fields)
        if (k == "actnorm_input" + std::to_string(i)) input = v;
      const auto           v = input.to_vector<float>();
      const std::size_t    n = v.size() / 2;
      std::vector<float>   bias(2), scale(2);
      for (int c = 0; c < 2; ++c) {
        double mean = 0, var = 0;
        for (std::size_t b = 0; b < n; ++b) mean += double(v[b * 2 + c]) / n;
        for (std::size_t b = 0; b < n; ++b) var += std::pow(double(v[b * 2 + c]) - mean, 2) / n;
        bias[c]  = float(-mean);
        scale[c] = float(-.5 * std::log(var + 1e-6));
      }
      store.set("actnorm" + std::to_string(i) + "/bias", nn::from_vector(bias, {2}));
      store.set("actnorm" + std::to_string(i) + "/log_scale", nn::from_vector(scale, {2}));
    }
  }
};

}  // namespace flowd
