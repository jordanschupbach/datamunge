// Planar (rank-one) and radial flows with constrained invertibility and explicit
// inverse logdets. Port of tflow's common/nn/planar_radial_density.h and
// common/ml/planar_radial_density.h (harness in flow_density.hpp).
//
// Planar: y = x + u tanh(w.x + b) with w = e^f (cos a, sin a), u = (p / e^f)(cos a,
// sin a) + q (-sin a, cos a), p = softplus(raw) - .9 > -1 so det = 1 + p (1 -
// tanh^2) > 0. Inverse: 32 bisection steps on t + p tanh(t) = c, then ONE Newton
// correction whose root and denominator are stop-gradient, so parameter
// gradients flow by the implicit-function rule (tflow's construction).
// Radial: y = c + (1 + beta / (alpha + r)) (x - c), alpha = .1 + softplus,
// beta = softplus - alpha > -alpha; the inverse solves the radius quadratic in
// stable closed form. Each of four blocks, then a global diagonal scale. Store
// seed 9903.
#pragma once

#include "common/real_nvp_density.hpp"

namespace flowd {

struct ElementaryFlowPath {
  Tensor value, logdet, factor, coordinate;
};

/// sqrt(|offset|^2) with a finite zero-radius derivative (tflow FlowRadius).
inline Tensor FlowRadius(const Tensor& offset) {
  const Tensor square = nn::sum(nn::square(offset), {1}, true);
  return nn::where(nn::greater(square, 0.0), nn::sqrt(nn::maximum(square, nn::full({}, 1e-30))), square * 0.0);
}

inline ElementaryFlowPath ElementaryFlow(nn::ParameterStore& p, const Tensor& input, int block, bool radial, bool inverse = false) {
  const std::string name = "flow" + std::to_string(block) + "/";
  auto scalar = [&](const std::string& key, double value) { return p.get(name + key, {}, nn::init::constant(value)); };
  Tensor value, ld, factor, coordinate;
  if (radial) {
    const Tensor center = p.get(name + "center", {2}, nn::init::normal(.2));
    const Tensor alpha  = 0.1 + nn::softplus(scalar("raw_alpha", .4));
    const Tensor beta   = nn::softplus(scalar("raw_beta", .56)) - alpha;
    const Tensor offset = input - center, distance = FlowRadius(offset);
    Tensor       radius = distance;
    if (inverse) {
      const Tensor a        = (alpha + beta) - distance;
      const Tensor disc     = nn::sqrt(nn::square(a) + 4.0 * (alpha * distance));
      const Tensor positive = (2.0 * (alpha * distance)) / nn::maximum(a + disc, nn::full({}, 1e-20));
      const Tensor negative = 0.5 * (disc - a);
      radius                = nn::where(nn::greater_equal(a, 0.0), positive, negative);
    }
    const Tensor denom        = alpha + radius;
    factor                    = 1.0 + beta / denom;
    const Tensor radial_factor = 1.0 + (beta * alpha) / nn::square(denom);
    const Tensor forward_ld    = nn::log(factor) + nn::log(radial_factor);
    value                      = center + (inverse ? offset / factor : offset * factor);
    ld                         = inverse ? -forward_ld : forward_ld;
    coordinate                 = radius;
  } else {
    const Tensor angle         = scalar("angle", double(.4f + .65f * block));
    const Tensor frequency     = nn::exp(scalar("log_frequency", 0));
    const Tensor projection    = nn::softplus(scalar("raw_projection", .4)) - 0.9;
    const Tensor perpendicular = scalar("perpendicular", double(.02f * (block % 2 ? -1 : 1)));
    const Tensor bias          = scalar("bias", .05);
    const Tensor direction     = nn::stack({nn::cos(angle), nn::sin(angle)});
    const Tensor tangent       = nn::stack({-nn::sin(angle), nn::cos(angle)});
    const Tensor w             = frequency * direction;
    const Tensor u             = (projection / frequency) * direction + perpendicular * tangent;
    const Tensor c             = nn::sum(input * w, {1}, true) + bias;
    Tensor       t             = c;
    if (inverse) {
      const Tensor margin = nn::sqrt(nn::square(projection)) + 1.0;
      Tensor       low = c - margin, high = c + margin;
      for (int iteration = 0; iteration < 32; ++iteration) {
        const Tensor mid      = 0.5 * (low + high);
        const Tensor residual = (mid + projection * nn::tanh(mid)) - c;
        const Tensor below    = nn::less(residual, 0.0);
        low                   = nn::where(below, mid, low);
        high                  = nn::where(below, high, mid);
      }
      const Tensor root        = nn::stop_gradient(0.5 * (low + high));
      const Tensor ht          = nn::tanh(root);
      const Tensor denominator = nn::stop_gradient(1.0 + projection * (1.0 - nn::square(ht)));
      const Tensor residual    = (c - root) - projection * ht;
      t                        = root + residual / denominator;
    }
    const Tensor activation = nn::tanh(t);
    factor                  = 1.0 + projection * (1.0 - nn::square(activation));
    const Tensor shift      = activation * u;
    value                   = inverse ? input - shift : input + shift;
    ld                      = inverse ? -nn::log(factor) : nn::log(factor);
    coordinate              = t;
  }
  return {value, nn::reshape(ld, {-1}), factor, coordinate};
}

class PlanarRadialDensity : public FlowModel {
 public:
  bool radial;
  explicit PlanarRadialDensity(bool training = true, bool radial_flow = false) : FlowModel(9903), radial(radial_flow) { Build(training); }
  FlowPath Forward(const Tensor& x) override {
    FlowPath out;
    Tensor   h = x, ld = ZeroLogdet(x);
    for (int i = 0; i < 4; ++i) {
      const auto block = ElementaryFlow(store, h, i, radial);
      h                = block.value;
      ld               = ld + block.logdet;
      CouplingFields(out.fields, i, h, block.logdet, block.factor, block.coordinate);
    }
    GlobalScale(store, out, h, ld);
    return out;
  }
  FlowInverse Inverse(const Tensor& z) override {
    auto [h, ld] = UnscaleGlobal(store, z);
    for (int i = 3; i >= 0; --i) {
      const auto block = ElementaryFlow(store, h, i, radial, true);
      h                = block.value;
      ld               = ld + block.logdet;
    }
    return {h, ld};
  }
  /// tflow InverseGradient: d/dparams of mean |inverse(z)|^2 + .3 mean inverse logdet.
  Map InverseGradient(const Tensor& z) {
    const auto   inv       = Inverse(z);
    const Tensor objective = nn::mean(nn::sum(nn::square(inv.value), {1}), {0}) + 0.3 * nn::mean(inv.logdet, {0});
    return replay::Named(store, nn::grad(objective, store.parameters()));
  }
};

}  // namespace flowd
