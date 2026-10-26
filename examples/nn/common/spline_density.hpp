// Neural spline flow: monotone rational-quadratic couplings (4 bins on [-4, 4])
// with an analytic quadratic inverse and C1 identity tails. Port of tflow's
// common/nn/spline_density.h and common/ml/spline_density.h (harness in
// flow_density.hpp).
//
// Heads (11 per point): 4 width and 4 height logits (softmax, minimum bin .01 of
// 8 units scaled by .96) and 3 interior knot derivatives .001 + softplus(raw +
// log(expm1(.999))), boundary derivatives 1. Outside [-4, 4] the map is the
// identity (logdet 0). Coupling i: 1 -> 16 tanh -> 11, W ~ N(0, .3) / N(0, .01).
// Store seed 9803.
#pragma once

#include "common/real_nvp_density.hpp"

namespace flowd {

struct RationalQuadraticPath {
  Tensor value, logdet, widths, heights, derivatives;
};

inline RationalQuadraticPath RationalQuadraticSpline(const Tensor& input, const Tensor& heads, bool inverse = false) {
  const Tensor widths  = 8.0 * (0.01 + 0.96 * nn::softmax(Col(heads, 0, 4), 1));
  const Tensor heights = 8.0 * (0.01 + 0.96 * nn::softmax(Col(heads, 4, 4), 1));
  const Tensor zeros = Col(heads, 0) * 0.0, left = zeros - 4.0, right = zeros + 4.0, ones = zeros + 1.0;
  const Tensor derivatives = nn::concat(
      {ones, 0.001 + nn::softplus(Col(heads, 8, 3) + double(float(std::log(std::expm1(.999))))), ones}, 1);
  std::vector<Tensor> xk{left}, yk{left};
  for (int j = 1; j < 4; ++j) {
    xk.push_back(-4.0 + nn::sum(Col(widths, 0, j), {1}, true));
    yk.push_back(-4.0 + nn::sum(Col(heights, 0, j), {1}, true));
  }
  xk.push_back(right), yk.push_back(right);
  const Tensor x0all = nn::concat({xk[0], xk[1], xk[2], xk[3]}, 1), x1all = nn::concat({xk[1], xk[2], xk[3], xk[4]}, 1);
  const Tensor y0all = nn::concat({yk[0], yk[1], yk[2], yk[3]}, 1), y1all = nn::concat({yk[1], yk[2], yk[3], yk[4]}, 1);
  const Tensor actual_widths = x1all - x0all, actual_heights = y1all - y0all;
  const Tensor edges   = inverse ? y1all : x1all;
  const Tensor bounded = nn::minimum(nn::maximum(input, nn::full({}, -4.0)), nn::full({}, 4.0));
  Tensor       index   = nn::sum(nn::cast(nn::greater_equal(bounded, edges), nn::DType::Int32), {1});
  index                = nn::minimum(index, nn::full({}, 3, nn::DType::Int32));
  const Tensor mask    = nn::one_hot(index, 4, 1.0, 0.0);
  const auto   pick    = [&](const Tensor& x) { return nn::sum(x * mask, {1}, true); };
  const Tensor x0 = pick(x0all), y0 = pick(y0all), w = pick(actual_widths), h = pick(actual_heights), d0 = pick(Col(derivatives, 0, 4)),
               d1 = pick(Col(derivatives, 1, 4));
  const Tensor delta = h / w, curvature = (d0 + d1) - 2.0 * delta;
  Tensor       theta, value;
  if (inverse) {
    const Tensor offset = bounded - y0;
    const Tensor a = offset * curvature + h * (delta - d0), b = h * d0 - offset * curvature, c = -(delta * offset);
    const Tensor discriminant = nn::square(b) - 4.0 * (a * c);
    theta                     = (2.0 * c) / (-b - nn::sqrt(discriminant));
    value                     = x0 + theta * w;
  } else {
    theta = (bounded - x0) / w;
  }
  const Tensor one_minus = 1.0 - theta, blend = theta * one_minus;
  const Tensor denominator = delta + curvature * blend;
  if (!inverse) value = y0 + h * (delta * nn::square(theta) + d0 * blend) / denominator;
  const Tensor derivative_numerator =
      nn::square(delta) * ((d1 * nn::square(theta) + 2.0 * (delta * blend)) + d0 * nn::square(one_minus));
  const Tensor logdet = nn::log(derivative_numerator) - 2.0 * nn::log(denominator);
  const Tensor inside = nn::logical_and(nn::greater_equal(input, -4.0), nn::less_equal(input, 4.0));
  return {nn::where(inside, value, input), nn::reshape(nn::where(inside, inverse ? -logdet : logdet, zeros), {-1}), actual_widths,
          actual_heights, derivatives};
}

inline RationalQuadraticPath SplineCoupling(nn::ParameterStore& p, const Tensor& input, int block, bool inverse = false) {
  const std::string name      = "coupling" + std::to_string(block);
  const int         condition = block % 2;
  const Tensor      a = Col(input, condition), b = Col(input, 1 - condition);
  const Tensor heads = Dense(p, name + "/heads", Dense(p, name + "/hidden", a, 1, 16, .3, .05, true), 16, 11, .01, 0);
  auto         out   = RationalQuadraticSpline(b, heads, inverse);
  out.value          = condition ? nn::concat({out.value, a}, 1) : nn::concat({a, out.value}, 1);
  return out;
}

class SplineDensity : public FlowModel {
 public:
  explicit SplineDensity(bool training = true) : FlowModel(9803) { Build(training); }
  FlowPath Forward(const Tensor& x) override {
    FlowPath out;
    Tensor   h = x, ld = ZeroLogdet(x);
    for (int i = 0; i < 4; ++i) {
      const auto c = SplineCoupling(store, h, i);
      h            = c.value;
      ld           = ld + c.logdet;
      // tflow stores bin widths / heights in the scale / shift diagnostic fields.
      CouplingFields(out.fields, i, h, c.logdet, c.widths, c.heights);
      out.fields.emplace_back("coupling_derivatives" + std::to_string(i), c.derivatives);
    }
    GlobalScale(store, out, h, ld);
    return out;
  }
  FlowInverse Inverse(const Tensor& z) override {
    auto [h, ld] = UnscaleGlobal(store, z);
    for (int i = 3; i >= 0; --i) {
      const auto c = SplineCoupling(store, h, i, true);
      h            = c.value;
      ld           = ld + c.logdet;
    }
    return {h, ld};
  }
};

}  // namespace flowd
