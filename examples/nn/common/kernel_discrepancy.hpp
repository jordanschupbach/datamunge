// Port of tflow's common/nn/kernel_discrepancy.h: the averaged inverse-multiquadric
// kernel and the unbiased two-sample MMD^2 with every intermediate term exposed
// (diagonals excluded only within each sample; unequal sample counts supported).
// gan.hpp has a value-only copy in namespace gan for the GAN pixel diagnostics.
#pragma once

#include <datamunge/nn/nn.hpp>

#include "common/checks.hpp"

namespace ae {

namespace nn = datamunge::nn;

/// Mean of c / (c + ||a_i - b_j||^2) for c = 1, 4, 16: [n, d] x [m, d] -> [n, m].
inline nn::Tensor ImqKernel(const nn::Tensor& a, const nn::Tensor& b) {
  const nn::Tensor d = nn::sum(nn::square(nn::expand_dims(a, 1) - nn::expand_dims(b, 0)), {2});
  return (1.0 / (d + 1.0) + 4.0 / (d + 4.0) + 16.0 / (d + 16.0)) / 3.0;
}

struct MmdOutput {
  nn::Tensor xx, yy, xy, within_x, within_y, cross, value;
};

/// Unbiased IMQ MMD^2 between samples x [n, d] and y [m, d]; needs n, m >= 2. Not clipped (can be negative).
inline MmdOutput ImqMmd(const nn::Tensor& x, const nn::Tensor& y) {
  const double n = double(x.size(0)), m = double(y.size(0));
  check::Require(n >= 2 && m >= 2, "unbiased MMD needs at least2samples EACH");
  MmdOutput o;
  o.xx       = ImqKernel(x, x);
  o.yy       = ImqKernel(y, y);
  o.xy       = ImqKernel(x, y);
  o.within_x = (nn::sum(o.xx) - n) / (n * (n - 1));
  o.within_y = (nn::sum(o.yy) - m) / (m * (m - 1));
  o.cross    = nn::mean(o.xy);
  o.value    = o.within_x + o.within_y - 2.0 * o.cross;
  return o;
}

}  // namespace ae
