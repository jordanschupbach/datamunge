// CMD: just run-nn custom_gradient
// Writing a custom gradient.
//
// Port of tflow's custom_gradient. TF looks gradients up per op type in a
// global registry (REGISTER_GRADIENT_OP). In datamunge an nn::Function pairs
// a forward with its own backward, so there is no registry to consult:
//   1. A mathematically correct gradient for a new op (Erfc).
//   2. A deliberately "wrong" gradient as a training trick: a
//      straight-through estimator for Round, whose true gradient is zero
//      almost everywhere.
//   3. The stop-gradient trick, which rewires the backward pass with no
//      custom function at all.

#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using Tensors = std::vector<Tensor>;

namespace {

const double kPi = 3.14159265358979323846;

// d/dx erfc(x) = -2/sqrt(pi) * exp(-x^2). The backward is written with nn
// ops, so it is differentiable too (second-order gradients work).
const nn::Function Erfc(
    "Erfc", [](const Tensors& in) { return Tensors{nn::erfc(in[0])}; },
    [](const Tensors& in, const Tensors&, const Tensors& grad) {
      return Tensors{grad[0] * nn::exp(-nn::square(in[0])) * (-2.0 / std::sqrt(kPi))};
    });

// Straight-through estimator: pretend round(x) has derivative 1. Used for
// quantization-aware training, where the true gradient (0 a.e.) is useless.
const nn::Function RoundStraightThrough(
    "RoundSTE", [](const Tensors& in) { return Tensors{nn::round(in[0])}; },
    [](const Tensors&, const Tensors&, const Tensors& grad) { return Tensors{grad[0]}; });

void MissingGradient() {
  Header("What happens without a gradient");
  // ceil has zero gradient everywhere it is defined, so it records no
  // history; asking for a gradient through it is reported, not silently 0.
  Tensor            x       = nn::tensor({0.5}).requires_grad_();
  const std::string message = check::Throws([&] { (void)nn::grad(nn::sum(nn::ceil(x)), {x}); }, "ceil gradient");
  std::cout << "  grad through ceil -> " << message << "\n";
}

void ErfcGradient() {
  Header("Custom gradient for Erfc, checked against finite differences");
  const std::vector<float> xs = {-1.5f, -0.5f, 0.f, 0.5f, 1.5f};
  Tensor                   x  = nn::from_vector(xs).requires_grad_();
  const Tensor             g  = nn::grad(nn::sum(Erfc(x)), {x})[0];
  const float              eps = 1e-3f;
  std::cout << "      x     custom       closed form   finite diff\n";
  for (int i = 0; i < 5; ++i) {
    const float xi     = xs[static_cast<std::size_t>(i)];
    const float closed = -2.f / std::sqrt(static_cast<float>(kPi)) * std::exp(-xi * xi);
    const float fd     = (std::erfc(xi + eps) - std::erfc(xi - eps)) / (2 * eps);
    std::printf("  %6.2f   %10.5f   %10.5f   %10.5f\n", xi, g.data_ptr<float>()[i], closed, fd);
    check::Near(g.data_ptr<float>()[i], closed, 1e-6, "Erfc gradient");
  }
  // In float64 the gradient and its own derivative pass nn::gradcheck.
  Tensor     x64 = nn::cast(x.detach(), nn::DType::Float64).requires_grad_();
  const auto r1  = nn::gradcheck([](const Tensors& in) { return Erfc(in[0]); }, {x64});
  const auto r2  = nn::gradgradcheck([](const Tensors& in) { return Erfc(in[0]); }, {x64});
  std::cout << "  gradcheck ok=" << r1.ok << " (max error " << r1.max_abs_error << "), gradgradcheck ok=" << r2.ok
            << "\n";
  check::Require(r1.ok && r2.ok, "Erfc gradient checks");
}

void StraightThrough() {
  Header("Straight-through estimator for Round (quantization-aware training)");
  // Quantize x to a grid of 0.25 and push the quantized value toward a target
  // on that grid. With the true (zero) gradient x would never move; with the
  // straight-through gradient, loss gradients reach x and training works.
  Tensor x = nn::scalar(0.1f).requires_grad_();
  Tensor q, loss, g;
  for (int i = 0; i <= 30; ++i) {
    q    = RoundStraightThrough(x * 4.0) / 4.0;
    loss = nn::square(q - 2.5);
    g    = nn::grad(loss, {x})[0];
    if (i % 5 == 0)
      std::printf("  step %2d  x %.4f  quantized %.2f  loss %.4f  dloss/dx %.4f\n", i, x.item(), q.item(), loss.item(),
                  g.item());
    nn::NoGradGuard guard;
    x.sub_(g, 0.1);
  }
  check::Near(q.item(), 2.5, 1e-6, "quantized value reaches the target");
}

void StopGradientTrick() {
  Header("The stop-gradient trick (no custom function needed)");
  // y = x + stop(f(x) - x) evaluates to f(x) in the forward pass but has
  // gradient 1 in the backward pass, whatever f is. Here f = ceil, which has
  // no gradient at all: the backward pass never visits it.
  Tensor       x    = nn::tensor({0.2, 1.7, -0.4}).requires_grad_();
  const Tensor y    = x + nn::stop_gradient(nn::ceil(x) - x);
  const Tensor loss = nn::sum(nn::square(y), {0});
  const Tensor g    = nn::grad(loss, {x})[0];
  std::cout << "  x        = " << x.values_str() << "\n"
            << "  y        = " << y.values_str() << "  (ceil(x) in the forward pass)\n"
            << "  dloss/dx = " << g.values_str() << "  (2*y, as if y = x)\n";
  check::Require(g.to_vector<double>() == std::vector<double>{2, 4, 0}, "straight-through gradient is 2*y");
}

}  // namespace

int main() {
  try {
    MissingGradient();
    ErfcGradient();
    StraightThrough();
    StopGradientTrick();
    std::cout << "\nAll custom gradient checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
