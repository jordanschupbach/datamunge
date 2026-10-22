// CMD: just run-nn autodiff
// Automatic differentiation with nn::grad.
//
// Port of tflow's autodiff (AddSymbolicGradients). Ops record themselves on
// a tape as they run; nn::grad(outputs, inputs) walks it backwards. With
// create_graph the backward pass is itself recorded, so a gradient can be
// differentiated again. Placeholders and feeds become function arguments;
// Variables become leaf tensors with requires_grad, updated in place.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

// ---------------------------------------------------------------------------
// 1. Scalar function, first and second derivative
// ---------------------------------------------------------------------------

void ScalarDerivatives() {
  Header("Scalar: y = x^3 + 2x");
  for (float xv : {0.5f, 1.f, 2.f}) {
    Tensor          x = nn::scalar(xv).requires_grad_();
    Tensor          y = nn::pow(x, 3.0) + 2.0 * x;
    nn::GradOptions create;
    create.create_graph = true;  // keep dy/dx differentiable
    Tensor dy  = nn::grad(y, {x}, create)[0];
    Tensor d2y = nn::grad(dy, {x})[0];
    std::printf("  x=%.1f  y=%8.3f  dy/dx=%8.3f (expect %8.3f)  d2y/dx2=%7.3f (expect %7.3f)\n", xv, y.item(),
                dy.item(), 3 * xv * xv + 2, d2y.item(), 6 * xv);
    check::Near(dy.item(), 3 * xv * xv + 2, 1e-5, "first derivative");
    check::Near(d2y.item(), 6 * xv, 1e-5, "second derivative");
  }
}

// ---------------------------------------------------------------------------
// 2. Gradients of a loss w.r.t. matrices, checked against finite differences
// ---------------------------------------------------------------------------

void MatrixGradients() {
  Header("Matrices: loss = mean((xW + b - t)^2), checked numerically");
  const Tensor x = nn::tensor({{1, 2, 3}, {0, -1, 1}, {2, 2, 2}, {-1, 0, 1}});  // [4,3]
  const Tensor t = nn::tensor({{1, 0}, {0, 1}, {1, 1}, {0, 0}});               // [4,2]
  auto         loss_at = [&](const Tensor& W, const Tensor& b) {
    return nn::mean(nn::squared_difference(nn::matmul(x, W) + b, t), {0, 1});
  };

  Tensor W = nn::tensor({{0.1, -0.2}, {0.3, 0.4}, {-0.5, 0.6}}).requires_grad_();
  Tensor b = nn::tensor({0.05, -0.05}).requires_grad_();
  Tensor loss  = loss_at(W, b);
  auto   grads = nn::grad(loss, {W, b});
  std::cout << "  loss    = " << loss.item() << "\n"
            << "  dL/dW   = " << grads[0].values_str() << "\n"
            << "  dL/db   = " << grads[1].values_str() << "\n";

  // Central finite differences on every element of W and b.
  nn::NoGradGuard guard;
  const float     eps     = 1e-2f;
  float           max_err = 0;
  auto            perturb = [&](const Tensor& base, std::int64_t i, float delta) {
    Tensor p = base.detach().clone();
    p.data_ptr<float>()[i] += delta;
    return p;
  };
  for (std::int64_t i = 0; i < 6; ++i) {
    const float fd = (loss_at(perturb(W, i, eps), b).item<float>() - loss_at(perturb(W, i, -eps), b).item<float>()) /
                     (2 * eps);
    max_err = std::max(max_err, std::abs(fd - grads[0].data_ptr<float>()[i]));
  }
  for (std::int64_t i = 0; i < 2; ++i) {
    const float fd = (loss_at(W, perturb(b, i, eps)).item<float>() - loss_at(W, perturb(b, i, -eps)).item<float>()) /
                     (2 * eps);
    max_err = std::max(max_err, std::abs(fd - grads[1].data_ptr<float>()[i]));
  }
  std::cout << "  max |analytic - finite difference| = " << max_err << "\n";
  check::Require(max_err < 1e-3f, "float32 finite differences agree");

  // The same check in float64 with nn::gradcheck, to tight tolerance.
  const auto r = nn::gradcheck(
      [&](const std::vector<Tensor>& in) {
        return nn::mean(nn::squared_difference(nn::matmul(nn::cast(x, DType::Float64), in[0]) + in[1],
                                               nn::cast(t, DType::Float64)));
      },
      {nn::cast(W.detach(), DType::Float64).requires_grad_(), nn::cast(b.detach(), DType::Float64).requires_grad_()});
  std::cout << "  nn::gradcheck (float64): ok=" << r.ok << ", max error " << r.max_abs_error << "\n";
  check::Require(r.ok, "gradcheck");
}

// ---------------------------------------------------------------------------
// 3. Vector-Jacobian products and the full Jacobian
// ---------------------------------------------------------------------------

void Jacobian() {
  Header("Vector-Jacobian products: y = [x0*x1, x0+x1, sin(x0)]");
  Tensor       x  = nn::tensor({2, 3}).requires_grad_();
  const Tensor x0 = nn::select(x, 0, 0), x1 = nn::select(x, 0, 1);
  const Tensor y  = nn::stack({x0 * x1, x0 + x1, nn::sin(x0)});

  // An explicit seed v gives the row v^T J; one-hot seeds give J row by row.
  std::cout << "  at x = [2, 3], J rows via one-hot seeds:\n";
  for (int row = 0; row < 3; ++row) {
    nn::GradOptions retain;
    retain.retain_graph = true;  // walk the same graph three times
    const Tensor seed   = nn::one_hot(nn::scalar(row, DType::Int64), 3);
    const auto   vjp    = nn::grad({y}, {x}, {seed}, retain);
    std::cout << "    dy" << row << "/dx = " << vjp[0].values_str() << "\n";
  }
  std::cout << "  expected: [3 2], [1 1], [cos(2)=" << std::cos(2.f) << " 0]\n";
}

// ---------------------------------------------------------------------------
// 4. StopGradient
// ---------------------------------------------------------------------------

void StopGradient() {
  Header("StopGradient");
  Tensor x = nn::scalar(3.f).requires_grad_();
  // z = x * x, but the second factor is treated as a constant: dz/dx = x, not 2x.
  const double dz      = nn::grad(x * nn::stop_gradient(x), {x})[0].item();
  const double dz_full = nn::grad(x * x, {x})[0].item();
  std::cout << "  d(x*stop(x))/dx at 3 = " << dz << "   d(x*x)/dx at 3 = " << dz_full << "\n";
  check::Near(dz, 3, 1e-6, "stop_gradient");
  check::Near(dz_full, 6, 1e-6, "full gradient");
}

// ---------------------------------------------------------------------------
// 5. Gradients feeding an update: fit y = 3x - 1
// ---------------------------------------------------------------------------

void GradientDescent() {
  Header("Gradient descent: fit y = 3x - 1");
  const Tensor x = nn::tensor({-2, -1, 0, 1, 2, 3});
  const Tensor t = nn::tensor({-7, -4, -1, 2, 5, 8});
  Tensor       w = nn::scalar(0.f).requires_grad_();
  Tensor       c = nn::scalar(0.f).requires_grad_();
  const float  lr = 0.05f;
  for (int step = 0; step <= 100; ++step) {
    const Tensor loss = nn::mean(nn::square(x * w + c - t), {0});
    if (step % 20 == 0) std::printf("  step %3d  loss %.5f  w %.4f  c %.4f\n", step, loss.item(), w.item(), c.item());
    if (step == 100) {
      check::Near(w.item(), 3, 0.05, "slope");
      check::Near(c.item(), -1, 0.15, "intercept");
      break;
    }
    const auto grads = nn::grad(loss, {w, c});
    // ApplyGradientDescent: an in-place update outside the recorded graph.
    nn::NoGradGuard guard;
    w.sub_(grads[0], lr);
    c.sub_(grads[1], lr);
  }
}

}  // namespace

int main() {
  try {
    ScalarDerivatives();
    MatrixGradients();
    Jacobian();
    StopGradient();
    GradientDescent();
    std::cout << "\nAll autodiff checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
