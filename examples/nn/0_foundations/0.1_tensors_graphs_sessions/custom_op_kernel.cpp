// CMD: just run-nn custom_op_kernel
// Custom op kernel written in C++.
//
// Port of tflow's custom_op_kernel. In TF an op has two halves registered by
// static initializers: the op definition (inputs, outputs, attrs, shape
// function; REGISTER_OP) and per-device/dtype kernels
// (REGISTER_KERNEL_BUILDER); a Scope wrapper adds it to graphs and
// REGISTER_GRADIENT_OP makes it differentiable. datamunge::nn has no op
// registry (the plan lists TF's custom-op registry as covered by
// nn::Function): a custom op is an nn::Function pairing a forward kernel with
// its backward. The pieces map as follows:
//   op definition / attrs   -> an OpDef-like struct we print, and the attrs
//                              captured by value when the Function is built
//   shape function          -> validation at the top of the forward kernel
//                              (it runs eagerly, so "construction time" and
//                              "run time" are the same moment)
//   kernel per dtype        -> one templated loop dispatched on the dtype,
//                              parallelised with nn::parallel_for (TF: Eigen on
//                              the session's thread pool)
//   gradient registration   -> the Function's backward, checked with
//                              gradcheck/gradgradcheck in float64
// The printed values (ScaleShift, RowSums, the gradient) are identical to TF.

#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using Tensors = std::vector<Tensor>;

namespace {

// ---------------------------------------------------------------------------
// 1. Op definitions (descriptive: there is no registry to look them up in)
// ---------------------------------------------------------------------------

struct OpDef {
  std::string name, signature, doc;
};
const OpDef kOpDefs[] = {
    {"ScaleShift", "ScaleShift(x: T) -> (y: T); attr T: {float32, float64}; attr scale: float = 1.0; attr shift: float = 0.0",
     "Computes scale * x + shift."},
    {"RowSums", "RowSums(x: float32 [n, m]) -> (y: float32 [n])", "Sums each row of a matrix."},
};

// ---------------------------------------------------------------------------
// 2. Kernels
// ---------------------------------------------------------------------------

template <typename T>
void ScaleShiftKernel(const Tensor& x, Tensor& y, double scale, double shift) {
  const T* in  = x.data_ptr<T>();
  T*       out = y.data_ptr<T>();
  // Chunks of the flat range run on the intra-op thread pool.
  nn::parallel_for(0, x.numel(), 1 << 14, [&](std::int64_t lo, std::int64_t hi) {
    for (std::int64_t i = lo; i < hi; ++i) out[i] = in[i] * static_cast<T>(scale) + static_cast<T>(shift);
  });
}

Tensor ScaleShiftForward(const Tensor& x_in, double scale, double shift) {
  // The type constraint of the op definition: T in {float32, float64}.
  if (x_in.dtype() != nn::DType::Float32 && x_in.dtype() != nn::DType::Float64)
    throw std::invalid_argument("ScaleShift: Value for attr 'T' of " + nn::dtype_name(x_in.dtype()) +
                                " is not in the list of allowed values: float32, float64");
  const Tensor x = x_in.contiguous();
  Tensor       y = nn::empty(x.shape(), x.dtype());  // UnchangedShape
  if (x.dtype() == nn::DType::Float32)
    ScaleShiftKernel<float>(x, y, scale, shift);
  else
    ScaleShiftKernel<double>(x, y, scale, shift);
  return y;
}

Tensor RowSumsForward(const Tensor& x_in) {
  // The shape function: rank 2 in, [rows] out.
  if (x_in.dim() != 2)
    throw std::invalid_argument("RowSums: Shape must be rank 2 but is rank " + std::to_string(x_in.dim()) + " for input " +
                                nn::shape_str(x_in.shape()));
  if (x_in.dtype() != nn::DType::Float32)
    throw std::invalid_argument("RowSums: expected float32, got " + nn::dtype_name(x_in.dtype()));
  const Tensor       x    = x_in.contiguous();
  const std::int64_t rows = x.shape()[0], cols = x.shape()[1];
  Tensor             y    = nn::empty({rows}, nn::DType::Float32);
  const float*       xm   = x.data_ptr<float>();
  float*             yv   = y.data_ptr<float>();
  for (std::int64_t i = 0; i < rows; ++i) {
    float s = 0;
    for (std::int64_t j = 0; j < cols; ++j) s += xm[i * cols + j];
    yv[i] = s;
  }
  return y;
}

// ---------------------------------------------------------------------------
// 3. The ops as callables (TF: the Scope wrappers), with
// 4. their gradients: d(scale*x+shift)/dx = scale; d rowsum/dx = broadcast.
// ---------------------------------------------------------------------------

// Attrs are fixed when the op is built, as TF reads them at kernel construction.
nn::Function ScaleShift(double scale, double shift) {
  return nn::Function(
      "ScaleShift", [scale, shift](const Tensors& in) { return Tensors{ScaleShiftForward(in[0], scale, shift)}; },
      [scale](const Tensors&, const Tensors&, const Tensors& grad) { return Tensors{nn::mul(grad[0], scale)}; });
}

const nn::Function RowSums(
    "RowSums", [](const Tensors& in) { return Tensors{RowSumsForward(in[0])}; },
    [](const Tensors& in, const Tensors&, const Tensors& grad) {
      return Tensors{nn::broadcast_to(nn::expand_dims(grad[0], 1), in[0].shape())};
    });

}  // namespace

int main() {
  try {
    Header("Op definitions");
    for (const OpDef& d : kOpDefs) std::cout << "  " << d.signature << "\n      " << d.doc << "\n";

    Header("Using the ops");
    Tensor       x    = nn::tensor({{1, 2, 3}, {4, 5, 6}}).requires_grad_();
    const Tensor xd   = nn::cast(x.detach(), nn::DType::Float64);
    const Tensor y    = ScaleShift(2.0, 0.5)(x);
    const Tensor yd   = ScaleShift(0.5, -1.0)(xd);  // the double kernel
    const Tensor sums = RowSums(y);
    // Our ops compose with the built-in ones, including in gradients.
    const Tensor loss = nn::sum(nn::square(y));
    const Tensor g    = nn::grad(loss, {x})[0];
    std::cout << "  ScaleShift(x,2,0.5)   = " << y.values_str() << "\n"
              << "  ScaleShift(xd,0.5,-1) = " << yd.values_str() << " (" << nn::dtype_name(yd.dtype()) << ")\n"
              << "  RowSums(y)            = " << sums.values_str() << "\n"
              << "  d sum(y^2) / dx       = " << g.values_str() << "  (= 2*y*scale)\n";
    check::Require(y.to_vector<float>() == std::vector<float>{2.5f, 4.5f, 6.5f, 8.5f, 10.5f, 12.5f}, "ScaleShift float");
    check::Require(yd.dtype() == nn::DType::Float64 &&
                       yd.to_vector<double>() == std::vector<double>{-0.5, 0, 0.5, 1, 1.5, 2},
                   "ScaleShift double");
    check::Require(sums.to_vector<float>() == std::vector<float>{13.5f, 31.5f}, "RowSums");
    check::Require(g.to_vector<float>() == std::vector<float>{10, 18, 26, 34, 42, 50}, "gradient through ScaleShift");

    // The registered gradients, verified numerically (float64), to second order.
    Tensor     x64 = nn::cast(x.detach(), nn::DType::Float64).requires_grad_();
    const auto f   = [](const Tensors& in) { return ScaleShift(2.0, 0.5)(nn::square(in[0])); };
    const auto r1  = nn::gradcheck(f, {x64});
    const auto r2  = nn::gradgradcheck(f, {x64});
    std::cout << "  ScaleShift gradcheck ok=" << r1.ok << " gradgradcheck ok=" << r2.ok << "\n";
    check::Require(r1.ok && r2.ok, "ScaleShift gradient checks");
    // RowSums is float32-only, as in TF: check its gradient against the closed form.
    const Tensor gr = nn::grad(nn::sum(nn::square(RowSums(x))), {x})[0];
    std::cout << "  d sum(RowSums(x)^2) / dx = " << gr.values_str() << "\n";
    check::Require(gr.to_vector<float>() == std::vector<float>{12, 12, 12, 30, 30, 30}, "gradient through RowSums");

    Header("Errors caught when the op is called");
    std::cout << "  RowSums(rank-1): "
              << check::Throws([] { (void)RowSums(nn::tensor({1, 2, 3})); }, "RowSums rank 1") << "\n";
    std::cout << "  ScaleShift(int32): "
              << check::Throws([] { (void)ScaleShift(1.0, 0.0)(nn::tensor({1, 2, 3}, {}, nn::DType::Int32)); },
                               "ScaleShift int32")
              << "\n";

    // The kernel's parallel_for gives the same result at any thread count.
    nn::Generator g1(3);
    const Tensor  big = nn::normal({1 << 18}, 0.0, 1.0, g1);
    nn::set_num_threads(1);
    const Tensor serial = ScaleShift(3.0, -1.0)(big);
    nn::set_num_threads(0);
    const Tensor threaded = ScaleShift(3.0, -1.0)(big);
    std::cout << "  " << big.numel() << " elements, 1 thread vs " << nn::get_num_threads()
              << " threads: identical = " << (serial.to_vector<float>() == threaded.to_vector<float>()) << "\n";
    check::Require(serial.to_vector<float>() == threaded.to_vector<float>(), "kernel is thread-count independent");

    std::cout << "\nAll custom op checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
