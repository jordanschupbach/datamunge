#pragma once

/// @file derivatives.hpp
/// @brief Derivative operators built on reverse-mode autograd: vector-Jacobian and
///        Jacobian-vector products, full Jacobians, Hessians and Hessian-vector products.
///
/// For \f$f:\mathbb R^n\to\mathbb R^m\f$ with Jacobian \f$J\f$ at \f$x\f$, one backward pass gives
/// \f$v^\top J\f$ for any \f$v\f$ (vjp()). The Jacobian therefore takes \f$m\f$ backward passes
/// (jacobian()), and the gradient of a scalar takes one. A forward-mode product \f$Ju\f$ is
/// obtained from two backward passes by the double-backward identity
/// \f$Ju=\nabla_v\langle J^\top v,u\rangle\f$ (jvp()). Second derivatives differentiate the recorded
/// backward pass (hessian(), hvp()). Inputs and outputs may have any shape; matrices are
/// laid out over the flattened elements.

#include "datamunge/nn/tensor.hpp"

#include <functional>

namespace datamunge::nn {

/// @brief A differentiable function of one tensor.
using UnaryTensorFunction = std::function<Tensor(const Tensor&)>;

/// @brief The vector-Jacobian product \f$v^\top J_f(x)\f$, by one backward pass.
/// @param f  The function.
/// @param x  The point; a floating tensor (its requires_grad flag is not changed).
/// @param v  Cotangent, shaped like f(x).
/// @return A tensor shaped like x.
/// @throws std::invalid_argument  If v's shape differs from f(x)'s.
[[nodiscard]] Tensor vjp(const UnaryTensorFunction& f, const Tensor& x, const Tensor& v);

/// @brief The Jacobian-vector product \f$J_f(x)\,u\f$ (a directional derivative), by two
///        backward passes: \f$g(v)=J^\top v\f$ is recorded, then \f$\nabla_v\langle g(v),u\rangle=Ju\f$.
/// @param f  The function.
/// @param x  The point.
/// @param u  Tangent, shaped like x.
/// @return A tensor shaped like f(x).
/// @throws std::invalid_argument  If u's shape differs from x's.
[[nodiscard]] Tensor jvp(const UnaryTensorFunction& f, const Tensor& x, const Tensor& u);

/// @brief The full Jacobian, one vector-Jacobian product per output element.
/// @param f  The function.
/// @param x  The point.
/// @return A [f(x).numel(), x.numel()] tensor in x's dtype; row i is the gradient of output i.
[[nodiscard]] Tensor jacobian(const UnaryTensorFunction& f, const Tensor& x);

/// @brief The Hessian \f$\nabla^2f(x)\f$ of a scalar function, as the Jacobian of its gradient.
/// @param f  A function returning a one-element tensor.
/// @param x  The point.
/// @return A symmetric [x.numel(), x.numel()] tensor.
/// @throws std::invalid_argument  If f(x) has more than one element.
[[nodiscard]] Tensor hessian(const UnaryTensorFunction& f, const Tensor& x);

/// @brief The Hessian-vector product \f$\nabla^2f(x)\,v\f$ without forming the Hessian: the gradient
///        of \f$\langle\nabla f(x),v\rangle\f$, two backward passes.
/// @param f  A function returning a one-element tensor.
/// @param x  The point.
/// @param v  Direction, shaped like x.
/// @return A tensor shaped like x.
/// @throws std::invalid_argument  If f(x) has more than one element or v's shape differs from x's.
[[nodiscard]] Tensor hvp(const UnaryTensorFunction& f, const Tensor& x, const Tensor& v);

}  // namespace datamunge::nn
