/// @file derivatives.cpp
/// @brief vjp(), jvp(), jacobian(), hessian() and hvp() on top of nn::grad.

#include "datamunge/nn/derivatives.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

#include <stdexcept>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief A fresh leaf copy of x that requires grad (the caller's tensor is left alone).
/// @param x  The point.
/// @return The leaf.
Tensor leaf_copy(const Tensor& x) { return x.detach().clone().requires_grad_(); }

/// @brief Throws unless a and b have the same shape.
/// @param a  First tensor.
/// @param b  Second tensor.
/// @param what  Message prefix.
void require_same_shape(const Tensor& a, const Tensor& b, const char* what) {
    if (a.shape() != b.shape()) throw std::invalid_argument(std::string(what) + ": shape mismatch");
}

/// @brief The value of a scalar-valued f, checked to have one element.
/// @param f  The function.
/// @param x  The leaf point.
/// @param what  Message prefix.
/// @return f(x), reshaped to a scalar.
Tensor scalar_value(const UnaryTensorFunction& f, const Tensor& x, const char* what) {
    Tensor y = f(x);
    if (y.numel() != 1) throw std::invalid_argument(std::string(what) + ": f must return one element");
    return reshape(y, {});
}

/// @brief Gradient of the leaf, or zeros when the output does not depend on it.
/// @param out  Output.
/// @param x  Leaf.
/// @param seed  Cotangent (or undefined for a scalar output).
/// @param create  Whether to record the backward pass.
/// @return The gradient, shaped like x.
Tensor grad_or_zeros(const Tensor& out, const Tensor& x, const Tensor& seed, bool create) {
    GradOptions options;
    options.create_graph = create;
    options.allow_unused = true;
    if (!out.requires_grad()) return zeros_like(x);
    auto g = seed.defined() ? grad({out}, {x}, {seed}, options) : grad({out}, {x}, {}, options);
    return g[0].defined() ? g[0] : zeros_like(x);
}

}  // namespace

Tensor vjp(const UnaryTensorFunction& f, const Tensor& x, const Tensor& v) {
    const Tensor leaf = leaf_copy(x);
    const Tensor y    = f(leaf);
    require_same_shape(y, v, "vjp");
    return grad_or_zeros(y, leaf, v.detach(), false);
}

Tensor jvp(const UnaryTensorFunction& f, const Tensor& x, const Tensor& u) {
    require_same_shape(x, u, "jvp");
    const Tensor leaf = leaf_copy(x);
    const Tensor y    = f(leaf);
    // g(v) = J^T v with v a dummy leaf; J u = d/dv <g(v), u>, and g is linear in v.
    const Tensor v = zeros_like(y).requires_grad_();
    const Tensor g = grad_or_zeros(y, leaf, v, true);
    return grad_or_zeros(sum(mul(g, u.detach())), v, Tensor(), false);
}

Tensor jacobian(const UnaryTensorFunction& f, const Tensor& x) {
    const Tensor       leaf = leaf_copy(x);
    const Tensor       y    = f(leaf);
    const std::int64_t m = y.numel(), n = leaf.numel();
    std::vector<Tensor> rows;
    rows.reserve(static_cast<std::size_t>(m));
    GradOptions options;
    options.retain_graph = true;
    options.allow_unused = true;
    for (std::int64_t i = 0; i < m; ++i) {
        if (!y.requires_grad()) {
            rows.push_back(zeros({n}, leaf.dtype()));
            continue;
        }
        std::vector<double> e(static_cast<std::size_t>(m), 0.0);
        e[static_cast<std::size_t>(i)] = 1.0;
        const Tensor seed = from_vector(e, {m}).to(y.dtype()).reshape(y.shape());
        const Tensor g    = grad({y}, {leaf}, {seed}, options)[0];
        rows.push_back(reshape(g.defined() ? g : zeros_like(leaf), {n}));
    }
    return stack(rows, 0);
}

Tensor hessian(const UnaryTensorFunction& f, const Tensor& x) {
    // Validate once, then differentiate the recorded gradient.
    (void)scalar_value(f, leaf_copy(x), "hessian");
    return jacobian(
        [&](const Tensor& z) {
            const Tensor value = scalar_value(f, z, "hessian");
            return grad_or_zeros(value, z, Tensor(), true);
        },
        x);
}

Tensor hvp(const UnaryTensorFunction& f, const Tensor& x, const Tensor& v) {
    require_same_shape(x, v, "hvp");
    const Tensor leaf  = leaf_copy(x);
    const Tensor value = scalar_value(f, leaf, "hvp");
    const Tensor g     = grad_or_zeros(value, leaf, Tensor(), true);
    return grad_or_zeros(sum(mul(g, v.detach())), leaf, Tensor(), false);
}

}  // namespace datamunge::nn
