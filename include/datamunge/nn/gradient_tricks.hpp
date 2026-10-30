#pragma once

/// @file gradient_tricks.hpp
/// @brief Functions whose backward pass is deliberately not the derivative of their forward pass:
///        straight-through estimators, fake quantisation, gradient reversal, clipping and scaling.
///
/// Each is an nn::Function (a forward paired with a hand-written backward), or the
/// stop-gradient identity \f$x+\mathrm{stop}(v-x)\f$, which evaluates to \f$v\f$ but differentiates as \f$x\f$.
/// They are the standard tools for training through non-differentiable steps
/// (rounding, quantisation, sampling) and for adversarial objectives (domain-adversarial
/// training). In every case the forward value is exact and only the gradient is altered.

#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// @brief A tensor equal to `value` whose gradient flows to `x` as if the result were `x`.
///
/// Computes \f$x+\mathrm{stop}(v-x)\f$. The forward value is `value`, up to the rounding of the sum and
/// difference, and the gradient with respect to `x` is the identity. `value` receives no gradient.
/// @param x  The tensor gradients flow to.
/// @param value  The forward value, shaped like `x` (typically a non-differentiable function of x).
/// @return The straight-through tensor.
/// @throws std::invalid_argument  If the shapes differ.
[[nodiscard]] Tensor straight_through(const Tensor& x, const Tensor& value);

/// @brief Rounds to the nearest integer (halves to even, like nn::round), with a straight-through
///        gradient: the forward value is exactly round(x), the backward is the identity.
/// @param x  Input.
/// @return round(x).
[[nodiscard]] Tensor round_ste(const Tensor& x);

/// @brief Stochastic rounding with a straight-through gradient.
///
/// Each element becomes \f$\lfloor x\rfloor+B\f$ with \f$B\sim\mathrm{Bernoulli}(x-\lfloor x\rfloor)\f$, so the result is unbiased,
/// \f$\mathbb E[\mathrm{round}_s(x)]=x\f$, and the identity backward is the exact derivative of that expectation.
/// @param x  Input (a floating dtype).
/// @param gen  Random generator.
/// @return The rounded tensor, in x's dtype.
[[nodiscard]] Tensor stochastic_round_ste(const Tensor& x, Generator& gen);

/// @brief Uniform fake quantisation with the clipped straight-through gradient.
///
/// Forward: \f$q(x)=\Delta\cdot\mathrm{clip}\bigl(\mathrm{round}(x/\Delta),\,q_{\min},\,q_{\max}\bigr)\f$, the nearest point of the grid
/// \f$\{\Delta q_{\min},\dots,\Delta q_{\max}\}\f$ (as TensorFlow's FakeQuant does). Backward: the gradient passes
/// unchanged where \f$\Delta q_{\min}\le x\le\Delta q_{\max}\f$ and is zero outside, where the forward is constant.
/// @param x  Input.
/// @param step  Grid spacing \f$\Delta>0\f$.
/// @param qmin  Lowest grid index.
/// @param qmax  Highest grid index (\f$\ge q_{\min}\f$).
/// @return The quantised tensor, in x's dtype.
/// @throws std::invalid_argument  If step <= 0 or qmax < qmin.
[[nodiscard]] Tensor fake_quantize(const Tensor& x, double step, std::int64_t qmin, std::int64_t qmax);

/// @brief Identity in the forward pass; multiplies the gradient by \f$-\lambda\f$ in the backward pass.
///
/// Placed between a feature extractor and an adversary (Ganin and Lempitsky, 2015), it makes one
/// gradient step minimise the adversary's loss in the adversary and maximise it in the features.
/// @param x  Input.
/// @param lambda  Reversal strength (\f$\lambda\ge0\f$ in the usual use).
/// @return A tensor equal to x.
[[nodiscard]] Tensor gradient_reversal(const Tensor& x, double lambda = 1.0);

/// @brief Identity in the forward pass; multiplies the gradient by `factor` in the backward pass.
/// @param x  Input.
/// @param factor  Gradient multiplier.
/// @return A tensor equal to x.
[[nodiscard]] Tensor scale_gradient(const Tensor& x, double factor);

/// @brief Identity in the forward pass; clips every element of the incoming gradient to [low, high].
/// @param x  Input.
/// @param low  Lower bound.
/// @param high  Upper bound (>= low).
/// @return A tensor equal to x.
/// @throws std::invalid_argument  If high < low.
[[nodiscard]] Tensor clip_gradient(const Tensor& x, double low, double high);

}  // namespace datamunge::nn
