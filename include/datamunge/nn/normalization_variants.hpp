#pragma once

/// @file normalization_variants.hpp
/// @brief Functional normalisation over any axes, closed-form Jacobians of layer norm and RMSNorm, ScaleNorm,
///        Dynamic Tanh and weight standardisation.
///
/// The layers in layers.hpp (layer_norm(), rms_norm(), group_norm(), instance_norm(), batch_norm()) differ only in
/// the axes their statistics are taken over and in whether they centre. normalize() exposes that single operation
/// without parameters, which is useful as a reference and for custom groupings. layer_norm_jacobian() and
/// rms_norm_jacobian() give the exact Jacobians of the parameter-free maps, whose null spaces explain why the
/// gradient flowing into a normaliser is orthogonal to the constant vector and (nearly) to the centred input.
/// scale_norm() (Nguyen and Salazar 2019) replaces RMSNorm's per-feature gain by one scalar. dynamic_tanh()
/// (Zhu et al. 2025) is a statistics-free elementwise replacement for layer norm. weight_standardize() (Qiao et al.
/// 2019) normalises the weights instead of the activations.

#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/params.hpp"
#include "datamunge/nn/tensor.hpp"

#include <string>

namespace datamunge::nn {

/// @brief Standardises x over the given axes: \f$(x-\mu)/\sqrt{\sigma^2+\epsilon}\f$, or \f$x/\sqrt{\overline{x^2}+\epsilon}\f$ without centring.
///
/// The mean \f$\mu\f$ and the biased variance \f$\sigma^2\f$ (or the mean square) are taken over `axes` separately for every index of
/// the remaining axes. Layer norm is `normalize(x, {-1})`, RMSNorm is `normalize(x, {-1}, false)`, and batch norm in
/// training mode is the normalisation over every axis but the last.
///
/// @param x  Floating input.
/// @param axes  Axes to reduce over (negative values count from the end); must be non-empty.
/// @param center  Subtract the mean; if false, divide by the root mean square instead of the standard deviation.
/// @param eps  Added to the variance (or mean square) before the square root.
/// @return The standardised tensor, same shape and dtype as x. Differentiable.
/// @throws std::invalid_argument  If axes is empty or eps is negative.
[[nodiscard]] Tensor normalize(const Tensor& x, const Axes& axes, bool center = true, double eps = 1e-5);

/// @brief Exact Jacobian of parameter-free layer norm of a vector: \f$J=\frac1\sigma\big(I-\frac1d\mathbf 1\mathbf 1^\top-\frac1d\hat y\hat y^\top\big)\f$.
///
/// Here \f$\sigma=\sqrt{\mathrm{var}(x)+\epsilon}\f$ and \f$\hat y=(x-\bar x)/\sigma\f$. J is symmetric, \f$J\mathbf 1=0\f$ exactly, and
/// \f$J(x-\bar x)=\frac{\epsilon}{\sigma^2}\cdot\frac{x-\bar x}{\sigma}\f$, which is zero when \f$\epsilon=0\f$.
///
/// @param x  Input vector of shape [d], d >= 1.
/// @param eps  Variance epsilon, as in layer_norm().
/// @return J of shape [d, d], with \f$J_{ij}=\partial y_i/\partial x_j\f$, in x's floating dtype.
/// @throws std::invalid_argument  If x is not rank 1 or is empty.
[[nodiscard]] Tensor layer_norm_jacobian(const Tensor& x, double eps = 1e-5);

/// @brief Exact Jacobian of parameter-free RMSNorm of a vector: \f$J=\frac1r\big(I-\frac1d\hat y\hat y^\top\big)\f$.
///
/// Here \f$r=\sqrt{\overline{x^2}+\epsilon}\f$ and \f$\hat y=x/r\f$. J is symmetric and \f$Jx=\frac{\epsilon}{r^2}\hat y\f$.
///
/// @param x  Input vector of shape [d], d >= 1.
/// @param eps  Mean-square epsilon, as in rms_norm().
/// @return J of shape [d, d] in x's floating dtype.
/// @throws std::invalid_argument  If x is not rank 1 or is empty.
[[nodiscard]] Tensor rms_norm_jacobian(const Tensor& x, double eps = 1e-6);

/// @brief ScaleNorm (Nguyen and Salazar 2019): \f$y=g\,x/\sqrt{\lVert x\rVert^2+\epsilon}\f$ over the last axis, with one learned scalar g.
///
/// The parameter "<name>/g" has shape [1] and starts at \f$\sqrt d\f$, so that at initialisation ScaleNorm equals
/// parameter-free RMSNorm with epsilon \f$\epsilon/d\f$.
///
/// @param store  Parameter store.
/// @param name  Parameter prefix.
/// @param x  Input with features on the last axis.
/// @param eps  Added to the squared norm.
/// @return The output, same shape as x.
/// @throws std::invalid_argument  If x is a scalar.
[[nodiscard]] Tensor scale_norm(ParameterStore& store, const std::string& name, const Tensor& x, double eps = 1e-6);

/// @brief Dynamic Tanh (Zhu et al. 2025): \f$y=\gamma\odot\tanh(\alpha x)+\beta\f$, a normalisation-free replacement for layer norm.
///
/// \f$\alpha\f$ is one learned scalar "<name>/alpha" (shape [1], initialised to `alpha0`); "<name>/gamma" (ones) and
/// "<name>/beta" (zeros) have the size of the last axis. No statistics are computed, so the output of each element
/// depends on that element only.
///
/// @param store  Parameter store.
/// @param name  Parameter prefix.
/// @param x  Input with features on the last axis.
/// @param alpha0  Initial value of alpha.
/// @return The output, same shape as x.
/// @throws std::invalid_argument  If x is a scalar.
[[nodiscard]] Tensor dynamic_tanh(ParameterStore& store, const std::string& name, const Tensor& x,
                                  double alpha0 = 0.5);

/// @brief Weight standardisation (Qiao et al. 2019): each output unit's weights get mean 0 and variance 1.
///
/// For a weight with output units on the last axis (dense [in, out] or convolution [H, W, in, out]) the statistics
/// are taken over every other axis: `normalize(w, {0, ..., rank - 2}, true, eps)`. Apply it to the stored weight
/// before use; gradients flow back to the raw weight.
///
/// @param w  Weight of rank >= 2, output units last.
/// @param eps  Variance epsilon.
/// @return The standardised weight, same shape.
/// @throws std::invalid_argument  If w has rank < 2.
[[nodiscard]] Tensor weight_standardize(const Tensor& w, double eps = 1e-5);

}  // namespace datamunge::nn
