#pragma once

/// @file differential_privacy.hpp
/// @brief Per-example gradient norms of dense layers without per-example backward passes, clipping factors,
///        and a Rényi-differential-privacy accountant for DP-SGD.
///
/// DP-SGD (Abadi et al. 2016) clips each example's gradient to norm at most C, sums, adds Gaussian noise of
/// standard deviation σC and takes a step. The expensive part is the per-example norm. For a dense layer
/// y = xW + b the gradient of example i is \f$X_i^\top\Delta_i\f$, whose norm follows from the layer's input and
/// output gradient alone (Goodfellow 2015), so one ordinary backward pass of the summed loss gives every norm;
/// a second backward pass of the loss reweighted by clip_factors() then gives the clipped sum.
/// dp_sgd_epsilon() turns (sampling rate, noise multiplier, steps) into an (ε, δ) guarantee through the Rényi
/// divergences of the Poisson-subsampled Gaussian mechanism at integer orders (Mironov, Talwar and Zhang 2019).

#include "datamunge/nn/tensor.hpp"

#include <cstdint>

namespace datamunge::nn {

/// @brief Norm of each example's gradient with respect to a dense layer's weight (and bias), from the layer's
///        input and the gradient of the loss with respect to its pre-activation output.
///
/// For x [N, ..., in] and dy [N, ..., out] (the same middle axes, flattened to T positions per example) the weight
/// gradient of example i is \f$G_i=X_i^\top\Delta_i\f$ with \f$X_i\f$ [T, in] and \f$\Delta_i\f$ [T, out]. For T = 1,
/// \f$\|G_i\|_F=\|x_i\|\,\|\delta_i\|\f$. Otherwise \f$\|G_i\|_F^2=\langle X_iX_i^\top,\Delta_i\Delta_i^\top\rangle\f$ (two T x T Gram
/// matrices), or \f$G_i\f$ itself when in·out is smaller than T². With @p bias the bias gradient
/// \f$\sum_t\delta_{i,t}\f$ is included. dy must be the gradient of a sum of per-example losses, so that its row i depends on
/// example i alone; that is what makes these the per-example gradients.
///
/// @param x  Layer input [N, ..., in].
/// @param dy  Gradient of the summed loss with respect to the layer's output x W + b, [N, ..., out].
/// @param bias  Include the bias gradient.
/// @return [N] float64 norms.
/// @throws std::invalid_argument  If the ranks are below 2 or the leading axes differ.
[[nodiscard]] Tensor dense_per_example_grad_norms(const Tensor& x, const Tensor& dy, bool bias = true);

/// @brief DP-SGD clipping factors \f$\min(1,C/\|g_i\|)\f$: weights that scale each example's gradient to norm at most C.
/// @param norms  Per-example gradient norms [N] (summed over every layer as \f$\sqrt{\sum_l\|g_{i,l}\|^2}\f$).
/// @param max_norm  Clipping norm C > 0.
/// @return [N] factors in (0, 1], same dtype as @p norms; a zero norm gives 1.
/// @throws std::invalid_argument  If max_norm is not positive.
[[nodiscard]] Tensor clip_factors(const Tensor& norms, double max_norm);

/// @brief Rényi divergence of integer order α of the Poisson-subsampled Gaussian mechanism with sampling rate q and
///        noise multiplier σ (sensitivity 1).
///
/// \f[ \varepsilon(\alpha)=\frac1{\alpha-1}\log\sum_{k=0}^{\alpha}\binom\alpha k(1-q)^{\alpha-k}q^k\exp\Big(\frac{k^2-k}{2\sigma^2}\Big), \f]
/// the divergence \f$D_\alpha\big((1-q)\mathcal N(0,\sigma^2)+q\mathcal N(1,\sigma^2)\,\big\|\,\mathcal N(0,\sigma^2)\big)\f$, computed in log space. For q = 1 it is
/// \f$\alpha/(2\sigma^2)\f$, the plain Gaussian mechanism.
///
/// @param q  Sampling rate in [0, 1].
/// @param noise_multiplier  σ > 0, the noise standard deviation in units of the clipping norm.
/// @param order  Integer order α >= 2.
/// @return ε(α) for one step.
/// @throws std::invalid_argument  If q is outside [0, 1], σ <= 0 or α < 2.
[[nodiscard]] double rdp_subsampled_gaussian(double q, double noise_multiplier, std::int64_t order);

/// @brief An (ε, δ) guarantee and the Rényi order that gave it.
struct DpEpsilon {
    /// Privacy loss ε.
    double epsilon{0};
    /// The integer order α at which the conversion was tightest.
    std::int64_t order{0};
};

/// @brief ε of DP-SGD after @p steps steps with sampling rate q and noise multiplier σ, at failure probability δ.
///
/// Rényi divergences add over steps, and an order-α guarantee converts to \f$\varepsilon=T\varepsilon(\alpha)+\log(1/\delta)/(\alpha-1)\f$
/// (Mironov 2017); the result is the minimum over α = 2, ..., @p max_order. The conversion is the classical one, slightly
/// looser than later refinements.
///
/// @param q  Sampling rate (expected batch size / dataset size).
/// @param noise_multiplier  σ > 0.
/// @param steps  Number of steps T >= 0.
/// @param delta  δ in (0, 1).
/// @param max_order  Largest order tried (default 256).
/// @return ε and the best order.
/// @throws std::invalid_argument  On out-of-range arguments.
[[nodiscard]] DpEpsilon dp_sgd_epsilon(double q, double noise_multiplier, std::int64_t steps, double delta, std::int64_t max_order = 256);

}  // namespace datamunge::nn
