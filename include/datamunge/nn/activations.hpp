#pragma once

/// @file activations.hpp
/// @brief Gated linear units and the Gaussian moments of an activation.
///
/// The elementwise activations themselves are nn::Activation and nn::activate() (layers.hpp).
/// This header adds the gated family used in transformer feed-forward blocks (GLU, SwiGLU, GeGLU,
/// ReGLU) and activation_moments(), which gives the numbers that decide how an activation passes a
/// Gaussian signal and its gradient from layer to layer.

#include "datamunge/nn/layers.hpp"

#include <cstdint>

namespace datamunge::nn {

/// @brief A gated linear unit: splits @p x in two halves a, b along @p axis and returns
///        \f$a\odot g(b)\f$ (Dauphin et al. 2017; Shazeer 2020).
///
/// `gate` Sigmoid gives GLU, Silu SwiGLU, Gelu GeGLU and Relu ReGLU. The output has half the length of @p x
/// along @p axis. Differentiable through both halves.
///
/// @param x  Input; its length along @p axis must be even.
/// @param gate  Activation applied to the second half (default Sigmoid).
/// @param axis  Axis to split; negative counts from the end (default -1).
/// @return The gated tensor.
/// @throws std::invalid_argument  If the axis length is odd or the axis is out of range.
[[nodiscard]] Tensor glu(const Tensor& x, Activation gate = Activation::Sigmoid, std::int64_t axis = -1);

/// @brief Moments of an activation f and of its derivative f' under a Gaussian input \f$Z\sim\mathcal N(0,q)\f$.
struct ActivationMoments {
    /// \f$\mathbb E[f(Z)]\f$.
    double mean{0};
    /// \f$\mathbb E[f(Z)^2]\f$: the factor in the forward variance map \f$q_{l+1}=s\,\mathbb E[f(Z)^2]\f$.
    double second_moment{0};
    /// \f$\operatorname{Var}f(Z)=\mathbb E[f^2]-\mathbb E[f]^2\f$.
    double variance{0};
    /// \f$\mathbb E[f'(Z)]\f$.
    double derivative_mean{0};
    /// \f$\mathbb E[f'(Z)^2]\f$: the factor \f$\chi=s\,\mathbb E[f'^2]\f$ by which gradient variance changes per layer.
    double derivative_second_moment{0};
};

/// @brief Computes ActivationMoments by quadrature: the midpoint rule on about 24 000 cells over
///        \f$\pm12\sqrt q\f$, with f' from autograd, in float64.
///
/// When the range reaches past \f$\pm1\f$ (and \f$q\le10^6\f$) the cells are aligned to the integers, so the kinks of the built-in
/// activations (at 0, \f$\pm1\f$, \f$\pm3\f$ and 6) fall on cell boundaries and never bias the derivative moments.
/// @param activation  The activation.
/// @param q  Input variance (default 1).
/// @param alpha  LeakyReLU slope, passed to activate().
/// @return The moments, accurate to about 1e-7 (ReLU's \f$\mathbb E f'(Z)=1/2\f$ is reproduced to 1e-12).
/// @throws std::invalid_argument  If q is not positive and finite.
[[nodiscard]] ActivationMoments activation_moments(Activation activation, double q = 1.0, double alpha = 0.2);

}  // namespace datamunge::nn
