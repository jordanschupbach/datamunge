#pragma once

/// @file dropout_variants.hpp
/// @brief Gaussian dropout, alpha dropout for self-normalising networks, spatial (channel) dropout and drop path
///        (stochastic depth).
///
/// nn::dropout() (layers.hpp) zeroes each element independently with probability p and scales the survivors by
/// 1/(1 - p). The variants here change what is dropped or what replaces it. gaussian_dropout() multiplies by
/// continuous noise with the same mean and variance as the dropout mask (Srivastava et al. 2014).
/// alpha_dropout() sets dropped units to SELU's negative saturation value and applies an affine correction, so that
/// zero-mean, unit-variance inputs keep those moments (Klambauer et al. 2017). spatial_dropout() drops whole
/// channels, sharing one mask over the positions of an example (Tompson et al. 2015). drop_path() drops a whole
/// residual branch per example (Huang et al. 2016). Every function is the identity when `training` is false and
/// is differentiable with respect to x.

#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// @brief Gaussian (multiplicative) dropout: \f$y=x\odot\xi\f$ with \f$\xi\sim\mathcal N\big(1,\,p/(1-p)\big)\f$ while training.
///
/// The noise has the mean and variance of inverted dropout's mask \f$m/(1-p)\f$, \f$m\sim\mathrm{Bernoulli}(1-p)\f$.
///
/// @param x  Input.
/// @param rate  Dropout rate p in [0, 1).
/// @param training  Apply noise; otherwise return x.
/// @param gen  Random stream.
/// @return The noisy input, same shape and dtype.
/// @throws std::invalid_argument  If rate is outside [0, 1).
[[nodiscard]] Tensor gaussian_dropout(const Tensor& x, double rate, bool training, Generator& gen);

/// @brief Alpha dropout (Klambauer et al. 2017): dropped units take SELU's saturation value
///        \f$\alpha'=-\lambda\alpha\approx-1.7581\f$, then \f$y=a\,(x\odot m+\alpha'(1-m))+b\f$.
///
/// With keep probability \f$k=1-p\f$, \f$a=\big(k(1+p\,\alpha'^2)\big)^{-1/2}\f$ and \f$b=-a\,p\,\alpha'\f$, chosen so that an input with mean 0
/// and variance 1 keeps mean 0 and variance 1. Use with SELU and LeCun-normal weights.
///
/// @param x  Input.
/// @param rate  Dropout rate p in [0, 1).
/// @param training  Apply dropout; otherwise return x.
/// @param gen  Random stream.
/// @return The output, same shape and dtype.
/// @throws std::invalid_argument  If rate is outside [0, 1).
[[nodiscard]] Tensor alpha_dropout(const Tensor& x, double rate, bool training, Generator& gen);

/// @brief Spatial dropout: inverted dropout with one mask per (example, channel), shared over every axis in between.
///
/// For x [N, ..., C] (for example NHWC images or [N, T, C] sequences) whole feature maps are dropped, which
/// regularises when neighbouring positions are strongly correlated and elementwise dropout removes little information.
///
/// @param x  Input of rank >= 2, channels last.
/// @param rate  Dropout rate p in [0, 1).
/// @param training  Apply dropout; otherwise return x.
/// @param gen  Random stream.
/// @return The output, same shape and dtype.
/// @throws std::invalid_argument  If rate is outside [0, 1) or x has rank < 2.
[[nodiscard]] Tensor spatial_dropout(const Tensor& x, double rate, bool training, Generator& gen);

/// @brief Drop path (stochastic depth): zeroes all of example i with probability p and scales the rest by 1/(1 - p).
///
/// Apply it to a residual branch, h + drop_path(f(h)), so that each example skips the branch at random
/// (Huang et al. 2016). One mask value per index of the first axis.
///
/// @param x  Input [N, ...].
/// @param rate  Drop rate p in [0, 1).
/// @param training  Apply dropping; otherwise return x.
/// @param gen  Random stream.
/// @return The output, same shape and dtype.
/// @throws std::invalid_argument  If rate is outside [0, 1) or x is a scalar.
[[nodiscard]] Tensor drop_path(const Tensor& x, double rate, bool training, Generator& gen);

}  // namespace datamunge::nn
