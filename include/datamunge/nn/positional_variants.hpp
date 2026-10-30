#pragma once

/// @file positional_variants.hpp
/// @brief RoPE frequency scaling for longer contexts (position interpolation and NTK-aware), rotary embedding at
///        arbitrary real positions, 2-D sinusoidal tables for image patches, and Fourier-feature encodings of
///        continuous coordinates.
///
/// attention.hpp provides the standard sequence encodings (sinusoidal_encoding(), learned_encoding(),
/// apply_rotary(), alibi_bias(), relative_position_bias()). This header adds the pieces needed to stretch RoPE past
/// its training length and to encode positions that are not token indices. rope_inverse_frequencies() returns the
/// per-pair rotation rates with optional position interpolation (Chen et al. 2023), which divides every rate by the
/// extension factor, or NTK-aware scaling, which enlarges the base so that low frequencies stretch and high ones
/// barely change. apply_rotary_at() rotates by those rates at any real positions. sinusoidal_encoding_2d() is the
/// fixed 2-D table used for vision transformer patches (He et al. 2022). fourier_features() is NeRF's octave encoding
/// (Mildenhall et al. 2020), and gaussian_fourier_features() the random Fourier features of Tancik et al. (2020),
/// which let a coordinate MLP fit high-frequency functions.

#include "datamunge/nn/tensor.hpp"

#include <cstdint>
#include <vector>

namespace datamunge::nn {

/// @brief How rope_inverse_frequencies() stretches RoPE to a longer context.
enum class RopeScaling {
  None,                   ///< \f$\theta_j=b^{-2j/d}\f$.
  PositionInterpolation,  ///< \f$\theta_j=b^{-2j/d}/s\f$: positions are compressed by the factor s (Chen et al. 2023).
  Ntk                     ///< \f$\theta_j=(b\,s^{d/(d-2)})^{-2j/d}\f$: the base grows so the lowest frequency is divided by s and the highest is unchanged.
};

/// @brief Inverse frequencies \f$\theta_j\f$, j = 0, ..., d/2 - 1, of rotary embeddings, optionally scaled for a context s times longer.
///
/// With RopeScaling::None these are the rates used by apply_rotary(). Position interpolation divides all of them by
/// s, so position \f$s\,t\f$ gets the angles that position t had. NTK-aware scaling replaces the base b by \f$b\,s^{d/(d-2)}\f$, which
/// leaves \f$\theta_0=1\f$ unchanged and divides \f$\theta_{d/2-1}\f$ by exactly s.
///
/// @param d  Head dimension, even and >= 2 (>= 4 for RopeScaling::Ntk).
/// @param base  RoPE base b > 0.
/// @param scaling  Scaling rule.
/// @param factor  Context extension factor s >= 1 (ignored for RopeScaling::None).
/// @return The d/2 inverse frequencies.
/// @throws std::invalid_argument  For odd or too small d, base <= 0 or factor < 1.
[[nodiscard]] std::vector<double> rope_inverse_frequencies(std::int64_t d, double base = 10000.0,
                                                           RopeScaling scaling = RopeScaling::None,
                                                           double factor = 1.0);

/// @brief Rotary embedding at real positions: pair (2j, 2j+1) of row t is rotated by the angle \f$p_t\theta_j\f$.
///
/// This is apply_rotary()'s interleaved layout with explicit positions and rates, so it supports fractional positions
/// and the scaled rates of rope_inverse_frequencies(). Differentiable with respect to x.
///
/// @param x  Input [..., T, d].
/// @param positions  T positions.
/// @param inv_freq  d/2 rates \f$\theta_j\f$.
/// @return The rotated tensor, same shape as x.
/// @throws std::invalid_argument  If the sizes do not match x.
[[nodiscard]] Tensor apply_rotary_at(const Tensor& x, const std::vector<double>& positions,
                                     const std::vector<double>& inv_freq);

/// @brief Fixed 2-D sinusoidal table [H W, d] for a grid of patches: the first d/2 columns encode the row, the last d/2 the column.
///
/// Row r of the grid position (r, c) uses sinusoidal_encoding(H, d/2, base) at r and column c uses the same table at
/// c, so the dot product between two grid positions is a function of their row offset plus a function of their column
/// offset. Rows of the result are ordered r W + c.
///
/// @param H  Grid height >= 1.
/// @param W  Grid width >= 1.
/// @param d  Encoding size, a multiple of 4.
/// @param base  Wavelength base.
/// @param dtype  Floating dtype of the result.
/// @return The table [H W, d].
/// @throws std::invalid_argument  If d is not a positive multiple of 4 or H, W < 1.
[[nodiscard]] Tensor sinusoidal_encoding_2d(std::int64_t H, std::int64_t W, std::int64_t d, double base = 10000.0,
                                            DType dtype = DType::Float64);

/// @brief NeRF's positional encoding of coordinates: \f$(x,\ \sin(2^k\pi x),\ \cos(2^k\pi x))_{k=0}^{L-1}\f$ applied to each coordinate.
///
/// For x [..., D] the output is [..., D (2L + 1)] with `include_input`, or [..., 2 D L] without, in the order x, then
/// sin and cos of each octave k = 0, ..., L - 1 (each block holding all D coordinates). Coordinates are usually
/// scaled to [-1, 1] first.
///
/// @param x  Coordinates [..., D].
/// @param num_frequencies  Number of octaves L >= 1.
/// @param include_input  Prepend x itself.
/// @return The encoded coordinates.
/// @throws std::invalid_argument  If num_frequencies < 1 or x is a scalar.
[[nodiscard]] Tensor fourier_features(const Tensor& x, std::int64_t num_frequencies, bool include_input = true);

/// @brief Gaussian random Fourier features (Tancik et al. 2020): \f$\gamma(x)=[\cos(2\pi xB),\ \sin(2\pi xB)]\f$.
///
/// With the entries of B drawn \f$\mathcal N(0,\sigma^2)\f$, \f$\gamma(x)\cdot\gamma(x')=\sum_j\cos(2\pi b_j\cdot(x-x'))\f$, whose expectation is the Gaussian kernel
/// \f$m\,e^{-2\pi^2\sigma^2\lVert x-x'\rVert^2}\f$ (Rahimi and Recht 2007); \f$\sigma\f$ sets the bandwidth a coordinate MLP can fit.
///
/// @param x  Coordinates [N, D].
/// @param B  Projection [D, m].
/// @return Features [N, 2m]: the m cosines, then the m sines.
/// @throws std::invalid_argument  If x is not rank 2 or B's first axis differs from x's second.
[[nodiscard]] Tensor gaussian_fourier_features(const Tensor& x, const Tensor& B);

}  // namespace datamunge::nn
