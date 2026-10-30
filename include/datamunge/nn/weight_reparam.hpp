#pragma once

/// @file weight_reparam.hpp
/// @brief Weight reparameterisations of a dense layer: weight normalisation (Salimans and Kingma 2016) and spectral
///        normalisation by power iteration (Miyato et al. 2018), with a Lipschitz bound for stacks of layers.
///
/// Both write the weight matrix as a function of other tensors so that its scale is controlled explicitly. Weight
/// normalisation gives each column a direction and a separate gain, \f$W_{:,j}=g_j\,v_{:,j}/\lVert v_{:,j}\rVert\f$. Spectral normalisation divides the
/// matrix by an estimate of its largest singular value, so the layer's map \f$x\mapsto xW\f$ is (approximately) 1-Lipschitz in the
/// Euclidean norm.

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/params.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::nn {

/// @brief Output of a reparameterised dense layer: the activations together with the effective weight matrix.
struct ReparamDense {
  Tensor y;      ///< Layer output, act(x W + b).
  Tensor W;      ///< Effective weight matrix [in, units] (differentiable with respect to the underlying parameters).
  Tensor sigma;  ///< Spectral-norm estimate used by spectral_norm_dense() (a scalar); undefined for weight normalisation.
};

/// @brief Weight normalisation of the columns of a matrix: \f$W_{:,j}=g_j\,v_{:,j}/\lVert v_{:,j}\rVert\f$.
/// @param v  Directions [in, out].
/// @param g  Gains [out].
/// @return The effective weights [in, out]; column j has Euclidean norm \f$|g_j|\f$.
/// @throws std::invalid_argument  If v is not rank 2 or g does not have v.size(1) entries.
[[nodiscard]] Tensor weight_normalize(const Tensor& v, const Tensor& g);

/// @brief A weight-normalised dense layer: \f$y=\phi(xW+b)\f$ with \f$W\f$ from weight_normalize().
///
/// Parameters "<name>/v" [in, units] (Glorot uniform), "<name>/g" [units] and "<name>/b" [units] (zeros). On creation g
/// is set to the column norms of v, so the layer starts with \f$W=v\f$. The gradients are
/// \f$\partial L/\partial g_j=(\partial L/\partial W_{:,j})\cdot v_{:,j}/\lVert v_{:,j}\rVert\f$ and \f$\partial L/\partial v_{:,j}=\frac{g_j}{\lVert v_{:,j}\rVert}\big(I-\hat v_j\hat v_j^\top\big)\partial L/\partial W_{:,j}\f$, which is orthogonal to \f$v_{:,j}\f$.
///
/// @param store  Parameter store.
/// @param name  Parameter prefix.
/// @param x  Input [..., in].
/// @param units  Output size.
/// @param activation  Nonlinearity \f$\phi\f$ applied after the bias.
/// @return The output and the effective weight matrix (sigma is left undefined).
[[nodiscard]] ReparamDense weight_norm_dense(ParameterStore& store, const std::string& name, const Tensor& x,
                                             std::int64_t units, Activation activation = Activation::None);

/// @brief Result of power iteration for the largest singular value of a matrix.
struct SpectralNormEstimate {
  Tensor sigma;  ///< \f$\hat\sigma=vWu^\top\f$, a scalar; differentiable in W with gradient \f$v^\top u\f$ (u and v are constants).
  Tensor u;      ///< Right singular vector estimate [1, out], unit norm, detached.
  Tensor v;      ///< Left singular vector estimate [1, in], unit norm, detached.
};

/// @brief Estimates the largest singular value of W [in, out] by power iteration from the row vector u0 [1, out].
///
/// Each iteration sets \f$v\leftarrow uW^\top/\lVert uW^\top\rVert\f$ and \f$u\leftarrow vW/\lVert vW\rVert\f$. The estimate \f$\hat\sigma=vWu^\top=\lVert vW\rVert\f$ never exceeds \f$\sigma_1\f$, and its
/// relative error decays like \f$(\sigma_2/\sigma_1)^{4k}\f$ for k iterations (each applies \f$W^\top W\f$ once) when u0 has a component along the top
/// right singular vector. Exactly, with \f$c_i\f$ the coefficients of u0 on the right singular vectors and \f$A_m=\sum_ic_i^2\sigma_i^m\f$, \f$\hat\sigma_k=\sqrt{A_{4k}/A_{4k-2}}\f$.
///
/// @param W  Matrix [in, out].
/// @param u0  Starting vector [1, out]; need not be normalised.
/// @param iterations  Number of iterations, at least 1.
/// @return The estimate and the final singular-vector pair.
/// @throws std::invalid_argument  If W is not rank 2, u0 is not [1, out] or iterations < 1.
[[nodiscard]] SpectralNormEstimate spectral_norm_power_iteration(const Tensor& W, const Tensor& u0, int iterations = 1);

/// @brief A spectrally normalised dense layer: \f$y=\phi(xW/\hat\sigma+b)\f$ with \f$\hat\sigma\f$ from spectral_norm_power_iteration().
///
/// Parameters "<name>/W" [in, units] (Glorot uniform) and "<name>/b" [units] live in `params`; the power-iteration vector
/// "<name>/u" [1, units] (standard normal at creation) lives in `state`, which should not be optimised. With
/// `update_state` the refined u is written back, so that one iteration per training step keeps the estimate converged
/// while the weights move slowly. With a 1-Lipschitz activation the layer is 1-Lipschitz once \f$\hat\sigma=\sigma_1(W)\f$.
///
/// @param params  Trainable parameter store.
/// @param state  Store for the non-trainable vector u.
/// @param name  Parameter prefix.
/// @param x  Input [..., in].
/// @param units  Output size.
/// @param iterations  Power iterations per call.
/// @param activation  Nonlinearity \f$\phi\f$ applied after the bias.
/// @param update_state  Whether to store the refined u for the next call.
/// @return The output, the normalised weight matrix \f$W/\hat\sigma\f$ and \f$\hat\sigma\f$.
[[nodiscard]] ReparamDense spectral_norm_dense(ParameterStore& params, ParameterStore& state, const std::string& name,
                                               const Tensor& x, std::int64_t units, int iterations = 1,
                                               Activation activation = Activation::None, bool update_state = true);

/// @brief Upper bound on the Lipschitz constant of a stack of dense layers with 1-Lipschitz activations: \f$\prod_l\sigma_1(W_l)\f$.
///
/// The largest singular values are computed exactly with svd(). Biases do not affect the constant.
/// @param weights  Weight matrices [in, out] in any order.
/// @return The product of their spectral norms (1 for an empty list).
/// @throws std::invalid_argument  If a matrix is not rank 2.
[[nodiscard]] double lipschitz_bound(const std::vector<Tensor>& weights);

}  // namespace datamunge::nn
