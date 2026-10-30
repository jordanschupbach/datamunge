#pragma once

/// @file attention_variants.hpp
/// @brief Linear-time kernel attention, positive random features for the softmax kernel (FAVOR+),
///        sliding-window masks and the entropy of attention weights.
///
/// Softmax attention costs \f$O(T_qT_kd)\f$ time and \f$O(T_qT_k)\f$ memory per head. Writing the
/// unnormalised weight \f$\exp(s\,q\cdot k)\f$ as an inner product of feature maps
/// \f$\varphi(q)\cdot\varphi(k)\f$ lets the sum over keys be done once:
/// \f$\sum_j\varphi(q)^\top\varphi(k_j)v_j^\top=\varphi(q)^\top\big(\sum_j\varphi(k_j)v_j^\top\big)\f$, which costs
/// \f$O((T_q+T_k)\,m\,d_v)\f$ for \f$m\f$ features (Katharopoulos et al. 2020). kernel_attention() does this for any
/// non-negative feature map; positive_random_features() gives the map whose inner products are unbiased for the
/// softmax kernel (Choromanski et al. 2021). Shapes follow attention.hpp: queries [..., Tq, d], keys [..., Tk, d],
/// values [..., Tk, dv], and causal masking aligns the last query with the last key.

#include "datamunge/nn/random.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cstdint>

namespace datamunge::nn {

/// @brief Attention with weights \f$\varphi(q_i)\cdot\varphi(k_j)\f$ normalised over the keys, computed without
///        forming the [Tq, Tk] matrix.
///
/// Output row i is \f$\varphi(q_i)^\top S_i\,/\,(\varphi(q_i)^\top z_i+\varepsilon)\f$ with
/// \f$S_i=\sum_j\varphi(k_j)v_j^\top\f$ and \f$z_i=\sum_j\varphi(k_j)\f$, the sums running over all keys, or, when
/// @p causal, over keys \f$j\le i+(T_k-T_q)\f$ (prefix sums, as in attention.hpp). With an exact feature map for
/// \f$\exp(s\,q\cdot k)\f$ this equals softmax attention. Built from recorded ops, so differentiable in all inputs.
/// The causal form holds the prefix sums, [..., Tk, m, dv], in memory.
///
/// @param phi_q  Query features [..., Tq, m]; must be non-negative for the weights to be.
/// @param phi_k  Key features [..., Tk, m].
/// @param v  Values [..., Tk, dv].
/// @param causal  Restrict each query to the keys at or before its aligned position.
/// @param eps  Added to the normaliser (default 1e-6).
/// @return Output [..., Tq, dv].
/// @throws std::invalid_argument  If the feature or time axes disagree, or causal with Tq > Tk.
[[nodiscard]] Tensor kernel_attention(const Tensor& phi_q, const Tensor& phi_k, const Tensor& v, bool causal = false,
                                      double eps = 1e-6);

/// @brief The feature map \f$\mathrm{elu}(x)+1\f$ of Katharopoulos et al. (2020): positive, and linear for x > 0.
/// @param x  Input [..., d].
/// @return \f$\mathrm{elu}(x)+1\f$, same shape.
[[nodiscard]] Tensor elu_feature_map(const Tensor& x);

/// @brief Positive random features for the softmax kernel (FAVOR+, Choromanski et al. 2021):
///        \f$\varphi(x)=m^{-1/2}\exp\!\big(W x'-\lVert x'\rVert^2/2\big)\f$ with \f$x'=\sqrt{s}\,x\f$.
///
/// If the rows of @p w are standard normal, then \f$\mathbb E[\varphi(q)\cdot\varphi(k)]=\exp(s\,q\cdot k)\f$ exactly, so
/// kernel_attention() with these features estimates softmax attention with scale s. Every feature is positive,
/// so the estimated weights are too. No stabilising shift is applied: keep \f$s\lVert x\rVert^2\f$ moderate.
///
/// @param x  Queries or keys [..., T, d].
/// @param w  Projection [m, d], e.g. normal or orthogonal_gaussian_projection().
/// @param scale  The softmax scale s; 0 means \f$1/\sqrt d\f$ (default).
/// @return Features [..., T, m].
/// @throws std::invalid_argument  If w is not [m, d] with d = x.size(-1).
[[nodiscard]] Tensor positive_random_features(const Tensor& x, const Tensor& w, double scale = 0.0);

/// @brief An [m, d] projection whose rows are orthogonal within each block of d and have the norms of
///        independent standard-normal vectors (orthogonal random features, Yu et al. 2016).
///
/// Each row is marginally \f$\mathcal N(0,I_d)\f$, so positive_random_features() stays unbiased, while the
/// orthogonality makes the features less correlated and the estimate less variable.
///
/// @param m  Number of rows (features), positive.
/// @param d  Row length, positive.
/// @param gen  Random generator.
/// @param dtype  Floating dtype (default float32).
/// @return The projection [m, d].
/// @throws std::invalid_argument  If m or d is not positive.
[[nodiscard]] Tensor orthogonal_gaussian_projection(std::int64_t m, std::int64_t d, Generator& gen,
                                                    DType dtype = DType::Float32);

/// @brief Bool [Tq, Tk] keep mask for sliding-window (local) attention.
///
/// With aligned position \f$i'=i+(T_k-T_q)\f$, causal windows keep keys \f$i'-w<j\le i'\f$ (the current key and the
/// w - 1 before it, as in Mistral); two-sided windows keep \f$\lvert j-i'\rvert<w\f$ (as in Longformer).
///
/// @param Tq  Number of queries.
/// @param Tk  Number of keys.
/// @param window  Window size w, at least 1.
/// @param causal  One-sided (default) or two-sided window.
/// @return The keep mask.
/// @throws std::invalid_argument  If window < 1 or a size is negative.
[[nodiscard]] Tensor local_keep_mask(std::int64_t Tq, std::int64_t Tk, std::int64_t window, bool causal = true);

/// @brief Shannon entropy, in nats, of each row of attention weights: \f$-\sum_jw_j\log w_j\f$ over the last axis.
///
/// Zero weights contribute nothing. A row spread evenly over n keys has entropy \f$\log n\f$; a one-hot row has 0.
/// Differentiable where the weights are positive.
///
/// @param weights  Attention weights [..., Tq, Tk], rows summing to one (or zero for fully masked rows).
/// @return Entropies [..., Tq].
[[nodiscard]] Tensor attention_entropy(const Tensor& weights);

}  // namespace datamunge::nn
