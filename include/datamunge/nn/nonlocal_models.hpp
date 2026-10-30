#pragma once
/// @file nonlocal_models.hpp
/// @brief Tensor non-local affinity blocks and explicit residual-backbone placement policy.
#include "datamunge/nn/attention_cnn_models.hpp"
#include <cmath>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Choose the non-local pairwise affinity and its normalization.
///
/// Value metadata; no model state or fitting. Dot/concatenation modes
/// are not row-probability distributions.
///
/// @par Example
/// @code
/// const auto f = dm::nn::models::Pairwise::kEmbeddedGaussian;
/// @endcode
/// @par References
/// - Wang et al. (2018), https://arxiv.org/abs/1711.07971; https://github.com/facebookresearch/video-nonlocal-net
/// @see NonLocalBlock()
enum class Pairwise {
  kEmbeddedGaussian, ///< Softmax of embedded dot products divided by sqrt(inner).
  kGaussian,         ///< Softmax of raw-input dot products, no scale/embeddings.
  kDotProduct,       ///< Embedded dot products divided by M key positions; may be signed.
  kConcatenation     ///< ReLU of separated linear theta/phi scores divided by M.
};

/// @brief Non-local affinity weights and residual output tensors.
///
/// Tensor handles own storage references and preserve full differentiation
/// graphs; source learned leaves remain in the stores. No fitting occurs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::NonLocalBlock(p, state, "nl", n::ones({2, 4, 4, 8}),
///                                         n::models::Pairwise::kEmbeddedGaussian, false, false);
/// @endcode
/// @par References
/// - Wang et al. (2018), https://arxiv.org/abs/1711.07971; https://github.com/facebookresearch/video-nonlocal-net
/// @see NonLocalBlock()
struct NonLocalOutputs {
  nn::Tensor attention; ///< [N,P,M] pairwise weights, probabilities only in Gaussian modes.
  nn::Tensor
      output; ///< Input plus projected/BN attention response, retaining all derivatives.
};

/// @brief Apply a differentiable non-local pairwise response with zero-scale BN residual.
///
/// P=H*W, M=H*W without subsampling or floor(H/2)*floor(W/2)
/// with it. Inner width is floor(C/2). Every position mixes all M values
/// through g. Gaussian uses raw x affinity; embedded Gaussian divides
/// theta*phi by sqrt(inner), then softmax over keys. Dot divides raw
/// embedded products by M; concatenation uses ReLU(w1*theta+w2*phi)/M.
/// These latter rows need not sum to one, and dot weights may be negative.
/// Output projects to C with bias then learned BN eps 1e-5 and adds x.
/// Gamma starts at zero so initialization is exact signed identity.
/// This source convention pools input BEFORE phi/g convolutions; it is
/// not the same operation as max-pooling already projected embeddings.
/// Work/storage includes O(N*P*M*inner)/O(N*P*M) affinity, plus filters.
/// No fitting or conditional speedup is hidden; synchronize shared stores.
///
/// @param p Mutable floating learned store, lazy biased g/theta/phi/W embeddings and BN affine leaves.
/// @param state Mutable matching-dtype output BN moving moments.
/// @param name Non-local parameter prefix.
/// @param x Nonempty floating NHWC input matching stores, C at least two.
/// @param f One of the four Pairwise modes.
/// @param subsample True max-pools INPUT by 2x2 VALID before key/value projections; both sides at least two.
/// @param training True uses batch BN, false saved moments.
/// @param momentum Moving-moment retention in [0,1), default .9.
/// @param update_stats Permit eager training moving updates, default true.
/// @return Pairwise [N,P,M] weights and same-shape residual output.
/// @throws std::invalid_argument For invalid rank/size/mode/BN setting, unsupported dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::NonLocalBlock(p, state, "nl", n::ones({2, 4, 4, 8}),
///                                         n::models::Pairwise::kEmbeddedGaussian, true, false);
/// @endcode
/// @par References
/// - Wang et al. (2018), https://arxiv.org/abs/1711.07971; https://github.com/facebookresearch/video-nonlocal-net
/// @see Pairwise, NonLocalResNetHook(), AttentionResNet()
inline NonLocalOutputs NonLocalBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                     const std::string &name, const nn::Tensor &x, Pairwise f,
                                     bool subsample, bool training, double momentum = .9,
                                     bool update_stats = true) {
  if (x.dim() != 4 || x.numel() == 0 || x.size(3) < 2 ||
      (subsample && (x.size(1) < 2 || x.size(2) < 2)) || !(momentum >= 0 && momentum < 1) ||
      int(f) < 0 || int(f) > 3)
    throw std::invalid_argument("invalid non-local tensor/affinity/BN options");
  const std::int64_t n = x.shape()[0], height = x.shape()[1], width = x.shape()[2],
                     channels = x.shape()[3], inner = channels / 2;
  if (inner < 1)
    throw std::invalid_argument("non-local blocks need at least two channels");
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  /// Apply one biased He-uniform pointwise embedding or output projection.
  auto conv = [&](const std::string &part, const nn::Tensor &in, std::int64_t to) {
    return nn::conv2d(p, name + "/" + part, in, to, 1, cfg);
  };
  /// Flatten spatial positions while retaining batch and channel axes.
  auto flatten = [&](const nn::Tensor &t) {
    return nn::reshape(t, {n, -1, t.shape()[3]});
  }; // [N, H*W, C]
  const nn::Tensor pooled = subsample ? nn::max_pool2d(x, 2, 2, nn::Padding::Valid) : x;
  const nn::Tensor g = flatten(conv("g", pooled, inner)); // [N, M, inner]
  nn::Tensor affinity;
  if (f == Pairwise::kGaussian) {
    affinity = nn::matmul(flatten(x), flatten(pooled), false, true);
  } else {
    const nn::Tensor theta = flatten(conv("theta", x, inner)),
                     phi = flatten(conv("phi", pooled, inner));
    if (f == Pairwise::kConcatenation) {
      // w . [theta_i, phi_j] = w1 . theta_i + w2 . phi_j for every pair (i, j);
      // He-uniform with the fan-in 2*inner of the concatenated vector.
      const auto init = nn::init::uniform(std::sqrt(6.0 / double(2 * inner)));
      const nn::Tensor w1 = p.get(name + "/concat/W1", {inner, 1}, init),
                       w2 = p.get(name + "/concat/W2", {inner, 1}, init);
      const nn::Tensor a = nn::matmul(theta, w1),
                       b = nn::matmul(phi, w2); // [N, P, 1], [N, M, 1]
      affinity = nn::relu(a + nn::transpose(b, {0, 2, 1}));
    } else {
      affinity = nn::matmul(theta, phi, false, true); // [N, P, M]
      if (f == Pairwise::kEmbeddedGaussian)
        affinity = affinity * (1 / std::sqrt(double(inner)));
    }
  }
  NonLocalOutputs r;
  if (f == Pairwise::kEmbeddedGaussian || f == Pairwise::kGaussian)
    r.attention = nn::softmax(affinity, -1);
  else
    r.attention = affinity / double(affinity.shape()[2]);
  const nn::Tensor y = nn::reshape(nn::matmul(r.attention, g), {n, height, width, inner});
  const nn::Tensor w = conv("W", y, channels);
  p.get(name + "/bn/gamma", {channels},
        nn::init::zeros()); // zero scale: the block starts as the identity
  nn::BatchNormOptions bn;
  bn.momentum = momentum;
  bn.eps = 1e-5;
  bn.update_stats = update_stats;
  r.output = nn::batch_norm(p, state, name + "/bn", w, training, bn) + x;
  return r;
}
/// @brief Create the five-block non-local policy for a ResNet-50 backbone.
///
/// Select zero-based stages one/two and even block indices. For depth 50
/// this adds two modules after stage-two blocks 0/2 and three after stage-
/// three blocks 0/2/4. Deeper backbones get additional even-index modules,
/// not a hard-coded total of five. Uses embedded Gaussian and prefix
/// <block>/non_local. Captures scalar options by value; no weights until
/// called. Set update_stats consistently with the outer backbone when
/// freezing moments. Tensor inputs and borrowed stores are used only at call
/// time; retained graphs require their leaves to outlive backward.
///
/// @param momentum Captured BN moving retention in [0,1).
/// @param update_stats Captured permission for eager training BN updates.
/// @param subsample Max-pool inputs before keys/values, default true.
/// @return Owned after-block hook, returning unchanged maps outside selected positions.
/// @throws std::invalid_argument For invalid captured momentum or a later NonLocalBlock() error.
/// @par Example
/// @code
/// const auto hook = dm::nn::models::NonLocalResNetHook(.9, false);
/// @endcode
/// @par References
/// - Wang et al. (2018), https://arxiv.org/abs/1711.07971; https://github.com/facebookresearch/video-nonlocal-net
/// @see AttentionResNetConfig, NonLocalBlock()

inline AfterBlockHook NonLocalResNetHook(double momentum, bool update_stats,
                                         bool subsample = true) {
  if (!(momentum >= 0 && momentum < 1))
    throw std::invalid_argument("invalid non-local hook BN momentum");
  return [=](nn::ParameterStore &p, nn::ParameterStore &state, const std::string &name,
             const nn::Tensor &x, int stage, int block, bool training) {
    if ((stage == 1 || stage == 2) && block % 2 == 0)
      return NonLocalBlock(p, state, name + "/non_local", x, Pairwise::kEmbeddedGaussian,
                           subsample, training, momentum, update_stats)
          .output;
    return x;
  };
}

} // namespace datamunge::nn::models
