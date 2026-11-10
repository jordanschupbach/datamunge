#pragma once
/// @file conditional_autoencoder.hpp
/// @brief Class-conditioned Gaussian VAE forwards and target-free conditional decoding.
#include <limits>

#include "datamunge/nn/gaussian_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Validation of discrete class conditions before any learned leaves are created.
namespace conditional_detail {
/// @brief Validate positive widths and a representable concatenated code width.
/// @param features Pixel width.
/// @param classes One-hot condition width.
/// @param latent Gaussian code width.
/// @param hidden ReLU width.
/// @throws std::invalid_argument For nonpositive widths or latent+classes overflow.
inline void
Widths(std::int64_t features, std::int64_t classes, std::int64_t latent, std::int64_t hidden) {
  aae_detail::Widths(features, latent, hidden);
  if (classes <= 0 || latent > std::numeric_limits<std::int64_t>::max() - classes ||
      features > std::numeric_limits<std::int64_t>::max() - classes)
    throw std::invalid_argument(
        "conditional VAE requires positive representable concatenated widths");
}
/// @brief Require one valid integer condition per row, rejecting one-hot's all-off fallback.
/// @param labels Nonempty Int32/64 vector [batch].
/// @param batch Positive row count.
/// @param classes Positive exclusive upper bound for conditions.
/// @throws std::invalid_argument For wrong shape/dtype or labels outside [0,classes).
inline void Labels(const Tensor& labels, std::int64_t batch, std::int64_t classes) {
  if (batch <= 0 || classes <= 0 || !labels.defined() || labels.shape() != Shape{batch} ||
      (labels.dtype() != DType::Int32 && labels.dtype() != DType::Int64))
    throw std::invalid_argument("conditional VAE requires one Int32/64 class per row");
  for (auto c : labels.contiguous().to_vector<std::int64_t>())
    if (c < 0 || c >= classes)
      throw std::invalid_argument("conditional VAE class outside [0,classes)");
}
} // namespace conditional_detail

/// @brief Decode explicit Gaussian codes with a required supplied class and no image input.
///
/// Concatenates code then one_hot(labels,classes) and applies a hidden ReLU/linear
/// decoder, returning logits. Names are decoder/W[latent+classes,hidden], decoder/b,
/// output/W[hidden,features], output/b, ordinary Glorot-uniform weights and zero
/// biases; these are the full model's same four decoder leaves. There is no image,
/// posterior, noise, RNG, prediction of labels or hidden fitting. The prior is
/// supplied by the caller and need not depend on class. Apply sigmoid for pixel
/// probabilities, which are not sampled Bernoulli pixels.
///
/// Finite signed/zero nonempty Float32/64 codes must match store dtype. Integer
/// conditions [B] are validated before one-hot encoding or leaf creation; unlike
/// one_hot(), invalid classes never silently produce all-off rows. Strides and
/// continuous code/leaf gradients are retained; labels have no gradients. Inputs
/// unchanged; output owns lifetime. Incompatible leaves can fail after validation;
/// computed overflow is not repaired. Work O(B*hidden*(latent+classes+features));
/// shared stores require external synchronization.
/// @param parameters Mutable floating store containing/creating the decoder leaves.
/// @param code Finite nonempty [B,latent] Float32/64 code matching store dtype.
/// @param labels Required Int32/64 [B] supplied classes in [0,classes).
/// @param features Positive pixel width, default784.
/// @param classes Positive one-hot width, default10.
/// @param latent Positive code width, default2.
/// @param hidden Positive decoder ReLU width, default64.
/// @return Owning differentiable logits [B,features].
/// @throws std::invalid_argument For invalid inputs/options or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2111);
/// const auto labels=n::from_vector(std::vector<int>{3,8},{2});
/// const auto pixels=n::sigmoid(n::models::ConditionalVaeDecode(p,n::zeros({2,2}),labels));
/// // [2,784] probabilities conditioned on supplied classes3 and8, without images.
/// @endcode
/// @par References
/// - Kingma and Welling (2013/2014), Auto-Encoding Variational Bayes,
///   https://arxiv.org/abs/1312.6114 . Uses a fixed standard-normal prior.
/// - Sohn, Lee and Yan (2015), Learning Structured Output Representation using
///   Deep Conditional Generative Models,
///   https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html
///   . This teaching variant uses supplied digit classes and a fixed shared prior, not the paper's
///   learned conditional-prior structured-output architecture.
/// @see ConditionalVariationalAutoencoder(), GaussianVaeDecode()
inline Tensor ConditionalVaeDecode(
    ParameterStore& parameters,
    const Tensor&   code,
    const Tensor&   labels,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    latent   = 2,
    std::int64_t    hidden   = 64) {
  conditional_detail::Widths(features, classes, latent, hidden);
  aae_detail::Matrix(code, latent, parameters.dtype(), false);
  conditional_detail::Labels(labels, code.size(0), classes);
  const auto condition = one_hot(labels, classes, 1, 0, parameters.dtype());
  return GaussianVaeDecode(
      parameters, concat({code, condition}, 1), features, latent + classes, hidden);
}

/// @brief Own Gaussian VAE endpoints plus the exact condition and both concatenated inputs.
///
/// Inherits all nine owning GaussianVaeOutput fields with their continuous
/// autograd histories. loss and unweighted_loss are the same scalar for beta1.
/// condition is detached discrete data. No RNG, optimizer or training state is
/// stored. Labels are supplied conditions, not model predictions; decode(mu) is
/// not a posterior predictive mean.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2111);
/// const auto out=n::models::ConditionalVariationalAutoencoder(p,n::zeros({1,784}),
///     n::zeros({1},n::DType::Int32),n::zeros({1,2}));
/// // out.condition[1,10], encoder_input[1,794], decoder_input[1,12] own lifetime.
/// @endcode
/// @see ConditionalVariationalAutoencoder(), GaussianVaeOutput
struct ConditionalVaeOutput : GaussianVaeOutput {
  Tensor condition;     ///< Detached one-hot supplied labels [B,classes], in computation dtype.
  Tensor encoder_input; ///< Image then condition [B,features+classes], with image gradients.
  Tensor decoder_input; ///< Sample z then the same condition [B,latent+classes], with sample
                        ///< gradients.
};

/// @brief Evaluate a class-conditioned Gaussian/Bernoulli VAE with explicit posterior noise.
///
/// Conditions BOTH q(z|image,class) and p(image|z,class): image then one-hot class
/// -> hidden ReLU -> separate linear mu/log_variance heads; z=mu+exp(lv/2)*epsilon;
/// z then the same class -> hidden ReLU -> pixel logits. The prior is shared
/// N(0,I), not learned or class-specific. Names/order: encoder/W,b; mu/W,b;
/// log_variance/W,b; decoder/W,b; output/W,b. Ordinary Glorot weights/zero biases
/// give10 leaves/102932 default scalars. Bernoulli NLL sums features, Gaussian KL
/// sums latent coordinates, then images are averaged: loss=mean(NLL+KL), beta1.
/// Binary pixels give a one-draw negative ELBO estimate, not exact conditional
/// log likelihood. Soft [0,1] targets give Bernoulli cross-entropy rather than
/// a likelihood density for continuous-valued observations.
/// No label prediction, KL annealing, random draw, fitting or selection is hidden.
///
/// Finite unit-range images and finite epsilon match store Float32/64. Conditions
/// are one Int32/64 index per image and are checked before leaves are created.
/// Continuous image/epsilon/leaf paths and strides are preserved. Returned fields
/// own lifetime; inputs unchanged. Incompatible stored leaves/nonfinite computed
/// states may fail later; overflow is not repaired. Work
/// O(B*hidden*(features+classes+latent)), storage O(B*(features+classes+hidden+latent)).
/// Shared stores require synchronization. Supplied classes remain required at inference.
/// @param parameters Mutable floating parameter store.
/// @param image Finite nonempty unit-range [B,features] image rows.
/// @param labels Required Int32/64 [B] supplied classes in [0,classes).
/// @param epsilon Explicit finite [B,latent] standard-normal noise, computation dtype.
/// @param features Positive pixel width, default784.
/// @param classes Positive one-hot width, default10.
/// @param latent Positive Gaussian code width, default2.
/// @param hidden Positive encoder/decoder ReLU width, default64.
/// @return Owning twelve-field ConditionalVaeOutput, including both concatenation endpoints.
/// @throws std::invalid_argument For invalid input/options or incompatible stored leaves;
///         nonfinite computed states may fail validation later.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2111);
/// const auto out=n::models::ConditionalVariationalAutoencoder(p,n::full({2,784},.2),
///     n::from_vector(std::vector<int>{0,9},{2}),n::zeros({2,2}));
/// const auto gradients=n::grad(out.loss,p.parameters()); //10 gradients; caller owns updates.
/// @endcode
/// @par References
/// - Kingma and Welling (2013/2014), https://arxiv.org/abs/1312.6114 .
/// - Sohn, Lee and Yan (2015),
/// https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html
/// .
///   This small class-conditioned teaching model fixes the shared prior to N(0,I).
/// @see ConditionalVaeOutput, ConditionalVaeDecode(), GaussianReparameterize(),
/// DiagonalGaussianKl(), BernoulliPixelNll()
inline ConditionalVaeOutput ConditionalVariationalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   labels,
    const Tensor&   epsilon,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    latent   = 2,
    std::int64_t    hidden   = 64) {
  conditional_detail::Widths(features, classes, latent, hidden);
  aae_detail::Matrix(image, features, parameters.dtype(), true);
  aae_detail::Matrix(epsilon, latent, parameters.dtype(), false);
  conditional_detail::Labels(labels, image.size(0), classes);
  if (image.size(0) != epsilon.size(0))
    throw std::invalid_argument("conditional VAE batches must match");
  ConditionalVaeOutput out;
  out.condition     = one_hot(labels, classes, 1, 0, parameters.dtype());
  out.encoder_input = concat({image, out.condition}, 1);
  const auto h      = dense(parameters, "encoder", out.encoder_input, hidden, Activation::Relu);
  out.mu            = dense(parameters, "mu", h, latent);
  out.log_variance  = dense(parameters, "log_variance", h, latent);
  out.z             = GaussianReparameterize(out.mu, out.log_variance, epsilon);
  out.decoder_input = concat({out.z, out.condition}, 1);
  out.logits = GaussianVaeDecode(parameters, out.decoder_input, features, latent + classes, hidden);
  out.probability        = sigmoid(out.logits);
  out.reconstruction_nll = BernoulliPixelNll(out.logits, image);
  out.kl                 = DiagonalGaussianKl(out.mu, out.log_variance);
  out.loss               = mean(out.reconstruction_nll + out.kl, {0});
  out.unweighted_loss    = out.loss;
  return out;
}
} // namespace datamunge::nn::models
