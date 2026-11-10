#pragma once
/// @file gumbel_categorical_autoencoder.hpp
/// @brief Explicit Gumbel banks, differentiable categorical relaxation and Bernoulli categorical
/// VAEs.
#include <limits>
#include <random>

#include "datamunge/nn/gaussian_autoencoder.hpp"

namespace datamunge::nn::models {
/// @brief Input checks shared by categorical relaxation and target-free decoders.
namespace gumbel_detail {
/// @brief Check architecture widths and return the flattened categorical width without overflow.
/// @param features Positive pixel width.
/// @param groups Positive independent categorical-variable count.
/// @param classes At least two categories, fitting Int32 model IDs.
/// @param hidden Positive hidden width.
/// @return groups*classes as a checked positive int64 width.
/// @throws std::invalid_argument For nonpositive widths, unsupported classes or multiplication
/// overflow.
inline std::int64_t
Width(std::int64_t features, std::int64_t groups, std::int64_t classes, std::int64_t hidden) {
  if (features <= 0 || groups <= 0 || classes < 2 ||
      classes > std::numeric_limits<std::int32_t>::max() || hidden <= 0 ||
      groups > std::numeric_limits<std::int64_t>::max() / classes)
    throw std::invalid_argument(
        "categorical architecture needs positive widths and at least two classes");
  return groups * classes;
}
/// @brief Validate temperature and retain the source's Float32 scalar rounding where appropriate.
/// @param temperature Finite positive temperature, representable in the computation dtype.
/// @param type Float32 or Float64 computation dtype.
/// @return Positive rounded scalar (Float32 promoted to double), or the Float64 input.
/// @throws std::invalid_argument For invalid temperature/dtype or Float32 overflow/underflow.
inline double Temperature(double temperature, DType type) {
  if (!std::isfinite(temperature) || temperature <= 0 ||
      (type != DType::Float32 && type != DType::Float64) ||
      (type == DType::Float32 && (!std::isfinite(float(temperature)) || float(temperature) <= 0)))
    throw std::invalid_argument("finite representable positive Gumbel temperature required");
  return type == DType::Float32 ? double(float(temperature)) : temperature;
}
/// @brief Validate nonempty grouped simplex codes, including finite values and per-group sums.
/// @param code Matching floating [B,groups,classes] code; strides allowed.
/// @param groups Expected categorical-variable count.
/// @param classes Expected category count.
/// @param type Store computation dtype.
/// @throws std::invalid_argument For invalid shape/dtype, values or normalization.
inline void Simplex(const Tensor& code, std::int64_t groups, std::int64_t classes, DType type) {
  if (!code.defined() || code.dim() != 3 || code.size(0) <= 0 || code.size(1) != groups ||
      code.size(2) != classes || code.dtype() != type ||
      (type != DType::Float32 && type != DType::Float64))
    throw std::invalid_argument("grouped floating simplex codes required");
  const auto   v         = code.contiguous().to_vector<double>();
  const double tolerance = type == DType::Float32 ? 2e-6 : 1e-12;
  for (std::size_t i = 0; i < v.size(); i += std::size_t(classes)) {
    double total = 0;
    for (std::int64_t k = 0; k < classes; ++k) {
      const auto x = v[i + k];
      if (!std::isfinite(x) || x < 0 || x > 1)
        throw std::invalid_argument("simplex values must lie in [0,1]");
      total += x;
    }
    if (std::abs(total - 1) >= tolerance)
      throw std::invalid_argument("each categorical code must sum to one");
  }
}
} // namespace gumbel_detail

/// @brief Own posterior categorical probabilities, a relaxed draw and hard Gumbel-max indices.
///
/// All floating fields retain their continuous input gradients and own lifetime.
/// Indices are detached Int64, with the lowest class index winning ties.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto sample=n::models::GumbelCategorical(n::zeros({1,3}),n::zeros({1,3}));
/// // probability and relaxed are uniform; tied hard index is0.
/// @endcode
/// @see GumbelCategorical()
struct GumbelCategoricalSample {
  Tensor log_probability; ///< log_softmax(logits), [B,C], independent of noise/temperature.
  Tensor probability;     ///< exp(log_probability), [B,C], independent of noise/temperature.
  Tensor relaxed;         ///< softmax((logits+noise)/temperature), [max(B,M),C].
  Tensor indices;         ///< Detached argmax(logits+noise), Int64 [max(B,M)].
};

/// @brief Reparameterize an explicit Gumbel draw with softmax and retain its hard categorical
/// sample.
///
/// q=exp(log_softmax(logits)), y=softmax((logits+noise)/temperature), and
/// hard=argmax(logits+noise). Hard samples ignore positive temperature; relaxed
/// values and gradients do not. No straight-through estimator is used. Both
/// logits and noise receive relaxed-path derivatives; q does not depend on noise.
/// Inputs are finite nonempty matching Float32/64 matrices with C>=2. Row counts
/// B,M must match or one is1, supporting one categorical distribution with many
/// noise draws. Strides/broadcast gradients are retained; inputs unchanged.
/// Temperature is rounded to Float32 before division for Float32 inputs, matching
/// the original experiment; invalid/unrepresentable values are rejected. Computed
/// score/division overflow is not repaired. No RNG, fitting, clipping or shared
/// state; output handles own lifetime. Work/storage O(max(B,M)*C).
/// @param logits Finite floating unnormalized log probabilities [B,C].
/// @param noise Finite same-dtype explicit Gumbel values [M,C], compatible rows.
/// @param temperature Finite representable positive scalar, default .7 (narrowed for Float32).
/// @return Owning categorical sample; posterior fields keep B rows, draw fields use broadcast rows.
/// @throws std::invalid_argument For invalid shapes/dtypes/values or temperature.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto logits=n::zeros({1,3}).requires_grad_();
/// const auto sample=n::models::GumbelCategorical(logits,n::tensor({.1,.4,-.3},{1,3}));
/// const auto gradients=n::grad(n::sum(n::square(sample.relaxed)),{logits});
/// // Hard index1; no hard argmax derivative is inserted into fitting.
/// @endcode
/// @par References
/// - Jang, Gu and Poole (2016/2017), Categorical Reparameterization with Gumbel-Softmax,
///   https://arxiv.org/abs/1611.01144 . Uses the soft path, without straight-through gradients.
/// - Maddison, Mnih and Teh (2016/2017), The Concrete Distribution: A Continuous
///   Relaxation of Discrete Random Variables, https://arxiv.org/abs/1611.00712 .
/// @see GumbelNoise(), CategoricalUniformKl(), GumbelCategoricalAutoencoder()
inline GumbelCategoricalSample
GumbelCategorical(const Tensor& logits, const Tensor& noise, double temperature = .7) {
  if (!logits.defined() || logits.dim() != 2 || logits.size(1) < 2)
    throw std::invalid_argument("categorical logits need [B,C], C>=2");
  aae_detail::Matrix(logits, logits.size(1), logits.dtype(), false);
  aae_detail::Matrix(noise, logits.size(1), logits.dtype(), false);
  if (logits.size(0) != noise.size(0) && logits.size(0) != 1 && noise.size(0) != 1)
    throw std::invalid_argument("categorical draw row counts must broadcast");
  const auto              tau = gumbel_detail::Temperature(temperature, logits.dtype());
  GumbelCategoricalSample out;
  out.log_probability = log_softmax(logits, -1);
  out.probability     = exp(out.log_probability);
  const auto scores   = logits + noise;
  out.relaxed         = softmax(scores / tau, -1);
  out.indices         = argmax(scores, -1);
  return out;
}

/// @brief Sum categorical KL to a uniform distribution over each row's classes.
///
/// Returns sum(q*(logq+log C),axis=1), without sampling or averaging classes.
/// Float32 rounds log C to Float32 before addition; Float64 uses full precision.
/// q and logq must be finite nonempty matching Float32/64 [B,C], C>=2, with
/// q values in[0,1]. The caller supplies normalized q and its matching logq;
/// that relationship is not checked, allowing independent derivatives of both
/// arguments. Stable finite log-softmax values allow q=0 from underflow without
/// evaluating log(0). Negative roundoff is retained, not clipped. Inputs unchanged;
/// all continuous paths/strides and owning lifetime retained. O(B*C), no shared state.
/// @param probability Finite unit-range categorical probabilities [B,C].
/// @param log_probability Finite same-shape/dtype log probabilities [B,C].
/// @return Owning differentiable per-row KL [B], nats, when supplied probabilities are consistent.
/// @throws std::invalid_argument For invalid shapes, dtype or values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto s=n::models::GumbelCategorical(n::zeros({2,3}),n::zeros({2,3}));
/// const auto kl=n::models::CategoricalUniformKl(s.probability,s.log_probability);
/// // Uniform posterior: each row's KL is0, up to roundoff.
/// @endcode
/// @par References
/// - Jang, Gu and Poole (2016/2017), https://arxiv.org/abs/1611.01144 . This is
///   the discrete categorical KL, not a KL between continuous Concrete densities.
/// @see GumbelCategorical(), GumbelCategoricalAutoencoder()
inline Tensor CategoricalUniformKl(const Tensor& probability, const Tensor& log_probability) {
  gaussian_detail::Match(probability, log_probability);
  if (probability.size(1) < 2)
    throw std::invalid_argument("at least two categorical classes required");
  aae_detail::Matrix(probability, probability.size(1), probability.dtype(), true);
  double log_classes = std::log(double(probability.size(1)));
  if (probability.dtype() == DType::Float32)
    log_classes = double(float(log_classes));
  return sum(probability * (log_probability + log_classes), {-1});
}

/// @brief Own the source-compatible uniform/Gumbel banks and their Float32 model inputs.
///
/// All fields are detached and own lifetime. A local MT19937 seed determines
/// sample-major, row-major, group-major then class-major values.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto bank=dm::nn::models::GumbelNoise(2,4,2927);
/// // All fields have shape[4,2,4,8]; bank.noise is Float32.
/// @endcode
/// @see GumbelNoise()
struct GumbelNoiseBank {
  Tensor uniform; ///< Open-unit-interval Float64 midpoint uniforms [S,B,G,C].
  Tensor gumbel;  ///< Float64 -log(-log(uniform)), same shape.
  Tensor noise;   ///< Float32 cast of gumbel, same shape; no gradient history.
};

/// @brief Generate explicit Gumbel noise from a local seeded MT19937 midpoint-uniform stream.
///
/// Each uint32 engine draw r gives u=(double(r)+.5)/2^32 and g=-log(-log(u)).
/// No normal/distribution-library sampler or global RNG is involved. Saves both
/// Float64 intermediates and the exact Float32 cast, without endpoint clipping.
/// Positive dimensions/C>=2 and element/byte-count overflow are checked before
/// allocation. Outputs own lifetime; no inputs or shared state are mutated.
/// Work/storage O(samples*n*groups*classes). Reproducibility assumes matching
/// floating log implementations; engine values are prescribed MT19937 outputs.
/// @param n Positive rows per draw.
/// @param samples Positive independent draw count.
/// @param seed Full unsigned32 local engine seed.
/// @param groups Positive categorical-variable count, default4.
/// @param classes At least two classes per variable, default8.
/// @return Three detached owning banks of shape[samples,n,groups,classes].
/// @throws std::invalid_argument For invalid dimensions or element/byte-count overflow.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto bank=dm::nn::models::GumbelNoise(5,2,2969,1,3);
/// const auto draw=dm::nn::select(bank.noise,0,0).contiguous();
/// // draw has shape[5,1,3]; the model never samples implicitly.
/// @endcode
/// @par References
/// - Jang, Gu and Poole (2016/2017), https://arxiv.org/abs/1611.01144 , Gumbel transform.
/// - Maddison, Mnih and Teh (2016/2017), https://arxiv.org/abs/1611.00712 .
/// @see GumbelNoiseBank, GumbelCategorical()
inline GumbelNoiseBank GumbelNoise(
    std::int64_t  n,
    std::int64_t  samples,
    std::uint32_t seed,
    std::int64_t  groups  = 4,
    std::int64_t  classes = 8) {
  if (n <= 0 || samples <= 0 || groups <= 0 || classes < 2)
    throw std::invalid_argument("positive categorical noise dimensions, C>=2 required");
  std::size_t count = 1;
  for (const auto size : {samples, n, groups, classes}) {
    if (std::uint64_t(size) > std::uint64_t(std::numeric_limits<std::size_t>::max() / count) ||
        std::uint64_t(size) > std::uint64_t(std::numeric_limits<std::int64_t>::max()) / count)
      throw std::invalid_argument("categorical noise shape overflow");
    count *= std::size_t(size);
  }
  if (count > std::vector<double>().max_size() || count > std::vector<float>().max_size() ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::invalid_argument("categorical noise byte count overflow");
  std::vector<double> uniform(count), gumbel(count);
  std::vector<float>  noise(count);
  std::mt19937        random(seed);
  for (std::size_t i = 0; i < count; ++i) {
    const double u = (double(random()) + .5) / 4294967296., g = -std::log(-std::log(u));
    uniform[i] = u;
    gumbel[i]  = g;
    noise[i]   = float(g);
  }
  const Shape shape{samples, n, groups, classes};
  return {from_vector(uniform, shape), from_vector(gumbel, shape), from_vector(noise, shape)};
}

/// @brief Decode supplied grouped simplex codes into Bernoulli probabilities without image targets.
///
/// Flattens groups/classes to one code axis and uses GaussianVaeDecode's shared
/// decoder/output names, with ordinary Glorot weights and zero biases, followed
/// by sigmoid. Each group sums to1 within strict2e-6(Float32)/1e-12(Float64).
/// All values are finite and in[0,1]. Validation precedes leaf creation; incompatible
/// stored leaves/nonfinite computed states may fail later. Inputs unchanged;
/// outputs own lifetime and all continuous code/parameter gradients, including
/// strided inputs. No RNG, fitting or no-grad scope is hidden. Synchronize stores.
/// Work O(B*hidden*(groups*classes+features)); default four leaves/26928 scalars.
/// @param parameters Mutable matching-dtype floating decoder store.
/// @param simplex Nonempty matching floating code[B,groups,classes].
/// @param features Positive output pixel width, default784.
/// @param groups Positive categorical-variable count, default4.
/// @param classes At least two categories, default8, fitting Int32 model IDs.
/// @param hidden Positive decoder ReLU width, default32.
/// @return Owning differentiable Bernoulli probabilities[B,features], not sampled pixels.
/// @throws std::invalid_argument For invalid code/options/overflow or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2903);
/// const auto probability=n::models::GumbelDecodeSoft(p,n::full({2,4,8},.125));
/// // [2,784] output; target images and an encoder are unnecessary.
/// @endcode
/// @see GumbelDecodeIds(), GumbelCategoricalAutoencoder(), GaussianVaeDecode()
inline Tensor GumbelDecodeSoft(
    ParameterStore& parameters,
    const Tensor&   simplex,
    std::int64_t    features = 784,
    std::int64_t    groups   = 4,
    std::int64_t    classes  = 8,
    std::int64_t    hidden   = 32) {
  const auto width = gumbel_detail::Width(features, groups, classes, hidden);
  gumbel_detail::Simplex(simplex, groups, classes, parameters.dtype());
  return sigmoid(GaussianVaeDecode(
      parameters, reshape(simplex, {simplex.size(0), width}), features, width, hidden));
}

/// @brief Decode hard categorical IDs through the same target-free Bernoulli decoder as soft codes.
///
/// Accepts finite in-range Int32/Int64 IDs; constructs same-store-dtype one-hot
/// codes then calls GumbelDecodeSoft. IDs are discrete; decoder parameters retain
/// gradients. No encoder/image/RNG/fitting is used; input unchanged, output owns
/// lifetime. Validation precedes leaf creation; incompatible leaves may fail later.
/// Strides accepted. Work includes O(B*groups*classes) one-hot storage; synchronize stores.
/// @param parameters Mutable Float32/64 decoder store.
/// @param ids Nonempty Int32/64 category IDs[B,groups], each in[0,classes).
/// @param features Positive output pixel width, default784.
/// @param groups Positive categorical-variable count, default4.
/// @param classes At least two categories, default8, fitting Int32 model IDs.
/// @param hidden Positive decoder ReLU width, default32.
/// @return Owning differentiable Bernoulli probabilities[B,features].
/// @throws std::invalid_argument For invalid IDs/options/dtype/overflow or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2903);
/// const auto images=n::models::GumbelDecodeIds(p,n::from_vector(std::vector<int>{0,1,2,3},{1,4}));
/// // One [1,784] decoder-probability row, independent of a target image.
/// @endcode
/// @see GumbelDecodeSoft(), GumbelCategoricalAutoencoder()
inline Tensor GumbelDecodeIds(
    ParameterStore& parameters,
    const Tensor&   ids,
    std::int64_t    features = 784,
    std::int64_t    groups   = 4,
    std::int64_t    classes  = 8,
    std::int64_t    hidden   = 32) {
  gumbel_detail::Width(features, groups, classes, hidden);
  if (!ids.defined() || ids.dim() != 2 || ids.size(0) <= 0 || ids.size(1) != groups ||
      (ids.dtype() != DType::Int32 && ids.dtype() != DType::Int64))
    throw std::invalid_argument("grouped Int32/64 category IDs required");
  for (auto value : ids.to_vector<std::int64_t>())
    if (value < 0 || value >= classes)
      throw std::invalid_argument("category ID out of range");
  return GumbelDecodeSoft(
      parameters,
      cast(one_hot(ids, classes), parameters.dtype()),
      features,
      groups,
      classes,
      hidden);
}

/// @brief Own categorical posterior, soft/hard sample, decoder and objective endpoints.
///
/// All fields own lifetime. Floating paths retain mathematical gradients; discrete
/// indices/hard codes carry no logits/noise derivative. No straight-through path.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2903);
/// const auto out=n::models::GumbelCategoricalAutoencoder(
///     p,n::zeros({2,784}),n::zeros({2,4,8}));
/// // Eight parameter leaves/53104 scalars; code is [2,32].
/// @endcode
/// @see GumbelCategoricalAutoencoder()
struct GumbelCategoricalAutoencoderOutput {
  Tensor logits;          ///< Posterior logits[B,G,C], independent of supplied noise/temperature.
  Tensor log_probability; ///< Posterior log probabilities[B,G,C].
  Tensor probability;     ///< Posterior probabilities[B,G,C].
  Tensor relaxed;         ///< Reparameterized soft draw[B,G,C].
  Tensor indices;         ///< Detached Int32 Gumbel-max category IDs[B,G].
  Tensor hard;            ///< Same-computation-dtype one-hot hard sample[B,G,C].
  Tensor code;            ///< Relaxed draw flattened to[B,G*C], the soft decoder input.
  Tensor soft_prediction; ///< Soft-draw decoder probabilities[B,features].
  Tensor hard_prediction; ///< Matched hard-draw decoder probabilities[B,features].
  Tensor nll;             ///< Per-image summed pixel soft Bernoulli NLL/cross-entropy[B].
  Tensor hard_nll;        ///< Per-image summed pixel hard Bernoulli NLL/cross-entropy[B].
  Tensor kl;              ///< Analytic discrete KL to G independent uniform categorical priors[B].
  Tensor loss; ///< Scalar mean(nll+kl), a biased continuous surrogate, not a discrete ELBO.
  Tensor hard_objective; ///< Scalar mean(hard_nll+kl), a discrete Monte Carlo negative ELBO
                         ///< diagnostic.
};

/// @brief Evaluate a categorical VAE trained by a soft Gumbel reconstruction plus discrete KL
/// surrogate.
///
/// Encoder features->hidden ReLU->groups*classes logits; independent Gumbel-softmax
/// draws form flattened relaxed codes. One shared ReLU/linear Bernoulli decoder
/// evaluates both relaxed and matched one-hot codes. Parameters are encoder/W,b;
/// categorical/W,b;decoder/W,b;output/W,b, created in that order with ordinary
/// Glorot-uniform weights and zero biases. Defaults are four8-way variables,
/// hidden32,784 pixels and eight leaves/53104 scalars, seed2903 in the report.
///
/// loss=mean(sum_pixels soft NLL+sum_groups discrete KL(q||Uniform(C))). This mixes
/// a relaxed likelihood and a discrete KL: it is a biased continuous surrogate,
/// not an exact discrete ELBO or Concrete-density KL. hard_objective instead uses
/// actual Gumbel-max categorical draws, yielding a discrete Monte Carlo negative
/// ELBO diagnostic. Training inserts no straight-through derivative. Temperature
/// affects soft values/gradients only, not q, discrete KL or hard sample indices.
/// Float32 narrows temperature and log C to preserve the original scalar arithmetic.
///
/// Unit-range finite nonempty images[B,F] and matching finite supplied noise[B,G,C]
/// are validated with positive widths/C>=2 and multiplication overflow before
/// creating leaves. No RNG, fitting, gradient request, optimizer or checkpoint
/// selection is hidden. Continuous image/noise/parameter paths stay live; hard
/// argmax paths do not. Soft-target NLL is cross-entropy rather than binary NLL.
/// Computed nonfinite states may fail later; no clipping/repair. Inputs unchanged,
/// owning results/strides preserved; synchronize stores. Work includes
/// O(B*hidden*(features+groups*classes)); the hard decoder is evaluated separately.
/// @param parameters Mutable matching Float32/64 parameter store.
/// @param image Finite pixel rows[B,features], B>0, in[0,1].
/// @param noise Finite matching-dtype explicit Gumbels[B,groups,classes].
/// @param temperature Positive representable scalar, default .7 (narrowed for Float32); no annealing.
/// @param features Positive image width, default784.
/// @param groups Positive number of categorical variables, default4.
/// @param classes At least two categories, default8, fitting Int32 model IDs.
/// @param hidden Positive encoder/decoder ReLU width, default32.
/// @return Owning fourteen-field forward output with soft loss and hard diagnostic.
/// @throws std::invalid_argument For invalid inputs/options/overflow or incompatible leaves.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore p(2903);
/// const auto out=n::models::GumbelCategoricalAutoencoder(
///     p,n::full({2,784},.2),n::zeros({2,4,8}));
/// const auto gradients=n::grad(out.loss,p.parameters());
/// // Caller owns Adam and fitting; all eight leaves receive soft-path/KL gradients.
/// @endcode
/// @par References
/// - Jang, Gu and Poole (2016/2017), Categorical Reparameterization with Gumbel-Softmax,
///   https://arxiv.org/abs/1611.01144 . Uses relaxed training, without straight-through gradients.
/// - Maddison, Mnih and Teh (2016/2017), The Concrete Distribution: A Continuous Relaxation
///   of Discrete Random Variables, https://arxiv.org/abs/1611.00712 . This migrated
///   small architecture is a teaching experiment, not a benchmark reproduction.
/// @see GumbelCategoricalAutoencoderOutput, GumbelCategorical(), CategoricalUniformKl(),
/// GumbelDecodeSoft(), GumbelDecodeIds(), GumbelNoise()
inline GumbelCategoricalAutoencoderOutput GumbelCategoricalAutoencoder(
    ParameterStore& parameters,
    const Tensor&   image,
    const Tensor&   noise,
    double          temperature = .7,
    std::int64_t    features    = 784,
    std::int64_t    groups      = 4,
    std::int64_t    classes     = 8,
    std::int64_t    hidden      = 32) {
  const auto width = gumbel_detail::Width(features, groups, classes, hidden);
  aae_detail::Matrix(image, features, parameters.dtype(), true);
  const auto tau = gumbel_detail::Temperature(temperature, parameters.dtype());
  if (!noise.defined() || noise.dim() != 3 || noise.size(0) != image.size(0) ||
      noise.size(1) != groups || noise.size(2) != classes || noise.dtype() != parameters.dtype())
    throw std::invalid_argument("explicit matching categorical Gumbels[B,G,C] required");
  if (image.size(0) > std::numeric_limits<std::int64_t>::max() / groups)
    throw std::invalid_argument("categorical row count overflow");
  aae_detail::Matrix(
      reshape(noise, {image.size(0) * groups, classes}), classes, parameters.dtype(), false);
  const auto n = image.size(0);
  const auto h = dense(parameters, "encoder", image, hidden, Activation::Relu);
  const auto flat_logits =
      reshape(dense(parameters, "categorical", h, width), {n * groups, classes});
  const auto sample = GumbelCategorical(flat_logits, reshape(noise, {n * groups, classes}), tau);
  GumbelCategoricalAutoencoderOutput out;
  out.logits             = reshape(flat_logits, {n, groups, classes});
  out.log_probability    = reshape(sample.log_probability, {n, groups, classes});
  out.probability        = reshape(sample.probability, {n, groups, classes});
  out.relaxed            = reshape(sample.relaxed, {n, groups, classes});
  out.indices            = cast(reshape(sample.indices, {n, groups}), DType::Int32);
  out.hard               = cast(one_hot(out.indices, classes), parameters.dtype());
  out.code               = reshape(out.relaxed, {n, width});
  const auto soft_logits = GaussianVaeDecode(parameters, out.code, features, width, hidden),
             hard_logits = GaussianVaeDecode(
                 parameters, reshape(out.hard, {n, width}), features, width, hidden);
  out.soft_prediction = sigmoid(soft_logits);
  out.hard_prediction = sigmoid(hard_logits);
  out.nll             = BernoulliPixelNll(soft_logits, image);
  out.hard_nll        = BernoulliPixelNll(hard_logits, image);
  out.kl              = sum(
      reshape(CategoricalUniformKl(sample.probability, sample.log_probability), {n, groups}), {1});
  out.loss           = mean(out.nll + out.kl, {0});
  out.hard_objective = mean(out.hard_nll + out.kl, {0});
  return out;
}
} // namespace datamunge::nn::models
