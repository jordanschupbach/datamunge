#pragma once
/// @file info_gan.hpp
/// @brief Explicit generated-code priors, separate image-only recognition and live InfoGAN
/// objectives.
#include <random>

#include "datamunge/nn/auxiliary_classifier_gan.hpp"
#include "datamunge/nn/dense_gan.hpp"

namespace datamunge::nn::models {
/// @brief Internal positive code geometry and bounded prior-value checks before leaf creation.
namespace info_gan_detail {
/// @brief Check positive representable concatenated noise/category/continuous widths.
/// @param noise Positive noise or image width.
/// @param classes Positive categorical cardinality C.
/// @param continuous Positive continuous-code width U.
/// @return Sum of the three positive widths.
/// @throws std::invalid_argument For nonpositive widths or int64 sum overflow.
inline std::int64_t Width(std::int64_t noise, std::int64_t classes, std::int64_t continuous) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (noise <= 0 || classes <= 0 || continuous <= 0 || noise > limit - classes ||
      noise + classes > limit - continuous)
    throw std::invalid_argument("InfoGAN widths must be positive and representable");
  return noise + classes + continuous;
}
/// @brief Validate finite matching CPU continuous-prior rows in the closed unit interval.
/// @param code Float32/64 CPU[rows,U], U>0, strides allowed.
/// @param rows Required positive number of generated rows.
/// @param type Required floating computation dtype.
/// @throws std::invalid_argument For invalid geometry/dtype/values or values outside[-1,1].
/// @throws std::logic_error For non-CPU storage.
inline void Continuous(const Tensor& code, std::int64_t rows, DType type) {
  if (!code.defined() || code.dim() != 2 || code.size(0) != rows)
    throw std::invalid_argument("InfoGAN needs one continuous-prior row per generated image");
  gan_detail::Matrix(code, code.size(1), type);
  for (double v : code.contiguous().to_vector<double>())
    if (std::abs(v) > 1.)
      throw std::invalid_argument("InfoGAN continuous prior must lie in[-1,1]");
}
/// @brief Check the three recognition-head dense products and their activation products.
/// @param rows Positive number of images.
/// @param features Positive image width F.
/// @param classes Positive class-head width C.
/// @param continuous Positive mean-head width U.
/// @param hidden Positive shared tanh width H.
/// @throws std::invalid_argument For invalid axes or int64 sum/product overflow.
inline void Heads(
    std::int64_t rows,
    std::int64_t features,
    std::int64_t classes,
    std::int64_t continuous,
    std::int64_t hidden) {
  Width(features, classes, continuous);
  dense_gan_detail::Geometry(rows, features, classes + continuous, hidden);
}
} // namespace info_gan_detail

/// @brief Own categorical codes, continuous codes and the accepted raw words of an explicit prior.
///
/// category is Int32[N], continuous is Float32/64[N,U], raw is Float64[N,U+1].
/// Raw column0 holds the accepted categorical word; subsequent columns hold
/// the continuous words. Rejected words are not stored. All fields own storage
/// after this record is destroyed, have no autograd history, and retain no RNG.
/// Synchronize external mutation of shared storage. Consumers revalidate codes.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto codes=dm::nn::models::InfoPrior(3,5382);
/// // category:[3], continuous:[3,2], raw:[3,3]; no digit label is an input.
/// @endcode
/// @see InfoPrior(), InfoGanGenerate(), InfoGan()
struct InfoCodes {
  Tensor category;   ///< Owning Int32[N] generated categorical IDs, unrelated to dataset labels.
  Tensor continuous; ///< Owning floating[N,U] uniform codes, with final-cast rounding.
  Tensor raw; ///< Owning Float64[N,U+1] exact accepted uint32 words, represented without precision
              ///< loss.
};

/// @brief Sample row-interleaved uniform categorical and continuous InfoGAN prior codes.
///
/// A local std::mt19937(seed) supplies an accepted category word v below
/// floor(2^32/C)*C, then U continuous words, row by row. category=v modulo C.
/// Each continuous value is computed in Double as2*((word+.5)/2^32)-1,
/// then cast once to dtype. Float32 rounding can attain-1 or1; values therefore
/// lie in[-1,1]. Defaults C10/U2 reproduce the source stream including its
/// rejection limit4294967290. This differs from separately seeded draws for
/// each component. Category frequencies are random, not forced balanced.
/// All returned tensors own lifetime and are detached. No global RNG, data,
/// fitting, optimizer or mutable shared state is accessed; independent calls
/// are reentrant. Work/storage O(N*(U+1)), with rejection sampling of categories.
/// @param rows Positive sample count N with representable output products.
/// @param seed Explicit uint32 local engine seed; no global seed is consumed.
/// @param classes Cardinality C in[1,INT32_MAX], default10, matching Int32 output IDs.
/// @param continuous Positive continuous width U, default2.
/// @param dtype Float32(default) or Float64 continuous output; raw stays Float64.
/// @return Owning InfoCodes with category[N],continuous[N,U],raw[N,U+1].
/// @throws std::invalid_argument For invalid counts, dtype, axes or int64 products.
/// @throws std::length_error If standard output containers cannot represent the requested capacity.
/// @throws std::bad_alloc If output storage cannot be allocated.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto a=dm::nn::models::InfoPrior(5,5382);
/// const auto b=dm::nn::models::InfoPrior(5,5382);
/// // All three a/b fields are bit exact, with no other random stream affected.
/// @endcode
/// @par References
/// - Chen et al. (2016), InfoGAN, https://arxiv.org/abs/1606.03657, section7.2.
///   The explicit row-wise engine, raw audit words and rejection protocol follow
///   the migrated teaching source rather than prescribing the paper's RNG.
/// @see MidpointUniformNoise(), InfoGanGenerate(), InfoCodes
inline InfoCodes InfoPrior(
    std::int64_t  rows,
    std::uint32_t seed,
    std::int64_t  classes    = 10,
    std::int64_t  continuous = 2,
    DType         dtype      = DType::Float32) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (rows <= 0 || classes <= 0 || classes > std::numeric_limits<std::int32_t>::max() ||
      continuous <= 0 || continuous == limit || rows > limit / (continuous + 1) ||
      (dtype != DType::Float32 && dtype != DType::Float64))
    throw std::invalid_argument("InfoGAN prior axes/dtype must be positive and representable");
  std::vector<std::int32_t> category(static_cast<std::size_t>(rows));
  std::vector<double>       values(static_cast<std::size_t>(rows * continuous)),
      raw(static_cast<std::size_t>(rows * (continuous + 1)));
  std::mt19937        rng(seed);
  const std::uint64_t domain   = std::uint64_t{1} << 32,
                      accepted = (domain / static_cast<std::uint64_t>(classes)) *
                                 static_cast<std::uint64_t>(classes);
  for (std::int64_t row = 0; row < rows; ++row) {
    std::uint32_t word;
    do
      word = rng();
    while (static_cast<std::uint64_t>(word) >= accepted);
    category[row] = static_cast<std::int32_t>(word % static_cast<std::uint64_t>(classes));
    raw[row * (continuous + 1)] = word;
    for (std::int64_t axis = 0; axis < continuous; ++axis) {
      word                                   = rng();
      raw[row * (continuous + 1) + axis + 1] = word;
      values[row * continuous + axis]        = 2 * ((double(word) + .5) / 4294967296.) - 1.;
    }
  }
  return {
      from_vector(category, {rows}),
      cast(from_vector(values, {rows, continuous}), dtype),
      from_vector(raw, {rows, continuous + 1})};
}

/// @brief Generate sigmoid pixels from explicit noise and generated-prior codes only.
///
/// Concatenates(noise[M,Z],onehot(category,C),continuous[M,U]) in that order
/// and reuses MlpGanGenerate(). All axes are positive; finite signed Float32/64
/// CPU noise and unit-interval continuous codes match the store dtype, while
/// categories are Int32/64[M] in[0,C). Strides and continuous first/higher paths
/// are retained. No image, real digit label, D/Q, loss, fit or input RNG is hidden.
/// Hidden/output W,b creation with Z16,C10,U2,H32,F784 gives26800 values/four
/// leaves. Input/geometry checks precede leaves; incompatible existing leaves
/// or nonfinite output may fail later. Output owns lifetime, inputs unchanged.
/// Work O(M*H*(Z+C+U+F)), storage O(M*(Z+C+U+H+F)); synchronize shared stores.
/// @param generator Mutable floating G store.
/// @param noise Finite matching CPU[M,Z], M,Z>0.
/// @param category Integer generated categories[M], not supervised digit labels.
/// @param continuous Finite matching CPU[M,U] in[-1,1], U>0.
/// @param features Positive sigmoid output width F, default784.
/// @param classes Positive categorical cardinality C, default10.
/// @param hidden Positive hidden tanh width H, default32.
/// @return Owning live generated coordinates[M,F].
/// @throws std::invalid_argument For invalid codes/geometry/dtypes, incompatible leaves or
/// nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5303);const auto c=n::models::InfoPrior(3,5351);
/// const auto image=n::models::InfoGanGenerate(g,n::zeros({3,16}),c.category,c.continuous);
/// // image:[3,784]; no training image or recognizer enters this call.
/// @endcode
/// @par References
/// - Chen et al. (2016), InfoGAN, https://arxiv.org/abs/1606.03657, latent-code input.
///   This source uses a compact dense generator rather than the paper's convolutional model.
/// @see MlpGanGenerate(), InfoPrior(), InfoGan()
inline Tensor InfoGanGenerate(
    ParameterStore& generator,
    const Tensor&   noise,
    const Tensor&   category,
    const Tensor&   continuous,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    hidden   = 32) {
  if (!noise.defined() || noise.dim() != 2 || !continuous.defined() || continuous.dim() != 2)
    throw std::invalid_argument("InfoGAN noise/continuous codes must be matrices");
  const auto width = info_gan_detail::Width(noise.size(1), classes, continuous.size(1));
  dense_gan_detail::Geometry(noise.size(0), features, width, hidden);
  gan_detail::Matrix(noise, noise.size(1), generator.dtype());
  info_gan_detail::Continuous(continuous, noise.size(0), generator.dtype());
  gan_detail::Labels(category, noise.size(0), classes);
  return MlpGanGenerate(
      generator,
      concat(
          {noise,
           cast(one_hot(cast(category, DType::Int64), classes), generator.dtype()),
           continuous},
          1),
      features,
      hidden);
}

/// @brief Own live image-only categorical predictions and continuous means from Q.
///
/// logits/probability are[N,C]; mean is[N,U], unbounded with fixed unit
/// recognition variance. All handles retain storage and graph history after
/// this record/store is destroyed. No code target, noise, digit label, D/G or
/// optimizer is retained. Probability is a convenience field; loss uses logits.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore q(5317);
/// const auto out=dm::nn::models::InfoGanRecognize(q,dm::nn::zeros({2,784}));
/// // logits/probability:[2,10], mean:[2,2], all from images alone.
/// @endcode
/// @see InfoGanRecognize(), InfoGanRecognitionLoss()
struct InfoRecognition {
  Tensor logits;      ///< Owning live categorical logits[N,C].
  Tensor probability; ///< Owning live row-softmax predictions[N,C], not supervised digit
                      ///< probabilities.
  Tensor mean;        ///< Owning live unbounded unit-Gaussian mean predictions[N,U].
};

/// @brief Recognize generated-prior codes from an image alone with an independent tanh Q network.
///
/// Hidden tanh H feeds categorical logits C then continuous means U. Parameter
/// creation order is hidden/W,b, category/W,b, continuous/W,b. Defaults give
/// 25516 values/six leaves. Q shares no parameters with D; no target/noise/code
/// is an input. Finite signed Float32/64 CPU[N,F] matches the store dtype;
/// strides and continuous first/higher paths remain live. Positive geometry
/// validates before leaves; incompatible existing leaves/nonfinite output may
/// fail later. Inputs unchanged; outputs own lifetime. Work O(N*H*(F+C+U)),
/// storage O(N*(H+C+U)); synchronize callers sharing a mutable store.
/// @param recognizer Mutable matching floating Q store, separate from D.
/// @param image Finite signed CPU[N,F], N>0, matching store dtype.
/// @param features Positive image width F, default784.
/// @param classes Positive categorical width C, default10.
/// @param continuous Positive linear mean width U, default2.
/// @param hidden Positive shared tanh width H, default32.
/// @return Owning InfoRecognition with logits,softmax probability and means.
/// @throws std::invalid_argument For invalid inputs/geometry, incompatible leaves or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore q(5317);
/// const auto out=dm::nn::models::InfoGanRecognize(q,dm::nn::ones({3,784}));
/// // No original code, category target, G or D is consulted.
/// @endcode
/// @par References
/// - Chen et al. (2016), InfoGAN, https://arxiv.org/abs/1606.03657, section6.
///   This teaching variant uses a separate Q rather than a shared D/Q trunk.
/// @see InfoGanRecognitionLoss(), InfoGan(), AuxiliaryGanDiscriminate()
inline InfoRecognition InfoGanRecognize(
    ParameterStore& recognizer,
    const Tensor&   image,
    std::int64_t    features   = 784,
    std::int64_t    classes    = 10,
    std::int64_t    continuous = 2,
    std::int64_t    hidden     = 32) {
  gan_detail::Matrix(image, features, recognizer.dtype());
  info_gan_detail::Heads(image.size(0), features, classes, continuous, hidden);
  auto            h      = dense(recognizer, "hidden", image, hidden, Activation::Tanh);
  auto            logits = dense(recognizer, "category", h, classes);
  auto            mean   = dense(recognizer, "continuous", h, continuous);
  InfoRecognition out{logits, softmax(logits), mean};
  gan_detail::Finite(out.logits);
  gan_detail::Finite(out.probability);
  gan_detail::Finite(out.mean);
  return out;
}

/// @brief Own categorical, unit-Gaussian, total recognition and variational-bound scalars.
///
/// All four fields own storage and applicable graphs after this record is
/// destroyed. Bound=log(C)+U*log(2)-total assumes independent uniform category
/// and uniform[-1,1] continuous priors. It may be negative and is not an actual
/// mutual-information measurement or proof of disentanglement. No fitting state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore q(5317);const auto codes=n::models::InfoPrior(2,5351);
/// const auto pred=n::models::InfoGanRecognize(q,n::zeros({2,784}));
/// const auto loss=n::models::InfoGanRecognitionLoss(pred,codes.category,codes.continuous);
/// // Q minimizes loss.total; the diagnostic lower_bound differs by prior entropy.
/// @endcode
/// @see InfoGanRecognitionLoss(), InfoGanOutput
struct InfoRecognitionLoss {
  Tensor categorical; ///< Owning scalar mean categorical CE against generated-prior IDs.
  Tensor continuous;  ///< Owning scalar unit-Gaussian NLL, summed over U then averaged over rows.
  Tensor total;       ///< Owning scalar categorical+continuous recognition objective R.
  Tensor lower_bound; ///< Owning scalar log(C)+U*log(2)-R, possibly negative.
};

/// @brief Compute generated-code recognition NLL and its fixed-prior variational information bound.
///
/// Reuses AuxiliaryClassLoss(logits,category). Continuous NLL is
/// .5*mean(sum((target-mean)^2,axis1))+.5*U*log(2*pi), preserving the source
/// grouping and log(2*pi) constant at U2. R=category NLL+continuous NLL;
/// lower_bound=(log(C)+U*log(2))-R. Priors are assumed independent uniform
/// categories and continuous[-1,1] coordinates; values alone cannot establish
/// that sampling assumption. Unit recognition variance is fixed, not fitted.
/// Q.probability is not consumed or validated. Finite Float32/64 CPU logits[N,C],
/// mean/target[N,U] share dtype and positive N,C,U; Int32/64 category[N] is
/// in[0,C). Target continuous codes lie in[-1,1]. Strides and first/higher paths
/// through logits,mean and target remain live; categories are discrete.
/// Inputs unchanged; scalar outputs own lifetime. No fit, RNG, optimizer or
/// posterior sampler. Work/storage O(N*(C+U)); synchronize external mutation.
/// Finite inputs whose squared residual/reduction overflows are rejected.
/// @param prediction Q logits and means; probability convenience field is unused.
/// @param category Generated-prior integer targets[N], not supervised image labels.
/// @param continuous Matching floating continuous-prior targets[N,U] in[-1,1].
/// @return Owning four-field InfoRecognitionLoss in computation dtype.
/// @throws std::invalid_argument For invalid shapes/dtypes/ranges or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::models::InfoRecognition q{n::zeros({2,2}),{},n::zeros({2,1})};
/// const auto
/// loss=n::models::InfoGanRecognitionLoss(q,n::tensor({0,1},{2},n::DType::Int64),n::tensor({-1,1},{2,1}));
/// // categorical=log2, continuous=(1+log(2*pi))/2; lower_bound about-.72579.
/// @endcode
/// @par References
/// - Chen et al. (2016), InfoGAN, https://arxiv.org/abs/1606.03657, equation5.
///   This source fixes Gaussian recognition variance at one and uses a separate Q.
/// @see AuxiliaryClassLoss(), InfoGanRecognize(), InfoGan()
inline InfoRecognitionLoss InfoGanRecognitionLoss(
    const InfoRecognition& prediction,
    const Tensor&          category,
    const Tensor&          continuous) {
  if (!prediction.logits.defined() || prediction.logits.dim() != 2 || !prediction.mean.defined() ||
      prediction.mean.dim() != 2)
    throw std::invalid_argument("InfoGAN recognition logits and means must be matrices");
  const auto type = prediction.logits.dtype();
  const auto rows = prediction.logits.size(0), classes = prediction.logits.size(1),
             width = prediction.mean.size(1);
  gan_detail::Matrix(prediction.logits, classes, type);
  gan_detail::Matrix(prediction.mean, width, type);
  if (prediction.mean.size(0) != rows)
    throw std::invalid_argument("InfoGAN recognition rows must match");
  info_gan_detail::Continuous(continuous, rows, type);
  if (continuous.size(1) != width)
    throw std::invalid_argument("InfoGAN continuous target width must match means");
  gan_detail::Labels(category, rows, classes);
  auto cat  = AuxiliaryClassLoss(prediction.logits, category);
  auto cont = .5 * mean(sum(square(continuous - prediction.mean), {1})) +
              (.5 * double(width)) * std::log(2 * std::acos(-1.));
  auto                total = cat + cont;
  InfoRecognitionLoss out{
      cat, cont, total, (std::log(double(classes)) + double(width) * std::log(2.)) - total};
  gan_detail::Finite(out.continuous);
  gan_detail::Finite(out.total);
  gan_detail::Finite(out.lower_bound);
  return out;
}

/// @brief Own fifteen live separate-Q InfoGAN endpoints without an experiment runner.
///
/// Every Tensor retains storage/graph after this record or its G/D/Q stores
/// are destroyed. Q sees fake images alone; category and continuous codes enter
/// G and the recognition loss, not D. No true digit label, checkpoint, optimizer,
/// fitting counter or quadratic MMD is retained. The bound may be negative.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5303),d(5309),q(5317);const auto c=n::models::InfoPrior(3,5351);
/// const auto
/// out=n::models::InfoGan(g,d,q,n::zeros({2,784}),n::zeros({3,16}),c.category,c.continuous);
/// // real logits have2 rows; fake/Q fields have3. No MMD enters fitting.
/// @endcode
/// @see InfoGan(), InfoRecognitionLoss
struct InfoGanOutput {
  Tensor fake;               ///< Live sigmoid generated coordinates[M,F].
  Tensor real_logit;         ///< Live image-only D source logits[N,1].
  Tensor fake_logit;         ///< Live image-only D source logits[M,1].
  Tensor real_probability;   ///< Live sigmoid real-source reporting values[N,1].
  Tensor fake_probability;   ///< Live sigmoid fake-source reporting values[M,1].
  Tensor discriminator_loss; ///< Live scalar stable logistic D objective.
  Tensor generator_loss;     ///< Live scalar non-saturating source G loss plus recognition R, with
                             ///< weight1.
  Tensor source_generator_loss; ///< Live scalar mean softplus(-fake_logit), without recognition.
  Tensor recognition_loss;      ///< Live scalar categorical CE plus unit-Gaussian continuous NLL.
  Tensor categorical_loss;      ///< Live scalar generated-category mean CE.
  Tensor continuous_loss;       ///< Live scalar per-row summed unit-Gaussian NLL.
  Tensor
      mi_lower_bound; ///< Live fixed-prior entropy minus R; not measured actual mutual information.
  Tensor q_logits;    ///< Live separate image-only Q categorical logits[M,C].
  Tensor q_probability; ///< Live Q softmax generated-prior predictions[M,C].
  Tensor q_mean;        ///< Live Q unbounded unit-Gaussian means[M,U].
};

/// @brief Evaluate explicit InfoGAN inputs with separate G,D,Q stores and caller-owned writes.
///
/// InfoGanGenerate() maps(noise,category,continuous) to sigmoid pixels;
/// MlpGanDiscriminate() scores real and fake images alone; InfoGanRecognize()
/// feeds only fake images to independent Q. D minimizes LogisticGanObjectives().
/// Q minimizes R=InfoGanRecognitionLoss().total. G minimizes non-saturating
/// source loss+R with weight1; no prior entropy constant is subtracted from G.
/// The diagnostic bound is separately Hprior-R and may be negative.
/// Defaults Z16,C10,U2,H32,F784 give G26800+D25153+Q25516=77469 values in
/// fourteen leaves, each store creating hidden before heads; Q category before
/// continuous. Positive real N/fake M may differ. Finite signed Float32/64 CPU
/// images/noise/continuous codes match all three distinct stores; Int32/64[M]
/// categories are generated-prior IDs in[0,C), continuous[M,U] lies in[-1,1].
/// Strides and applicable first/higher continuous paths remain live. Continuous
/// input is also a recognition target, giving both direct and mapped paths.
/// All user input/options/geometry checks precede leaves; incompatible existing
/// leaves or computed nonfinite values can fail later. Inputs unchanged, fields
/// own lifetime. Synchronize shared stores. Caller recomputes D then Q then G
/// after each selected-store write using independent optimizer states. There is
/// no hidden RNG, fit, freeze, optimizer, step, checkpoint, real-label supervision
/// or MMD. Work O(M*H*(Z+C+U+F)+(N+M)*H*F+M*H*(F+C+U)); storage follows
/// dense activations plus host validation.
/// @param generator Mutable floating G store.
/// @param discriminator Distinct matching floating image-only D store.
/// @param recognizer Third distinct matching floating image-only Q store.
/// @param image Finite matching signed CPU[N,F] real coordinates, N>0.
/// @param noise Finite matching signed CPU[M,Z] explicit noise, M,Z>0.
/// @param category Int32/64 CPU[M] generated-prior IDs in[0,C).
/// @param continuous Matching floating CPU[M,U] prior/recognition targets in[-1,1], U>0.
/// @param features Positive image/output width F, default784.
/// @param classes Positive categorical cardinality C, default10.
/// @param hidden Positive tanh width H for each independent network, default32.
/// @return Owning fifteen-field InfoGanOutput with live endpoints and selected objectives.
/// @throws std::invalid_argument For invalid inputs/geometry, aliased/mismatched stores,
/// incompatible leaves or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5303),d(5309),q(5317);const auto c=n::models::InfoPrior(2,5382);
/// const auto
/// out=n::models::InfoGan(g,d,q,n::zeros({2,784}),n::zeros({2,16}),c.category,c.continuous); const
/// auto dq=n::grad(out.recognition_loss,q.parameters());
/// // Caller applies dq only to Q and recomputes before its next selected G step.
/// @endcode
/// @par References
/// - Chen, Duan, Houthooft, Schulman, Sutskever and Abbeel (2016), InfoGAN:
///   Interpretable Representation Learning by Information Maximizing Generative
///   Adversarial Nets, https://arxiv.org/abs/1606.03657, equations5-6.
///   The migrated teaching variant has a separate unit-variance Q and explicit D-Q-G schedule.
/// @see InfoPrior(), InfoGanGenerate(), InfoGanRecognize(), InfoGanRecognitionLoss()
inline InfoGanOutput InfoGan(
    ParameterStore& generator,
    ParameterStore& discriminator,
    ParameterStore& recognizer,
    const Tensor&   image,
    const Tensor&   noise,
    const Tensor&   category,
    const Tensor&   continuous,
    std::int64_t    features = 784,
    std::int64_t    classes  = 10,
    std::int64_t    hidden   = 32) {
  if (&generator == &discriminator || &generator == &recognizer || &discriminator == &recognizer ||
      generator.dtype() != discriminator.dtype() || generator.dtype() != recognizer.dtype())
    throw std::invalid_argument("InfoGAN requires three distinct matching floating stores");
  if (!noise.defined() || noise.dim() != 2 || !continuous.defined() || continuous.dim() != 2)
    throw std::invalid_argument("InfoGAN noise/continuous codes must be matrices");
  const auto width = info_gan_detail::Width(noise.size(1), classes, continuous.size(1));
  dense_gan_detail::Geometry(noise.size(0), features, width, hidden);
  info_gan_detail::Heads(noise.size(0), features, classes, continuous.size(1), hidden);
  gan_detail::Matrix(image, features, generator.dtype());
  dense_gan_detail::Geometry(image.size(0), features, 1, hidden);
  gan_detail::Matrix(noise, noise.size(1), generator.dtype());
  info_gan_detail::Continuous(continuous, noise.size(0), generator.dtype());
  gan_detail::Labels(category, noise.size(0), classes);
  auto fake = InfoGanGenerate(generator, noise, category, continuous, features, classes, hidden);
  auto real = MlpGanDiscriminate(discriminator, image, features, hidden),
       generated    = MlpGanDiscriminate(discriminator, fake, features, hidden);
  const auto source = LogisticGanObjectives(real, generated);
  const auto q = InfoGanRecognize(recognizer, fake, features, classes, continuous.size(1), hidden);
  const auto r = InfoGanRecognitionLoss(q, category, continuous);
  auto       total = source.generator + r.total;
  gan_detail::Finite(total);
  return {
      fake,
      real,
      generated,
      sigmoid(real),
      sigmoid(generated),
      source.discriminator,
      total,
      source.generator,
      r.total,
      r.categorical,
      r.continuous,
      r.lower_bound,
      q.logits,
      q.probability,
      q.mean};
}
} // namespace datamunge::nn::models
