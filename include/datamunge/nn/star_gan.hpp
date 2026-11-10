#pragma once
/// @file star_gan.hpp
/// @brief Explicit three-domain U-Net, dual-head image critic and shared reconstruction game.
#include "datamunge/nn/pix2pix.hpp"
namespace datamunge::nn::models {
/// @brief Own the conditioned input and both actual StarGAN U-Net concatenation skips.
///
/// Nine flat Tensor fields retain storage and applicable source/weight graphs
/// after parents/stores are collected. code is detached tiled one-hot target[N,28,28,3];
/// input concatenates source then code. No target image, noise or fitted state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6301);
/// auto out=n::models::StarGanGenerate(g,n::zeros({2,28,28,1}),n::zeros({2},n::DType::Int64));
/// // input[2,28,28,4],joined14[2,14,14,8],joined28[2,28,28,6],image[2,28,28,1].
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see StarGanGenerate(), StarGanOutput
struct StarGanGenerated {
  Tensor code;     ///< Owning detached tiled one-hot target[N,28,28,3], Float32 source encoding.
  Tensor input;    ///< Owning live concat(source,code)[N,28,28,4], in source computation dtype.
  Tensor down14;   ///< Owning live first encoder ReLU[N,14,14,4].
  Tensor down7;    ///< Owning live second encoder ReLU[N,7,7,8].
  Tensor up14;     ///< Owning live first learned transpose/ReLU[N,14,14,4].
  Tensor joined14; ///< Owning live concat(up14,down14)[N,14,14,8].
  Tensor up28;     ///< Owning live second learned transpose/ReLU[N,28,28,2].
  Tensor joined28; ///< Owning live concat(up28,input)[N,28,28,6].
  Tensor image;    ///< Owning live3x3 sigmoid grayscale[N,28,28,1].
};
/// @brief Generate from source plus integer target domain through one conditioned two-skip U-Net.
///
/// Finite signed matching Float32/64 CPU source[N,28,28,1], N>0, strides allowed;
/// Int32/64 CPU target[N] in0..2. Validate domains before one-hot or parameter leaves.
/// Tile one_hot(target,3) over28x28 then concat(source,code). Biased normal(.02)
/// SAME4x4/stride2 ReLU encoder4->4->8; learned transpose4x4/stride2 ReLU8->4,
/// concat(up14,down14), transpose8->2, concat(up28,full conditioned input),
/// 3x3 sigmoid6->1. Source seed6301/names/order creates1609 values in10 leaves.
/// No normalization, dropout, noise, original-domain input or paired target image.
/// Only missing leaves initialize; no existing-leaf write, fitting, optimizer,
/// later RNG or hidden no-grad. Applicable first/higher source/weight paths remain
/// live through both actual skips; target is discrete/detached. Owning fields
/// retain graphs/storage. Inputs unchanged. Stored-leaf/math errors may follow
/// initialization; invalid supplied inputs fail before leaves. O(N) fixed-map work/
/// storage, synchronize shared stores. Code is Float32, cast to store dtype for concat.
/// @param generator Caller-owned floating G parameter store, source seed6301.
/// @param source Finite signed matching floating CPU[N,28,28,1], positive N.
/// @param target Aligned integer CPU[N] requested domains0..2.
/// @return Owning StarGanGenerated with input/code, all U-Net boundaries and sigmoid image.
/// @throws std::invalid_argument For invalid geometry, dtype, values, domain IDs or alignment.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6301);
/// auto x=n::zeros({1,28,28,1});auto requested=n::full({1},2,n::DType::Int32);
/// auto image=n::models::StarGanGenerate(g,x,requested).image;
/// // Source + requested domain only; G owns1609 values, without a training call.
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see StarGanGenerated, StarGan(), Pix2PixGenerate()
inline StarGanGenerated
StarGanGenerate(ParameterStore& generator, const Tensor& source, const Tensor& target) {
  pix_detail::Image(source, generator.dtype());
  if (source.size(0) > std::numeric_limits<std::int64_t>::max() / 4704)
    throw std::invalid_argument("Star generator activation product overflow");
  gan_detail::Labels(target, source.size(0), 3);
  StarGanGenerated o;
  o.code = tile(
      reshape(one_hot(cast(target, DType::Int64), 3), {source.size(0), 1, 1, 3}), {1, 28, 28, 1});
  o.input    = concat({source, cast(o.code, source.dtype())}, 3);
  o.down14   = pix_detail::Conv(generator, "down14", o.input, 4, 4, 2, Activation::Relu);
  o.down7    = pix_detail::Conv(generator, "down7", o.down14, 8, 4, 2, Activation::Relu);
  o.up14     = pix_detail::Conv(generator, "up14", o.down7, 4, 4, 2, Activation::Relu, true);
  o.joined14 = concat({o.up14, o.down14}, 3);
  o.up28     = pix_detail::Conv(generator, "up28", o.joined14, 2, 4, 2, Activation::Relu, true);
  o.joined28 = concat({o.up28, o.input}, 3);
  o.image    = pix_detail::Conv(generator, "output", o.joined28, 1, 3, 1, Activation::Sigmoid);
  gan_detail::Finite(o.image);
  return o;
}
/// @brief Own image-only source patches, domain logits/probabilities and critic feature boundaries.
///
/// Five flat live fields retain graph/storage beyond parent/store lifetime.
/// No target-domain feed, class label, noise, training flag or mutable batch state.
/// Domain probabilities are a three-class softmax; raw source patches are not probabilities.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore d(6311);
/// auto out=n::models::StarGanDiscriminate(d,n::ones({2,28,28,1}));
/// // source[2,7,7,1],domain_logits/domain_probability[2,3].
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see StarGanDiscriminate(), StarGanOutput
struct StarGanDiscriminated {
  Tensor source;             ///< Owning live raw49-patch source map[N,7,7,1].
  Tensor domain_logits;      ///< Owning live pooled three-domain raw logits[N,3].
  Tensor domain_probability; ///< Owning live softmax(domain_logits,-1)[N,3].
  Tensor down14;             ///< Owning live leaky.2 first map[N,14,14,4].
  Tensor down7;              ///< Owning live leaky.2 second map[N,7,7,8], shared by both heads.
};
/// @brief Score source patches and classify three domains from image alone with one shared trunk.
///
/// Finite signed matching Float32/64 CPU image[N,28,28,1], N>0, strides
/// allowed. Biased normal(.02) SAME4x4/stride2 conv1->4->8, leaky.2 after each;
/// mean spatial features[N,8], native Glorot dense3 domain head, then biased
/// normal3x3 source head8->1. Domain head initializes BEFORE source head, preserving
/// source seed6311/order:688 values in8 leaves. No domain/label/noise input or
/// batch statistics. Source49 raw scores have nominal18x18 receptive fields;
/// softmax probabilities apply only to the domain head. Applicable first/higher
/// image/weight paths live, both heads share the trunk. Only missing leaves
/// initialize; no fitting, optimizer writes, later RNG or hidden no-grad. Owning
/// fields retain graph/storage. Validate supplied image before leaves; stored-leaf/
/// math errors may follow initialization. O(N) work/storage; synchronize stores.
/// @param critic Caller-owned floating D store, source seed6311.
/// @param image Finite signed matching floating CPU[N,28,28,1], positive N.
/// @return Owning StarGanDiscriminated with raw source, raw domains, probabilities and trunk maps.
/// @throws std::invalid_argument For invalid geometry, dtype, values, domain IDs or alignment.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore d(6311);
/// auto fields=n::models::StarGanDiscriminate(d,n::zeros({1,28,28,1}));
/// // Image-only inference initializes688 values/8 leaves, without an optimizer.
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see StarGanDiscriminated, StarDomainLoss(), StarGan()
inline StarGanDiscriminated StarGanDiscriminate(ParameterStore& critic, const Tensor& image) {
  pix_detail::Image(image, critic.dtype());
  StarGanDiscriminated o;
  o.down14             = leaky_relu(pix_detail::Conv(critic, "down14", image, 4, 4, 2), .2);
  o.down7              = leaky_relu(pix_detail::Conv(critic, "down7", o.down14, 8, 4, 2), .2);
  o.domain_logits      = dense(critic, "domain", mean(o.down7, {1, 2}), 3);
  o.source             = pix_detail::Conv(critic, "source", o.down7, 1, 3, 1);
  o.domain_probability = softmax(o.domain_logits, -1);
  gan_detail::Finite(o.source);
  gan_detail::Finite(o.domain_logits);
  gan_detail::Finite(o.domain_probability);
  return o;
}
/// @brief Compute source-grouped stable mean three-domain softmax cross entropy.
///
/// Finite Float32/64 CPU logits[N,3], N>0, strides allowed; aligned Int32/64
/// CPU target[N] in0..2 checked before one-hot. mean(-sum(one_hot(target,3)*
/// log_softmax(logits,-1),{1})), with source grouping, no probability clipping.
/// All applicable first/higher logits paths live; integer targets detached.
/// No fitting, mutation, RNG or state; owning scalar retains graph/storage.
/// O(N*3) work/storage; synchronize external input writes; nonfinite loss rejected.
/// @param logits Finite floating CPU[N,3] raw domain scores, positive N.
/// @param target Aligned integer CPU[N] domain IDs0..2.
/// @return Owning live mean cross entropy in logits dtype.
/// @throws std::invalid_argument For invalid geometry, dtype, values, domain IDs or alignment.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto loss=n::models::StarDomainLoss(n::zeros({2,3}),n::zeros({2},n::DType::Int32));
/// // Uniform three-way scores give log(3), approximately1.0986123.
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see StarGanDiscriminate(), StarGan()
inline Tensor StarDomainLoss(const Tensor& logits, const Tensor& target) {
  gan_detail::Matrix(logits, 3, logits.dtype());
  gan_detail::Labels(target, logits.size(0), 3);
  auto loss = mean(-sum(one_hot(cast(target, DType::Int64), 3) * log_softmax(logits, -1), {1}));
  gan_detail::Finite(loss);
  return loss;
}
/// @brief Own all fifteen source translation, reconstruction, source/domain and objective fields.
///
/// Flat live Tensor handles retain applicable first/higher paths and storage
/// beyond parent/store lifetime. Both generator invocations share one store; both
/// critic heads share one trunk. No captured partial, optimizer, state or schedule.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6301),d(6311);
/// auto
/// out=n::models::StarGan(g,d,n::ones({1,28,28,1}),n::zeros({1},n::DType::Int32),n::ones({1},n::DType::Int64));
/// // fake/reconstruction[1,28,28,1]; generator_loss includes the shared G round trip.
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see StarGan(), StarGanGenerated, StarGanDiscriminated
struct StarGanOutput {
  Tensor fake;                    ///< Owning live G(source,target)[N,28,28,1].
  Tensor reconstruction;          ///< Owning live G(fake,original)[N,28,28,1], same G store.
  Tensor real_source;             ///< Owning live real raw source patches[N,7,7,1].
  Tensor fake_source;             ///< Owning live fake raw source patches[N,7,7,1].
  Tensor real_domain_logits;      ///< Owning live real pooled domain logits[N,3].
  Tensor fake_domain_logits;      ///< Owning live fake pooled domain logits[N,3].
  Tensor real_domain_probability; ///< Owning live real domain softmax[N,3].
  Tensor fake_domain_probability; ///< Owning live fake domain softmax[N,3].
  Tensor source_loss_g;           ///< Owning live full fake-to1 patch MSE mean, no half.
  Tensor source_loss_d;           ///< Owning live half(real-to1 mean plus fake-to0 mean).
  Tensor real_domain_loss;        ///< Owning live real classification CE against original domains.
  Tensor fake_domain_loss;        ///< Owning live fake classification CE against requested domains.
  Tensor reconstruction_loss;     ///< Owning live mean absolute shared-G roundtrip error.
  Tensor generator_loss; ///< Owning live(source_loss_g+fake_domain_loss)+10*reconstruction_loss.
  Tensor discriminator_loss; ///< Owning live source_loss_d+real_domain_loss; no fake CE term.
};
/// @brief Evaluate the explicit two-store three-domain game with a shared-generator round trip.
///
/// Distinct matching Float32/64 G/D stores; finite signed matching CPU
/// source[N,28,28,1], positive N, strides allowed. Int32/64 CPU original/target[N]
/// in0..2; each may differ by row and target may equal original. Validate all
/// supplied images/domains and aliases before leaves. Source seeds6301/6311,
/// G1609/10 and D688/8 values/leaves,2297/18 total. G(source,target) returns fake;
/// G(fake,original) reconstructs source with the SAME G. D reads image alone.
/// G=(full fake-to1 source mean+fake-domain CE)+10*mean(abs(reconstruction-source));
/// D=half(real-to1 source mean+fake-to0 source mean)+real-domain CE. Preserve source
/// grouping; known transformed images/digit labels never enter either loss.
/// All applicable first/higher source/parameter paths live, including BOTH G calls
/// in reconstruction and current D in G partials. No hidden detach/capture,
/// optimizer, fitting, no-grad, batch state, noise or RNG after initialization.
/// Caller captures/steps D, then makes a fresh G forward through updated D with
/// same source/original/target batch and steps only G. Existing leaves/inputs
/// unchanged, only missing leaves initialize. Owning fields retain storage/graphs.
/// Stored-leaf/math errors may follow initialization. O(N) fixed-map work/storage;
/// synchronize shared stores.
/// @param generator Distinct floating G store, source seed6301.
/// @param critic Distinct same-dtype D store, source seed6311.
/// @param source Finite signed matching floating CPU[N,28,28,1], positive N.
/// @param original Aligned integer CPU[N] observed source domains0..2.
/// @param target Aligned integer CPU[N] requested output domains0..2.
/// @return Owning StarGanOutput with fifteen live source fields and explicit objectives.
/// @throws std::invalid_argument For invalid geometry, dtype, values, domain IDs or alignment.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore g(6301),d(6311);
/// auto x=n::zeros({2,28,28,1});auto
/// o=n::zeros({2},n::DType::Int32),t=n::ones({2},n::DType::Int32); auto
/// out=n::models::StarGan(g,d,x,o,t);auto dd=n::grad(out.discriminator_loss,d.parameters());
/// // Caller applies D partials, makes a fresh forward, then captures/applies G partials.
/// @endcode
/// @par References
/// - Choi,Choi,Kim,Ha,Kim and Choo (2018), StarGAN: Unified Generative Adversarial Networks for
/// Multi-Domain Image-to-Image Translation, https://arxiv.org/abs/1711.09020 . Compact teaching
/// domains/maps differ from the paper architecture.
/// @see StarGanGenerate(), StarGanDiscriminate(), StarDomainLoss(), SplitStarDomains()
inline StarGanOutput StarGan(
    ParameterStore& generator,
    ParameterStore& critic,
    const Tensor&   source,
    const Tensor&   original,
    const Tensor&   target) {
  if (&generator == &critic || generator.dtype() != critic.dtype())
    throw std::invalid_argument("StarGAN needs distinct matching G/D stores");
  pix_detail::Image(source, generator.dtype());
  gan_detail::Labels(original, source.size(0), 3);
  gan_detail::Labels(target, source.size(0), 3);
  StarGanOutput o;
  o.fake           = StarGanGenerate(generator, source, target).image;
  o.reconstruction = StarGanGenerate(generator, o.fake, original).image;
  auto real = StarGanDiscriminate(critic, source), generated = StarGanDiscriminate(critic, o.fake);
  o.real_source             = real.source;
  o.fake_source             = generated.source;
  o.real_domain_logits      = real.domain_logits;
  o.fake_domain_logits      = generated.domain_logits;
  o.real_domain_probability = real.domain_probability;
  o.fake_domain_probability = generated.domain_probability;
  o.source_loss_g           = mean(square(generated.source - 1.));
  o.source_loss_d       = .5 * (mean(square(real.source - 1.)) + mean(square(generated.source)));
  o.real_domain_loss    = StarDomainLoss(real.domain_logits, original);
  o.fake_domain_loss    = StarDomainLoss(generated.domain_logits, target);
  o.reconstruction_loss = mean(abs(o.reconstruction - source));
  o.generator_loss      = (o.source_loss_g + o.fake_domain_loss) + 10. * o.reconstruction_loss;
  o.discriminator_loss  = o.source_loss_d + o.real_domain_loss;
  gan_detail::Finite(o.generator_loss);
  gan_detail::Finite(o.discriminator_loss);
  return o;
}
} // namespace datamunge::nn::models
