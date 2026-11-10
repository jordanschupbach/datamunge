#pragma once
/// @file wasserstein_gan.hpp
/// @brief Compact Wasserstein maps and differentiable critic-input penalty, without hidden writes.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/bidirectional_gan.hpp"

namespace datamunge::nn::models {
/// @brief Clip every tensor of a selected critic store, including biases, without an optimizer
/// step.
///
/// Requires a nonempty finite Float32/64 CPU store and finite positive bound
/// representable in its dtype, default.01. Validates all tensors and computes all
/// candidates before writing any; invalid input leaves the store unchanged.
/// Under NoGradGuard copies clip(value,-bound,bound) into every existing named
/// leaf, retaining names, order and gradient flags. No initialization, fitting,
/// optimizer/counter reset or other-store mutation. This is the clipped WGAN
/// constraint, including biases; it does not certify a1-Lipschitz critic.
/// Caller initializes the network and clips once before fitting, then clips
/// immediately after EVERY selected critic step, never a G step. Work/storage
/// O(parameter count); synchronize shared stores and do not reuse a pre-write graph.
/// @param critic Mutable nonempty finite floating critic parameter store.
/// @param bound Positive finite representable symmetric magnitude, default.01.
/// @throws std::invalid_argument For empty/nonfloating/nonfinite parameters or invalid bound.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore c(5503);
/// dm::nn::models::BiganMap(c,dm::nn::zeros({1,2}),1,16);
/// dm::nn::models::ClipWassersteinCritic(c);
/// // All four tensors, including both biases, lie in[-.01f,.01f].
/// @endcode
/// @par References
/// - Arjovsky,Chintala and Bottou (2017), Wasserstein GAN, https://arxiv.org/abs/1701.07875 .
/// @see WassersteinCritic(), WassersteinGan(), clip()
inline void ClipWassersteinCritic(ParameterStore& critic, double bound = .01) {
  const auto   type    = critic.dtype();
  const double rounded = type == DType::Float32 ? double(float(bound)) : bound;
  if (critic.size() == 0 || (type != DType::Float32 && type != DType::Float64) ||
      !std::isfinite(bound) || bound <= 0 || !std::isfinite(rounded) || rounded <= 0)
    throw std::invalid_argument(
        "Clipped WGAN needs finite nonempty parameters and positive representable bound");
  NoGradGuard         guard;
  std::vector<Tensor> candidates;
  const auto          names = critic.names();
  for (const auto& name : names) {
    auto value = critic.at(name);
    gan_detail::Finite(value);
    candidates.push_back(clip(value.detach(), -bound, bound));
  }
  for (std::size_t k = 0; k < names.size(); ++k)
    critic.set(names[k], candidates[k]);
}

/// @brief Map explicit two-coordinate noise through the compact tanh/linear Wasserstein generator.
///
/// Finite matching Float32/64 CPU noise[N,2], N>0, strides allowed. Delegates to
/// existing BiganMap() with hidden16/output2: hidden/W[2,16],hidden/b[16],
/// output/W[16,2],output/b[2],82 values/four leaves, source Glorot initialization.
/// Unbounded output[N,2] depends only on noise and G parameters, with applicable
/// first/higher paths. No clipping, input distribution, critic, optimizer, fit or
/// checkpoint hidden. Input checks precede leaves; stored-leaf/nonfinite math may
/// fail after earlier missing leaves initialize. Owning output; input unchanged.
/// O(N) work/storage for the fixed architecture; synchronize mutable stores.
/// @param generator Mutable matching floating generator owner.
/// @param noise Finite signed matching CPU[N,2], positive N and representable products.
/// @return Owning live generated coordinates[N,2], without bounded activation.
/// @throws std::invalid_argument For invalid geometry/dtype/products or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore g(5501);
/// const auto fake=dm::nn::models::WassersteinGenerate(g,dm::nn::ones({3,2}));
/// // fake is[3,2], without selecting a real-data row or fitting a model.
/// @endcode
/// @par References
/// - Arjovsky et al. (2017), https://arxiv.org/abs/1701.07875 .
/// @see WassersteinCritic(), WassersteinGan(), BiganMap()
inline Tensor WassersteinGenerate(ParameterStore& generator, const Tensor& noise) {
  gan_detail::Matrix(noise, 2, generator.dtype());
  return BiganMap(generator, noise, 2, 16);
}

/// @brief Score explicit two-coordinate points with the compact tanh/linear Wasserstein critic.
///
/// Finite matching Float32/64 CPU point[N,2], positive N, strides allowed. Reuses
/// BiganMap() hidden16/output1: four source-named leaves/65 values, Glorot kernels
/// and zero biases. Returns unbounded raw score[N,1], without sigmoid or class
/// probability. Applicable first/higher point/parameter paths live. No G call,
/// clipping, Lipschitz certificate, optimizer or fitting. Geometry/product checks
/// precede leaves; existing-leaf/nonfinite math errors may follow initialization.
/// Owning output; input unchanged, O(N) work/storage; synchronize stores.
/// @param critic Mutable matching floating critic owner.
/// @param point Finite signed matching CPU[N,2], N>0.
/// @return Owning live raw critic scores[N,1].
/// @throws std::invalid_argument For invalid geometry/dtype/products or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore c(5503);
/// const auto score=dm::nn::models::WassersteinCritic(c,dm::nn::zeros({3,2}));
/// // score is[3,1]; caller chooses clipping or a gradient penalty explicitly.
/// @endcode
/// @par References
/// - Arjovsky et al. (2017), https://arxiv.org/abs/1701.07875 .
/// @see WassersteinGenerate(), WassersteinGan(), ClipWassersteinCritic()
inline Tensor WassersteinCritic(ParameterStore& critic, const Tensor& point) {
  gan_detail::Matrix(point, 2, critic.dtype());
  return BiganMap(critic, point, 1, 16);
}

/// @brief Own eleven Wasserstein endpoints, including symbolic and analytic critic input gradients.
///
/// All fields own storage/applicable graphs after parent/store collection.
/// interpolation is detached; penalty stops all external real/alpha/G paths.
/// symbolic/penalty retain critic parameter history only when requested, while
/// input_gradient is a separately computed analytic Jacobian diagnostic. Base
/// critic loss still has a G path, so choose update groups explicitly. No mode,
/// optimizer, update counter or clipping callback is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5501),c(5503);
/// auto out=n::models::WassersteinGan(g,c,n::ones({2,2}),n::zeros({2,2}),n::full({2,1},.5));
/// // out.norm is[2]; penalty is the mean squared distance of each norm from1.
/// @endcode
/// @par References
/// - Gulrajani et al. (2017), https://arxiv.org/abs/1704.00028 .
/// @see WassersteinGan(), WassersteinGenerate(), WassersteinCritic()
struct WassersteinGanOutput {
  Tensor fake;           ///< Owning live G coordinates[N,2].
  Tensor real_score;     ///< Owning live real critic scores[N,1].
  Tensor fake_score;     ///< Owning live generated critic scores[N,1].
  Tensor interpolation;  ///< Owning detached alpha*real+(1-alpha)*stop(fake)[N,2].
  Tensor input_gradient; ///< Owning analytic critic Jacobian[N,2], with detached interpolation.
  Tensor
      symbolic_input_gradient; ///< Owning autograd critic Jacobian[N,2]; optional critic history.
  Tensor norm;                 ///< Owning sqrt(sum(symbolic_gradient^2,axis1)+1e-12)[N].
  Tensor penalty;    ///< Owning mean(square(norm-1)) scalar; no external real/alpha/G path.
  Tensor critic_gap; ///< Owning mean(real_score)-mean(fake_score), an unconverged game diagnostic.
  Tensor generator_loss; ///< Owning -mean(fake_score) scalar, unaffected by penalty weight.
  Tensor critic_loss;    ///< Owning -gap, plus positive penalty_weight*penalty when selected.
};

/// @brief Evaluate the compact Wasserstein game with an explicit differentiable critic-input
/// penalty.
///
/// Distinct matching Float32/64 CPU stores, finite real/noise[N,2] and alpha[N,1]
/// in[0,1], positive aligned rows and representable products. G/C reuse the source
/// tanh16 maps (82/65 values, eight tensors). Creates interpolation as a detached
/// leaf alpha*real+(1-alpha)*stop(fake); no external real/alpha/G penalty path.
/// Under EnableGradGuard differentiates sum(critic(interpolation)) with respect
/// to that leaf even when called under NoGradGuard. With penalty_graph=true
/// (default), create_graph retains critic parameter history through this input
/// derivative; false detaches it. Other outputs obey ambient recording. Outer
/// NoGradGuard plus penalty_graph=false yields entirely detached evaluation.
///
/// Also computes analytic J=((1-tanh(xhat*W1+b1)^2)*w2^T)*W1^T from detached
/// interpolation as an independent diagnostic. Norm=sqrt(sum(J_symbolic^2)+1e-12),
/// penalty=mean(square(norm-1)), gap=mean(real)-mean(fake), G=-mean(fake),
/// C=-gap+penalty_weight*penalty. Weight0 skips the added term but still returns
/// all diagnostics, preserving the clipped-source forward; default10 is WGAN-GP.
/// This fixed1e-12 guard and operation grouping preserve the source arithmetic.
///
/// No optimizer, clipping, fitting, freeze, checkpoint or distance certification.
/// Caller chooses parameter gradients, RMSProp/clipping or Adam/penalty, three
/// fresh-noise critic steps then a fresh G step. Applicable higher paths live;
/// no existing leaves are mutated. Checks precede creation; stored-leaf/nonfinite
/// math errors may occur later. Owning outputs, inputs unchanged. O(N) work/storage
/// for fixed width; synchronize stores and exclude concurrent input writes.
/// @param generator Mutable floating G owner, distinct from critic.
/// @param critic Mutable matching floating C owner.
/// @param real Finite signed matching CPU[N,2] real coordinates, N>0.
/// @param noise Finite signed matching CPU[N,2] explicit latent bank, same N.
/// @param alpha Finite matching CPU[N,1] interpolation coefficients in[0,1].
/// @param penalty_weight Finite nonnegative coefficient, default10;0 for clipped WGAN's base loss.
/// @param penalty_graph Retain critic history through symbolic input gradient when true(default);
/// detach it when false.
/// @return Owning eleven-field WassersteinGanOutput with both Jacobians and explicit objectives.
/// @throws std::invalid_argument For invalid geometry/stores/options/ranges or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore g(5501),c(5503);
/// auto
/// out=n::models::WassersteinGan(g,c,n::ones({2,2}),n::zeros({2,2}),n::full({2,1},.5),10.,true);
/// auto critic_gradient=n::grad(out.critic_loss,c.parameters());
/// // No Adam/clipping/update occurs; G gradient requires a fresh forward after C writes.
/// @endcode
/// @par References
/// - Arjovsky,Chintala and Bottou (2017), https://arxiv.org/abs/1701.07875 .
/// - Gulrajani,Ahmed,Arjovsky,Dumoulin and Courville (2017), Improved Training of
///   Wasserstein GANs, https://arxiv.org/abs/1704.00028 . The tiny teaching network
///   and three-critic-step schedule are the repository source's choices.
/// @see WassersteinGanOutput, WassersteinGenerate(), WassersteinCritic(), ClipWassersteinCritic()
inline WassersteinGanOutput WassersteinGan(
    ParameterStore& generator,
    ParameterStore& critic,
    const Tensor&   real,
    const Tensor&   noise,
    const Tensor&   alpha,
    double          penalty_weight = 10.,
    bool            penalty_graph  = true) {
  if (&generator == &critic || generator.dtype() != critic.dtype())
    throw std::invalid_argument("Wasserstein G and C stores must be distinct and matching");
  gan_detail::Matrix(real, 2, generator.dtype());
  gan_detail::Matrix(noise, 2, generator.dtype());
  gan_detail::Matrix(alpha, 1, generator.dtype());
  if (real.size(0) != noise.size(0) || real.size(0) != alpha.size(0) ||
      real.size(0) > std::numeric_limits<std::int64_t>::max() / 16 ||
      !std::isfinite(penalty_weight) || penalty_weight < 0)
    throw std::invalid_argument("Invalid Wasserstein row alignment, products or coefficient");
  for (double a : alpha.contiguous().to_vector<double>())
    if (a < 0 || a > 1)
      throw std::invalid_argument("Wasserstein alpha outside[0,1]");
  auto fake       = WassersteinGenerate(generator, noise);
  auto real_score = WassersteinCritic(critic, real), fake_score = WassersteinCritic(critic, fake);
  auto interpolation = (alpha * real + (1. - alpha) * stop_gradient(fake)).detach();
  interpolation.requires_grad_();
  Tensor symbolic;
  {
    EnableGradGuard enable;
    GradOptions     options;
    options.create_graph = penalty_graph;
    symbolic = grad(sum(WassersteinCritic(critic, interpolation)), {interpolation}, options)[0];
  }
  if (!penalty_graph)
    symbolic = symbolic.detach();
  auto hidden   = dense(critic, "hidden", interpolation.detach(), 16, Activation::Tanh);
  auto slope    = (1. - square(hidden)) * reshape(critic.at("output/W"), {1, 16});
  auto analytic = matmul(slope, critic.at("hidden/W"), false, true);
  auto norm     = sqrt(sum(square(symbolic), {1}) + 1e-12);
  auto penalty  = mean(square(norm - 1.));
  auto gap      = mean(real_score) - mean(fake_score);
  auto loss     = -gap;
  if (penalty_weight > 0)
    loss = loss + penalty_weight * penalty;
  WassersteinGanOutput out{
      fake,
      real_score,
      fake_score,
      interpolation.detach(),
      analytic,
      symbolic,
      norm,
      penalty,
      gap,
      -mean(fake_score),
      loss};
  for (const auto& value :
       {out.fake,
        out.real_score,
        out.fake_score,
        out.interpolation,
        out.input_gradient,
        out.symbolic_input_gradient,
        out.norm,
        out.penalty,
        out.critic_gap,
        out.generator_loss,
        out.critic_loss})
    gan_detail::Finite(value);
  return out;
}
} // namespace datamunge::nn::models
