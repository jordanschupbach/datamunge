#pragma once
/// @file relativistic_gan.hpp
/// @brief Live unequal-pool relativistic-average logistic game, with explicit half-scaled means.
#include "datamunge/nn/gan_models.hpp"
namespace datamunge::nn::models {
/// @brief Own relativistic-average D/G losses and both live pool-centered score fields.
///
/// Four flat Tensor fields retain storage and applicable first/higher real/fake
/// paths beyond parent collection. Pool means stay live, no stop/clipping/state.
/// No model, optimizer, parameter mutation or fitting is hidden.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto out=n::models::RelativisticAverageGan(n::zeros({3,1}),n::zeros({2,1}));
/// // Both losses are log(2); pool sizes may differ.
/// @endcode
/// @par References
/// - Ledig et al.(2017), Photo-Realistic Single Image Super-Resolution Using a Generative
/// Adversarial Network, https://arxiv.org/abs/1609.04802 ; Wang et al.(2018), ESRGAN: Enhanced
/// Super-Resolution Generative Adversarial Networks, https://arxiv.org/abs/1809.00219 . Fixed-width
/// x2 teaching adaptations with a bicubic image shortcut, not pretrained benchmark models.
/// @see RelativisticAverageGan(), AdversarialSR()
struct RelativisticGanLoss {
  Tensor discriminator; ///< Owning live half(mean softplus(-rr)+mean softplus(rf)).
  Tensor generator;     ///< Owning live half(mean softplus(rr)+mean softplus(-rf)).
  Tensor real_relative; ///< Owning live real-mean(fake)[N,1], including opposite-pool derivative.
  Tensor fake_relative; ///< Owning live fake-mean(real)[M,1], including opposite-pool derivative.
};
/// @brief Evaluate stable relativistic-average logistic losses with live opposite-pool means.
///
/// Finite matching Float32/64 CPU real[N,1],fake[M,1], positive counts; unequal
/// pools and strides allowed. rr=real-mean(fake),rf=fake-mean(real).
/// D=.5*(mean softplus(-rr)+mean softplus(rf));
/// G=.5*(mean softplus(rr)+mean softplus(-rf)), preserving source grouping.
/// Opposite-pool means remain live at every derivative order. A common score shift
/// cancels up to floating rounding. No sigmoid/log clipping, hidden detach, mutation,
/// RNG, fitting or optimizer. Owning outputs retain graphs/storage; inputs unchanged.
/// Nonfinite computed relatives/losses rejected. O(N+M) work/storage; synchronize
/// external writes. These source halves differ from LogisticGanObjectives defaults.
/// @param real Finite floating CPU[N,1] real raw scores, N>0.
/// @param fake Finite matching floating CPU[M,1] fake raw scores, M>0.
/// @return Owning RelativisticGanLoss with both losses and live rr/rf.
/// @throws std::invalid_argument For invalid geometry, dtype, values, alignment, products, aliased
/// stores or incompatible leaves.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto real=n::tensor({-2.,0.,3.},{3,1},n::DType::Float64).requires_grad_();
/// auto fake=n::tensor({-1.,2.},{2,1},n::DType::Float64).requires_grad_();
/// auto loss=n::models::RelativisticAverageGan(real,fake);
/// auto partial=n::grad(loss.generator,{real,fake});
/// // G has nonzero partials through BOTH pools, including their cross-pool means.
/// @endcode
/// @par References
/// - Jolicoeur-Martineau(2018), The relativistic discriminator: a key element missing from standard
/// GAN, https://arxiv.org/abs/1807.00734 ; Wang et al.(2018), https://arxiv.org/abs/1809.00219 .
/// Source SR teaching game keeps explicit half-scaled pool means.
/// @see RelativisticGanLoss, LogisticGanObjectives(), AdversarialSR()
inline RelativisticGanLoss RelativisticAverageGan(const Tensor& real, const Tensor& fake) {
  gan_detail::Matrix(real, 1, real.dtype());
  gan_detail::Matrix(fake, 1, real.dtype());
  auto rr = real - mean(fake), rf = fake - mean(real);
  gan_detail::Finite(rr);
  gan_detail::Finite(rf);
  auto d = .5 * (mean(softplus(-rr)) + mean(softplus(rf))),
       g = .5 * (mean(softplus(rr)) + mean(softplus(-rf)));
  gan_detail::Finite(d);
  gan_detail::Finite(g);
  return {d, g, rr, rf};
}
} // namespace datamunge::nn::models
