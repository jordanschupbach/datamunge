#pragma once
/// @file cycle_gan.hpp
/// @brief Four-store unpaired CycleGAN with shared generator objectives and explicit caller-owned
/// writes.
#include <array>

#include "datamunge/nn/pix2pix.hpp"

namespace datamunge::nn::models {
/// @brief Score grayscale images with the source unconditional49-patch least-squares critic.
///
/// Finite signed matching Float32/64 CPU image[N,28,28,1], positive N, strides
/// allowed. Biased normal(.02) SAME4x4/stride2 conv1->4/leaky.2, then4->8/leaky.2,
/// then biased SAME3x3/stride1 raw8->1 patch scores[N,7,7,1]. Biases start zero;
/// source names/order create661 values in6 leaves. Source A/B critic seeds6217/
/// 6221. No condition concatenation, sigmoid, batch state or normalization.
///
/// Only absent leaves initialize; no fitting, optimizer, hidden no-grad or later
/// RNG. Applicable first/higher input/weight paths stay live. Owning output
/// retains graph/storage; input unchanged. Validate geometry before leaves;
/// stored-leaf/nonfinite-math failures may follow earlier initialization.
/// O(N) work/storage for fixed map; synchronize shared stores.
/// @param critic Caller-owned floating unconditional critic parameter store.
/// @param image Finite signed matching CPU[N,28,28,1], positive N.
/// @return Owning live raw critic patch map[N,7,7,1].
/// @throws std::invalid_argument For invalid inputs/leaves/products or nonfinite scores.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// dm::nn::ParameterStore a(6217);
/// auto score=dm::nn::models::CycleGanDiscriminate(a,dm::nn::ones({2,28,28,1}));
/// //49 raw scores per image;661 parameters, without a source/target pair input.
/// @endcode
/// @par References
/// - Zhu,Park,Isola and Efros (2017), Unpaired Image-to-Image Translation using
///   Cycle-Consistent Adversarial Networks, https://arxiv.org/abs/1703.10593 .
///   This repository uses a small18x18-receptive-field teaching critic.
/// @see CycleGan(), Pix2PixDiscriminate(), Pix2PixGenerate()
inline Tensor CycleGanDiscriminate(ParameterStore& critic, const Tensor& image) {
  pix_detail::Image(image, critic.dtype());
  auto first  = leaky_relu(pix_detail::Conv(critic, "down14", image, 4, 4, 2), .2);
  auto second = leaky_relu(pix_detail::Conv(critic, "down7", first, 8, 4, 2), .2);
  auto score  = pix_detail::Conv(critic, "output", second, 1, 3, 1);
  gan_detail::Finite(score);
  return score;
}

/// @brief Own both translations/cycles/identities, four patch maps and all explicit CycleGAN terms.
///
/// Nineteen flat Tensor fields retain storage/applicable graphs after parent or
/// store collection. Unpaired domain row counts may differ. No optimizer,
/// derivative capture, hidden stop/no-grad, mutable batch state or schedule.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore ab(6203),ba(6211),da(6217),db(6221);
/// auto out=n::models::CycleGan(ab,ba,da,db,n::zeros({2,28,28,1}),n::ones({3,28,28,1}));
/// // fake_b/cycle_a have2 rows; fake_a/cycle_b have3 rows. The batches are unpaired.
/// @endcode
/// @par References
/// - Zhu et al. (2017), https://arxiv.org/abs/1703.10593 .
/// @see CycleGan(), CycleGanDiscriminate(), Pix2PixGenerate()
struct CycleGanOutput {
  Tensor fake_a;          ///< Owning live G_ba(b)[M,28,28,1], from B source alone.
  Tensor fake_b;          ///< Owning live G_ab(a)[N,28,28,1], from A source alone.
  Tensor cycle_a;         ///< Owning live G_ba(G_ab(a))[N,28,28,1].
  Tensor cycle_b;         ///< Owning live G_ab(G_ba(b))[M,28,28,1].
  Tensor identity_a;      ///< Owning live G_ba(a)[N,28,28,1], source-domain identity control.
  Tensor identity_b;      ///< Owning live G_ab(b)[M,28,28,1], source-domain identity control.
  Tensor real_a_score;    ///< Owning live D_a(a)[N,7,7,1] raw patch scores.
  Tensor real_b_score;    ///< Owning live D_b(b)[M,7,7,1] raw patch scores.
  Tensor fake_a_score;    ///< Owning live D_a(fake_a)[M,7,7,1] raw patch scores.
  Tensor fake_b_score;    ///< Owning live D_b(fake_b)[N,7,7,1] raw patch scores.
  Tensor g_loss;          ///< Owning common G objective: adversarial sum+(cycle_weight*cycle
                          ///< sum+identity_weight*identity sum).
  Tensor d_a_loss;        ///< Owning half*(mean((real_a-1)^2)+mean(fake_a_score^2)).
  Tensor d_b_loss;        ///< Owning half*(mean((real_b-1)^2)+mean(fake_b_score^2)).
  Tensor adversarial_a;   ///< Owning mean((fake_a_score-1)^2), without half.
  Tensor adversarial_b;   ///< Owning mean((fake_b_score-1)^2), without half.
  Tensor cycle_loss_a;    ///< Owning mean(abs(cycle_a-a)), over all A pixels.
  Tensor cycle_loss_b;    ///< Owning mean(abs(cycle_b-b)), over all B pixels.
  Tensor identity_loss_a; ///< Owning mean(abs(identity_a-a)), over all A pixels.
  Tensor identity_loss_b; ///< Owning mean(abs(identity_b-b)), over all B pixels.
};

/// @brief Evaluate the unpaired four-store game, retaining one shared coupled generator objective.
///
/// Four distinct matching Float32/64 stores; finite signed CPU a[N,28,28,1] and
/// b[M,28,28,1], N,M>0, independent counts/strides allowed. Reuses the canonical
/// two-skip Pix2PixGenerate maps for G_ab/G_ba(1390 values/10 leaves each), and
/// unconditional CycleGanDiscriminate for D_a/D_b(661/6 each):4102/32 total.
/// Source seeds6203,6211,6217,6221. Translations read their source image alone.
/// Both cycles and identities always evaluate, including when a weight is zero.
/// No known transform, opposite-domain target pairing, labels or noise input.
///
/// G=(adversarial_a+adversarial_b)+(cycle_weight*(cycle_loss_a+cycle_loss_b)+
/// identity_weight*(identity_loss_a+identity_loss_b)), preserving source grouping.
/// Defaults10 and5. Adversarial G losses are full mean squared distance to1;
/// each D loss is half the sum of real-to1/fake-to0 patch means. L1 terms average
/// all pixels independently in each domain. Weights must be nonnegative finite,
/// finite after dtype rounding; zero retains diagnostics. Abs uses subgradient0
/// at equality. All applicable first/higher input/parameter paths remain live,
/// including both repeated generators in each cycle and G paths through critics.
///
/// Caller owns all captures/writes: D_a then D_b, a fresh common G forward,
/// capture BOTH G parameter vectors from that one forward BEFORE either G write,
/// then apply G_ab/G_ba. Recomputing G_ba's gradient after G_ab changes the method.
/// No fitting, optimizer, hidden stop/no-grad, persistent graph flag or batch
/// state. Only missing leaves initialize; existing values/inputs unchanged.
/// Owning fields retain graph/storage. Validate all stores/inputs/weights before
/// leaves; stored-leaf/math failures may follow earlier initialization. O(N+M)
/// work/storage for fixed maps; synchronize shared stores and exclude metadata.
/// @param a_to_b Caller-owned G_ab parameter store, source seed6203.
/// @param b_to_a Distinct matching G_ba store, source seed6211.
/// @param critic_a Distinct matching unconditional D_a store, source seed6217.
/// @param critic_b Distinct matching unconditional D_b store, source seed6221.
/// @param a Finite signed matching CPU[N,28,28,1] independent A-domain images.
/// @param b Finite signed matching CPU[M,28,28,1] independent B-domain images.
/// @param cycle_weight Nonnegative finite representable cycle-L1 weight, default10.
/// @param identity_weight Nonnegative finite representable identity-L1 weight, default5.
/// @return Owning CycleGanOutput with nineteen complete live fields and common G loss.
/// @throws std::invalid_argument For invalid/aliased stores, inputs, weights, leaves or nonfinite
/// loss.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// n::ParameterStore ab(6203),ba(6211),da(6217),db(6221);
/// auto out=n::models::CycleGan(ab,ba,da,db,n::zeros({2,28,28,1}),n::ones({2,28,28,1}));
/// n::GradOptions keep;keep.retain_graph=true;
/// auto first=n::grad(out.g_loss,ab.parameters(),keep);
/// auto second=n::grad(out.g_loss,ba.parameters(),keep);
/// // Both captures now belong to the same snapshot; caller may explicitly step both stores.
/// @endcode
/// @par References
/// - Zhu,Park,Isola and Efros (2017), Unpaired Image-to-Image Translation using
///   Cycle-Consistent Adversarial Networks, https://arxiv.org/abs/1703.10593 .
///   This small U-Net teaching variant uses default cycle10/identity5 and no image buffer.
/// @see CycleGanOutput, CycleGanDiscriminate(), Pix2PixGenerate(), SplitCycleDomains()
inline CycleGanOutput CycleGan(
    ParameterStore& a_to_b,
    ParameterStore& b_to_a,
    ParameterStore& critic_a,
    ParameterStore& critic_b,
    const Tensor&   a,
    const Tensor&   b,
    double          cycle_weight    = 10,
    double          identity_weight = 5) {
  const std::array<ParameterStore*, 4> stores{&a_to_b, &b_to_a, &critic_a, &critic_b};
  for (std::size_t i = 0; i < stores.size(); ++i) {
    if (stores[i]->dtype() != a_to_b.dtype())
      throw std::invalid_argument("CycleGAN stores must have matching dtypes");
    for (std::size_t j = 0; j < i; ++j)
      if (stores[i] == stores[j])
        throw std::invalid_argument("CycleGAN stores must be distinct");
  }
  pix_detail::Image(a, a_to_b.dtype());
  pix_detail::Image(b, a_to_b.dtype());
  for (double weight : {cycle_weight, identity_weight})
    if (!std::isfinite(weight) || weight < 0 ||
        (a_to_b.dtype() == DType::Float32 && !std::isfinite(float(weight))))
      throw std::invalid_argument(
          "CycleGAN weights must be finite and nonnegative in the computation dtype");
  CycleGanOutput o;
  o.fake_a          = Pix2PixGenerate(b_to_a, b).image;
  o.fake_b          = Pix2PixGenerate(a_to_b, a).image;
  o.cycle_a         = Pix2PixGenerate(b_to_a, o.fake_b).image;
  o.cycle_b         = Pix2PixGenerate(a_to_b, o.fake_a).image;
  o.identity_a      = Pix2PixGenerate(b_to_a, a).image;
  o.identity_b      = Pix2PixGenerate(a_to_b, b).image;
  o.real_a_score    = CycleGanDiscriminate(critic_a, a);
  o.real_b_score    = CycleGanDiscriminate(critic_b, b);
  o.fake_a_score    = CycleGanDiscriminate(critic_a, o.fake_a);
  o.fake_b_score    = CycleGanDiscriminate(critic_b, o.fake_b);
  o.adversarial_a   = mean(square(o.fake_a_score - 1.));
  o.adversarial_b   = mean(square(o.fake_b_score - 1.));
  o.cycle_loss_a    = mean(abs(o.cycle_a - a));
  o.cycle_loss_b    = mean(abs(o.cycle_b - b));
  o.identity_loss_a = mean(abs(o.identity_a - a));
  o.identity_loss_b = mean(abs(o.identity_b - b));
  o.g_loss          = (o.adversarial_a + o.adversarial_b) +
             (cycle_weight * (o.cycle_loss_a + o.cycle_loss_b) +
              identity_weight * (o.identity_loss_a + o.identity_loss_b));
  o.d_a_loss = .5 * (mean(square(o.real_a_score - 1.)) + mean(square(o.fake_a_score)));
  o.d_b_loss = .5 * (mean(square(o.real_b_score - 1.)) + mean(square(o.fake_b_score)));
  gan_detail::Finite(o.g_loss);
  gan_detail::Finite(o.d_a_loss);
  gan_detail::Finite(o.d_b_loss);
  return o;
}
} // namespace datamunge::nn::models
