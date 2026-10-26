// Port of tflow's common/nn/cycle_gan.h and common/ml/cycle_gan.h: CycleGAN
// with two U-Net generators (a_to_b, b_to_a), two unconditional local critics
// and least-squares adversarial, L1 cycle (x10) and identity (x5) terms. Groups
// 0 = a_to_b, 1 = b_to_a, 2 = critic_a, 3 = critic_b ("<k>/..."). The two
// generators share one objective, so their gradients must be captured TOGETHER
// before either write (GradientOf of several groups, then Apply).
#pragma once

#include "common/pix2pix.hpp"

namespace gan {

/// tflow CycleDiscriminate: image -> 4x4/2 conv 4 leaky .2 -> 4x4/2 conv 8 leaky .2 -> 3x3 conv 1 patch score.
inline Tensor CycleDiscriminate(nn::ParameterStore& p, const Tensor& image) {
  const Tensor down14 = nn::leaky_relu(PixConv(p, "down14", image, 1, 4, 4, 2), .2);
  const Tensor down7  = nn::leaky_relu(PixConv(p, "down7", down14, 4, 8, 4, 2), .2);
  return PixConv(p, "output", down7, 8, 1, 3, 1);
}
inline Tensor CycleL1(const Tensor& a, const Tensor& b) { return nn::mean(nn::abs(a - b)); }
inline Tensor CycleFakeMse(const Tensor& score) { return nn::mean(nn::square(score - 1.0)); }
inline Tensor CycleDiscriminatorLoss(const Tensor& real, const Tensor& fake) {
  return .5 * (CycleFakeMse(real) + nn::mean(nn::square(fake)));
}

class CycleGanModel : public GroupedGan {
 public:
  explicit CycleGanModel(bool training = true, std::vector<std::uint64_t> seeds = {6203, 6211, 6217, 6221})
      : GroupedGan(seeds) {
    {
      nn::NoGradGuard guard;
      const Tensor    zero = nn::zeros({1, 28, 28, 1});
      PixGenerate(Group(0), zero);
      PixGenerate(Group(1), zero);
      CycleDiscriminate(Group(2), zero);
      CycleDiscriminate(Group(3), zero);
    }
    if (training) MakeAdam(.0002, .5);
  }

  Arrays Forward(const Tensor& a, const Tensor& b) {
    auto&        ab = Group(0);
    auto&        ba = Group(1);
    const Tensor fake_a = PixGenerate(ba, b), fake_b = PixGenerate(ab, a);
    const Tensor cycle_a = PixGenerate(ba, fake_b), cycle_b = PixGenerate(ab, fake_a);
    const Tensor identity_a = PixGenerate(ba, a), identity_b = PixGenerate(ab, b);
    const Tensor real_a_score = CycleDiscriminate(Group(2), a), real_b_score = CycleDiscriminate(Group(3), b);
    const Tensor fake_a_score = CycleDiscriminate(Group(2), fake_a), fake_b_score = CycleDiscriminate(Group(3), fake_b);
    const Tensor adversarial_a = CycleFakeMse(fake_a_score), adversarial_b = CycleFakeMse(fake_b_score);
    const Tensor cycle_loss_a = CycleL1(cycle_a, a), cycle_loss_b = CycleL1(cycle_b, b);
    const Tensor identity_loss_a = CycleL1(identity_a, a), identity_loss_b = CycleL1(identity_b, b);
    const Tensor g_loss = (adversarial_a + adversarial_b) +
                          (10.0 * (cycle_loss_a + cycle_loss_b) + 5.0 * (identity_loss_a + identity_loss_b));
    return {{"fake_a", fake_a},
            {"fake_b", fake_b},
            {"cycle_a", cycle_a},
            {"cycle_b", cycle_b},
            {"identity_a", identity_a},
            {"identity_b", identity_b},
            {"real_a_score", real_a_score},
            {"real_b_score", real_b_score},
            {"fake_a_score", fake_a_score},
            {"fake_b_score", fake_b_score},
            {"g_loss", g_loss},
            {"d_a_loss", CycleDiscriminatorLoss(real_a_score, fake_a_score)},
            {"d_b_loss", CycleDiscriminatorLoss(real_b_score, fake_b_score)},
            {"adversarial_a", adversarial_a},
            {"adversarial_b", adversarial_b},
            {"cycle_loss_a", cycle_loss_a},
            {"cycle_loss_b", cycle_loss_b},
            {"identity_loss_a", identity_loss_a},
            {"identity_loss_b", identity_loss_b}};
  }
  Arrays Evaluate(const Tensor& a, const Tensor& b) {
    nn::NoGradGuard guard;
    Arrays          out = Forward(a, b);
    RequireFinite(out, "finite CycleGAN fields");
    return out;
  }
  /// SOURCE ONLY translation (no opposite-domain image/target).
  Tensor Translate(bool a_to_b, const Tensor& image) {
    nn::NoGradGuard guard;
    return PixGenerate(Group(a_to_b ? 0 : 1), image);
  }
  /// Gradients of every requested group's objective from ONE forward (no write in between).
  Arrays Gradient(const std::vector<int>& requested, const Tensor& a, const Tensor& b) {
    const auto fields = Forward(a, b);
    Arrays     out;
    for (int k : requested) {
      nn::GradOptions o;
      o.retain_graph   = true;
      auto&      store = Group(k);
      const auto g     = nn::grad(fields.at(k < 2 ? "g_loss" : k == 2 ? "d_a_loss" : "d_b_loss"), store.parameters(), o);
      for (std::size_t i = 0; i < g.size(); ++i) out.emplace(std::to_string(k) + "/" + store.names()[i], g[i]);
    }
    return out;
  }
};

}  // namespace gan
