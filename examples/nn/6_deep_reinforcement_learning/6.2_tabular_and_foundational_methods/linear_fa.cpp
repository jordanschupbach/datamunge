// CMD: just run-nn linear_fa
// Linear function approximation: tile coding + semi-gradient SARSA.
//
// Port of tflow's linear_fa. MountainCar has a continuous state, so Q(s, a)
// is approximated as w . phi(s, a), where phi is a sparse binary feature
// vector from tile coding (Sutton & Barto 10.1). As in TF, w is a trainable
// tensor, q is a gather of the active tiles summed, the TD target goes
// through a stop-gradient, and the semi-gradient step is reverse-mode
// gradients + plain SGD (nn::gather, nn::stop_gradient, nn::Optimizer in place
// of GatherV2, StopGradient, AddSymbolicGradients and ApplyGradientDescent);
// only the active tile indices are fed each step. The weights start at zero
// and the policy is greedy, so the run is deterministic and follows TF's
// episode for episode (up to float summation order). The vsim video of the
// greedy policy is omitted: the best of the same 3 seeded episodes is played
// and reported instead. The port adds checks (exit 1 on failure).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/play_best.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

// `tilings` overlapping grids of `size` x `size` tiles over the 2-D state
// box, each shifted by a different offset, with a separate block of tiles
// per action. phi(s, a) has exactly `tilings` ones.
class TileCoder {
 public:
  TileCoder(int tilings, int size, int actions, float lo0, float hi0, float lo1, float hi1)
      : tilings_(tilings), size_(size), actions_(actions), lo_{lo0, lo1}, hi_{hi0, hi1} {}
  int                  num_features() const { return tilings_ * size_ * size_ * actions_; }
  int                  tilings() const { return tilings_; }
  std::vector<int64_t> Active(float s0, float s1, int action) const {
    std::vector<int64_t> idx(tilings_);
    const float          u0 = (s0 - lo_[0]) / (hi_[0] - lo_[0]) * size_;
    const float          u1 = (s1 - lo_[1]) / (hi_[1] - lo_[1]) * size_;
    for (int t = 0; t < tilings_; ++t) {
      // Asymmetric offsets (1, 3) * t / tilings, as recommended for 2-D.
      const int c0 = std::min(size_ - 1, std::max(0, static_cast<int>(u0 + 1.f * t / tilings_)));
      const int c1 = std::min(size_ - 1, std::max(0, static_cast<int>(u1 + 3.f * t / tilings_)));
      idx[t] = ((action * tilings_ + t) * size_ + c0) * size_ + c1;
    }
    return idx;
  }

 private:
  int   tilings_, size_, actions_;
  float lo_[2], hi_[2];
};

Tensor IndexTensor(const std::vector<int64_t>& v) {
  return nn::from_vector(v, {static_cast<int64_t>(v.size())});
}

}  // namespace

int main() {
  try {
    rl::MountainCar env(/*max_steps=*/1000);
    const int       A = 3;
    TileCoder       coder(/*tilings=*/8, /*size=*/8, A, -1.2f, 0.6f, -0.07f, 0.07f);
    const float     alpha = 0.5f / coder.tilings(), gamma = 1.f, eps = 0.f;
    const int       episodes = 60;
    std::printf("== MountainCar with %d features (8 tilings x 8x8 tiles x %d actions), "
                "semi-gradient SARSA, alpha %.4f, gamma %.0f, epsilon %.0f (optimistic zero init)\n",
                coder.num_features(), A, alpha, gamma, eps);

    // ---- Model: q(s, a) = sum of the active weights.
    Tensor        w = nn::zeros({coder.num_features()}).requires_grad_();
    nn::Optimizer sgd({w}, alpha, {nn::OptimizerKind::Sgd});
    auto          q_of = [&](const Tensor& idx) { return nn::sum(nn::gather(w, idx, 0)); };  // [tilings] -> scalar
    auto          all_indices = [&](const rl::Obs& o) {  // [A, tilings]
      std::vector<int64_t> v;
      for (int a = 0; a < A; ++a) {
        const std::vector<int64_t> t = coder.Active(o[0], o[1], a);
        v.insert(v.end(), t.begin(), t.end());
      }
      return nn::from_vector(v, {A, static_cast<int64_t>(coder.tilings())});
    };
    auto q_all = [&](const rl::Obs& o) {  // [A]
      nn::NoGradGuard no_grad;
      return nn::sum(nn::gather(w, all_indices(o), 0), {1}).to_vector<float>();
    };
    std::mt19937_64 rng(0);
    auto            choose = [&](const rl::Obs& o) {
      if (std::uniform_real_distribution<float>(0, 1)(rng) < eps) return static_cast<int>(rng() % A);
      const std::vector<float> qa = q_all(o);
      int                      best = 0;
      for (int a = 1; a < A; ++a)
        if (qa[a] > qa[best]) best = a;
      return best;
    };

    // ---- Episodes
    std::vector<int> steps_per_episode;
    std::printf("  %-10s %s\n", "episode", "steps to the flag (1000 = time limit)");
    for (int ep = 1; ep <= episodes; ++ep) {
      rl::Obs obs = env.reset(ep);
      int     a = choose(obs);
      int     steps = 0;
      while (true) {
        rl::StepResult r = env.step(a);
        ++steps;
        const int a2 = r.terminated ? 0 : choose(r.obs);
        // Semi-gradient: the target is treated as a constant.
        const Tensor q = q_of(IndexTensor(coder.Active(obs[0], obs[1], a)));
        const Tensor q_next = q_of(IndexTensor(coder.Active(r.obs[0], r.obs[1], a2)));
        const Tensor target = nn::stop_gradient(r.reward + gamma * ((r.terminated ? 0.f : 1.f) * q_next));
        const Tensor diff = target - q;
        sgd.minimize(0.5f * diff * diff);
        obs = r.obs;
        a = a2;
        if (r.done()) break;
      }
      steps_per_episode.push_back(steps);
      if (ep <= 10 || ep % 10 == 0) std::printf("  %-10d %d\n", ep, steps);
    }

    // ---- The learned cost-to-go, -max_a q(s, a), on a coarse grid
    std::cout << "\n-- Cost-to-go -max_a q(position, velocity) after training (rows: velocity high to low)\n";
    std::printf("  %-8s", "vel\\pos");
    for (int i = 0; i < 9; ++i) std::printf(" %6.2f", -1.2f + i * 0.2f);
    std::printf("\n");
    float max_cost = 0;
    for (int j = 8; j >= 0; --j) {
      const float v = -0.07f + j * 0.0175f;
      std::printf("  %-8.3f", v);
      for (int i = 0; i < 9; ++i) {
        const float              p = -1.2f + i * 0.2f;
        const std::vector<float> qa = q_all({p, v});
        const float              cost = -std::max(qa[0], std::max(qa[1], qa[2]));
        max_cost = std::max(max_cost, cost);
        std::printf(" %6.1f", cost);
      }
      std::printf("\n");
    }
    std::cout << "\n-- The greedy policy on the hill (vsim film omitted)\n";
    auto greedy = [&](const rl::Obs& o) {  // the tile-coded q(s, .) without exploration
      return rl::GreedyQ(q_all(o), "q(s,a) from the tiles");
    };
    const rl::Playback best = rl::PlayBestOfSeeds(env, greedy, "linear_fa_mountain_car", 4242, 3);

    // Checks. A random policy almost never reaches the flag within 1000 steps;
    // TF needs 132 steps by episode 60 and plays a 155-step greedy episode.
    int late_max = 0;
    for (int e = episodes - 10; e < episodes; ++e) late_max = std::max(late_max, steps_per_episode[e]);
    check::Require(late_max < 300, "the last 10 episodes should all reach the flag in < 300 steps");
    check::Require(best.steps < 250 && best.ret > -250.f, "the greedy policy should reach the flag in < 250 steps");
    // The cost-to-go is about the number of steps left: positive and bounded.
    check::Require(max_cost > 20.f && max_cost < 200.f, "cost-to-go should look like steps-to-go (20-200)");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
