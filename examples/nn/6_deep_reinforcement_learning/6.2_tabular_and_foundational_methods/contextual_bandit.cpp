// CMD: just run-nn contextual_bandit
// Contextual bandits with a neural policy.
//
// Port of tflow's contextual_bandit. Each round shows a context vector; the
// agent picks one of K arms and sees only that arm's (Bernoulli) reward. A
// small network maps the context to a softmax over arms and is trained with
// the one-step policy gradient
//   grad log pi(a | x) * (r - baseline),
// the simplest member of the REINFORCE family (6.4). TF builds the graph,
// AddSymbolicGradients and ApplyGradientDescent ops and runs them in a
// session; here the same loss is eager nn ops on four leaf tensors, and
// nn::Optimizer (plain SGD) applies the update. The environment
// (common/rl/bandit.hpp) and the action-sampling stream are identical to
// tflow; the initial weights come from nn's generator (seeds 1 and 2 as in
// TF, but a different normal stream), so the learning curve matches TF
// statistically, not digit for digit. The vsim video of 30 greedy rounds is
// omitted: the rounds are played and summarised instead. The port adds
// checks on the learning curve (exit 1 on failure).

#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/bandit.hpp"
#include "common/rl/pg_common.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main() {
  try {
    const int   dim = 8, K = 4, hidden = 32, batch = 32, rounds = 6000;
    const float lr = 0.1f, entropy_beta = 0.01f;
    rl::ContextualBandit env(dim, K, /*seed=*/11);
    env.Seed(0);
    std::printf("== %s: %d-dim context, %d arms, Bernoulli rewards; %d rounds, batches of %d, lr %.2f\n",
                env.name().c_str(), dim, K, rounds, batch, lr);

    // ---- Policy network: x -> tanh(x W1 + b1) -> logits = h W2 + b2
    nn::Generator g1(1), g2(2);
    Tensor        W1 = nn::normal({dim, hidden}, 0.0, std::sqrt(1.0 / dim), g1).requires_grad_();
    Tensor        b1 = nn::zeros({hidden}).requires_grad_();
    Tensor        W2 = nn::normal({hidden, K}, 0.0, std::sqrt(1.0 / hidden), g2).requires_grad_();
    Tensor        b2 = nn::zeros({K}).requires_grad_();
    auto          log_policy = [&](const Tensor& x) {  // [B, dim] -> log pi [B, K]
      const Tensor h = nn::tanh(nn::matmul(x, W1) + b1);
      return nn::log_softmax(nn::matmul(h, W2) + b2);
    };
    nn::Optimizer sgd({W1, b1, W2, b2}, lr, {nn::OptimizerKind::Sgd});

    // ---- Loss: -mean( logp[a] * (r - baseline) ) - beta * entropy
    auto loss_of = [&](const Tensor& x, const Tensor& a, const Tensor& r) {
      const Tensor logp = log_policy(x);
      const Tensor probs = nn::exp(logp);
      const Tensor chosen_logp = rl::LogProbOf(logp, a, K);              // [B]
      const Tensor advantage = r - nn::stop_gradient(nn::mean(r, {0}));  // baseline: the batch mean reward
      const Tensor entropy = -nn::sum(probs * logp, {1});                // [B]
      return -nn::mean(chosen_logp * advantage, {0}) - entropy_beta * nn::mean(entropy, {0});
    };
    auto probs_of = [&](const rl::Obs& ctx) {
      nn::NoGradGuard no_grad;
      return nn::exp(log_policy(nn::from_vector(ctx, {1, dim}))).to_vector<float>();
    };

    // ---- Interaction loop
    std::mt19937_64      rng(0);
    std::vector<rl::Obs> xb;
    std::vector<int>     ab;
    std::vector<float>   rb;
    double               window_reward = 0, window_oracle = 0, window_random = 0, window_greedy = 0;
    int                  window = 0;
    std::vector<double>  policy_curve, greedy_curve, oracle_curve, random_curve;
    std::printf("  %-12s %12s %12s %12s %12s\n", "rounds", "policy", "greedy", "oracle", "random");
    for (int t = 0; t < rounds; ++t) {
      rl::Obs                  ctx = env.reset();
      const std::vector<float> p = probs_of(ctx);
      // Sample an arm from the policy.
      const int arm = rl::SampleAction(p.data(), K, rng);
      int       greedy_arm = 0;
      for (int k = 1; k < K; ++k)
        if (p[k] > p[greedy_arm]) greedy_arm = k;
      const float reward = env.step(arm).reward;

      // Exact expected rewards make the evaluation low-variance.
      window_reward += env.ExpectedReward(arm);
      window_greedy += env.ExpectedReward(greedy_arm);
      window_oracle += env.ExpectedReward(env.BestArm());
      window_random += env.ExpectedReward(static_cast<int>(rng() % K));
      ++window;

      xb.push_back(ctx), ab.push_back(arm), rb.push_back(reward);
      if (static_cast<int>(xb.size()) == batch) {
        sgd.minimize(loss_of(rl::ObsMatrix(xb), rl::IntVector(ab), rl::FloatVector(rb)));
        xb.clear(), ab.clear(), rb.clear();
      }

      if ((t + 1) % 1000 == 0) {
        std::printf("  %5d-%-6d %12.3f %12.3f %12.3f %12.3f\n", t + 2 - window, t + 1, window_reward / window,
                    window_greedy / window, window_oracle / window, window_random / window);
        policy_curve.push_back(window_reward / window), greedy_curve.push_back(window_greedy / window);
        oracle_curve.push_back(window_oracle / window), random_curve.push_back(window_random / window);
        window_reward = window_oracle = window_random = window_greedy = 0;
        window = 0;
      }
    }
    std::cout << "  (mean expected reward of the arm chosen; 'policy' samples, 'greedy' takes the argmax)\n";

    // tflow films 30 rounds of the trained policy's greedy arm here.
    int    wins = 0, best_picks = 0;
    double greedy_expected = 0;
    for (int t = 0; t < 30; ++t) {
      rl::Obs                  ctx = env.reset();
      const std::vector<float> p = probs_of(ctx);
      int                      arm = 0;
      for (int k = 1; k < K; ++k)
        if (p[k] > p[arm]) arm = k;
      best_picks += arm == env.BestArm();
      greedy_expected += env.ExpectedReward(arm);
      wins += env.step(arm).reward > 0;
    }
    std::printf("  30 greedy rounds (vsim film omitted): %d rewards, best arm chosen %d times, mean expected %.3f\n",
                wins, best_picks, greedy_expected / 30);

    // Checks. TF: policy 0.502 -> 0.724, greedy 0.568 -> 0.785, oracle ~0.80, random ~0.46.
    check::Require(policy_curve.back() > policy_curve.front() + 0.1, "the sampled policy should improve");
    check::Require(policy_curve.back() > random_curve.back() + 0.15, "the trained policy should beat random pulls");
    check::Require(greedy_curve.back() > oracle_curve.back() - 0.06, "the greedy arm should be near the oracle");
    check::Require(greedy_curve.back() >= policy_curve.back(), "argmax should beat sampling once trained");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
