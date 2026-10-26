// CMD: just run-nn action_tricks
// Action-space tricks: tanh squashing and parameterized actions.
//
// Port of tflow's action_tricks.
// Part 1. A Gaussian policy produces unbounded actions; environments want
// a box. Squashing with tanh keeps the policy differentiable but changes
// the density: log pi(a) = log N(u) - sum_i log(1 - tanh(u_i)^2). The
// correction is checked against a Monte Carlo histogram, and compared with
// the clipping alternative, which piles probability mass on the bounds.
//
// Part 2. Parameterized (hybrid) actions: a discrete choice with its own
// continuous parameters, log pi(k, u | s) = log pi_d(k|s) + log N(u | mu_k(s), sigma_k).
// A toy reaching task is solved with REINFORCE on that joint log-probability.
//
// Part 1 draws the same std::mt19937_64 samples as TF and evaluates the shared
// SquashedSample (common/rl/continuous.hpp) eagerly, so its table matches TF's
// digit for digit. Part 2's TF graph becomes eager nn ops over a
// ParameterStore; its task draws and samples use the same stream as TF but
// the weights use nn's generator, so the learning curve matches statistically.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/continuous.hpp"
#include "common/rl/pg_common.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void SquashingDemo() {
  std::printf("#### Part 1: tanh squashing\n");
  const float mu = 0.8f, sigma = 0.7f;
  const int samples = 2000000, bins = 40;
  std::mt19937_64 rng(0);
  std::normal_distribution<float> normal(mu, sigma);
  std::vector<double> hist_tanh(bins, 0), hist_clip(bins, 0);
  int clipped = 0;
  for (int i = 0; i < samples; ++i) {
    const float u = normal(rng);
    const float a = std::tanh(u);
    hist_tanh[std::min(bins - 1, static_cast<int>((a + 1) / 2 * bins))] += 1.0;
    const float c = std::max(-1.f, std::min(1.f, u));
    clipped += std::abs(u) > 1;
    hist_clip[std::min(bins - 1, static_cast<int>((c + 1) / 2 * bins))] += 1.0;
  }
  // The squashed density from the formula, via the shared SquashedSample.
  nn::NoGradGuard no_grad;
  const rl::Gaussian g{nn::full({1, 1}, mu), nn::full({1, 1}, std::log(sigma))};
  std::printf("  u ~ N(%.1f, %.1f^2), a = tanh(u). Density of a: Monte Carlo vs formula, and the same without the Jacobian term\n", mu, sigma);
  std::printf("  %8s %10s %12s %14s %14s\n", "a", "MC", "with corr.", "w/o corr.", "clipped MC");
  double worst_rel = 0;
  for (int b = 4; b < bins; b += 5) {
    const float a = -1 + (b + 0.5f) * 2.f / bins;
    const float u = std::atanh(a), e = (u - mu) / sigma;
    const Tensor et = nn::full({1, 1}, e);
    const rl::Squashed sq = rl::SquashedSample(g, et);
    const Tensor gauss_only = -0.5 * nn::square(et) - (std::log(sigma) + 0.5f * std::log(2 * 3.14159265f));
    const double mc = hist_tanh[b] / samples * bins / 2, with = std::exp(sq.logp.item<float>());
    std::printf("  %8.3f %10.4f %12.4f %14.4f %14.4f\n", a, mc, with, std::exp(gauss_only.item<float>()), hist_clip[b] / samples * bins / 2);
    if (b < bins - 1) worst_rel = std::max(worst_rel, std::abs(mc - with) / with);
    if (b == 19) check::Near(mc, 0.2845, 1e-4, "MC density at a=-0.025 (TF table: 0.2845)");
  }
  std::printf("  clipping instead: %.1f%% of samples land exactly on a bound (a point mass the Gaussian log-prob ignores)\n", 100.0 * clipped / samples);
  check::Require(worst_rel < 0.02, "corrected density matches the Monte Carlo histogram within 2% away from the bound");
  check::Near(100.0 * clipped / samples, 39.3, 0.05, "clipped fraction (TF: 39.3%)");
}

// Toy task: move from x to a target t in one step. The discrete action is
// the direction (left / right) and its continuous parameter the magnitude
// in [0, 1]; reward = -|x + dir * magnitude - t|. The oracle scores 0.
void HybridDemo() {
  std::printf("\n#### Part 2: parameterized actions on a one-step reaching task\n");
  constexpr int K = 2;
  const int batch = 256, iterations = 1500;
  const float entropy_coef = 0.01f, dirs[K] = {-1.f, 1.f};
  std::mt19937_64 rng(1);
  nn::ParameterStore params(0);
  struct Out {
    Tensor logits, log_softmax, probs, means, log_std;
  };
  auto net = [&](const Tensor& obs) {  // obs [B, 2]: position, target
    Out o;
    const Tensor h = nn::mlp(params, "trunk", obs, {64}, nn::Activation::Tanh);
    o.logits = nn::dense(params, "discrete", h, K);
    o.log_softmax = nn::log_softmax(o.logits);
    o.probs = nn::softmax(o.logits);
    o.means = nn::dense(params, "means", h, K);  // mu_k(s): magnitude for each direction
    o.log_std = params.get("log_std", {K}, nn::init::constant(-1.0));
    return o;
  };
  {
    nn::NoGradGuard no_grad;
    net(nn::zeros({1, 2}));
  }
  rl::Adam adam(params, 3e-3, 1.0);

  auto reward = [&](float x, float target, int k, float u) {
    return -std::abs(x + dirs[k] * std::max(0.f, std::min(1.f, u)) - target);
  };
  std::normal_distribution<float> normal(0.f, 1.f);
  std::uniform_real_distribution<float> unif(-1.f, 1.f);
  std::vector<float> ob(batch * 2), uv(batch * K), advs(batch);
  std::vector<int> ks(batch);
  double last_mean = 0;
  std::printf("  %10s %12s %12s\n", "iteration", "mean reward", "sigma");
  for (int it = 1; it <= iterations; ++it) {
    for (int i = 0; i < batch; ++i) ob[i * 2] = unif(rng), ob[i * 2 + 1] = unif(rng);
    const Tensor obs = nn::from_vector(ob, {batch, 2});
    std::vector<float> probs, means, log_std;
    {
      nn::NoGradGuard no_grad;
      const Out o = net(obs);
      probs = o.probs.to_vector<float>(), means = o.means.to_vector<float>(), log_std = o.log_std.to_vector<float>();
    }
    double total = 0;
    for (int i = 0; i < batch; ++i) {
      const int k = rl::SampleAction(probs.data() + i * K, K, rng);
      ks[i] = k;
      for (int j = 0; j < K; ++j) uv[i * K + j] = means[i * K + j] + std::exp(log_std[j]) * normal(rng);
      advs[i] = reward(ob[i * 2], ob[i * 2 + 1], k, uv[i * K + k]);
      total += advs[i];
    }
    const float baseline = static_cast<float>(total / batch);
    for (float& a : advs) a -= baseline;
    const float sigma = std::exp(log_std[0]);
    // Joint log-probability of (k, u_k) and REINFORCE with a small entropy
    // bonus on the discrete head, which keeps both branches alive while their
    // parameters are still being learned.
    const Out o = net(obs);
    const Tensor onehot = nn::one_hot(rl::IntVector(ks), K);
    const Tensor logp_k = nn::sum(o.log_softmax * onehot, {1});
    const Tensor zk = (nn::from_vector(uv, {batch, K}) - o.means) / nn::exp(o.log_std);
    const Tensor logp_u = nn::sum((-0.5 * nn::square(zk) - (o.log_std + 0.5 * std::log(2 * 3.14159265))) * onehot, {1});
    const Tensor logp = logp_k + logp_u;
    const Tensor disc_entropy = nn::mean(-nn::sum(o.probs * o.log_softmax, {1}), {0});
    const Tensor loss = -nn::mean(logp * rl::FloatVector(advs), {0}) - entropy_coef * disc_entropy;
    check::Require(std::isfinite(loss.item<float>()), "finite REINFORCE loss");
    adam.Minimize(loss);
    last_mean = total / batch;
    if (it % 250 == 0 || it == 1) std::printf("  %10d %12.3f %12.3f\n", it, total / batch, sigma);
  }
  double random = 0;
  for (int i = 0; i < 10000; ++i) random += reward(unif(rng), unif(rng), rng() % 2, std::abs(unif(rng)));
  std::printf("  oracle mean reward 0.000, random policy %.3f\n", random / 10000);
  std::printf("  learned policy from x = 0 for several targets (P(right), magnitude mean of the chosen direction):\n ");
  int correct = 0;
  for (float d : {-0.9f, -0.3f, 0.3f, 0.9f}) {
    nn::NoGradGuard no_grad;
    const Out o = net(nn::from_vector(std::vector<float>{0.f, d}, {1, 2}));
    const std::vector<float> p = o.probs.to_vector<float>(), m = o.means.to_vector<float>();
    const int k = p[1] > 0.5f ? 1 : 0;
    std::printf("  t=%+.1f: P(right)=%.2f, magnitude=%.2f;", d, p[1], m[k]);
    correct += (k == 1) == (d > 0) && std::abs(m[k] - std::abs(d)) < 0.2f;
  }
  std::printf("\n");
  // TF: final mean reward -0.113 (random -0.802); all four probes pick the right direction.
  check::Require(last_mean > -0.3, "hybrid REINFORCE beats random play by a wide margin (final mean reward > -0.3)");
  check::TrainingQuality(correct >= 3, "learned direction and magnitude for at least 3 of 4 probe targets");
}

}  // namespace

int main() {
  SquashingDemo();
  HybridDemo();
  return 0;
}
