// CMD: just run-nn sac_vsim
// SAC on vsim tasks, with a video of the trained policy.
//
// Port of tflow's sac_vsim: the soft actor-critic from sac.cpp, unchanged,
// pointed at the vsim environments InvertedPendulum (default; TF solves it at
// a return of 1000 within the default 30k steps) or Hopper (task=hopper).
//
// PARTIAL PORT. vsim's physics engine, its scenes and the renderer that
// films the trained policy are out of scope for the datamunge ports (see
// 6.1_environments/vsim_envs.cpp), so the training, the evaluations and the
// video are skipped and no simulated number is printed. What remains
// testable without the engine is the agent at the task's sizes: the same
// networks (hidden 256), the same SAC update (soft twin-critic target, the
// reparameterised actor step and the learned temperature, batch 256), and the
// unit-box -> task action scaling. They are exercised on a synthetic replay
// buffer of random transitions with the task's observation and action sizes
// (reward +1 per step as in InvertedPendulum), and the checks cover what that
// can show: finite losses, the critic fitting the bootstrapped target, the
// temperature falling while the entropy is above target, and ScaleAction.
// Options as in TF: task=inverted_pendulum|hopper and the CConfig keys
// (updates=N sets the number of synthetic updates, default 300).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/continuous.hpp"
#include "common/rl/replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  rl::CConfig cfg;
  cfg.actor_lr = cfg.critic_lr = 3e-4f;
  cfg.batch = 256;
  cfg.hidden = 256;
  cfg.log_every = 2000;
  cfg.eval_every = 4000;
  std::string task = "inverted_pendulum";
  int updates = 300;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("task=", 0) == 0) task = arg.substr(5);
    if (arg.rfind("updates=", 0) == 0) updates = std::atoi(arg.c_str() + 8);
  }
  std::vector<char*> rest;  // updates= is this port's own option
  for (int i = 0; i < argc; ++i)
    if (std::string(argv[i]).rfind("updates=", 0) != 0) rest.push_back(argv[i]);
  rl::ParseCConfig(static_cast<int>(rest.size()), rest.data(), &cfg);
  // The tasks' interfaces (tflow's vsim_env.h): InvertedPendulum observes
  // [x, angle, x_dot, angle_dot] and pushes the cart with force/100 in [-3, 3];
  // Hopper observes height, pitch, 3 joint angles and 6 rates and drives 3 joints in [-1, 1].
  rl::ActionSpace space;
  int obs_dim;
  std::string env_name;
  if (task == "hopper") {
    space = rl::ActionSpace::Continuous({-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}), obs_dim = 11, env_name = "Hopper";
    if (cfg.steps == 20000) cfg.steps = 100000;
  } else {
    space = rl::ActionSpace::Continuous({-3.f}, {3.f}), obs_dim = 4, env_name = "InvertedPendulum";
    if (cfg.steps == 20000) cfg.steps = 30000;
  }
  const int act_dim = space.dim;
  const float target_entropy = -static_cast<float>(act_dim);
  std::printf("== SAC on vsim %s: obs %d, actions %d, %d steps, batch %d, hidden %d\n", env_name.c_str(), obs_dim, act_dim, cfg.steps, cfg.batch, cfg.hidden);
  std::printf("  SKIPPED: training, evaluation and the video need vsim's physics engine and renderer (out of scope);\n"
              "  exercising the agent with %d updates on synthetic transitions of the task's sizes instead\n", updates);

  nn::ParameterStore actor(cfg.seed), critic(cfg.seed + 1), critic_t(cfg.seed + 2), temperature(cfg.seed + 3);
  const Tensor log_alpha = temperature.get("log_alpha", {}, nn::init::zeros());
  auto policy = [&](const Tensor& s) { return rl::GaussianActor(actor, "pi", s, act_dim, cfg.hidden); };
  auto q = [&](nn::ParameterStore& p, const char* name, const Tensor& s, const Tensor& a) { return rl::QNetwork(p, name, s, a, cfg.hidden); };
  {
    nn::NoGradGuard no_grad;
    const Tensor s0 = nn::zeros({1, obs_dim}), a0 = nn::zeros({1, act_dim});
    policy(s0);
    for (nn::ParameterStore* p : {&critic, &critic_t}) q(*p, "q1", s0, a0), q(*p, "q2", s0, a0);
  }
  rl::Adam critic_adam(critic, cfg.critic_lr), actor_adam(actor, cfg.actor_lr), alpha_adam(temperature, cfg.actor_lr);
  critic_t.copy_from(critic, 1.0);
  const auto mlp_count = [&](int in, int out) { return in * cfg.hidden + cfg.hidden + cfg.hidden * cfg.hidden + cfg.hidden + (cfg.hidden + 1) * out; };
  std::printf("  actor %lld parameters, twin critics %lld\n", static_cast<long long>(actor.count()), static_cast<long long>(critic.count()));
  check::Require(actor.count() == mlp_count(obs_dim, act_dim) + (cfg.hidden + 1) * act_dim, "actor parameter count (trunk + mean + log-std heads)");
  check::Require(critic.count() == 2 * mlp_count(obs_dim + act_dim, 1), "twin critic parameter count");

  // A synthetic replay buffer: random states and unit-box actions, +1 reward,
  // 5% terminal transitions (the pole falls).
  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> normal(0.f, 1.f);
  std::uniform_real_distribution<float> unif(-1.f, 1.f);
  for (int i = 0; i < 5000; ++i) {
    rl::Obs s(obs_dim), s2(obs_dim);
    std::vector<float> u(act_dim);
    for (float& x : s) x = 0.3f * normal(rng);
    for (float& x : s2) x = 0.3f * normal(rng);
    for (float& x : u) x = unif(rng);
    buffer.Add({s, 0, 1.f, s2, unif(rng) < -0.9f ? 1.f : 0.f, u});
  }

  float first_loss = 0, last_loss = 0, alpha_value = 1, entropy = 0;
  for (int it = 1; it <= updates; ++it) {
    const rl::Batch b = buffer.Sample(cfg.batch, rng);
    const Tensor eps = rl::NormalNoise(cfg.batch, act_dim, rng), eps2 = rl::NormalNoise(cfg.batch, act_dim, rng);
    const Tensor alpha = nn::exp(log_alpha);
    Tensor y;
    {
      nn::NoGradGuard no_grad;
      const rl::Squashed next = rl::SquashedSample(policy(b.s2), eps2);
      const Tensor soft_next = nn::minimum(q(critic_t, "q1", b.s2, next.action), q(critic_t, "q2", b.s2, next.action)) - alpha.detach() * next.logp;
      y = b.r + (cfg.gamma * (1.0 - b.done)) * soft_next;
    }
    const Tensor critic_loss = nn::mean(nn::square(y - q(critic, "q1", b.s, b.act_c)), {0}) + nn::mean(nn::square(y - q(critic, "q2", b.s, b.act_c)), {0});
    critic_adam.Minimize(critic_loss);
    const rl::Squashed sample = rl::SquashedSample(policy(b.s), eps);
    const Tensor actor_loss = nn::mean(alpha.detach() * sample.logp - nn::minimum(q(critic, "q1", b.s, sample.action), q(critic, "q2", b.s, sample.action)), {0});
    const Tensor alpha_loss = -nn::mean(log_alpha * (sample.logp.detach() + target_entropy), {0});
    alpha_value = alpha.item<float>();
    entropy = -nn::mean(sample.logp.detach(), {0}).item<float>();
    actor_adam.Minimize(actor_loss);
    alpha_adam.Minimize(alpha_loss);
    critic_t.copy_from(critic, cfg.tau);
    const float cl = critic_loss.item<float>();
    check::Require(std::isfinite(cl) && std::isfinite(actor_loss.item<float>()), "finite SAC losses");
    if (it <= 20) first_loss += cl / 20;
    if (it > updates - 20) last_loss += cl / 20;
    if (it % 100 == 0 || it == 1) std::printf("  update %4d: critic loss %.4f, alpha %.4f, entropy estimate %.2f (target %.1f)\n", it, cl, alpha_value, entropy, target_entropy);
  }
  std::printf("  critic loss, first 20 vs last 20 updates: %.4f -> %.4f\n", first_loss, last_loss);
  // Unit-box actions of the mean policy map onto the task's box.
  std::vector<float> unit;
  {
    nn::NoGradGuard no_grad;
    unit = nn::tanh(policy(nn::zeros({1, obs_dim})).mean).to_vector<float>();
  }
  const rl::Action a = rl::ScaleAction(unit, space);
  bool in_box = true;
  for (int i = 0; i < act_dim; ++i)
    in_box &= std::abs(a.continuous[i] - (space.low[i] + (unit[i] + 1) * 0.5f * (space.high[i] - space.low[i]))) < 1e-5f &&
              a.continuous[i] >= space.low[i] && a.continuous[i] <= space.high[i];
  check::Require(in_box, "mean action scaled onto the task's action box");
  if (updates >= 100) {
    check::Require(last_loss < first_loss, "the twin critics fit the bootstrapped soft target (loss falls)");
    check::Require(alpha_value < 1.f, "the temperature falls while the policy entropy is above the target");
  }
  std::printf("  SKIPPED: final evaluation and the vsim video of the trained policy (no vsim engine)\n");
  return 0;
}
