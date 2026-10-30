// CMD: just run-nn sac_vsim
/// @file sac_vsim.cpp
/// @brief Train SAC on actual native InvertedPendulum/Hopper transitions and evaluate its policy.
/// CPU NN training and physics are native datamunge code; no synthetic replay or external runtime.
/// The native task definitions differ from Jolt/Gym. Smoke runs establish execution and finite
/// losses, not convergence. Use --full for longer experiments and task=hopper for the legged task.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <string>

#include "common/checks.hpp"
#include "common/rl/continuous.hpp"
#include "common/rl/native_tasks.hpp"
#include "common/rl/replay.hpp"
namespace nn = datamunge::nn;
using nn::Tensor;
int main(int argc, char** argv) {
  rl::ValidateNativeDeviceArguments(argc, argv);
  rl::CConfig cfg;
  cfg.steps    = 30000;
  cfg.actor_lr = cfg.critic_lr = 3e-4f;
  cfg.hidden                   = 64;
  cfg.batch                    = 128;
  std::string task             = "inverted_pendulum";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]).rfind("task=", 0) == 0)
      task = std::string(argv[i]).substr(5);
  rl::ParseCConfig(argc, argv, &cfg);
  if (!example_runtime::Full())
    cfg.hidden = std::min(cfg.hidden, 32);
  if (cfg.steps < 1 || cfg.warmup < 0 || cfg.batch < 1 || cfg.buffer < cfg.batch ||
      cfg.hidden < 1 || cfg.eval_episodes < 1)
    throw std::invalid_argument("invalid native SAC configuration");
  const auto        native = rl::NativeTask(task);
  const int         limit  = example_runtime::Full() ? 1000 : 100;
  rl::NativeTaskEnv env(native, limit), eval_env(native, limit);
  const auto        space   = env.action_space();
  const int         obs_dim = env.obs_size(), act_dim = space.dim;
  const float       target_entropy = -static_cast<float>(act_dim);
  std::printf(
      "== SAC on %s: %d native transitions, batch %d, hidden %d\n",
      env.name().c_str(),
      cfg.steps,
      cfg.batch,
      cfg.hidden);
  nn::ParameterStore actor(cfg.seed), critic(cfg.seed + 1), critic_t(cfg.seed + 2),
      temperature(cfg.seed + 3);
  const Tensor log_alpha = temperature.get("log_alpha", {}, nn::init::zeros());
  auto         policy    = [&](const Tensor& s) {
    return rl::GaussianActor(actor, "pi", s, act_dim, cfg.hidden);
  };
  auto q = [&](nn::ParameterStore& p, const char* name, const Tensor& s, const Tensor& a) {
    return rl::QNetwork(p, name, s, a, cfg.hidden);
  };
  {
    nn::NoGradGuard no_grad;
    const Tensor    s0 = nn::zeros({1, obs_dim}), a0 = nn::zeros({1, act_dim});
    policy(s0);
    for (nn::ParameterStore* p : {&critic, &critic_t})
      q(*p, "q1", s0, a0), q(*p, "q2", s0, a0);
  }
  rl::Adam critic_adam(critic, cfg.critic_lr), actor_adam(actor, cfg.actor_lr),
      alpha_adam(temperature, cfg.actor_lr);
  critic_t.copy_from(critic, 1.0);
  const auto mlp_count = [&](int in, int out) {
    return in * cfg.hidden + cfg.hidden + cfg.hidden * cfg.hidden + cfg.hidden +
           (cfg.hidden + 1) * out;
  };
  std::printf(
      "  actor %lld parameters, twin critics %lld\n",
      static_cast<long long>(actor.count()),
      static_cast<long long>(critic.count()));
  check::Require(
      actor.count() == mlp_count(obs_dim, act_dim) + (cfg.hidden + 1) * act_dim,
      "actor parameter count (trunk + mean + log-std heads)");
  check::Require(
      critic.count() == 2 * mlp_count(obs_dim + act_dim, 1), "twin critic parameter count");

  const auto       initial_actor = actor.flatten();
  rl::ReplayBuffer buffer(cfg.buffer, obs_dim);
  std::mt19937_64  rng(cfg.seed);
  auto             act = [&](const rl::Obs& observation, bool stochastic) {
    nn::NoGradGuard guard;
    const auto      distribution = policy(rl::Row(observation));
    return stochastic ? rl::SquashedSample(distribution, rl::NormalNoise(1, act_dim, rng))
                            .action.to_vector<float>()
                                  : nn::tanh(distribution.mean).to_vector<float>();
  };
  const float before = rl::EvaluateC(
      eval_env, [&](const rl::Obs& o) { return act(o, false); }, cfg.eval_episodes, 777);
  auto       obs     = env.reset(cfg.seed);
  int        updates = 0, episodes = 0;
  float      last_loss = 0, alpha_value = 1, entropy = 0;
  const auto training_start = std::chrono::steady_clock::now();
  for (int step = 1; step <= cfg.steps; ++step) {
    std::vector<float> unit(act_dim);
    if (step <= cfg.warmup)
      for (auto& value : unit)
        value = std::uniform_real_distribution<float>(-1, 1)(rng);
    else
      unit = act(obs, true);
    const auto result = env.step(rl::ScaleAction(unit, space));
    // Only physical termination masks the bootstrap; a time limit retains its final state.
    buffer.Add({obs, 0, result.reward, result.obs, result.terminated ? 1.f : 0.f, unit});
    obs = result.obs;
    if (result.done()) {
      ++episodes;
      obs = env.reset();
    }
    if (step <= cfg.warmup || buffer.size() < cfg.batch)
      continue;
    const rl::Batch b   = buffer.Sample(cfg.batch, rng);
    const Tensor    eps = rl::NormalNoise(cfg.batch, act_dim, rng),
                 eps2   = rl::NormalNoise(cfg.batch, act_dim, rng);
    const Tensor alpha  = nn::exp(log_alpha);
    Tensor       y;
    {
      nn::NoGradGuard    no_grad;
      const rl::Squashed next = rl::SquashedSample(policy(b.s2), eps2);
      const Tensor       soft_next =
          nn::minimum(q(critic_t, "q1", b.s2, next.action), q(critic_t, "q2", b.s2, next.action)) -
          alpha.detach() * next.logp;
      y = b.r + (cfg.gamma * (1.0 - b.done)) * soft_next;
    }
    const Tensor critic_loss = nn::mean(nn::square(y - q(critic, "q1", b.s, b.act_c)), {0}) +
                               nn::mean(nn::square(y - q(critic, "q2", b.s, b.act_c)), {0});
    critic_adam.Minimize(critic_loss);
    const rl::Squashed sample     = rl::SquashedSample(policy(b.s), eps);
    const Tensor       actor_loss = nn::mean(
        alpha.detach() * sample.logp -
            nn::minimum(q(critic, "q1", b.s, sample.action), q(critic, "q2", b.s, sample.action)),
        {0});
    const Tensor alpha_loss = -nn::mean(log_alpha * (sample.logp.detach() + target_entropy), {0});
    alpha_value             = alpha.item<float>();
    entropy                 = -nn::mean(sample.logp.detach(), {0}).item<float>();
    actor_adam.Minimize(actor_loss);
    alpha_adam.Minimize(alpha_loss);
    critic_t.copy_from(critic, cfg.tau);
    const float cl = critic_loss.item<float>();
    check::Require(
        std::isfinite(cl) && std::isfinite(actor_loss.item<float>()), "finite SAC losses");

    ++updates;
    last_loss = cl;
  }
  const double training_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - training_start).count();
  std::printf(
      "  training loop %.4f seconds, %.1f native transitions/s (CPU physics and NN)\n",
      training_seconds,
      cfg.steps / training_seconds);
  const float after = rl::EvaluateC(
      eval_env, [&](const rl::Obs& o) { return act(o, false); }, cfg.eval_episodes, 777);
  check::Require(std::isfinite(before) && std::isfinite(after), "finite native policy evaluations");
  if (cfg.steps > std::max(cfg.warmup, cfg.batch))
    check::Require(updates > 0, "native replay performs optimizer updates");
  const auto final_actor = actor.flatten();
  double     actor_delta = 0;
  for (std::size_t i = 0; i < final_actor.size(); ++i)
    actor_delta = std::max(actor_delta, std::abs(final_actor[i] - initial_actor[i]));
  if (updates && cfg.actor_lr > 0)
    check::Require(actor_delta > 0, "native SAC updates actor parameters");
  std::printf("  maximum actor parameter change %.6g\n", actor_delta);
  std::printf(
      "  transitions %d, optimizer updates %d, completed episodes %d\n",
      cfg.steps,
      updates,
      episodes);
  std::printf(
      "  deterministic evaluation before %.2f, after %.2f; final critic loss %.5f, alpha %.4f, "
      "entropy %.3f\n",
      before,
      after,
      last_loss,
      alpha_value,
      entropy);
  return 0;
}
