// CMD: just run-nn acer
// ACER: actor-critic with experience replay.
//
// Port of tflow's acer. Wang et al. 2017 make actor-critic sample-efficient
// by replaying stored trajectories, with three ingredients for the
// resulting off-policy data:
//   - Retrace targets for Q(s, a) with truncated importance weights,
//   - a policy gradient with truncated importance sampling plus a bias
//     correction term that sums over all actions,
//   - an efficient trust region: the gradient w.r.t. the policy logits is
//     projected so the policy stays close to a slowly moving average policy,
//     then back-propagated into the network.
// TF's AddSymbolicGradients(grad_inputs) overload becomes nn::grad with
// grad_outputs: one reverse pass from {logits, critic+entropy loss} with
// seeds {projected logit gradient, 1} yields the summed parameter gradient.
//
// The average network is a second ParameterStore updated by copy_from(tau).
// The vectorised CartPole, the action sampling and the replay sampling use
// the same std::mt19937_64 streams as TF; the weights use nn's generator, so
// results match TF statistically, not digit for digit. The vsim video is
// replaced by PlayBestOfSeeds (same episode, printed, no video).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <deque>
#include <memory>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"
#include "common/rl/vec_env.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Trajectory {
  std::vector<rl::Obs> obs;      // T + 1
  std::vector<int> act;          // T
  std::vector<float> rew, done;  // T
  std::vector<float> mu;         // T x A behaviour probabilities
};

struct Net {
  Tensor logits, logp, probs, q, v;  // [N,A] [N,A] [N,A] [N,A] [N]
};

Net Build(nn::ParameterStore& p, const Tensor& obs, int A, int hidden) {
  Net n;
  const Tensor h = nn::mlp(p, "trunk", obs, {hidden, hidden}, nn::Activation::Tanh);
  n.logits = nn::dense(p, "logits", h, A);
  n.logp = nn::log_softmax(n.logits);
  n.probs = nn::exp(n.logp);
  n.q = nn::dense(p, "q", h, A);
  n.v = nn::sum(n.probs.detach() * n.q, {1});
  return n;
}

// Retrace targets for one trajectory (T steps) from the current network's
// probabilities p, Q-values q and values v over its T + 1 observations
// (rows base .. base + T).
std::vector<float> Retrace(const Trajectory& tr, const float* p, const float* q, const float* v, int base, int T, int A,
                           float gamma) {
  float ret = v[base + T];  // bootstrap; a terminal inside the trajectory zeroes it via done
  std::vector<float> qret(T);
  for (int t = T - 1; t >= 0; --t) {
    if (tr.done[t] > 0) ret = 0;
    ret = tr.rew[t] + gamma * ret;
    qret[t] = ret;
    const int a = tr.act[t];
    const float rho_bar = std::min(1.f, p[(base + t) * A + a] / std::max(tr.mu[t * A + a], 1e-8f));
    ret = rho_bar * (ret - q[(base + t) * A + a]) + v[base + t];
  }
  return qret;
}

}  // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.n_envs = 4;
  cfg.n_steps = 20;
  cfg.lr = 5e-4f;
  rl::ParsePgConfig(argc, argv, &cfg);
  const float c_trunc = 10.f, trust_delta = 1.f, avg_tau = 0.01f;
  const int replay_ratio = 4, replay_capacity = 2000;
  rl::VecEnv venv([] { return std::make_unique<rl::CartPole>(); }, cfg.n_envs, cfg.seed);
  rl::CartPole eval_env;
  const int obs_dim = venv.obs_size(), A = venv.action_space().n, T = cfg.n_steps;
  std::printf(
      "== ACER on CartPole: %d envs x %d steps per trajectory, replay ratio %d, c %.0f, trust delta %.1f, %d steps total\n",
      cfg.n_envs, T, replay_ratio, c_trunc, trust_delta, cfg.steps);

  // Retrace on a hand-checked on-policy trajectory (rho = 1, so
  // Q_ret_t = r_t + gamma (Q_ret_{t+1} - Q(s,a) + V)): with Q = V the
  // targets are plain bootstrapped returns.
  {
    Trajectory tr;
    tr.act = {0, 1};
    tr.rew = {1, 1};
    tr.done = {0, 0};
    tr.mu = {0.5f, 0.5f, 0.5f, 0.5f};
    const float p[6] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}, q[6] = {2, 2, 2, 2, 2, 2}, v[3] = {2, 2, 3};
    const std::vector<float> qr = Retrace(tr, p, q, v, 0, 2, 2, 0.5f);
    check::Near(qr[1], 1 + 0.5 * 3, 1e-6, "Retrace target at the last step = r + gamma V(s_T)");
    check::Near(qr[0], 1 + 0.5 * (2.5 - 2 + 2), 1e-6, "Retrace target with rho = 1");
  }

  nn::ParameterStore params(cfg.seed), avg(cfg.seed + 1);
  {
    nn::NoGradGuard no_grad;
    Build(params, nn::zeros({1, obs_dim}), A, cfg.hidden);
    Build(avg, nn::zeros({1, obs_dim}), A, cfg.hidden);
  }
  avg.copy_from(params, 1.0);  // init_avg
  rl::Adam adam(params, cfg.lr, cfg.clip_norm);
  const std::vector<Tensor>& vars = adam.optimizer().parameters();

  rl::Monitor monitor(rl::MonitorConfig(cfg), "kl(avg)");
  std::mt19937_64 rng(cfg.seed + 3);
  std::deque<Trajectory> replay;
  auto explain = [&](const rl::Obs& o) {  // pi(.|s) and V(s) = sum_a pi Q: the most probable action
    nn::NoGradGuard no_grad;
    const Net n = Build(params, rl::Row(o), A, cfg.hidden);
    return rl::GreedyPi(n.probs.to_vector<float>(), n.v.item<float>());
  };
  auto greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };

  bool checked = false;
  // One update from a set of trajectories: Retrace targets on the host, then
  // the gradient step. Returns the mean KL(avg || pi) before the step.
  auto update = [&](const std::vector<const Trajectory*>& trs) {
    std::vector<rl::Obs> all_obs;
    for (auto* tr : trs) all_obs.insert(all_obs.end(), tr->obs.begin(), tr->obs.end());
    std::vector<float> P, Q, V;
    {
      nn::NoGradGuard no_grad;
      const Net n = Build(params, rl::ObsMatrix(all_obs), A, cfg.hidden);
      P = n.probs.to_vector<float>(), Q = n.q.to_vector<float>(), V = n.v.to_vector<float>();
    }
    std::vector<rl::Obs> t_obs;
    std::vector<int> t_act;
    std::vector<float> t_qret, t_mu;
    for (size_t b = 0; b < trs.size(); ++b) {
      const Trajectory& tr = *trs[b];
      const std::vector<float> qret = Retrace(tr, P.data(), Q.data(), V.data(), static_cast<int>(b) * (T + 1), T, A, cfg.gamma);
      for (int t = 0; t < T; ++t) {
        t_obs.push_back(tr.obs[t]), t_act.push_back(tr.act[t]), t_qret.push_back(qret[t]);
        t_mu.insert(t_mu.end(), tr.mu.begin() + t * A, tr.mu.begin() + (t + 1) * A);
      }
    }
    const std::int64_t N = static_cast<std::int64_t>(t_act.size());
    const Tensor obs = rl::ObsMatrix(t_obs), mu = nn::from_vector(t_mu, {N, A}), qret = rl::FloatVector(t_qret);
    const Tensor onehot = nn::one_hot(rl::IntVector(t_act), A);
    const Net net = Build(params, obs, A, cfg.hidden);
    Net avg_net;
    {
      nn::NoGradGuard no_grad;
      avg_net = Build(avg, obs, A, cfg.hidden);
    }
    const Tensor probs_c = net.probs.detach(), q_c = net.q.detach(), v_c = net.v.detach();
    const Tensor rho = probs_c / nn::maximum(mu, nn::scalar(1e-8));  // [N, A]
    const Tensor rho_a = nn::sum(rho * onehot, {1});                  // [N]
    const Tensor logp_a = nn::sum(net.logp * onehot, {1});
    const Tensor q_a = nn::sum(net.q * onehot, {1});
    // Truncated importance sampling with bias correction (eq. 7 of the paper).
    const Tensor term1 = nn::minimum(rho_a, nn::scalar(c_trunc)) * (qret - v_c) * logp_a;
    const Tensor correction = nn::maximum(1.f - c_trunc / rho, nn::scalar(0.0));  // [N, A]
    const Tensor term2 = nn::sum(correction * probs_c * ((q_c - nn::expand_dims(v_c, 1)) * net.logp), {1});
    const Tensor policy_loss = -nn::mean(term1 + term2, {0});
    const Tensor entropy = nn::mean(-nn::sum(net.probs * net.logp, {1}), {0});
    const Tensor value_loss = nn::mean(nn::square(qret - q_a), {0});
    // Efficient trust region on the logits: g <- g - max(0, (k.g - delta)/(k.k)) k,
    // with k the gradient of KL(avg || pi) w.r.t. the logits.
    const Tensor kl = nn::sum(avg_net.probs * (avg_net.logp - net.logp), {1});  // [N]
    nn::GradOptions keep;
    keep.retain_graph = true;
    const Tensor g = nn::grad(policy_loss, {net.logits}, keep)[0];
    const Tensor k = nn::grad(nn::sum(kl, {0}), {net.logits}, keep)[0];
    const Tensor kg = nn::sum(k * g, {1}, true), kk = nn::sum(k * k, {1}, true);
    const Tensor scale = nn::maximum((kg - trust_delta) / nn::maximum(kk, nn::scalar(1e-8)), nn::scalar(0.0));
    const Tensor g_adjusted = g - scale * k;
    // Back-propagate the adjusted logit gradient (it reaches the trunk and
    // the logits head) together with the critic and entropy terms.
    const Tensor other_loss = cfg.value_coef * value_loss - cfg.entropy_coef * entropy;
    if (!checked) {
      // The projection keeps k . g_adjusted <= delta in every row, and seeding
      // the logits with the unprojected g reproduces the plain gradient of
      // policy_loss + other_loss (the grad_outputs mechanism used below).
      check::Require(nn::max(nn::sum(k * g_adjusted, {1})).item<float>() <= trust_delta + 1e-4f, "trust region: k . g <= delta");
      const std::vector<Tensor> seeded = nn::grad({net.logits, other_loss}, vars, {g, nn::ones_like(other_loss)}, keep);
      const std::vector<Tensor> plain = nn::grad(policy_loss + other_loss, vars, keep);
      double diff = 0;
      for (size_t i = 0; i < vars.size(); ++i) diff = std::max(diff, nn::max(nn::abs(seeded[i] - plain[i])).item<double>());
      check::Near(diff, 0.0, 1e-5, "logit-seeded backprop = direct gradient");
      checked = true;
    }
    std::vector<Tensor> grads = nn::grad({net.logits, other_loss}, vars, {g_adjusted, nn::ones_like(other_loss)});
    if (cfg.clip_norm > 0) grads = nn::clip_by_global_norm(grads, cfg.clip_norm).first;
    adam.optimizer().step(grads);
    avg.copy_from(params, avg_tau);  // update_avg
    const float vl = value_loss.item<float>();
    check::Require(std::isfinite(vl) && std::isfinite(policy_loss.item<float>()), "finite losses");
    monitor.AddLoss(vl);
    return nn::mean(kl, {0}).item<float>();
  };

  std::vector<rl::Obs> cur = venv.reset();
  std::vector<float> ep_ret(cfg.n_envs, 0.f);
  long steps = 0, next_log = cfg.log_every;
  float last_kl = 0, best_eval = 0;
  while (steps < cfg.steps) {
    // Collect one trajectory per environment with the current policy as mu.
    std::vector<Trajectory> fresh(cfg.n_envs);
    for (int t = 0; t < T; ++t) {
      std::vector<float> P;
      {
        nn::NoGradGuard no_grad;
        P = Build(params, rl::ObsMatrix(cur), A, cfg.hidden).probs.to_vector<float>();
      }
      std::vector<rl::Action> actions(cfg.n_envs);
      for (int i = 0; i < cfg.n_envs; ++i) {
        const float* pi = P.data() + i * A;
        const int a = rl::SampleAction(pi, A, rng);
        actions[i] = rl::Action(a);
        fresh[i].obs.push_back(cur[i]);
        fresh[i].act.push_back(a);
        fresh[i].mu.insert(fresh[i].mu.end(), pi, pi + A);
      }
      rl::VecStep s = venv.step(actions);
      for (int i = 0; i < cfg.n_envs; ++i) {
        fresh[i].rew.push_back(s.reward[i]);
        fresh[i].done.push_back(s.terminated[i] ? 1.f : 0.f);
        ep_ret[i] += s.reward[i];
        if (s.done(i)) monitor.EndEpisode(ep_ret[i]), ep_ret[i] = 0;
      }
      cur = s.obs;
      steps += cfg.n_envs;
    }
    std::vector<const Trajectory*> on_policy;
    for (int i = 0; i < cfg.n_envs; ++i) {
      fresh[i].obs.push_back(cur[i]);
      on_policy.push_back(&fresh[i]);
    }
    last_kl = update(on_policy);
    check::Require(last_kl > -1e-5f, "KL(avg || pi) is nonnegative");
    for (auto& tr : fresh) {
      replay.push_back(tr);
      if (static_cast<int>(replay.size()) > replay_capacity) replay.pop_front();
    }
    // Off-policy updates from replayed trajectories.
    for (int r = 0; r < replay_ratio && static_cast<int>(replay.size()) >= cfg.n_envs; ++r) {
      std::vector<const Trajectory*> sample;
      for (int i = 0; i < cfg.n_envs; ++i) sample.push_back(&replay[rng() % replay.size()]);
      last_kl = update(sample);
    }
    if (steps >= next_log) {
      const float ev =
          next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + next_log) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(static_cast<int>(steps), last_kl, ev);
      next_log += cfg.log_every;
    }
  }
  std::printf("  (the 'loss' column is the Retrace critic loss; 'kl(avg)' the KL to the average policy)\n");
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
  rl::PlayBestOfSeeds(eval_env, explain, "acer_cartpole", 4242);
  // A random policy averages about 22. Over seeds 0..4 the best greedy evaluation
  // averages 393 for TF (final 307), 452 here (final 254).
  check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "ACER learned to balance (best greedy evaluation >= 150)");
  return 0;
}
