// CMD: just run-nn trpo
// Trust region policy optimization (TRPO).
//
// Port of tflow's trpo. Schulman et al. 2015: maximise the surrogate
// L(theta) = E[pi_theta/pi_old A] subject to KL(pi_old || pi_theta) <= delta.
// The constrained step is
//   theta = theta_old + sqrt(2 delta / (x^T F x)) x,   F x = grad L,
// where F is the Fisher information (the Hessian of the KL at theta_old).
// F x is never formed: conjugate gradient only needs Fisher-vector
// products, computed by differentiating (grad KL . v) a second time. A
// backtracking line search then checks the real surrogate and KL before
// accepting the step.
//
// TF builds the double-backward graph with AddSymbolicGradients and the
// registered second-order gradients (common/nn/second_order_grads.h); here
// nn::grad(kl, actor, create_graph) records the backward pass, and a second
// nn::grad of (grad KL . v) gives F v, reusing the same first-order graph for
// every CG iteration. The Assign ops that load a flat parameter vector become
// ParameterStore::set. The vectorised CartPole and the action sampling use the
// same std::mt19937_64 streams as TF; the weights use nn's generator, so
// results match TF statistically, not digit for digit. The vsim video is
// replaced by PlayBestOfSeeds (same episode, printed).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"
#include "common/rl/vec_env.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

// Flat vectors over all parameter tensors.
using Flat = std::vector<std::vector<float>>;
Flat FromTensors(const std::vector<Tensor>& ts) {
  Flat f;
  for (const Tensor& t : ts) f.push_back(t.to_vector<float>());
  return f;
}
double Dot(const Flat& a, const Flat& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < a[i].size(); ++j) s += double(a[i][j]) * b[i][j];
  return s;
}
void Axpy(float alpha, const Flat& x, Flat* y) {
  for (size_t i = 0; i < x.size(); ++i)
    for (size_t j = 0; j < x[i].size(); ++j) (*y)[i][j] += alpha * x[i][j];
}

}  // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.n_steps = 128;
  cfg.lr = 1e-3f;  // critic only
  rl::ParsePgConfig(argc, argv, &cfg);
  const float max_kl = 0.01f, damping = 0.1f;
  const int cg_iters = 10, critic_epochs = 5;
  rl::VecEnv venv([] { return std::make_unique<rl::CartPole>(); }, cfg.n_envs, cfg.seed);
  rl::CartPole eval_env;
  const int obs_dim = venv.obs_size(), A = venv.action_space().n;
  std::printf("== TRPO on CartPole: batches of %d x %d steps, max KL %.3f, %d CG iterations, %d steps total\n",
              cfg.n_envs, cfg.n_steps, max_kl, cg_iters, cfg.steps);

  // Actor (tanh, so the KL is twice differentiable) and a separate critic.
  nn::ParameterStore actor(cfg.seed), critic(cfg.seed + 1);
  auto pi = [&](const Tensor& obs) {
    return rl::BuildActorCritic(actor, obs, A, cfg.hidden, /*with_value=*/false, nn::Activation::Tanh);
  };
  auto value = [&](const Tensor& obs) {
    const Tensor h = nn::mlp(critic, "mlp", obs, {cfg.hidden, cfg.hidden}, nn::Activation::Tanh);
    return nn::squeeze(nn::dense(critic, "v", h, 1), {1});
  };
  {
    nn::NoGradGuard no_grad;
    pi(nn::zeros({1, obs_dim}));
    value(nn::zeros({1, obs_dim}));
  }
  const std::vector<Tensor> theta_vars = actor.parameters();
  const size_t P = theta_vars.size();
  auto set_params = [&](const Flat& f) {
    nn::NoGradGuard no_grad;
    for (size_t i = 0; i < P; ++i) actor.set(actor.names()[i], nn::from_vector(f[i], theta_vars[i].shape()));
  };
  rl::Adam critic_adam(critic, cfg.lr, cfg.clip_norm);

  rl::Monitor monitor(rl::MonitorConfig(cfg), "kl");
  auto policy = [&](const std::vector<rl::Obs>& o, std::vector<float>* probs, std::vector<float>* values) {
    nn::NoGradGuard no_grad;
    const Tensor x = rl::ObsMatrix(o);
    *probs = pi(x).probs.to_vector<float>();
    *values = value(x).to_vector<float>();
  };
  auto explain = [&](const rl::Obs& o) {  // pi(.|s) and V(s): the most probable action, for evaluation and the replay
    nn::NoGradGuard no_grad;
    const Tensor x = rl::Row(o);
    return rl::GreedyPi(pi(x).probs.to_vector<float>(), value(x).item<float>());
  };
  auto greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };
  rl::RolloutCollector collector(venv, A, cfg.seed + 2);
  long next_log = cfg.log_every;
  int accepted = 0, iterations = 0;
  float last_kl = 0, best_eval = 0, value_loss = 0;
  bool checked = false;
  while (collector.total_steps() < cfg.steps) {
    rl::Rollout ro = collector.Collect(cfg.n_steps, policy, &monitor);
    std::vector<float> advantages, returns, logps, a_i, r_i;
    for (int i = 0; i < ro.n_envs; ++i) {
      rl::Gae(ro.rew[i], ro.val[i], ro.done[i], ro.last_val[i], cfg.gamma, cfg.lam, &a_i, &r_i);
      advantages.insert(advantages.end(), a_i.begin(), a_i.end());
      returns.insert(returns.end(), r_i.begin(), r_i.end());
      logps.insert(logps.end(), ro.logp[i].begin(), ro.logp[i].end());
    }
    const float mean = std::accumulate(advantages.begin(), advantages.end(), 0.f) / advantages.size();
    float var = 0;
    for (float a : advantages) var += (a - mean) * (a - mean);
    const float sd = std::sqrt(var / advantages.size()) + 1e-8f;
    for (float& a : advantages) a = (a - mean) / sd;
    const Tensor obs = rl::ObsMatrix(ro.FlatObs()), act = rl::IntVector(ro.FlatAct());
    const Tensor adv = rl::FloatVector(advantages), logp_old = rl::FloatVector(logps);
    Tensor probs_old, log_probs_old;
    {
      nn::NoGradGuard no_grad;
      probs_old = pi(obs).probs;
      std::vector<float> p = probs_old.to_vector<float>();
      for (float& x : p) x = std::log(std::max(x, 1e-8f));
      log_probs_old = nn::from_vector(p, probs_old.shape());
    }
    auto surrogate_of = [&](const rl::ActorCritic& ac) {
      return nn::mean(nn::exp(rl::LogProbOf(ac.logp, act, A) - logp_old) * adv, {0});
    };
    auto kl_of = [&](const rl::ActorCritic& ac) {
      return nn::mean(nn::sum(probs_old * (log_probs_old - ac.logp), {1}), {0});
    };

    // 1. Policy gradient and current parameters.
    const rl::ActorCritic ac = pi(obs);
    const Tensor surrogate = surrogate_of(ac);
    nn::GradOptions retain;  // the KL below reuses the forward pass
    retain.retain_graph = true;
    const Flat g = FromTensors(nn::grad(surrogate, theta_vars, retain));
    const Flat theta = FromTensors(theta_vars);
    const float surr_old = surrogate.item<float>();

    // 2. Conjugate gradient for F x = g, using Fisher-vector products
    //    F v = grad_theta[(grad_theta KL) . v] + damping v.
    nn::GradOptions create;
    create.create_graph = true;
    const std::vector<Tensor> kl_grad = nn::grad(kl_of(ac), theta_vars, create);
    auto fvp = [&](const Flat& v) {
      Tensor gvp = nn::zeros({});
      for (size_t i = 0; i < P; ++i) gvp = gvp + nn::sum(kl_grad[i] * nn::from_vector(v[i], theta_vars[i].shape()));
      nn::GradOptions keep;
      keep.retain_graph = true;
      Flat Fv = FromTensors(nn::grad({gvp}, theta_vars, {}, keep));
      Axpy(damping, v, &Fv);
      return Fv;
    };
    if (!checked) {
      // At theta_old the KL and its gradient vanish, and F is symmetric
      // positive semi-definite: u.Fv = v.Fu and v.Fv >= damping |v|^2.
      check::Near(kl_of(ac).item<float>(), 0.0, 1e-6, "KL(pi_old || pi_theta) = 0 at theta_old");
      double kg = 0;
      for (const Tensor& t : kl_grad) kg = std::max(kg, double(nn::max(nn::abs(t)).item<float>()));
      check::Require(kg < 1e-5, "grad KL = 0 at theta_old");
      Flat u = g, v = g;
      for (size_t i = 0; i < P; ++i)
        for (size_t j = 0; j < u[i].size(); ++j) u[i][j] = std::sin(1.0f + i + 0.37f * j), v[i][j] = std::cos(0.5f * i + 0.11f * j);
      const double uFv = Dot(u, fvp(v)), vFu = Dot(v, fvp(u)), vFv = Dot(v, fvp(v));
      std::printf("  Fisher-vector product: u.Fv %.6f, v.Fu %.6f, v.Fv %.6f >= damping |v|^2 %.6f\n", uFv, vFu, vFv,
                  damping * Dot(v, v));
      check::Near(uFv, vFu, 1e-4 * std::max(1.0, std::abs(uFv)), "the Fisher-vector product is symmetric");
      check::Require(vFv >= damping * Dot(v, v) * (1 - 1e-4), "the damped Fisher matrix is positive definite");
    }
    Flat x = g, r = g, p = g;
    for (auto& xi : x) std::fill(xi.begin(), xi.end(), 0.f);
    double rr = Dot(r, r);
    const double rr0 = rr;
    for (int it = 0; it < cg_iters; ++it) {
      Flat Fp = fvp(p);
      const double alpha = rr / std::max(1e-10, Dot(p, Fp));
      Axpy(static_cast<float>(alpha), p, &x);
      Axpy(static_cast<float>(-alpha), Fp, &r);
      const double rr_new = Dot(r, r);
      const double beta_cg = rr_new / rr;
      rr = rr_new;
      for (size_t i = 0; i < P; ++i)
        for (size_t j = 0; j < p[i].size(); ++j) p[i][j] = r[i][j] + static_cast<float>(beta_cg) * p[i][j];
      if (rr < 1e-10) break;
    }
    // 3. Scale to the trust region and line-search on the real objective.
    const Flat Fx = fvp(x);
    const double xFx = Dot(x, Fx);
    if (!checked) {
      // CG residual |F x - g| is small compared with |g|.
      Flat res = Fx;
      Axpy(-1.f, g, &res);
      std::printf("  conjugate gradient: |Fx - g| / |g| = %.2e after %d iterations\n", std::sqrt(Dot(res, res) / rr0),
                  cg_iters);
      check::Require(std::sqrt(Dot(res, res) / rr0) < 0.1, "conjugate gradient solves F x = g (relative residual < 0.1)");
      check::Require(Dot(g, x) > 0, "the natural-gradient direction ascends the surrogate");
      checked = true;
    }
    const double step_scale = std::sqrt(2 * max_kl / std::max(1e-10, xFx));
    const double expected = Dot(g, x) * step_scale;
    bool ok = false;
    ++iterations;
    for (int k = 0; k < 10 && !ok; ++k) {
      const float frac = std::pow(0.5f, k);
      Flat candidate = theta;
      Axpy(static_cast<float>(frac * step_scale), x, &candidate);
      set_params(candidate);
      nn::NoGradGuard no_grad;
      const rl::ActorCritic ac_new = pi(obs);
      const float surr_new = surrogate_of(ac_new).item<float>(), kl_new = kl_of(ac_new).item<float>();
      if (kl_new <= 1.5f * max_kl && surr_new - surr_old > 0.1f * frac * expected) {
        ok = true;
        last_kl = kl_new;
        ++accepted;
        check::Require(kl_new >= -1e-6, "accepted step has KL >= 0");
      }
    }
    if (!ok) set_params(theta);  // revert
    // 4. Critic regression.
    const Tensor ret = rl::FloatVector(returns);
    for (int e = 0; e < critic_epochs; ++e) {
      const Tensor vl = nn::mean(nn::square(ret - value(obs)), {0});
      critic_adam.Minimize(vl);
      value_loss = vl.item<float>();
    }
    check::Require(std::isfinite(value_loss), "finite value loss");
    monitor.AddLoss(value_loss);
    if (collector.total_steps() >= next_log) {
      const float ev =
          next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + next_log) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(static_cast<int>(collector.total_steps()), last_kl, ev);
      next_log += cfg.log_every;
    }
  }
  std::printf("  (the 'loss' column is the critic's value loss; the 'kl' column the KL of the accepted step)\n");
  std::printf("  %d of %d trust-region steps accepted by the line search\n", accepted, iterations);
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
  rl::PlayBestOfSeeds(eval_env, explain, "trpo_cartpole", 4242);
  // tflow has no checks. TF seed 0 accepts 98 of 98 steps; best 438.8, final 247.4.
  check::Require(accepted >= iterations / 2, "the line search accepts most trust-region steps");
  check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "TRPO learned to balance (best greedy evaluation >= 150)");
  return 0;
}
