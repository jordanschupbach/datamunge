// CMD: just run-nn acktr
// ACKTR: actor-critic with Kronecker-factored trust region.
//
// Port of tflow's acktr. Wu et al. 2017 replace Adam with an approximate
// natural-gradient step using K-FAC (Martens & Grosse 2015). For a dense
// layer with inputs a and pre-activation gradients g, the Fisher block is
// approximated by a Kronecker product A (x) G with A = E[a a^T] and
// G = E[g g^T], so the natural gradient is the cheap
// (A + lambda I)^-1 dW (G + lambda I)^-1. The step is then rescaled so that
// the quadratic model of the KL stays below a trust-region radius. The
// critic uses Adam.
//
// TF fetches the layer inputs and the gradients w.r.t. the pre-activations
// from the graph, does the small matrix algebra on the host with Eigen and
// writes the weights back through Assign ops. Here one nn::grad call returns
// the gradients w.r.t. the weights, biases and pre-activations (intermediate
// tensors), the factor algebra uses nn ops (matmul, trace, linalg solve in
// float32, as TF's Eigen code), and the new weights are written with
// ParameterStore::set. The vectorised CartPole and the action sampling use
// the same std::mt19937_64 streams as TF; the weights use nn's generator, so
// results match TF statistically, not digit for digit. The vsim video is
// replaced by PlayBestOfSeeds (same episode, printed, no video).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <memory>
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

struct Layer {
  int in = 0, out = 0;
  std::string w_name, b_name;
  Tensor A, G;  // running Kronecker factors [in+1, in+1], [out, out]
  bool initialised = false;
};

// Forward pass of the hand-built actor; keeps each layer's input and
// pre-activation so K-FAC can use them.
struct ActorOut {
  std::vector<Tensor> inputs, pre;
  Tensor logits, logp, probs;
};

}  // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.n_steps = 20;
  rl::ParsePgConfig(argc, argv, &cfg);
  const float damping = 0.01f, trust_delta = 0.001f, lr_max = 0.25f, ema = 0.95f;
  rl::VecEnv venv([] { return std::make_unique<rl::CartPole>(); }, cfg.n_envs, cfg.seed);
  rl::CartPole eval_env;
  const int obs_dim = venv.obs_size(), A = venv.action_space().n, H = cfg.hidden;
  std::printf("== ACKTR on CartPole: %d envs x %d steps, K-FAC damping %.3f, trust region %.4f, %d steps total\n", cfg.n_envs,
              cfg.n_steps, damping, trust_delta, cfg.steps);

  // The actor is built by hand so the layer inputs and pre-activations are
  // available; glorot-uniform weights and zero biases as in TF.
  nn::ParameterStore actor(cfg.seed), critic(cfg.seed + 1);
  std::vector<Layer> layers;
  const int sizes[4] = {obs_dim, H, H, A};
  for (int l = 0; l < 3; ++l) {
    Layer L;
    L.in = sizes[l];
    L.out = sizes[l + 1];
    L.w_name = "W" + std::to_string(l);
    L.b_name = "b" + std::to_string(l);
    actor.get(L.w_name, {L.in, L.out}, nn::init::uniform(std::sqrt(6.0 / (L.in + L.out))));
    actor.get(L.b_name, {L.out}, nn::init::zeros());
    layers.push_back(L);
  }
  auto run_actor = [&](const Tensor& obs) {
    ActorOut o;
    Tensor h = obs;
    for (int l = 0; l < 3; ++l) {
      o.inputs.push_back(h);
      o.pre.push_back(nn::bias_add(nn::matmul(h, actor.at(layers[l].w_name)), actor.at(layers[l].b_name)));
      h = l < 2 ? nn::tanh(o.pre.back()) : o.pre.back();
    }
    o.logits = h;
    o.logp = nn::log_softmax(o.logits);
    o.probs = nn::exp(o.logp);
    return o;
  };
  auto run_critic = [&](const Tensor& obs) {
    return nn::squeeze(nn::dense(critic, "v", nn::mlp(critic, "mlp", obs, {H, H}, nn::Activation::Tanh), 1), {1});
  };
  {
    nn::NoGradGuard no_grad;
    run_critic(nn::zeros({1, obs_dim}));
  }
  rl::Adam critic_adam(critic, cfg.lr, cfg.clip_norm);

  rl::Monitor monitor(rl::MonitorConfig(cfg), "step eta");
  auto policy = [&](const std::vector<rl::Obs>& o, std::vector<float>* p, std::vector<float>* v) {
    nn::NoGradGuard no_grad;
    const Tensor x = rl::ObsMatrix(o);
    *p = run_actor(x).probs.to_vector<float>();
    *v = run_critic(x).to_vector<float>();
  };
  auto explain = [&](const rl::Obs& o) {  // pi(.|s) and V(s): the most probable action, for evaluation and the replay
    nn::NoGradGuard no_grad;
    const Tensor x = rl::Row(o);
    return rl::GreedyPi(run_actor(x).probs.to_vector<float>(), run_critic(x).item<float>());
  };
  auto greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };

  rl::RolloutCollector collector(venv, A, cfg.seed + 2);
  long next_log = cfg.log_every;
  float eta = 0, best_eval = 0;
  bool checked = false;
  while (collector.total_steps() < cfg.steps) {
    rl::Rollout ro = collector.Collect(cfg.n_steps, policy, &monitor);
    std::vector<float> advantages, returns, a_i, r_i;
    for (int i = 0; i < ro.n_envs; ++i) {
      rl::Gae(ro.rew[i], ro.val[i], ro.done[i], ro.last_val[i], cfg.gamma, cfg.lam, &a_i, &r_i);
      advantages.insert(advantages.end(), a_i.begin(), a_i.end());
      returns.insert(returns.end(), r_i.begin(), r_i.end());
    }
    const int N = static_cast<int>(advantages.size());
    const Tensor obs = rl::ObsMatrix(ro.FlatObs());
    const ActorOut ao = run_actor(obs);
    const Tensor entropy = nn::mean(-nn::sum(ao.probs * ao.logp, {1}), {0});
    const Tensor logp_a = rl::LogProbOf(ao.logp, rl::IntVector(ro.FlatAct()), A);
    const Tensor actor_loss = -nn::mean(logp_a * rl::FloatVector(advantages), {0}) - cfg.entropy_coef * entropy;
    // Gradients w.r.t. parameters and w.r.t. pre-activations (for G).
    std::vector<Tensor> wrt;
    for (int l = 0; l < 3; ++l) wrt.push_back(actor.at(layers[l].w_name)), wrt.push_back(actor.at(layers[l].b_name)), wrt.push_back(ao.pre[l]);
    const std::vector<Tensor> grads = nn::grad(actor_loss, wrt);

    // K-FAC: factors, damped inverses, natural gradient, trust region.
    std::vector<Tensor> updates;
    double quad = 0;
    for (int l = 0; l < 3; ++l) {
      Layer& L = layers[l];
      const Tensor dW = grads[3 * l], db = grads[3 * l + 1];
      const Tensor g = grads[3 * l + 2] * static_cast<float>(N);  // per-sample gradients
      const Tensor a = nn::concat({ao.inputs[l].detach(), nn::ones({N, 1})}, 1);  // inputs with a 1 for the bias
      const Tensor A_now = nn::matmul(a, a, true) / static_cast<float>(N), G_now = nn::matmul(g, g, true) / static_cast<float>(N);
      if (!L.initialised) L.A = A_now, L.G = G_now, L.initialised = true;
      else L.A = ema * L.A + (1 - ema) * A_now, L.G = ema * L.G + (1 - ema) * G_now;
      // Factored Tikhonov damping (Martens & Grosse, section 6.3).
      const float trA = nn::trace(L.A).item<float>(), trG = nn::trace(L.G).item<float>();
      const float pi = std::sqrt((trA / (L.in + 1)) / std::max(1e-8f, trG / L.out));
      const Tensor A_damped = L.A + pi * std::sqrt(damping) * nn::eye(L.in + 1);
      const Tensor G_damped = L.G + std::sqrt(damping) / pi * nn::eye(L.out);
      const Tensor grad = nn::concat({dW, nn::reshape(db, {1, L.out})}, 0);  // [in+1, out]
      const Tensor nat = nn::matmul(nn::solve(A_damped, grad), nn::solve(G_damped, nn::eye(L.out)));
      if (!checked) {  // A_damped nat G_damped recovers the gradient
        const double resid = nn::max(nn::abs(nn::matmul(nn::matmul(A_damped, nat), G_damped) - grad)).item<float>();
        const double scale = nn::max(nn::abs(grad)).item<float>();
        check::Require(resid <= 1e-3 * scale + 1e-7, "K-FAC natural gradient solves (A + pi sqrt(l) I) X (G + sqrt(l)/pi I) = dW, layer " + std::to_string(l));
      }
      quad += nn::trace(nn::matmul(nn::matmul(nn::matmul(nat, L.A, true), nat), L.G)).item<float>();
      updates.push_back(nat);
    }
    checked = true;
    check::Require(std::isfinite(quad) && quad >= -1e-6, "the quadratic KL model is nonnegative");
    eta = std::min(lr_max, static_cast<float>(std::sqrt(2 * trust_delta / std::max(1e-10, quad))));
    {
      nn::NoGradGuard no_grad;
      for (int l = 0; l < 3; ++l) {
        const Layer& L = layers[l];
        actor.set(L.w_name, actor.at(L.w_name) - eta * nn::slice_axis(updates[l], 0, 0, L.in));
        actor.set(L.b_name, actor.at(L.b_name) - eta * nn::reshape(nn::slice_axis(updates[l], 0, L.in, L.in + 1), {L.out}));
      }
    }
    // Critic: three Adam epochs on the GAE returns.
    const Tensor ret_t = rl::FloatVector(returns);
    float value_loss = 0;
    for (int e = 0; e < 3; ++e) {
      const Tensor vl = nn::mean(nn::square(ret_t - run_critic(obs)), {0});
      critic_adam.Minimize(vl);
      value_loss = vl.item<float>();  // TF fetches it alongside the step, i.e. before the update
    }
    check::Require(std::isfinite(value_loss), "finite value loss");
    monitor.AddLoss(value_loss);
    if (collector.total_steps() >= next_log) {
      const float ev =
          next_log % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + next_log) : NAN;
      if (!std::isnan(ev)) best_eval = std::max(best_eval, ev);
      monitor.Log(static_cast<int>(collector.total_steps()), eta, ev);
      next_log += cfg.log_every;
    }
  }
  std::printf("  (the 'loss' column is the critic's value loss; 'step eta' the trust-region step size)\n");
  const float final_eval = rl::Evaluate(eval_env, greedy, 20, 777);
  std::printf("  final greedy evaluation over 20 episodes: %.1f\n", final_eval);
  rl::PlayBestOfSeeds(eval_env, explain, "acktr_cartpole", 4242);
  // A random policy averages about 22. Over seeds 0..4 TF's final evaluation is
  // 500 every time, this port's 484..500 (mean 497).
  check::TrainingQuality(std::max(best_eval, final_eval) >= 150, "ACKTR learned to balance (best greedy evaluation >= 150)");
  return 0;
}
