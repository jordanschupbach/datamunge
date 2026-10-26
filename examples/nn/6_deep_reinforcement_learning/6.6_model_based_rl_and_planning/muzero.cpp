// CMD: just run-nn muzero
// MuZero: planning with a learned latent dynamics model.
//
// Port of tflow's muzero.
// Schrittwieser et al. 2020 keep AlphaZero's search but replace the real
// simulator with three learned functions: a representation h(o) -> s_0,
// dynamics g(s, a) -> (r, s'), and prediction f(s) -> (policy, value). The
// model is never asked to reconstruct observations; it is trained end to end
// by unrolling K steps from replayed positions and matching the rewards,
// the n-step returns and the search policies that actually occurred. This
// version plays CartPole with 25 simulations per move; hidden states are
// min-max scaled and search values are normalised as in the paper.
//
// The TF graphs (initial / recurrent inference, K-step training unroll) become
// eager nn functions on one ParameterStore with tflow's layer names; Adam
// clips the global norm at 5 as tflow's nn::Adam. Search, replay, targets and
// every random draw use the same std::mt19937_64 stream as TF; weights use
// nn's generator, so returns match TF statistically. The vsim video
// (RecordBest) is replaced by PlayBestOfSeeds (same episodes, printed, no
// video). Options as tflow's (steps=, warmup=, batch=, seed=, log_every=,
// eval_every=, ...) plus threads=N. The port adds checks (exit 1 on failure).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/hud.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

// The paper (appendix F) trains value and reward heads in a transformed space
// h(x) = sign(x)(sqrt(|x| + 1) - 1) + 0.001 x and inverts it in the search. At
// CartPole's scale the identity learned faster in our runs (the transform
// shrinks the targets so much that the value head barely moves), so it is
// off by default; flip kUseValueTransform to try it.
constexpr bool kUseValueTransform = false;
inline float Transform(float x) {
  if (!kUseValueTransform) return x;
  return (x >= 0 ? 1.f : -1.f) * (std::sqrt(std::abs(x) + 1) - 1) + 0.001f * x;
}
inline float InverseTransform(float y) {
  if (!kUseValueTransform) return y;
  const float s = y >= 0 ? 1.f : -1.f;
  const float t = (std::sqrt(1 + 4 * 0.001f * (std::abs(y) + 1 + 0.001f)) - 1) / (2 * 0.001f);
  return s * (t * t - 1);
}

struct Step {
  rl::Obs obs;
  int action;
  float reward;
  std::vector<float> search_policy;
  float search_value;
};
struct Episode {
  std::vector<Step> steps;
  bool terminated = false;
};

// Search node in latent space.
struct Node {
  std::vector<float> state;  // hidden state (after expansion)
  float reward = 0.f, prior = 0.f, value_sum = 0.f;
  int visits = 0;
  std::vector<std::unique_ptr<Node>> children;
  bool expanded() const { return !children.empty(); }
  float Q() const { return visits ? value_sum / visits : 0.f; }
};

}  // namespace

int main(int argc, char** argv) {
  try {
  rl::DqnConfig cfg;
  cfg.steps = 20000;
  cfg.warmup = 500;
  cfg.batch = 64;
  cfg.log_every = 2500;
  cfg.eval_every = 5000;
  rl::ParseConfig(argc, argv, &cfg);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("threads=", 0) == 0) nn::set_num_threads(std::atoi(arg.c_str() + 8));
  }
  const int S = 32, K = 5, n_step = 10, simulations = 25;
  const float gamma = 0.997f, c1 = 1.25f, alpha_dir = 0.25f;
  rl::CartPole env, eval_env;
  const int obs_dim = env.obs_size(), A = env.action_space().n;
  std::printf("== MuZero on CartPole: %d steps, %d simulations per move, unroll K=%d, %d-step value targets\n", cfg.steps, simulations, K, n_step);

  nn::ParameterStore params(cfg.seed);
  auto scale = [&](const Tensor& s) {  // min-max scale each hidden state to [0, 1]
    const Tensor mn = nn::min(s, {1}, true), mx = nn::max(s, {1}, true);
    return (s - mn) / ((mx - mn) + 1e-6);
  };
  auto represent = [&](const Tensor& o) { return scale(nn::dense(params, "h/out", nn::mlp(params, "h/mlp", o, {64}), S)); };
  struct Dyn {
    Tensor state, reward;
  };
  auto dynamics = [&](const Tensor& s, const Tensor& a_onehot) {
    const Tensor x = nn::concat({s, a_onehot}, 1);
    const Tensor h = nn::mlp(params, "g/mlp", x, {64});
    return Dyn{scale(nn::dense(params, "g/state", h, S)), nn::squeeze(nn::dense(params, "g/reward", h, 1), {1})};
  };
  struct Pred {
    Tensor logits, value;
  };
  auto predict = [&](const Tensor& s) {
    const Tensor h = nn::mlp(params, "f/mlp", s, {64});
    return Pred{nn::dense(params, "f/policy", h, A), nn::squeeze(nn::dense(params, "f/value", h, 1), {1})};
  };
  auto one_hot_rows = [&](const std::vector<int>& actions) {
    std::vector<float> v(actions.size() * A, 0.f);
    for (size_t i = 0; i < actions.size(); ++i) v[i * A + actions[i]] = 1.f;
    return nn::from_vector(v, {static_cast<std::int64_t>(actions.size()), A});
  };
  {  // build every weight once so Adam sees them
    nn::NoGradGuard no_grad;
    const Tensor s0 = represent(nn::zeros({1, obs_dim}));
    predict(s0);
    dynamics(s0, one_hot_rows({0}));
  }
  rl::Adam adam(params, 1e-3, 5.0);

  std::mt19937_64 rng(cfg.seed);
  // Inference "graphs" for search: initial (s0, softmax policy, value) and recurrent.
  struct Inference {
    std::vector<float> state, probs;
    float reward = 0, value = 0;
  };
  auto initial_inference = [&](const rl::Obs& obs) {
    nn::NoGradGuard no_grad;
    const Tensor s0 = represent(rl::Row(obs));
    const Pred p = predict(s0);
    return Inference{s0.to_vector<float>(), nn::softmax(p.logits, -1).to_vector<float>(), 0.f, p.value.item<float>()};
  };
  auto recurrent_inference = [&](const std::vector<float>& state, int action) {
    nn::NoGradGuard no_grad;
    const Dyn d = dynamics(nn::from_vector(state, {1, S}), one_hot_rows({action}));
    const Pred p = predict(d.state);
    return Inference{d.state.to_vector<float>(), nn::softmax(p.logits, -1).to_vector<float>(), d.reward.item<float>(), p.value.item<float>()};
  };
  // MCTS in latent space with min-max normalised Q values.
  auto search = [&](const rl::Obs& obs, bool add_noise, std::vector<float>* policy_out) {
    Node tree;
    const Inference init = initial_inference(obs);
    tree.state = init.state;
    std::vector<float> prior = init.probs;
    if (add_noise) {
      std::gamma_distribution<float> gd(alpha_dir, 1.f);
      std::vector<float> n(A);
      float sum = 0;
      for (float& x : n) sum += (x = gd(rng));
      for (int a = 0; a < A; ++a) prior[a] = 0.75f * prior[a] + 0.25f * n[a] / sum;
    }
    for (int a = 0; a < A; ++a) {
      tree.children.push_back(std::make_unique<Node>());
      tree.children.back()->prior = prior[a];
    }
    float q_min = 1e9f, q_max = -1e9f;
    auto norm_q = [&](float q) { return q_max > q_min ? (q - q_min) / (q_max - q_min) : q; };
    for (int sim = 0; sim < simulations; ++sim) {
      Node* node = &tree;
      std::vector<Node*> path{node};
      std::vector<int> actions;
      while (node->expanded()) {
        int best = 0;
        float best_score = -1e9f;
        for (int a = 0; a < A; ++a) {
          const Node* ch = node->children[a].get();
          const float q = ch->visits ? norm_q(ch->reward + gamma * ch->Q()) : 0.f;
          const float score = q + ch->prior * std::sqrt(static_cast<float>(node->visits)) / (1 + ch->visits) * c1;
          if (score > best_score) best_score = score, best = a;
        }
        actions.push_back(best);
        node = node->children[best].get();
        path.push_back(node);
      }
      // Expand with the dynamics + prediction networks.
      Node* parent = path[path.size() - 2];
      const Inference inf = recurrent_inference(parent->state, actions.back());
      node->state = inf.state;
      node->reward = InverseTransform(inf.reward);
      for (int a = 0; a < A; ++a) {
        node->children.push_back(std::make_unique<Node>());
        node->children.back()->prior = inf.probs[a];
      }
      float value = InverseTransform(inf.value);
      for (int d = static_cast<int>(path.size()) - 1; d >= 0; --d) {
        path[d]->value_sum += value;
        path[d]->visits += 1;
        q_min = std::min(q_min, path[d]->reward + gamma * path[d]->Q());
        q_max = std::max(q_max, path[d]->reward + gamma * path[d]->Q());
        value = path[d]->reward + gamma * value;
      }
    }
    policy_out->assign(A, 0.f);
    for (int a = 0; a < A; ++a) (*policy_out)[a] = static_cast<float>(tree.children[a]->visits) / simulations;
    return tree.Q();
  };
  // ---- Training: unroll K steps from the observed position.
  auto train = [&](const std::vector<float>& O, const std::vector<std::vector<int>>& acts, const std::vector<std::vector<float>>& V,
                   const std::vector<std::vector<float>>& R, const std::vector<std::vector<float>>& P, const std::vector<std::vector<float>>& M) {
    const int Bn = cfg.batch;
    Tensor loss;
    Tensor st = represent(nn::from_vector(O, {Bn, obs_dim}));
    for (int k = 0; k <= K; ++k) {
      const Pred p = predict(st);
      const Tensor mask = nn::from_vector(M[k], {Bn});
      const Tensor v_loss = nn::square(nn::from_vector(V[k], {Bn}) - p.value);
      const Tensor p_loss = -nn::sum(nn::from_vector(P[k], {Bn, A}) * nn::log_softmax(p.logits, -1), {1});
      Tensor step_loss = mask * (0.25 * v_loss + p_loss);
      if (k < K) {
        const Dyn d = dynamics(st, one_hot_rows(acts[k]));
        step_loss = step_loss + mask * nn::square(nn::from_vector(R[k], {Bn}) - d.reward);
        // Scale the gradient flowing into the dynamics by 1/K (paper, appendix G).
        st = (1.0 / K) * d.state + (1.0 - 1.0 / K) * d.state.detach();
      }
      const Tensor l = nn::mean(step_loss, {0});
      loss = k ? loss + l : l;
    }
    const float value = loss.item<float>();
    adam.Minimize(loss);
    return value;
  };

  std::deque<Episode> replay;
  Episode current;
  rl::Monitor monitor(cfg, "root value");
  auto greedy = [&](const rl::Obs& o) {
    std::vector<float> pi;
    search(o, false, &pi);
    return static_cast<int>(std::max_element(pi.begin(), pi.end()) - pi.begin());
  };
  rl::Obs obs = env.reset(cfg.seed);
  float ret = 0, root_value = 0, first_loss = -1;
  std::vector<float> evals;
  for (int step = 1; step <= cfg.steps; ++step) {
    std::vector<float> pi;
    root_value = search(obs, true, &pi);
    // Sample from the visit distribution early in training (temperature 1), argmax later.
    const int act = step < cfg.steps / 2 ? rl::SampleAction(pi.data(), A, rng) : static_cast<int>(std::max_element(pi.begin(), pi.end()) - pi.begin());
    rl::StepResult sr = env.step(act);
    current.steps.push_back({obs, act, sr.reward, pi, root_value});
    ret += sr.reward;
    obs = sr.obs;
    if (sr.done()) {
      current.terminated = sr.terminated;
      replay.push_back(std::move(current));
      current = Episode();
      while (replay.size() > 200) replay.pop_front();
      monitor.EndEpisode(ret);
      ret = 0;
      obs = env.reset();
    }
    if (step >= cfg.warmup && step % 2 == 0 && !replay.empty()) {
      // Build a batch of K-step unrolls with n-step value targets.
      const int Bn = cfg.batch;
      std::vector<float> O(static_cast<size_t>(Bn) * obs_dim);
      std::vector<std::vector<int>> Ac(K, std::vector<int>(Bn));
      std::vector<std::vector<float>> V(K + 1, std::vector<float>(Bn)), R(K, std::vector<float>(Bn)), M(K + 1, std::vector<float>(Bn)),
          P(K + 1, std::vector<float>(static_cast<size_t>(Bn) * A));
      for (int b = 0; b < Bn; ++b) {
        const Episode& ep = replay[rng() % replay.size()];
        const int T = static_cast<int>(ep.steps.size());
        const int pos = static_cast<int>(rng() % T);
        for (int j = 0; j < obs_dim; ++j) O[b * obs_dim + j] = ep.steps[pos].obs[j];
        for (int k = 0; k <= K; ++k) {
          const int t = pos + k;
          const bool inside = t < T;
          M[k][b] = inside ? 1.f : 0.f;
          // n-step return bootstrapped from the stored search value.
          float target = 0;
          if (inside) {
            float disc = 1;
            for (int i = t; i < std::min(T, t + n_step); ++i) target += disc * ep.steps[i].reward, disc *= gamma;
            if (t + n_step < T) target += disc * ep.steps[t + n_step].search_value;
            else if (!ep.terminated) target += disc * ep.steps[T - 1].search_value;  // truncated: bootstrap
          }
          V[k][b] = Transform(target);
          for (int a = 0; a < A; ++a) P[k][b * A + a] = inside ? ep.steps[t].search_policy[a] : 1.f / A;
          if (k < K) {
            Ac[k][b] = inside ? ep.steps[t].action : static_cast<int>(rng() % A);
            R[k][b] = Transform(inside ? ep.steps[t].reward : 0.f);
          }
        }
      }
      const float l = train(O, Ac, V, R, P, M);
      check::Require(std::isfinite(l), "finite MuZero loss");
      if (first_loss < 0) first_loss = l;
      monitor.AddLoss(l);
    }
    if (step % cfg.log_every == 0) {
      const float ev = step % cfg.eval_every == 0 ? rl::Evaluate(eval_env, greedy, cfg.eval_episodes, 1000 + step) : NAN;
      if (!std::isnan(ev)) evals.push_back(ev);
      monitor.Log(step, root_value, ev);
    }
  }
  const float final_eval = rl::Evaluate(eval_env, greedy, 10, 777);
  std::printf("  final evaluation over 10 episodes (search with %d simulations, no noise): %.1f\n", simulations, final_eval);
  // On film in TF: the search's visit share per action and the root value it backed up.
  rl::PlayBestOfSeeds(eval_env, [&](const rl::Obs& o) {
    std::vector<float> pi;
    const float v = search(o, false, &pi);
    rl::Decision d = rl::GreedyPi(pi, v, "visit share");
    d.value_name = "root value";
    d.note = "MCTS, " + std::to_string(simulations) + " simulations in the learned model";
    return d;
  }, "muzero_cartpole", 4242, 2);

  // Checks added by the port (tflow prints the table only).
  // Thresholds from 4 seeds of the port (max periodic eval 347/445/368/158, final 260/388/378/143;
  // TF seed 0: 226 and 234). (The unrolled loss grows with training as the value targets grow with the episodes, so it is not checked.)
  check::Require(first_loss > 0, "MuZero trained");
  if (cfg.steps >= 20000) {
    check::TrainingQuality(*std::max_element(evals.begin(), evals.end()) >= 150, "search in the learned model reaches 150 in some periodic evaluation");
    check::TrainingQuality(final_eval >= 120, "search in the learned model balances the pole (final 10-episode evaluation >= 120)");
  }
  std::printf("\nchecks passed (vsim video omitted)\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
