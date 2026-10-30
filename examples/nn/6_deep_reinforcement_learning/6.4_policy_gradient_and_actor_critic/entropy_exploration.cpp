// CMD: just run-nn entropy_exploration
// Entropy regularization and exploration bonuses.
//
// Port of tflow's entropy_exploration. Two ways to keep a policy-gradient
// agent exploring:
//   1. An entropy bonus in the loss, -beta H(pi(.|s)), which penalises
//      premature collapse to a deterministic policy. A2C on CartPole is
//      trained with three coefficients and the final policy entropy shown.
//   2. An intrinsic reward for visiting rarely seen states, r+ = c / sqrt(N(s))
//      over a coarse discretisation of the state (count-based exploration).
//      MountainCar's reward is -1 everywhere until the flag, so the plain
//      gradient has nothing to follow; the bonus pulls the agent toward
//      states it has not seen, which is how it finds the hill.
//
// The TF graph becomes eager nn ops over a ParameterStore. The vectorised
// environments and the action sampling use the same std::mt19937_64 streams
// as TF; the weights use nn's generator, so results match TF statistically,
// not digit for digit. The vsim videos are replaced by PlayBestOfSeeds (same
// candidate episodes, printed), and TF's XLA hint has no counterpart.
// Extra options as in TF: bonus=<scale> grid=<cells per axis> mc_steps=<n>.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/pg_common.hpp"
#include "common/rl/play_best.hpp"
#include "common/rl/vec_env.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Result {
  float eval, entropy, success_rate;
};

// MountainCar with observations scaled to about [-1, 1]: the raw velocity
// (+-0.07) is far too small for an unnormalised network to use.
class NormalizedMountainCar : public rl::Env {
 public:
  std::string              name() const override { return "NormalizedMountainCar"; }
  int                      obs_size() const override { return 2; }
  rl::ActionSpace          action_space() const override { return inner_.action_space(); }
  std::vector<std::string> action_names() const override { return inner_.action_names(); }
  void                     Seed(uint64_t seed) override { inner_.Seed(seed); }
  int                      max_steps() const override { return inner_.max_steps(); }
  rl::StepResult           step(const rl::Action& a) override {
    rl::StepResult r = inner_.step(a);
    r.obs            = Scale(r.obs);
    return r;
  }
  static rl::Obs Scale(const rl::Obs& o) { return {(o[0] + 0.3f) / 0.9f, o[1] / 0.07f}; }

 protected:
  rl::Obs Reset() override { return Scale(inner_.reset()); }

 private:
  rl::MountainCar inner_;
};

using Explain = std::function<rl::Decision(const rl::Obs&)>;

// A2C with GAE; `bonus_fn(obs)` adds an intrinsic reward (0 for none).
// `film` replays the greedy policy's best episode once training is done.

std::string Tag(float x) {
  std::string t = std::to_string(x).substr(0, 4);
  std::replace(t.begin(), t.end(), '.', '_');
  return t;
}

} // namespace

int main(int argc, char** argv) {
  rl::PgConfig cfg;
  cfg.n_steps    = 20;
  cfg.steps      = 60000;
  cfg.log_every  = 20000;
  cfg.eval_every = 20000;
  rl::ParsePgConfig(argc, argv, &cfg);
  // Extra knobs for part 2: bonus=<scale> grid=<cells per axis> mc_steps=<n>
  float bonus_scale = 0.1f;
  int   grid = 40, mc_steps = 200000;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("bonus=", 0) == 0)
      bonus_scale = std::strtof(a.c_str() + 6, nullptr);
    else if (a.rfind("grid=", 0) == 0)
      grid = std::atoi(a.c_str() + 5);
    else if (a.rfind("mc_steps=", 0) == 0)
      mc_steps = std::atoi(a.c_str() + 9);
  }
  auto no_bonus = [](const rl::Obs&) { return 0.f; };

  std::printf("#### Part 1: entropy regularisation on CartPole\n");
  rl::CartPole                          cartpole_eval;
  std::vector<std::pair<float, Result>> part1;
  for (float coef : {0.f, 0.01f, 0.1f}) {
    rl::PgConfig c = cfg;
    c.entropy_coef = coef;
    auto film      = [&](const Explain& explain) {
      rl::PlayBestOfSeeds(
          cartpole_eval, explain, "entropy_exploration_coef_" + Tag(coef) + "_cartpole", 4242);
    };
    part1.push_back(
        {coef, [&]() -> Result {
           std::printf(
               "\n== %s: entropy coef %.3f, %d steps\n",
               "A2C on CartPole",
               c.entropy_coef,
               c.steps);
           rl::VecEnv venv([] { return std::make_unique<rl::CartPole>(); }, c.n_envs, c.seed);
           const int  obs_dim = venv.obs_size(), A = venv.action_space().n;
           nn::ParameterStore params(c.seed);
           auto               net = [&](const Tensor& obs) {
             return rl::BuildActorCritic(params, obs, A, c.hidden);
           };
           {
             nn::NoGradGuard no_grad;
             net(nn::zeros({1, obs_dim}));
           }
           nn::OptimizerConfig optimizer_config;
           optimizer_config.kind = nn::OptimizerKind::Adam;
           nn::Optimizer optimizer(params, c.lr, optimizer_config);

           rl::Monitor monitor(rl::MonitorConfig(c), "entropy");
           auto        policy = [&](const std::vector<rl::Obs>& o,
                             std::vector<float>*         probs,
                             std::vector<float>*         values) {
             nn::NoGradGuard       no_grad;
             const rl::ActorCritic ac = net(rl::ObsMatrix(o));
             *probs                   = ac.probs.to_vector<float>();
             *values                  = ac.value.to_vector<float>();
           };
           auto explain = [&](const rl::Obs& o) { // pi(.|s) and V(s): the most probable action, for
                                                  // evaluation and the replay
             nn::NoGradGuard       no_grad;
             const rl::ActorCritic ac = net(rl::Row(o));
             return rl::GreedyPi(ac.probs.to_vector<float>(), ac.value.item<float>());
           };
           auto greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };
           rl::RolloutCollector collector(venv, A, c.seed + 1);
           float                ent      = 0;
           long                 next_log = c.log_every;
           while (collector.total_steps() < c.steps) {
             rl::Rollout        ro = collector.Collect(c.n_steps, policy, &monitor);
             std::vector<float> advantages, returns, a_i, r_i;
             for (int i = 0; i < ro.n_envs; ++i) {
               std::vector<float> shaped = ro.rew[i];
               for (int t = 0; t < ro.n_steps; ++t)
                 shaped[t] += no_bonus(ro.obs[i][t]); // intrinsic reward
               rl::Gae(shaped, ro.val[i], ro.done[i], ro.last_val[i], c.gamma, c.lam, &a_i, &r_i);
               advantages.insert(advantages.end(), a_i.begin(), a_i.end());
               returns.insert(returns.end(), r_i.begin(), r_i.end());
             }
             const rl::ActorCritic ac      = net(rl::ObsMatrix(ro.FlatObs()));
             const Tensor          logp_a  = rl::LogProbOf(ac.logp, rl::IntVector(ro.FlatAct()), A);
             const Tensor          entropy = nn::mean(ac.entropy, {0});
             const Tensor          loss =
                 -nn::mean(logp_a * rl::FloatVector(advantages), {0}) +
                 c.value_coef * nn::mean(nn::square(rl::FloatVector(returns) - ac.value), {0}) -
                 c.entropy_coef * entropy;
             auto gradients = nn::grad(loss, optimizer.parameters());
             if (c.clip_norm > 0)
               gradients = nn::clip_by_global_norm(gradients, c.clip_norm).first;
             optimizer.step(gradients);
             monitor.AddLoss(loss.item<float>());
             ent = entropy.item<float>();
             check::Require(
                 std::isfinite(loss.item<float>()) && ent >= 0 && ent <= std::log(double(A)) + 1e-5,
                 "finite loss, entropy in [0, log A]");
             if (collector.total_steps() >= next_log) {
               monitor.Log(
                   static_cast<int>(collector.total_steps()),
                   ent,
                   next_log % c.eval_every == 0
                       ? rl::Evaluate(cartpole_eval, greedy, c.eval_episodes, 1000 + next_log)
                       : NAN);
               next_log += c.log_every;
             }
           }
           Result res{0, ent, 0};
           int    successes = 0;
           for (int e = 0; e < 20; ++e) {
             rl::Obs o       = cartpole_eval.reset(777 + e);
             float   r_total = 0;
             while (true) {
               rl::StepResult sr = cartpole_eval.step(greedy(o));
               r_total += sr.reward;
               o = sr.obs;
               if (sr.done())
                 break;
             }
             res.eval += r_total / 20;
             successes += r_total > 1e9f;
           }
           res.success_rate = successes / 20.f;
           std::printf(
               "  final greedy evaluation over 20 episodes: %.1f, policy entropy %.3f\n",
               res.eval,
               res.entropy);
           film(explain);
           return res;
         }()});
  }

  std::printf(
      "\n#### Part 2: count-based exploration bonus on MountainCar (reward -1 until the flag)\n");
  // Visit counts over a grid of (position, velocity); shared by all
  // environments and persisting across episodes.
  std::map<std::pair<int, int>, int> counts;
  std::printf(
      "  bonus %.2f / sqrt(N(cell)) over a %dx%d grid, %d steps\n",
      bonus_scale,
      grid,
      grid,
      mc_steps);
  auto count_bonus = [&](const rl::Obs& o) {
    // o is the normalised observation, both coordinates in about [-1, 1].
    const int px = std::min(grid - 1, std::max(0, static_cast<int>((o[0] + 1.f) / 2.f * grid)));
    const int vx = std::min(grid - 1, std::max(0, static_cast<int>((o[1] + 1.f) / 2.f * grid)));
    const int n  = ++counts[{px, vx}];
    return bonus_scale / std::sqrt(static_cast<float>(n));
  };
  { // the bonus decays as 1 / sqrt(N) in one cell, then the counts are cleared
    const float b1 = count_bonus({0.f, 0.f}), b2 = count_bonus({0.f, 0.f}),
                b4 = (count_bonus({0.f, 0.f}), count_bonus({0.f, 0.f}));
    check::Near(b1, bonus_scale, 1e-6, "first visit earns the full bonus");
    check::Near(b2, bonus_scale / std::sqrt(2.0), 1e-6, "second visit earns bonus / sqrt(2)");
    check::Near(b4, bonus_scale / 2, 1e-6, "fourth visit earns bonus / 2");
    counts.clear();
  }
  NormalizedMountainCar mc_eval;
  auto                  film_mc = [&](const std::string& name) {
    return
        [&, name](const Explain& explain) { rl::PlayBestOfSeeds(mc_eval, explain, name, 4242, 5); };
  };
  rl::PgConfig mc    = cfg;
  mc.steps           = mc_steps;
  mc.entropy_coef    = 0.01f;
  mc.log_every       = 50000;
  mc.eval_every      = 50000;
  const Result plain = [&]() -> Result {
    std::printf(
        "\n== %s: entropy coef %.3f, %d steps\n",
        "A2C on MountainCar, no bonus",
        mc.entropy_coef,
        mc.steps);
    rl::VecEnv venv([] { return std::make_unique<NormalizedMountainCar>(); }, mc.n_envs, mc.seed);
    const int  obs_dim = venv.obs_size(), A = venv.action_space().n;
    nn::ParameterStore params(mc.seed);
    auto net = [&](const Tensor& obs) { return rl::BuildActorCritic(params, obs, A, mc.hidden); };
    {
      nn::NoGradGuard no_grad;
      net(nn::zeros({1, obs_dim}));
    }
    nn::OptimizerConfig optimizer_config;
    optimizer_config.kind = nn::OptimizerKind::Adam;
    nn::Optimizer optimizer(params, mc.lr, optimizer_config);

    rl::Monitor monitor(rl::MonitorConfig(mc), "entropy");
    auto        policy =
        [&](const std::vector<rl::Obs>& o, std::vector<float>* probs, std::vector<float>* values) {
          nn::NoGradGuard       no_grad;
          const rl::ActorCritic ac = net(rl::ObsMatrix(o));
          *probs                   = ac.probs.to_vector<float>();
          *values                  = ac.value.to_vector<float>();
        };
    auto explain = [&](const rl::Obs& o) { // pi(.|s) and V(s): the most probable action, for
                                           // evaluation and the replay
      nn::NoGradGuard       no_grad;
      const rl::ActorCritic ac = net(rl::Row(o));
      return rl::GreedyPi(ac.probs.to_vector<float>(), ac.value.item<float>());
    };
    auto                 greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };
    rl::RolloutCollector collector(venv, A, mc.seed + 1);
    float                ent      = 0;
    long                 next_log = mc.log_every;
    while (collector.total_steps() < mc.steps) {
      rl::Rollout        ro = collector.Collect(mc.n_steps, policy, &monitor);
      std::vector<float> advantages, returns, a_i, r_i;
      for (int i = 0; i < ro.n_envs; ++i) {
        std::vector<float> shaped = ro.rew[i];
        for (int t = 0; t < ro.n_steps; ++t)
          shaped[t] += no_bonus(ro.obs[i][t]); // intrinsic reward
        rl::Gae(shaped, ro.val[i], ro.done[i], ro.last_val[i], mc.gamma, mc.lam, &a_i, &r_i);
        advantages.insert(advantages.end(), a_i.begin(), a_i.end());
        returns.insert(returns.end(), r_i.begin(), r_i.end());
      }
      const rl::ActorCritic ac      = net(rl::ObsMatrix(ro.FlatObs()));
      const Tensor          logp_a  = rl::LogProbOf(ac.logp, rl::IntVector(ro.FlatAct()), A);
      const Tensor          entropy = nn::mean(ac.entropy, {0});
      const Tensor          loss =
          -nn::mean(logp_a * rl::FloatVector(advantages), {0}) +
          mc.value_coef * nn::mean(nn::square(rl::FloatVector(returns) - ac.value), {0}) -
          mc.entropy_coef * entropy;
      auto gradients = nn::grad(loss, optimizer.parameters());
      if (mc.clip_norm > 0)
        gradients = nn::clip_by_global_norm(gradients, mc.clip_norm).first;
      optimizer.step(gradients);
      monitor.AddLoss(loss.item<float>());
      ent = entropy.item<float>();
      check::Require(
          std::isfinite(loss.item<float>()) && ent >= 0 && ent <= std::log(double(A)) + 1e-5,
          "finite loss, entropy in [0, log A]");
      if (collector.total_steps() >= next_log) {
        monitor.Log(
            static_cast<int>(collector.total_steps()),
            ent,
            next_log % mc.eval_every == 0
                ? rl::Evaluate(mc_eval, greedy, mc.eval_episodes, 1000 + next_log)
                : NAN);
        next_log += mc.log_every;
      }
    }
    Result res{0, ent, 0};
    int    successes = 0;
    for (int e = 0; e < 20; ++e) {
      rl::Obs o       = mc_eval.reset(777 + e);
      float   r_total = 0;
      while (true) {
        rl::StepResult sr = mc_eval.step(greedy(o));
        r_total += sr.reward;
        o = sr.obs;
        if (sr.done())
          break;
      }
      res.eval += r_total / 20;
      successes += r_total > -200.f;
    }
    res.success_rate = successes / 20.f;
    std::printf(
        "  final greedy evaluation over 20 episodes: %.1f, policy entropy %.3f\n",
        res.eval,
        res.entropy);
    film_mc("entropy_exploration_no_bonus_mountain_car")(explain);
    return res;
  }();
  const Result bonus = [&]() -> Result {
    std::printf(
        "\n== %s: entropy coef %.3f, %d steps\n",
        "A2C on MountainCar, count-based bonus",
        mc.entropy_coef,
        mc.steps);
    rl::VecEnv venv([] { return std::make_unique<NormalizedMountainCar>(); }, mc.n_envs, mc.seed);
    const int  obs_dim = venv.obs_size(), A = venv.action_space().n;
    nn::ParameterStore params(mc.seed);
    auto net = [&](const Tensor& obs) { return rl::BuildActorCritic(params, obs, A, mc.hidden); };
    {
      nn::NoGradGuard no_grad;
      net(nn::zeros({1, obs_dim}));
    }
    nn::OptimizerConfig optimizer_config;
    optimizer_config.kind = nn::OptimizerKind::Adam;
    nn::Optimizer optimizer(params, mc.lr, optimizer_config);

    rl::Monitor monitor(rl::MonitorConfig(mc), "entropy");
    auto        policy =
        [&](const std::vector<rl::Obs>& o, std::vector<float>* probs, std::vector<float>* values) {
          nn::NoGradGuard       no_grad;
          const rl::ActorCritic ac = net(rl::ObsMatrix(o));
          *probs                   = ac.probs.to_vector<float>();
          *values                  = ac.value.to_vector<float>();
        };
    auto explain = [&](const rl::Obs& o) { // pi(.|s) and V(s): the most probable action, for
                                           // evaluation and the replay
      nn::NoGradGuard       no_grad;
      const rl::ActorCritic ac = net(rl::Row(o));
      return rl::GreedyPi(ac.probs.to_vector<float>(), ac.value.item<float>());
    };
    auto                 greedy = [&](const rl::Obs& o) { return explain(o).action.discrete; };
    rl::RolloutCollector collector(venv, A, mc.seed + 1);
    float                ent      = 0;
    long                 next_log = mc.log_every;
    while (collector.total_steps() < mc.steps) {
      rl::Rollout        ro = collector.Collect(mc.n_steps, policy, &monitor);
      std::vector<float> advantages, returns, a_i, r_i;
      for (int i = 0; i < ro.n_envs; ++i) {
        std::vector<float> shaped = ro.rew[i];
        for (int t = 0; t < ro.n_steps; ++t)
          shaped[t] += count_bonus(ro.obs[i][t]); // intrinsic reward
        rl::Gae(shaped, ro.val[i], ro.done[i], ro.last_val[i], mc.gamma, mc.lam, &a_i, &r_i);
        advantages.insert(advantages.end(), a_i.begin(), a_i.end());
        returns.insert(returns.end(), r_i.begin(), r_i.end());
      }
      const rl::ActorCritic ac      = net(rl::ObsMatrix(ro.FlatObs()));
      const Tensor          logp_a  = rl::LogProbOf(ac.logp, rl::IntVector(ro.FlatAct()), A);
      const Tensor          entropy = nn::mean(ac.entropy, {0});
      const Tensor          loss =
          -nn::mean(logp_a * rl::FloatVector(advantages), {0}) +
          mc.value_coef * nn::mean(nn::square(rl::FloatVector(returns) - ac.value), {0}) -
          mc.entropy_coef * entropy;
      auto gradients = nn::grad(loss, optimizer.parameters());
      if (mc.clip_norm > 0)
        gradients = nn::clip_by_global_norm(gradients, mc.clip_norm).first;
      optimizer.step(gradients);
      monitor.AddLoss(loss.item<float>());
      ent = entropy.item<float>();
      check::Require(
          std::isfinite(loss.item<float>()) && ent >= 0 && ent <= std::log(double(A)) + 1e-5,
          "finite loss, entropy in [0, log A]");
      if (collector.total_steps() >= next_log) {
        monitor.Log(
            static_cast<int>(collector.total_steps()),
            ent,
            next_log % mc.eval_every == 0
                ? rl::Evaluate(mc_eval, greedy, mc.eval_episodes, 1000 + next_log)
                : NAN);
        next_log += mc.log_every;
      }
    }
    Result res{0, ent, 0};
    int    successes = 0;
    for (int e = 0; e < 20; ++e) {
      rl::Obs o       = mc_eval.reset(777 + e);
      float   r_total = 0;
      while (true) {
        rl::StepResult sr = mc_eval.step(greedy(o));
        r_total += sr.reward;
        o = sr.obs;
        if (sr.done())
          break;
      }
      res.eval += r_total / 20;
      successes += r_total > -200.f;
    }
    res.success_rate = successes / 20.f;
    std::printf(
        "  final greedy evaluation over 20 episodes: %.1f, policy entropy %.3f\n",
        res.eval,
        res.entropy);
    film_mc("entropy_exploration_count_bonus_mountain_car")(explain);
    return res;
  }();

  std::printf("\n#### Summary\n  CartPole entropy coefficient -> final return / policy entropy:");
  for (auto& [coef, r] : part1)
    std::printf("  %.2f -> %.0f / %.3f", coef, r.eval, r.entropy);
  std::printf(
      "\n  MountainCar flag reached (greedy, 20 episodes): no bonus %.0f%%, count-based bonus "
      "%.0f%% (visited %zu grid cells)\n",
      100 * plain.success_rate,
      100 * bonus.success_rate,
      counts.size());
  // tflow has no checks. Whether the bonus beats plain A2C is seed-dependent
  // in TF too (flag reached, no bonus / bonus, seeds 0..4: 0/100, 90/70, 0/55,
  // 0/0, 0/0 %), so it is reported, not checked; CartPole must be learned.
  float best_cartpole = 0;
  for (auto& [coef, r] : part1)
    best_cartpole = std::max(best_cartpole, r.eval);
  check::TrainingQuality(
      best_cartpole >= 150,
      "A2C learned CartPole for some entropy coefficient (final greedy evaluation >= 150)");
  return 0;
}
