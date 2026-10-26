// CMD: just run-nn vec_env
// Vectorized environments for parallel rollouts.
//
// Port of tflow's vec_env. rl::VecEnv (common/rl/vec_env.hpp, copied from
// tflow, TF-free) steps N environments with one call and resets any that
// finish, so a learner always has a full batch of observations. It can use a
// thread pool; this program measures when that is worth it. The port adds
// checks: the auto-reset bookkeeping (final_obs, done flags, episode count)
// and that a threaded VecEnv produces exactly the single-threaded rollout.
// Throughput numbers depend on the machine and its load.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>

#include "common/checks.hpp"
#include "common/rl/classic_control.hpp"
#include "common/rl/env.hpp"
#include "common/rl/vec_env.hpp"

namespace {

// Wraps an environment and burns CPU on every step, standing in for an
// expensive simulator.
class Expensive : public rl::Env {
 public:
  Expensive(std::unique_ptr<rl::Env> inner, int work) : inner_(std::move(inner)), work_(work) {}
  std::string name() const override { return "Expensive(" + inner_->name() + ")"; }
  int obs_size() const override { return inner_->obs_size(); }
  rl::ActionSpace action_space() const override { return inner_->action_space(); }
  rl::Obs Reset() override { return inner_->reset(); }
  void Seed(uint64_t seed) override { inner_->Seed(seed); }
  rl::StepResult step(const rl::Action& a) override {
    volatile float sink = 0;
    for (int i = 0; i < work_; ++i) sink += std::sin(static_cast<float>(i));
    return inner_->step(a);
  }

 private:
  std::unique_ptr<rl::Env> inner_;
  int work_;
};

double StepsPerSecond(rl::VecEnv& venv, int iterations, uint64_t seed) {
  std::mt19937_64 rng(seed);
  const rl::ActionSpace space = venv.action_space();
  std::vector<rl::Action> actions(venv.size());
  venv.reset();
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iterations; ++it) {
    for (auto& a : actions) a = space.Sample(rng);
    venv.step(actions);
  }
  auto t1 = std::chrono::steady_clock::now();
  return static_cast<double>(venv.size()) * iterations /
         std::chrono::duration<double>(t1 - t0).count();
}

// Steps two VecEnvs with the same random actions; true when every batch matches.
bool SameRollout(rl::VecEnv& a, rl::VecEnv& b, int iterations, uint64_t seed) {
  std::mt19937_64 rng(seed);
  const rl::ActionSpace space = a.action_space();
  std::vector<rl::Action> actions(a.size());
  if (a.reset() != b.reset()) return false;
  for (int it = 0; it < iterations; ++it) {
    for (auto& act : actions) act = space.Sample(rng);
    const rl::VecStep sa = a.step(actions), sb = b.step(actions);
    if (sa.obs != sb.obs || sa.reward != sb.reward || sa.terminated != sb.terminated ||
        sa.truncated != sb.truncated)
      return false;
  }
  return true;
}

}  // namespace

int main() {
  try {
    std::cout << "== Batched stepping with auto-reset\n";
    rl::VecEnv venv([] { return std::make_unique<rl::CartPole>(); }, /*n=*/8, /*seed=*/0);
    std::vector<rl::Obs> obs = venv.reset();
    std::printf("  %d x %s, batch observation %zu x %d\n", venv.size(), venv.env(0).name().c_str(),
                obs.size(), venv.obs_size());
    std::mt19937_64 rng(0);
    std::vector<rl::Action> actions(venv.size());
    int episodes_finished = 0;
    bool bookkeeping_ok = true;
    std::vector<float> returns(venv.size(), 0.f);
    for (int t = 1; t <= 100; ++t) {
      for (auto& a : actions) a = venv.action_space().Sample(rng);
      rl::VecStep s = venv.step(actions);
      bookkeeping_ok &= static_cast<int>(s.obs.size()) == venv.size();
      for (int i = 0; i < venv.size(); ++i) {
        returns[i] += s.reward[i];
        if (s.done(i)) {
          ++episodes_finished;
          // CartPole ends when |x| > 2.4 or |theta| > 0.2095 (or at 500 steps); a fresh episode
          // starts within +-0.05 of rest.
          bookkeeping_ok &= s.final_obs[i].size() == 4 &&
                            (std::abs(s.final_obs[i][0]) > 2.4f || std::abs(s.final_obs[i][2]) > 0.2094f);
          bookkeeping_ok &= std::abs(s.obs[i][0]) <= 0.05f && std::abs(s.obs[i][2]) <= 0.05f;
          if (episodes_finished <= 5)
            std::printf("  t=%-3d env %d finished with return %.0f; final x=%+.2f, new episode starts at x=%+.2f\n",
                        t, i, returns[i], s.final_obs[i][0], s.obs[i][0]);
          returns[i] = 0;
        }
      }
    }
    std::printf("  after 100 batched steps: %d episodes finished across %d environments\n",
                episodes_finished, venv.size());

    std::cout << "\n== Throughput: environment steps per second (64 envs, random actions)\n";
    std::printf("  %-28s %10s %10s %10s %10s\n", "", "1 thread", "2 threads", "4 threads",
                "8 threads");
    for (int work : {0, 2000, 20000}) {
      std::printf("  %-28s", work == 0 ? "CartPole (cheap)"
                                       : ("CartPole + " + std::to_string(work) + " sins").c_str());
      for (int threads : {1, 2, 4, 8}) {
        rl::VecEnv v(
            [work]() -> std::unique_ptr<rl::Env> {
              std::unique_ptr<rl::Env> e = std::make_unique<rl::CartPole>();
              if (work == 0) return e;
              return std::make_unique<Expensive>(std::move(e), work);
            },
            64, 0, threads);
        const int iters = work == 0 ? 2000 : (work < 10000 ? 200 : 30);
        std::printf(" %10.0f", StepsPerSecond(v, iters, 1));
        std::fflush(stdout);
      }
      std::printf("\n");
    }
    std::cout << "  Threads only help once a step costs more than the synchronisation.\n";

    auto cartpole = [] { return std::make_unique<rl::CartPole>(); };
    rl::VecEnv serial(cartpole, 64, 0, 1), threaded(cartpole, 64, 0, 4);
    const bool same = SameRollout(serial, threaded, 300, 7);
    std::printf("\n  4-thread rollout identical to the 1-thread rollout (64 envs, 300 steps): %s\n", same ? "yes" : "NO");

    check::Require(obs.size() == 8 && venv.obs_size() == 4, "the batch observation should be 8 x 4");
    check::Require(bookkeeping_ok, "finished environments should report a terminal final_obs and restart near rest");
    check::Require(episodes_finished == 34, "random CartPole from seed 0 should finish 34 episodes in 100 steps (as in tflow)");
    check::Require(same, "threading must not change the rollout");
    std::cout << "\nchecks passed\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
