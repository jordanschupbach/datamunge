// CMD: just run-nn bandit_algorithms
// Bandit algorithms: epsilon-greedy, UCB, Thompson sampling.
//
// Port of tflow's bandit_algorithms (TF-free in tflow too, so the numbers
// match it exactly). The exploration/exploitation trade-off in its purest
// form. Each algorithm is run on many random bandit instances and judged by
// average reward, how often it pulls the best arm, and cumulative regret.
// The environments come from common/rl/bandit.hpp (copied from tflow). The
// vsim video of 40 Thompson pulls is omitted; the pulls are still made and
// summarised. The port adds checks on the regret ordering the program
// teaches (exit 1 on failure).

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/bandit.hpp"

namespace {

struct Algorithm {
  std::string name;
  // Returns the arm to pull given the counts and estimates so far.
  virtual int  Choose(const std::vector<int>& n, const std::vector<float>& q, int t, std::mt19937_64& rng) = 0;
  virtual void Observe(int, float) {}
  virtual void Reset() {}
  virtual ~Algorithm() = default;
};

struct EpsilonGreedy : Algorithm {
  float eps;
  EpsilonGreedy(float e, std::string label) : eps(e) { name = label; }
  int Choose(const std::vector<int>&, const std::vector<float>& q, int, std::mt19937_64& rng) override {
    if (std::uniform_real_distribution<float>(0, 1)(rng) < eps) return static_cast<int>(rng() % q.size());
    int best = 0;
    for (size_t a = 1; a < q.size(); ++a)
      if (q[a] > q[best]) best = static_cast<int>(a);
    return best;
  }
};

struct Ucb : Algorithm {
  float c;
  explicit Ucb(float c_) : c(c_) { name = "UCB1 c=" + std::to_string(c).substr(0, 3); }
  int Choose(const std::vector<int>& n, const std::vector<float>& q, int t, std::mt19937_64&) override {
    int   best = 0;
    float best_score = -1e30f;
    for (size_t a = 0; a < q.size(); ++a) {
      const float score = n[a] == 0 ? 1e30f : q[a] + c * std::sqrt(std::log(static_cast<float>(t + 1)) / n[a]);
      if (score > best_score) best_score = score, best = static_cast<int>(a);
    }
    return best;
  }
};

// Thompson sampling for Gaussian rewards with known unit variance and a
// N(0, 1) prior per arm: the posterior mean is sum / (n + 1) and the
// posterior variance 1 / (n + 1). Pull the arm with the best posterior draw.
struct ThompsonGaussian : Algorithm {
  std::vector<float> sum;
  ThompsonGaussian() { name = "Thompson (Gaussian)"; }
  void Reset() override { sum.clear(); }
  int  Choose(const std::vector<int>& n, const std::vector<float>&, int, std::mt19937_64& rng) override {
    if (sum.empty()) sum.assign(n.size(), 0.f);
    int   best = 0;
    float best_draw = -1e30f;
    for (size_t a = 0; a < n.size(); ++a) {
      const float mean = sum[a] / (n[a] + 1), sd = 1.f / std::sqrt(n[a] + 1.f);
      const float draw = std::normal_distribution<float>(mean, sd)(rng);
      if (draw > best_draw) best_draw = draw, best = static_cast<int>(a);
    }
    return best;
  }
  void Observe(int a, float r) override { sum[a] += r; }
};

// Thompson sampling for Bernoulli rewards with Beta(1, 1) priors.
struct ThompsonBeta : Algorithm {
  std::vector<float> wins, losses;
  ThompsonBeta() { name = "Thompson (Beta)"; }
  void Reset() override { wins.clear(), losses.clear(); }
  int  Choose(const std::vector<int>& n, const std::vector<float>&, int, std::mt19937_64& rng) override {
    if (wins.empty()) wins.assign(n.size(), 0.f), losses.assign(n.size(), 0.f);
    int   best = 0;
    float best_draw = -1.f;
    for (size_t a = 0; a < n.size(); ++a) {
      std::gamma_distribution<float> ga(wins[a] + 1, 1.f), gb(losses[a] + 1, 1.f);
      const float                    x = ga(rng), y = gb(rng);
      const float                    draw = x / (x + y);  // Beta(wins+1, losses+1) sample
      if (draw > best_draw) best_draw = draw, best = static_cast<int>(a);
    }
    return best;
  }
  void Observe(int a, float r) override { (r > 0 ? wins : losses)[a] += 1; }
};

struct Summary {
  std::vector<double> reward, optimal, regret;  // per step, averaged over runs
};

Summary Evaluate(Algorithm& alg, bool bernoulli, int runs, int steps, float q0) {
  Summary s{std::vector<double>(steps), std::vector<double>(steps), std::vector<double>(steps)};
  for (int run = 0; run < runs; ++run) {
    std::mt19937_64       rng(run);
    rl::MultiArmedBandit bandit = bernoulli ? rl::MultiArmedBandit::Bernoulli([&] {
      std::vector<float> p(10);
      for (float& v : p) v = std::uniform_real_distribution<float>(0, 1)(rng);
      return p;
    }())
                                            : rl::MultiArmedBandit::Testbed(10, run);
    bandit.Seed(run + 1);
    const int          K = bandit.num_arms();
    std::vector<int>   n(K, 0);
    std::vector<float> q(K, q0);  // sample-average estimates, optionally optimistic
    alg.Reset();
    double regret = 0;
    for (int t = 0; t < steps; ++t) {
      const int a = alg.Choose(n, q, t, rng);
      bandit.reset();
      const float r = bandit.step(a).reward;
      n[a]++;
      q[a] += (r - q[a]) / n[a];
      alg.Observe(a, r);
      regret += bandit.best_mean() - bandit.mean(a);
      s.reward[t] += r;
      s.optimal[t] += a == bandit.best_arm();
      s.regret[t] += regret;
    }
  }
  for (int t = 0; t < steps; ++t) s.reward[t] /= runs, s.optimal[t] /= runs, s.regret[t] /= runs;
  return s;
}

// Prints the table and returns the summaries (for the checks).
std::vector<Summary> Table(const std::vector<std::pair<Algorithm*, float>>& algs, bool bernoulli, int runs, int steps) {
  std::vector<Summary> all;
  std::printf("  %-22s %28s   %28s   %10s\n", "", "average reward at step", "% optimal action at step", "regret");
  std::printf("  %-22s", "algorithm");
  for (int t : {10, 100, 500, 1000}) std::printf(" %6d", t);
  std::printf("   ");
  for (int t : {10, 100, 500, 1000}) std::printf(" %6d", t);
  std::printf("   %10d\n", steps);
  for (auto& [alg, q0] : algs) {
    Summary s = Evaluate(*alg, bernoulli, runs, steps, q0);
    std::printf("  %-22s", (alg->name + (q0 != 0 ? " Q0=5" : "")).c_str());
    for (int t : {10, 100, 500, 1000}) std::printf(" %6.3f", s.reward[t - 1]);
    std::printf("   ");
    for (int t : {10, 100, 500, 1000}) std::printf(" %5.0f%%", 100 * s.optimal[t - 1]);
    std::printf("   %10.1f\n", s.regret[steps - 1]);
    all.push_back(std::move(s));
  }
  return all;
}

}  // namespace

int main() {
  try {
    const int runs = 500, steps = 1000;
    std::printf("== 10-armed Gaussian testbed (means ~ N(0,1), unit noise), %d runs x %d steps\n", runs, steps);
    EpsilonGreedy greedy(0.f, "greedy"), eps01(0.01f, "eps-greedy 0.01"), eps1(0.1f, "eps-greedy 0.1"),
        optimistic(0.f, "optimistic greedy");
    Ucb              ucb(2.f);
    ThompsonGaussian thompson;
    const std::vector<Summary> g =
        Table({{&greedy, 0}, {&eps01, 0}, {&eps1, 0}, {&optimistic, 5.f}, {&ucb, 0}, {&thompson, 0}}, false, runs, steps);

    std::printf("\n== 10 Bernoulli arms (p ~ U(0,1)), %d runs x %d steps\n", runs, steps);
    ThompsonBeta               beta;
    Ucb                        ucb_b(1.f);
    const std::vector<Summary> b = Table({{&eps1, 0}, {&ucb_b, 0}, {&beta, 0}}, true, runs, steps);
    std::cout << "\n  Regret grows linearly for fixed-epsilon policies and logarithmically for UCB and Thompson.\n";

    // tflow films the first 40 pulls of one Thompson sampling run on a fresh
    // testbed here; the film is omitted, the pulls are made and summarised.
    std::cout << "\n-- 40 pulls of Thompson sampling on a 10-armed testbed (vsim film omitted)\n";
    int best_pulls = 0;
    {
      rl::MultiArmedBandit bandit = rl::MultiArmedBandit::Testbed(10, 7);
      bandit.Seed(8);
      std::mt19937_64    rng(3);
      std::vector<int>   n(10, 0);
      std::vector<float> q(10, 0.f);
      thompson.Reset();
      for (int t = 0; t < 40; ++t) {
        const int a = thompson.Choose(n, q, t, rng);
        bandit.reset();
        const float r = bandit.step(a).reward;
        n[a]++, q[a] += (r - q[a]) / n[a];
        thompson.Observe(a, r);
      }
      best_pulls = n[bandit.best_arm()];
      std::printf("  best arm %d (mean %.2f) was pulled %d of 40 times\n", bandit.best_arm(), bandit.best_mean(),
                  best_pulls);
    }

    // ---- Checks: the lessons the tables teach.
    // Gaussian rows: 0 greedy, 1 eps 0.01, 2 eps 0.1, 3 optimistic, 4 UCB, 5 Thompson.
    const auto regret = [](const Summary& s) { return s.regret.back(); };
    for (int i : {2, 3, 4, 5})
      check::Require(regret(g[i]) < regret(g[0]), "an exploring algorithm should have less regret than pure greedy");
    check::Require(regret(g[4]) < regret(g[2]) && regret(g[5]) < regret(g[2]),
                   "UCB and Thompson should have less regret than eps-greedy 0.1");
    check::Require(g[2].optimal.back() > g[0].optimal.back() + 0.2,
                   "eps-greedy 0.1 should find the best arm far more often than greedy");
    check::Require(g[5].optimal.back() > 0.8, "Thompson should pull the best arm > 80% of the time by step 1000");
    // Bernoulli rows: 0 eps 0.1, 1 UCB c=1, 2 Thompson (Beta).
    check::Require(regret(b[2]) < 0.5 * regret(b[0]) && regret(b[2]) < 0.5 * regret(b[1]),
                   "Bernoulli regret: Thompson (Beta) should beat eps-greedy and UCB by a wide margin");
    // Linear vs logarithmic growth: eps-greedy's regret over steps 500-1000 is
    // still a large share of its first 500 steps; Thompson's is much smaller.
    const auto late_share = [](const Summary& s) { return (s.regret[999] - s.regret[499]) / s.regret[499]; };
    std::printf("\n  regret over steps 501-1000 / regret over steps 1-500: eps-greedy 0.1 %.2f, Thompson (Beta) %.2f\n",
                late_share(b[0]), late_share(b[2]));
    check::Require(late_share(b[0]) > 2 * late_share(b[2]),
                   "regret should grow ~linearly for eps-greedy and sublinearly for Thompson");
    check::Require(best_pulls >= 10, "the filmed Thompson run should already favour the best arm");
    std::cout << "\nchecks passed (vsim video omitted)\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
