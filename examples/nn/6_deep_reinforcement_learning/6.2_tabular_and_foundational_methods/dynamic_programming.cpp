// CMD: just run-nn dynamic_programming
// Dynamic programming: policy iteration and value iteration.
//
// Port of tflow's dynamic_programming. Both methods need the MDP's model,
// which the grid environments expose as p(s', r | s, a). The model is packed
// into two tensors, P[s, a, s'] and R[s, a], and the Bellman backups become
// batched tensor ops: one einsum computes the expected next-state value for
// every (s, a) at once, and policy evaluation is a single linear solve
// (nn::einsum and nn::solve in place of TF's Einsum and MatrixSolve). The vsim
// videos of the greedy policy are omitted; the port adds one check, that
// policy and value iteration agree.

#include <cmath>
#include <cstdio>
#include <datamunge/nn/einsum.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/grid_world.hpp"
#include "common/rl/tabular.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

// P[s, a, s'] = probability of continuing in s'; transitions into terminal
// states contribute reward but no continuation. R[s, a] = expected reward.
struct Model {
  int    S, A;
  Tensor P, R;
};

Model BuildModel(const rl::TabularEnv& env) {
  Model m{env.num_states(), env.action_space().n, nn::zeros({env.num_states(), env.action_space().n, env.num_states()}),
          nn::zeros({env.num_states(), env.action_space().n})};
  float* P = m.P.data_ptr<float>();
  float* R = m.R.data_ptr<float>();
  for (int s = 0; s < m.S; ++s)
    for (int a = 0; a < m.A; ++a)
      for (const rl::Transition& t : env.transitions(s, a)) {
        R[s * m.A + a] += t.prob * t.reward;
        if (!t.terminal) P[(s * m.A + a) * m.S + t.next_state] += t.prob;
      }
  return m;
}

std::vector<float> ToVector(const Tensor& t) { return t.to_vector<float>(); }

std::vector<int> ToInts(const Tensor& t) { return t.to_vector<int>(); }

// Monte Carlo check of V(start) by running the policy in the real environment.
float DiscountedReturn(rl::TabularEnv& env, const std::vector<int>& policy, float gamma, int episodes) {
  double total = 0;
  for (int e = 0; e < episodes; ++e) {
    rl::Obs obs = env.reset(1000 + e);
    double  g = 0, discount = 1;
    while (true) {
      rl::StepResult r = env.step(policy[static_cast<int>(obs[0])]);
      g += discount * r.reward;
      discount *= gamma;
      obs = r.obs;
      if (r.done()) break;
    }
    total += g;
  }
  return static_cast<float>(total / episodes);
}

void Solve(rl::GridBase& env, float gamma) {
  std::cout << "\n==== " << env.name() << " (gamma = " << gamma << ")\n";
  env.reset(0);
  const Model  model = BuildModel(env);
  const int    S = model.S, A = model.A;
  const Tensor &P = model.P, &R = model.R;

  // ---- Value iteration: V <- max_a [ R[s,a] + gamma * sum_s' P[s,a,s'] V[s'] ]
  const auto q_of = [&](const Tensor& V) { return R + gamma * nn::einsum("sat,t->sa", P, V); };
  // ---- Policy evaluation as a linear system: (I - gamma P_pi) V = R_pi
  const auto evaluate = [&](const Tensor& pi) {
    const Tensor onehot = nn::one_hot(pi, A);                        // [S, A]
    const Tensor P_pi   = nn::einsum("sa,sat->st", onehot, P);       // [S, S]
    const Tensor R_pi   = nn::sum(onehot * R, {1});                  // [S]
    const Tensor system = nn::eye(S) - gamma * P_pi;
    return nn::squeeze(nn::solve(system, nn::expand_dims(R_pi, 1)), {1});
  };

  // Value iteration loop.
  Tensor V = nn::zeros({S});
  Tensor greedy;
  int    sweeps = 0;
  float  delta  = 0;
  do {
    const Tensor Q     = q_of(V);
    const Tensor V_out = nn::max(Q, {1});
    greedy             = nn::argmax(Q, 1);
    delta              = nn::max(nn::abs(V_out - V)).item<float>();
    V                  = V_out;
    ++sweeps;
  } while (delta > 1e-6f && sweeps < 10000);
  const std::vector<int> vi_policy = ToInts(greedy);
  std::printf("\n-- Value iteration converged in %d sweeps (max change %.1e)\n", sweeps, delta);
  std::cout << "  V*:\n" << rl::ValueGrid(env, ToVector(V)) << "  greedy policy:\n" << rl::PolicyGrid(env, vi_policy);

  // Policy iteration loop, starting from "always up".
  std::vector<int>   pi(S, 0);
  int                iterations = 0;
  std::vector<float> V_final;
  while (true) {
    const Tensor V_pi   = evaluate(nn::from_vector(pi));
    // Policy improvement: greedy w.r.t. Q_pi
    const auto   pi_out = ToInts(nn::argmax(q_of(V_pi), 1));
    ++iterations;
    bool stable = true;
    for (int s = 0; s < S; ++s) {
      // Keep the old action when it is still optimal (avoids flip-flopping on ties).
      if (pi[s] != pi_out[s]) {
        stable = false;
        pi[s]  = pi_out[s];
      }
    }
    V_final = ToVector(V_pi);
    if (stable || iterations > 100) break;
  }
  std::printf("\n-- Policy iteration converged in %d evaluate/improve rounds\n", iterations);
  std::cout << "  V^pi:\n" << rl::ValueGrid(env, V_final) << "  policy:\n" << rl::PolicyGrid(env, pi);
  float max_diff = 0;
  const auto v_star = ToVector(V);
  for (int s = 0; s < S; ++s) max_diff = std::max(max_diff, std::abs(V_final[s] - v_star[s]));
  std::printf("  max |V_policy_iteration - V_value_iteration| = %.2e\n", max_diff);
  check::Require(max_diff < 1e-4f, "policy and value iteration agree");

  const int start = env.state();  // still at the start cell after reset()
  std::printf("  V*(start) = %.3f; Monte Carlo return of the greedy policy over 2000 episodes = %.3f\n", v_star[start],
              DiscountedReturn(env, vi_policy, gamma, 2000));
}

}  // namespace

int main() {
  try {
    rl::GridWorld grid(/*slip_prob=*/0.2f);
    Solve(grid, 0.95f);
    rl::CliffWalking cliff;
    Solve(cliff, 0.99f);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
