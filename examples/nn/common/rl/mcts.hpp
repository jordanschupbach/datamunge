// datamunge::nn port: copied from tflow_examples src/common/rl/mcts.h (TF-free code, unchanged).
// Monte Carlo tree search over rl::Game, with a pluggable leaf evaluator:
// random rollouts (classic UCT) or a policy/value network (AlphaZero).
// Values are always from the point of view of the player to move at the
// node, and are negated on the way up the tree.
#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "common/rl/board_games.hpp"

namespace rl {

struct Evaluation {
  std::vector<float> prior;  // one entry per action (only legal ones are used)
  float value = 0.f;         // expected outcome for the player to move, in [-1, 1]
};
using Evaluator = std::function<Evaluation(const Game&)>;

// Terminal value for the player to move: the game just ended, so if there
// is a winner it is the opponent.
inline float TerminalValue(const Game& g) { return g.Winner() == 0 ? 0.f : -1.f; }

// Uniform prior; value from `rollouts` random playouts.
inline Evaluator RolloutEvaluator(std::mt19937_64* rng, int rollouts = 1) {
  return [rng, rollouts](const Game& g) {
    Evaluation e;
    e.prior.assign(g.num_actions(), 1.f);
    const int me = g.CurrentPlayer();
    float total = 0;
    for (int i = 0; i < rollouts; ++i) {
      std::unique_ptr<Game> sim = g.Clone();
      while (!sim->Terminal()) {
        std::vector<int> legal = sim->LegalActions();
        sim->Apply(legal[(*rng)() % legal.size()]);
      }
      total += sim->Winner() == 0 ? 0.f : (sim->Winner() == me ? 1.f : -1.f);
    }
    e.value = total / rollouts;
    return e;
  };
}

class Mcts {
 public:
  Mcts(Evaluator evaluator, float c_puct = 1.4f, float dirichlet_alpha = 0.3f, float noise_eps = 0.25f)
      : evaluator_(std::move(evaluator)), c_puct_(c_puct), alpha_(dirichlet_alpha), eps_(noise_eps) {}

  // Runs `simulations` from `root` and returns the visit count per action.
  std::vector<int> Search(const Game& root, int simulations, std::mt19937_64& rng, bool add_noise = false) {
    Node tree;
    Expand(&tree, root);
    if (add_noise) AddDirichletNoise(&tree, rng);
    for (int i = 0; i < simulations; ++i) {
      std::unique_ptr<Game> game = root.Clone();
      std::vector<Node*> path{&tree};
      Node* node = &tree;
      // Select down to a leaf.
      while (node->expanded && !game->Terminal()) {
        Node* child = SelectChild(node);
        game->Apply(child->action);
        path.push_back(child);
        node = child;
      }
      float value;
      if (game->Terminal()) value = TerminalValue(*game);
      else value = Expand(node, *game);
      // Back up, flipping sign each ply.
      for (int d = static_cast<int>(path.size()) - 1; d >= 0; --d) {
        path[d]->visits += 1;
        path[d]->value_sum += value;
        value = -value;
      }
    }
    std::vector<int> counts(root.num_actions(), 0);
    for (const auto& c : tree.children) counts[c->action] = c->visits;
    last_root_value_ = tree.Q();
    return counts;
  }
  // The root's backed-up value from the last search, from the point of view
  // of the player to move (+1 a sure win, -1 a sure loss).
  float last_root_value() const { return last_root_value_; }

 private:
  struct Node {
    int action = -1;
    float prior = 0.f;
    int visits = 0;
    float value_sum = 0.f;
    bool expanded = false;
    std::vector<std::unique_ptr<Node>> children;
    float Q() const { return visits ? value_sum / visits : 0.f; }
  };

  float Expand(Node* node, const Game& game) {
    Evaluation e = evaluator_(game);
    std::vector<int> legal = game.LegalActions();
    float norm = 0;
    for (int a : legal) norm += e.prior[a];
    for (int a : legal) {
      auto child = std::make_unique<Node>();
      child->action = a;
      child->prior = norm > 0 ? e.prior[a] / norm : 1.f / legal.size();
      node->children.push_back(std::move(child));
    }
    node->expanded = true;
    last_root_value_ = e.value;
    return e.value;
  }
  // PUCT: Q from the child's own perspective is the negation of what the
  // parent (who chooses) will get, so pick max of -Q + exploration.
  Node* SelectChild(Node* node) const {
    Node* best = nullptr;
    float best_score = -1e9f;
    const float sqrt_n = std::sqrt(static_cast<float>(node->visits) + 1e-8f);
    for (const auto& c : node->children) {
      const float score = -c->Q() + c_puct_ * c->prior * sqrt_n / (1 + c->visits);
      if (score > best_score) best_score = score, best = c.get();
    }
    return best;
  }
  void AddDirichletNoise(Node* root, std::mt19937_64& rng) {
    std::gamma_distribution<float> gamma(alpha_, 1.f);
    std::vector<float> noise(root->children.size());
    float sum = 0;
    for (float& x : noise) sum += (x = gamma(rng));
    for (size_t i = 0; i < noise.size(); ++i)
      root->children[i]->prior = (1 - eps_) * root->children[i]->prior + eps_ * noise[i] / sum;
  }

  Evaluator evaluator_;
  float c_puct_, alpha_, eps_;
  float last_root_value_ = 0.f;
};

}  // namespace rl

