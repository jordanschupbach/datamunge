#pragma once
/// @file mcts.hpp
/// @brief Native source PUCT search over owning game clones and pluggable leaf evaluations.
#include <algorithm>
#include <cmath>
#include <datamunge/rl/environments/board_games.hpp>
#include <functional>
#include <limits>
#include <memory>
#include <random>
/// @brief Owning planning data, explicit cloned-game search and caller-driven model components.
namespace datamunge::rl::planning {
/// @brief Polymorphic two-player game contract used by cloned-state tree search.
///
/// @par Example
/// @code
/// using Game=dm::rl::planning::Game;
/// @endcode
/// @see environments::Game
using Game = environments::Game;
/// @brief Policy prior and current-player value returned by a leaf evaluator.
///
/// @par Example
/// @code
/// dm::rl::planning::Evaluation leaf{{1,1},0};
/// @endcode
/// @see Evaluator
struct Evaluation {
  std::vector<float> prior; ///< One finite nonnegative entry per action; legal entries normalized.
  float              value = 0.f; ///< Expected outcome for player to move, finite in[-1,1].
};
/// @brief Caller-supplied game-to-leaf evaluator, copied into a search object.
///
/// Callable accepts a borrowed immutable game; returned arrays own their values. It must satisfy
/// Evaluation bounds and may create caller model parameters. Search is not safe concurrently on the
/// same instance.
///
/// @par Example
/// @code
/// dm::rl::planning::Evaluator f=[](const dm::rl::planning::Game& g){return
/// dm::rl::planning::Evaluation{std::vector<float>(g.num_actions(),1),0};};
/// @endcode
/// @see Evaluation
using Evaluator = std::function<Evaluation(const Game&)>;

/// @brief Return the source terminal outcome from the player-to-move perspective.
///
/// @param g Terminal alternating two-player game; caller establishes terminality.
/// @return Zero for a draw, otherwise-1: a winner is assumed to be the previous mover.
/// @par Example
/// @code
/// dm::rl::environments::TicTacToe g;for(int a:{0,3,1,4,2})g.Apply(a);
/// auto value=dm::rl::planning::TerminalValue(g);
/// @endcode
/// @see Game::Winner()
inline float TerminalValue(const Game& g) {
  return g.Winner() == 0 ? 0.f : -1.f;
}

/// @brief Create a uniform-prior evaluator using random playouts on game clones.
///
/// Requires finite terminating game episodes. Throws std::invalid_argument for invalid
/// configuration, std::logic_error for nonterminal games without legal actions. Original game is
/// never mutated. Callers synchronize engine access.
///
/// @param rng Nonnull borrowed engine; must outlive returned callable and its copies.
/// @param rollouts Number of complete playouts per evaluation, default1;1..1000000.
/// @return Owning callable borrowing rng; each clone consumes raw modulo action draws.
/// @par Example
/// @code
/// dm::rl::environments::TicTacToe game;std::mt19937_64 rng(0);
/// dm::rl::planning::Mcts search(dm::rl::planning::RolloutEvaluator(&rng));
/// auto visits=search.Search(game,20,rng);
/// @endcode
/// @par References
/// - Silver et al. (2017), Mastering the game of Go without human knowledge,
/// https://doi.org/10.1038/nature24270 .
/// @see Mcts
inline Evaluator RolloutEvaluator(std::mt19937_64* rng, int rollouts = 1) {
  if (!rng || rollouts < 1 || rollouts > 1000000)
    throw std::invalid_argument("caller RNG and positive bounded rollout count required");
  return [rng, rollouts](const Game& g) {
    Evaluation e;
    e.prior.assign(g.num_actions(), 1.f);
    const int me    = g.CurrentPlayer();
    float     total = 0;
    for (int i = 0; i < rollouts; ++i) {
      std::unique_ptr<Game> sim = g.Clone();
      while (!sim->Terminal()) {
        std::vector<int> legal = sim->LegalActions();
        if (legal.empty())
          throw std::logic_error("nonterminal game has no legal action");
        sim->Apply(legal[(*rng)() % legal.size()]);
      }
      total += sim->Winner() == 0 ? 0.f : (sim->Winner() == me ? 1.f : -1.f);
    }
    e.value = total / rollouts;
    return e;
  };
}

/// @brief PUCT tree search with cloned games, sign-flipping backup and optional root noise.
///
/// Each Search discards the previous tree and borrows its root without mutation. The evaluator is
/// owned by value; any captured stores/engines remain caller-owned. No match, self-play, fitting or
/// replay loop is owned. Calls on one search object require synchronization.
///
/// @par Example
/// @code
/// dm::rl::environments::TicTacToe game;std::mt19937_64 rng(0);
/// dm::rl::planning::Mcts search(dm::rl::planning::RolloutEvaluator(&rng));
/// auto visits=search.Search(game,20,rng);
/// @endcode
/// @par References
/// - Silver et al. (2017), Mastering the game of Go without human knowledge,
/// https://doi.org/10.1038/nature24270 .
/// @see Evaluator, Search()
class Mcts {
 public:
  /// @brief Store the leaf evaluator and validate PUCT/noise configuration.
  ///
  /// Invalid configuration throws std::invalid_argument; parameter stores captured by the evaluator
  /// remain borrowed.
  ///
  /// @param evaluator Nonempty callable copied/moved into this object.
  /// @param c_puct Finite nonnegative exploration scale, default1.4.
  /// @param dirichlet_alpha Finite positive root gamma shape, default.3.
  /// @param noise_eps Finite root noise mixture fraction in[0,1], default.25.
  /// @par Example
  /// @code
  /// dm::rl::environments::TicTacToe game;std::mt19937_64 rng(0);
  /// dm::rl::planning::Mcts search(dm::rl::planning::RolloutEvaluator(&rng));
  /// auto visits=search.Search(game,20,rng);
  /// @endcode
  /// @par References
  /// - Silver et al. (2017), Mastering the game of Go without human knowledge,
  /// https://doi.org/10.1038/nature24270 .
  /// @see Search()
  Mcts(
      Evaluator evaluator,
      float     c_puct          = 1.4f,
      float     dirichlet_alpha = 0.3f,
      float     noise_eps       = 0.25f)
      : evaluator_(std::move(evaluator)),
        c_puct_(c_puct),
        alpha_(dirichlet_alpha),
        eps_(noise_eps) {
    if (!evaluator_ || !std::isfinite(c_puct_) || c_puct_ < 0 || !std::isfinite(alpha_) ||
        alpha_ <= 0 || !std::isfinite(eps_) || eps_ < 0 || eps_ > 1)
      throw std::invalid_argument("valid evaluator, PUCT scale and Dirichlet parameters required");
  }

  /// @brief Expand a fresh root and run the requested number of cloned-state simulations.
  ///
  /// Evaluator is called once for root even at zero simulations or terminal roots. Zero simulations
  /// stores root value0 after backup. Ties retain legal-action order. Invalid priors, values or
  /// dimensions throw std::invalid_argument; nonterminal empty legal actions throw
  /// std::logic_error. Partial state and caller engine may have changed on evaluator failure.
  ///
  /// @param root Borrowed immutable game with1..1000000 actions and valid legal indices.
  /// @param simulations Number of simulated paths,0..1000000; default none is not implied.
  /// @param rng Borrowed engine used only for optional root noise; evaluator may borrow it too.
  /// @param add_noise Mix sampled Dirichlet root priors when true, default false.
  /// @return One visit count per action; illegal actions have zero counts.
  /// @par Example
  /// @code
  /// dm::rl::environments::TicTacToe game;std::mt19937_64 rng(0);
  /// dm::rl::planning::Mcts search(dm::rl::planning::RolloutEvaluator(&rng));
  /// auto visits=search.Search(game,20,rng);
  /// @endcode
  /// @par References
  /// - Silver et al. (2017), Mastering the game of Go without human knowledge,
  /// https://doi.org/10.1038/nature24270 .
  /// @see last_root_value()
  std::vector<int>
  Search(const Game& root, int simulations, std::mt19937_64& rng, bool add_noise = false) {
    if (simulations < 0 || simulations > 1000000 || root.num_actions() < 1 ||
        root.num_actions() > 1000000)
      throw std::invalid_argument("bounded search simulation/action counts required");
    Node tree;
    Expand(&tree, root);
    if (add_noise)
      AddDirichletNoise(&tree, rng);
    for (int i = 0; i < simulations; ++i) {
      std::unique_ptr<Game> game = root.Clone();
      std::vector<Node*>    path{&tree};
      Node*                 node = &tree;
      // Select down to a leaf.
      while (node->expanded && !game->Terminal()) {
        Node* child = SelectChild(node);
        game->Apply(child->action);
        path.push_back(child);
        node = child;
      }
      float value;
      if (game->Terminal())
        value = TerminalValue(*game);
      else
        value = Expand(node, *game);
      // Back up, flipping sign each ply.
      for (int d = static_cast<int>(path.size()) - 1; d >= 0; --d) {
        path[d]->visits += 1;
        path[d]->value_sum += value;
        value = -value;
      }
    }
    std::vector<int> counts(root.num_actions(), 0);
    for (const auto& c : tree.children)
      counts[c->action] = c->visits;
    last_root_value_ = tree.Q();
    return counts;
  }
  /// @brief Read the last backed-up root mean from the current-player perspective.
  ///
  /// @return Mean in[-1,1] after a successful search;0 before any search or after zero simulations.
  /// @see Search()
  float last_root_value() const { return last_root_value_; }

 private:
  /// @brief Owning node with statistics expressed for its player to move.
  struct Node {
    int                                action    = -1;    ///< Incoming action; root sentinel-1.
    float                              prior     = 0.f;   ///< Normalized incoming policy mass.
    int                                visits    = 0;     ///< Number of backed-up paths.
    float                              value_sum = 0.f;   ///< Sum of own-player values.
    bool                               expanded  = false; ///< Leaf evaluator has been applied.
    std::vector<std::unique_ptr<Node>> children; ///< Owned legal successors in source order.

    /// @brief Read the node mean, using0 for an unvisited node.
    ///
    /// @return Own-player mean value.
    /// @see Search()
    float Q() const { return visits ? value_sum / visits : 0.f; }
  };

  /// @brief Evaluate and validate a leaf, then create normalized legal successors.
  ///
  /// Invalid evaluation throws std::invalid_argument; empty nonterminal legal actions throw
  /// std::logic_error.
  ///
  /// @param node Borrowed unexpanded node mutated in place.
  /// @param game Borrowed corresponding game state.
  /// @return Finite own-player leaf value.
  /// @see Evaluation
  float Expand(Node* node, const Game& game) {
    Evaluation       e     = evaluator_(game);
    std::vector<int> legal = game.LegalActions();
    if (e.prior.size() != size_t(game.num_actions()) || !std::isfinite(e.value) || e.value < -1 ||
        e.value > 1)
      throw std::invalid_argument("aligned MCTS priors and bounded finite value required");
    for (float p : e.prior)
      if (!std::isfinite(p) || p < 0)
        throw std::invalid_argument("finite nonnegative MCTS priors required");
    for (int a : legal)
      if (a < 0 || a >= game.num_actions())
        throw std::invalid_argument("legal action out of range");
    if (!game.Terminal() && legal.empty())
      throw std::logic_error("nonterminal game has no legal action");
    float norm = 0;
    for (int a : legal)
      norm += e.prior[a];
    if (!std::isfinite(norm))
      throw std::invalid_argument("finite legal prior mass required");
    for (int a : legal) {
      auto child    = std::make_unique<Node>();
      child->action = a;
      child->prior  = norm > 0 ? e.prior[a] / norm : 1.f / legal.size();
      node->children.push_back(std::move(child));
    }
    node->expanded   = true;
    last_root_value_ = e.value;
    return e.value;
  }
  /// @brief Select first maximum of negated child value plus PUCT bonus.
  ///
  /// @param node Borrowed expanded nonterminal node with nonempty children.
  /// @return Borrowed winning child, invalidated with its tree.
  /// @par References
  /// - Silver et al. (2017), Mastering the game of Go without human knowledge,
  /// https://doi.org/10.1038/nature24270 .
  /// @see Search()
  Node* SelectChild(Node* node) const {
    Node*       best       = nullptr;
    float       best_score = -std::numeric_limits<float>::infinity();
    const float sqrt_n     = std::sqrt(static_cast<float>(node->visits) + 1e-8f);
    for (const auto& c : node->children) {
      const float score = -c->Q() + c_puct_ * c->prior * sqrt_n / (1 + c->visits);
      if (score > best_score)
        best_score = score, best = c.get();
    }
    return best;
  }
  /// @brief Mix root priors with gamma-normalized noise using source draw order.
  ///
  /// Nonfinite/underflowed total gamma mass uses uniform noise after the same draws.
  ///
  /// @param root Borrowed expanded root; priors mutated.
  /// @param rng Borrowed engine consumed by gamma sampling.
  /// @see Search()
  void AddDirichletNoise(Node* root, std::mt19937_64& rng) {
    std::gamma_distribution<float> gamma(alpha_, 1.f);
    std::vector<float>             noise(root->children.size());
    float                          sum = 0;
    for (float& x : noise)
      sum += (x = gamma(rng));
    if (!std::isfinite(sum) || sum <= 0) {
      std::fill(noise.begin(), noise.end(), 1.f);
      sum = float(noise.size());
    }
    for (size_t i = 0; i < noise.size(); ++i)
      root->children[i]->prior = (1 - eps_) * root->children[i]->prior + eps_ * noise[i] / sum;
  }

  Evaluator evaluator_;             ///< Owned callable; captured references remain borrowed.
  float     c_puct_;                ///< PUCT exploration scale.
  float     alpha_;                 ///< Gamma shape for root noise.
  float     eps_;                   ///< Root noise mixture weight.
  float     last_root_value_ = 0.f; ///< Last root mean, overwritten during search.
};

} // namespace datamunge::rl::planning
