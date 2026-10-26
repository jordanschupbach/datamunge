// CMD: just run-nn alphazero
// AlphaZero: self-play, MCTS and a policy/value network.
//
// Port of tflow's alphazero. Silver et al. 2017, scaled down to tic-tac-toe.
// A network maps the board (from the player to move's point of view) to move
// probabilities and a value in [-1, 1]. MCTS uses those as priors and leaf
// values instead of random playouts (PUCT). Self-play games, with Dirichlet
// noise at the root for exploration, produce training targets: the search's
// visit distribution for the policy and the game outcome for the value.
// Training and self-play alternate until the agent stops losing.
//
// The TF graph and session become an eager nn network over a ParameterStore
// (the MCTS evaluator runs it under NoGradGuard); the search (common/rl/
// mcts.hpp, copied from tflow), the games and the std::mt19937_64 stream are
// tflow's. Weights use nn's generator, so the training numbers match tflow
// only statistically. tflow films one game of the trained agent against
// MCTS(200) on vsim's 3D board; the film is omitted but the game is played
// and its outcome printed. Options: seed=N (weight seed, default 0),
// threads=N. The port adds checks (exit 1 on failure).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <deque>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/rl/board_games.hpp"
#include "common/rl/dqn_common.hpp"
#include "common/rl/mcts.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

struct Example {
  rl::Obs board;
  std::vector<float> pi;
  float z;
};

}  // namespace

int main(int argc, char** argv) {
  std::uint64_t seed = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = arg.substr(0, eq), val = arg.substr(eq + 1);
    if (key == "seed") seed = std::strtoull(val.c_str(), nullptr, 10);
    else if (key == "threads") nn::set_num_threads(std::atoi(val.c_str()));
    else if (!rl::IsHardwareOption(key)) std::fprintf(stderr, "unknown option %s\n", arg.c_str());
  }
  try {
    rl::TicTacToe game;
    const int cells = game.num_actions(), A = cells, hidden = 64;
    const int iterations = 25, games_per_iter = 20, simulations = 50, batch = 64, train_steps = 60;
    std::printf("== AlphaZero on tic-tac-toe: %d iterations of %d self-play games with %d simulations per move\n",
                iterations, games_per_iter, simulations);

    nn::ParameterStore params(seed);
    struct Out {
      Tensor logp, value;  // [B, 9], [B]
    };
    auto net = [&](const Tensor& board) {
      const Tensor h = nn::mlp(params, "trunk", board, {hidden, hidden}, nn::Activation::Relu);
      Out o;
      o.logp = nn::log_softmax(nn::dense(params, "policy", h, A), 1);
      o.value = nn::squeeze(nn::tanh(nn::dense(params, "value", h, 1)), {1});
      return o;
    };
    {
      nn::NoGradGuard no_grad;
      net(nn::zeros({1, cells}));
    }
    rl::Adam adam(params, 1e-3, 1.0);

    std::mt19937_64 rng(0);
    // The network as an MCTS evaluator: illegal moves are masked by the search.
    auto probe = [&](const rl::Game& g) {
      nn::NoGradGuard no_grad;
      const Out o = net(rl::Row(g.Observation()));
      rl::Evaluation e;
      e.prior = nn::exp(o.logp).to_vector<float>();
      e.value = o.value.item<float>();
      return e;
    };
    rl::Evaluator net_eval = probe;
    rl::Mcts search(net_eval, /*c_puct=*/1.5f);
    rl::Mcts rollout_search(rl::RolloutEvaluator(&rng), 1.4f);
    std::deque<Example> replay;

    // Play `games` games of agent (net + MCTS, no noise, argmax) vs an opponent.
    auto evaluate = [&](int games, const std::function<int(rl::Game&)>& opponent, int* wins, int* losses, int* draws) {
      *wins = *losses = *draws = 0;
      for (int g = 0; g < games; ++g) {
        game.Reset();
        const int agent_player = g % 2 + 1;
        while (!game.Terminal()) {
          int action;
          if (game.CurrentPlayer() == agent_player) {
            std::vector<int> counts = search.Search(game, simulations, rng, false);
            action = static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
          } else {
            action = opponent(game);
          }
          game.Apply(action);
        }
        if (game.Winner() == 0) ++*draws;
        else if (game.Winner() == agent_player) ++*wins;
        else ++*losses;
      }
    };
    auto random_opponent = [&](rl::Game& g) {
      auto l = g.LegalActions();
      return l[rng() % l.size()];
    };
    auto mcts_opponent = [&](rl::Game& g) {
      std::vector<int> counts = rollout_search.Search(g, 200, rng);
      return static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
    };

    std::printf("  %4s %8s %8s %22s %22s\n", "iter", "examples", "loss", "vs random (W/L/D)", "vs MCTS-200 (W/L/D)");
    int last_l1 = -1, last_l2 = -1, last_w1 = 0;
    float first_loss = 0, last_loss = 0;
    for (int it = 1; it <= iterations; ++it) {
      // ---- Self-play with exploration: Dirichlet noise at the root and
      // sampling moves from the visit distribution for the first moves.
      for (int g = 0; g < games_per_iter; ++g) {
        game.Reset();
        std::vector<Example> episode;
        std::vector<int> players;
        while (!game.Terminal()) {
          std::vector<int> counts = search.Search(game, simulations, rng, /*add_noise=*/true);
          std::vector<float> pi(A, 0.f);
          float total = 0;
          for (int a = 0; a < A; ++a) total += counts[a];
          for (int a = 0; a < A; ++a) pi[a] = counts[a] / total;
          episode.push_back({game.Observation(), pi, 0.f});
          players.push_back(game.CurrentPlayer());
          int action;
          if (static_cast<int>(episode.size()) <= 2) {  // temperature 1 early, then greedy
            float u = std::uniform_real_distribution<float>(0, 1)(rng), acc = 0;
            action = A - 1;
            for (int a = 0; a < A; ++a) {
              acc += pi[a];
              if (u < acc) {
                action = a;
                break;
              }
            }
          } else {
            action = static_cast<int>(std::max_element(pi.begin(), pi.end()) - pi.begin());
          }
          game.Apply(action);
        }
        for (size_t i = 0; i < episode.size(); ++i) {
          episode[i].z = game.Winner() == 0 ? 0.f : (game.Winner() == players[i] ? 1.f : -1.f);
          replay.push_back(episode[i]);
        }
        while (replay.size() > 4000) replay.pop_front();
      }
      // ---- Training on sampled examples.
      for (int step = 0; step < train_steps; ++step) {
        std::vector<float> b(static_cast<size_t>(batch) * cells), p(static_cast<size_t>(batch) * A), z(batch);
        for (int i = 0; i < batch; ++i) {
          const Example& e = replay[rng() % replay.size()];
          for (int j = 0; j < cells; ++j) b[i * cells + j] = e.board[j];
          for (int j = 0; j < A; ++j) p[i * A + j] = e.pi[j];
          z[i] = e.z;
        }
        const Tensor pi_t = nn::from_vector(p, {batch, A}), z_t = nn::from_vector(z, {batch});
        const Out o = net(nn::from_vector(b, {batch, cells}));
        const Tensor loss = nn::mean(nn::square(z_t - o.value), {0}) - nn::mean(nn::sum(pi_t * o.logp, {1}), {0});
        adam.Minimize(loss);
        last_loss = loss.item<float>();
        check::Require(std::isfinite(last_loss), "finite training loss");
      }
      if (it == 1) first_loss = last_loss;
      if (it % 5 == 0 || it == 1) {
        int w1, l1, d1, w2, l2, d2;
        evaluate(40, random_opponent, &w1, &l1, &d1);
        evaluate(20, mcts_opponent, &w2, &l2, &d2);
        std::printf("  %4d %8zu %8.3f %10d/%d/%-9d %10d/%d/%-9d\n", it, replay.size(), last_loss, w1, l1, d1, w2, l2, d2);
        std::fflush(stdout);
        last_w1 = w1, last_l1 = l1, last_l2 = l2;
      }
    }
    std::printf("  (tic-tac-toe is a draw with perfect play: a trained agent never loses and beats random often)\n");

    // In tflow, on film: the trained agent (X) against the MCTS opponent. Played here without the video.
    game.Reset();
    while (!game.Terminal()) {
      const bool agent = game.CurrentPlayer() == 1;
      std::vector<int> counts = agent ? search.Search(game, simulations, rng, false) : rollout_search.Search(game, 200, rng);
      game.Apply(static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin()));
    }
    const int filmed_winner = game.Winner();
    std::printf("  played game vs MCTS(200) (tflow films it): %s\n",
                filmed_winner == 0 ? "draw" : filmed_winner == 1 ? "the agent wins" : "the agent loses");
    // The network alone, without search, on the empty board.
    game.Reset();
    const rl::Evaluation e = probe(game);
    std::printf("  raw network on the empty board: value %.2f, move probabilities:\n", e.value);
    for (int r = 0; r < 3; ++r) {
      std::printf("   ");
      for (int c = 0; c < 3; ++c) std::printf(" %5.2f", e.prior[r * 3 + c]);
      std::printf("\n");
    }

    check::Require(last_loss < first_loss, "the training loss falls from the first iteration");
    check::Require(last_l1 <= 1 && last_w1 >= 20, "the trained agent (almost) never loses to random play and beats it in >= half the games");
    check::Require(last_l2 <= 5, "the trained agent loses at most 5 of 20 games to MCTS(200)");  // TF: 0 (iter 25), 3 (iter 20); ours seeds 0-5: 0-5
    check::Require(filmed_winner != 2, "the trained agent does not lose the game against MCTS(200)");
    std::printf("\nchecks passed (vsim video omitted)\n");
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "CHECK FAILED: %s\n", ex.what());
    return 1;
  }
  return 0;
}
