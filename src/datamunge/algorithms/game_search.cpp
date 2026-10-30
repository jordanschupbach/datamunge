/// @file game_search.cpp
/// @brief Minimax, alpha-beta, move ordering, random and tic-tac-toe trees.

#include <datamunge/algorithms/game_search.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief Recursive helpers for the search routines.
namespace {

/// @brief Positive infinity, the initial alpha-beta window edge.
constexpr double kInf = std::numeric_limits<double>::infinity();

/// @brief Whether @p n is searched as a leaf (flagged, or with no children).
/// @param n the node.
/// @return true for a leaf.
bool is_leaf(const GameNode& n) { return n.leaf || n.children.empty(); }

/// @brief Plain minimax below @p node, counting leaves and nodes into @p r.
/// @param node the node to value.
/// @param maximizing whether max moves at @p node.
/// @param r counters to update.
/// @return the minimax value of @p node.
double minimax_rec(const GameNode& node, bool maximizing, GameResult& r) {
    ++r.nodes_visited;
    if (is_leaf(node)) {
        ++r.leaves_evaluated;
        return node.value;
    }
    double best = maximizing ? -kInf : kInf;
    for (const GameNode& c : node.children) {
        const double v = minimax_rec(c, !maximizing, r);
        best           = maximizing ? std::max(best, v) : std::min(best, v);
    }
    return best;
}

/// @brief Fail-soft alpha-beta below @p node with window (@p alpha, @p beta).
/// @param node the node to value.
/// @param maximizing whether max moves at @p node.
/// @param alpha lower window edge.
/// @param beta upper window edge.
/// @param r counters to update (leaves, nodes, cutoffs).
/// @return the value, exact inside the window and a bound outside it.
double alpha_beta_rec(const GameNode& node, bool maximizing, double alpha, double beta, GameResult& r) {
    ++r.nodes_visited;
    if (is_leaf(node)) {
        ++r.leaves_evaluated;
        return node.value;
    }
    if (maximizing) {
        double best = -kInf;
        for (const GameNode& c : node.children) {
            best  = std::max(best, alpha_beta_rec(c, false, alpha, beta, r));
            alpha = std::max(alpha, best);
            if (alpha >= beta) { ++r.cutoffs; break; }  // beta cutoff: the minimizer would never allow this
        }
        return best;
    }
    double best = kInf;
    for (const GameNode& c : node.children) {
        best = std::min(best, alpha_beta_rec(c, true, alpha, beta, r));
        beta = std::min(beta, best);
        if (alpha >= beta) { ++r.cutoffs; break; }  // alpha cutoff: the maximizer would never allow this
    }
    return best;
}

/// @brief Index of the first root child whose minimax value equals @p value.
/// @param root the root.
/// @param maximizing whether max moves at the root.
/// @param value the root's minimax value.
/// @return the child index, or -1 for a leaf root.
int find_best_child(const GameNode& root, bool maximizing, double value) {
    if (is_leaf(root)) return -1;
    for (std::size_t i = 0; i < root.children.size(); ++i) {
        GameResult scratch;
        if (minimax_rec(root.children[i], !maximizing, scratch) == value) return static_cast<int>(i);
    }
    return -1;
}

/// @brief Sorts the children of every node below @p node in place by minimax value.
/// @param node the subtree to reorder.
/// @param maximizing whether max moves at @p node.
/// @param order best-first or worst-first for the mover.
/// @return the minimax value of @p node.
double order_rec(GameNode& node, bool maximizing, MoveOrder order) {
    if (is_leaf(node)) return node.value;
    std::vector<double> values(node.children.size());
    for (std::size_t i = 0; i < node.children.size(); ++i)
        values[i] = order_rec(node.children[i], !maximizing, order);
    std::vector<std::size_t> idx(values.size());
    std::iota(idx.begin(), idx.end(), 0);
    const bool descending = (order == MoveOrder::BestFirst) == maximizing;
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return descending ? values[a] > values[b] : values[a] < values[b];
    });
    std::vector<GameNode> sorted;
    sorted.reserve(idx.size());
    for (std::size_t i : idx) sorted.push_back(std::move(node.children[i]));
    node.children = std::move(sorted);
    return maximizing ? *std::max_element(values.begin(), values.end())
                      : *std::min_element(values.begin(), values.end());
}

/// @brief Builds a uniform subtree with Uniform(0,1) leaves.
/// @param depth plies remaining.
/// @param branching children per node.
/// @param rng random engine.
/// @param u the leaf distribution.
/// @return the subtree root.
GameNode random_rec(int depth, int branching, std::mt19937_64& rng, std::uniform_real_distribution<double>& u) {
    GameNode n;
    if (depth == 0) {
        n.leaf  = true;
        n.value = u(rng);
        return n;
    }
    n.children.reserve(static_cast<std::size_t>(branching));
    for (int i = 0; i < branching; ++i) n.children.push_back(random_rec(depth - 1, branching, rng, u));
    return n;
}

}  // namespace

GameResult minimax(const GameNode& root, bool maximizing) {
    GameResult r;
    r.value      = minimax_rec(root, maximizing, r);
    r.best_child = find_best_child(root, maximizing, r.value);
    return r;
}

GameResult alpha_beta(const GameNode& root, bool maximizing) { return alpha_beta(root, maximizing, -kInf, kInf); }

GameResult alpha_beta(const GameNode& root, bool maximizing, double alpha, double beta) {
    GameResult r;
    r.value = alpha_beta_rec(root, maximizing, alpha, beta, r);
    if (r.value > alpha && r.value < beta) r.best_child = find_best_child(root, maximizing, r.value);
    return r;
}

GameNode random_game_tree(int depth, int branching, std::uint64_t seed) {
    if (depth < 0) throw std::invalid_argument("random_game_tree: depth must be non-negative");
    if (branching < 1) throw std::invalid_argument("random_game_tree: branching must be at least 1");
    std::mt19937_64                        rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    return random_rec(depth, branching, rng, u);
}

GameNode order_moves(const GameNode& root, bool maximizing, MoveOrder order) {
    GameNode copy = root;
    order_rec(copy, maximizing, order);
    return copy;
}

long long minimal_tree_leaves(int branching, int depth) {
    if (branching < 1) throw std::invalid_argument("minimal_tree_leaves: branching must be at least 1");
    if (depth < 0) throw std::invalid_argument("minimal_tree_leaves: depth must be non-negative");
    auto ipow = [](long long b, int e) {
        long long r = 1;
        for (int i = 0; i < e; ++i) r *= b;
        return r;
    };
    return ipow(branching, (depth + 1) / 2) + ipow(branching, depth / 2) - 1;
}

double alpha_beta_branching_factor(int branching) {
    if (branching < 1) throw std::invalid_argument("alpha_beta_branching_factor: branching must be at least 1");
    if (branching == 1) return 1.0;
    double lo = 0.0, hi = 1.0;  // x^b + x - 1 is increasing on [0,1], -1 at 0 and +1 at 1
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        (std::pow(mid, branching) + mid - 1.0 < 0.0 ? lo : hi) = mid;
    }
    const double xi = 0.5 * (lo + hi);
    return xi / (1.0 - xi);
}

namespace tictactoe {

/// @brief Board helpers for tic-tac-toe.
namespace {

/// @brief Throws unless @p board is 9 characters of X, O and '.'.
/// @param board the board to check.
void validate(const std::string& board) {
    if (board.size() != 9) throw std::invalid_argument("tictactoe: board must have 9 characters");
    for (char c : board)
        if (c != 'X' && c != 'O' && c != '.')
            throw std::invalid_argument("tictactoe: board characters must be X, O or '.'");
}

/// @brief The eight winning lines: three rows, three columns, two diagonals.
constexpr int kLines[8][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
                              {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};

/// @brief The winner of a board already validated.
/// @param b the board.
/// @return 'X', 'O' or '.'.
char winner_unchecked(const std::string& b) {
    for (const auto& l : kLines)
        if (b[l[0]] != '.' && b[l[0]] == b[l[1]] && b[l[1]] == b[l[2]]) return b[l[0]];
    return '.';
}

/// @brief Builds the game tree below @p b, editing @p b in place and restoring it.
/// @param b the board (restored on return).
/// @param x_moves whether X moves next.
/// @return the subtree root.
GameNode tree_rec(std::string& b, bool x_moves) {
    GameNode   n;
    const char w = winner_unchecked(b);
    if (w != '.' || b.find('.') == std::string::npos) {
        n.leaf  = true;
        n.value = w == 'X' ? 1.0 : (w == 'O' ? -1.0 : 0.0);
        return n;
    }
    for (std::size_t i = 0; i < 9; ++i) {
        if (b[i] != '.') continue;
        b[i] = x_moves ? 'X' : 'O';
        n.children.push_back(tree_rec(b, !x_moves));
        b[i] = '.';
    }
    return n;
}

}  // namespace

char winner(const std::string& board) {
    validate(board);
    return winner_unchecked(board);
}

bool x_to_move(const std::string& board) {
    validate(board);
    const auto xs = std::count(board.begin(), board.end(), 'X');
    const auto os = std::count(board.begin(), board.end(), 'O');
    return xs == os;
}

GameNode game_tree(const std::string& board) {
    validate(board);
    const auto xs = std::count(board.begin(), board.end(), 'X');
    const auto os = std::count(board.begin(), board.end(), 'O');
    if (xs != os && xs != os + 1) throw std::invalid_argument("tictactoe::game_tree: impossible move counts");
    std::string b = board;
    return tree_rec(b, xs == os);
}

}  // namespace tictactoe

}  // namespace datamunge::algorithms
