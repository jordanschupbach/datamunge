#pragma once

/// @file game_search.hpp
/// @brief Two-player zero-sum game-tree search: minimax, alpha-beta pruning, move ordering,
///        random and tic-tac-toe trees, and the Knuth-Moore and Pearl leaf-count formulas.

#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief A node of a two-player game tree: a @ref leaf carrying a heuristic @ref value, or an
///        internal node with @ref children. Levels alternate between the maximizing and minimizing
///        player.
struct GameNode {
    bool                  leaf{false}; ///< true if this is a terminal/evaluated position.
    double                value{0.0};  ///< the value at a leaf (ignored for internal nodes).
    std::vector<GameNode> children;    ///< child positions (moves) for an internal node.
};

/// @brief The outcome of a game-tree search: the minimax @ref value of the root and the number of
///        @ref leaves_evaluated to obtain it (the work done).
struct GameResult {
    double    value{0.0};            ///< the minimax value of the root.
    long long leaves_evaluated{0};   ///< number of leaf positions actually evaluated.
    long long nodes_visited{0};      ///< number of nodes entered, leaves and internal nodes together.
    long long cutoffs{0};            ///< number of times a node stopped early (alpha >= beta).
    int       best_child{-1};        ///< index of the first root child achieving @ref value (-1 at a leaf).
};

/// @brief The order in which order_moves() arranges each node's children.
enum class MoveOrder {
    BestFirst,   ///< the mover's best child first: max nodes descending, min nodes ascending.
    WorstFirst,  ///< the mover's worst child first: the reverse of BestFirst, which defeats most cutoffs.
};

/// @brief *Minimax*: the value of a game position under optimal play. The maximizing player picks the
///        child of greatest value, the minimizing player the child of least value, alternating down
///        the tree; leaf values propagate up. Plain minimax evaluates *every* leaf.
///
/// @param root the root position.
/// @param maximizing whether the player to move at the root is maximizing.
/// @return the minimax @ref GameResult (value and the full leaf count).
GameResult minimax(const GameNode& root, bool maximizing);

/// @brief *Alpha-beta pruning*: minimax that carries a window @c [alpha,beta] of values still
///        relevant and *prunes* a subtree as soon as it cannot affect the result (a max node whose
///        value already exceeds @c beta, or a min node below @c alpha). It returns the *identical*
///        value as @ref minimax but can skip large parts of the tree; with good move ordering it
///        roughly *halves the effective depth*, examining @c O(b^{d/2}) leaves instead of @c O(b^d).
///
/// @param root the root position.
/// @param maximizing whether the player to move at the root is maximizing.
/// @return the @ref GameResult with the same value as minimax and the (smaller) leaf count.
GameResult alpha_beta(const GameNode& root, bool maximizing);

/// @brief Alpha-beta started from the window @c (alpha,beta) instead of @c (-inf,+inf).
///
/// If the true value @c v lies strictly inside the window the result is exact. Otherwise it is
/// a bound: a result @c <= alpha means @c v <= result (fail low), and @c >= beta means
/// @c v >= result (fail high). Narrow windows prune more, which aspiration search and
/// null-window tests exploit.
/// @param root the root position.
/// @param maximizing whether the player to move at the root is maximizing.
/// @param alpha lower end of the initial window.
/// @param beta upper end of the initial window.
/// @return the @ref GameResult; @c value is exact only when it falls inside the window.
GameResult alpha_beta(const GameNode& root, bool maximizing, double alpha, double beta);

/// @brief A complete game tree with @p branching children per internal node and independent
///        Uniform(0,1) leaf values at depth @p depth (the model of Knuth-Moore and Pearl).
/// @param depth number of plies; 0 gives a single leaf.
/// @param branching children per internal node (at least 1).
/// @param seed random seed for the leaf values.
/// @return the root of a tree with @c branching^depth leaves.
/// @throws std::invalid_argument if @p depth is negative or @p branching is below 1.
GameNode random_game_tree(int depth, int branching, std::uint64_t seed);

/// @brief Copy of @p root with every node's children sorted by their minimax values.
///
/// The player to move at each node sees its children in the chosen @p order. BestFirst gives
/// alpha-beta its best case, the *minimal tree*. WorstFirst removes every cutoff that comes
/// from a node's own siblings, though bounds inherited from higher up can still prune.
/// Ties keep their original order.
/// @param root the root position.
/// @param maximizing whether the player to move at the root is maximizing.
/// @param order the ordering to apply.
/// @return the reordered tree (same minimax value).
GameNode order_moves(const GameNode& root, bool maximizing, MoveOrder order);

/// @brief Leaves in the Knuth-Moore minimal tree of a uniform tree:
///        \f$b^{\lceil d/2\rceil}+b^{\lfloor d/2\rfloor}-1\f$.
///
/// This is the number of leaves alpha-beta evaluates with perfect move ordering and distinct
/// leaf values, and a lower bound for any ordering.
/// @param branching branching factor @c b (at least 1).
/// @param depth depth @c d (at least 0).
/// @return the leaf count.
/// @throws std::invalid_argument for a branching factor below 1 or a negative depth.
long long minimal_tree_leaves(int branching, int depth);

/// @brief Pearl's asymptotic branching factor of alpha-beta on random uniform trees,
///        \f$\xi_b/(1-\xi_b)\f$ where \f$\xi_b\f$ is the positive root of \f$x^b+x-1=0\f$.
///
/// With randomly ordered children and continuous independent leaf values, alpha-beta evaluates
/// about \f$R_b^{\,d}\f$ leaves as the depth grows; \f$R_b\approx b^{3/4}\f$ for large @c b.
/// @param branching branching factor @c b (at least 1).
/// @return the asymptotic effective branching factor (1 when @c b is 1).
/// @throws std::invalid_argument for a branching factor below 1.
double alpha_beta_branching_factor(int branching);

/// @brief Tic-tac-toe positions as 9-character strings read row by row: @c 'X', @c 'O' or
///        @c '.' for an empty square. X moves first, so the mover is X when both have
///        played equally often.
namespace tictactoe {

/// @brief The winner of a board: @c 'X', @c 'O', or @c '.' if neither has three in a row.
/// @param board a 9-character board.
/// @return the winning mark or @c '.'.
/// @throws std::invalid_argument if @p board is not 9 characters of @c X, @c O and @c '.'.
char winner(const std::string& board);

/// @brief The full game tree below @p board, with X as the maximizing player.
///
/// A position is a leaf when someone has won or the board is full. Leaf values are +1 for an
/// X win, -1 for an O win and 0 for a draw. Children follow the empty squares in reading
/// order. From the empty board the tree has 255 168 leaves (complete games).
/// @param board the starting position; the default is the empty board.
/// @return the root of the tree.
/// @throws std::invalid_argument for a malformed board or impossible move counts.
GameNode game_tree(const std::string& board = ".........");

/// @brief Whether X is to move on @p board (X has not moved more often than O).
/// @param board a 9-character board.
/// @return @c true when X moves next.
/// @throws std::invalid_argument for a malformed board.
bool x_to_move(const std::string& board);

}  // namespace tictactoe

} // namespace datamunge::algorithms
