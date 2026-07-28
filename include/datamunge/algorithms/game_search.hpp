#pragma once

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

} // namespace datamunge::algorithms
