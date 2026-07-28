#include <datamunge/algorithms/game_search.hpp>

#include <algorithm>
#include <limits>

namespace datamunge::algorithms {

namespace {

double minimax_rec(const GameNode& node, bool maximizing, long long& leaves) {
    if (node.leaf || node.children.empty()) {
        ++leaves;
        return node.value;
    }
    if (maximizing) {
        double best = -std::numeric_limits<double>::infinity();
        for (const GameNode& c : node.children) best = std::max(best, minimax_rec(c, false, leaves));
        return best;
    }
    double best = std::numeric_limits<double>::infinity();
    for (const GameNode& c : node.children) best = std::min(best, minimax_rec(c, true, leaves));
    return best;
}

double alpha_beta_rec(const GameNode& node, bool maximizing, double alpha, double beta, long long& leaves) {
    if (node.leaf || node.children.empty()) {
        ++leaves;
        return node.value;
    }
    if (maximizing) {
        double best = -std::numeric_limits<double>::infinity();
        for (const GameNode& c : node.children) {
            best  = std::max(best, alpha_beta_rec(c, false, alpha, beta, leaves));
            alpha = std::max(alpha, best);
            if (alpha >= beta) break; // beta cutoff: the minimizer would never allow this
        }
        return best;
    }
    double best = std::numeric_limits<double>::infinity();
    for (const GameNode& c : node.children) {
        best = std::min(best, alpha_beta_rec(c, true, alpha, beta, leaves));
        beta = std::min(beta, best);
        if (alpha >= beta) break; // alpha cutoff: the maximizer would never allow this
    }
    return best;
}

} // namespace

GameResult minimax(const GameNode& root, bool maximizing) {
    GameResult r;
    r.value = minimax_rec(root, maximizing, r.leaves_evaluated);
    return r;
}

GameResult alpha_beta(const GameNode& root, bool maximizing) {
    GameResult   r;
    const double inf = std::numeric_limits<double>::infinity();
    r.value          = alpha_beta_rec(root, maximizing, -inf, inf, r.leaves_evaluated);
    return r;
}

} // namespace datamunge::algorithms
