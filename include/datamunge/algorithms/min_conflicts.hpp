#pragma once

#include <vector>

namespace datamunge::algorithms {

/// @brief The result of a min-conflicts search for the @c n-queens problem: the placement
///        @ref queens (one column per row), whether it was @ref solved, and the number of repair
///        @ref steps taken.
struct NQueensResult {
    std::vector<int> queens;       ///< queens[row] = column of the queen in that row.
    bool             solved{false}; ///< true if a conflict-free placement was found within the step cap.
    int              steps{0};      ///< number of repair iterations performed.
};

/// @brief The *min-conflicts* heuristic (Minton et al., 1992) solving the @c n-queens problem. It is
///        *local search* for constraint satisfaction: start from a complete (conflicting) assignment,
///        then repeatedly pick a variable that is in conflict and move it to the value that *minimizes*
///        the number of conflicts (ties broken randomly). Despite its simplicity it solves the
///        million-queens problem in seconds -- famously, its runtime is nearly independent of @c n --
///        because most CSPs have solutions densely scattered through the assignment space, reachable
///        by a short greedy repair walk. One queen per row is fixed, so only column conflicts (shared
///        column or diagonal) matter.
///
/// @param n the board size / number of queens.
/// @param max_steps the maximum number of repair moves before giving up (a restart cap).
/// @param seed the RNG seed (for the random initial assignment and tie-breaking).
/// @return the @ref NQueensResult; @c solved is true iff @c queens is conflict-free.
NQueensResult min_conflicts_nqueens(int n, int max_steps, unsigned long long seed);

} // namespace datamunge::algorithms
