#pragma once
/// @file min_conflicts.hpp
/// @brief Greedy-random n-queens initialization and actual min-conflicts repairs.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned final assignment and honest terminal status of one capped search.
/// One queen per row; columns need not be distinct on an unsuccessful run.
/// @par Example
/// @code
/// auto r=dm::algorithms::min_conflicts_nqueens(1,0,42);
/// std::print("{} {}\n",r.solved,r.steps); // true 0
/// @endcode
/// @see min_conflicts_nqueens()
struct NQueensResult {
  std::vector<int> queens;        ///< Final queen column for each row, owned.
  bool             solved{false}; ///< True exactly when no attacking pairs remain.
  int              steps{0};      ///< Actual repair moves, excluding initialization.
};
/// @brief Actual initialization choice or repair, with exact candidate costs and RNG words.
/// Initialization has column_before=-1 and no conflicted list/row draw. Repair
/// costs are measured after temporary queen removal, including all n columns.
/// Conflicts count unordered attacking pairs; a pair cannot share two constraints.
/// @par Example
/// @code
/// auto t=dm::algorithms::min_conflicts_nqueens_trace(1,0,42);
/// std::print("{}\n",t.moves.front().column_after); // 0
/// @endcode
/// @see NQueensTrace
struct NQueensMove {
  bool             initialization{false}; ///< True for one greedy-random initial placement.
  int              row{0};                ///< Actually selected row.
  int              column_before{-1};     ///< Previous column, or−1 before initial placement.
  int              column_after{0};       ///< Actual selected column.
  std::vector<int> conflicted;            ///< Actual sorted conflicting rows before a repair.
  std::vector<int> costs;      ///< Actual conflict cost for every column, in column order.
  std::vector<int> candidates; ///< Actual sorted minimum-cost tied columns.
  std::uint64_t    row_draw{
      0}; ///< Actual MT19937-64 repair-row word; zero placeholder on initialization.
  std::uint64_t column_draw{0};      ///< Actual MT19937-64 tied-column word.
  long long     conflicts_before{0}; ///< Actual pair count before placement/repair.
  long long     conflicts_after{0};  ///< Actual pair count after placement/repair.
};
/// @brief Owned inputs, actual initialization/repair history and final assignment.
/// Capture units are three scalar inputs, final n columns, and per move one
/// record plus every stored conflicting-row ID, cost and tied-column ID.
/// Driver occupancy arrays and temporary candidates are not capture units.
/// @par Example
/// @code
/// auto t=dm::algorithms::min_conflicts_nqueens_trace(1,0,42);
/// std::print("{} {}\n",t.moves.size(),t.captured_entries); // 1 7
/// @endcode
/// @see min_conflicts_nqueens_trace()
struct NQueensTrace {
  int                      n{0};                ///< Original board size.
  int                      max_steps{0};        ///< Original repair cap.
  unsigned long long       seed{0};             ///< Original MT19937-64 seed.
  NQueensResult            result;              ///< Actual terminal assignment/status.
  std::vector<NQueensMove> moves;               ///< Actual initialization first, then repair moves.
  std::size_t              captured_entries{0}; ///< Admitted capture units.
};
/// @brief Search for nonattacking queens using actual minimum-conflict repairs.
/// Initialize each row in increasing order at a minimum-conflict column, using
/// one MT19937-64 word modulo tie count. Each repair draws a conflicting row,
/// removes its queen, scans every column, then draws among minimum-cost ties.
/// Modulo sampling has finite-word bias. Seed and legacy choices are preserved.
/// Always checks terminal status after the last allowed move, including zero.
/// n=0 is solved. No success guarantee or restart. O(n^2+n*max_steps) work,
/// O(n) driver storage, so no size-independent runtime claim is made.
/// @param n Nonnegative board size, at most INT_MAX/3 for safe occupancy sums.
/// @param max_steps Nonnegative repair cap; zero still initializes and checks.
/// @param seed Arbitrary RNG seed, including zero.
/// @return Owned actual final assignment and verified status.
/// @throws std::invalid_argument If n or max_steps is negative.
/// @throws std::length_error If occupancy/count geometry is unrepresentable.
/// @par Example
/// @code
/// auto r=dm::algorithms::min_conflicts_nqueens(1,0,42);
/// std::print("{}\n",r.solved); // true
/// @endcode
/// @par References
/// - S. Minton et al. (1992), Minimizing conflicts: a heuristic repair method for constraint satisfaction and scheduling problems, Artificial Intelligence58,161–205, doi:10.1016/0004-3702(92)90007-K.
/// @see min_conflicts_nqueens_trace()
NQueensResult min_conflicts_nqueens(int n, int max_steps, unsigned long long seed);
/// @brief Capture actual decisions of the same n-queens search producer.
/// No replay or changed RNG scheduling. Actual capture admission checks units
/// before appending; no partial trace is returned on quota exhaustion. Trace
/// storage is proportional to captured entries; driver work follows ordinary search.
/// @param n Nonnegative board size with the ordinary count guard.
/// @param max_steps Nonnegative repair cap.
/// @param seed Arbitrary MT19937-64 seed.
/// @param max_entries Capture quota, default1000000; zero disables.
/// @return Owned actual inputs, history and terminal result.
/// @throws std::invalid_argument If size or step cap is negative.
/// @throws std::length_error If geometry or capture quota exceeds its limit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::min_conflicts_nqueens_trace(1,0,42);
/// std::print("{}\n",t.result.solved); // true
/// @endcode
/// @see NQueensTrace, min_conflicts_nqueens()
NQueensTrace min_conflicts_nqueens_trace(
    int                n,
    int                max_steps,
    unsigned long long seed,
    std::size_t        max_entries = 1000000);
} // namespace datamunge::algorithms
