#pragma once
/// @file cannon.hpp
/// @brief Checked sequential Cannon scheduling for scalar or square-block matrix products.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Counts of scalar arithmetic and buffer assignments in this Cannon implementation.
/// A multiply-add counts one multiply-and-accumulate expression, not two measured
/// machine instructions. Copies count double assignments, including the final closing
/// shift and self-copies when the mesh has one cell. These are not network messages,
/// bytes, timings, allocations or hardware performance counters.
/// @par Example
/// @code
/// const auto w=dm::algorithms::cannon_work(4,2);
/// std::print("{} {}\n",w.block_updates,w.multiply_adds); // 8 64
/// @endcode
/// @see cannon_work(), CannonTrace
struct CannonWork {
  std::size_t block_updates=0; ///< Mesh-cell block products, q cubed for nonempty input.
  std::size_t multiply_adds=0; ///< Scalar multiply-accumulate expressions, n cubed.
  std::size_t skew_copies=0; ///< Initial direct skew assignments, 2*n*n.
  std::size_t shift_copies=0; ///< Post-round double assignments, 2*q*n*n.
};
/// @brief One actual round's source-block ownership and accumulated global output.
/// Arrays a_sources/b_sources have q*q row-major cells. Each entry identifies the
/// original input block by row*q+column, not a scalar offset. partial has n*n
/// row-major values after this round's block products, before the next shifts.
/// Values/indices are owned and independent of inputs and later rounds.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cannon_matmul_trace({1,2,3,4},{5,6,7,8},2,2);
/// std::print("{}\n",t.rounds[0].a_sources); // [0, 1, 3, 2]
/// @endcode
/// @see CannonTrace
struct CannonRound {
  std::size_t step=0; ///< Zero-based multiplication round; equals round vector position.
  std::vector<std::size_t> a_sources; ///< Actual original A block ids held by each cell.
  std::vector<std::size_t> b_sources; ///< Actual original B block ids held by each cell.
  std::vector<double> partial; ///< Owned global C after this round's accumulation.
};
/// @brief Owned product, actual carried source ids, snapshots and loop work counters.
/// The shared ordinary/traced kernel records without replaying multiplication or
/// deriving source ids from a closed-form round formula. Source ids travel through
/// the same source-cell selection as numeric shift buffers. Inputs are borrowed only
/// during the call and never mutated; input values are not copied into this result.
/// With q rounds, snapshots use O(q*(n*n+q*q)) storage. Empty input has no rounds.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cannon_matmul_trace({1,2,3,4},{5,6,7,8},2,2);
/// std::print("{} {}\n",t.output,t.rounds.size()); // [19, 22, 43, 50] 2
/// @endcode
/// @see cannon_matmul_trace(), CannonRound
struct CannonTrace {
  int dimension=0; ///< Matrix side n; zero for the empty product.
  int mesh_dimension=0; ///< Mesh side q; zero only for the empty product.
  int block_dimension=0; ///< Block side n/q, or zero when empty.
  std::vector<double> output; ///< Owned row-major n-by-n product.
  std::vector<CannonRound> rounds; ///< Actual round-order ownership and partial products.
  std::vector<std::size_t> final_a_sources; ///< Actual A ownership after the last closing shift.
  std::vector<std::size_t> final_b_sources; ///< Actual B ownership after the last closing shift.
  CannonWork work; ///< Actual increments from multiplication and numeric-copy loops.
};
/// @brief Predict exact loop counts without allocating matrices or running the kernel.
/// The empty shape is n=q=0. Otherwise n>0 and q is a positive divisor of n.
/// Counts include the final closing shift and one-cell self-copies; they do not
/// model initial-skew network hops, messages, latency or execution time. O(1) work
/// and storage, no shared mutable state. Shape/count checks precede kernel allocation.
/// @param n Nonnegative square matrix side.
/// @param mesh_dimension Mesh side q, positive divisor of nonzero n; zero for empty n.
/// @return Checked count formulas for this sequential schedule.
/// @throws std::invalid_argument Negative dimensions or invalid/divisibility/empty shape.
/// @throws std::length_error A size/count product exceeds size_t.
/// @par Example
/// @code
/// const auto w=dm::algorithms::cannon_work(4,2);
/// std::print("{} {}\n",w.skew_copies,w.shift_copies); // 32 64
/// @endcode
/// @par References
/// - J. Demmel, CS267 lecture notes, Cannon's algorithm on a 2D mesh.
///   https://people.eecs.berkeley.edu/~demmel/cs267/lecture11/lecture11.html
/// @see cannon_block_matmul(), CannonWork
CannonWork cannon_work(int n,int mesh_dimension);
/// @brief Multiply square row-major matrices using the original scalar-cell Cannon API.
/// Equivalent to cannon_block_matmul(A,B,n,n): n-by-n logical cells, one scalar
/// per cell, n rounds and a closing shift after every round including the last.
/// The cyclic summation order is retained. Floating-point results can differ from
/// ascending-k multiplication; NaN/infinity and overflow propagate through ordinary
/// double arithmetic. This is a sequential simulation, with no MPI/network execution.
/// Inputs are checked for exact shape, borrowed and unchanged; output owns storage.
/// Empty n=0 requires empty inputs and returns empty. O(n^3) work, O(n^2) extra
/// storage excluding inputs; ordinary calls allocate no ownership/snapshot records.
/// @param A Row-major n-by-n left matrix, exactly n*n entries.
/// @param B Row-major n-by-n right matrix, exactly n*n entries; may alias A.
/// @param n Nonnegative matrix side; the logical mesh side is also n.
/// @return Owned row-major product, subject to floating-point rounding.
/// @throws std::invalid_argument Negative n or input lengths do not match n*n.
/// @throws std::length_error Shape or exact loop-count products exceed size_t.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cannon_matmul({1,2,3,4},{5,6,7,8},2);
/// std::print("{}\n",c); // [19, 22, 43, 50]
/// @endcode
/// @par References
/// - L. E. Cannon (1969), A cellular computer to implement the Kalman filter algorithm,
///   PhD thesis, Montana State University.
///   https://scholarworks.montana.edu/server/api/core/bitstreams/643669d9-055d-4345-a467-41ad612599c8/content
/// @see cannon_block_matmul(), cannon_matmul_trace()
std::vector<double> cannon_matmul(const std::vector<double>& A,const std::vector<double>& B,int n);
/// @brief Multiply with square blocks on a q-by-q logical toroidal processor mesh.
/// Block side b=n/q. Initial direct copies skew A block row i left by i and B
/// block column j up by j. Each of q rounds accumulates a block product, then
/// copies A left and B up one block, including a redundant final closing shift.
/// Local inner coordinates run in ascending k order; block order is cyclic and
/// can change floating-point rounding. NaN/infinity propagate without rejection.
/// No actual processors, messages or parallel execution are created. O(n^3+q*n^2)
/// sequential work, O(n^2) temporary storage. Inputs are borrowed, never mutated;
/// ordinary calls store no source ids or snapshots. Empty shape is n=q=0 only.
/// @param A Row-major n-by-n left matrix, exactly n*n entries.
/// @param B Row-major n-by-n right matrix, exactly n*n entries; may alias A.
/// @param n Nonnegative square matrix side.
/// @param mesh_dimension Mesh side q, positive divisor of nonzero n; zero when n is zero.
/// @param max_block_updates Whole-call q^3 budget; zero (default) means unlimited.
/// @return Owned row-major product; no partial result is returned on exception.
/// @throws std::invalid_argument Invalid dimensions/divisibility or mismatched input lengths.
/// @throws std::length_error A shape/count product overflows size_t or the budget is exceeded.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cannon_block_matmul({1,2,3,4},{5,6,7,8},2,1);
/// std::print("{}\n",c); // [19, 22, 43, 50]
/// @endcode
/// @par References
/// - J. Demmel, CS267, Cannon's algorithm on a 2D mesh (same block schedule).
///   https://people.eecs.berkeley.edu/~demmel/cs267/lecture11/lecture11.html
/// @see cannon_matmul(), cannon_work(), cannon_matmul_trace()
std::vector<double> cannon_block_matmul(const std::vector<double>& A,const std::vector<double>& B,
    int n,int mesh_dimension,std::size_t max_block_updates=0);
/// @brief Execute Cannon once, retaining actual round ownership, partials and work.
/// Uses the same numeric kernel as cannon_block_matmul(), with carried original
/// block ids and owned C snapshots. No multiplication or shifting is replayed to
/// construct the trace. Final ownership follows the last closing shift and equals
/// initial skew ownership, even though numerical C is complete before that shift.
/// Tracing adds O(q*(n*n+q*q)) storage; shape, arithmetic, mutation and empty rules
/// match cannon_block_matmul(). Snapshot allocations can fail without changing inputs.
/// @param A Borrowed row-major n-by-n left matrix, exactly n*n entries.
/// @param B Borrowed row-major n-by-n right matrix, exactly n*n entries; may alias A.
/// @param n Nonnegative square matrix side.
/// @param mesh_dimension Mesh side q, positive divisor of nonzero n; zero when empty.
/// @param max_block_updates Whole-call q^3 budget; zero (default) is unlimited; checked before buffers.
/// @return Owned complete trace and product; no partial result on any exception.
/// @throws std::invalid_argument Invalid dimensions/divisibility or mismatched input lengths.
/// @throws std::length_error A shape/count product overflows size_t or the budget is exceeded.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cannon_matmul_trace({1,2,3,4},{5,6,7,8},2,2,8);
/// std::print("{} {}\n",t.work.multiply_adds,t.work.shift_copies); // 8 16
/// @endcode
/// @par References
/// - J. Demmel, CS267, Cannon's algorithm on a 2D mesh.
///   https://people.eecs.berkeley.edu/~demmel/cs267/lecture11/lecture11.html
/// @see cannon_block_matmul(), CannonTrace
CannonTrace cannon_matmul_trace(const std::vector<double>& A,const std::vector<double>& B,
    int n,int mesh_dimension,std::size_t max_block_updates=0);
} // namespace datamunge::algorithms
