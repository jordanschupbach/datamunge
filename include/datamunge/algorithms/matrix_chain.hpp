#pragma once

/// @file matrix_chain.hpp
/// @brief Canonical matrix chain API, producing histories and checked native contracts.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Minimum scalar multiplication cost and one first-minimum parenthesization.
/// @details Matrices are named A1 through An. Empty chains have an empty expression.
/// @par Example
/// @code
/// auto r=dm::algorithms::matrix_chain_order({10,30,5,60});
/// std::print("{} {}\n",r.cost,r.parenthesization); // 4500 ((A1A2)A3)
/// @endcode
/// @see matrix_chain_order
/// @par Reference
/// Cormen et al., Introduction to Algorithms, matrix-chain multiplication recurrence.
struct MatrixChainResult {
  long long   cost{0};          ///< Exact representable optimum in scalar multiplications.
  std::string parenthesization; ///< First strict minimum under increasing split order.
};

/// @brief Actual iteration counts from the canonical bottom-up producer.
/// @details Counts exclude allocation, formatting and reconstruction character copies.
/// @par Example
/// @code
/// auto a=dm::algorithms::matrix_chain_analyze({10,30,5,60});
/// std::print("{}\n",a.work.trials); // 4
/// @endcode
/// @see matrix_chain_analyze
/// @par Reference
/// Standard interval dynamic programming for matrix-chain ordering.
struct MatrixChainWork {
  std::size_t trials{0};       ///< Actual len/i/j/k split evaluations, n(n-1)(n+1)/6.
  std::size_t improvements{0}; ///< Initial selections and strict smaller candidate replacements.
  std::size_t cells{0};        ///< Committed nontrivial intervals, n(n-1)/2.
};

/// @brief Same-run result, complete one-based tables and actual work counts.
/// @details Costs saturate at LLONG_MAX+1, represented as uint64_t. This sentinel
/// marks an unrepresentable subproblem, never a fabricated finite exact cost.
/// Unused table entries are zero. Splits keep the first minimum, even if saturated.
/// @par Example
/// @code
/// auto a=dm::algorithms::matrix_chain_analyze({10,30,5,60});
/// std::print("{} {}\n",a.costs[1][3],a.splits[1][3]); // 4500 2
/// @endcode
/// @see matrix_chain_trace
/// @par Reference
/// Matrix-chain interval recurrence with nonnegative cost saturation.
struct MatrixChainAnalysis {
  MatrixChainResult                       result; ///< Exact full-chain optimum and its expression.
  std::vector<std::vector<std::uint64_t>> costs;  ///< One-based capped cost table.
  std::vector<std::vector<std::size_t>>   splits; ///< One-based selected split table.
  MatrixChainWork                         work;   ///< Actual producer operations.
};

/// @brief One actual split evaluation in increasing interval length/start/split order.
/// @details Capped values equal LLONG_MAX+1 on overflow; exact smaller costs remain exact.
/// @par Example
/// @code
/// auto t=dm::algorithms::matrix_chain_trace({2,3,4});
/// std::print("{} {}\n",t.trials[0].cost,t.trials[0].selected); // 24 true
/// @endcode
/// @see MatrixChainTrace
/// @par Reference
/// Standard matrix-chain dynamic-programming recurrence.
struct MatrixChainTrial {
  std::size_t   i{0};       ///< First matrix of the actual interval.
  std::size_t   j{0};       ///< Last matrix of the actual interval.
  std::size_t   k{0};       ///< Actual tested split between k and k+1.
  std::uint64_t left{0};    ///< Actual committed left subproblem cost, possibly capped.
  std::uint64_t right{0};   ///< Actual committed right subproblem cost, possibly capped.
  std::uint64_t product{0}; ///< Capped dimensions[i-1]*dimensions[k]*dimensions[j].
  std::uint64_t cost{0};    ///< Capped candidate sum.
  bool selected{false};     ///< Whether this candidate initialized or strictly improved the cell.
};

/// @brief One committed nontrivial interval after all of its split candidates.
/// @par Example
/// @code
/// auto t=dm::algorithms::matrix_chain_trace({2,3,4});
/// std::print("{}\n",t.commits[0].split); // 1
/// @endcode
/// @see MatrixChainTrace
/// @par Reference
/// Bottom-up interval DP dependency order.
struct MatrixChainCommit {
  std::size_t   i{0};     ///< Actual first matrix.
  std::size_t   j{0};     ///< Actual last matrix.
  std::size_t   split{0}; ///< Actual first minimizing split.
  std::uint64_t cost{0};  ///< Actual final capped cell cost.
};

/// @brief Owned input and actual trial/commit records from one complete producer execution.
/// @details Captured entries count dimension scalars, both complete returned square
/// tables, every trial and every commit. No row headers, record fields, output
/// expression bytes, container capacities or runtime storage are counted.
/// @par Example
/// @code
/// auto t=dm::algorithms::matrix_chain_trace({2,3,4});
/// std::print("{}\n",t.captured_entries); // 23
/// @endcode
/// @see matrix_chain_trace
/// @par Reference
/// Matrix-chain interval dynamic programming, captured at actual producing operations.
struct MatrixChainTrace {
  std::vector<int>               dimensions;          ///< Owned original positive dimensions.
  MatrixChainAnalysis            analysis;            ///< Actual same-run result, tables and work.
  std::vector<MatrixChainTrial>  trials;              ///< Actual candidate evaluations.
  std::vector<MatrixChainCommit> commits;             ///< Actual final cell writes.
  std::size_t                    captured_entries{0}; ///< Complete logical entry count.
};

/// @brief Find the minimum-cost parenthesization by one bottom-up DP execution.
/// @details Positive dimensions describe Ai as dims[i-1] by dims[i]. Zero or one
/// dimension denotes an empty chain; all supplied dimensions are validated first.
/// Increasing split order and strict replacement select the first minimum.
/// Time O(n^3), table space O(n^2); an explicit stack builds the expression
/// in output-linear character work without recursive reconstruction. It plans arithmetic and does not multiply matrices.
/// Irrelevant overflowing subchains saturate, allowing a representable full optimum.
/// @param dims Positive int dimensions, with n+1 entries for n matrices.
/// @return Exact minimum cost and first minimizing expression; empty for no matrices.
/// @throws std::invalid_argument If any dimension is nonpositive.
/// @throws std::length_error If complete table or iteration geometry overflows size_t.
/// @throws std::overflow_error If the optimal full-chain cost exceeds LLONG_MAX.
/// @par Example
/// @code
/// auto r=dm::algorithms::matrix_chain_order({10,30,5,60});
/// std::print("{}\n",r.cost); // 4500
/// @endcode
/// @see matrix_chain_analyze, matrix_chain_trace
/// @par Reference
/// Cormen et al., Introduction to Algorithms, matrix-chain multiplication.
MatrixChainResult matrix_chain_order(const std::vector<int>& dims);

/// @brief Return the canonical producer's complete tables and work as well as its result.
/// @param dims Positive dimension sequence, with the same empty-chain convention as matrix_chain_order.
/// @return Same-run exact optimum, capped cost table, selected splits and actual counts.
/// @throws std::invalid_argument On nonpositive dimensions.
/// @throws std::length_error On unrepresentable geometry.
/// @throws std::overflow_error On an unrepresentable full-chain optimum.
/// @par Example
/// @code
/// auto a=dm::algorithms::matrix_chain_analyze({2,3,4});
/// std::print("{} {}\n",a.result.cost,a.work.cells); // 24 1
/// @endcode
/// @see matrix_chain_order, matrix_chain_trace
/// @par Reference
/// Standard bottom-up matrix-chain recurrence.
MatrixChainAnalysis matrix_chain_analyze(const std::vector<int>& dims);

/// @brief Capture every actual split trial and cell commit from the canonical producer.
/// @details Complete admission occurs before table allocation. For d supplied dimensions,
/// n=max(d-1,0), the quota is d+2*(n+1)^2+n*(n-1)*(n+1)/6+n*(n-1)/2.
/// Empty input still has two one-cell tables, so its logical count is two.
/// Capture is O(n^3) storage; it never changes arithmetic or tie decisions.
/// @param dims Positive dimension sequence.
/// @param max_entries Complete logical capture limit, default one million; zero disables it.
/// @return Owned inputs, actual tables/trials/commits and same-run optimum.
/// @throws std::invalid_argument On nonpositive dimensions.
/// @throws std::length_error Before allocation if geometry or the full logical count exceeds its limit.
/// @throws std::overflow_error If the optimal full-chain cost exceeds LLONG_MAX.
/// @par Example
/// @code
/// auto t=dm::algorithms::matrix_chain_trace({2,3,4},23);
/// std::print("{}\n",t.trials.size()); // 1
/// @endcode
/// @see matrix_chain_analyze, MatrixChainTrace
/// @par Reference
/// Matrix-chain recurrence, retaining actual producing evaluations and writes.
MatrixChainTrace matrix_chain_trace(
    const std::vector<int>& dims,
    std::size_t             max_entries = 1000000);

/// @brief Cost of the fully left-associated product under the same checked dimension contract.
/// @param dims Positive dimensions, empty chains allowed.
/// @return Exact cost of (((A1 A2) A3)...), zero for fewer than two matrices.
/// @throws std::invalid_argument On nonpositive dimensions.
/// @throws std::overflow_error If this chosen order's cost exceeds LLONG_MAX.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::matrix_chain_left_cost({10,30,5,60})); // 4500
/// @endcode
/// @see matrix_chain_right_cost, matrix_chain_order
/// @par Reference
/// Scalar multiplication count for a p-by-q times q-by-r product: p*q*r.
long long matrix_chain_left_cost(const std::vector<int>& dims);

/// @brief Cost of the fully right-associated product under the same checked dimension contract.
/// @param dims Positive dimensions, empty chains allowed.
/// @return Exact cost of (A1(A2(...An))), zero for fewer than two matrices.
/// @throws std::invalid_argument On nonpositive dimensions.
/// @throws std::overflow_error If this chosen order's cost exceeds LLONG_MAX.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::matrix_chain_right_cost({10,30,5,60})); // 27000
/// @endcode
/// @see matrix_chain_left_cost, matrix_chain_order
/// @par Reference
/// Scalar multiplication count for a p-by-q times q-by-r product: p*q*r.
long long matrix_chain_right_cost(const std::vector<int>& dims);

/// @brief Original six-matrix CLRS fixture, retained as native reusable input.
/// @return Dimensions 30,35,15,5,10,20,25.
/// @par Example
/// @code
/// auto r=dm::algorithms::matrix_chain_order(dm::algorithms::matrix_chain_demo_dimensions());
/// std::print("{}\n",r.cost); // 15125
/// @endcode
/// @see matrix_chain_order
/// @par Reference
/// Cormen et al., Introduction to Algorithms, matrix-chain worked example.
std::vector<int> matrix_chain_demo_dimensions();

} // namespace datamunge::algorithms
