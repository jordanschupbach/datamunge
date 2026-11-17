#pragma once
/// @file simon.hpp
/// @brief Exact statevector Simon support enumeration and actual binary elimination.
#include <cstddef>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief SimonResult stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SimonResult value{};
/// std::print("{}\n",value.recovered_period); // 0
/// @endcode
struct SimonResult {
  std::size_t
      recovered_period{}; ///< Nullspace direction recovered from enumerated nonzero support rows.
  std::vector<std::size_t>
      observed_y; ///< Historical name: all nonzero support strings, not sampled observations.
};
/// @brief SimonOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SimonOptions value{};
/// std::print("{}\n",value.max_amplitudes); // 1048576
/// @endcode
struct SimonOptions {
  std::size_t max_amplitudes{
      1048576}; ///< Complete 2^(2n) complex amplitude count; zero disables policy.
  std::size_t max_history_values{
      1000000}; ///< Captured reduction records and all copied rows; zero disables.
};
/// @brief SimonReduction stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SimonReduction value{};
/// std::print("{}\n",value.column); // 0
/// @endcode
struct SimonReduction {
  int         column{};    ///< Actual pivot column in low-to-high bit order.
  std::size_t pivot_row{}; ///< Actual destination pivot row.
  std::vector<std::size_t>
      rows; ///< Actual complete binary row matrix after this elimination pivot.
};
/// @brief SimonRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SimonRun value{};
/// std::print("{}\n",value.rank); // 0
/// @endcode
struct SimonRun {
  SimonResult         result;           ///< Preserved support-based recovery result.
  std::vector<double> probabilities;    ///< Actual complete input marginal including y=0.
  std::size_t         amplitudes{};     ///< Actual full two-register amplitude count.
  std::size_t         hadamard_calls{}; ///< Actual native one-qubit Hadamard calls, 2n.
  std::size_t
      oracle_writes{}; ///< Actual nonzero-input oracle writes in this specialized known-zero-output simulation.
  std::size_t                 row_xors{}; ///< Actual binary elimination row XORs.
  int                         rank{};     ///< Actual rank of enumerated nonzero support rows.
  std::vector<SimonReduction> reduction;  ///< Optional actual elimination states.
};
/// @brief Simulate Simon probability support.
/// n>=1, nonzero period<2^n; canonical promised oracle f(x)=min(x,x XOR period). Uses the actual native two-register statevector: n Hadamards, specialized oracle on the known output-zero state, n Hadamards, full marginal, then elimination of every nonzero-probability nonzero y. This is exhaustive classical support enumeration, not random quantum measurements or an O(n)-query simulator. y=0 has positive probability and is retained in probabilities but omitted from historical observed_y. Threshold1e-9 is safe for allowed n<=15, enforced intrinsically along with shift capacity; default amplitude policy limits n<=10. State work O(n*4^n), storage O(4^n); elimination over O(2^n) support rows adds O(n^2*2^n) binary work. No global state or randomness; owned outputs.
/// @param n Input bit width1..15 subject to shift/amplitude limits.
/// @param period Nonzero promised hidden XOR period.
/// @param options Complete amplitude and optional history policies; zero disables policies.
/// @return Actual complete marginal, support recovery, gate/write counts, rank and optional reduction.
/// @throws std::invalid_argument Invalid width/zero or out-of-range period.
/// @throws std::length_error Amplitude/history/shift/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::simon_run(2,3);
/// std::print("{} {}\n",r.result.recovered_period,r.rank); // 3 1
/// @endcode
/// @par References
/// - D. R. Simon, On the Power of Quantum Computation, SIAM J. Comput.26(5), 1997, DOI 10.1137/S0097539796298637; conference1994.
/// @see simon()
SimonRun simon_run(int n, std::size_t period, const SimonOptions& options = {});
/// @brief Simulate Simon probability support with actual binary elimination states.
/// n>=1, nonzero period<2^n; canonical promised oracle f(x)=min(x,x XOR period). Uses the actual native two-register statevector: n Hadamards, specialized oracle on the known output-zero state, n Hadamards, full marginal, then elimination of every nonzero-probability nonzero y. This is exhaustive classical support enumeration, not random quantum measurements or an O(n)-query simulator. y=0 has positive probability and is retained in probabilities but omitted from historical observed_y. Threshold1e-9 is safe for allowed n<=15, enforced intrinsically along with shift capacity; default amplitude policy limits n<=10. State work O(n*4^n), storage O(4^n); elimination over O(2^n) support rows adds O(n^2*2^n) binary work. No global state or randomness; owned outputs.
/// @param n Input bit width1..15 subject to shift/amplitude limits.
/// @param period Nonzero promised hidden XOR period.
/// @param options Complete amplitude and optional history policies; zero disables policies.
/// @return Actual complete marginal, support recovery, gate/write counts, rank and optional reduction.
/// @throws std::invalid_argument Invalid width/zero or out-of-range period.
/// @throws std::length_error Amplitude/history/shift/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::simon_trace(2,3);
/// std::print("{} {}\n",r.result.recovered_period,r.rank); // 3 1
/// @endcode
/// @par References
/// - D. R. Simon, On the Power of Quantum Computation, SIAM J. Comput.26(5), 1997, DOI 10.1137/S0097539796298637; conference1994.
/// @see simon()
SimonRun simon_trace(int n, std::size_t period, const SimonOptions& options = {});
/// @brief Return the preserved support-based Simon result.
/// Historical observed_y denotes enumerated nonzero support, not random observations. Uses checked statevector producer and default policies.
/// @param n Input bits,1..10 under default amplitude bound.
/// @param period Nonzero period below2^n.
/// @return Owned period and all nonzero support strings.
/// @throws std::invalid_argument Invalid promised oracle parameters.
/// @throws std::length_error Default amplitude bound/capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::simon(4,11).recovered_period); // 11
/// @endcode
/// @see simon_run()
SimonResult simon(int n, std::size_t period);
} // namespace datamunge::algorithms
