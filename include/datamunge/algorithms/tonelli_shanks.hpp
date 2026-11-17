#pragma once
/// @file tonelli_shanks.hpp
/// @brief Checked full-uint64 prime square roots with actual two-adic refinements.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own ModSqrt values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ModSqrt r{};
/// (void)r;
/// @endcode
struct ModSqrt {
  bool          exists{}; ///< Actual mathematical residue verdict for the validated prime modulus.
  std::uint64_t root{};   ///< Actual smaller root min(r,p-r); zero on no root or zero residue.
};
/// @brief Own TonelliOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TonelliOptions r{};
/// (void)r;
/// @endcode
struct TonelliOptions {
  std::size_t max_nonresidue_tests{
      1000000}; ///< Actual sequential auxiliary-candidate tests; zero disables.
  std::size_t max_products{
      100000000}; ///< Actual root-kernel modular products, excluding separate primality validation; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual candidate/refinement records plus copied tested-square entries; zero disables.
};
/// @brief Own TonelliCandidate values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TonelliCandidate r{};
/// (void)r;
/// @endcode
struct TonelliCandidate {
  std::uint64_t candidate{}; ///< Actual tested auxiliary z.
  std::uint64_t euler{};     ///< Actual z^((p-1)/2) mod p.
};
/// @brief Own TonelliStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TonelliStep r{};
/// (void)r;
/// @endcode
struct TonelliStep {
  std::uint64_t M_before{}; ///< Actual remaining two-adic exponent before this refinement.
  std::uint64_t i{};        ///< Actual first exponent with t^(2^i)=1, strictly below M.
  std::vector<std::uint64_t> squares;   ///< Actual initial t and each searched square through1.
  std::uint64_t              b{};       ///< Actual c^(2^(M-i-1)) correction.
  std::uint64_t              c_after{}; ///< Actual b squared modulo p.
  std::uint64_t              t_after{}; ///< Actual t*c_after modulo p.
  std::uint64_t root_after{}; ///< Actual R*b modulo p before smaller-root canonicalization.
};
/// @brief Own TonelliRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TonelliRun r{};
/// (void)r;
/// @endcode
struct TonelliRun {
  ModSqrt       result;     ///< Actual smaller-root verdict.
  std::uint64_t reduced{};  ///< Actual n mod validated p.
  std::uint64_t odd_part{}; ///< Actual odd Q in p-1=Q*2^S for general branch, zero on shortcuts.
  std::uint64_t twos{};     ///< Actual initial S, zero on shortcuts.
  std::uint64_t euler{}; ///< Actual residue test n^((p-1)/2), zero when zero/p2 shortcuts avoid it.
  std::uint64_t nonresidue{};       ///< Actual selected z on general branch, zero on shortcuts.
  std::size_t   nonresidue_tests{}; ///< Actual sequential z tests.
  std::size_t
      products{}; ///< Actual root-kernel modular products, excluding deterministic primality check.
  std::size_t                   refinements{}; ///< Actual completed strictly decreasing-M updates.
  std::vector<TonelliCandidate> candidates;    ///< Optional actual auxiliary tests.
  std::vector<TonelliStep>      steps; ///< Optional actual refinements and searched square chains.
};
/// @brief Compute prime-field square roots through actual checked Tonelli-Shanks states.
/// First validate p as prime over the complete uint64 domain with the existing deterministic twelve-base primality checker, even for n0. Then reduce n modulo p. Handle n0 and p2 directly, reject nonresidues by Euler criterion, and use n^((p+1)/4) for p mod4=3. Otherwise p-1=Q*2^S, choose the first sequential nonresidue z>=2, initialize c=z^Q,t=n^Q,R=n^((Q+1)/2),M=S. Find least i with t^(2^i)=1, set b=c^(2^(M-i-1)), then R*=b,t*=b^2,c=b^2,M=i. M strictly decreases. Private unsigned128 products avoid overflow; return the smaller complementary root. Sequential auxiliary search has its own potentially variable cost; no unconditional O(log^2 p) claim includes this search or full primality validation. Kernel product counts exclude separate validation. No constant-time/security guarantee, owned results and no shared state.
/// @param n Complete uint64 radicand, reduced after validating modulus.
/// @param p Complete prime uint64 modulus, including2.
/// @param options Actual auxiliary/product/capture policies; zero disables budgets.
/// @return Owned actual smaller-root verdict, selected work and optional candidate/refinement records.
/// @throws std::invalid_argument Modulus is not prime, including0 and1.
/// @throws std::length_error Actual auxiliary/product/history or counter capacity exceeded.
/// @throws std::logic_error Validated prime-field refinement invariant unexpectedly fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::tonelli_shanks_run(10,13);
/// std::print("{} {}\n",r.result.exists,r.result.root); // true 6
/// @endcode
/// @par References
/// - Shanks, Five Number-theoretic Algorithms, Proceedings of the Second Manitoba Conference on Numerical Mathematics (1973),51-70.
/// @see tonelli_shanks()
TonelliRun tonelli_shanks_run(std::uint64_t n, std::uint64_t p, const TonelliOptions& options = {});
/// @brief Compute prime-field square roots through actual checked Tonelli-Shanks states.
/// First validate p as prime over the complete uint64 domain with the existing deterministic twelve-base primality checker, even for n0. Then reduce n modulo p. Handle n0 and p2 directly, reject nonresidues by Euler criterion, and use n^((p+1)/4) for p mod4=3. Otherwise p-1=Q*2^S, choose the first sequential nonresidue z>=2, initialize c=z^Q,t=n^Q,R=n^((Q+1)/2),M=S. Find least i with t^(2^i)=1, set b=c^(2^(M-i-1)), then R*=b,t*=b^2,c=b^2,M=i. M strictly decreases. Private unsigned128 products avoid overflow; return the smaller complementary root. Sequential auxiliary search has its own potentially variable cost; no unconditional O(log^2 p) claim includes this search or full primality validation. Kernel product counts exclude separate validation. No constant-time/security guarantee, owned results and no shared state.
/// @param n Complete uint64 radicand, reduced after validating modulus.
/// @param p Complete prime uint64 modulus, including2.
/// @param options Actual auxiliary/product/capture policies; zero disables budgets.
/// @return Owned actual smaller-root verdict, selected work and optional candidate/refinement records.
/// @throws std::invalid_argument Modulus is not prime, including0 and1.
/// @throws std::length_error Actual auxiliary/product/history or counter capacity exceeded.
/// @throws std::logic_error Validated prime-field refinement invariant unexpectedly fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::tonelli_shanks_trace(10,13);
/// std::print("{} {}\n",r.result.exists,r.result.root); // true 6
/// @endcode
/// @par References
/// - Shanks, Five Number-theoretic Algorithms, Proceedings of the Second Manitoba Conference on Numerical Mathematics (1973),51-70.
/// @see tonelli_shanks()
TonelliRun
tonelli_shanks_trace(std::uint64_t n, std::uint64_t p, const TonelliOptions& options = {});
/// @brief Preserve checked smaller prime-field modular square roots.
/// First validate p as prime over the complete uint64 domain with the existing deterministic twelve-base primality checker, even for n0. Then reduce n modulo p. Handle n0 and p2 directly, reject nonresidues by Euler criterion, and use n^((p+1)/4) for p mod4=3. Otherwise p-1=Q*2^S, choose the first sequential nonresidue z>=2, initialize c=z^Q,t=n^Q,R=n^((Q+1)/2),M=S. Find least i with t^(2^i)=1, set b=c^(2^(M-i-1)), then R*=b,t*=b^2,c=b^2,M=i. M strictly decreases. Private unsigned128 products avoid overflow; return the smaller complementary root. Sequential auxiliary search has its own potentially variable cost; no unconditional O(log^2 p) claim includes this search or full primality validation. Kernel product counts exclude separate validation. No constant-time/security guarantee, owned results and no shared state.
/// @param n Complete uint64 radicand, reduced after validating modulus.
/// @param p Complete prime uint64 modulus, including2.
/// @return Owned actual smaller-root residue verdict.
/// @throws std::invalid_argument Modulus not prime.
/// @throws std::length_error Default actual work/capture or counter capacity exceeded.
/// @throws std::logic_error Unexpected prime-field invariant failure.
/// @par Example
/// @code
/// auto r=dm::algorithms::tonelli_shanks(10,13);
/// std::print("{}\n",r.root); // 6
/// @endcode
/// @see tonelli_shanks_run()
ModSqrt tonelli_shanks(std::uint64_t n, std::uint64_t p);
} // namespace datamunge::algorithms
