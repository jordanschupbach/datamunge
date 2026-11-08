#pragma once
/// @file chakravala.hpp
/// @brief Checked signed-128 Pell solutions and actual cyclic Chakravala multiplier records.
#include <datamunge/algorithms/integer_arithmetic.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned positive solution of x*x-N*y*y=1, retaining the legacy field layout.
/// A successful solver returns the fundamental positive solution. Default or
/// square-input results are {0,0,false}; arithmetic/budget failures throw rather
/// than returning a partial pair. Manually constructed records are not validated.
/// Fields use the compiler's signed __int128 extension, not arbitrary precision.
/// @par Example
/// @code
/// const auto p=dm::algorithms::chakravala(13);
/// std::print("{} {} {}\n",dm::algorithms::int128_text(p.x),dm::algorithms::int128_text(p.y),p.found); // 649 180 true
/// @endcode
/// @see chakravala(), pell_is_solution()
struct PellSolution {
  __int128 x{0}; ///< Positive fundamental x on success, zero for square-input solver results.
  __int128 y{0}; ///< Positive fundamental y on success, zero for square-input solver results.
  bool found{false}; ///< True only on successful return; false for default or square-input results.
};
/// @brief Owned auxiliary triple satisfying a*a-N*b*b=k in an executed cyclic solve.
/// The kernel keeps positive a,b and a signed nonzero k for nonsquare N.
/// Default construction is only a record, with no invariant validation.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chakravala_trace(13);
/// std::print("{} {} {}\n",dm::algorithms::int128_text(t.initial.a),dm::algorithms::int128_text(t.initial.b),dm::algorithms::int128_text(t.initial.k)); // 3 1 -4
/// @endcode
/// @see ChakravalaStep, chakravala_trace()
struct ChakravalaState {
  __int128 a{0}; ///< Positive x-like coordinate, initially floor(sqrt(N)).
  __int128 b{0}; ///< Positive y-like coordinate, initially one.
  __int128 k{0}; ///< Signed auxiliary norm a*a-N*b*b; zero for square-input initial records.
};
/// @brief Actual congruence class, bracketing candidates and minimum-norm multiplier.
/// Admissible positive m satisfy m=residue mod modulus, with modulus=abs(k).
/// below is the largest such m<=floor(sqrt(N)), or zero when none exists;
/// above is the smallest such m>floor(sqrt(N)). Errors are abs(m*m-N),
/// computed exactly in unsigned 128-bit arithmetic. A missing below has error
/// zero but is never eligible. Equal errors select the smaller positive m.
/// @par Example
/// @code
/// const auto c=dm::algorithms::chakravala_trace(13).steps[0].choice;
/// std::print("{} {} {}\n",c.below,c.above,c.chosen); // 1 5 1
/// @endcode
/// @par References
/// - A. Bauval (2014), An elementary proof of the halting property for Chakravala
///   algorithm, Section 2, https://arxiv.org/abs/1406.6809
/// @see ChakravalaStep
struct ChakravalaChoice {
  std::uint64_t modulus{0}; ///< Positive abs(k), not a field modulus.
  std::uint64_t residue{0}; ///< Least nonnegative solution of b*m=-a mod abs(k).
  std::uint64_t below{0}; ///< Lower positive candidate, or zero when absent.
  std::uint64_t above{0}; ///< Upper positive candidate, always present in actual records.
  unsigned __int128 below_error{0}; ///< abs(below*below-N), or zero when below is absent.
  unsigned __int128 above_error{0}; ///< abs(above*above-N), exactly widened before multiplication.
  std::uint64_t chosen{0}; ///< Actual minimum-error positive multiplier, smaller on ties.
};
/// @brief Owned before, multiplier decision and after records from one actual cyclic update.
/// The ordinary and traced solver share arithmetic. Records are captured during
/// execution; neither multiplier choices nor coordinate updates are replayed.
/// @par Example
/// @code
/// const auto s=dm::algorithms::chakravala_trace(2).steps[0];
/// std::print("{} {}\n",s.choice.chosen,dm::algorithms::int128_text(s.after.a)); // 1 3
/// @endcode
/// @see chakravala_trace(), ChakravalaChoice
struct ChakravalaStep {
  ChakravalaState before; ///< Actual triple immediately before this update.
  ChakravalaChoice choice; ///< Actual selected congruence and candidates.
  ChakravalaState after; ///< Actual reduced triple immediately after this update.
};
/// @brief Owned initial triple, chronological cyclic steps and final Pell solution.
/// Square inputs have initial={floor(sqrt(N)),1,0}, no steps and found=false.
/// Nonsquares begin at the legacy floor-root triple and stop at k=+1, without
/// the historical shortcuts for norms -1, +/-2 or +/-4. Their path therefore
/// differs from variants starting at the nearest square or stopping at |k|=1.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chakravala_trace(13);
/// std::print("{} {}\n",t.steps.size(),t.solution.found); // 6 true
/// @endcode
/// @see chakravala_trace(), PellSolution
struct ChakravalaTrace {
  std::uint64_t N{0}; ///< Original nonnegative coefficient, owned by value.
  ChakravalaState initial; ///< Actual floor-root setup, including square-input setup.
  std::vector<ChakravalaStep> steps; ///< Every actual cyclic update, in chronological order.
  PellSolution solution; ///< Completed positive Pell pair, or the square-input sentinel.
};
/// @brief Solve positive Pell's equation using checked cyclic Chakravala arithmetic.
/// Preserves the original signature and result layout. Every uint64 N is
/// accepted; squares (including zero/one) return {0,0,false}. Nonsquares start
/// with a=floor(sqrt(N)), b=1 and minimise abs(m*m-N) over positive admissible
/// m, choosing the smaller m on ties. Corrects the old k=-1 floor-only choice.
/// Uses no floating arithmetic. Quotient/remainder decomposition avoids forming
/// oversized composition numerators when the reduced coordinates still fit.
/// Coordinates are limited to 2^127-1; oversized reduced triples throw. At most
/// 100000 cyclic updates, with no partial result on failure. Ordinary calls keep
/// no provenance records; O(S log N) fixed-word arithmetic for S updates and
/// constant auxiliary storage, not an arbitrary-precision bit-complexity bound.
/// Independent calls share no mutable state and inputs are passed by value.
/// @param N Nonnegative uint64 coefficient; nonsquares are at least two.
/// @return Fundamental positive pair on success, square-input sentinel otherwise.
/// @throws std::overflow_error If a reduced coordinate exceeds signed 128-bit range or an auxiliary value exceeds uint64.
/// @throws std::length_error If 100000 updates do not reach k=1.
/// @throws std::logic_error If an internal exact-divisibility or coprimality invariant fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::chakravala(61);
/// std::print("{}\n",dm::algorithms::int128_text(p.x)); // 1766319049
/// @endcode
/// @par References
/// - A. Bauval (2014), An elementary proof of the halting property for Chakravala
///   algorithm, https://arxiv.org/abs/1406.6809; minimum-norm cyclic steps.
/// - A. Sathaye (2018), Note on Chakravala Method of Bhaskara II,
///   https://www.ms.uky.edu/~sohum/ma330/files/chakravala.pdf; composition identity.
///   This implementation starts at the floor root and continues to positive norm one.
/// @see chakravala_trace(), pell_is_solution(), integer_sqrt_u64()
PellSolution chakravala(std::uint64_t N);
/// @brief Run the same checked solver while retaining every actual decision and reduced triple.
/// Same square, tie, overflow and fixed-width semantics as chakravala(). The
/// explicit budget counts complete cyclic updates; zero allows no updates,
/// hence only square inputs return with a zero budget. An exceeded budget or
/// allocation failure returns no partial trace. No caller-owned state is changed.
/// O(S) owned trace storage in addition to the ordinary fixed-word solver work.
/// @param N Nonnegative uint64 coefficient, including zero and one.
/// @param max_steps Maximum permitted updates, default 100000; zero is a strict zero budget.
/// @return Owned setup, actual steps and completed solution, with no borrowed data.
/// @throws std::overflow_error If a reduced coordinate exceeds signed 128-bit range or an auxiliary value exceeds uint64.
/// @throws std::length_error If the step budget is exhausted or vector capacity is exceeded.
/// @throws std::bad_alloc If record allocation fails.
/// @throws std::logic_error If an internal exact-divisibility or coprimality invariant fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chakravala_trace(13,6);
/// std::print("{}\n",dm::algorithms::int128_text(t.steps.back().after.k)); // 1
/// @endcode
/// @par References
/// - A. Bauval (2014), https://arxiv.org/abs/1406.6809; variant described in chakravala().
/// @see chakravala(), ChakravalaTrace
ChakravalaTrace chakravala_trace(std::uint64_t N,std::size_t max_steps=100000);
/// @brief Check a positive Pell pair exactly without squaring either signed-128 coordinate.
/// Requires found=true, N>=2, x>1 and y>0. Cancels gcd factors between
/// (x-1)*(x+1) and N*y*y, avoiding every large product. Accepts nonfundamental
/// solutions too; equation satisfaction alone does not establish minimality.
/// All manually supplied signed-128 values are handled without overflow. No
/// allocation or mutation; a constant number of Euclidean gcd computations.
/// @param N Coefficient for the proposed equation, any uint64.
/// @param solution Owned or borrowed record to inspect, without mutation.
/// @return True precisely when the flagged positive pair satisfies x*x-N*y*y=1.
/// @par Example
/// @code
/// const auto p=dm::algorithms::chakravala(661);
/// std::print("{}\n",dm::algorithms::pell_is_solution(661,p)); // true
/// @endcode
/// @see chakravala(), PellSolution
bool pell_is_solution(std::uint64_t N,const PellSolution& solution) noexcept;
} // namespace datamunge::algorithms
