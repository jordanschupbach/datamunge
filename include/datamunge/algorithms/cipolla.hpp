#pragma once
/// @file cipolla.hpp
/// @brief Checked full-uint64 prime modular square roots with actual candidate and quadratic-field traces.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned coefficients of x+y*s in one Cipolla call's quadratic extension.
/// The call supplies p and s^2=w; this plain record stores neither and does not
/// validate manually assigned coefficients. Returned coefficients are canonical
/// uint64 residues. The basis symbol s is formal, not the complex number i.
/// @par Example
/// @code
/// const auto e=dm::algorithms::cipolla_sqrt_trace(10,13).final_element;
/// std::print("{} {}\n",e.x,e.y); // 6 0
/// @endcode
/// @see CipollaPowerStep
struct CipollaElement {
  std::uint64_t x{0}; ///< Coefficient of one, in [0,p) for computed records.
  std::uint64_t y{0}; ///< Coefficient of s, in [0,p); zero for a completed root.
};
/// @brief Owned exact modular-root verdict and both unsigned root representatives.
/// root follows deterministic auxiliary search/powering order, not size order.
/// other_root is -root modulo p. They are distinct for nonzero residues at odd
/// primes; zero and p=2 have one distinct root. Failure has exists=false,
/// root=other_root=0 and legacy_root=-1. Default storage is not a completed solve.
/// legacy_root equals root when it fits long long; otherwise it uses other_root,
/// which fits because one of two complementary uint64 roots is at most p/2.
/// @par Example
/// @code
/// const auto r=dm::algorithms::cipolla_sqrt_result(10,13);
/// std::print("{} {} {}\n",r.exists,r.root,r.other_root); // true 6 7
/// @endcode
/// @see cipolla_sqrt_result(), cipolla_sqrt()
struct CipollaResult {
  bool exists{false}; ///< Whether a root exists for the validated prime modulus.
  std::uint64_t root{0}; ///< Actual unsigned extension result or direct shortcut root.
  std::uint64_t other_root{0}; ///< Additive opposite modulo p, equal to root for zero or p=2.
  long long legacy_root{-1}; ///< Representable root for the legacy signed API, or -1 for nonresidues.
};
/// @brief Actual route through the checked Cipolla solver.
/// Shortcuts still validate primality first. NonResidue is a mathematical Euler
/// verdict, while malformed moduli and exhausted work caps throw. Unprocessed
/// describes default storage, without a solved input.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cipolla_sqrt_trace(11,1009);
/// std::print("{}\n",t.path==dm::algorithms::CipollaPath::NonResidue); // true
/// @endcode
/// @see CipollaTrace
enum class CipollaPath {
  Unprocessed, ///< Default storage; no solve has executed.
  BinaryField, ///< Direct unique root in F_2.
  Zero, ///< Direct zero root for an odd prime.
  NonResidue, ///< Nonzero input fails Euler's criterion and has no root.
  Extension ///< Actual nonresidue search followed by quadratic-extension exponentiation.
};
/// @brief Whole-call actual-candidate and extension-product limits; zero is unlimited.
/// Candidate order is 1,2,...,p-1,0, fixing the missing-zero case while retaining
/// previous successful nonzero choices. Validation precedes limits and shortcuts.
/// Products count result multiplications plus every base square, including the
/// final unused square retained from the legacy loop. Primality and scalar Euler
/// powers are outside the extension-product cap. No partial result escapes a cap.
/// @par Example
/// @code
/// dm::algorithms::CipollaOptions options;options.max_candidates=2;options.max_extension_products=6;
/// std::print("{}\n",dm::algorithms::cipolla_sqrt_result(10,13,options).root); // 6
/// @endcode
/// @see cipolla_sqrt_trace(), CipollaWork
struct CipollaOptions {
  std::size_t max_candidates{0}; ///< Actual auxiliary candidates admitted; zero is unlimited.
  std::size_t max_extension_products{0}; ///< Actual pair multiplications/squares admitted; zero is unlimited.
};
/// @brief Actual tested auxiliary value, difference and Euler residue.
/// w=a^2-n modulo p. euler_value is w^((p-1)/2) modulo p, equal to zero,
/// one or p-1; p-1 marks acceptance. Every rejected test is retained in a trace.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cipolla_sqrt_trace(10,13).candidates.back();
/// std::print("{} {} {}\n",c.a,c.w,c.euler_value); // 2 7 12
/// @endcode
/// @see CipollaTrace
struct CipollaCandidate {
  std::uint64_t a{0}; ///< Actual canonical auxiliary candidate, including the final zero fallback.
  std::uint64_t w{0}; ///< Actual canonical a^2-normalized_n.
  std::uint64_t euler_value{0}; ///< Actual Euler-test residue; p-1 means a quadratic nonresidue.
};
/// @brief Owned actual state transition of one right-to-left exponent bit.
/// If remaining_exponent is odd, result is multiplied by base; otherwise result
/// stays unchanged. Base is always squared, including the final unused square.
/// Captures the executing arithmetic, not a recomputed explanation.
/// @par Example
/// @code
/// const auto s=dm::algorithms::cipolla_sqrt_trace(10,13).steps.back();
/// std::print("{} {} {}\n",s.remaining_exponent,s.result_after.x,s.result_after.y); // 1 6 0
/// @endcode
/// @see CipollaTrace, CipollaElement
struct CipollaPowerStep {
  std::uint64_t remaining_exponent{0}; ///< Actual positive exponent before this bit iteration.
  CipollaElement result_before; ///< Accumulator before the optional multiply.
  CipollaElement base_before; ///< Base before this iteration's square.
  bool multiplied{false}; ///< Whether the low exponent bit caused a result multiplication.
  CipollaElement result_after; ///< Actual accumulator after the optional multiply.
  CipollaElement base_after; ///< Actual squared base, including the closing square.
};
/// @brief Actual named operation counts, excluding arithmetic inside primality validation.
/// Scalar Euler powers retain a square in every bit iteration, including the
/// closing unused square. Each extension product performs exactly five modular
/// scalar products and two reduced additions, even for zero coefficients. Counts
/// are executed operations, not timing, secret-independent work or bit complexity.
/// @par Example
/// @code
/// const auto w=dm::algorithms::cipolla_sqrt_trace(10,13).work;
/// std::print("{} {} {}\n",w.candidate_tests,w.extension_result_products,w.extension_base_squares); // 2 3 3
/// @endcode
/// @see CipollaOptions, CipollaTrace
struct CipollaWork {
  std::size_t primality_checks{0}; ///< Actual deterministic uint64 primality-validator calls.
  std::size_t residue_power_products{0}; ///< Modular products in the input Euler test.
  std::size_t candidate_tests{0}; ///< Actual admitted auxiliary candidate visits.
  std::size_t candidate_squares{0}; ///< Actual a*a modular products.
  std::size_t candidate_power_products{0}; ///< Modular products in auxiliary Euler tests.
  std::size_t extension_result_products{0}; ///< Actual accumulator-by-base pair multiplications.
  std::size_t extension_base_squares{0}; ///< Actual pair base squares, including the final unused one.
  std::size_t extension_scalar_products{0}; ///< Actual reduced scalar products inside all pair operations.
  std::size_t verification_products{0}; ///< Actual final root-square checks; one on the extension path.
};
/// @brief Owned actual prime-field path, candidate tests, bit states and full-range root result.
/// The ordinary, unsigned-result and traced calls share a kernel without replay.
/// input_n is the caller's uint64; normalized_n is input_n modulo the validated
/// prime. residue_euler is zero when the p=2/zero paths skip its computation.
/// a,w,final_element and both record arrays stay zero/empty off the extension
/// path. No shared state or input mutation. Trace storage is O(T+log p) for T
/// actual candidates; no allocation is reserved from the modulus magnitude.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cipolla_sqrt_trace(10,13);
/// std::print("{} {} {}\n",t.a,t.w,t.steps.size()); // 2 7 3
/// @endcode
/// @see cipolla_sqrt_trace(), CipollaResult
struct CipollaTrace {
  std::uint64_t input_n{0}; ///< Original caller residue.
  std::uint64_t normalized_n{0}; ///< Original residue reduced modulo the checked prime.
  std::uint64_t prime{0}; ///< Validated prime, zero for default storage.
  CipollaPath path{CipollaPath::Unprocessed}; ///< Actual shortcut, nonresidue or extension path.
  CipollaResult result; ///< Full unsigned roots and representable legacy root.
  std::uint64_t residue_euler{0}; ///< Actual input Euler residue; zero when skipped.
  std::uint64_t a{0}; ///< Actual accepted auxiliary value on the extension path.
  std::uint64_t w{0}; ///< Actual nonresidue a^2-normalized_n modulo prime.
  CipollaElement final_element; ///< Actual final pair, with y=0 for a successful extension.
  std::vector<CipollaCandidate> candidates; ///< Every actual rejected/accepted auxiliary test.
  std::vector<CipollaPowerStep> steps; ///< Actual exponent-bit transitions and closing square.
  CipollaWork work; ///< Actual named counts; primality's internal products are excluded.
};
/// @brief Find a modular square root with the legacy signed result and full uint64 prime validation.
/// Validates p with the existing deterministic 64-bit Miller-Rabin API before
/// reducing n, avoiding modulo-by-zero and invalid zero shortcuts. Supports p=2
/// and n=0. Nonresidues return -1. The deterministic auxiliary order is
/// 1,2,...,p-1,0; including zero fixes sqrt(1) modulo 3. Previously successful
/// nonzero auxiliary choices remain. Products use unsigned-128 intermediates;
/// sums reduce components before adding, and differences never form a+p-n.
/// Returns the actual computed root when it fits long long, otherwise its
/// complementary root, which fits. This avoids returning a negative value for a valid root.
/// The algorithm is variable-time; sequential auxiliary search has no constant
/// expected-trial guarantee. O((T+1)*log p) named fixed-word products including
/// bounded prime validation, O(1) auxiliary storage; no allocation or shared state.
/// @param n Arbitrary unsigned residue, normalized modulo p.
/// @param p Prime uint64 modulus, including two.
/// @return Nonnegative representable root, or -1 iff the normalized residue is a nonresidue.
/// @throws std::invalid_argument If p is not prime, including zero and one.
/// @throws std::overflow_error If an actual work counter cannot fit size_t.
/// @throws std::logic_error If a validated-field invariant or final root check fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cipolla_sqrt(2,1009)); // 439
/// @endcode
/// @par References
/// - M. Cipolla (1903), Un metodo per la risoluzione della congruenza di secondo grado.
/// - A. Menezes, P. van Oorschot and S. Vanstone (1996), Handbook of Applied Cryptography,
///   Algorithm 3.39 and section 3.12 notes, https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
///   Uses the equivalent a+sqrt(a^2-n) basis and deterministic search instead of random b.
/// @see cipolla_sqrt_result(), cipolla_sqrt_trace(), cipolla_root_satisfies()
long long cipolla_sqrt(std::uint64_t n,std::uint64_t p);
/// @brief Return both unsigned roots and the compatible legacy root with optional work caps.
/// Same actual arithmetic/search as cipolla_sqrt(); the raw extension root keeps
/// its uint64 range even when above LLONG_MAX. Zero/p=2 have one distinct root.
/// Caps cover actual candidates and pair products; zero is unlimited. Validation
/// and mathematical nonresidue/shortcut paths precede unused limits. No partial
/// result escapes a cap failure. No input mutation, allocation or shared state.
/// @param n Unsigned input residue, normalized after prime validation.
/// @param p Valid uint64 prime, including two.
/// @param options Whole-call candidate/extension-product caps, default unlimited.
/// @return Complete root verdict with both unsigned roots and representable legacy value.
/// @throws std::invalid_argument If p is not prime.
/// @throws std::length_error If a positive work cap is exhausted.
/// @throws std::overflow_error If a work counter overflows size_t.
/// @throws std::logic_error If the validated-field invariant or final check fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::cipolla_sqrt_result(1,3);
/// std::print("{} {}\n",r.root,r.other_root); // 2 1
/// @endcode
/// @see cipolla_sqrt(), CipollaOptions
CipollaResult cipolla_sqrt_result(std::uint64_t n,std::uint64_t p,CipollaOptions options={});
/// @brief Retain actual candidate rejections and every extension exponent bit without replay.
/// Same kernel and final result as cipolla_sqrt_result(). Includes the closing
/// unused base square, and counts all five scalar products per pair operation.
/// Validation happens before shortcuts/caps. No partial trace escapes a cap or
/// allocation failure. O(T+log p) owned records for T actual candidates, with no
/// giant reservation based on p. Calls share no mutable state.
/// @param n Unsigned residue, normalized modulo the validated prime.
/// @param p Prime uint64 modulus, including two.
/// @param options Whole-call candidate/pair-product limits, default unlimited.
/// @return Owned actual field construction, bit states, result and named counts.
/// @throws std::invalid_argument If p is not prime.
/// @throws std::length_error If a work cap or record capacity is exceeded.
/// @throws std::overflow_error If a work counter cannot fit size_t.
/// @throws std::logic_error If the validated-field invariant or final check fails.
/// @throws std::bad_alloc If record allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cipolla_sqrt_trace(1,3);
/// std::print("{} {}\n",t.candidates.size(),t.candidates.back().a); // 3 0
/// @endcode
/// @par References
/// - A. Menezes, P. van Oorschot and S. Vanstone (1996), Algorithm 3.39,
///   https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
/// @see cipolla_sqrt_result(), CipollaTrace
CipollaTrace cipolla_sqrt_trace(std::uint64_t n,std::uint64_t p,CipollaOptions options={});
/// @brief Check one unsigned candidate root with widened modular multiplication.
/// Validates p first, including on zero inputs, then compares root^2 modulo p
/// with n modulo p. Arbitrary uint64 n/root representatives are permitted.
/// Empty/no-root result records must be checked via exists before using fields.
/// This verifies the equation, not root ordering or a timing/security property.
/// No allocation or mutation; prime validation plus one scalar modular product.
/// @param n Unsigned target residue.
/// @param p Prime uint64 modulus, including two.
/// @param root Unsigned candidate, normalized modulo p before squaring.
/// @return True iff the candidate satisfies the modular equation.
/// @throws std::invalid_argument If p is not prime.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cipolla_root_satisfies(2,1009,439)); // true
/// @endcode
/// @see cipolla_sqrt_result(), cipolla_sqrt()
bool cipolla_root_satisfies(std::uint64_t n,std::uint64_t p,std::uint64_t root);
}
