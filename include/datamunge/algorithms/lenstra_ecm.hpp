#pragma once
/// @file lenstra_ecm.hpp
/// @brief Checked unsigned64 stage1 affine ECM, actual curve/denominator evidence and explicit finite search limits.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Explicit bounded stage1 search settings; no stage2 or primality certificate.
/// @par Example
/// @code
/// dm::algorithms::ECMOptions o;
/// std::print("{}\n",o.bound); // 300
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see lenstra_ecm_analyze(), lenstra_ecm_trace()
struct ECMOptions {
  std::uint32_t bound{300}; ///< Prime-power smoothness bound in[2,1000000], default300.
  std::size_t   max_curves{
      3000}; ///< Maximum random curve attempts in[0,1000000], default3000;0 gives only small trial division.
};
/// @brief Owned normalized affine residue coordinates or the point at infinity.
/// @par Example
/// @code
/// const dm::algorithms::ECMPoint p{0,1,false};
/// std::print("{}\n",p.infinity); // false
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see lenstra_ecm_analyze(), lenstra_ecm_trace()
struct ECMPoint {
  std::uint64_t x{0};           ///< Normalized horizontal residue when finite.
  std::uint64_t y{0};           ///< Normalized vertical residue when finite.
  bool          infinity{true}; ///< True for identity; coordinate placeholders0.
};
/// @brief Actual affine addition operands, denominator GCD, inversion and resulting point.
/// @par Example
/// @code
/// const auto e=dm::algorithms::ecm_add(91,12,1,{0,1,false},{7,8,false});
/// std::print("{}\n",e.factor); // 7
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see lenstra_ecm_analyze(), lenstra_ecm_trace()
struct ECMAddition {
  std::size_t   curve{0};     ///< Actual random curve attempt index,0 for standalone addition.
  ECMPoint      left;         ///< Actual original first point.
  ECMPoint      right;        ///< Actual original second point.
  std::uint64_t numerator{0}; ///< Actual normalized slope numerator,0 if no denominator evaluated.
  std::uint64_t denominator{0};   ///< Actual normalized slope denominator,0 if unevaluated.
  std::uint64_t gcd{0};           ///< Actual gcd(denominator,n),0 if unevaluated.
  std::uint64_t slope{0};         ///< Actual modular slope only when inverted=true.
  bool          evaluated{false}; ///< True only when a slope denominator was actually evaluated.
  bool          inverted{false};  ///< True only when a coprime denominator was actually inverted.
  std::uint64_t factor{0};        ///< Actual proper denominator GCD if found, otherwise0.
  ECMPoint      result; ///< Actual returned point; infinity placeholder on failed inverse.
};
/// @brief Actual selected stage1 operation counts; excludes validation, sieve, allocation and elapsed time.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lenstra_ecm_analyze(10);
/// std::print("{}\n",a.work.curves); // 0
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see lenstra_ecm_analyze(), lenstra_ecm_trace()
struct ECMWork {
  std::size_t trial_divisions{0};   ///< Actual small-prime modulus checks, including even check.
  std::size_t curves{0};            ///< Actual random curves generated.
  std::size_t discriminant_gcds{0}; ///< Actual curve discriminant GCD evaluations.
  std::size_t singular_curves{0};   ///< Actual discriminant-GCD-equals-n discarded curves.
  std::size_t multiplications{0};   ///< Actual prime-power scalar multiplication calls.
  std::size_t additions{0}; ///< Actual affine add calls, including identity/inverse shortcuts.
  std::size_t denominator_gcds{0}; ///< Actual nonshortcut slope-denominator GCD evaluations.
  std::size_t inversions{0};       ///< Actual coprime modular inverse evaluations.
};
/// @brief Owned actual bounded-search result and work; n means failure or no proper factor, never a prime certificate.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lenstra_ecm_analyze(91);
/// std::print("{} {}\n",a.factor,a.proper); // 7 true
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see lenstra_ecm_analyze(), lenstra_ecm_trace()
struct ECMAnalysis {
  std::uint64_t n{0};      ///< Original input, including0/1 identity cases.
  std::uint64_t seed{0};   ///< Original caller seed;0 maps to documented nonzero generator state.
  ECMOptions    options;   ///< Actual validated stage1 limits.
  std::uint64_t factor{0}; ///< Proper factor on success; original n otherwise, including0/1.
  bool          proper{false}; ///< True exactly when1<factor<n and factor divides n.
  ECMWork       work;          ///< Actual same-execution selected operations.
};
/// @brief Owned actual generated curve and its final observed outcome.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lenstra_ecm_trace(10403,1,{.bound=20,.max_curves=1});
/// std::print("{}\n",t.curves.size()); // 1
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see lenstra_ecm_analyze(), lenstra_ecm_trace()
struct ECMCurve {
  std::size_t   index{0}; ///< Actual zero-based attempted curve index.
  ECMPoint      initial;  ///< Actual generated original point.
  std::uint64_t a{0};     ///< Actual normalized linear coefficient.
  std::uint64_t b{0};     ///< Actual coefficient chosen so initial lies on y squared=x cubed+a*x+b.
  std::uint64_t discriminant_gcd{0}; ///< Actual gcd(4*a cubed+27*b squared,n).
  std::uint64_t factor{0};           ///< Actual proper factor found on this curve,0 otherwise.
  ECMPoint
      final; ///< Actual point after performed stage1 multiplications; initial on discriminant exit.
};
/// @brief Owned complete actually generated curves and evaluated slope denominators; no replay.
/// Complete result or exception: each record quota is admitted before appending;
/// unknown future curve/denominator counts are not reserved as invented work.
/// Default one million actual records;0 disables optional cap. Excludes fields,
/// point components, capacities, output/runtime copies and bytes/RAM.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lenstra_ecm_trace(91);
/// std::print("{}\n",t.denominators.size()); // 0
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see lenstra_ecm_analyze(), lenstra_ecm_trace()
struct ECMTrace {
  ECMAnalysis analysis; ///< Actual final bounded result/options/work.
  std::vector<std::uint64_t>
      prime_powers; ///< Actual increasing-prime stage1 powers, generated once when search is reached.
  std::vector<ECMCurve> curves; ///< Actual completed curve records, including discriminant exits.
  std::vector<ECMAddition>
      denominators; ///< Actual evaluated denominator records in producer order; identity/inverse shortcuts excluded.
  std::size_t captured_entries{0}; ///< Actual prime-power/curve/denominator logical record count.
};
/// @brief Evaluate one checked on-curve affine addition modulo a composite or prime modulus.
/// Identity and additive-inverse shortcuts do not evaluate a denominator. If a
/// denominator has gcd n, return infinity without a proper factor. No primality
/// or nonsingular-curve certificate is implied. No mutation/shared state.
/// @param n Modulus>=2.
/// @param a Normalized curve coefficient, less than n.
/// @param b Normalized curve coefficient, less than n.
/// @param left Normalized finite on-curve point or infinity.
/// @param right Normalized finite on-curve point or infinity.
/// @return Owned actual operand/denominator/inversion/output evidence; proper GCD signals factor.
/// @throws std::invalid_argument On invalid modulus, residue range or off-curve points.
/// @par Example
/// @code
/// const auto e=dm::algorithms::ecm_add(91,12,1,{0,1,false},{7,8,false});
/// std::print("{} {}\n",e.denominator,e.gcd); // 7 7
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see ECMTrace, ECMAnalysis
ECMAddition
ecm_add(std::uint64_t n, std::uint64_t a, std::uint64_t b, ECMPoint left, ECMPoint right);
/// @brief Return a factor found by bounded stage1 affine ECM with historical default limits.
/// Small trial primes through47 precede up to3000 seeded curves, bound300.
/// Each curve chooses b so its generated point lies on the curve; discriminant
/// GCDs and failed slope inversions can expose factors. Full unsigned128 modular
/// products avoid former signed128 overflow for the unsigned64 domain. No stage2,
/// complete factorization, prime test, constant-time cryptography or success guarantee.
/// Search order/PRNG/default limits retain the old defined-range behavior.
/// @param n Original unsigned64 input;0/1 return themselves without a proper factor.
/// @param seed Xorshift64 seed, default0x243F6A8885A308D3;0 uses0x9E3779B97F4A7C15.
/// @return Proper factor if found; n otherwise, including prime, exhausted composite and0/1.
/// @throws std::bad_alloc On stage1 sieve/runtime allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::lenstra_ecm(91)); // 7
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see ECMTrace, ECMAnalysis
std::uint64_t lenstra_ecm(std::uint64_t n, std::uint64_t seed = 0x243F6A8885A308D3ULL);
/// @brief Return actual bounded stage1 factor/status/work with explicit finite search controls.
/// Same producing kernel and modular arithmetic as lenstra_ecm(), with no capture
/// or reruns. Zero curve attempts retains small trial division only.
/// @param n Original unsigned64 input;0/1 return themselves without a proper factor.
/// @param seed Xorshift64 seed, default0x243F6A8885A308D3;0 uses0x9E3779B97F4A7C15.
/// @param options Bound[2,1000000] and at most1000000 curves, default300/3000.
/// @return Owned actual result with proper-factor status, not a primality certificate.
/// @throws std::invalid_argument On invalid explicit stage1 limits.
/// @throws std::bad_alloc On sieve/runtime allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lenstra_ecm_analyze(10403,1,{.bound=20,.max_curves=0});
/// std::print("{} {}\n",a.factor,a.proper); // 10403 false
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see ECMTrace, ECMAnalysis
ECMAnalysis lenstra_ecm_analyze(
    std::uint64_t n,
    std::uint64_t seed    = 0x243F6A8885A308D3ULL,
    ECMOptions    options = {});
/// @brief Capture every actual generated curve and evaluated slope denominator of one complete bounded run.
/// Actual capture counts prime powers, curve records and evaluated denominator
/// records, not identity/inverse shortcuts or unused future attempts. Quota is
/// checked before each record append; failure throws with no partial returned
/// result. Count-only execution remains available for capture-heavy searches.
/// @param n Original unsigned64 input;0/1 return themselves without a proper factor.
/// @param seed Xorshift64 seed, default0x243F6A8885A308D3;0 uses0x9E3779B97F4A7C15.
/// @param options Same explicit validated limits as analysis.
/// @param max_entries Actual logical record cap, default1000000;0 disables only optional cap.
/// @return Owned complete evidence and actual result/work; never a silently truncated prefix.
/// @throws std::invalid_argument On invalid explicit stage1 limits.
/// @throws std::length_error On actual logical count/quota/container limits.
/// @throws std::bad_alloc On sieve/runtime/capture allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lenstra_ecm_trace(91);
/// std::print("{}\n",t.analysis.factor); // 7
/// @endcode
/// @par References
/// - H. W. Lenstra Jr.(1987), Factoring Integers with Elliptic Curves, Annals of Mathematics126(3),649–673, https://doi.org/10.2307/1971363.
/// @see ECMTrace, ECMAnalysis
ECMTrace lenstra_ecm_trace(
    std::uint64_t n,
    std::uint64_t seed        = 0x243F6A8885A308D3ULL,
    ECMOptions    options     = {},
    std::size_t   max_entries = 1000000);
/// @brief Exact uniform integer count of B-smooth candidates in a small prime Hasse interval.
/// This is not a distribution of elliptic-curve orders or a measured ECM success
/// probability. Prime-power divisibility by the stage1 multiplier is stricter
/// than B-smoothness alone. Counts include1 as the empty prime product.
/// @par Example
/// @code
/// const auto h=dm::algorithms::ecm_hasse_smoothness(101,50);
/// std::print("{} {}\n",h.lower,h.upper); // 82 122
/// @endcode
/// @see ecm_hasse_smoothness()
struct ECMHasseStudy {
  std::uint32_t prime{0};  ///< Actual validated prime, at most one million.
  std::uint32_t bound{0};  ///< Actual B-smoothness bound, at least2 and at most one million.
  std::uint32_t lower{0};  ///< Actual inclusive p+1-floor(2*sqrt(p)).
  std::uint32_t upper{0};  ///< Actual inclusive p+1+floor(2*sqrt(p)).
  std::size_t   smooth{0}; ///< Actual interval integers with no prime factor larger than bound.
  std::size_t   total{0};  ///< Actual complete integer interval count.
};
/// @brief Count smooth integers in the complete small-prime Hasse interval as an explicitly uniform proxy.
/// No curve sampling or factor search is performed. Trial division validates the
/// prime and every interval integer. O(sqrt(p)*sqrt(p+sqrt(p))) simple trial work.
/// @param prime Prime in[2,1000000], completely checked.
/// @param bound B-smooth prime-factor threshold in[2,1000000].
/// @return Owned complete interval bounds and exact integer counts.
/// @throws std::invalid_argument On nonprime/out-of-range prime or bound.
/// @par Example
/// @code
/// const auto h=dm::algorithms::ecm_hasse_smoothness(101,50);
/// std::print("{}\n",h.total); // 41
/// @endcode
/// @par References
/// - Lenstra(1987), https://doi.org/10.2307/1971363; uniform interval smoothness is explicitly only a proxy here.
/// @see lenstra_ecm_trace()
ECMHasseStudy ecm_hasse_smoothness(std::uint32_t prime, std::uint32_t bound);
} // namespace datamunge::algorithms
