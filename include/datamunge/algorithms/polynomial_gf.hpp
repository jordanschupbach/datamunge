#pragma once
/// @file polynomial_gf.hpp
/// @brief Checked prime-field arithmetic, actual Chien root search and squarefree odd-field factorization.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned low-to-high polynomial coefficients over a checked signed-64-bit prime field.
/// Public functions accept arbitrary int64 coefficients and trailing zeros, then
/// reduce into [0,p) and trim. Empty represents zero; nonzero degree is size()-1.
/// No modulus is stored in this alias. Copies own their coefficients independently.
/// @par Example
/// @code
/// const dm::algorithms::GFPoly f{4,0,1}; // x^2+4 over GF(5)
/// std::print("{}\n",dm::algorithms::poly_string_gf(f,5)); // x^2 + 4
/// @endcode
/// @see poly_normalize_gf(), poly_divmod_gf()
using GFPoly=std::vector<std::int64_t>;
/// @brief Owned normalized quotient and remainder of prime-field long division.
/// For nonzero divisor b, a=q*b+r modulo p, with r zero or lower degree than b.
/// The original aggregate layout is retained; neither vector borrows its inputs.
/// @par Example
/// @code
/// const auto qr=dm::algorithms::poly_divmod_gf({4,0,1},{4,1},5);
/// std::print("{} {}\n",qr.quotient,qr.remainder); // [1, 1] []
/// @endcode
/// @see poly_divmod_gf()
struct PolyDivMod {
  GFPoly quotient; ///< Owned normalized quotient.
  GFPoly remainder; ///< Owned normalized remainder, empty for exact divisibility.
};
/// @brief One actual distinct-degree step on the current remaining monic polynomial.
/// frobenius is x^(p^degree) modulo remaining_before. bucket is the monic gcd
/// with frobenius-x, possibly the unit [1]. remaining_after is the exact quotient.
/// Previously removed degree buckets are absent. Records own all coefficients.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cantor_zassenhaus_trace({4,4,4,2,1},5);
/// std::print("{} {}\n",t.distinct_degree[0].degree,t.distinct_degree[0].bucket); // 1 [2, 2, 1]
/// @endcode
/// @see CantorZassenhausTrace
struct CZDistinctDegreeStep {
  std::size_t degree=0; ///< Irreducible degree being extracted, starting at one.
  GFPoly remaining_before; ///< Actual monic remainder before this step.
  GFPoly frobenius; ///< Actual reduced Frobenius iterate.
  GFPoly bucket; ///< Actual monic gcd, [1] when no factor has this degree.
  GFPoly remaining_after; ///< Actual remainder after bucket division, or unchanged.
};
/// @brief One actual random equal-degree splitting attempt, including rejected constants.
/// Candidate coefficients come from the legacy seeded xorshift stream modulo p;
/// they are deterministic and slightly biased, not ideal independent uniform samples.
/// A constant candidate is rejected without gcd/powering (all following polynomials
/// empty). Otherwise initial_gcd is computed first. A proper gcd is accepted directly;
/// else used_character=true and character is h^((p^d-1)/2) mod target. split_gcd then
/// comes from gcd(character-1,target). quotient is set only for an accepted split.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cantor_zassenhaus_trace({4,0,1},5);
/// std::print("{}\n",t.equal_degree.back().accepted); // true
/// @endcode
/// @see CantorZassenhausTrace
struct CZEqualDegreeTrial {
  std::size_t attempt=0; ///< Zero-based whole-call attempt index.
  std::size_t degree=0; ///< Common irreducible degree of this target bucket.
  GFPoly target; ///< Actual pending monic polynomial being split.
  GFPoly candidate; ///< Actual normalized random polynomial of lower degree.
  GFPoly initial_gcd; ///< Actual gcd(candidate,target), empty only if skipped.
  bool used_character=false; ///< Whether a character power and second gcd were computed.
  GFPoly character; ///< Actual modular character value, empty if not computed or zero.
  GFPoly split_gcd; ///< Actual proposed monic factor, possibly trivial.
  GFPoly quotient; ///< Actual complementary factor for accepted splits; empty otherwise.
  bool accepted=false; ///< True only for a proper nonconstant split.
};
/// @brief Seed and complete-call work caps for odd-field squarefree factorization.
/// Zero caps mean unlimited. Trials include constant candidates; modular products
/// count multiply-then-reduce calls in polynomial powering/character construction,
/// including DDF. Caps throw before excess work and never return a partial factor list.
/// They do not bound coefficient operations, primality validation, gcd/division or time.
/// @par Example
/// @code
/// dm::algorithms::CantorZassenhausOptions o;
/// o.max_random_trials=100;
/// std::print("{}\n",o.max_modular_products); // 0
/// @endcode
/// @see cantor_zassenhaus_trace(), cantor_zassenhaus_with_options()
struct CantorZassenhausOptions {
  std::uint64_t seed=0x9E3779B97F4A7C15ULL; ///< Legacy xorshift seed; zero maps to 0x1234567.
  std::size_t max_random_trials=0; ///< Whole-call random-attempt cap, zero unlimited.
  std::size_t max_modular_products=0; ///< Whole-call modular polynomial-product cap, zero unlimited.
};
/// @brief Actual high-level work counts from the shared factorization kernel.
/// Counts do not measure scalar field operations or wall time. Gcd calls include
/// squarefree validation, DDF and EDF; modular products count only multiply/reduce
/// requests in powering/character computation. Ordinary calls keep these counters
/// internally but allocate no DDF/EDF records.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cantor_zassenhaus_trace({4,4,4,2,1},5);
/// std::print("{}\n",t.work.distinct_degree_steps); // 1
/// @endcode
/// @see CantorZassenhausTrace
struct CantorZassenhausWork {
  std::size_t distinct_degree_steps=0; ///< Number of actual DDF iterations.
  std::size_t random_trials=0; ///< Number of actual candidate polynomials drawn.
  std::size_t random_words=0; ///< Xorshift outputs consumed for candidate coefficients.
  std::size_t modular_power_calls=0; ///< Actual binary polynomial-powering requests.
  std::size_t modular_products=0; ///< Actual multiply-and-reduce requests in powering/characters.
  std::size_t gcd_calls=0; ///< Actual polynomial gcd requests including input squarefree validation.
};
/// @brief Owned input normalization, factors and actual DDF/EDF provenance without replay.
/// input is normalized but retains its leading coefficient. monic_input is the
/// divided-by-leading-coefficient polynomial. Product(factors)=monic_input; multiply
/// by leading_coefficient to recover input. residual is the final DDF remainder;
/// if nonconstant it is emitted directly as the last irreducible factor. Constants
/// have no factors and residual=[1]. Invalid/zero/repeated-factor inputs throw.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cantor_zassenhaus_trace({4,4,4,2,1},5);
/// std::print("{} {}\n",t.factors.size(),t.residual); // 3 [2, 0, 1]
/// @endcode
/// @see cantor_zassenhaus_trace(), CZDistinctDegreeStep, CZEqualDegreeTrial
struct CantorZassenhausTrace {
  std::int64_t modulus=0; ///< Checked odd prime p.
  std::int64_t leading_coefficient=0; ///< Nonzero normalized input leading coefficient.
  GFPoly input; ///< Owned normalized input, not necessarily monic.
  GFPoly monic_input; ///< Owned input divided by its leading coefficient.
  GFPoly squarefree_gcd; ///< Actual gcd(monic_input,derivative), [1] on success.
  GFPoly residual; ///< Actual final DDF remainder, possibly [1].
  std::vector<GFPoly> factors; ///< Owned monic irreducibles, degree-bucket then EDF depth-first order.
  std::vector<CZDistinctDegreeStep> distinct_degree; ///< Actual incremental Frobenius/gcd steps.
  std::vector<CZEqualDegreeTrial> equal_degree; ///< Actual random attempts, including failures.
  CantorZassenhausOptions options; ///< Effective seed and whole-call work caps.
  CantorZassenhausWork work; ///< Actual checked kernel counters.
};
/// @brief Reduce coefficients into [0,p) and trim trailing zeros without signed overflow.
/// O(n) work/storage, owned result, inputs unchanged, no shared state. Validates the
/// signed-64-bit prime using the existing deterministic 64-bit Miller-Rabin API.
/// @param f Borrowed arbitrary int64 coefficients, including empty and trailing zeros.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return Owned normalized coefficients, empty for zero.
/// @throws std::invalid_argument p is not a positive prime.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::poly_normalize_gf({-1,7,0},5)); // [4, 2]
/// @endcode
/// @see GFPoly, poly_multiply_gf()
GFPoly poly_normalize_gf(const GFPoly& f,std::int64_t p);
/// @brief Schoolbook polynomial product with overflow-safe modular accumulation.
/// Normalizes arbitrary signed inputs; zero times anything is zero. Uses 128-bit
/// intermediates, O(a.size()*b.size()) field operations and O(a.size()+b.size())
/// result storage. Borrows inputs, owns result, no shared mutable state.
/// @param a Borrowed low-to-high coefficients.
/// @param b Borrowed low-to-high coefficients, possibly aliasing a.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return Owned normalized product.
/// @throws std::invalid_argument Modulus is not prime.
/// @throws std::length_error Product coefficient count overflows size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::poly_multiply_gf({4,1},{3,1},5)); // [2, 2, 1]
/// @endcode
/// @see poly_divmod_gf(), poly_normalize_gf()
GFPoly poly_multiply_gf(const GFPoly& a,const GFPoly& b,std::int64_t p);
/// @brief Long division over a checked prime field, accepting arbitrary signed coefficients.
/// Normalizes both operands and rejects a divisor that becomes zero. Remainder degree
/// is strictly smaller than the normalized divisor; division by a nonzero constant
/// returns zero remainder. Schoolbook quadratic field work; inputs unchanged, owned result.
/// @param a Borrowed dividend, possibly zero.
/// @param b Borrowed divisor, nonzero after reduction modulo p.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return Owned quotient/remainder satisfying a=q*b+r modulo p.
/// @throws std::invalid_argument Invalid prime modulus or zero reduced divisor.
/// @par Example
/// @code
/// const auto qr=dm::algorithms::poly_divmod_gf({4,0,1},{4,1},5);
/// std::print("{} {}\n",qr.quotient,qr.remainder); // [1, 1] []
/// @endcode
/// @par References
/// - V. Shoup, A Computational Introduction to Number Theory and Algebra, Chapter 17.
///   https://shoup.net/ntb/ntb-v2.pdf
/// @see PolyDivMod, poly_multiply_gf()
PolyDivMod poly_divmod_gf(const GFPoly& a,const GFPoly& b,std::int64_t p);
/// @brief Monic Euclidean polynomial gcd over a checked prime field.
/// gcd(0,0)=0; any nonzero constant gcd normalizes to [1]. O(n^2) schoolbook
/// field work plus leading-coefficient inversion, owned result, inputs unchanged.
/// @param a Borrowed arbitrary signed coefficients.
/// @param b Borrowed arbitrary signed coefficients.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return Owned normalized monic gcd, or empty for two zero inputs.
/// @throws std::invalid_argument Modulus is not prime.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::poly_gcd_gf({4,0,1},{4,1},5)); // [4, 1]
/// @endcode
/// @par References
/// - V. Shoup, Chapter 17. https://shoup.net/ntb/ntb-v2.pdf
/// @see poly_divmod_gf(), poly_is_squarefree_gf()
GFPoly poly_gcd_gf(const GFPoly& a,const GFPoly& b,std::int64_t p);
/// @brief Evaluate a polynomial by overflow-safe modular Horner recurrence.
/// Normalizes arbitrary signed coefficients and x. Empty gives zero; O(n) field
/// work and constant extra storage, input unchanged, no shared state.
/// @param f Borrowed low-to-high coefficients, trailing zeros permitted.
/// @param x Any signed-64-bit evaluation point, reduced modulo p.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return Value in [0,p).
/// @throws std::invalid_argument Modulus is not prime.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::poly_evaluate_gf({4,0,1},1,5)); // 0
/// @endcode
/// @see poly_normalize_gf(), chien_search()
std::int64_t poly_evaluate_gf(const GFPoly& f,std::int64_t x,std::int64_t p);
/// @brief Return the normalized formal derivative, including characteristic-p zeros.
/// Coefficient i becomes i*f[i] modulo p, without narrowing the degree or signed
/// overflow. O(n) work/storage, owned output, borrowed input unchanged.
/// @param f Borrowed arbitrary signed coefficients.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return Owned derivative, empty for constants and zero derivatives.
/// @throws std::invalid_argument Modulus is not prime.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::poly_derivative_gf({1,2,3},5)); // [2, 1]
/// @endcode
/// @see poly_is_squarefree_gf()
GFPoly poly_derivative_gf(const GFPoly& f,std::int64_t p);
/// @brief Test absence of repeated irreducible factors using gcd(f,f').
/// Zero returns false; nonzero constants return true. Characteristic-p zero
/// derivatives of nonconstant polynomials return false. Quadratic schoolbook gcd
/// field work, inputs unchanged and no shared mutable state.
/// @param f Borrowed arbitrary signed coefficients.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return True for nonzero squarefree polynomials, including units.
/// @throws std::invalid_argument Modulus is not prime.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::poly_is_squarefree_gf({1,2,1},5)); // false
/// @endcode
/// @par References
/// - V. Shoup, Section 20.3. https://shoup.net/ntb/ntb-v2.pdf
/// @see poly_derivative_gf(), cantor_zassenhaus()
bool poly_is_squarefree_gf(const GFPoly& f,std::int64_t p);
/// @brief Deterministically test irreducibility by excluding all small-degree factors.
/// A nonconstant polynomial is reducible iff it has an irreducible factor of degree
/// at most half its degree. Repeated Frobenius powering and gcds test these degrees;
/// zero/constants return false, all linears true. O(n^3*log(p)) schoolbook field
/// operations as an upper bound, O(n) scratch storage, inputs unchanged.
/// @param f Borrowed arbitrary signed coefficients; monicity is not required.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return True iff the reduced polynomial is nonconstant and irreducible over GF(p).
/// @throws std::invalid_argument Modulus is not prime.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::poly_is_irreducible_gf({2,0,1},5)); // true
/// @endcode
/// @par References
/// - V. Shoup, Sections 20.1 and 20.4. https://shoup.net/ntb/ntb-v2.pdf
/// @see poly_is_squarefree_gf(), cantor_zassenhaus()
bool poly_is_irreducible_gf(const GFPoly& f,std::int64_t p);
/// @brief Format normalized coefficients as a descending-power ASCII polynomial.
/// Omits zero terms and coefficient one on nonconstant terms; zero is "0". Owned
/// string, inputs unchanged, linear coefficient traversal plus output formatting.
/// @param f Borrowed arbitrary signed coefficients, possibly empty.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return Owned canonical descending polynomial string, e.g. "x^2 + 4".
/// @throws std::invalid_argument Modulus is not prime.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::poly_string_gf({4,4,4,2,1},5)); // x^4 + 2x^3 + 4x^2 + 4x + 4
/// @endcode
/// @see GFPoly, poly_normalize_gf()
std::string poly_string_gf(const GFPoly& f,std::int64_t p);
/// @brief Whole-call limits for Chien enumeration and generator discovery; zero means unlimited.
/// Point/term caps are checked from the normalized degree before generator work
/// or sweep allocation. Generator steps count trial divisors plus primitive-root
/// power tests, checked before each such operation; scalar arithmetic within a
/// test and repeated division by an accepted factor are not separate steps.
/// All shortcut paths use zero field-point and generator steps. Limits bound
/// these named operations, not primality validation, normalization or elapsed time.
/// @par Example
/// @code
/// dm::algorithms::ChienOptions options;options.max_points=7;
/// std::print("{}\n",dm::algorithms::chien_search_with_options({6,2,1},7,options)); // [3, 2]
/// @endcode
/// @see chien_search_with_options(), chien_search_trace()
struct ChienOptions {
  std::size_t max_points=0; ///< Maximum field-point checks, including the zero coefficient check; zero is unlimited.
  std::size_t max_term_products=0; ///< Maximum coefficient-state update products; zero is unlimited.
  std::size_t max_generator_steps=0; ///< Maximum trial-divisor attempts plus generator power tests; zero is unlimited.
};
/// @brief Actual path used by the ordinary/traced Chien kernel after normalization.
/// Shortcuts do not pretend to enumerate points or retain recurrence states.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chien_search_trace({2,2},5);
/// std::print("{}\n",t.path==dm::algorithms::ChienPath::Linear); // true
/// @endcode
/// @see ChienTrace
enum class ChienPath {
  Zero, ///< Legacy zero-polynomial sentinel: no roots enumerated, although every field element is mathematically a root.
  Constant, ///< Nonzero constant: no roots and no sweep.
  Linear, ///< One distinct root obtained by inversion, without finding a generator or sweeping.
  Incremental ///< Zero checked directly, then all nonzero points visited through primitive-root powers.
};
/// @brief Actual named operations, including the legacy constant/closing state updates.
/// Excludes prime testing, coefficient normalization, allocations and integer
/// division/modulo work in factoring p-1. Each modular scalar multiplication
/// is counted once, regardless of its bit cost; additions count all accumulator
/// updates, including zero coefficients. Counters are checked size_t values.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chien_search_trace({4,3,2,1},13);
/// std::print("{} {}\n",t.work.field_points,t.work.term_products); // 13 48
/// @endcode
/// @see ChienTrace, ChienOptions
struct ChienWork {
  std::size_t field_points=0; ///< Actual zero/nonzero point checks; zero for shortcut paths.
  std::size_t generator_steps=0; ///< Sum of trial-divisor attempts and primitive-root power tests.
  std::size_t generator_divisors=0; ///< Trial divisors attempted while factoring p-1, excluding repeated successful divisions.
  std::size_t generator_tests=0; ///< Scalar-power comparisons g^((p-1)/q)!=1, including rejected candidates.
  std::size_t generator_products=0; ///< Modular products inside those scalar-power tests.
  std::size_t setup_products=0; ///< Modular products while computing g^j independently for every coefficient index.
  std::size_t term_products=0; ///< Actual state updates, including j=0 and the final unused closing update.
  std::size_t point_products=0; ///< Actual point*=g updates, including the final return to one.
  std::size_t sum_additions=0; ///< Modular term-accumulator additions at nonzero points.
  std::size_t linear_products=0; ///< Inversion scalar-power products plus the root product on the linear shortcut.
};
/// @brief One actual generator power test, retaining rejected candidates without replay.
/// q is a distinct prime divisor of p-1; candidate passes this test iff value!=1.
/// A primitive root must pass every distinct-factor test. Records own their values.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chien_search_trace({6,2,1},7);
/// std::print("{} {}\n",t.generator_tests[0].candidate,t.generator_tests[0].value); // 2 1
/// @endcode
/// @see ChienTrace
struct ChienGeneratorTest {
  std::int64_t candidate=0; ///< Actual positive generator candidate, tested in ascending order.
  std::int64_t q=0; ///< Distinct prime divisor of p-1 determining this exponent.
  std::int64_t value=0; ///< Actual candidate^((p-1)/q) modulo p.
};
/// @brief Owned actual coefficient-state vector and evaluation at one nonzero point.
/// terms[j]=input[j]*point^j mod p immediately before the next update; value
/// is their modular sum. The first record has exponent zero and point one.
/// Recording does not re-evaluate the polynomial with Horner's method.
/// @par Example
/// @code
/// const auto s=dm::algorithms::chien_search_trace({6,2,1},7).steps[1];
/// std::print("{} {} {}\n",s.point,s.terms,s.value); // 3 [6, 6, 2] 0
/// @endcode
/// @see chien_search_trace(), ChienTrace
struct ChienStep {
  std::size_t exponent=0; ///< Actual exponent i in point=g^i.
  std::int64_t point=0; ///< Actual nonzero field element visited.
  GFPoly terms; ///< Owned coefficient states before this point's closing update.
  std::int64_t value=0; ///< Actual modular term sum.
  bool root=false; ///< Whether value equals zero.
};
/// @brief Owned normalized input, actual generator/recurrence records, roots and work.
/// Zero, constant and linear shortcuts leave generator, multipliers, zero-check
/// and recurrence records empty/default; path identifies them explicitly. For
/// Incremental, zero is checked separately, steps covers p-1 nonzero points,
/// and the retained final update returns final_point=1 and final_terms=input.
/// This is a prime-field implementation, not arithmetic over GF(2^m) for m>1.
/// All records are independent of the input and other calls; nothing is replayed.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chien_search_trace({6,2,1},7);
/// std::print("{} {}\n",t.generator,t.roots); // 3 [3, 2]
/// @endcode
/// @see chien_search_trace(), ChienWork
struct ChienTrace {
  std::int64_t modulus=0; ///< Checked prime p, owned by value.
  GFPoly input; ///< Owned normalized/trailing-zero-trimmed coefficients.
  ChienPath path=ChienPath::Zero; ///< Actual shortcut or recurrence branch.
  std::int64_t generator=0; ///< Smallest primitive root for Incremental; one for GF(2), zero on shortcuts.
  std::vector<std::int64_t> generator_factors; ///< Distinct prime factors of p-1 used by the actual search.
  std::vector<ChienGeneratorTest> generator_tests; ///< Actual power comparisons, including rejected candidates.
  GFPoly multipliers; ///< Fixed actual g^j values, one per normalized coefficient.
  bool zero_checked=false; ///< Whether input[0] was checked as f(0) in the incremental path.
  std::int64_t zero_value=0; ///< Actual f(0) when zero_checked, default zero otherwise.
  std::vector<ChienStep> steps; ///< Actual nonzero evaluations in multiplicative order.
  GFPoly final_terms; ///< Actual coefficient states after the final closing update.
  std::int64_t final_point=0; ///< Actual point after all updates, one for Incremental.
  std::vector<std::int64_t> roots; ///< Distinct roots, zero first when checked/root, then primitive-power order; one root for Linear.
  ChienWork work; ///< Actual checked operation counts.
};
/// @brief Find roots by the existing primitive-root Chien evaluation order.
/// Checked prime p; arbitrary signed/trailing-zero coefficients are normalized.
/// Retains the legacy zero-polynomial convention of returning no roots (rather
/// than enumerating the whole field). Constants return empty; linears are solved
/// directly. Higher degrees return zero first if a root, then nonzero roots in
/// successive powers of the smallest primitive root. No sorting or multiplicities.
/// Field enumeration is O(p*n) and generator search trial-factors p-1; intended
/// for small fields, not huge p. Inputs unchanged, owned result, no shared state.
/// @param f Borrowed low-to-high coefficients.
/// @param p Prime modulus in [2,INT64_MAX].
/// @return Owned distinct roots in the documented legacy order.
/// @throws std::invalid_argument Modulus is not prime.
/// @throws std::length_error If checked sweep/vector capacity is exceeded.
/// @throws std::overflow_error If an actual work counter overflows size_t.
/// @throws std::bad_alloc If coefficient/root allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::chien_search({4,0,1},5)); // [1, 4]
/// @endcode
/// @par References
/// - R. T. Chien (1964), Cyclic Decoding Procedures for Bose-Chaudhuri-Hocquenghem Codes.
///   https://doi.org/10.1109/TIT.1964.1053699
/// @see poly_evaluate_gf(), chien_search_trace(), chien_search_with_options()
std::vector<std::int64_t> chien_search(const GFPoly& f,std::int64_t p);
/// @brief Run the legacy Chien root search with explicit whole-call operation limits.
/// Same normalization, shortcuts, distinct-root order and zero convention as
/// chien_search(). Point and state-product forecasts reject before generator
/// search/sweep work; generator-step caps are checked during its actual search.
/// Reaching any cap returns no partial root set. Ordinary calls retain no
/// generator/term provenance. Inputs are unchanged; independent calls share no state.
/// @param f Borrowed low-to-high signed coefficients, including trailing zeros.
/// @param p Prime modulus in [2,INT64_MAX]; this is GF(p), not a composite-order extension field.
/// @param options Named complete-call limits, default all unlimited.
/// @return Owned distinct roots in the legacy branch-dependent order.
/// @throws std::invalid_argument If p is not prime.
/// @throws std::length_error If a limit or checked sweep/vector capacity is exceeded.
/// @throws std::overflow_error If an actual scalar work counter overflows size_t.
/// @throws std::bad_alloc If local coefficient/root allocation fails.
/// @par Example
/// @code
/// dm::algorithms::ChienOptions options;options.max_term_products=18;
/// std::print("{}\n",dm::algorithms::chien_search_with_options({6,2,1},7,options)); // [3, 2]
/// @endcode
/// @par References
/// - R. T. Chien (1964), Cyclic Decoding Procedures for Bose-Chaudhuri-Hocquenghem Codes,
///   https://doi.org/10.1109/TIT.1964.1053699; prime-field adaptation and limits here.
/// @see chien_search(), chien_search_trace(), ChienOptions
std::vector<std::int64_t> chien_search_with_options(const GFPoly& f,std::int64_t p,ChienOptions options={});
/// @brief Retain actual generator tests and every coefficient state from the shared Chien search.
/// Same numeric/limit semantics as chien_search_with_options(), including constant
/// and direct-linear shortcuts and the zero-polynomial sentinel. Generator tests
/// and term vectors are captured in the executing kernel, never replayed.
/// Incremental trace storage is O((p-1)*n) coefficient values plus generator
/// tests and roots; max_points/max_term_products also bound that sweep storage.
/// No partial trace escapes on failure; caller inputs are unchanged. Prime checking,
/// trial-factoring p-1 and setup are additional to the O(p*n) sweep.
/// @param f Borrowed low-to-high signed coefficients, normalized into the owned trace.
/// @param p Prime modulus in [2,INT64_MAX], intended small for degree at least two.
/// @param options Named complete-call limits, default all unlimited.
/// @return Owned actual path, setup, evaluations, roots and operation counts.
/// @throws std::invalid_argument If p is not prime.
/// @throws std::length_error If a limit or checked sweep/vector capacity is exceeded.
/// @throws std::overflow_error If an actual scalar work counter overflows size_t.
/// @throws std::bad_alloc If local trace allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::chien_search_trace({6,2,1},7);
/// std::print("{} {}\n",t.final_point,t.final_terms); // 1 [6, 2, 1]
/// @endcode
/// @par References
/// - R. T. Chien (1964), https://doi.org/10.1109/TIT.1964.1053699; actual-state diagnostics here.
/// @see chien_search(), chien_search_with_options(), ChienTrace
ChienTrace chien_search_trace(const GFPoly& f,std::int64_t p,ChienOptions options={});
/// @brief Factor a nonzero squarefree polynomial over an odd prime field.
/// Preserves the original signature, default seed, coefficient alias and output
/// traversal policy. Nonmonic inputs are divided by their leading coefficient;
/// returned monic factors reconstruct that monic associate, not the original unit.
/// Nonzero constants return no factors; zero/repeated factors and nonprime/even p
/// throw before random splitting. No squarefree decomposition or GF(2) splitter is
/// supplied. DDF uses incremental Frobenius; EDF uses an explicit depth-first stack.
/// Character powering avoids materializing p^d, so no uint64 exponent wraps.
/// The legacy deterministic modulo-reduced xorshift sampler is retained; ideal
/// uniform-random expected-time claims do not guarantee termination for every seed.
/// Ordinary calls allocate no DDF/EDF records. Inputs are unchanged, result owns
/// coefficients, no shared state. See the configured variant for work/space costs.
/// @param f Borrowed arbitrary signed coefficients, nonzero and squarefree modulo p.
/// @param p Odd prime in [3,INT64_MAX].
/// @param seed Legacy xorshift seed; default 0x9E3779B97F4A7C15, zero maps to 0x1234567.
/// @return Owned monic irreducible factors, degree-bucket/EDF depth-first order.
/// @throws std::invalid_argument Invalid field, zero input or repeated factors.
/// @throws std::length_error A polynomial product coefficient count exceeds size_t.
/// @throws std::overflow_error Actual work counters exceed size_t.
/// @par Example
/// @code
/// const auto factors=dm::algorithms::cantor_zassenhaus({4,4,4,2,1},5);
/// std::print("{}\n",factors.size()); // 3
/// @endcode
/// @par References
/// - D. G. Cantor and H. Zassenhaus (1981), A New Algorithm for Factoring Polynomials
///   Over Finite Fields, Mathematics of Computation 36(154), 587-592.
///   https://doi.org/10.1090/S0025-5718-1981-0606517-5
/// @see cantor_zassenhaus_with_options(), cantor_zassenhaus_trace(), poly_is_irreducible_gf()
std::vector<GFPoly> cantor_zassenhaus(const GFPoly& f,std::int64_t p,
    std::uint64_t seed=0x9E3779B97F4A7C15ULL);
/// @brief Run checked squarefree odd-field factorization with whole-call work caps.
/// Shares the original function's kernel, normalization, ordering and arithmetic.
/// Schoolbook modular products/divisions cost O(n^2) field work. DDF costs at most
/// O(n^3*log(p)); one EDF trial on an m-degree degree-d bucket costs
/// O(d*m^2*log(p)), excluding scalar inversions and sampling. No fixed trial bound
/// follows from the deterministic sampler. Untraced pending polynomials and dense
/// scratch use O(n) coefficient storage, in addition to the output and inputs.
/// Both budgets exclude scalar primality/gcd/division work; no partial factors return.
/// @param f Borrowed nonzero squarefree coefficients; nonmonic inputs allowed.
/// @param p Odd prime in [3,INT64_MAX].
/// @param options Seed and complete-call random-trial/modular-product budgets; defaults unlimited.
/// @return Owned monic irreducibles in the original traversal order.
/// @throws std::invalid_argument Invalid field, zero input or repeated factors.
/// @throws std::length_error Product size overflow or a work cap would be exceeded.
/// @throws std::overflow_error Actual work counters exceed size_t.
/// @par Example
/// @code
/// dm::algorithms::CantorZassenhausOptions o;o.max_random_trials=100;
/// std::print("{}\n",dm::algorithms::cantor_zassenhaus_with_options({4,0,1},5,o).size()); // 2
/// @endcode
/// @par References
/// - Cantor and Zassenhaus (1981). https://doi.org/10.1090/S0025-5718-1981-0606517-5
/// @see cantor_zassenhaus(), CantorZassenhausOptions
std::vector<GFPoly> cantor_zassenhaus_with_options(const GFPoly& f,std::int64_t p,
    CantorZassenhausOptions options={});
/// @brief Retain actual DDF steps and EDF attempts from one shared factorization run.
/// No factorization/powering is replayed. Character powers use the recurrence
/// e_(j+1)=p*e_j+(p-1)/2 via modular powering, never storing p^d. This fixes wrapped
/// uint64 exponents while retaining coefficient sampling and depth-first order.
/// Input validation and caps match the configured ordinary variant. Owned records
/// add storage proportional to the sum of recorded polynomial lengths across all
/// steps/trials; potentially large traces should use explicit caps. Inputs unchanged.
/// @param f Borrowed nonzero squarefree coefficients, possibly nonmonic.
/// @param p Odd prime in [3,INT64_MAX].
/// @param options Effective seed and whole-call work caps; defaults unlimited.
/// @return Owned normalization, factors, residual shortcut, actual records and work.
/// @throws std::invalid_argument Invalid field, zero input or repeated factors.
/// @throws std::length_error Product size overflow or a work cap would be exceeded.
/// @throws std::overflow_error Actual work counters exceed size_t.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cantor_zassenhaus_trace({4,4,4,2,1},5);
/// std::print("{} {}\n",t.leading_coefficient,t.work.distinct_degree_steps); // 1 1
/// @endcode
/// @par References
/// - Cantor and Zassenhaus (1981). https://doi.org/10.1090/S0025-5718-1981-0606517-5
/// @see cantor_zassenhaus(), CantorZassenhausTrace
CantorZassenhausTrace cantor_zassenhaus_trace(const GFPoly& f,std::int64_t p,
    CantorZassenhausOptions options={});
} // namespace datamunge::algorithms
